"""单 GPU 执行：完成正常推理、故障快照、KV 恢复和严格回放。

先读 FailureRunner.run；父进程由 experiment 调用 worker_main，
本文件调用 cpu_kv 备份/加载，使用 workload 准备输入、metrics/report 记录结果。
"""
from bisect import bisect_right
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from workload import SEED, make_prompt, request_prompt, workload_identity
from metrics import StabilityGate, latency_summary, running_snapshot

from .cpu_kv import KVTransfer, SourceKVBackup, WallProfile, profile_phase
from .report import compact_record, metric_summary, update_latency


@dataclass(frozen=True)
class ReplayContext:
    """一次运行所需的回放目录、故障事件和共享 CPU 内存句柄。

    这些是进程资源，不是额外的命令行参数；inference_only 表示各策略独立执行在线备份。
    """
    mode: str = None
    directory: str = None
    failure_event: object = None
    cpu_handle: object = None
    inference_only: bool = False  # 各备份策略独立产生自己的 DMA 和槽位状态。
    diagnostics_directory: str = None


class PrefailureReplay:
    """逐 Step 记录/校验真实推理，保证轮内故障队列和生成进度可比。

    回放仍执行模型前向；inference_only 不重用其他策略的 CPU 备份操作或字节。
    """
    def __init__(self, context, worker):
        self.mode = context.mode
        self.inference_only, self.worker = context.inference_only, worker
        self.path = Path(context.directory)/f'worker_{worker}.json' if self.mode else None
        self.data = json.loads(self.path.read_text()) if self.mode=='replay' else dict(steps=[])
        if self.mode=='replay' and self.data.get('inference_only',False)!=self.inference_only:
            raise ValueError('Inference-only and legacy CPU-operation replay files cannot be mixed')
        self.index = 0
        self.details=[]
        self.divergence=None
        detail_path=(Path(context.diagnostics_directory)/f'worker_{worker}_steps.json'
                     if context.diagnostics_directory else None)
        self.reference_details=json.loads(detail_path.read_text()) if detail_path and detail_path.exists() else None

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def next_step(self):
        if self.mode=='replay' and self.index<len(self.data['steps']):
            return self.data['steps'][self.index]
        return None

    def step(self, cursor, begin, end, batches, outputs, cpu_ops=None, diagnostic=None):
        """每次真实前向后校验提交游标、调度 token 数和完整输出签名。

        记录模式追加参考步骤；回放模式遇到首个差异立即报错，不反复尝试到一致。
        """
        if not self.mode:
            return
        signature = self.digest([
            [[r['request_id'],r['scheduled_tokens']] for b in batches for r in b['scheduled_requests']],
            [[r.request_id,r.finished,list(r.outputs[0].token_ids) if r.outputs else []] for r in outputs]])
        row = dict(cursor=cursor,start_s=begin,end_s=end,signature=signature)
        if diagnostic is not None:
            detail=dict(step=self.index,cursor=cursor,signature=signature,
                scheduled=[[r['request_id'],r['scheduled_tokens'],r['computed_tokens_before_step']]
                           for b in batches for r in b['scheduled_requests']],
                outputs=[[r.request_id,r.finished,len(r.outputs[0].token_ids),r.outputs[0].token_ids[-1]]
                         for r in outputs if r.outputs and r.outputs[0].token_ids],**diagnostic)
            self.details.append(detail)
        if cpu_ops is not None and not self.inference_only:
            row['cpu_ops'] = cpu_ops
        if self.mode=='record':
            self.data['steps'].append(row)
        elif row!=self.next_step():
            self.divergence=dict(kind='forward',worker=self.worker,step=self.index,
                verified_steps=self.index,expected=self.next_step(),actual=row,
                actual_detail=self.details[-1] if diagnostic is not None else None,
                reference_detail=(self.reference_details[self.index]
                    if self.reference_details is not None and self.index<len(self.reference_details) else None))
            raise RuntimeError(f'Strict pre-failure replay diverged at forward {self.index} on worker {self.worker}; '
                               f"cursor actual={cursor}, expected={self.next_step()['cursor']}")
        self.index += 1

    @staticmethod
    def native_state(scheduler,ids,cursor):
        return dict(cursor=cursor,
            running=[ids[r.request_id] for r in scheduler.running],
            waiting=[[ids[r.request_id] for r in q] for q in (scheduler.waiting,scheduler.skipped_waiting)],
            requests=sorted((ids[rid],r.num_computed_tokens,list(r.output_token_ids))
                            for rid,r in scheduler.requests.items()))

    def finish(self, scheduler, ids, cursor, now, stability=None, cpu_snapshot=None):
        """故障边界核对请求队列和 token 进度，返回校验摘要。

        推理独立回放还校验幸存 GPU 的缓存、引用数及淘汰顺序，避免只复现请求表面状态。
        """
        if not self.mode:
            return {}
        state = self.native_state(scheduler,ids,cursor)
        signature = self.digest(state)
        cache_signature = None
        if self.inference_only and self.worker != 0:
            # 幸存实例必须具有相同的 APC 内容、引用数和淘汰顺序；
            # 仅有相同队列长度不足以说明恢复起点公平。
            pool = scheduler.kv_cache_manager.block_pool
            cache_signature = self.digest([
                [[b.block_id, b.block_hash.hex() if isinstance(b.block_hash,bytes) else repr(b.block_hash),
                  b.block_hash_num_tokens, b.ref_cnt] for b in pool.blocks],
                [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]])
        if self.mode=='record':
            self.data.update(failure_s=now,state_signature=signature,
                kv_blocks=getattr(getattr(scheduler,'kv_cache_config',None),'num_blocks',None),
                inference_only=self.inference_only, gpu_cache_signature=cache_signature)
            if stability is not None:
                self.data['stability'] = stability
            if cpu_snapshot is not None and not self.inference_only:
                self.data['cpu_kv_snapshot'] = cpu_snapshot
            if not self.inference_only:
                self.close()
        elif (self.index!=len(self.data['steps']) or signature!=self.data['state_signature']):
            self.divergence=dict(kind='fault_state',worker=self.worker,step=self.index,
                expected_signature=self.data['state_signature'],actual_signature=signature)
            raise RuntimeError('Strict pre-failure snapshot mismatch: queues, computed tokens or output tokens differ')
        elif cache_signature!=self.data.get('gpu_cache_signature'):
            self.divergence=dict(kind='survivor_cache',worker=self.worker,step=self.index,
                expected_signature=self.data.get('gpu_cache_signature'),actual_signature=cache_signature)
            raise RuntimeError('Strict pre-failure survivor GPU cache/eviction state differs')
        return dict(prefailure_verified=True,prefailure_steps=self.index,prefailure_state_signature=signature,
                    prefailure_gpu_cache_signature=cache_signature)

    def close(self):
        # 严格备份对照不在故障屏障上序列化参考日志；
        # 源在收到派发命令后写出，幸存实例在最终退出时写出。
        if self.mode=='record' and 'state_signature' in self.data:
            self.path.write_text(json.dumps(self.data,separators=(',',':')))


def merge_pending(requests, arrivals, cursor, forwarded):
    """把转发流量按原到达时间并入尚未提交的输入，返回新请求表和到达表。

    cursor 之前已入引擎的请求保持原位置，不能重新提交。
    """
    pending = list(zip(requests[cursor:], arrivals[cursor:]))
    pending.extend((row, row['arrival_s']) for row in forwarded)
    pending.sort(key=lambda item: item[1])
    return (requests[:cursor]+[r for r, _ in pending],
            arrivals[:cursor]+[t for _, t in pending])


def recovery_input(checkpoint, mode):
    """由检查点构造恢复输入，返回（输入 token、还需生成的数量）。

    resume 使用原 prompt 加已输出 token；restart 只使用原 prompt。
    """
    generated = checkpoint['generated_token_ids'] if mode == 'resume' else []
    remaining = checkpoint['output_length']-len(generated)
    if remaining < 1:
        raise ValueError('A completed request must not enter the recovery cohort')
    return checkpoint['prompt_token_ids']+generated, remaining


def promote_recoveries(scheduler, recovery_order):
    """仅把仍在等待的恢复请求按 recovery_order 提到队首，不移动 running。"""
    rank = {rid: i for i, rid in enumerate(recovery_order)}
    for queue in (scheduler.waiting, scheduler.skipped_waiting):
        selected = sorted((r for r in queue if r.request_id in rank),
                          key=lambda r: rank[r.request_id])
        for request in selected:
            queue.remove_request(request)
        for request in reversed(selected):
            queue.prepend_request(request)


def recovery_queue(scheduler, recovery_order, fallback, transfer=None, deferred=None):
    """为原生 Scheduler 选择优先队列，并临时绕过等待后缀拼接的队首。

    返回 waiting/skipped_waiting 中的一个队列；deferred 保存本轮移出的请求，稍后必须放回。
    """
    if transfer is not None and transfer.pending and deferred is not None:
        for queue in (scheduler.skipped_waiting,scheduler.waiting):
            while (queue and len(scheduler.waiting)+len(scheduler.skipped_waiting)>1
                   and transfer.waiting_for_join(queue.peek_request())):
                request=queue.pop_request()
                deferred.append((queue,request))
                metrics=transfer.pending[request.request_id]['metrics']
                metrics['join_wait_deferrals']=metrics.get('join_wait_deferrals',0)+1
        # 全部 waiter 都等拼接时留一个走原生零 token 退出；
        # 否则选到空队列会破坏原生 peek 的前置条件。
    rank = {rid: i for i, rid in enumerate(recovery_order)}
    candidates = [q for q in (scheduler.skipped_waiting, scheduler.waiting)
                  if q and q.peek_request().request_id in rank]
    queue=(min(candidates, key=lambda q: rank[q.peek_request().request_id])
           if candidates else fallback())
    if transfer is not None and getattr(transfer,'demand_members',None) and queue:
        request=queue.peek_request()
        if not transfer.waiting_for_join(request):
            transfer.load_demand(request,inside_schedule=True)
        if (transfer.waiting_for_join(request)
                and len(scheduler.waiting)+len(scheduler.skipped_waiting)>1):
            deferred.append((queue,queue.pop_request()))
            return recovery_queue(scheduler,recovery_order,fallback,transfer,deferred)
    return queue


def restore_deferred_recoveries(scheduler, recovery_order, deferred):
    """原生调度结束或异常时放回临时移出的请求，恢复恢复队列优先级。"""
    if not deferred:
        return
    for queue,request in reversed(deferred):
        queue.prepend_request(request)
    deferred.clear()
    promote_recoveries(scheduler,recovery_order)


def failure_stability(args, load_rows, arrivals, records, now):
    """用故障前 load 与延迟记录检查连续滚动窗口，返回 qualified 和检查明细。"""
    gate = StabilityGate(args)
    gate.rows.extend((r['start_s'], r['end_s'], r) for r in load_rows)
    records = list(records)
    completions = sorted(r['completed_s'] for r in records if r.get('completed_s') is not None)
    events = [(r['first_s'],r['ttft_ms']) for r in records if r.get('first_s') is not None]
    checks = [gate.evaluate(now-i*args.check_every, arrivals, completions, events)
              for i in reversed(range(args.stable_checks))]
    return dict(qualified=all(c['stable'] for c in checks), checks=checks)


@contextmanager
def engine_session(args, worker, context, warmup_token):
    """创建并预热一个 TP=1 引擎，清空预热缓存后交给 worker 使用。

    回放时复用参考 GPU 块数；退出上下文时关闭引擎。
    """
    from vllm import EngineArgs, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    saved = (json.loads((Path(context.directory)/f'worker_{worker}.json').read_text())
             if context.mode == 'replay' else {})
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype=args.dtype, tensor_parallel_size=1,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.batch_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=16, enable_prefix_caching=True, enable_chunked_prefill=True,
        async_scheduling=False, scheduling_policy='fcfs', enforce_eager=args.eager,
        seed=SEED, num_gpu_blocks_override=saved.get('kv_blocks'),
        kv_transfer_config=None), enable_multiprocessing=False)
    try:
        length = min(128, max(1, args.max_model_len//2))
        engine.add_request('__initialization__', {'prompt_token_ids': [warmup_token]*length},
            SamplingParams(temperature=0, max_tokens=min(8, args.max_model_len-length),
                           ignore_eos=True, detokenize=False))
        while engine.has_unfinished_requests():
            engine.step()
        if not engine.engine_core.reset_prefix_cache(reset_connector=True):
            raise RuntimeError('Failed to clear initialization prefix cache')
        yield engine
    finally:
        engine.engine_core.shutdown()


def worker_main(worker, gpu, args, requests, arrivals, output, connection, context):
    """子进程入口：绑定物理 GPU、准备 token、创建引擎并运行 FailureRunner。

    通过 connection 向父进程发送 ready、现场和结果；异常也通过同一管道上报。
    """
    log = open(os.devnull, 'w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    original_print = builtins.print
    def quiet_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith('[SCHED_TRACE]'):
            return
        return original_print(*values, **kwargs)
    builtins.print = quiet_print
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_NO_USAGE_STATS='1', VLLM_LOGGING_LEVEL='ERROR',
        OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for key in list(os.environ):
        if key.startswith(('KV_P1_', 'KV_P2_', 'KV_P3_', 'KV_PREFILL_', 'P1_KV_')):
            del os.environ[key]
    def terminate(signum, frame):
        raise KeyboardInterrupt('Worker termination requested')
    signal.signal(signal.SIGTERM, terminate)
    try:
        from transformers import AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        specials = set(tokenizer.all_special_ids)
        allowed = tuple(i for i in range(min(tokenizer.vocab_size,
            args.model_vocab_size or tokenizer.vocab_size)) if i not in specials)
        if not allowed:
            raise ValueError('Tokenizer has no usable non-special tokens')
        warmup_token = allowed[0]
        requests = [dict(row, tokens=request_prompt(row, allowed, args.trace_block_tokens))
                    for row in requests]
        del tokenizer, allowed
        properties = torch.cuda.get_device_properties(0)
        identity = dict(gpu=gpu, pid=os.getpid(),
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            trace=requests[0].get('trace_file', args.trace) if requests else args.trace,
            cache_salt=None, device_uuid=str(properties.uuid), device_name=properties.name,
            free_bytes_before_engine=torch.cuda.mem_get_info(0)[0],
            **workload_identity(requests, arrivals))
        with engine_session(args, worker, context, warmup_token) as engine:
            scheduler = engine.engine_core.engine_core.scheduler
            connection.send(dict(kind='ready', worker=worker, gpu=gpu,
                dtype=str(engine.model_config.dtype), identity=identity,
                cpu_cache_gb=0., kv_transfer_config=None,
                kv_blocks=scheduler.kv_cache_config.num_blocks))
            command = connection.recv()
            if command is not None:
                time.sleep(max(0., command['start']-time.perf_counter()))
                summary = FailureRunner(engine, scheduler, requests, arrivals, args,
                    worker, gpu, output, connection, command['start'], context).run()
                connection.send(dict(kind='result', **summary))
    except BaseException:
        try:
            connection.send(dict(kind='error', worker=worker, gpu=gpu,
                                 traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
        builtins.print = original_print
        log.close()


class FailureRunner:
    """一个实例的推理状态机：源实例到故障即停止，幸存实例继续恢复和原流量。"""
    def __init__(self, engine, scheduler, requests, arrivals, args, worker, gpu,
                 output, connection, start, context=None):
        from vllm import SamplingParams
        self.sampling_params = SamplingParams
        self.context = context or ReplayContext()
        self.engine = engine
        self.scheduler = scheduler
        self.requests = requests
        self.arrivals = arrivals
        self.args = args
        self.worker = worker
        self.gpu = gpu
        self.output = output
        self.connection = connection
        self.start = start
        self.profiler=WallProfile(getattr(args,'profile_backup_overhead',False),start)
        self.profile_saved=False
        self.manifest = {r['request_id']: r for r in self.requests}
        # manifest/normal/recovered 用稳定的公开 ID；internal_ids 将原生 ID 映射回来。
        # recovery_order 保存原生 ID，因它直接用于操作 vLLM 队列。
        self.internal_ids, self.normal, self.recovered, self.recovery_order = {}, {}, {}, []
        with self.profiler.stage('replay_setup'):
            self.journal = PrefailureReplay(self.context,self.worker)
        self.cursor, self.previous, self.paused = 0, 0., False
        # paused 表示已经进入过故障屏障，不会在 resume 后恢复为 False；用它停止再次故障/备份。
        self.planned_original_requests = len(self.requests)
        self.load_rows, self.batch_rows, self.prefill_batches = [], [], []
        self.failure_time = self.pause_s = self.resumed_s = None
        self.restore_timing=None
        self.recovery_batch_id = 0
        self.join_deferred=[]
        self.cpu_transfer = self.cpu_backup = self.cpu_snapshot = None
        self.step_cpu_ops = None
        self.original_schedule = self.scheduler.schedule
        self.original_select = self.scheduler._select_waiting_queue_for_scheduling
        self.original_allocate=None
        # 旧引用操作日志无法表达调度中途的释放，故不在该路径加回收重试；
        # 正常运行和推理独立备份对照则使用同一非阻塞重试。
        if (worker==0 and getattr(args,'cpu_kv_backup_gb',0)
                and (not self.context.mode or self.context.inference_only)):
            self.original_allocate=self.scheduler.kv_cache_manager.allocate_slots
            self.scheduler.kv_cache_manager.allocate_slots=self.allocate_slots
        self.original_clip=None
        if (getattr(args,'backup_policy','full')=='suffix-first'
                or getattr(args,'kv_load_policy','eager')=='demand'):
            self.original_clip=getattr(self.scheduler,'_reserve_prefill_lookahead',None)
            if self.original_clip is None:
                raise RuntimeError('This vLLM branch lacks the required prefill boundary API')
            def clip(request,computed,count):
                count=self.original_clip(request,computed,count)
                return self.cpu_transfer.clip_prefill(request,computed,count) if self.cpu_transfer else count
            self.scheduler._reserve_prefill_lookahead=clip
        self.levels, self.batches = {}, []
        self.scheduler._select_waiting_queue_for_scheduling = lambda: recovery_queue(
            self.scheduler, self.recovery_order, self.original_select,self.cpu_transfer,self.join_deferred)
        self.scheduler.schedule = self.schedule

    def run(self):
        """单实例主循环：备份回收 → 故障检查 → 请求提交 → 真实 Step。

        严格回放使用参考逻辑时间，故障后恢复实际计时；返回该实例结果，异常也执行清理。
        """
        try:
            if getattr(self.args, 'cpu_kv_backup_gb', 0):
                with self.profiler.stage('backup_setup'):
                    self.cpu_transfer = KVTransfer(self.engine,self.scheduler,self.context.cpu_handle,self.profiler)
                if self.worker == 0 and (self.journal.mode != 'replay' or self.context.inference_only):
                    self.cpu_backup = SourceKVBackup(self.cpu_transfer,getattr(self.args,'backup_policy','full'),
                        getattr(self.args,'manifest_mode','delta'),getattr(self.args,'manifest_debug_oracle',False),
                        queue_config=dict(survivors=list(range(1,self.args.instances)),
                            min_shared_tokens=self.args.kv_min_shared_tokens,split_factor=self.args.kv_split_factor,
                            mode=self.args.recovery_mode)
                        if getattr(self.args,'backup_policy','full')=='group-prefix-queue' else None)
                    self.cpu_backup.publish_invalidations=self.publish_cpu_updates
            while True:
                if self.cpu_backup is not None and not self.paused:
                    # 先发布已完成的备份再检查故障；冻结时不额外等待 DMA。
                    self.cpu_backup.poll()
                    self.publish_cpu_updates()
                now = time.perf_counter()-self.start
                forced = None
                if not self.paused and self.journal.mode=='replay':
                    # 参考只确定本步提交到哪个游标和逻辑时刻；下面仍运行真实 engine.step。
                    forced = self.journal.next_step()
                    now = forced['start_s'] if forced else self.journal.data['failure_s']
                    delay = now-(time.perf_counter()-self.start)
                    if delay>0:
                        with self.profiler.stage('replay_pacing'):
                            time.sleep(min(delay,.05))
                        continue
                pending_due = self.cursor < len(self.arrivals) and self.arrivals[self.cursor] < self.args.max_duration
                if now >= self.args.max_duration and (
                        now >= self.args.max_duration+self.args.recovery_drain
                        or (not pending_due and not self.engine.has_unfinished_requests())):
                    break
                if not self.paused and ((self.journal.mode=='replay' and forced is None)
                                   or (self.journal.mode!='replay' and self.failure_requested(now))):
                    if not self.enter_failure_barrier(now):
                        self.previous = now
                        break
                    # 单独记录故障屏障暂停，不把它计入普通前向区间。
                    self.previous = time.perf_counter()-self.start
                    continue
                with self.profiler.stage('request_admission'):
                    while (self.cursor < forced['cursor'] if forced else
                           self.cursor < len(self.arrivals) and self.arrivals[self.cursor] <= now and self.arrivals[self.cursor] < self.args.max_duration):
                        # 积压提交期间也检查故障，避免大量到达请求推迟故障边界。
                        if not self.paused and not forced and self.journal.mode!='replay' and self.failure_requested(now):
                            break
                        row = self.requests[self.cursor]
                        internal = self.engine.add_request(row['request_id'], dict(prompt_token_ids=row['tokens']),
                                                      self.sampling_params(temperature=0,
                                                                     max_tokens=row['output_length'],
                                                                     ignore_eos=True, detokenize=False))
                        self.internal_ids[internal] = row['request_id']
                        if getattr(self.scheduler.requests[internal], 'cache_salt', None) is not None:
                            raise RuntimeError('Normal/forwarded request unexpectedly entered an isolated cache namespace')
                        admitted = time.perf_counter()-self.start
                        self.normal[row['request_id']] = self.state_row(row, self.arrivals[self.cursor], admitted)

                        if self.cpu_backup is not None and not self.paused:
                            self.send_cpu_metadata([dict(request_id=row['request_id'],prompt_token_ids=row['tokens'],
                                                        output_length=row['output_length'],arrival_s=self.arrivals[self.cursor])])
                        self.cursor += 1
                        now = forced['start_s'] if forced else time.perf_counter()-self.start
                if not self.paused and not forced and self.journal.mode!='replay' and self.failure_requested(now):
                    continue
                self.execute_step(forced)
        except BaseException as error:
            self.save_profile('diverged' if self.journal.divergence else 'error',repr(error))
            raise
        finally:
            try:
                self.close()
            finally:
                if self.profiler.enabled and getattr(self.args,'profile_dir',None):
                    path=Path(self.args.profile_dir)
                    path.mkdir(parents=True,exist_ok=True)
                    (path/f'worker_{self.worker}_steps.json').write_text(json.dumps(self.journal.details))
        return dict(worker=self.worker, gpu=self.gpu, status='failed_instance' if self.worker == 0 else 'complete',
                    samples=0, required_samples=0, stable_seconds=0., stable_periods=[],
                    request_routing=self.args.request_routing, per_instance_qps=self.args.per_instance_qps,
                    elapsed_s=self.previous, failure_s=self.failure_time, paused_s=self.pause_s, resumed_s=self.resumed_s,
                    latency=self.latencies, submitted=self.cursor,
                    future_not_submitted=len(self.arrivals)-self.cursor,
                    recovery_requests=len(self.recovered), recovery_completed=sum(
                        r['status'] == 'completed' for r in self.recovered.values()),
                    forwarded_requests=len(self.forwarded), forwarded_completed=sum(
                        r['status'] == 'completed' for r in self.forwarded),
                    failure_results=dict(planned_original_requests=self.planned_original_requests,
                        normal=list(self.normal.values()),recovery=list(self.recovered.values()),forwarded=self.forwarded,
                        load_rows=self.load_rows,batch_rows=self.batch_rows,recovery_prefill_batches=self.prefill_batches,
                        recovery_batches=getattr(self,'recovery_batches',[]),
                        timing=dict(paused_s=self.pause_s,resumed_s=self.resumed_s,
                            restore=self.restore_timing,
                            barrier_pause_s=self.resumed_s-self.pause_s if self.resumed_s is not None else None)))

    def allocate_slots(self, *positional, **keywords):
        """原生分配失败时非阻塞回收已完成 D2H 引用，至多重试一次。

        返回原生分配结果；未完成的复制不能等待或提前释放，之后仍交由原生抢占处理。
        """
        result=self.original_allocate(*positional,**keywords)
        backup=self.cpu_backup
        if result is not None or backup is None or backup.frozen or not backup.jobs:
            return result
        pool=self.scheduler.kv_cache_manager.block_pool
        with self.profiler.stage('allocation_reclaim'):
            counters=backup.allocation_reclaim
            pending=sum(len(job['blocks']) for job in backup.jobs)
            free=pool.get_num_free_blocks()
            counters['checks']+=1
            backup.poll()  # 仅查询已完成事件，不等待 PENDING DMA。
            released=pending-sum(len(job['blocks']) for job in backup.jobs)
            counters['released_blocks']+=released
            counters['freed_gpu_blocks']+=pool.get_num_free_blocks()-free
        if not released:
            return result
        # 释放的只是复制额外引用，即便块仍被活跃请求占用，
        # 之后原生抢占该请求时也能像无备份基线一样回收页。
        counters['retries']+=1
        with self.profiler.stage('allocation_retry'):
            result=self.original_allocate(*positional,**keywords)
        counters['retry_successes']+=int(result is not None)
        return result

    @profile_phase('schedule_observation')
    def schedule(self, *positional, **keywords):
        """包装原生调度：推进 KV 拼接、提高恢复队列优先级，再记录真实批次。

        返回原生调度结果；普通 running 仍参加同一批次，不在这里执行模型前向。
        """
        if self.cpu_transfer is not None:
            self.cpu_transfer.advance_restores()
        if (self.cpu_transfer is not None and self.worker == 0 and self.failure_time is None
                and not self.context.inference_only):
            if self.journal.mode == 'replay':
                # 只有旧 CPU 操作对照才复现引用顺序；inference_only 各组真实独立备份。
                self.step_cpu_ops = self.journal.next_step().get('cpu_ops', [])
                self.cpu_transfer.replay_operations(self.step_cpu_ops)
            else:
                self.step_cpu_ops = self.cpu_backup.take_operations()
        promote_recoveries(self.scheduler, self.recovery_order)
        schedule_start = time.perf_counter()-self.start
        if self.failure_time is not None:
            pool=getattr(getattr(self.scheduler,'kv_cache_manager',None),'block_pool',None)
            free_before=pool.get_num_free_blocks() if pool is not None else None
        try:
            with self.profiler.stage('native_scheduler'):
                scheduled = self.original_schedule(*positional, **keywords)
        finally:
            restore_deferred_recoveries(self.scheduler,self.recovery_order,self.join_deferred)
        if self.failure_time is not None:
            free_after=pool.get_num_free_blocks() if pool is not None else None
        if self.cpu_transfer is not None:
            self.cpu_transfer.release_imports()
        if self.failure_time is not None and scheduled.num_scheduled_tokens:
            self.recovery_batch_id += 1
        now = time.perf_counter()-self.start
        running, waiting = self.scheduler.get_request_counts()
        self.levels.update(running=running, waiting=waiting)
        for rid, count in scheduled.num_scheduled_tokens.items():
            state = self.recovered.get(self.internal_ids[rid])
            if count > 0 and state is not None and state['first_scheduled_s'] is None:
                state['first_scheduled_s'] = now
                state['first_schedule_batch'] = self.recovery_batch_id
                request = self.scheduler.requests[rid]
                cached = max(0, min(request.num_computed_tokens-count, request.num_prompt_tokens))
                state['gpu_cached_tokens_first_schedule'] = cached
        # 故障前也记录真实批次，便于核对提高 QPS 后是否用满预算；
        # 不能仅凭 waiting 或 running 数量推断一次前向是否已满。
        if scheduled.num_scheduled_tokens:
            snapshot = running_snapshot(self.scheduler, scheduled, self.manifest, self.internal_ids)
            active = []
            for row in snapshot['requests']:
                count = row['scheduled_tokens']
                if not count:
                    continue
                prefill = min(count, max(0, row['input_length']-row['computed_tokens_before_step']))
                active.append(dict(row, cohort=('recovery' if row.get('is_recovery') else
                                               'forwarded' if self.manifest[row['request_id']].get('is_forwarded')
                                               else 'normal'), prefill_tokens=prefill, decode_tokens=count-prefill))
            total = sum(r['scheduled_tokens'] for r in active)
            budget = min(getattr(self.scheduler, 'max_num_scheduled_tokens', self.args.batch_tokens),
                         self.args.batch_tokens)
            self.batches.append(dict(batch_id=self.recovery_batch_id,schedule_start_s=schedule_start, scheduled_s=now,
                                after_failure=self.failure_time is not None,
                                running=snapshot['running_count'], waiting=snapshot['waiting_count'],
                                total_scheduled_tokens=total, token_budget=budget,
                                scheduled_token_fraction=total/budget,
                                scheduled_requests=active))
            if self.failure_time is not None:
                self.batches[-1].update(free_kv_blocks_before=free_before,free_kv_blocks_after=free_after)
        event=getattr(self,'demand_head_event',None)
        if event is not None:
            event.record()  # 排除调度中注册内存、构造描述符及提交 H2D 的主机时间。
        return scheduled

    def failure_requested(self, now):
        """判断是否到达故障边界；busy 模式由源实例有 running 时发出共享事件。"""
        if not getattr(self.args, 'failure_when_busy', False):
            return now >= self.args.failure_time_s
        if self.worker == 0 and now >= self.args.failure_time_s and self.scheduler.running:
            self.context.failure_event.set()
        return self.context.failure_event.is_set()

    def state_row(self, row, arrival, admitted):
        """建立普通/转发请求的延迟记录；arrival 是计划到达，admitted 是实际登记时间。"""
        state = dict(request_id=row['request_id'], source_index=row['source_index'],
                    trace_file=row.get('trace_file', self.args.trace),
                    source_line=row['source_index']+1, input_length=row['input_length'],
                    output_length=row['output_length'], arrival_s=arrival, admitted_s=admitted,
                    first_s=None, completed_s=None, last_token_s=None, last_tokens=0,
                    ttft_ms=None, tpot_ms=None, arrival_epoch=None, first_epoch=None,
                    completion_epoch=None, status='unfinished')
        if row.get('is_forwarded'):
            state.update(is_forwarded=True, original_request_id=row['original_request_id'],
                         source_worker=0, source_gpu=self.args.gpus[0], target_worker=self.worker, target_gpu=self.gpu,
                         source_queue_at_failure=row['source_queue_at_failure'],
                         source_generated_tokens=row.get('source_generated_tokens', 0),
                         forward_received_s=self.resumed_s, target_queue_to_first_ms=None)
        return state

    def capture_failure(self, now):
        """完整 Step 后保存原生队列、计算/生成进度及已发布 CPU 检查点。

        now 可能是回放逻辑时间；另外保存实际停止单调时钟，供物理恢复计时使用。
        """
        # 物理停止时刻要在冻结、目录校验和现场序列化之前记录，这些开销属于恢复中断。
        stopped_clock = time.perf_counter()
        self.profiler.stop()
        stability = None
        if (getattr(self.args, 'require_stable_before_failure', False) and
                (self.worker == 0 or getattr(self.args, 'failure_stability_scope', 'all') == 'all')):
            # 回放的稳定性沿用参考逻辑历史，包含空闲区间；
            # 真实重建的队列和计算/生成进度仍会独立校验。
            stability = (self.journal.data.get('stability') if self.journal.mode=='replay' else
                         failure_stability(self.args, self.load_rows, self.arrivals, self.normal.values(), now))
            if not stability or not stability['qualified']:
                failed = sorted({k for c in (stability or {}).get('checks', [])
                                 for k, ok in c['checks'].items() if not ok})
                raise RuntimeError(f'GPU {self.gpu} is not stable at failure t={now:.3f}s: {failed}')
        self.paused, self.pause_s = True, now
        # 这里没有计划下一轮前向；快照包含全部原生 running，不只上一轮被安排的请求。
        snapshot = running_snapshot(self.scheduler, None, self.manifest, self.internal_ids)
        lower = max(self.args.warmup, now-self.args.window)
        waiting = [self.internal_ids[r.request_id] for q in (self.scheduler.skipped_waiting, self.scheduler.waiting)
                   for r in q]
        checkpoints = []
        if self.worker == 0:
            if self.cpu_transfer is not None:
                if self.journal.mode == 'replay' and not self.context.inference_only:
                    self.cpu_snapshot = self.journal.data['cpu_kv_snapshot']
                else:
                    self.cpu_snapshot = self.cpu_backup.freeze([r.request_id for r in self.scheduler.running])
                    self.cpu_snapshot['requests'] = {self.internal_ids[rid]: value
                        for rid,value in self.cpu_snapshot['requests'].items()}
            snapshot_states = {r['request_id']: r for r in snapshot['requests']}
            for request in self.scheduler.running:
                rid = self.internal_ids[request.request_id]
                row, state = self.manifest[rid], self.normal[rid]
                generated = list(request.output_token_ids)
                if len(generated) != state['last_tokens']:
                    raise RuntimeError('Checkpoint/output token accounting mismatch')
                checkpoints.append(dict(
                    request_id=rid, source_index=row['source_index'],
                    input_length=row['input_length'], output_length=row['output_length'],
                    trace_file=row.get('trace_file', self.args.trace),
                    source_status=getattr(getattr(request, 'status', None), 'name', 'RUNNING'),
                    source_phase=snapshot_states[rid]['phase'],
                    source_computed_tokens=request.num_computed_tokens,
                    prompt_token_ids=row['tokens'], generated_token_ids=generated,
                    source_arrival_s=state['arrival_s'], source_first_s=state['first_s'],
                    source_last_token_s=state['last_token_s']))
                if self.cpu_snapshot is not None:
                    checkpoints[-1]['cpu_kv'] = self.cpu_snapshot['requests'][rid]
        verification = self.journal.finish(self.scheduler,self.internal_ids,self.cursor,now,stability=stability,
                                      cpu_snapshot=self.cpu_snapshot)
        self.save_profile('fault_verified')
        before = dict(kind='pre_failure', worker=self.worker, gpu=self.gpu, time_s=now,
                      stopped_monotonic_s=stopped_clock, prefailure_wall_s=stopped_clock-self.start,
                      target_time_s=self.args.failure_time_s,
                      point='after_completed_step_before_failure',
                      baseline_start_s=lower, baseline=metric_summary(self.normal.values(), lower, now),
                      waiting_ids=waiting, future_not_submitted=len(self.arrivals)-self.cursor,
                      waiting_request_states=[compact_record(self.normal[rid]) for rid in waiting],
                      checkpoints=checkpoints, **snapshot,**verification)
        if stability is not None:
            before['stability'] = stability
        if self.cpu_snapshot is not None:
            before['cpu_backup'] = self.cpu_snapshot['stats']
            before['cpu_directory'] = self.cpu_snapshot.get('slot_directory',{})
            before['cpu_epoch'] = self.cpu_snapshot.get('epoch')
        if self.worker == 0:
            before['waiting_requests'] = [dict(self.manifest[rid], arrival_s=self.normal[rid]['arrival_s'],
                source_generated_tokens=self.normal[rid]['last_tokens']) for rid in waiting]
            before['future_requests'] = [dict(row, arrival_s=arrival)
                                         for row, arrival in zip(self.requests[self.cursor:], self.arrivals[self.cursor:])]
        before['latencies'] = list(self.normal.values())
        return before

    def enter_failure_barrier(self, now):
        """发布现场并等待父进程指令；源返回 False 结束，目标加载后返回 True。"""
        before = self.capture_failure(now)
        self.connection.send(before)
        if not self.connection.poll(self.args.startup_timeout):
            raise TimeoutError('Failure snapshot barrier timed out')
        command = self.connection.recv()
        self.start = command.get('timeline_start',self.start)
        if self.worker == 0:
            if command['action'] != 'fail':
                raise RuntimeError('Failed instance must not resume inference')
            running_ids = set(before['running_ids'])
            for rid, state in self.normal.items():
                if state['status'] != 'completed':
                    state['status'] = 'failed_running' if rid in running_ids else 'forwarded_waiting'
            self.failure_time = command['failure_s']
            return False
        return self.restore_requests(command)

    def restore_requests(self, command):
        """按 command 登记恢复请求、合并转发流量，加载真实 KV 后等待统一放行。

        返回 True 继续主循环；所有恢复请求先入原生队列，因此 TTFT 包含加载与屏障等待。
        """
        if command['action'] != 'recover':
            raise RuntimeError('Expected recovery assignments')
        self.recovery_batches=[]
        self.failure_time = command['failure_s']
        forwarded = command['forwarded']
        if any(row['request_id'] in self.manifest for row in forwarded):
            raise RuntimeError('Forwarded request ID collides with target traffic')
        self.manifest.update((r['request_id'], r) for r in forwarded)
        self.requests, self.arrivals = merge_pending(self.requests, self.arrivals, self.cursor, forwarded)
        for checkpoint in command['checkpoints']:
            tokens, remaining = recovery_input(checkpoint, self.args.recovery_mode)
            rid = checkpoint['recovery_id']
            row = dict(request_id=rid, source_index=checkpoint['source_index'],
                       trace_file=checkpoint.get('trace_file', self.args.trace),
                       input_length=len(tokens), output_length=remaining,
                       original_request_id=checkpoint['request_id'], is_recovery=True)
            self.manifest[rid] = row
            enqueued = time.perf_counter()-self.start
            # 恢复输入已经包含旧输出；这里等的是本次恢复后的第一个新 token，不是重放旧答案。
            # 全部按选定顺序直接进入 vLLM；缓存命中由原生调度器判断。
            internal = self.engine.add_request(rid,dict(prompt_token_ids=tokens),self.sampling_params(
                temperature=0,max_tokens=remaining,ignore_eos=True,detokenize=False))
            if getattr(self.scheduler.requests[internal],'cache_salt',None) is not None:
                raise RuntimeError('Recovery request unexpectedly entered an isolated cache namespace')
            self.internal_ids[internal] = rid
            self.recovery_order.append(internal)
            self.recovered[rid] = dict(
                self.state_row(row, enqueued, time.perf_counter()-self.start),
                scheduler_request_id=internal,
                is_recovery=True, original_request_id=checkpoint['request_id'],
                source_worker=0, source_gpu=self.args.gpus[0], target_worker=self.worker, target_gpu=self.gpu,
                recovery_mode=self.args.recovery_mode, failure_s=self.failure_time, enqueued_s=enqueued,
                source_failure_s=command.get('source_failure_s', self.failure_time),
                recovery_group=checkpoint.get('recovery_group'),
                shared_prefix_tokens=checkpoint.get('shared_prefix_tokens',0),
                estimated_prefill_tokens=checkpoint['estimated_prefill_tokens'],
                latency_measurement=self.args.latency_measurement,
                source_trace=checkpoint.get('trace_file', self.args.trace),
                target_trace=self.args.survivor_trace or self.args.trace,
                recovery_cache_policy=self.args.recovery_cache_policy,
                gpu_cached_tokens_first_schedule=None,
                first_schedule_batch=None,first_token_batch=None,prefill_scheduled_tokens=0,
                prefill_step_ms=0.,prefill_batch_ids=[],scheduled_batch_ids=[],
                source_generated_tokens=len(checkpoint['generated_token_ids']),
                source_status=checkpoint.get('source_status'),
                source_phase=checkpoint.get('source_phase'),
                source_computed_tokens=checkpoint.get('source_computed_tokens'),
                source_remaining_tokens=checkpoint['output_length']-len(checkpoint['generated_token_ids']),
                original_input_length=checkpoint['input_length'],
                original_output_length=checkpoint['output_length'],
                source_arrival_s=checkpoint['source_arrival_s'], source_first_s=checkpoint['source_first_s'],
                source_last_token_s=checkpoint['source_last_token_s'],
                first_scheduled_s=None, queue_wait_ms=None, scheduled_to_first_ms=None,
                failure_to_first_ms=None, interruption_ms=None, generated_tokens_after_restart=0,
                continued_output_index=(len(checkpoint['generated_token_ids'])+1
                                        if self.args.recovery_mode=='resume' else 1),
                first_token_index=None, first_token_observed_count=None)
        if self.cpu_transfer is not None:
            # 整批先入队再加载，使每个恢复请求的 TTFT 都包含
            # 目标 H2D 和统一放行屏障，不能先加载再开始计时。
            items=[]
            for checkpoint in command['checkpoints']:
                state = self.recovered[checkpoint['recovery_id']]
                request = self.scheduler.requests[state['scheduler_request_id']]
                items.append((request,checkpoint['cpu_kv'],state))
                manifest=checkpoint['cpu_kv']
                state['cpu_backup_tokens'] = len(manifest.get('blocks',manifest['slots']))*16
                state['cpu_unpublished_ready_blocks']=manifest.get('unpublished_ready_blocks',0)
                refs=manifest.get('blocks',[])
                state['cpu_backup_ranges'] = ([(refs[0]['logical_block']*16,(refs[-1]['logical_block']+1)*16)]
                                               if refs else [])
            mode=getattr(self.args,'kv_restore_mode','serial')
            demand=getattr(self.args,'kv_load_policy','eager')=='demand'
            self.restore_timing=(self.cpu_transfer.register_demand(items,mode,self.start) if demand
                                 else self.cpu_transfer.restore_cohort(items,mode,self.execute_step))
            if any(state['first_s'] is not None for state in self.recovered.values()):
                raise RuntimeError('A recovery emitted a new token before the global resume barrier')
            if not demand:
                for state in self.recovered.values():
                    state['preload_complete_s']=time.perf_counter()-self.start
        promote_recoveries(self.scheduler, self.recovery_order)
        queue_ids = {name: [self.internal_ids[r.request_id] for r in getattr(self.scheduler, name)]
                     for name in ('waiting', 'skipped_waiting')}
        queued = {r.request_id for q in (self.scheduler.waiting, self.scheduler.skipped_waiting) for r in q}
        active={r.request_id for r in self.scheduler.running if r.request_id in self.recovery_order}
        if not set(self.recovery_order) <= queued|active:
            raise RuntimeError('Recovery requests are missing from native queues/running')
        ack = dict(kind='recovery_enqueued', worker=self.worker, gpu=self.gpu,
                   time_s=time.perf_counter()-self.start,
                   recovery_ids=list(self.recovered), queue_ids=queue_ids,
                   admitted_recovery_ids=list(self.recovered),
                   running_recovery_ids=[self.internal_ids[rid] for rid in active],
                   kv_restore_mode=getattr(self.args,'kv_restore_mode','serial'))
        if self.cpu_transfer is not None:
            demand=getattr(self.args,'kv_load_policy','eager')=='demand'
            ack.update(restore_protocol='on-demand' if demand else 'eager-preload',
                preload_complete=(False if demand else all(
                    state.get('preload_complete') for state in self.recovered.values())))
            if not demand and not ack['preload_complete']:
                raise RuntimeError('Recovery ACK preceded KV preload completion')
        self.connection.send(ack)
        if not self.connection.poll(self.args.startup_timeout) or self.connection.recv().get('action') != 'resume':
            raise RuntimeError('Recovery resume barrier failed')
        self.resumed_s = time.perf_counter()-self.start
        return True

    def send_cpu_metadata(self, updates, recovery_plan=None):
        """发送请求/token/目录增量和可选恢复计划，记录 IPC，不发送完整 KV 字节。"""
        message=dict(kind='cpu_kv_metadata',worker=self.worker,gpu=self.gpu,updates=updates)
        if recovery_plan is not None:
            message['recovery_plan']=recovery_plan
        with self.profiler.stage('ipc_serialization'):
            payload=ForkingPickler.dumps(message)
        self.profiler.add('ipc_bytes',len(payload))
        self.profiler.add('ipc_messages',1)
        if any('manifest_delta' in row or 'signature' in row for row in updates):
            self.profiler.add('manifest_ipc_bytes',len(payload))
            self.profiler.add('manifest_ipc_messages',1)
        with self.profiler.stage('ipc_transport_wait'):
            self.connection.send_bytes(payload)

    @profile_phase('metadata_publish')
    def publish_cpu_updates(self):
        """发布变化的 READY 引用和最新预测计划；故障检查前父进程即可持有两者。"""
        updates=self.cpu_backup.updates(self.internal_ids)
        plan=self.cpu_backup.take_recovery_plan(self.internal_ids)
        if updates or plan is not None:
            self.send_cpu_metadata(updates,recovery_plan=plan)

    def allocator_summary(self):
        """只读记录 GPU 空闲块和队列头，用于定位严格回放的首次分歧。"""
        pool=getattr(getattr(self.scheduler,'kv_cache_manager',None),'block_pool',None)
        if pool is None:
            return {}
        queue=pool.free_block_queue
        block=queue.fake_free_list_head.next_free_block
        head=[]
        while block is not queue.fake_free_list_tail and len(head)<8:
            head.append([block.block_id,block.block_hash.hex()[:16] if isinstance(block.block_hash,bytes) else None])
            block=block.next_free_block
        return dict(free_blocks=pool.get_num_free_blocks(),
            cached_hashes=len(pool.cached_block_hash_to_block),free_head=head)

    def save_profile(self, status, error=None):
        """故障或出错时保存一次物理耗时与分歧现场，不把之后清理时间混入故障前耗时。"""
        if not self.profiler.enabled or self.profile_saved:
            return
        self.profiler.stop()
        path=Path(self.args.profile_dir)
        path.mkdir(parents=True,exist_ok=True)
        record=dict(status=status,worker=self.worker,gpu=self.gpu,backup_policy=self.args.backup_policy,
            replay_backup_mode=getattr(self.args,'replay_backup_mode','async'),
            verified_steps=self.journal.index,reference_steps=len(self.journal.data.get('steps',[])),
            clock=self.profiler.snapshot(),allocator=self.allocator_summary(),
            divergence=self.journal.divergence,error=error,
            native_state=self.journal.native_state(self.scheduler,self.internal_ids,self.cursor),
            cpu_backup=self.cpu_backup.stats() if self.cpu_backup else None)
        if self.cpu_backup:
            record['dma_jobs_observed_after_stop']=[dict(blocks=len(job['blocks']),
                event_complete=job['end'].query(),gpu_refs=[[b.block_id,b.ref_cnt] for b in job['blocks'][:8]])
                for job in list(self.cpu_backup.jobs)[:4]]
        (path/f'worker_{self.worker}.json').write_text(json.dumps(record))
        self.profile_saved=True


    def execute_step(self, forced=None):
        """执行一次真实 Step，更新输出、延迟和批次，再提交源端异步备份。

        forced 为参考逻辑步骤；返回本轮调度量，后缀并行阶段也用它执行真实前段重算。
        """
        self.levels.clear()
        self.batches.clear()
        if self.engine.has_unfinished_requests():
            if (self.cpu_backup is not None and self.context.inference_only and not self.paused
                    and getattr(self.args,'replay_backup_mode','async')=='step-fenced'):
                self.cpu_backup.before_replayed_step()
                self.publish_cpu_updates()
            step_start = forced['start_s'] if forced else time.perf_counter()-self.start
            physical_before=self.profiler.snapshot() if self.profiler.enabled and not self.paused else None
            observe=(self.cpu_transfer is not None
                     and getattr(self.cpu_transfer,'demand_mode',None)=='suffix-overlap'
                     and any(g['state']!='ready' for g in self.cpu_transfer.demand_groups.values()))
            if observe:
                head_begin=self.cpu_transfer.torch.cuda.Event(enable_timing=True)
                head_end=self.cpu_transfer.torch.cuda.Event(enable_timing=True)
                self.demand_head_event=head_begin
            with self.profiler.stage('engine_step'):
                outputs = self.engine.step()
            if observe:
                self.demand_head_event=None
                head_end.record()
            ended = forced['end_s'] if forced else time.perf_counter()-self.start
            if observe:
                heads=sum(min(row['prefill_tokens'],max(0,plan['start']-row['computed_tokens_before_step']))
                    for batch in self.batches for row in batch['scheduled_requests']
                    if (plan:=self.cpu_transfer.pending.get(row['scheduler_request_id'])) is not None)
                self.cpu_transfer.demand_step_timing(head_begin,head_end,1000*(ended-step_start),heads)
            if not self.paused:
                with self.profiler.stage('replay_verification'):
                    diagnostic=(dict(before_forward=physical_before,allocator=self.allocator_summary())
                                if physical_before is not None else None)
                    self.journal.step(self.cursor,step_start,ended,self.batches,outputs,
                                      cpu_ops=self.step_cpu_ops,diagnostic=diagnostic)
            with self.profiler.stage('output_bookkeeping'):
                # 引擎返回累计输出列表；只把新增 token 发给父进程，延迟以本轮返回时刻记账。
                for result in outputs:
                    state = self.recovered.get(result.request_id) or self.normal[result.request_id]
                    tokens = len(result.outputs[0].token_ids) if result.outputs else 0
                    if self.cpu_backup is not None and not self.paused:
                        self.send_cpu_metadata([dict(request_id=result.request_id,
                            new_tokens=(list(result.outputs[0].token_ids[state['last_tokens']:]) if result.outputs else []),
                            completed=result.finished)])
                    if result.finished and (not result.outputs
                                            or result.outputs[0].finish_reason != 'length'
                                            or tokens != state['output_length']):
                        raise RuntimeError(f'Unexpected completion: {result.request_id}')
                    first = update_latency(state, tokens, ended, result.finished)
                    if first and state.get('is_recovery'):
                        state['first_token_batch'] = self.recovery_batch_id
                        state['first_new_token_id'] = result.outputs[0].token_ids[0]
                    if result.finished and state.get('is_recovery'):
                        state['continuation_sha256'] = hashlib.sha256(
                            json.dumps(list(result.outputs[0].token_ids)).encode()).hexdigest()
                        if getattr(self.args, 'record_recovery_tokens', False):
                            state['continuation_token_ids'] = list(result.outputs[0].token_ids)
            if self.cpu_backup is not None and not self.paused:
                self.cpu_backup.submit()
            if self.cpu_transfer is not None:
                # 调度中空间不足的按需导入，到真实前向结束后才允许抢占并腾页。
                for key in list(getattr(self.cpu_transfer,'demand_deferred',{})):
                    self.cpu_transfer.load_demand(self.cpu_transfer.demand_groups[key]['items'][0][0])
            with self.profiler.stage('batch_bookkeeping'):
                if self.batches:
                    for batch in self.batches:
                        active = batch['scheduled_requests']
                        for row in active:
                            state = self.recovered.get(row['request_id'])
                            if state is not None:
                                state['scheduled_batch_ids'].append(batch['batch_id'])
                                if row['prefill_tokens']:
                                    state['prefill_scheduled_tokens'] += row['prefill_tokens']
                                    # 此值累计整轮 Step 墙钟，含其他请求工作；并非该请求独占的 Prefill 内核耗时。
                                    state['prefill_step_ms'] += 1000*(ended-step_start)
                                    state['prefill_batch_ids'].append(batch['batch_id'])
                        self.batch_rows.append(dict(time_s=step_start, tokens=batch['total_scheduled_tokens'],
                            full=batch['total_scheduled_tokens']==batch['token_budget'],
                            mixed={'normal','recovery'} <= {r['cohort'] for r in active},
                            kv_usage=getattr(getattr(self.scheduler, 'kv_cache_manager', None), 'usage', None)))
                        if any(r['cohort']=='recovery' and r['prefill_tokens'] for r in active):
                            self.prefill_batches.append(dict(batch_id=batch['batch_id'],step_start_s=step_start,step_end_s=ended,
                                total_scheduled_tokens=batch['total_scheduled_tokens'],
                                token_budget=batch['token_budget'],waiting=batch['waiting'],
                                scheduled_requests=[{k:r[k] for k in ('request_id','cohort','phase',
                                    'scheduled_tokens','prefill_tokens','decode_tokens')} for r in active]))
                        if batch['after_failure'] and any(r['cohort']=='recovery' for r in active):
                            self.recovery_batches.append(dict(batch,step_start_s=step_start,step_end_s=ended,
                                step_ms=1000*(ended-step_start)))
                for state in self.recovered.values():
                    if state['first_s'] is not None and 'prefill_to_first_tokens' not in state:
                        state['prefill_to_first_tokens'] = state['prefill_scheduled_tokens']
        else:
            with self.profiler.stage('idle'):
                time.sleep(.01)
            ended = time.perf_counter()-self.start
            self.levels.update(running=0, waiting=0)
        with self.profiler.stage('loop_bookkeeping'):
            outstanding = sum(s['status'] != 'completed' for s in self.normal.values())
            outstanding += sum(s['status'] != 'completed' for s in self.recovered.values())
            ingress = max(0, bisect_right(self.arrivals, min(ended, self.args.max_duration))-self.cursor)
            self.load_rows.append(dict(start_s=self.previous,end_s=ended,**self.levels,
                                  outstanding=outstanding+ingress,ingress=ingress))
            self.previous = ended

        return dict(scheduled_tokens=sum(b['total_scheduled_tokens'] for b in self.batches),
            head_tokens=sum(row['prefill_tokens'] for b in self.batches for row in b['scheduled_requests']
                            if row['cohort']=='recovery'))

    def close(self):
        """恢复 Scheduler 钩子，结束 DMA 并释放引用，再终止剩余请求、保存结束状态。"""
        self.scheduler.schedule = self.original_schedule
        self.scheduler._select_waiting_queue_for_scheduling = self.original_select
        if self.original_allocate is not None:
            self.scheduler.kv_cache_manager.allocate_slots=self.original_allocate
        if self.original_clip is not None:
            self.scheduler._reserve_prefill_lookahead=self.original_clip
        if self.cpu_backup is not None:
            self.cpu_backup.close()
        if self.cpu_transfer is not None:
            if hasattr(self.cpu_transfer,'demand_groups'):
                self.restore_timing=self.cpu_transfer.demand_summary()
            self.cpu_transfer.close()
        if self.engine.has_unfinished_requests():
            self.engine.abort_request(list(self.normal)+list(self.recovered))
        if self.engine.has_unfinished_requests():
            raise RuntimeError('Abort did not clear unfinished requests')
        for state in [*self.normal.values(), *self.recovered.values()]:
            if state['status'] == 'unfinished':
                state['status'] = 'aborted_at_end'
        self.forwarded = [s for s in self.normal.values() if s.get('is_forwarded')]
        self.forwarded.extend(dict(self.state_row(r, arrival, None), status='not_submitted_at_end')
                         for r, arrival in zip(self.requests[self.cursor:], self.arrivals[self.cursor:]) if r.get('is_forwarded'))
        self.latencies = latency_summary(self.normal.values(), self.args.warmup)
        self.journal.close()


