"""普通 QPS/CPU offload 实验的单 GPU 执行，不是当前故障恢复入口。

run.py 启动 worker_main，workload 提供输入，metrics 计算固定窗口指标。
run_trial 按到达表真实推理，同时观察原生 Scheduler。
"""
import bisect
import builtins
import csv
import json
import os
from pathlib import Path
import time
import traceback

from workload import SEED, make_prompt, request_prompt

from metrics import CacheMetrics, transfer_config, interval_summary, summarize

MAX_NUM_SEQS = 256
BATCH_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90


def memory_locality():
    """读取本进程 NUMA 常驻页与 CPU 亲和性，用于分析内存位置，不代表仅有 KV 的占用。"""
    pages = {}
    try:
        for line in Path('/proc/self/numa_maps').read_text().splitlines():
            for part in line.split():
                if part.startswith('N') and '=' in part:
                    node, count = part.split('=', 1)
                    if node[1:].isdigit():
                        pages[node] = pages.get(node, 0) + int(count)
    except OSError:
        pass
    return {'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'numa_resident_pages': pages}


def observe(scheduler, scheduled):
    """原生 schedule 完成、前向尚未执行时读取 running、预算、KV 和 offload 占用。"""
    scheduled_tokens = scheduled.num_scheduled_tokens
    running, waiting = scheduler.get_request_counts()
    decode = 0
    scheduled_decode = 0
    prefill_tokens = 0
    for req in scheduler.running:
        # 原生调度已推进计算计数；减去本轮安排量才能得到前向前的历史进度。
        before = req.num_computed_tokens - scheduled_tokens.get(req.request_id, 0)
        is_decode = before >= req.num_prompt_tokens and bool(req.output_token_ids)
        decode += int(is_decode)
        scheduled_decode += int(is_decode and scheduled_tokens.get(req.request_id, 0) > 0)
        prefill_tokens += min(scheduled_tokens.get(req.request_id, 0),
                              max(0, req.num_prompt_tokens-before))
    cpu_occupied = 0.0
    if scheduler.connector is not None:
        manager = scheduler.connector.scheduler.manager
        # 原生 offload 占用包含可淘汰缓存条目；
        # 官方 usage 指标只看传输期间的保护页，不能代替实际缓存占用。
        cpu_occupied = ((manager._num_allocated_chunks - len(manager._free_list))
                        / manager._num_chunks if manager._num_chunks else 0.0)
    return {
        'running': running,
        'decode': decode,
        'scheduled_decode': scheduled_decode,
        'forward_requests': sum(n > 0 for n in scheduled_tokens.values()),
        'waiting': waiting,
        'gpu_kv_usage': scheduler.kv_cache_manager.usage,
        'cpu_cache_occupied_fraction': cpu_occupied,
        'scheduled_tokens': scheduled.total_num_scheduled_tokens,
        'preemptions': len(scheduled.preempted_req_ids or ()),
        'prefill_tokens': prefill_tokens,
    }


def run_trial(engine, scheduler, counts, telemetry, requests, worker, gpu, args, output, command):
    """按 command 到达表运行到 duration，返回一组逐 GPU 指标并保存原始观测。

    请求用固定输出长度排除提前 EOS；窗口结束保留未完成样本，再终止剩余推理。
    """
    from vllm import SamplingParams

    qps, start = command['qps'], command['start']
    arrivals = command['arrivals']
    if len(arrivals) > len(requests):
        raise RuntimeError('Not enough prepared requests')

    cursor = 0
    submitted = []
    admission_times = []
    completions = []
    first_token_events = []
    completed_latencies = []
    states = {}
    rows = []
    request_records = {}
    token_events = []
    generated_tokens = 0
    telemetry['start'] = start
    telemetry['metrics'] = CacheMetrics(args.warmup, args.duration)
    next_print = 0.0
    previous = start
    path = output / f'qps_{qps:g}_gpu{gpu}.csv'

    time.sleep(max(0, start-time.perf_counter()))
    with path.open('w', newline='') as stream:
        writer = None
        while time.perf_counter() < start + args.duration:
            elapsed = time.perf_counter() - start
            while cursor < len(arrivals) and arrivals[cursor] <= elapsed:
                if time.perf_counter() >= start + args.duration:
                    break
                request = requests[cursor]
                rid = request['request_id']
                engine.add_request(
                    rid,
                    {'prompt_token_ids': request['tokens']},
                    SamplingParams(temperature=0,
                                   max_tokens=request['output_length'],
                                   ignore_eos=True, detokenize=False),
                )
                admitted_at = time.perf_counter() - start
                states[rid] = {
                    'request_id': rid, 'input_length': request['input_length'],
                    'output_length': request['output_length'],
                    'arrival_s': arrivals[cursor], 'admitted_s': admitted_at,
                    'first_s': None, 'completed_s': None, 'last_tokens': 0,
                    'last_token_s': None,
                    'ttft_ms': None, 'tpot_ms': None, 'e2e_ms': None,
                    'finish_reason': None, 'status': 'unfinished',
                }
                request_records[rid] = states[rid]
                submitted.append(rid)
                admission_times.append(admitted_at)
                cursor += 1
                elapsed = time.perf_counter() - start

            if engine.has_unfinished_requests():
                counts.clear()
                step_started = time.perf_counter()
                outputs = engine.step()  # 真实执行 GPU Prefill/Decode，并非模拟耗时。
                ended = time.perf_counter()
                engine_step_ms = 1000*(ended-step_started)
                t = ended - start
                for result in outputs:
                    rid = result.request_id
                    state = states.get(rid)
                    if state is None:
                        continue
                    token_count = (len(result.outputs[0].token_ids)
                                   if result.outputs else 0)
                    if token_count > state['last_tokens']:
                        delta = token_count-state['last_tokens']
                        token_events.append({
                            'request_id': rid, 'time_s': t, 'new_tokens': delta,
                            'interval_ms': (1000*(t-state['last_token_s'])
                                            if state['last_token_s'] is not None else None),
                            'engine_step_ms': engine_step_ms,
                            'prefill_tokens': counts.get('prefill_tokens', 0),
                        })
                        state['last_token_s'] = t
                        if args.warmup <= t < args.duration:
                            generated_tokens += token_count - state['last_tokens']
                        if state['first_s'] is None:
                            state['first_s'] = t
                            state['ttft_ms'] = 1000*(t-state['arrival_s'])
                            first_token_events.append(
                                (t, 1000*(t-state['arrival_s'])))
                        state['last_tokens'] = token_count
                    if result.finished:
                        reason = result.outputs[0].finish_reason if result.outputs else None
                        if reason != 'length' or token_count != state['output_length']:
                            raise RuntimeError(f'Unexpected request termination {rid}: '
                                               f'{reason}, tokens={token_count}')
                        completions.append(t)
                        first = state['first_s']
                        n = state['last_tokens']
                        tpot = (1000*(t-first)/(n-1)) if first is not None and n > 1 else None
                        completed_latencies.append({
                            'completed_s': t,
                            'tpot_ms': tpot,
                            'output_tokens': n,
                        })
                        state.update(completed_s=t, tpot_ms=tpot,
                                     e2e_ms=1000*(t-state['arrival_s']),
                                     finish_reason=reason, status='completed')
                        del states[rid]
                current = counts.copy() if counts else {
                    'running': 0, 'decode': 0, 'scheduled_decode': 0,
                    'forward_requests': 0, 'waiting': 0,
                    'gpu_kv_usage': scheduler.kv_cache_manager.usage,
                    'cpu_cache_occupied_fraction': 0.0,
                    'scheduled_tokens': 0, 'preemptions': 0,
                    'prefill_tokens': 0,
                }
            else:
                upcoming = arrivals[cursor] if cursor < len(arrivals) else args.duration
                delay = min(0.05, max(0, upcoming - (time.perf_counter()-start)))
                if delay > 0:
                    time.sleep(delay)
                ended = time.perf_counter()
                engine_step_ms = 0.0
                current = {
                    'running': 0, 'decode': 0, 'scheduled_decode': 0,
                    'forward_requests': 0, 'waiting': 0,
                    'gpu_kv_usage': scheduler.kv_cache_manager.usage,
                    'cpu_cache_occupied_fraction': (
                        counts.get('cpu_cache_occupied_fraction', 0.0)),
                    'scheduled_tokens': 0, 'preemptions': 0,
                    'prefill_tokens': 0,
                }

            # 本轮 Scheduler 快照对应这一段实际工作，按持续时长参与负载统计。
            midpoint = (previous+ended)/2-start
            due = bisect.bisect_right(arrivals, midpoint)
            admitted_by_mid = bisect.bisect_right(admission_times, midpoint)
            completed_by_mid = bisect.bisect_right(completions, midpoint)
            row = {
                'start_s': max(0, previous-start),
                'end_s': ended-start,
                'engine_step_ms': engine_step_ms,
                **current,
                'ingress_waiting': max(0, due-admitted_by_mid),
                'outstanding': max(0, due-completed_by_mid),
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            rows.append(row)
            previous = ended

            if ended-start >= next_print:
                print(f'QPS={qps:g} GPU={gpu} t={ended-start:.1f}s {current}', flush=True)
                stream.flush()
                next_print = ended-start+10

    unfinished = engine.get_num_unfinished_requests()
    if cursor != len(completions)+unfinished:
        raise RuntimeError('Request accounting mismatch: '
                           f'submitted={cursor}, finished={len(completions)}, pending={unfinished}')
    not_admitted = len(arrivals)-cursor
    telemetry['start'] = None
    for index in range(cursor, len(arrivals)):
        request = requests[index]
        request_records[request['request_id']] = {
            'request_id': request['request_id'], 'input_length': request['input_length'],
            'output_length': request['output_length'], 'arrival_s': arrivals[index],
            'admitted_s': None, 'first_s': None, 'completed_s': None,
            'last_tokens': 0, 'ttft_ms': None, 'tpot_ms': None, 'e2e_ms': None,
            'finish_reason': None, 'status': 'not_admitted',
        }
    with (output / f'requests_qps_{qps:g}_gpu{gpu}.jsonl').open('w') as stream:
        for record in request_records.values():
            stream.write(json.dumps(record)+'\n')
    cache_metrics = telemetry['metrics']
    with (output / f'token_events_qps_{qps:g}_gpu{gpu}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'request_id', 'time_s', 'new_tokens', 'interval_ms', 'engine_step_ms', 'prefill_tokens'])
        writer.writeheader()
        writer.writerows(token_events)
    (output / f'cache_qps_{qps:g}_gpu{gpu}.json').write_text(json.dumps({
        'full_run': dict(cache_metrics.full),
        'measurement': dict(cache_metrics.measured),
        'native_metrics': cache_metrics.native,
    }, indent=2))
    if unfinished:
        engine.abort_request(submitted)
    if engine.has_unfinished_requests():
        raise RuntimeError('Abort did not clear unfinished requests')

    return {
        'global_qps': qps, 'worker': worker, 'gpu': gpu,
        'cpu_cache_gb': args.cpu_cache_gb,
        'cpu_cache_bytes': int(args.cpu_cache_gb * 1_000_000_000),
        'repeat': args.repeat,
        'trace_shard': (worker-args.routing_offset) % len(args.gpus),
        **summarize(rows, args.warmup, args.duration, qps, arrivals,
                    admission_times, completions, first_token_events,
                    completed_latencies, batch_tokens=BATCH_TOKENS),
        'planned': len(arrivals), 'submitted': cursor,
        'not_admitted': not_admitted, 'unfinished': unfinished,
        'completed_total': len(completions),
        'generation_tokens_per_s': generated_tokens / (args.duration-args.warmup),
        **cache_metrics.summary(),
        **interval_summary(token_events, args.warmup, args.duration),
        'scheduler_trace_lines_suppressed': telemetry['suppressed'][0],
    }


def worker_main(worker, gpu, args, requests, output, connection):
    """普通负载子进程入口：准备输入、预热清缓存、接收统一启动命令并运行一组配置。"""
    output = Path(output)
    log = (output / f'worker{worker}.log').open('w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    # 本地 vLLM 会逐 Step 强制刷新调试输出，可能干扰耗时；
    # 仅在本 worker 过滤该调试前缀，仍保留错误和警告。
    original_print = builtins.print
    suppressed = [0]
    def quiet_print(*values, **kwargs):
        if (not args.scheduler_trace and values and isinstance(values[0], str)
                and values[0].startswith('[SCHED_TRACE]')):
            suppressed[0] += 1
            return
        return original_print(*values, **kwargs)
    builtins.print = quiet_print
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_NO_USAGE_STATS='1',
        OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads),
    )
    for key in list(os.environ):
        if key.startswith(('KV_P1_', 'KV_P2_', 'KV_P3_', 'KV_PREFILL_', 'P1_KV_')):
            del os.environ[key]

    engine = None
    try:
        # 只构造本 worker 的输入，避免每个进程重复准备全部 trace。
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        special = set(tokenizer.all_special_ids)
        vocab_size = min(tokenizer.vocab_size, args.model_vocab_size or tokenizer.vocab_size)
        allowed = tuple(i for i in range(vocab_size) if i not in special)
        if not allowed:
            raise ValueError('Model/tokenizer has no usable non-special input token IDs')
        warmup_token = allowed[0]
        requests = [dict(r, tokens=request_prompt(r, allowed, args.trace_block_tokens)) for r in requests]
        del tokenizer, allowed

        import vllm
        from vllm import EngineArgs, SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.config import KVTransferConfig

        kv_config = transfer_config(args.cpu_cache_gb, args.cpu_cache_storage)
        gpu_kv_gb = getattr(args, 'gpu_kv_cache_gb', None)

        engine = LLMEngine.from_engine_args(
            EngineArgs(
                model=args.model, dtype=args.dtype, tensor_parallel_size=1,
                max_model_len=args.max_model_len, max_num_seqs=MAX_NUM_SEQS,
                max_num_batched_tokens=BATCH_TOKENS,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                kv_cache_memory_bytes=(int(gpu_kv_gb * 1_000_000_000)
                                       if gpu_kv_gb is not None else None),
                block_size=16, enable_prefix_caching=True,
                enable_chunked_prefill=True, async_scheduling=False,
                enforce_eager=args.eager, disable_log_stats=False, seed=SEED,
                kv_transfer_config=KVTransferConfig(**kv_config) if kv_config else None,
            ),
            enable_multiprocessing=False,
        )
        scheduler = engine.engine_core.engine_core.scheduler
        (output/f'locality_gpu{gpu}.json').write_text(json.dumps(memory_locality(), indent=2))
        warmup_input = min(128, max(1, args.max_model_len//2))
        engine.add_request('warmup', {'prompt_token_ids': [warmup_token]*warmup_input},
                           SamplingParams(temperature=0, max_tokens=min(8, args.max_model_len-warmup_input),
                                          ignore_eos=True, detokenize=False))
        while engine.has_unfinished_requests():
            engine.step()

        counts = {}
        telemetry = {'start': None, 'metrics': None, 'suppressed': suppressed}
        original_schedule = scheduler.schedule
        def schedule(*positional, **keywords):
            scheduled = original_schedule(*positional, **keywords)
            counts.update(observe(scheduler, scheduled))
            return scheduled
        scheduler.schedule = schedule

        original_make_stats = scheduler.make_stats
        def make_stats(*positional, **keywords):
            connector_stats = (positional[1] if len(positional) > 1
                               else keywords.get('kv_connector_stats'))
            stats = original_make_stats(*positional, **keywords)
            if telemetry['start'] is not None:
                telemetry['metrics'].record(
                    time.perf_counter()-telemetry['start'], stats, connector_stats)
            return stats
        scheduler.make_stats = make_stats

        if engine.has_unfinished_requests():
            raise RuntimeError('Initialization warmup did not terminate cleanly')
        if not engine.engine_core.reset_prefix_cache(reset_connector=True):
            raise RuntimeError('Failed to reset prefix cache')
        connection.send({
            'kind': 'ready', 'worker': worker, 'gpu': gpu,
            'vllm': vllm.__version__,
            'model': args.model, 'dtype': str(engine.model_config.dtype),
            'max_model_len': args.max_model_len,
            'trace_block_tokens': args.trace_block_tokens,
            'kv_blocks': scheduler.kv_cache_config.num_blocks,
            'cpu_cache_gb': args.cpu_cache_gb,
            'cpu_cache_bytes': int(args.cpu_cache_gb * 1_000_000_000),
            'kv_transfer_config': kv_config,
            'cpu_cache_chunks': (scheduler.connector.scheduler.manager._num_chunks
                                 if scheduler.connector is not None else 0),
            'gpu_kv_cache_gb_requested': gpu_kv_gb,
            'cpu_threads': args.cpu_threads,
            'trace_shard': (worker-args.routing_offset) % len(args.gpus),
        })
        command = connection.recv()
        if command is None:
            return
        result = run_trial(engine, scheduler, counts, telemetry, requests, worker, gpu,
                           args, output, command)
        connection.send({'kind': 'result', 'worker': worker, 'summary': result})
    except BaseException:
        try:
            connection.send({'kind': 'error', 'worker': worker,
                             'traceback': traceback.format_exc()})
        except Exception:
            pass
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()
        connection.close()
        builtins.print = original_print
        log.close()
