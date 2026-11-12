# 正常推理稳定后随机采样 running；main → run_workers → replay。
# 共用 workload/metrics；inject-failure 是旧入口，当前故障实验另见 failure_recovery.main。
"""Observe native running queues, or inject instance-0 failure and recover its work.

Each GPU is an independent TP=1 instance. --inject-failure enables the next stage.
GPU imports stay inside the spawned worker; --dry-run and tests are CPU-only.
"""
import argparse
import bisect
import builtins
import copy
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time
import traceback

from workload import (load_trace, load_probe_traces, worker_workload,
                      validate_replicas, workload_identity, request_prompt, SEED)
from metrics import percentile, latency_summary, StabilityGate, running_snapshot
from report import write_csv


def replay(engine, scheduler, requests, arrivals, args, worker, gpu, output, connection, start):
    """在正常推理中检查动态稳定性，合格后随机采样 running，返回采样与延迟汇总。

    采样放在原生调度后、前向前；失稳会结束当前稳定区间并重新等待合格。
    """
    from vllm import SamplingParams

    gate = StabilityGate(args)
    rng = random.Random(args.seed+worker)
    manifest = {r['request_id']: r for r in requests}
    request_ids = {}  # add_request() 返回的原生 ID 可能带随机后缀，需映射回稳定的数据集编号。
    records, stable_periods = {}, []
    submitted, completions = [], []
    cursor, sample_count, next_check, next_running_log = 0, 0, 0., 0.
    target, qualified_since, first_qualified = None, None, None
    current = {}
    previous = 0.
    original_schedule = scheduler.schedule
    samples_path = output/f'samples_gpu{gpu}.jsonl'
    checks_path = output/f'stability_gpu{gpu}.jsonl'

    def active_epoch():
        return len(stable_periods) if stable_periods and stable_periods[-1]['end_s'] is None else None

    def stable_seconds(now):
        return sum((p['end_s'] if p['end_s'] is not None else now)-p['start_s']
                   for p in stable_periods)

    def lose_stability(now):
        nonlocal target, qualified_since
        if active_epoch() is not None:
            stable_periods[-1]['end_s'] = now
        target, qualified_since = None, None

    with (samples_path.open('w', buffering=1) as samples,
          (output/f'running_timeline_gpu{gpu}.jsonl').open('w', buffering=1) as timeline,
          checks_path.open('w', buffering=1) as checks,
          (output/f'load_gpu{gpu}.jsonl').open('w', buffering=65536) as load,
          (output/f'events_gpu{gpu}.jsonl').open('w', buffering=1) as events):
        def write_check(record, reason):
            checks.write(json.dumps(dict(record, reason=reason))+'\n')

        def maybe_sample(scheduled):
            nonlocal next_check, target, sample_count, qualified_since, first_qualified, next_running_log
            now = time.perf_counter()-start
            if now >= args.max_duration:
                return
            if now >= next_check:
                proof = gate.check(now, arrivals, completions,
                    [(r['first_s'], r['ttft_ms']) for r in records.values() if r['first_s'] is not None])
                write_check(proof, 'periodic')
                connection.send(dict(kind='stability', worker=worker, gpu=gpu, **proof))
                progress = dict(time_s=now, stable_seconds=stable_seconds(now),
                                samples=sample_count, **latency_summary(records.values(), args.warmup))
                (output/f'latency_progress_gpu{gpu}.json').write_text(json.dumps(progress, indent=2))
                next_check = now+args.check_every
                if not proof['stable']:
                    lose_stability(now)
                elif proof['qualified'] and active_epoch() is None:
                    qualified_since = now
                    first_qualified = first_qualified if first_qualified is not None else now
                    stable_periods.append(dict(start_s=now, end_s=None))
                    target = now+rng.uniform(*args.sample_gap)
            if sample_count < args.samples and target is not None and now >= target:
                # 采样当刻再次检查稳定性，不能只相信上次周期检查。
                proof = gate.evaluate(now, arrivals, completions,
                    [(r['first_s'], r['ttft_ms']) for r in records.values() if r['first_s'] is not None])
                write_check(proof, 'sample_validation')
                if not proof['stable']:
                    gate.streak = 0
                    lose_stability(now)
                else:
                    snapshot = dict(
                        sample=sample_count+1, worker=worker, gpu=gpu,
                        time_s=now, target_time_s=target, lateness_s=now-target,
                        qualified_since_s=qualified_since, stability=proof,
                        point='after_schedule_before_forward' if scheduled is not None else 'idle',
                        **running_snapshot(scheduler, scheduled, manifest, request_ids))
                    samples.write(json.dumps(snapshot, ensure_ascii=False)+'\n')
                    connection.send(dict(kind='sample', **snapshot))
                    sample_count += 1
                    target = now+rng.uniform(*args.sample_gap)
            if now >= next_running_log:
                timeline.write(json.dumps(dict(
                    time_s=now, worker=worker, gpu=gpu, qualified_at_last_check=active_epoch() is not None,
                    stable_epoch=active_epoch(),
                    point='after_schedule_before_forward' if scheduled is not None else 'idle',
                    **running_snapshot(scheduler, scheduled, manifest, request_ids)), ensure_ascii=False)+'\n')
                next_running_log = now+args.running_log_every

        def schedule(*positional, **keywords):
            scheduled = original_schedule(*positional, **keywords)
            running, waiting = scheduler.get_request_counts()
            current.update(running=running, waiting=waiting)
            maybe_sample(scheduled)
            return scheduled

        scheduler.schedule = schedule
        try:
            while time.perf_counter()-start < args.max_duration:
                now = time.perf_counter()-start
                while cursor < len(arrivals) and arrivals[cursor] <= now:
                    if time.perf_counter()-start >= args.max_duration:
                        break
                    row = requests[cursor]
                    internal_id = engine.add_request(
                        row['request_id'], {'prompt_token_ids': row['tokens']},
                        SamplingParams(temperature=0, max_tokens=row['output_length'],
                                       ignore_eos=True, detokenize=False))
                    request_ids[internal_id] = row['request_id']
                    admitted = time.perf_counter()-start
                    epoch = active_epoch()
                    records[row['request_id']] = dict(
                        request_id=row['request_id'], source_index=row['source_index'],
                        source_line=row['source_index']+1, input_length=row['input_length'],
                        output_length=row['output_length'], arrival_s=arrivals[cursor], admitted_s=admitted,
                        first_s=None, completed_s=None, last_tokens=0, ttft_ms=None, tpot_ms=None,
                        arrival_epoch=(epoch if epoch is not None
                                       and arrivals[cursor] >= qualified_since else None),
                        first_epoch=None, completion_epoch=None, status='unfinished')
                    submitted.append(admitted)
                    events.write(json.dumps(dict(event='admitted', time_s=admitted,
                                                 request_id=row['request_id'],
                                                 scheduler_request_id=internal_id))+'\n')
                    cursor += 1
                    now = time.perf_counter()-start
                current.clear()
                if engine.has_unfinished_requests():
                    outputs = engine.step()
                    ended = time.perf_counter()-start
                    for result in outputs:
                        state = records[result.request_id]
                        tokens = len(result.outputs[0].token_ids) if result.outputs else 0
                        if tokens > state['last_tokens']:
                            if state['first_s'] is None:
                                state.update(first_s=ended, ttft_ms=1000*(ended-state['arrival_s']),
                                             first_epoch=active_epoch())
                                events.write(json.dumps(dict(event='first_token', time_s=ended,
                                                             request_id=result.request_id,
                                                             ttft_ms=state['ttft_ms']))+'\n')
                            state['last_tokens'] = tokens
                        if result.finished:
                            expected = manifest[result.request_id]['output_length']
                            if (not result.outputs or result.outputs[0].finish_reason != 'length'
                                    or len(result.outputs[0].token_ids) != expected):
                                raise RuntimeError(f'Unexpected completion: {result.request_id}')
                            completions.append(ended)
                            state.update(completed_s=ended, completion_epoch=active_epoch(),
                                         tpot_ms=(1000*(ended-state['first_s'])/(tokens-1)
                                                  if tokens > 1 else None), status='completed')
                            events.write(json.dumps(dict(event='completed', time_s=ended,
                                                         request_id=result.request_id,
                                                         ttft_ms=state['ttft_ms'],
                                                         tpot_ms=state['tpot_ms']))+'\n')
                else:
                    current.update(running=0, waiting=0)
                    maybe_sample(None)  # 空快照也有效，避免只采忙时产生偏差。
                    upcoming = arrivals[cursor] if cursor < len(arrivals) else args.max_duration
                    time.sleep(min(.02, max(0., upcoming-(time.perf_counter()-start))))
                    ended = time.perf_counter()-start
                midpoint = (previous+ended)/2
                due = bisect.bisect_right(arrivals, midpoint)
                levels = dict(**current,
                              outstanding=due-bisect.bisect_right(completions, midpoint),
                              ingress=due-bisect.bisect_right(submitted, midpoint))
                gate.add(previous, ended, **levels)
                load.write(json.dumps(dict(start_s=previous, end_s=ended, **levels))+'\n')
                previous = ended
        finally:
            scheduler.schedule = original_schedule
            unfinished = engine.get_num_unfinished_requests()
            if unfinished:
                engine.abort_request([r['request_id'] for r in requests[:cursor]])
            if engine.has_unfinished_requests():
                raise RuntimeError('Abort did not clear unfinished requests')
            if active_epoch() is not None:
                stable_periods[-1]['end_s'] = previous
            for state in records.values():
                if state['status'] != 'completed':
                    state['status'] = 'aborted_at_end'
            with (output/f'latencies_gpu{gpu}.jsonl').open('w') as stream:
                for state in records.values():
                    stream.write(json.dumps(state)+'\n')
            latencies = latency_summary(records.values(), args.warmup)
            (output/f'latency_gpu{gpu}.json').write_text(json.dumps(latencies, indent=2))
    complete = sample_count == args.samples and stable_seconds(previous) >= args.measurement_duration
    return dict(worker=worker, gpu=gpu, samples=sample_count, required_samples=args.samples,
                request_routing=args.request_routing, per_instance_qps=args.per_instance_qps,
                status=('complete' if complete else 'never_stable' if first_qualified is None
                        else 'insufficient_stable_measurement'),
                first_qualified_s=first_qualified, elapsed_s=previous,
                stable_seconds=stable_seconds(previous), stable_periods=stable_periods,
                required_stable_seconds=args.measurement_duration, latency=latencies,
                submitted=cursor, completed=len(completions), aborted_at_end=unfinished)


def worker_main(worker, gpu, args, requests, arrivals, output, connection):
    """采样子进程入口，独占引擎与 Scheduler；结束时显式清理。"""
    output = Path(output)
    log = open(os.devnull if args.inject_failure else output/f'worker_gpu{gpu}.log', 'w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    original_print = builtins.print

    def quiet_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith('[SCHED_TRACE]'):
            return
        return original_print(*values, **kwargs)

    builtins.print = quiet_print
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0',
                      VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_NO_USAGE_STATS='1',
                      OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                      OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    if args.inject_failure:
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    for key in list(os.environ):
        if key.startswith(('KV_P1_', 'KV_P2_', 'KV_P3_', 'KV_PREFILL_', 'P1_KV_')):
            del os.environ[key]
    engine = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Worker termination requested')
    signal.signal(signal.SIGTERM, terminate)
    try:
        from transformers import AutoTokenizer
        import torch
        from vllm import EngineArgs, SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        specials = set(tokenizer.all_special_ids)
        allowed = tuple(i for i in range(min(tokenizer.vocab_size, args.model_vocab_size
                                            or tokenizer.vocab_size)) if i not in specials)
        if not allowed:
            raise ValueError('Tokenizer has no usable non-special tokens')
        warmup_token = allowed[0]
        requests = [dict(r, tokens=request_prompt(r, allowed, args.trace_block_tokens)) for r in requests]
        del tokenizer, allowed
        properties = torch.cuda.get_device_properties(0)
        identity = dict(gpu=gpu, pid=os.getpid(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                        trace=requests[0].get('trace_file', args.trace) if requests else args.trace,
                        cache_salt=requests[0].get('cache_salt') if requests else None,
                        device_uuid=str(properties.uuid), device_name=properties.name,
                        free_bytes_before_engine=torch.cuda.mem_get_info(0)[0],
                        **workload_identity(requests, arrivals))
        if not args.inject_failure:
            (output/f'identity_gpu{gpu}.json').write_text(json.dumps(identity, indent=2))
        blocks_override = None
        if getattr(args,'_prefailure_mode',None)=='replay':
            saved = json.loads((Path(args._prefailure_dir)/f'worker_{worker}.json').read_text())
            blocks_override = saved['kv_blocks']
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=args.model, dtype=args.dtype, tensor_parallel_size=1,
            max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.batch_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
            block_size=16, enable_prefix_caching=True, enable_chunked_prefill=True,
            async_scheduling=False, scheduling_policy='fcfs', enforce_eager=args.eager, seed=SEED,
            num_gpu_blocks_override=blocks_override,
            kv_transfer_config=None,
        ), enable_multiprocessing=False)
        scheduler = engine.engine_core.engine_core.scheduler
        warmup_input = min(128, max(1, args.max_model_len//2))
        engine.add_request('__initialization__', {'prompt_token_ids': [warmup_token]*warmup_input},
                           SamplingParams(temperature=0, max_tokens=min(8, args.max_model_len-warmup_input),
                                          ignore_eos=True, detokenize=False))
        while engine.has_unfinished_requests():
            engine.step()
        if not engine.engine_core.reset_prefix_cache(reset_connector=True):
            raise RuntimeError('Failed to clear initialization prefix cache')
        connection.send(dict(kind='ready', worker=worker, gpu=gpu,
                             dtype=str(engine.model_config.dtype), identity=identity,
                             cpu_cache_gb=args.cpu_cache_gb, kv_transfer_config=None,
                             kv_blocks=scheduler.kv_cache_config.num_blocks))
        command = connection.recv()
        if command is not None:
            time.sleep(max(0., command['start']-time.perf_counter()))
            replay_method = replay
            if args.inject_failure:
                from failure_sampling import replay_failure
                replay_method = replay_failure
            summary = replay_method(engine, scheduler, requests, arrivals, args, worker, gpu,
                                    output, connection, command['start'])
            if not args.inject_failure:
                (output/f'summary_gpu{gpu}.json').write_text(json.dumps(summary, indent=2))
            connection.send(dict(kind='result', **summary))
    except BaseException:
        try:
            connection.send(dict(kind='error', worker=worker, gpu=gpu, traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()
        connection.close()
        builtins.print = original_print
        log.close()


def run_workers(args, requests, arrivals, output, capture=None):
    # 父进程持有匿名 RAM，源实例 0 退出后备份仍在；
    # 旧策略对照每轮共用一池，让不同排队策略读取同一检查点。
    """启动所选实例并收集稳定性、采样和结果；旧 inject-failure 分支仍保留历史协议。"""
    owned_pool = None
    pool = getattr(args, '_cpu_kv_pool', None)
    if getattr(args, 'cpu_kv_backup_gb', 0) and pool is None:
        from cpu_kv_recovery import SharedKVPool
        owned_pool = pool = SharedKVPool(args.cpu_kv_backup_gb)
    ctx = mp.get_context('spawn')
    if getattr(args, 'failure_when_busy', False):
        args = copy.copy(args)
        args._failure_event = ctx.Event()
    processes, connections, summaries, ready = [], [], [], []
    original_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for worker, gpu in enumerate(args.gpus):
            parent, child = ctx.Pipe()
            worker_requests, worker_arrivals = worker_workload(args, requests, arrivals, worker)
            local = copy.copy(args)
            if hasattr(local, '_cpu_kv_pool'):
                del local._cpu_kv_pool
            if pool is not None:
                local._cpu_kv_handle = pool.share()
            local.per_instance_qps = args.instance_qps[worker]
            process = ctx.Process(target=worker_main, args=(
                worker, gpu, local, worker_requests, worker_arrivals,
                str(output), child))
            # spawn 前就设置设备掩码，避免子进程早期导入看到其他 GPU。
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        pending = set(connections)
        deadline = time.perf_counter()+args.startup_timeout
        while pending:
            readable = wait(pending, timeout=max(0., deadline-time.perf_counter()))
            if not readable:
                raise TimeoutError('Engine startup timed out; inspect worker_gpu*.log')
            for connection in readable:
                message = connection.recv()
                if message['kind'] != 'ready':
                    raise RuntimeError(str(message))
                ready.append(message)
                pending.remove(connection)
        if not args.inject_failure:
            (output/'engines.json').write_text(json.dumps(ready, indent=2))
        validate_replicas(args, ready)
        start = time.perf_counter()+2
        failure = None
        if args.inject_failure:
            from failure_sampling import FailureCoordinator
            failure = FailureCoordinator(args, connections, output, start, engines=ready,
                                         cpu_metadata=pool.metadata if pool is not None else None)
        for connection in connections:
            connection.send(dict(start=start))
        pending = set(connections)
        deadline = start+args.max_duration+(args.recovery_drain if failure else 0)+120
        while pending:
            readable = wait(pending, timeout=min(5., max(0., deadline-time.perf_counter())))
            if not readable:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Replay timed out; inspect worker logs')
                continue
            for connection in readable:
                message = connection.recv()
                kind, gpu = message['kind'], message['gpu']
                if kind == 'error':
                    raise RuntimeError(message['traceback'])
                if failure is not None and failure.handle(message):
                    continue
                if kind == 'result':
                    summaries.append(message)
                    pending.remove(connection)
                elif kind == 'sample':
                    print(f"GPU={gpu} sample={message['sample']}/{args.samples} "
                          f"t={message['time_s']:.2f}s running={message['running_count']} "
                          f"waiting={message['waiting_count']} IDs={message['running_ids']} "
                          f"source_index={message['source_indices']} source_line={message['source_lines']}", flush=True)
                elif kind == 'stability':
                    failed = [k for k, passed in message['checks'].items() if not passed]
                    print(f"GPU={gpu} t={message['time_s']:.1f}s "
                          f"stable_checks={message['consecutive_passes']}/{args.stable_checks} "
                          f"arrived/completed={message['offered']}/{message['completed']} "
                          f"failed={failed}", flush=True)
        if failure is not None:
            summary = failure.finalize()
            if capture is not None:
                capture.update(before=failure.before_report,plan=failure.plan,after=failure.after_report)
            return summary
        return sorted(summaries, key=lambda s: s['worker'])
    finally:
        if original_visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_visible
        # 正常退出由 worker 关闭引擎；中断时先给子进程时间清理，再强制回收。
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in connections:
            connection.close()
        if owned_pool is not None:
            owned_pool.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 5, 6, 7])
    parser.add_argument('--qps', type=float, default=1.5,
                        help='Four-instance total QPS by default, including replicated copies')
    parser.add_argument('--qps-scope', choices=['global', 'per-instance'], default='global')
    parser.add_argument('--source-qps', type=float,
                        help='Explicit source rate; use with --survivor-qps and --survivor-trace, overrides --qps')
    parser.add_argument('--survivor-qps', type=float, help='Explicit rate of each background instance')
    parser.add_argument('--request-routing', choices=['replicate', 'round-robin'], default='replicate')
    parser.add_argument('--trace-order', choices=['file', 'timestamp'], default='file')
    parser.add_argument('--model', default='/data1/jhcheng/models/scchen/Qwen3-8B')
    parser.add_argument('--trace', default=str(Path(__file__).parent/'data/toolagent_trace.jsonl'))
    parser.add_argument('--survivor-trace',
                        help='Failure mode: instance 0 uses --trace; instances 1/2/3 replicate this trace')
    parser.add_argument('--recovery-cache-policy', choices=['shared'], default='shared',
                        help='All normal, forwarded and recovered requests share the local prefix pool without cache salts')
    parser.add_argument('--trace-block-tokens', type=int)
    parser.add_argument('--trace-time-unit', choices=['auto', 's', 'ms'], default='auto')
    parser.add_argument('--arrival', choices=['constant', 'trace'], default='constant')
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.90)
    parser.add_argument('--max-num-seqs', type=int, default=256)
    parser.add_argument('--batch-tokens', type=int, default=8192)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--cpu-cache-gb', type=float, choices=[0.0], default=0.0,
                        help='Generic CPU prefix offload stays disabled; live checkpoints use --cpu-kv-backup-gb')
    parser.add_argument('--cpu-kv-backup-gb', type=float, default=0.,
                        help='Source-only live KV checkpoint RAM budget (decimal GB); 0 disables. '
                             'Full blocks copy asynchronously; coordinator owns the RAM. Enables resume by default.')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--warmup', type=float, default=60, help='Earliest stability qualification, not a fixed sampling start')
    parser.add_argument('--window', type=float, default=60)
    parser.add_argument('--check-every', type=float, default=10)
    parser.add_argument('--stable-checks', type=int, default=3)
    parser.add_argument('--flow-tolerance', type=float, default=.15)
    parser.add_argument('--trend-tolerance', type=float, default=.05)
    parser.add_argument('--level-tolerance', type=float, default=.25)
    parser.add_argument('--count-tolerance', type=float, default=2)
    parser.add_argument('--max-ingress', type=float, default=1)
    parser.add_argument('--ingress-policy', choices=['caught-up','bounded'], default='caught-up',
                        help='Bounded checks that pending ingress does not grow; caught-up also caps its absolute mean')
    parser.add_argument('--ttft-stability', action='store_true',
                        help='Also require both rolling-window halves to have stable mean TTFT and >=3 samples')
    parser.add_argument('--ttft-relative-tolerance', type=float, default=.25)
    parser.add_argument('--ttft-absolute-tolerance-ms', type=float, default=200.)
    parser.add_argument('--samples', type=int, default=5, help='Snapshots per GPU')
    parser.add_argument('--measurement-duration', type=float, default=180,
                        help='Required cumulative qualified time for normal-inference latency statistics')
    parser.add_argument('--sample-gap', type=float, nargs=2, default=[10., 30.], metavar=('MIN', 'MAX'))
    parser.add_argument('--running-log-every', type=float, default=1,
                        help='Record native running IDs at this interval, including unqualified states')
    parser.add_argument('--max-duration', type=float, default=600, help='Replay timeout, excludes engine startup')
    parser.add_argument('--startup-timeout', type=float, default=1200)
    parser.add_argument('--inject-failure', action='store_true',
                        help='Fail instance 0: recover running at queue head; forward waiting/future to 1/2/3')
    parser.add_argument('--require-stable-before-failure', action='store_true',
                        help='Abort if the selected stability scope fails rolling checks at the fault')
    parser.add_argument('--failure-stability-scope', choices=['source', 'all'], default='source',
                        help='Only require the failed source to be stable by default; background rates are independent')
    parser.add_argument('--failure-when-busy', action='store_true',
                        help='After the random time, fail at the next nonempty source running boundary; notify survivors')
    parser.add_argument('--recovery-scheduler', choices=['equal','prefill-balanced','kv-aware'], default='equal',
                        help='Equal counts; full-prefill LPT; or shared-prefix groups with overload splitting')
    queue_orders = ['source','interleave','prefix-shortest','shortest','group-first','group-ratio']
    parser.add_argument('--recovery-order',choices=queue_orders,default='prefix-shortest',
                        help='Within each target: least estimated new prefill first by default; '
                             'source restores snapshot order; no placement change or delayed admission')
    parser.add_argument('--compare-recovery-schedulers',action='store_true',
                        help='Strict matched comparison: record the first combination, replay for all others')
    parser.add_argument('--comparison-orders',nargs='+',choices=queue_orders,
                        help='Compare each scheduler with these queue orders on the same fault state; shortest aliases prefix-shortest')
    parser.add_argument('--comparison-cases',nargs='+',metavar='SCHEDULER/ORDER',
                        help='Compare only these explicit combinations instead of the full scheduler/order product')
    parser.add_argument('--comparison-repeats',type=int,default=1,
                        help='Sample a new failure time per repeat; three strategies share that repeat\'s baseline')
    parser.add_argument('--kv-min-shared-tokens', type=int, default=16,
                        help='kv-aware: minimum shared prefix, a positive multiple of 16')
    parser.add_argument('--kv-split-factor', type=float, default=1.2,
                        help='kv-aware: group placement budget multiplier (>=1), default 1.2')
    parser.add_argument('--failure-delay', type=float, nargs=2, default=[10., 30.],
                        metavar=('MIN', 'MAX'), help='Uniform random seconds after warmup before failure')
    parser.add_argument('--recovery-mode', choices=['resume', 'restart'], default=None,
                        help='Defaults to restart without CPU checkpoints, resume with them. '
                             'Resume preserves saved output and measures the next new token.')
    parser.add_argument('--recovery-drain', type=float, default=180,
                        help='Maximum extra seconds after max-duration to drain all pending traffic')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--output', default=f'results/running_samples_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.cpu_kv_backup_gb) or args.cpu_kv_backup_gb < 0:
        parser.error('--cpu-kv-backup-gb must be finite and nonnegative')
    args.recovery_mode = args.recovery_mode or ('resume' if args.cpu_kv_backup_gb else 'restart')
    if args.cpu_kv_backup_gb and (not args.inject_failure or args.recovery_mode != 'resume'):
        parser.error('Live CPU KV recovery requires --inject-failure and --recovery-mode resume')
    positive = ('qps', 'window', 'check_every', 'max_duration', 'startup_timeout',
                'flow_tolerance', 'trend_tolerance', 'level_tolerance', 'count_tolerance', 'gpu_memory_utilization',
                'measurement_duration', 'running_log_every', 'ttft_relative_tolerance',
                'ttft_absolute_tolerance_ms')
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        parser.error('Rates, windows, timeouts and tolerances must be positive and finite')
    if (any(not math.isfinite(v) or v < 0 for v in [args.warmup, args.max_ingress, *args.sample_gap])
            or not 0 < args.sample_gap[0] <= args.sample_gap[1]
            or args.gpu_memory_utilization > 1):
        parser.error('Invalid warmup, ingress bound, sample gap or GPU utilization')
    if (not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)
            or min(args.stable_checks, args.samples, args.max_num_seqs, args.batch_tokens, args.cpu_threads) < 1):
        parser.error('GPUs must be unique/nonnegative; integer counts must be positive')
    if (not math.isfinite(args.recovery_drain) or args.recovery_drain < 0
            or any(not math.isfinite(v) or v < 0 for v in args.failure_delay)
            or args.failure_delay[0] > args.failure_delay[1]):
        parser.error('Failure delay and recovery drain must be finite and nonnegative')
    if args.inject_failure:
        if len(args.gpus) != 4:
            parser.error('--inject-failure requires four independent instances: 0 fails, 1/2/3 survive')
        if args.warmup+args.failure_delay[1] >= args.max_duration:
            parser.error('The full failure-delay range must fit after warmup and before max-duration')
    if (args.require_stable_before_failure or args.failure_when_busy) and not args.inject_failure:
        parser.error('--require-stable-before-failure and --failure-when-busy require --inject-failure')
    if args.compare_recovery_schedulers and not args.inject_failure:
        parser.error('--compare-recovery-schedulers requires --inject-failure')
    if args.comparison_orders is not None and not args.compare_recovery_schedulers:
        parser.error('--comparison-orders requires --compare-recovery-schedulers')
    if args.comparison_cases is not None:
        if not args.compare_recovery_schedulers or args.comparison_orders is not None:
            parser.error('--comparison-cases requires --compare-recovery-schedulers and excludes --comparison-orders')
        cases = []
        for case in args.comparison_cases:
            parts = case.split('/')
            if len(parts)!=2 or parts[0] not in ('equal','prefill-balanced','kv-aware') or parts[1] not in queue_orders:
                parser.error('Each --comparison-cases value must be a valid SCHEDULER/ORDER')
            cases.append((parts[0],'prefix-shortest' if parts[1]=='shortest' else parts[1]))
        if len(set(cases))!=len(cases):
            parser.error('--comparison-cases must contain distinct combinations')
        if any(order in ('group-first','group-ratio') and strategy!='kv-aware' for strategy,order in cases):
            parser.error('Group queue orders require kv-aware group assignments')
        args.comparison_cases = cases
        args.comparison_orders = list(dict.fromkeys(order for _,order in cases))
    args.recovery_order = 'prefix-shortest' if args.recovery_order=='shortest' else args.recovery_order
    args.comparison_orders = ['prefix-shortest' if order=='shortest' else order
                              for order in (args.comparison_orders or [args.recovery_order])]
    if len(set(args.comparison_orders))!=len(args.comparison_orders):
        parser.error('--comparison-orders must contain distinct queue orders')
    group_orders = {'group-first','group-ratio'}
    if (not args.comparison_cases and args.compare_recovery_schedulers and group_orders.intersection(args.comparison_orders)):
        parser.error('Use --comparison-cases to compare group queue orders only with kv-aware')
    if not args.compare_recovery_schedulers and args.recovery_order in group_orders and args.recovery_scheduler!='kv-aware':
        parser.error('Group queue orders require --recovery-scheduler kv-aware')
    if args.comparison_repeats<1 or (args.comparison_repeats!=1 and not args.compare_recovery_schedulers):
        parser.error('--comparison-repeats must be positive and requires --compare-recovery-schedulers')
    if args.compare_recovery_schedulers and args.comparison_repeats>1 and args.failure_delay[0]==args.failure_delay[1]:
        parser.error('Repeated comparisons require --failure-delay MIN MAX with MIN < MAX to sample new fault times')
    if not args.inject_failure and args.recovery_scheduler != 'equal':
        parser.error('--recovery-scheduler requires --inject-failure')
    elif not args.inject_failure and args.max_duration <= max(args.warmup, args.window)+(args.stable_checks-1)*args.check_every:
        parser.error('--max-duration must leave time for stability qualification and sampling')
    failure_rng = random.Random(args.seed)
    args.comparison_failure_times_s = [args.warmup+failure_rng.uniform(*args.failure_delay)
        for _ in range(args.comparison_repeats)] if args.inject_failure else []
    args.failure_time_s = args.comparison_failure_times_s[0] if args.inject_failure else None
    if args.survivor_trace and args.request_routing != 'replicate':
        parser.error('--survivor-trace requires --request-routing replicate')
    if args.survivor_trace and len(args.gpus) < 2:
        parser.error('--survivor-trace requires a source and at least one background instance')
    if args.source_qps is not None or args.survivor_qps is not None:
        if (not args.survivor_trace or min(args.source_qps or 0, args.survivor_qps or 0) <= 0
                or any(not math.isfinite(v) for v in (args.source_qps or 0, args.survivor_qps or 0))):
            parser.error('Decoupled rates require --survivor-trace and positive finite --source-qps/--survivor-qps')
    if args.kv_min_shared_tokens<16 or args.kv_min_shared_tokens%16 or not 1<=args.kv_split_factor<math.inf:
        parser.error('KV prefix must be a positive multiple of 16; split factor must be finite and >=1')
    args.latency_measurement = ('resume_cpu_kv_next_token_v1' if args.cpu_kv_backup_gb
                                else 'resume_next_token_v1' if args.recovery_mode=='resume'
                                else 'restart_first_token_v2')
    args.recovery_execution = 'source_order_native_queue_v3'
    args.instances = len(args.gpus)
    args.aggregate_offered_qps = args.qps*(args.instances if args.qps_scope == 'per-instance' else 1)
    args.per_instance_qps = args.aggregate_offered_qps/args.instances
    args.instance_qps = [args.per_instance_qps]*args.instances
    if args.source_qps is not None:
        args.instance_qps = [args.source_qps]+[args.survivor_qps]*(args.instances-1)
        args.aggregate_offered_qps = sum(args.instance_qps)
        args.per_instance_qps = None
    args.logical_qps = (args.per_instance_qps if args.request_routing == 'replicate'
                        else args.aggregate_offered_qps)
    if args.source_qps is not None:
        args.logical_qps = args.source_qps
    return args


def main():
    """采样入口：用共用输入读取器准备数据；普通采样与旧故障分支分别保存对应结果。"""
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if args.inject_failure:
        from failure_sampling import initialize_results, experiment_config, write_result, RESULT_FILES
        initialize_results(output, experiment_config(args))
        try:
            # 共用读取器需要临时输入清单，这些文件不放进公开实验结果目录。
            with tempfile.TemporaryDirectory(prefix='failure_inputs_') as temporary:
                requests, arrivals = load_probe_traces(args, Path(temporary))
            initialize_results(output, experiment_config(args))
            if args.dry_run:
                print(f'Dry run: {output}; three result files initialized', flush=True)
                return
            if args.compare_recovery_schedulers:
                from failure_sampling import compare_recovery_schedulers
                compare_recovery_schedulers(args,requests,arrivals,output)
                print(f'Comparison complete: {output}',flush=True)
                return
            summary = run_workers(args, requests, arrivals, output)
        except BaseException:
            write_result(output/RESULT_FILES[2], dict(status='error', error=traceback.format_exc()))
            raise
        print(f"QPS={args.aggregate_offered_qps:g}: recovery={summary['completed']}/{summary['requests']}, "
              f"forwarded={summary['forwarding']['completed']}/{summary['forwarding']['requests']}; "
              f"results={output}", flush=True)
        if (summary['status'] != 'complete' or summary['forwarding']['status'] != 'complete'
                or summary['traffic']['unfinished']):
            raise SystemExit(2)
        return
    # 这里只复用输入读取器，不导入或执行普通扫描入口 run.py。
    requests, arrivals = load_probe_traces(args, output)
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    source = output/'source'
    source.mkdir()
    for name in ('sample_running.py', 'failure_sampling.py', 'recovery_scheduling.py', 'cpu_kv_recovery.py', 'running_report.py', 'workload.py',
                 'worker.py', 'metrics.py', 'report.py', 'test_failure_sampling.py'):
        shutil.copy2(Path(__file__).parent/name, source/name)
    for name in ('failure_sampling.md',):
        shutil.copy2(Path(__file__).parent/'docs'/name, source/name)
    mode = (f'instance 0 fails at planned t={args.failure_time_s:.3f}s; '
            f'equal running recovery + source waiting/future forwarding on 1/2/3; '
            f'mode={args.recovery_mode}' if args.inject_failure else
            f'{args.samples} random snapshots per GPU after stability')
    print(f'Routing={args.request_routing}; QPS scope={args.qps_scope}; '
          f'instance QPS={args.instance_qps}; logical source trace QPS={args.logical_qps:g}; '
          f'aggregate offered QPS={args.aggregate_offered_qps:g}; trace order={args.trace_order}; '
          f'{args.instances} independent TP=1 instances; '
          f'{mode}; CPU KV offload disabled', flush=True)
    if args.survivor_trace:
        print(f'Instance 0 trace={args.trace}; instances 1/2/3 trace={args.survivor_trace}; '
              f'shared local prefix pool without cache salts', flush=True)
    if args.dry_run:
        print(f'Dry run: input manifest saved to {output}; no GPU initialized', flush=True)
        return
    try:
        summaries = run_workers(args, requests, arrivals, output)
        (output/'summary.json').write_text(json.dumps(summaries, indent=2))
        write_csv(output/'summary.csv', [dict(
            gpu=s['gpu'], status=s['status'], samples=s['samples'], stable_seconds=s['stable_seconds'],
            request_routing=args.request_routing, per_instance_qps=s['per_instance_qps'],
            aggregate_offered_qps=args.aggregate_offered_qps, qps_scope=args.qps_scope,
            **{f'{cohort}_{key}': value for cohort in ('stable', 'post_warmup')
               for key, value in s['latency'][cohort].items()},
            unfinished=s['latency']['unfinished']) for s in summaries])
    except BaseException:
        (output/'error.txt').write_text(traceback.format_exc())
        raise
    print(json.dumps(summaries, indent=2), flush=True)
    if any(s['status'] != 'complete' for s in summaries):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
