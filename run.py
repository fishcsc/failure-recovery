# 普通 QPS/CPU offload 扫描入口；main 建计划，worker 执行，metrics/report 汇总。
# 每个配置创建新引擎，避免上一组缓存污染下一组；不负责当前故障恢复实验。
"""Run independent per-GPU QPS and CPU KV-cache experiments."""
import argparse
import csv
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import subprocess
import shutil
import threading
import time

from worker import worker_main
from workload import configure_inputs, load_trace
from metrics import aggregate_results, transfer_config
from report import write_csv, write_matched_comparison, extract

MODEL = '/data1/jhcheng/models/scchen/Qwen3-8B'


def trace_shard(worker, offset=0, instances=4):
    """跨重复轮次轮换固定数据分片对应的 GPU，分片内部请求顺序不变。"""
    return (worker-offset) % instances


def host_snapshot():
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'memory_kib': {k: int(mem[k].split()[0]) for k in
                          ('MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree', 'Mlocked', 'Unevictable')},
            'vm_counters': {k: int(vm[k]) for k in ('pswpin', 'pswpout', 'pgmajfault')}}


def receive_all(connections, kind, timeout):
    """等待每个子进程的指定消息并按 worker 返回；任一子进程错误就终止本组。"""
    pending = set(connections)
    messages = []
    deadline = time.perf_counter()+timeout
    while pending:
        readable = wait(pending, timeout=max(0, deadline-time.perf_counter()))
        if not readable:
            raise TimeoutError(f'Waiting for {kind} timed out; inspect worker*.log')
        for connection in readable:
            try:
                msg = connection.recv()
            except EOFError as e:
                raise RuntimeError('Worker terminated; inspect worker*.log') from e
            if msg.get('kind') != kind:
                raise RuntimeError(f'Worker error: {msg}')
            messages.append(msg)
            pending.remove(connection)
    return sorted(messages, key=lambda r: r['worker'])


class GPUUtilSampler:
    """父进程独立线程采样 nvidia-smi，不把监控等待塞入引擎 Step。"""
    def __init__(self, gpus, start, period=1.0):
        self.gpus = set(gpus)
        self.start = start
        self.period = period
        self.rows = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            stamp = time.perf_counter()-self.start
            try:
                result = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits',
                ], text=True, timeout=5)
                for line in result.strip().splitlines():
                    cols = [s.strip() for s in line.split(',')]
                    if len(cols) < 3:
                        continue
                    gpu = int(cols[0])
                    if gpu in self.gpus:
                        self.rows.append({
                            'time_s': stamp, 'gpu': gpu,
                            'util_pct': float(cols[1].replace('%','')),
                            'memory_mib': float(cols[2].replace('MiB','')),
                        })
            except (OSError, ValueError, subprocess.SubprocessError) as e:
                self.error = str(e)
            self.stop_event.wait(self.period)

    def start_sampling(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=8)
        return self.rows


def available_host_bytes():
    """同时考虑主机 MemAvailable 和容器内存限额，返回可用字节数。"""
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0]) * 1024
    limit_path = Path('/sys/fs/cgroup/memory.max')
    used_path = Path('/sys/fs/cgroup/memory.current')
    if limit_path.exists() and used_path.exists():
        limit = limit_path.read_text().strip()
        if limit != 'max':
            available = min(available, max(0, int(limit)-int(used_path.read_text())))
    return available


def check_cpu_capacity(cpu_cache_gb, instances=4, storage='private'):
    """普通 offload 启动前检查全部实例容量总和，按所选后端额外检查共享内存。"""
    requested = int(cpu_cache_gb * 1_000_000_000) * instances
    if not requested:
        return
    available = available_host_bytes()
    shm = shutil.disk_usage('/dev/shm').free if storage == 'shm' else None
    reserve = 8_000_000_000
    if requested + reserve > available or (shm is not None and requested > shm):
        storage_info = (f'/dev/shm={shm/1e9:.1f} GB' if shm is not None
                        else 'private pinned CPU tensors; /dev/shm not used')
        raise RuntimeError(
            f'{instances} instances x {cpu_cache_gb:g} GB need {requested/1e9:.1f} GB '
            f'CPU KV memory; available host={available/1e9:.1f} GB '
            f'(reserve {reserve/1e9:g} GB), {storage_info}. '
            'Reduce --cpu-cache-gb or free memory; capacities are per instance.')


def run_configuration(args, requests, schedules, output):
    """为一个 QPS/容量/重复组合创建全新实例，返回逐卡和整体汇总。

    到达表按固定分片分给各 worker，ready 后统一起跑；结束回收全部子进程。
    """
    if len(args.qps) != 1:
        raise ValueError('Each engine group must run exactly one QPS')
    instances = len(args.gpus)
    ctx = mp.get_context('spawn')
    (output/'host_before.json').write_text(json.dumps(host_snapshot(), indent=2))
    connections, processes, summaries, aggregates = [], [], [], []
    try:
        for worker, gpu in enumerate(args.gpus):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=worker_main,
                args=(worker, gpu, args,
                      requests[trace_shard(worker, args.routing_offset, instances)::instances],
                      str(output), child),
            )
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)

        qps = args.qps[0]
        ready = receive_all(connections, 'ready', timeout=900)
        (output/'engines.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')
        start = time.perf_counter()+3
        for worker, connection in enumerate(connections):
            connection.send({
                'qps': qps, 'start': start,
                'arrivals': schedules[qps][trace_shard(worker, args.routing_offset, instances)::instances],
            })
        sampler = GPUUtilSampler(args.gpus, start)
        sampler.start_sampling()
        try:
            results = receive_all(connections, 'result', args.duration+180)
        finally:
            samples = sampler.finish()
        write_csv(output/f'gpu_qps_{qps:g}.csv', samples)
        trial_rows = []
        for result in results:
            row = result['summary']
            selected = [s for s in samples if s['gpu'] == row['gpu']
                        and args.warmup <= s['time_s'] < args.duration]
            row['avg_gpu_util_pct'] = (sum(s['util_pct'] for s in selected)/len(selected)
                                       if selected else None)
            row['avg_gpu_mem_mib'] = (sum(s['memory_mib'] for s in selected)/len(selected)
                                      if selected else None)
            row['gpu_util_samples'] = len(selected)
            row['case_directory'] = str(output)
            trial_rows.append(row)
            print(f"CPU={args.cpu_cache_gb:g}GB/instance repeat={args.repeat} "
                  f"QPS={qps:g} GPU={row['gpu']} stable={row['looks_stable']} "
                  f"running={row['mean_running']:.2f} waiting={row['mean_waiting']:.2f} "
                  f"TTFT={row['avg_ttft_ms']} TPOT={row['avg_tpot_ms']} "
                  f"ITL={row['window_avg_itl_ms']} "
                  f"CPU_load_GB={row['cpu_load_bytes']/1e9:.3f}", flush=True)
        summaries.extend(trial_rows)
        aggregates.append(aggregate_results(trial_rows, output, args, qps))
        if sampler.error:
            print(f'GPU monitoring warning: {sampler.error}', flush=True)
        write_csv(output/'summary.csv', summaries)
        (output/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        write_csv(output/'aggregate.csv', aggregates)
    finally:
        for connection in connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in processes:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        for connection in connections:
            connection.close()
        (output/'host_after.json').write_text(json.dumps(host_snapshot(), indent=2))
    return summaries, aggregates


def build_plan(args, output):
    """为每个容量×重复×QPS 建独立目录和参数，返回执行计划。

    偶数轮反转容量顺序，可选轮换 GPU 分片，减小运行顺序和硬件的混杂影响。
    """
    plan = []
    for repeat in range(1, args.repeats + 1):
        capacities = args.cpu_cache_gb if repeat % 2 else list(reversed(args.cpu_cache_gb))
        for capacity in capacities:
            for qps in args.qps:
                directory = output / f'cpu_gb_{capacity:g}/repeat_{repeat}/qps_{qps:g}'
                directory.mkdir(parents=True)
                case = argparse.Namespace(**vars(args))
                case.cpu_cache_gb, case.repeat, case.qps = capacity, repeat, [qps]
                case.instances = len(args.gpus)
                case.routing_offset = (repeat - 1) % case.instances if args.rotate_workers else 0
                metadata = dict(vars(case), kv_transfer_config=transfer_config(capacity, args.cpu_cache_storage))
                (directory/'config.json').write_text(json.dumps(metadata, indent=2))
                (directory/'routing.json').write_text(json.dumps([
                    {'worker': worker, 'gpu': gpu,
                     'trace_shard': trace_shard(worker, case.routing_offset, case.instances)}
                    for worker, gpu in enumerate(args.gpus)], indent=2))
                plan.append((case, directory))
    return plan


def main():
    """普通扫描入口：参数校验、数据/计划生成、逐组运行，再汇总匹配请求结果。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 0, 4, 5],
                   help='Physical GPU IDs; one independent TP=1 instance per GPU')
    p.add_argument('--qps', type=float, nargs='+', default=[2],
                   help='Global offered QPS across all selected instances (default: 2)')
    p.add_argument('--cpu-cache-gb', type=float, nargs='+', default=[0],
                   help='CPU KV-cache GB per instance; 0 = GPU prefix cache only')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--cpu-threads', type=int, default=1,
                   help='OMP/MKL/OpenBLAS threads per engine (default: 1)')
    p.add_argument('--scheduler-trace', action='store_true',
                   help='Keep the local vLLM per-step debug prints (off by default)')
    p.add_argument('--rotate-workers', action='store_true',
                   help='Rotate trace shards across GPUs each repeat to separate workload and hardware effects')
    p.add_argument('--cpu-cache-storage', choices=['private', 'shm'], default='private',
                   help='CPU buffer allocation: private pinned RAM (default) or /dev/shm')
    p.add_argument('--gpu-kv-cache-gb', type=float,
                   help='Optional fixed GPU KV-cache GB per instance, to isolate CPU capacity')
    p.add_argument('--duration', type=float, default=180)
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--arrival', choices=['constant', 'trace'], default='constant')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--trace', default=str(Path(__file__).parent/'data/toolagent_trace.jsonl'))
    p.add_argument('--trace-block-tokens', type=int,
                   help='Tokens per trace hash; auto: block_size, filename blksz_N, Qwen=16, legacy=512')
    p.add_argument('--trace-time-unit', choices=['auto', 's', 'ms'], default='auto',
                   help='Trace timestamp unit; auto: Qwen chat schema=s, legacy=ms')
    p.add_argument('--max-model-len', type=int, help='Optional context cap, at most the model config limit')
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    p.add_argument('--output', required=True)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--dry-run', action='store_true',
                   help='Write schedules/configs without importing vLLM or using GPUs')
    args = p.parse_args()

    if len(set(args.gpus)) != len(args.gpus) or len(set(args.qps)) != len(args.qps):
        p.error('GPUs and QPS levels must be unique')
    if any(gpu < 0 for gpu in args.gpus):
        p.error('GPU IDs must be non-negative')
    args.instances = len(args.gpus)
    if any(not math.isfinite(q) or q <= 0 for q in args.qps):
        p.error('All QPS values must be finite and positive')
    if (len(set(args.cpu_cache_gb)) != len(args.cpu_cache_gb)
            or any(not math.isfinite(g) or g < 0 for g in args.cpu_cache_gb)):
        p.error('CPU cache capacities must be unique, finite and non-negative')
    if args.repeats < 1:
        p.error('--repeats must be positive')
    if args.cpu_threads < 1:
        p.error('--cpu-threads must be positive')
    if args.trace_block_tokens is not None and args.trace_block_tokens <= 0:
        p.error('--trace-block-tokens must be positive')
    if args.max_model_len is not None and args.max_model_len <= 0:
        p.error('--max-model-len must be positive')
    if args.gpu_kv_cache_gb is not None and (
            not math.isfinite(args.gpu_kv_cache_gb) or args.gpu_kv_cache_gb <= 0):
        p.error('--gpu-kv-cache-gb must be finite and positive')
    if (not math.isfinite(args.duration) or not math.isfinite(args.warmup)
            or not (0 <= args.warmup < args.duration)
            or args.duration-args.warmup < 30):
        p.error('Measurement window must be finite and at least 30 seconds')

    if not args.dry_run:
        for gpu in args.gpus:
            used = subprocess.check_output([
                'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
            ], text=True)
            if int(used.strip()) > 1024:
                raise RuntimeError(f'GPU {gpu} seems occupied ({used.strip()} MiB)')
        check_cpu_capacity(max(args.cpu_cache_gb), instances=args.instances,
                           storage=args.cpu_cache_storage)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests, schedules = load_trace(args, output)
    (output/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    if not args.dry_run:
        (output/'gpu_topology.txt').write_text(subprocess.check_output(
            ['nvidia-smi', 'topo', '-m'], text=True))
    source_dir = output/'source'
    source_dir.mkdir()
    for name in ('run.py', 'workload.py', 'sample_running.py', 'failure_sampling.py', 'recovery_scheduling.py', 'running_report.py',
                 'worker.py', 'metrics.py', 'report.py', 'audit.py',
                 'cpu_offload_spec.py', 'launch.sh', 'README.md', 'test_experiment.py'):
        shutil.copy2(Path(__file__).parent/name, source_dir/name)
    shutil.copytree(Path(__file__).parent/'docs', source_dir/'docs')
    plan = build_plan(args, output)
    (output/'execution_order.json').write_text(json.dumps([
        {'repeat': a.repeat, 'cpu_cache_gb': a.cpu_cache_gb, 'global_qps': a.qps[0],
         'routing_offset': a.routing_offset, 'directory': str(d)} for a, d in plan], indent=2))
    (output/'schedule.json').write_text(json.dumps({
        str(qps): {'arrivals_s': times,
                   'planned': len(times),
                   'full_run_offered_qps': len(times)/args.duration,
                   'measurement_offered_qps': sum(
                       args.warmup <= t < args.duration for t in times)
                       / (args.duration-args.warmup)}
        for qps, times in schedules.items()
    }, indent=2))
    print(f'CPU cache capacities: {args.cpu_cache_gb} GB per instance; '
          f'{args.instances}-instance peak CPU KV budget='
          f'{args.instances*max(args.cpu_cache_gb):g} GB; '
          f'{len(plan)} fresh engine groups', flush=True)
    print(f'Global QPS levels: {args.qps}; per-instance average QPS: '
          f'{[qps/args.instances for qps in args.qps]}; round-robin routing', flush=True)
    if args.dry_run:
        print(f'Dry run prepared: {output}', flush=True)
        return

    summaries, aggregates = [], []
    for case_args, directory in plan:
        check_cpu_capacity(case_args.cpu_cache_gb, instances=args.instances,
                           storage=args.cpu_cache_storage)
        rows, combined = run_configuration(case_args, requests, schedules, directory)
        summaries.extend(rows)
        aggregates.extend(combined)
        write_csv(output/'summary.csv', summaries)
        (output/'summary.json').write_text(json.dumps(summaries, indent=2))
        write_csv(output/'aggregate.csv', aggregates)
        (output/'aggregate.json').write_text(json.dumps(aggregates, indent=2))
        write_matched_comparison(output)
    extract(output)


if __name__ == '__main__':
    main()
