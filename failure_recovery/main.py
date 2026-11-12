# 当前故障实验入口：参数 → workload 输入 → experiment 协调 → 三份报告。
# 先读 main，再按需读 parse_args；GPU 循环在 engine.FailureRunner.run。
"""Independent failure-recovery and request-scheduling experiment entry point.

Run from request_count with: python -m failure_recovery.main --help
"""
import argparse
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

# 直接执行该文件时也能找到项目模块，不依赖当前工作目录。
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 直接导入共用函数，同时保留旧代码从入口模块导入这些名字的用法。
from workload import (configure_inputs, qa_request, load_trace,
                      load_probe_traces as load_inputs)

from failure_recovery.experiment import (compare_recovery_schedulers,
                                        experiment_config, run_workers)
from failure_recovery.engine import ReplayContext
from failure_recovery.report import (FAILURE_REPORT_FILES, initialize_results,
                                     write_failure_report)

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    """故障实验入口：按 argv 准备输入，再执行单组或严格策略对照。

    输出目录只保留三份 Markdown；不完整恢复或回放分歧会报错，正常结束返回 None。
    """
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if args.profile_backup_overhead:
        args.profile_dir=str(Path(args.profile_dir or ROOT/'backups'/
            (output.parent.name+'_'+output.name+'_profiles')).expanduser().resolve())
        directory=Path(args.profile_dir)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Refusing to overwrite existing profiles')
        directory.mkdir(parents=True,exist_ok=True)
    initialize_results(output, experiment_config(args))
    try:
        # 输入清单只用于准备和核对；公开目录仍只保留三份报告。
        with tempfile.TemporaryDirectory(prefix='failure_inputs_') as temporary:
            requests, arrivals = load_inputs(args, Path(temporary))
        initialize_results(output, experiment_config(args))
        if args.dry_run:
            print(f'Dry run: {output}; three reports initialized; no GPU started', flush=True)
            return
        if args.compare_recovery_schedulers:
            compare_recovery_schedulers(args, requests, arrivals, output)
        else:
            context = None
            if args.inference_replay:
                # 参考文件只控制推理现场；各备份策略仍独立选块、复制和发布目录。
                directory=Path(args.inference_replay_dir).expanduser().resolve()
                directory.mkdir(parents=True,exist_ok=True)
                files=list(directory.glob('worker_*.json'))
                if args.inference_replay=='record' and files:
                    raise ValueError('Refusing to overwrite an existing inference reference')
                if args.inference_replay=='replay' and {p.name for p in files}!={
                        f'worker_{i}.json' for i in range(args.instances)}:
                    raise ValueError('Inference replay requires one reference journal per instance')
                context=ReplayContext(mode=args.inference_replay,directory=str(directory),inference_only=True,
                                      diagnostics_directory=args.replay_diagnostics_dir)
            summary = run_workers(args, requests, arrivals, output,context=context)
            if (summary['status'] != 'complete' or summary['forwarding']['status'] != 'complete'
                    or summary['traffic']['unfinished']):
                raise RuntimeError('Recovery or forwarded traffic did not complete')
        print(f'Experiment complete: {output}', flush=True)
    except BaseException:
        write_failure_report(output/FAILURE_REPORT_FILES[2],
            dict(status='error', error=traceback.format_exc()))
        raise


def parse_args(argv=None):
    """解析并校验实验参数，返回带派生速率和故障时刻的 Namespace。

    故障时刻由 seed 与预热后区间确定；只做参数准备，不启动 GPU。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 5, 6, 7])
    parser.add_argument('--qps', type=float, default=1.5,
                        help='Total QPS across all instances by default, including replicated copies')
    parser.add_argument('--qps-scope', choices=['global', 'per-instance'], default='global')
    parser.add_argument('--source-qps', type=float,
                        help='Explicit source rate; use with --survivor-qps and --survivor-trace, overrides --qps')
    parser.add_argument('--survivor-qps', type=float, help='Explicit rate of each background instance')
    parser.add_argument('--request-routing', choices=['replicate', 'round-robin'], default='replicate')
    parser.add_argument('--trace-order', choices=['file', 'timestamp'], default='file')
    parser.add_argument('--model', default='/data1/jhcheng/models/scchen/Qwen3-8B')
    parser.add_argument('--trace', default=str(ROOT/'data/toolagent_trace.jsonl'))
    parser.add_argument('--survivor-trace',
                        help='Instance 0 uses --trace; every surviving instance replicates this trace')
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
    parser.add_argument('--record-recovery-tokens', action='store_true',
                        help='Save fault prompts and recovery IDs to /tmp/<parent>_<output-name>_records.json')
    parser.add_argument('--backup-policy', choices=['none','full','prefix-first','suffix-first',
                        'request-fifo','group-prefix-queue'], default=None,
                        help='Online source backup policy; defaults to full with a positive CPU budget, otherwise none')
    parser.add_argument('--cpu-kv-backup-bytes', type=int,
                        help='Exact integer CPU pool bytes, exclusive with --cpu-kv-backup-gb; useful for half-slot budgets')
    parser.add_argument('--kv-restore-mode',choices=['serial','suffix-overlap'],default='serial',
                        help='Serial bulk preload, or overlap suffix H2D with native head recomputation')
    parser.add_argument('--kv-load-policy',choices=['eager','demand'],default='eager',
                        help='Eager cohort preload, or load each prefix group when native scheduling reaches it')
    parser.add_argument('--inference-replay', choices=['record','replay'],
                        help='Match native inference steps across independent online backup policies')
    parser.add_argument('--inference-replay-dir',
                        help='One persistent inference journal per instance, without CPU KV bytes or slot snapshots')
    parser.add_argument('--replay-backup-mode', choices=['async','step-fenced'],default='async',
                        help='Async never waits for D2H between steps; step-fenced retains the previous control')
    parser.add_argument('--profile-backup-overhead',action='store_true')
    parser.add_argument('--profile-dir',help='Private physical-clock profiles, outside the three Markdown reports')
    parser.add_argument('--replay-diagnostics-dir',help='E0 profiling directory used to explain the first replay mismatch')
    parser.add_argument('--manifest-mode',choices=['delta','legacy'],default='delta',
                        help='Incremental live directory, or original full-manifest publication control')
    parser.add_argument('--manifest-debug-oracle',action='store_true',
                        help='CPU-only shadow comparison against original manifests; disable for timing')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--warmup', type=float, default=60,
                        help='Failure time is sampled after this warmup; does not imply steady state')
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
    parser.add_argument('--max-duration', type=float, default=600, help='Replay timeout, excludes engine startup')
    parser.add_argument('--startup-timeout', type=float, default=1200)
    parser.add_argument('--inject-failure', action='store_true', default=True,
                        help=argparse.SUPPRESS)  # 保留旧命令兼容；此入口始终执行故障实验。
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
    parser.add_argument('--output', default='results/failure_recovery/run_'+
                        datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.inference_replay)!=bool(args.inference_replay_dir):
        parser.error('--inference-replay and --inference-replay-dir are required together')
    if args.inference_replay and (args.compare_recovery_schedulers or args.failure_when_busy):
        parser.error('Inference-only replay excludes legacy scheduler comparisons and failure-when-busy')
    if args.inference_replay and os.environ.get('VLLM_BATCH_INVARIANT')!='1':
        parser.error('Inference-only exact replay requires VLLM_BATCH_INVARIANT=1')
    if (args.profile_dir or args.replay_diagnostics_dir) and not args.profile_backup_overhead:
        parser.error('Profile and diagnostic directories require --profile-backup-overhead')
    if args.cpu_kv_backup_bytes is not None:
        if args.cpu_kv_backup_bytes<0 or args.cpu_kv_backup_gb:
            parser.error('Use one nonnegative CPU budget: --cpu-kv-backup-gb or --cpu-kv-backup-bytes')
        args.cpu_kv_backup_gb=args.cpu_kv_backup_bytes/1e9
    if not math.isfinite(args.cpu_kv_backup_gb) or args.cpu_kv_backup_gb < 0:
        parser.error('--cpu-kv-backup-gb must be finite and nonnegative')
    args.backup_policy=args.backup_policy or ('full' if args.cpu_kv_backup_gb else 'none')
    if args.kv_restore_mode=='suffix-overlap' and args.backup_policy!='suffix-first':
        parser.error('suffix-overlap requires --backup-policy suffix-first and a positive CPU budget')
    if (args.backup_policy=='none') != (args.cpu_kv_backup_gb==0):
        parser.error('none requires a zero CPU budget; backup policies require a positive CPU budget')
    if args.backup_policy not in ('none','full') and args.compare_recovery_schedulers:
        parser.error('This phase fixes the recovery scheduler; selective-backup replay comparisons are not enabled')
    args.recovery_mode = args.recovery_mode or ('resume' if args.cpu_kv_backup_gb else 'restart')
    if args.cpu_kv_backup_gb and args.recovery_mode != 'resume':
        parser.error('Live CPU KV recovery requires --recovery-mode resume')
    if args.inference_replay and args.recovery_mode!='resume':
        parser.error('Backup-policy inference replay requires --recovery-mode resume in every case')
    positive = ('qps', 'window', 'check_every', 'max_duration', 'startup_timeout',
                'flow_tolerance', 'trend_tolerance', 'level_tolerance', 'count_tolerance', 'gpu_memory_utilization',
                'ttft_relative_tolerance',
                'ttft_absolute_tolerance_ms')
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        parser.error('Rates, windows, timeouts and tolerances must be positive and finite')
    if (any(not math.isfinite(v) or v < 0 for v in [args.warmup, args.max_ingress])
            or args.gpu_memory_utilization > 1):
        parser.error('Invalid warmup, ingress bound or GPU utilization')
    if (not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)
            or min(args.stable_checks, args.max_num_seqs, args.batch_tokens, args.cpu_threads) < 1):
        parser.error('GPUs must be unique/nonnegative; integer counts must be positive')
    if (not math.isfinite(args.recovery_drain) or args.recovery_drain < 0
            or any(not math.isfinite(v) or v < 0 for v in args.failure_delay)
            or args.failure_delay[0] > args.failure_delay[1]):
        parser.error('Failure delay and recovery drain must be finite and nonnegative')
    if len(args.gpus) < 2:
        parser.error('At least two independent instances are required: instance 0 fails')
    if args.warmup+args.failure_delay[1] >= args.max_duration:
        parser.error('The full failure-delay range must fit after warmup and before max-duration')
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
    if args.backup_policy=='group-prefix-queue' and (args.recovery_scheduler!='kv-aware'
                                                   or args.recovery_order!='group-ratio'):
        parser.error('group-prefix-queue predicts --recovery-scheduler kv-aware --recovery-order group-ratio')
    if args.comparison_repeats<1 or (args.comparison_repeats!=1 and not args.compare_recovery_schedulers):
        parser.error('--comparison-repeats must be positive and requires --compare-recovery-schedulers')
    if args.compare_recovery_schedulers and args.comparison_repeats>1 and args.failure_delay[0]==args.failure_delay[1]:
        parser.error('Repeated comparisons require --failure-delay MIN MAX with MIN < MAX to sample new fault times')
    failure_rng = random.Random(args.seed)
    args.comparison_failure_times_s = [args.warmup+failure_rng.uniform(*args.failure_delay)
        for _ in range(args.comparison_repeats)]
    args.failure_time_s = args.comparison_failure_times_s[0]
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
    # worker 是逻辑实例号，gpus[worker] 才是物理卡；列表第一张卡始终为故障源。
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


if __name__ == '__main__':
    main()
