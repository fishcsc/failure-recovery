"""sample_running --inject-failure 的旧故障流程，保留历史对照与测试。

入口 replay_failure，旧协调器使用 recovery_scheduling 和 cpu_kv_recovery。
当前实验入口是 failure_recovery.main，不要把两套故障协议混在一起修改。
"""
import math
from bisect import bisect_right
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import time

from metrics import percentile
from report import FAILURE_REPORT_FILES as RESULT_FILES, write_failure_report as write_result
from recovery_scheduling import assign_recovery_requests as recovery_assignment, order_recovery_requests


class PrefailureReplay:
    """严格对照：记录提交边界与真实 forward；重放时每一步校验，绝不替换实际请求。"""
    def __init__(self, args, worker):
        self.mode = getattr(args, '_prefailure_mode', None)
        self.path = Path(args._prefailure_dir)/f'worker_{worker}.json' if self.mode else None
        self.data = json.loads(self.path.read_text()) if self.mode=='replay' else dict(steps=[])
        self.index = 0

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def next_step(self):
        if self.mode=='replay' and self.index<len(self.data['steps']):
            return self.data['steps'][self.index]
        return None

    def step(self, cursor, begin, end, batches, outputs, cpu_ops=None):
        if not self.mode:
            return
        signature = self.digest([
            [[r['request_id'],r['scheduled_tokens']] for b in batches for r in b['scheduled_requests']],
            [[r.request_id,r.finished,list(r.outputs[0].token_ids) if r.outputs else []] for r in outputs]])
        row = dict(cursor=cursor,start_s=begin,end_s=end,signature=signature)
        if cpu_ops is not None:
            row['cpu_ops'] = cpu_ops
        if self.mode=='record':
            self.data['steps'].append(row)
        elif row!=self.next_step():
            raise RuntimeError(f'Strict pre-failure replay diverged at forward {self.index}')
        self.index += 1

    def finish(self, scheduler, ids, cursor, now, stability=None, cpu_snapshot=None):
        if not self.mode:
            return {}
        state = dict(cursor=cursor,
            running=[ids[r.request_id] for r in scheduler.running],
            waiting=[[ids[r.request_id] for r in q] for q in (scheduler.waiting,scheduler.skipped_waiting)],
            requests=sorted((ids[rid],r.num_computed_tokens,list(r.output_token_ids))
                            for rid,r in scheduler.requests.items()))
        signature = self.digest(state)
        if self.mode=='record':
            self.data.update(failure_s=now,state_signature=signature,
                kv_blocks=getattr(getattr(scheduler,'kv_cache_config',None),'num_blocks',None))
            if stability is not None:
                self.data['stability'] = stability
            if cpu_snapshot is not None:
                self.data['cpu_kv_snapshot'] = cpu_snapshot
            self.path.write_text(json.dumps(self.data,separators=(',',':')))
        elif (self.index!=len(self.data['steps']) or signature!=self.data['state_signature']):
            raise RuntimeError('Strict pre-failure snapshot mismatch: queues, computed tokens or output tokens differ')
        return dict(prefailure_verified=True,prefailure_steps=self.index,prefailure_state_signature=signature)


def compare_recovery_schedulers(args, requests, arrivals, output):
    """轮间独立采样；轮内三策略×所选顺序共用一次记录，其他组合逐步重放。"""
    from sample_running import run_workers
    from report import write_failure_comparison, _failure_number

    results = []
    cases = args.comparison_cases or [(strategy,order) for strategy in ('equal','prefill-balanced','kv-aware')
                                     for order in args.comparison_orders]
    write_failure_comparison(output,results,vars(args))
    with tempfile.TemporaryDirectory(prefix='failure_comparison_') as temporary, ExitStack() as pools:
        for repeat,failure_s in enumerate(args.comparison_failure_times_s,1):
            cpu_pool = None
            if getattr(args, 'cpu_kv_backup_gb', 0):
                from cpu_kv_recovery import SharedKVPool
                cpu_pool = SharedKVPool(args.cpu_kv_backup_gb)
                pools.callback(cpu_pool.close)
            baseline = Path(temporary)/f'repeat_{repeat}'
            baseline.mkdir()
            reference,placements = None,{}
            for index,(strategy,order) in enumerate(cases):
                local = copy.copy(args)
                local.failure_time_s = failure_s
                local.recovery_scheduler = strategy
                local.recovery_order = order
                local._prefailure_dir,local._prefailure_mode = str(baseline),'replay' if index else 'record'
                if cpu_pool is not None:
                    local._cpu_kv_pool = cpu_pool
                stage = baseline/f'{strategy}_{order}'
                stage.mkdir()
                captured = {}
                print(f'Repeat {repeat}/{args.comparison_repeats}; comparison {index+1}/{len(cases)}: {strategy}/{order}; '
                      f'planned failure={failure_s:.3f}s; pre-failure={local._prefailure_mode}',flush=True)
                summary = run_workers(local,requests,arrivals,stage,capture=captured)
                # 故障前状态完全相同；同策略的不同排队顺序还必须保持目标分配相同。
                state = ([g['prefailure_state_signature'] for g in captured['before']['gpus']],
                         sorted((r['original_request_id'],r['source_generated_tokens'],r['source_computed_tokens'])
                                for r in captured['plan']['assignment']))
                if cpu_pool is not None:
                    state += (sorted((r['original_request_id'],r['cpu_backup_tokens'])
                                     for r in captured['plan']['assignment']),)
                if not all(g.get('prefailure_verified') for g in captured['before']['gpus']):
                    raise RuntimeError('Comparison requires verified pre-failure replay on all four instances')
                reference = state if reference is None else reference
                if state!=reference:
                    raise RuntimeError('Comparison changed the fault cohort, source progress or background state')
                placed = sorted((r['original_request_id'],r['target_worker'],r.get('recovery_group'))
                                for r in captured['plan']['assignment'])
                if placed!=placements.setdefault(strategy,placed):
                    raise RuntimeError('Queue order comparison unexpectedly changed GPU placement')
                results.append(dict(repeat=repeat,strategy=strategy,order=order,**captured))
                write_failure_comparison(output,results,vars(args))
                if (summary['status']!='complete' or summary['traffic']['unfinished']
                        or summary['forwarding']['status']!='complete'):
                    raise RuntimeError(f'{strategy}: incomplete recovery/traffic; comparison stopped')
                print(f'Repeat {repeat}: {strategy}/{order}: recovery={summary["completed"]}/{summary["requests"]}; '
                      f'TTFT={_failure_number(summary["avg_ttft_ms"])}ms; '
                      f'TPOT={_failure_number(summary["avg_tpot_ms"])}ms/token',flush=True)
            if cpu_pool is not None:
                cpu_pool.close()
    return results


def metric_summary(records, lower=0., upper=math.inf):
    """TTFT 按首 token 事件筛窗口，TPOT 按完成事件筛窗口，保留未完成的首 token 样本。"""
    result = {}
    for name, event in [('ttft', 'first_s'), ('tpot', 'completed_s')]:
        values = [r[f'{name}_ms'] for r in records if r.get(event) is not None
                  and lower <= r[event] < upper and r.get(f'{name}_ms') is not None]
        result.update({f'{name}_requests': len(values),
                       f'avg_{name}_ms': sum(values)/len(values) if values else None,
                       f'p50_{name}_ms': percentile(values, 50),
                       f'p95_{name}_ms': percentile(values, 95)})
    return result


def forwarding_assignment(waiting, future, survivors):
    """一次性转发源 waiting/未来流量并保留原到达时间，不与故障 running 重复。"""
    assignments = {worker: [] for worker in survivors}
    for index, (kind, row) in enumerate(
            [('waiting', r) for r in waiting]+[('future', r) for r in future]):
        target = survivors[index % len(survivors)]
        assignments[target].append(dict(
            row, request_id=f"FWD0_{row['request_id']}_{index:05d}",
            original_request_id=row['request_id'], is_forwarded=True,
            source_queue_at_failure=kind, target_worker=target))
    return assignments


def merge_pending(requests, arrivals, cursor, forwarded):
    """已提交请求不动，仅把转发流量并入尚未提交部分，返回请求/到达两份新列表。"""
    pending = list(zip(requests[cursor:], arrivals[cursor:]))
    pending.extend((row, row['arrival_s']) for row in forwarded)
    pending.sort(key=lambda item: item[1])
    return (requests[:cursor]+[r for r, _ in pending],
            arrivals[:cursor]+[t for _, t in pending])


def recovery_input(checkpoint, mode):
    generated = checkpoint['generated_token_ids'] if mode == 'resume' else []
    remaining = checkpoint['output_length']-len(generated)
    if remaining < 1:
        raise ValueError('A completed request must not enter the recovery cohort')
    return checkpoint['prompt_token_ids']+generated, remaining


def prompt_input(tokens):
    """正常、转发、恢复请求统一使用无 Salt 的实例本地前缀池。"""
    return dict(prompt_token_ids=tokens)


def promote_recoveries(scheduler, recovery_order):
    """只调整等待中的恢复请求到队首，不移动原有 running。"""
    rank = {rid: i for i, rid in enumerate(recovery_order)}
    for queue in (scheduler.waiting, scheduler.skipped_waiting):
        selected = sorted((r for r in queue if r.request_id in rank),
                          key=lambda r: rank[r.request_id])
        for request in selected:
            queue.remove_request(request)
        for request in reversed(selected):
            queue.prepend_request(request)


def recovery_queue(scheduler, recovery_order, fallback):
    """旧队列优先规则：skipped 的普通请求不能挡住 waiting 中的恢复请求。"""
    rank = {rid: i for i, rid in enumerate(recovery_order)}
    candidates = [q for q in (scheduler.skipped_waiting, scheduler.waiting)
                  if q and q.peek_request().request_id in rank]
    return (min(candidates, key=lambda q: rank[q.peek_request().request_id])
            if candidates else fallback())


def update_latency(state, tokens, now, finished):
    """记录本次首 token 和完成时间；TPOT 仅覆盖本次输出，恢复计时沿用旧协议。"""
    first = tokens > state['last_tokens'] and state['first_s'] is None
    if tokens > state['last_tokens']:
        if first:
            state.update(first_s=now, ttft_ms=1000*(now-state['arrival_s']))
            if state.get('is_forwarded'):
                state['target_queue_to_first_ms'] = 1000*(now-state['admitted_s'])
            if state.get('is_recovery'):
                scheduled = state['first_scheduled_s']
                state.update(failure_to_first_ms=1000*(now-state['failure_s']),
                             queue_wait_ms=1000*(scheduled-state['enqueued_s']),
                             scheduled_to_first_ms=1000*(now-scheduled),
                             first_token_index=1, first_token_observed_count=tokens,
                             interruption_ms=(1000*(now-state['source_last_token_s'])
                                              if state.get('source_last_token_s') is not None else None))
        state.update(last_tokens=tokens, last_token_s=now)
        if state.get('is_recovery'):
            state['generated_tokens_after_restart'] = tokens
    if finished:
        if state['first_s'] is None:
            raise RuntimeError('Completion without an observed first token')
        state.update(completed_s=now, status='completed',
                     tpot_ms=1000*(now-state['first_s'])/(tokens-1) if tokens > 1 else None)
    return first


def experiment_config(args, engines=()):
    keys = ('model', 'trace', 'survivor_trace', 'gpus', 'aggregate_offered_qps', 'per_instance_qps',
            'source_qps', 'survivor_qps', 'instance_qps', 'failure_stability_scope', 'cpu_kv_backup_gb',
            'ttft_stability', 'ttft_relative_tolerance', 'ttft_absolute_tolerance_ms',
            'ingress_policy',
            'qps_scope', 'arrival', 'request_routing', 'trace_order', 'cpu_cache_gb',
            'gpu_memory_utilization', 'max_num_seqs', 'batch_tokens', 'max_model_len', 'dtype',
            'warmup', 'window', 'max_duration', 'recovery_drain', 'recovery_mode', 'require_stable_before_failure', 'failure_when_busy',
            'recovery_cache_policy', 'recovery_scheduler', 'recovery_execution', 'recovery_order', 'comparison_orders', 'comparison_cases', 'comparison_repeats', 'kv_min_shared_tokens', 'kv_split_factor',
            'seed', 'failure_time_s', 'failure_delay', 'comparison_failure_times_s', 'latency_measurement', 'trace_workloads')
    return dict({k:getattr(args,k) for k in keys if hasattr(args,k)},
                engines=[dict(worker=e['worker'],gpu=e['gpu'],dtype=e['dtype'],kv_blocks=e['kv_blocks'],
                    device_name=e['identity']['device_name'],trace=e['identity']['trace']) for e in engines])


def compact_record(row):
    omitted = {'cache_salt', 'prompt_token_ids', 'generated_token_ids', 'tokens', 'hash_ids',
               'scheduler_request_id', 'arrival_epoch', 'first_epoch', 'completion_epoch'}
    return {k: v for k, v in row.items() if k not in omitted}


def initialize_results(output, config):
    for name in RESULT_FILES:
        write_result(output/name, dict(status='pending', **({'config': config} if name == RESULT_FILES[0] else {})))


def observation_summary(loads, batches, lower=0., upper=math.inf):
    duration, waiting_time = 0., 0.
    totals = {k: 0. for k in ('running', 'waiting', 'ingress', 'outstanding')}
    maxima = dict.fromkeys(totals, 0)
    for row in loads:
        seconds = max(0., min(row['end_s'], upper)-max(row['start_s'], lower))
        duration += seconds
        waiting_time += seconds*(row['waiting'] > 0)
        for key in totals:
            totals[key] += seconds*row[key]
            if seconds:
                maxima[key] = max(maxima[key], row[key])
    selected = [b for b in batches if lower <= b['time_s'] < upper]
    return dict(recorded_seconds=duration,
                **{f'mean_{k}': v/duration if duration else None for k, v in totals.items()},
                **{f'max_{k}': v for k, v in maxima.items()},
                waiting_positive_time_fraction=waiting_time/duration if duration else None,
                schedules=len(selected), full_schedules=sum(b['full'] for b in selected),
                max_scheduled_tokens=max((b['tokens'] for b in selected), default=0),
                normal_recovery_mixed_schedules=sum(b['mixed'] for b in selected),
                peak_gpu_kv_usage=max((b['kv_usage'] for b in selected if b['kv_usage'] is not None), default=None))


class FailureCoordinator:
    """旧四实例故障协调器，收集现场并保留三份结果文档。"""
    def __init__(self, args, connections, output, start, engines=None, cpu_metadata=None):
        self.args, self.connections, self.output, self.start = args, connections, output, start
        self.snapshots, self.acks, self.results = {}, {}, {}
        self.assignments, self.forwarding = None, None
        self.config = experiment_config(args, engines or [])
        self.cpu_metadata = cpu_metadata if cpu_metadata is not None else {}
        initialize_results(output, self.config)

    @staticmethod
    def public_snapshot(snapshot):
        omitted = {'checkpoints', 'waiting_requests', 'future_requests', 'latencies', 'scheduler_running_ids',
                   'source_indices', 'source_lines', 'scheduled_ids', 'kind'}
        row = {k: v for k, v in snapshot.items() if k not in omitted}
        row['requests'] = [compact_record(r) for r in row.get('requests', [])]
        return row

    def save_before(self):
        self.before_report = dict(status='recorded', config=self.config,
                                  failure_s=self.plan['failure_s'], gpus=[self.public_snapshot(s)
                                  for _, s in sorted(self.snapshots.items())])
        write_result(self.output/RESULT_FILES[0], self.before_report)

    def handle(self, message):
        kind, worker = message['kind'], message['worker']
        if kind == 'cpu_kv_metadata':
            if worker != 0 or not getattr(self.args, 'cpu_kv_backup_gb', 0):
                raise RuntimeError('Unexpected CPU checkpoint metadata publisher')
            for update in message['updates']:
                rid = update['request_id']
                if update.get('completed'):
                    self.cpu_metadata.pop(rid,None)
                    continue
                entry = self.cpu_metadata.setdefault(rid,dict(generated_token_ids=[],slots=[]))
                entry['generated_token_ids'].extend(update.get('new_tokens',[]))
                entry.update({k:v for k,v in update.items() if k not in ('new_tokens','request_id')})
            return True
        if kind == 'result':
            self.results[worker] = message.pop('failure_results')
            return False
        if kind == 'pre_failure':
            if worker in self.snapshots:
                raise RuntimeError('Duplicate failure snapshot')
            self.snapshots[worker] = message
            if len(self.snapshots) == self.args.instances:
                self.dispatch()
            return True
        if kind != 'recovery_enqueued':
            return False
        if worker in self.acks or self.assignments is None or worker not in self.assignments:
            raise RuntimeError('Unexpected recovery enqueue acknowledgement')
        expected = [r['recovery_id'] for r in self.assignments[worker]]
        if message['recovery_ids'] != expected:
            raise RuntimeError('Recovery assignment/admission mismatch')
        if message['admitted_recovery_ids'] != expected:
            raise RuntimeError('All recovery requests must enter native queues before resuming')
        if not set(expected) <= {rid for ids in message['queue_ids'].values() for rid in ids}:
            raise RuntimeError('Admitted recovery is missing from native queues')
        for ids in message['queue_ids'].values():
            ordinary_seen = False
            for rid in ids:
                if rid in expected and ordinary_seen:
                    raise RuntimeError('Recovery request is behind ordinary waiting requests')
                ordinary_seen |= rid not in expected
        self.acks[worker] = message
        if len(self.acks) == self.args.instances-1:
            for target in self.assignments:
                self.connections[target].send(dict(action='resume'))
            self.plan.update(resume_sent_s=time.perf_counter()-self.start,
                             enqueue_acknowledgements=list(self.acks.values()))
            write_result(self.output/RESULT_FILES[1], self.plan)
        return True

    def dispatch(self):
        source = self.snapshots[0]
        failure_s = max(s['time_s'] for s in self.snapshots.values())
        checkpoints = source['checkpoints']
        if [c['request_id'] for c in checkpoints] != source['running_ids']:
            raise RuntimeError('Failure cohort must equal the native source running queue')
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            for checkpoint in checkpoints:
                saved = self.cpu_metadata[checkpoint['request_id']]
                if (saved['slots'] != checkpoint['cpu_kv']['slots']
                        or saved['generated_token_ids'] != checkpoint['generated_token_ids']
                        or saved['prompt_token_ids'] != checkpoint['prompt_token_ids']
                        or saved.get('signature') != checkpoint['cpu_kv']['signature']):
                    raise RuntimeError('Parent-owned CPU checkpoint/progress differs from fault snapshot')
        timeline = {}
        if getattr(self.args,'_prefailure_mode',None):
            # 所有实例重建完成后统一时钟，故障后的原到达流仍严格保持全局 QPS。
            timeline['timeline_start'] = time.perf_counter()-failure_s
            self.start = timeline['timeline_start']
        self.assignments = recovery_assignment(checkpoints, list(range(1, self.args.instances)),
            self.args.recovery_scheduler,self.args.recovery_mode,self.args.kv_min_shared_tokens,self.args.kv_split_factor)
        self.assignments = {w:order_recovery_requests(items,self.args.recovery_order,
            self.args.recovery_mode) for w,items in self.assignments.items()}
        self.forwarding = forwarding_assignment(source['waiting_requests'], source['future_requests'], list(self.assignments))
        if {r['original_request_id'] for items in self.forwarding.values() for r in items} & set(source['running_ids']):
            raise RuntimeError('Forwarded traffic overlaps the running recovery cohort')
        self.plan = dict(status='scheduled', failed_worker=0, failed_gpu=self.args.gpus[0],
            recovery_protocol='running_priority_and_forward_all_v3', latency_measurement=self.args.latency_measurement,
            source_trace=self.args.trace, survivor_trace=self.args.survivor_trace,
            recovery_cache_policy=self.args.recovery_cache_policy, recovery_mode=self.args.recovery_mode,
            recovery_scheduler=self.args.recovery_scheduler,
            recovery_execution=self.args.recovery_execution,
            recovery_order=self.args.recovery_order,
            kv_min_shared_tokens=self.args.kv_min_shared_tokens,kv_split_factor=self.args.kv_split_factor,
            estimated_prefill_by_target=[dict(worker=w,gpu=self.args.gpus[w],
                input_tokens=sum(r['prefill_tokens'] for r in items),
                estimated_prefill_tokens=sum(r['estimated_prefill_tokens'] for r in items),
                groups=len({r['recovery_group'] for r in items if r.get('recovery_group')}))
                for w,items in self.assignments.items()],
            planned_failure_s=self.args.failure_time_s, failure_s=failure_s,
            source_stopped_forward_s=source['time_s'], dispatch_started_s=time.perf_counter()-self.start,
            policy=self.args.recovery_scheduler+'_queue_head', snapshots_s={str(w):s['time_s'] for w,s in self.snapshots.items()},
            snapshot_skew_s=failure_s-min(s['time_s'] for s in self.snapshots.values()),
            original_running_ids=source['running_ids'], source_waiting_forwarded=source['waiting_ids'],
            source_future_forwarded=source['future_not_submitted'],
            future_traffic_policy='survivors_keep_original_stream; source_waiting_and_future_round_robin',
            assignment=[dict(original_request_id=r['request_id'], source_index=r['source_index'],
                source_line=r['source_index']+1, recovery_id=r['recovery_id'], target_worker=w,
                target_gpu=self.args.gpus[w], source_generated_tokens=len(r['generated_token_ids']),
                source_status=r.get('source_status'), source_phase=r.get('source_phase'),
                source_computed_tokens=r.get('source_computed_tokens'),
                source_remaining_tokens=r['output_length']-len(r['generated_token_ids']),
                estimated_prefill_tokens=r['estimated_prefill_tokens'],recovery_group=r.get('recovery_group'),
                group_position=r.get('group_position'),queue_rank=r.get('queue_rank'),
                input_length=r['input_length'], output_length=r['output_length'])
                for w, items in self.assignments.items() for r in items],
            forwarding_assignment=[dict(original_request_id=r['original_request_id'], request_id=r['request_id'],
                source_line=r['source_index']+1, target_worker=w, target_gpu=self.args.gpus[w],
                source_queue_at_failure=r['source_queue_at_failure'], arrival_s=r['arrival_s'],
                source_generated_tokens=r.get('source_generated_tokens', 0),
                input_length=r['input_length'], output_length=r['output_length'])
                for w, items in self.forwarding.items() for r in items])
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            for entry in self.plan['assignment']:
                checkpoint = next(r for items in self.assignments.values() for r in items
                                  if r['recovery_id']==entry['recovery_id'])
                entry['cpu_backup_tokens'] = len(checkpoint['cpu_kv']['slots'])*16
        for snapshot in self.snapshots.values():
            lower = max(self.args.warmup, failure_s-self.args.window)
            snapshot.update(baseline_start_s=lower, baseline_end_s=failure_s,
                            baseline=metric_summary(snapshot['latencies'], lower, failure_s))
        self.save_before()
        write_result(self.output/RESULT_FILES[1], self.plan)
        self.connections[0].send(dict(action='fail', failure_s=failure_s,**timeline))
        for worker, items in self.assignments.items():
            self.connections[worker].send(dict(action='recover', failure_s=failure_s,
                checkpoints=items, forwarded=self.forwarding[worker],**timeline))
        print(f"Failure GPU={self.args.gpus[0]} t={failure_s:.3f}s: running={len(checkpoints)}, "
              f"waiting={len(source['waiting_ids'])}, future={source['future_not_submitted']}", flush=True)

    def finalize(self):
        if self.assignments is None or len(self.acks) != self.args.instances-1 or len(self.results) != self.args.instances:
            raise RuntimeError('Failure protocol did not complete')
        recovered = [r for data in self.results.values() for r in data['recovery']]
        forwarded = [r for data in self.results.values() for r in data['forwarded']]
        expected = {(r['recovery_id'], w) for w, items in self.assignments.items() for r in items}
        if {(r['request_id'],r['target_worker']) for r in recovered} != expected or len(recovered) != len(expected):
            raise RuntimeError('Recovery request accounting mismatch')
        expected_fwd = {(r['request_id'], w) for w, items in self.forwarding.items() for r in items}
        if {(r['request_id'],r['target_worker']) for r in forwarded} != expected_fwd or len(forwarded) != len(expected_fwd):
            raise RuntimeError('Source waiting/future traffic accounting mismatch')
        completed = sum(r['status']=='completed' for r in recovered)
        fwd_completed = sum(r['status']=='completed' for r in forwarded)
        summary = dict(status='no_running_requests' if not expected else
            'complete' if completed == len(expected) else 'incomplete_recovery',
            failed_worker=0, failed_gpu=self.args.gpus[0], failure_s=self.plan['failure_s'],
            recovery_protocol=self.plan['recovery_protocol'], latency_measurement=self.args.latency_measurement,
            recovery_scheduler=self.args.recovery_scheduler,
            recovery_execution=self.args.recovery_execution,
            recovery_order=self.args.recovery_order,
            requests=len(recovered), completed=completed, unfinished=len(recovered)-completed,
            **metric_summary(recovered), recovery_cache_policy=self.args.recovery_cache_policy,
            cache_observed_requests=sum(r['gpu_cached_tokens_first_schedule'] is not None for r in recovered),
            requests_with_gpu_cache_hits=sum((r['gpu_cached_tokens_first_schedule'] or 0)>0 for r in recovered),
            gpu_cached_prompt_tokens=sum(r['gpu_cached_tokens_first_schedule'] or 0 for r in recovered),
            forwarding=dict(status='complete' if fwd_completed==len(forwarded) else 'incomplete_forwarding',
                requests=len(forwarded), completed=fwd_completed, unfinished=len(forwarded)-fwd_completed,
                waiting=sum(r['source_queue_at_failure']=='waiting' for r in forwarded),
                future=sum(r['source_queue_at_failure']=='future' for r in forwarded), **metric_summary(forwarded)),
            by_target=[dict(worker=w,gpu=self.args.gpus[w],assigned=len(self.assignments[w]),
                completed=sum(r['status']=='completed' for r in recovered if r['target_worker']==w),
                **metric_summary([r for r in recovered if r['target_worker']==w])) for w in self.assignments])
        planned = sum(d['planned_original_requests'] for d in self.results.values())
        original_completed = sum(r['status']=='completed' and not r.get('is_forwarded')
                                 for d in self.results.values() for r in d['normal'])
        summary['traffic'] = dict(planned=planned, completed=original_completed+completed+fwd_completed,
                                 unfinished=planned-original_completed-completed-fwd_completed)
        last = max((r['completed_s'] for r in recovered if r['completed_s'] is not None), default=None)
        summary.update(recovery_completed_s=last,
                       recovery_elapsed_s=last-self.plan['failure_s'] if last is not None else None)
        measured = [r for r in recovered if r['ttft_ms'] is not None]
        summary['ttft_breakdown'] = {key:sum(r[key] for r in measured)/len(measured) if measured else None
                                    for key in ('queue_wait_ms','scheduled_to_first_ms')}
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            summary['cpu_kv'] = dict(
                source_backup=self.snapshots[0]['cpu_backup'],
                backup_tokens=sum(r['cpu_backup_tokens'] for r in recovered),
                loaded_tokens=sum(r['cpu_loaded_tokens'] for r in recovered),
                load_bytes=sum(r['cpu_load_bytes'] for r in recovered),
                load_work_ms=sum(r['cpu_load_ms'] for r in recovered),
                mean_prefill_to_first_tokens=(sum(r.get('prefill_to_first_tokens',0) for r in recovered)
                                              /len(recovered) if recovered else None))
        gpus = []
        failure_s = self.plan['failure_s']
        for worker, data in sorted(self.results.items()):
            before = self.snapshots[worker]
            lower = max(self.args.warmup, failure_s-self.args.window)
            before['load'] = observation_summary(data['load_rows'],data['batch_rows'],lower,failure_s)
            if worker == 0:
                continue
            normal = [r for r in data['normal'] if not r.get('is_forwarded')]
            windows = []
            begin = failure_s
            while begin < self.args.max_duration:
                end = min(begin+self.args.window, self.args.max_duration)
                windows.append(dict(start_s=begin,end_s=end,normal=metric_summary(normal,begin,end),
                    load=observation_summary(data['load_rows'],data['batch_rows'],begin,end)))
                begin = end
            gpus.append(dict(worker=worker,gpu=self.args.gpus[worker],timing=data['timing'],
                normal=metric_summary(normal,failure_s,self.args.max_duration),
                normal_planned=data['planned_original_requests'], normal_completed=sum(r['status']=='completed' for r in normal),
                post_failure=observation_summary(data['load_rows'],data['batch_rows'],failure_s,self.args.max_duration),
                windows=windows,
                recovery_prefill_batches=data['recovery_prefill_batches']))
        self.save_before()
        status = (summary['status'] if summary['status'] != 'complete' else
                  'complete' if not summary['traffic']['unfinished'] else 'incomplete')
        self.after_report = dict(status=status,summary=summary,gpus=gpus,
            metric_definitions=dict(ttft='(first_s - enqueued_s) * 1000; first output token of this attempt',
                tpot='(completed_s - first_s) * 1000 / (output_tokens - 1)',
                forwarded_ttft='(first_s - original arrival_s) * 1000'),
            recovery_requests=[compact_record(r) for r in recovered],
            forwarded_requests=[compact_record(r) for r in forwarded])
        write_result(self.output/RESULT_FILES[2], self.after_report)
        return summary


def failure_stability(args, load_rows, arrivals, records, now):
    """在故障边界用连续滚动窗口验证负载与延迟是否稳定。"""
    from sample_running import StabilityGate
    gate = StabilityGate(args)
    gate.rows.extend((r['start_s'], r['end_s'], r) for r in load_rows)
    records = list(records)
    completions = sorted(r['completed_s'] for r in records if r.get('completed_s') is not None)
    events = [(r['first_s'],r['ttft_ms']) for r in records if r.get('first_s') is not None]
    checks = [gate.evaluate(now-i*args.check_every, arrivals, completions, events)
              for i in reversed(range(args.stable_checks))]
    return dict(qualified=all(c['stable'] for c in checks), checks=checks)


def replay_failure(engine, scheduler, requests, arrivals, args, worker, gpu, output, connection, start):
    from vllm import SamplingParams
    from sample_running import latency_summary, running_snapshot

    manifest = {r['request_id']: r for r in requests}
    internal_ids, normal, recovered, recovery_order = {}, {}, {}, []
    journal = PrefailureReplay(args,worker)
    cursor, previous, paused = 0, 0., False
    planned_original_requests = len(requests)
    load_rows, batch_rows, prefill_batches = [], [], []
    failure_time = pause_s = resumed_s = None
    recovery_batch_id = 0
    cpu_transfer = cpu_backup = cpu_snapshot = None
    step_cpu_ops = None
    original_schedule = scheduler.schedule
    original_select = scheduler._select_waiting_queue_for_scheduling
    levels, batches = {}, []
    scheduler._select_waiting_queue_for_scheduling = lambda: recovery_queue(
        scheduler, recovery_order, original_select)

    def schedule(*positional, **keywords):
        nonlocal recovery_batch_id, step_cpu_ops
        if cpu_transfer is not None and worker == 0 and failure_time is None:
            if journal.mode == 'replay':
                step_cpu_ops = journal.next_step().get('cpu_ops', [])
                cpu_transfer.replay_operations(step_cpu_ops)
            else:
                step_cpu_ops = cpu_backup.take_operations()
        promote_recoveries(scheduler, recovery_order)
        schedule_start = time.perf_counter()-start
        scheduled = original_schedule(*positional, **keywords)
        if cpu_transfer is not None:
            cpu_transfer.release_imports()
        if failure_time is not None and scheduled.num_scheduled_tokens:
            recovery_batch_id += 1
        now = time.perf_counter()-start
        running, waiting = scheduler.get_request_counts()
        levels.update(running=running, waiting=waiting)
        for rid, count in scheduled.num_scheduled_tokens.items():
            state = recovered.get(internal_ids[rid])
            if count > 0 and state is not None and state['first_scheduled_s'] is None:
                state['first_scheduled_s'] = now
                state['first_schedule_batch'] = recovery_batch_id
                request = scheduler.requests[rid]
                cached = max(0, min(request.num_computed_tokens-count, request.num_prompt_tokens))
                state['gpu_cached_tokens_first_schedule'] = cached
        # 故障前也记录真实批次，便于核对提高 QPS 后是否用满预算；
        # 不能仅凭 waiting 或 running 数量推断一次前向是否已满。
        if scheduled.num_scheduled_tokens:
            snapshot = running_snapshot(scheduler, scheduled, manifest, internal_ids)
            active = []
            for row in snapshot['requests']:
                count = row['scheduled_tokens']
                if not count:
                    continue
                prefill = min(count, max(0, row['input_length']-row['computed_tokens_before_step']))
                active.append(dict(row, cohort=('recovery' if row.get('is_recovery') else
                                               'forwarded' if manifest[row['request_id']].get('is_forwarded')
                                               else 'normal'), prefill_tokens=prefill, decode_tokens=count-prefill))
            total = sum(r['scheduled_tokens'] for r in active)
            budget = min(getattr(scheduler, 'max_num_scheduled_tokens', args.batch_tokens),
                         args.batch_tokens)
            batches.append(dict(batch_id=recovery_batch_id,schedule_start_s=schedule_start, scheduled_s=now,
                                after_failure=failure_time is not None,
                                running=snapshot['running_count'], waiting=snapshot['waiting_count'],
                                total_scheduled_tokens=total, token_budget=budget,
                                scheduled_token_fraction=total/budget,
                                scheduled_requests=active))
        return scheduled

    scheduler.schedule = schedule

    def failure_requested(now):
        if not getattr(args, 'failure_when_busy', False):
            return now >= args.failure_time_s
        if worker == 0 and now >= args.failure_time_s and scheduler.running:
            args._failure_event.set()
        return args._failure_event.is_set()

    def state_row(row, arrival, admitted):
        state = dict(request_id=row['request_id'], source_index=row['source_index'],
                    trace_file=row.get('trace_file', args.trace),
                    source_line=row['source_index']+1, input_length=row['input_length'],
                    output_length=row['output_length'], arrival_s=arrival, admitted_s=admitted,
                    first_s=None, completed_s=None, last_token_s=None, last_tokens=0,
                    ttft_ms=None, tpot_ms=None, arrival_epoch=None, first_epoch=None,
                    completion_epoch=None, status='unfinished')
        if row.get('is_forwarded'):
            state.update(is_forwarded=True, original_request_id=row['original_request_id'],
                         source_worker=0, source_gpu=args.gpus[0], target_worker=worker, target_gpu=gpu,
                         source_queue_at_failure=row['source_queue_at_failure'],
                         source_generated_tokens=row.get('source_generated_tokens', 0),
                         forward_received_s=resumed_s, target_queue_to_first_ms=None)
        return state

    def enter_failure_barrier(now):
        nonlocal paused, pause_s, failure_time, resumed_s, requests, arrivals, start, cpu_snapshot
        stability = None
        if (getattr(args, 'require_stable_before_failure', False) and
                (worker == 0 or getattr(args, 'failure_stability_scope', 'all') == 'all')):
            # 回放的稳定性沿用参考逻辑历史，包含空闲区间；
            # 真实重建的队列和计算/生成进度仍会独立校验。
            stability = (journal.data.get('stability') if journal.mode=='replay' else
                         failure_stability(args, load_rows, arrivals, normal.values(), now))
            if not stability or not stability['qualified']:
                failed = sorted({k for c in (stability or {}).get('checks', [])
                                 for k, ok in c['checks'].items() if not ok})
                raise RuntimeError(f'GPU {gpu} is not stable at failure t={now:.3f}s: {failed}')
        paused, pause_s = True, now
        snapshot = running_snapshot(scheduler, None, manifest, internal_ids)
        lower = max(args.warmup, now-args.window)
        waiting = [internal_ids[r.request_id] for q in (scheduler.skipped_waiting, scheduler.waiting)
                   for r in q]
        checkpoints = []
        if worker == 0:
            if cpu_transfer is not None:
                if journal.mode == 'replay':
                    cpu_snapshot = journal.data['cpu_kv_snapshot']
                else:
                    cpu_snapshot = cpu_backup.freeze([r.request_id for r in scheduler.running])
                    cpu_snapshot['requests'] = {internal_ids[rid]: value
                        for rid,value in cpu_snapshot['requests'].items()}
            snapshot_states = {r['request_id']: r for r in snapshot['requests']}
            for request in scheduler.running:
                rid = internal_ids[request.request_id]
                row, state = manifest[rid], normal[rid]
                generated = list(request.output_token_ids)
                if len(generated) != state['last_tokens']:
                    raise RuntimeError('Checkpoint/output token accounting mismatch')
                checkpoints.append(dict(
                    request_id=rid, source_index=row['source_index'],
                    input_length=row['input_length'], output_length=row['output_length'],
                    trace_file=row.get('trace_file', args.trace),
                    source_status=getattr(getattr(request, 'status', None), 'name', 'RUNNING'),
                    source_phase=snapshot_states[rid]['phase'],
                    source_computed_tokens=request.num_computed_tokens,
                    prompt_token_ids=row['tokens'], generated_token_ids=generated,
                    source_arrival_s=state['arrival_s'], source_first_s=state['first_s'],
                    source_last_token_s=state['last_token_s']))
                if cpu_snapshot is not None:
                    checkpoints[-1]['cpu_kv'] = cpu_snapshot['requests'][rid]
        verification = journal.finish(scheduler,internal_ids,cursor,now,stability=stability,
                                      cpu_snapshot=cpu_snapshot)
        before = dict(kind='pre_failure', worker=worker, gpu=gpu, time_s=now,
                      target_time_s=args.failure_time_s,
                      point='after_completed_step_before_failure',
                      baseline_start_s=lower, baseline=metric_summary(normal.values(), lower, now),
                      waiting_ids=waiting, future_not_submitted=len(arrivals)-cursor,
                      waiting_request_states=[compact_record(normal[rid]) for rid in waiting],
                      checkpoints=checkpoints, **snapshot,**verification)
        if stability is not None:
            before['stability'] = stability
        if cpu_snapshot is not None:
            before['cpu_backup'] = cpu_snapshot['stats']
        if worker == 0:
            before['waiting_requests'] = [dict(manifest[rid], arrival_s=normal[rid]['arrival_s'],
                source_generated_tokens=normal[rid]['last_tokens']) for rid in waiting]
            before['future_requests'] = [dict(row, arrival_s=arrival)
                                         for row, arrival in zip(requests[cursor:], arrivals[cursor:])]
        before['latencies'] = list(normal.values())
        connection.send(before)
        if not connection.poll(args.startup_timeout):
            raise TimeoutError('Failure snapshot barrier timed out')
        command = connection.recv()
        start = command.get('timeline_start',start)
        if worker == 0:
            if command['action'] != 'fail':
                raise RuntimeError('Failed instance must not resume inference')
            running_ids = set(snapshot['running_ids'])
            for rid, state in normal.items():
                if state['status'] != 'completed':
                    state['status'] = 'failed_running' if rid in running_ids else 'forwarded_waiting'
            failure_time = command['failure_s']
            return False
        if command['action'] != 'recover':
            raise RuntimeError('Expected recovery assignments')
        failure_time = command['failure_s']
        forwarded = command['forwarded']
        if any(row['request_id'] in manifest for row in forwarded):
            raise RuntimeError('Forwarded request ID collides with target traffic')
        manifest.update((r['request_id'], r) for r in forwarded)
        requests, arrivals = merge_pending(requests, arrivals, cursor, forwarded)
        for checkpoint in command['checkpoints']:
            tokens, remaining = recovery_input(checkpoint, args.recovery_mode)
            rid = checkpoint['recovery_id']
            row = dict(request_id=rid, source_index=checkpoint['source_index'],
                       trace_file=checkpoint.get('trace_file', args.trace),
                       input_length=len(tokens), output_length=remaining,
                       original_request_id=checkpoint['request_id'], is_recovery=True)
            manifest[rid] = row
            enqueued = time.perf_counter()-start
            # 全部按选定顺序直接进入 vLLM；缓存命中由原生调度器判断。
            internal = engine.add_request(rid,prompt_input(tokens),SamplingParams(
                temperature=0,max_tokens=remaining,ignore_eos=True,detokenize=False))
            if getattr(scheduler.requests[internal],'cache_salt',None) is not None:
                raise RuntimeError('Recovery request unexpectedly entered an isolated cache namespace')
            internal_ids[internal] = rid
            recovery_order.append(internal)
            recovered[rid] = dict(
                state_row(row, enqueued, time.perf_counter()-start),
                scheduler_request_id=internal,
                is_recovery=True, original_request_id=checkpoint['request_id'],
                source_worker=0, source_gpu=args.gpus[0], target_worker=worker, target_gpu=gpu,
                recovery_mode=args.recovery_mode, failure_s=failure_time, enqueued_s=enqueued,
                recovery_group=checkpoint.get('recovery_group'),
                shared_prefix_tokens=checkpoint.get('shared_prefix_tokens',0),
                estimated_prefill_tokens=checkpoint['estimated_prefill_tokens'],
                latency_measurement=args.latency_measurement,
                source_trace=checkpoint.get('trace_file', args.trace),
                target_trace=args.survivor_trace or args.trace,
                recovery_cache_policy=args.recovery_cache_policy,
                gpu_cached_tokens_first_schedule=None,
                first_schedule_batch=None,first_token_batch=None,prefill_scheduled_tokens=0,
                prefill_step_ms=0.,prefill_batch_ids=[],
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
                                        if args.recovery_mode=='resume' else 1),
                first_token_index=None, first_token_observed_count=None)
        if cpu_transfer is not None:
            # 整批先入队再加载，使每个恢复请求的 TTFT 都包含
            # 目标 H2D 和统一放行屏障，不能先加载再开始计时。
            for checkpoint in command['checkpoints']:
                state = recovered[checkpoint['recovery_id']]
                request = scheduler.requests[state['scheduler_request_id']]
                state.update(cpu_transfer.restore(request,checkpoint['cpu_kv']))
                state['cpu_backup_tokens'] = len(checkpoint['cpu_kv']['slots'])*16
        promote_recoveries(scheduler, recovery_order)
        queue_ids = {name: [internal_ids[r.request_id] for r in getattr(scheduler, name)]
                     for name in ('waiting', 'skipped_waiting')}
        queued = {r.request_id for q in (scheduler.waiting, scheduler.skipped_waiting) for r in q}
        if not set(recovery_order) <= queued:
            raise RuntimeError('Recovery requests were not admitted into native waiting queues')
        ack = dict(kind='recovery_enqueued', worker=worker, gpu=gpu,
                   time_s=time.perf_counter()-start,
                   recovery_ids=list(recovered), queue_ids=queue_ids,
                   admitted_recovery_ids=list(recovered))
        connection.send(ack)
        if not connection.poll(args.startup_timeout) or connection.recv().get('action') != 'resume':
            raise RuntimeError('Recovery resume barrier failed')
        resumed_s = time.perf_counter()-start
        return True

    try:
        if getattr(args, 'cpu_kv_backup_gb', 0):
            from cpu_kv_recovery import KVTransfer, SourceKVBackup
            cpu_transfer = KVTransfer(engine,scheduler,args._cpu_kv_handle)
            if worker == 0 and journal.mode != 'replay':
                cpu_backup = SourceKVBackup(cpu_transfer)
        while True:
            if cpu_backup is not None and not paused:
                cpu_backup.poll()
                updates = cpu_backup.updates(internal_ids)
                if updates:
                    connection.send(dict(kind='cpu_kv_metadata',worker=worker,gpu=gpu,updates=updates))
            now = time.perf_counter()-start
            forced = None
            if not paused and journal.mode=='replay':
                forced = journal.next_step()
                now = forced['start_s'] if forced else journal.data['failure_s']
                delay = now-(time.perf_counter()-start)
                if delay>0:
                    time.sleep(min(delay,.05))
                    continue
            pending_due = cursor < len(arrivals) and arrivals[cursor] < args.max_duration
            if now >= args.max_duration and (
                    now >= args.max_duration+args.recovery_drain
                    or (not pending_due and not engine.has_unfinished_requests())):
                break
            if not paused and ((journal.mode=='replay' and forced is None)
                               or (journal.mode!='replay' and failure_requested(now))):
                if not enter_failure_barrier(now):
                    previous = now
                    break
                # 单独记录故障屏障暂停，不把它计入普通前向区间。
                previous = time.perf_counter()-start
                continue
            while (cursor < forced['cursor'] if forced else
                   cursor < len(arrivals) and arrivals[cursor] <= now and arrivals[cursor] < args.max_duration):
                # 积压提交期间也检查故障，避免大量到达请求推迟故障边界。
                if not paused and not forced and journal.mode!='replay' and failure_requested(now):
                    break
                row = requests[cursor]
                internal = engine.add_request(row['request_id'], prompt_input(row['tokens']),
                                              SamplingParams(temperature=0,
                                                             max_tokens=row['output_length'],
                                                             ignore_eos=True, detokenize=False))
                internal_ids[internal] = row['request_id']
                if getattr(scheduler.requests[internal], 'cache_salt', None) is not None:
                    raise RuntimeError('Normal/forwarded request unexpectedly entered an isolated cache namespace')
                admitted = time.perf_counter()-start
                normal[row['request_id']] = state_row(row, arrivals[cursor], admitted)
                if cpu_backup is not None and not paused:
                    connection.send(dict(kind='cpu_kv_metadata',worker=worker,gpu=gpu,
                        updates=[dict(request_id=row['request_id'],prompt_token_ids=row['tokens'],
                                      output_length=row['output_length'],arrival_s=arrivals[cursor])]))
                cursor += 1
                now = forced['start_s'] if forced else time.perf_counter()-start
            if not paused and not forced and journal.mode!='replay' and failure_requested(now):
                continue
            levels.clear()
            if engine.has_unfinished_requests():
                batches.clear()
                step_start = forced['start_s'] if forced else time.perf_counter()-start
                outputs = engine.step()
                ended = forced['end_s'] if forced else time.perf_counter()-start
                if not paused:
                    journal.step(cursor,step_start,ended,batches,outputs,cpu_ops=step_cpu_ops)
                for result in outputs:
                    state = recovered.get(result.request_id) or normal[result.request_id]
                    tokens = len(result.outputs[0].token_ids) if result.outputs else 0
                    if cpu_backup is not None and not paused:
                        connection.send(dict(kind='cpu_kv_metadata',worker=worker,gpu=gpu,
                            updates=[dict(request_id=result.request_id,
                                new_tokens=(list(result.outputs[0].token_ids[state['last_tokens']:]) if result.outputs else []),
                                completed=result.finished)]))
                    if result.finished and (not result.outputs
                                            or result.outputs[0].finish_reason != 'length'
                                            or tokens != state['output_length']):
                        raise RuntimeError(f'Unexpected completion: {result.request_id}')
                    first = update_latency(state, tokens, ended, result.finished)
                    if first and state.get('is_recovery'):
                        state['first_token_batch'] = recovery_batch_id
                        if cpu_transfer is not None:
                            state['first_new_token_id'] = result.outputs[0].token_ids[0]
                    if result.finished and state.get('is_recovery') and cpu_transfer is not None:
                        state['continuation_sha256'] = hashlib.sha256(
                            json.dumps(list(result.outputs[0].token_ids)).encode()).hexdigest()
                if cpu_backup is not None and not paused:
                    cpu_backup.submit()
                if batches:
                    for batch in batches:
                        active = batch['scheduled_requests']
                        for row in active:
                            state = recovered.get(row['request_id'])
                            if state is not None and row['prefill_tokens']:
                                state['prefill_scheduled_tokens'] += row['prefill_tokens']
                                state['prefill_step_ms'] += 1000*(ended-step_start)
                                state['prefill_batch_ids'].append(batch['batch_id'])
                        batch_rows.append(dict(time_s=step_start, tokens=batch['total_scheduled_tokens'],
                            full=batch['total_scheduled_tokens']==batch['token_budget'],
                            mixed={'normal','recovery'} <= {r['cohort'] for r in active},
                            kv_usage=getattr(getattr(scheduler, 'kv_cache_manager', None), 'usage', None)))
                        if any(r['cohort']=='recovery' and r['prefill_tokens'] for r in active):
                            prefill_batches.append(dict(batch_id=batch['batch_id'],step_start_s=step_start,step_end_s=ended,
                                total_scheduled_tokens=batch['total_scheduled_tokens'],
                                token_budget=batch['token_budget'],waiting=batch['waiting'],
                                scheduled_requests=[{k:r[k] for k in ('request_id','cohort','phase',
                                    'scheduled_tokens','prefill_tokens','decode_tokens')} for r in active]))
                if cpu_transfer is not None:
                    for state in recovered.values():
                        if state['first_s'] is not None and 'prefill_to_first_tokens' not in state:
                            state['prefill_to_first_tokens'] = state['prefill_scheduled_tokens']
            else:
                time.sleep(.01)
                ended = time.perf_counter()-start
                levels.update(running=0, waiting=0)
            outstanding = sum(s['status'] != 'completed' for s in normal.values())
            outstanding += sum(s['status'] != 'completed' for s in recovered.values())
            ingress = max(0, bisect_right(arrivals, min(ended, args.max_duration))-cursor)
            load_rows.append(dict(start_s=previous,end_s=ended,**levels,
                                  outstanding=outstanding+ingress,ingress=ingress))
            previous = ended
    finally:
        scheduler.schedule = original_schedule
        scheduler._select_waiting_queue_for_scheduling = original_select
        if cpu_backup is not None:
            cpu_backup.close()
        if cpu_transfer is not None:
            cpu_transfer.close()
        if engine.has_unfinished_requests():
            engine.abort_request(list(normal)+list(recovered))
        if engine.has_unfinished_requests():
            raise RuntimeError('Abort did not clear unfinished requests')
        for state in [*normal.values(), *recovered.values()]:
            if state['status'] == 'unfinished':
                state['status'] = 'aborted_at_end'
        forwarded = [s for s in normal.values() if s.get('is_forwarded')]
        forwarded.extend(dict(state_row(r, arrival, None), status='not_submitted_at_end')
                         for r, arrival in zip(requests[cursor:], arrivals[cursor:]) if r.get('is_forwarded'))
        latencies = latency_summary(normal.values(), args.warmup)
    return dict(worker=worker, gpu=gpu, status='failed_instance' if worker == 0 else 'complete',
                samples=0, required_samples=0, stable_seconds=0., stable_periods=[],
                request_routing=args.request_routing, per_instance_qps=args.per_instance_qps,
                elapsed_s=previous, failure_s=failure_time, paused_s=pause_s, resumed_s=resumed_s,
                latency=latencies, submitted=cursor,
                future_not_submitted=len(arrivals)-cursor,
                recovery_requests=len(recovered), recovery_completed=sum(
                    r['status'] == 'completed' for r in recovered.values()),
                forwarded_requests=len(forwarded), forwarded_completed=sum(
                    r['status'] == 'completed' for r in forwarded),
                failure_results=dict(planned_original_requests=planned_original_requests,
                    normal=list(normal.values()),recovery=list(recovered.values()),forwarded=forwarded,
                    load_rows=load_rows,batch_rows=batch_rows,recovery_prefill_batches=prefill_batches,
                    timing=dict(paused_s=pause_s,resumed_s=resumed_s,
                        barrier_pause_s=resumed_s-pause_s if resumed_s is not None else None)))
