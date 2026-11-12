"""共用 CPU 统计：延迟、负载、稳定性、running 快照和原生 offload 计数。

worker、sample_running 和故障 engine 调用本文件；不创建引擎或决定调度。
主要入口是 summarize、StabilityGate.evaluate、running_snapshot。
"""
from collections import defaultdict, deque
import bisect
import csv
import json
import math


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def interval_summary(events, warmup, duration):
    """按 token 间隔结束时间选窗口，统计相邻输出间隔，未完成请求也可贡献样本。

    首 token 没有前一 token，不能混入 TPOT/间隔；一次返回多个 token 时保留其平均间隔口径。
    """
    selected = [e for e in events if e['interval_ms'] is not None
                and warmup <= e['time_s'] < duration]
    tokens = sum(e['new_tokens'] for e in selected)
    elapsed = sum(e['interval_ms'] for e in selected)
    values = [e['interval_ms']/e['new_tokens'] for e in selected]
    return {
        'window_itl_tokens': tokens,
        'window_itl_sum_ms': elapsed,
        'window_avg_itl_ms': elapsed/tokens if tokens else None,
        'window_p50_itl_ms': percentile(values, 50),
        'window_p95_itl_ms': percentile(values, 95),
        'window_p99_itl_ms': percentile(values, 99),
        'window_max_itl_ms': max(values) if values else None,
        'window_coalesced_token_events': sum(e['new_tokens'] > 1 for e in selected),
    }


def transfer_config(cpu_cache_gb, storage='private'):
    """生成普通 CPU offload 的每实例配置，十进制 GB 为 0 时返回 None。

    这是 run.py 的原生缓存层配置，不是故障实验的源 CPU 检查点池。
    """
    if cpu_cache_gb == 0:
        return None
    if storage not in ('private', 'shm'):
        raise ValueError(f'Unknown CPU KV storage: {storage}')
    return {
        'kv_connector': 'OffloadingConnector',
        'kv_role': 'kv_both',
        'kv_connector_extra_config': {
            **({'spec_name': 'PrivateCPUOffloadingSpec',
                'spec_module_path': 'cpu_offload_spec'} if storage == 'private'
               else {'spec_name': 'CPUOffloadingSpec'}),
            'cpu_bytes_to_use': int(cpu_cache_gb * 1_000_000_000),
            'block_size': 16,
            'eviction_policy': 'lru',
            'offload_prompt_only': True,
            'store_threshold': 0,
        },
    }


class CacheMetrics:
    """在原生日志消费前读取一次增量计数，分全程和测量窗口累计。

    GPU 命中查询全部前缀，CPU 查询仅覆盖 GPU 未满足部分，两者命中率分母不同。
    """
    def __init__(self, warmup, duration):
        self.warmup, self.duration = warmup, duration
        self.full = defaultdict(float)
        self.measured = defaultdict(float)
        self.native = {'full_run': {}, 'measurement': {}}

    def record(self, elapsed, stats, connector_stats):
        targets = [(self.full, self.native['full_run'])]
        if self.warmup <= elapsed < self.duration:
            targets.append((self.measured, self.native['measurement']))
        for totals, native in targets:
            for tier, attr in [('gpu', 'prefix_cache_stats'),
                               ('cpu', 'connector_prefix_cache_stats')]:
                cache = getattr(stats, attr, None)
                if cache is not None:
                    for name in ('queries', 'hits', 'preempted_queries', 'preempted_hits'):
                        totals[f'{tier}_prefix_{name}'] += getattr(cache, name, 0)
            if connector_stats is None:
                continue
            payload = connector_stats.data
            for name, labels in payload['data'].items():
                kind = payload['types'][name]
                for labelvalues, value in labels.items():
                    key = name + (f':{labelvalues}' if labelvalues else '')
                    entry = native.setdefault(key, {'type': kind})
                    if kind == 'counter':
                        entry['sum'] = entry.get('sum', 0) + value
                    elif kind == 'histogram':
                        entry['count'] = entry.get('count', 0) + len(value)
                        entry['sum'] = entry.get('sum', 0) + sum(value)
                    elif kind == 'gauge':
                        entry['last'] = value
                        entry['max'] = max(entry.get('max', value), value)
            for direction in ('load', 'store'):
                for unit in ('bytes', 'time'):
                    key = f'vllm:kv_offload_{direction}_{unit}'
                    totals[f'cpu_{direction}_{unit}'] += sum(
                        payload['data'].get(key, {}).values())
            totals['cpu_store_allocation_failures'] += sum(
                payload['data'].get('vllm:kv_offload_allocation_failure', {}).values())

    def summary(self):
        result = {}
        for tier in ('gpu', 'cpu'):
            for name in ('queries', 'hits', 'preempted_queries', 'preempted_hits'):
                key = f'{tier}_prefix_{name}'
                result[key] = int(self.measured[key])
            queries = result[f'{tier}_prefix_queries']
            result[f'{tier}_prefix_hit_rate'] = (
                result[f'{tier}_prefix_hits'] / queries if queries else None)
        for direction in ('load', 'store'):
            result[f'cpu_{direction}_bytes'] = int(self.measured[f'cpu_{direction}_bytes'])
            result[f'cpu_{direction}_time_s'] = self.measured[f'cpu_{direction}_time']
        result['cpu_store_allocation_failures'] = int(
            self.measured['cpu_store_allocation_failures'])
        return result


def summarize(rows, warmup, duration, qps, arrivals, admission_times, completions,
              first_token_events, completed_latencies, batch_tokens):
    """在固定预热后窗口统计时长加权负载、事件延迟和积压趋势，返回逐实例指标。

    与动态稳定窗口不同：这里只给出 looks_stable，不自动把测量起点推迟到稳定时刻。
    """
    measured = []
    for row in rows:
        left = max(warmup, row['start_s'])
        right = min(duration, row['end_s'])
        if right > left:
            measured.append(dict(row, weight=right-left, midpoint=(left+right)/2))
    seconds = sum(r['weight'] for r in measured)
    if seconds < 1:
        raise RuntimeError('No measurement interval (forward/idle samples)')

    def avg(key):
        return sum(r['weight'] * r[key] for r in measured) / seconds

    def slope(key):
        center = avg('midpoint')
        denom = sum(r['weight'] * (r['midpoint']-center)**2 for r in measured)
        return (sum(r['weight'] * (r['midpoint']-center) * r[key]
                    for r in measured) / denom) if denom else 0.0

    def count_in_window(times):
        return bisect.bisect_left(times, duration) - bisect.bisect_left(times, warmup)

    win = duration - warmup
    offer = count_in_window(arrivals) / win
    admitted = count_in_window(admission_times) / win
    completed = count_in_window(completions) / win
    waiting_slope = slope('waiting')
    outstanding_slope = slope('outstanding')
    ingress_slope = slope('ingress_waiting')
    slope_tol = max(0.015, min(0.05, offer * 0.03))
    flow_tol = max(0.03, offer * 0.05)

    # 再比较窗口前后两半，避免单一斜率掩盖先涨后降等非线性积压。
    midpoint = warmup + win / 2
    early = [r for r in measured if r['midpoint'] < midpoint]
    late = [r for r in measured if r['midpoint'] >= midpoint]
    def half_mean(part, key):
        weight = sum(r['weight'] for r in part)
        return sum(r['weight'] * r[key] for r in part) / weight if weight else 0.0
    half_wait_delta = half_mean(late, 'waiting') - half_mean(early, 'waiting')
    half_out_delta = half_mean(late, 'outstanding') - half_mean(early, 'outstanding')
    half_tol = max(2.0, 0.05 * win * max(offer, 0.1))

    stable = (seconds >= 30
              and abs(waiting_slope) <= slope_tol
              and abs(outstanding_slope) <= slope_tol
              and abs(ingress_slope) <= slope_tol
              and abs(offer-completed) <= flow_tol
              and abs(half_wait_delta) <= half_tol
              and abs(half_out_delta) <= half_tol)

    ttft = [ttft_ms for event_time, ttft_ms in first_token_events
            if warmup <= event_time < duration]
    tpot = [r['tpot_ms'] for r in completed_latencies
            if warmup <= r['completed_s'] < duration and r['tpot_ms'] is not None]
    completed_decode = [r for r in completed_latencies
                        if warmup <= r['completed_s'] < duration and r['tpot_ms'] is not None]
    decode_intervals = sum(r['output_tokens']-1 for r in completed_decode)
    decode_ms = sum(r['tpot_ms']*(r['output_tokens']-1) for r in completed_decode)
    steps = [r['engine_step_ms'] for r in rows
             if warmup <= r['end_s'] < duration and r['engine_step_ms'] > 0]

    return {
        'looks_stable': stable,
        'mean_running': avg('running'),
        'mean_decode': avg('decode'),
        'mean_scheduled_decode': avg('scheduled_decode'),
        'mean_forward_requests': avg('forward_requests'),
        'mean_waiting': avg('waiting'),
        'mean_ingress_waiting': avg('ingress_waiting'),
        'mean_outstanding': avg('outstanding'),
        'max_running': max(r['running'] for r in measured),
        'max_waiting': max(r['waiting'] for r in measured),
        'mean_gpu_kv_usage': avg('gpu_kv_usage'),
        'max_gpu_kv_usage': max(r['gpu_kv_usage'] for r in measured),
        'mean_cpu_cache_occupied_fraction': avg('cpu_cache_occupied_fraction'),
        'max_cpu_cache_occupied_fraction': max(r['cpu_cache_occupied_fraction'] for r in measured),
        'preemptions': sum(r['preemptions'] for r in measured),
        'scheduled_token_budget_full_steps': sum(
            r['scheduled_tokens'] >= batch_tokens for r in measured),
        'offered_qps': offer,
        'admitted_qps': admitted,
        'completion_qps': completed,
        'avg_ttft_ms': sum(ttft)/len(ttft) if ttft else None,
        'avg_tpot_ms': sum(tpot)/len(tpot) if tpot else None,
        'ttft_samples': len(ttft),
        'tpot_samples': len(tpot),
        'completed_token_weighted_tpot_ms': decode_ms/decode_intervals if decode_intervals else None,
        'completed_decode_intervals': decode_intervals,
        'completed_decode_ms': decode_ms,
        'mean_engine_step_ms': sum(steps)/len(steps) if steps else None,
        'max_engine_step_ms': max(steps) if steps else None,
        **{f'p{p}_ttft_ms': percentile(ttft, p) for p in (50, 95, 99)},
        **{f'p{p}_tpot_ms': percentile(tpot, p) for p in (50, 95, 99)},
        'waiting_slope': waiting_slope,
        'ingress_slope': ingress_slope,
        'outstanding_slope': outstanding_slope,
        'half_waiting_delta': half_wait_delta,
        'half_outstanding_delta': half_out_delta,
        'slope_tolerance': slope_tol,
        'flow_tolerance': flow_tol,
        'measured_seconds': seconds,
    }


def aggregate_results(rows, directory, args, qps):
    """合并各 GPU 的原始请求样本再计算整体分位数，避免平均各卡分位数。"""
    samples = {'ttft': [], 'tpot': [], 'e2e': []}
    token_events = []
    for row in rows:
        events_path = directory / f'token_events_qps_{qps:g}_gpu{row["gpu"]}.csv'
        if events_path.exists():
            with events_path.open() as source:
                for event in csv.DictReader(source):
                    token_events.append({
                        'time_s': float(event['time_s']),
                        'new_tokens': int(event['new_tokens']),
                        'interval_ms': float(event['interval_ms']) if event['interval_ms'] else None,
                    })
        with (directory / f'requests_qps_{qps:g}_gpu{row["gpu"]}.jsonl').open() as source:
            for line in source:
                record = json.loads(line)
                first, done = record['first_s'], record['completed_s']
                if first is not None and args.warmup <= first < args.duration:
                    samples['ttft'].append(record['ttft_ms'])
                if done is not None and args.warmup <= done < args.duration:
                    samples['e2e'].append(record['e2e_ms'])
                    if record['tpot_ms'] is not None:
                        samples['tpot'].append(record['tpot_ms'])
    total = lambda key: sum(r[key] for r in rows)
    average = lambda key: total(key) / len(rows)
    result = {
        'cpu_cache_gb': args.cpu_cache_gb, 'repeat': args.repeat,
        'global_qps': qps, 'arrival': args.arrival,
        'all_look_stable': all(r['looks_stable'] for r in rows),
        'offered_qps': total('offered_qps'),
        'completion_qps': total('completion_qps'),
        'generation_tokens_per_s': total('generation_tokens_per_s'),
        'mean_running_per_gpu': average('mean_running'),
        'mean_waiting_per_gpu': average('mean_waiting'),
        'mean_gpu_kv_usage': average('mean_gpu_kv_usage'),
        'mean_cpu_cache_occupied_fraction': average('mean_cpu_cache_occupied_fraction'),
        'preemptions': total('preemptions'),
        'planned': total('planned'), 'completed_total': total('completed_total'),
        'unfinished': total('unfinished'), 'not_admitted': total('not_admitted'),
        'cpu_load_bytes': total('cpu_load_bytes'),
        'cpu_store_bytes': total('cpu_store_bytes'),
        'cpu_store_allocation_failures': total('cpu_store_allocation_failures'),
    }
    for tier in ('gpu', 'cpu'):
        queries, hits = total(f'{tier}_prefix_queries'), total(f'{tier}_prefix_hits')
        result[f'{tier}_prefix_queries'] = queries
        result[f'{tier}_prefix_hits'] = hits
        result[f'{tier}_prefix_hit_rate'] = hits / queries if queries else None
    for metric, values in samples.items():
        result[f'{metric}_samples'] = len(values)
        result[f'avg_{metric}_ms'] = sum(values)/len(values) if values else None
        for p in (50, 95, 99):
            result[f'p{p}_{metric}_ms'] = percentile(values, p)
    result.update(interval_summary(token_events, args.warmup, args.duration))
    intervals = sum(r.get('completed_decode_intervals', 0) for r in rows)
    decode_ms = sum(r.get('completed_decode_ms', 0) for r in rows)
    result['completed_token_weighted_tpot_ms'] = decode_ms/intervals if intervals else None
    return result

# 采样与故障恢复共用以下观测，避免两套口径漂移。

def latency_summary(records, warmup):
    """按全程、预热后和同稳定区间汇总请求；已出首 token 但未完成仍计 TTFT。"""
    def aggregate(ttft_rows, tpot_rows):
        ttft = [r['ttft_ms'] for r in ttft_rows if r['ttft_ms'] is not None]
        tpot = [r['tpot_ms'] for r in tpot_rows if r['tpot_ms'] is not None]
        return dict(ttft_requests=len(ttft), tpot_requests=len(tpot),
                    avg_ttft_ms=sum(ttft)/len(ttft) if ttft else None,
                    p50_ttft_ms=percentile(ttft, 50), p95_ttft_ms=percentile(ttft, 95),
                    avg_tpot_ms=sum(tpot)/len(tpot) if tpot else None,
                    p50_tpot_ms=percentile(tpot, 50), p95_tpot_ms=percentile(tpot, 95))
    records = list(records)
    stable_ttft = [r for r in records if r['arrival_epoch'] is not None
                   and r['arrival_epoch'] == r['first_epoch']]
    stable_tpot = [r for r in records if r['first_epoch'] is not None
                   and r['first_epoch'] == r['completion_epoch']]
    return dict(full_run=aggregate(records, records),
                post_warmup=aggregate([r for r in records if r['arrival_s'] >= warmup],
                                      [r for r in records if r['first_s'] is not None
                                       and r['first_s'] >= warmup]),
                stable=aggregate(stable_ttft, stable_tpot),
                submitted=len(records), completed=sum(r['status'] == 'completed' for r in records),
                unfinished=sum(r['status'] != 'completed' for r in records),
                unfinished_without_first_token=sum(r['first_s'] is None for r in records),
                stable_arrivals=sum(r['arrival_epoch'] is not None for r in records),
                stable_arrivals_without_first_token=sum(
                    r['arrival_epoch'] is not None and r['first_s'] is None for r in records))


class StabilityGate:
    """用滑动窗口的前后两半负载与到达/完成数量判断稳定，连续通过才合格。"""
    def __init__(self, args):
        self.args = args
        self.rows = deque()
        self.streak = 0

    def add(self, begin, end, **levels):
        if end > begin:
            self.rows.append((begin, end, levels))
        while self.rows and self.rows[0][1] < end-self.args.window:
            self.rows.popleft()

    def evaluate(self, now, arrivals, completions, ttft_events=()):
        """对 now 之前的窗口返回 stable 和分项检查，不改变连续通过次数。

        负载按区间重叠秒数加权；可选 TTFT 检查要求两半都有足够首 token 样本。
        """
        args = self.args
        left, middle = now-args.window, now-args.window/2
        totals = {k: [0., 0.] for k in ('running', 'waiting', 'outstanding', 'ingress')}
        covered = [0., 0.]
        for begin, end, values in self.rows:
            for half, (low, high) in enumerate(((left, middle), (middle, now))):
                seconds = max(0., min(end, high)-max(begin, low))
                covered[half] += seconds
                for key in totals:
                    totals[key][half] += values[key]*seconds
        means = {k: [v[i]/covered[i] if covered[i] else 0. for i in (0, 1)]
                 for k, v in totals.items()}
        offered = bisect.bisect_right(arrivals, now)-bisect.bisect_right(arrivals, left)
        finished = (bisect.bisect_right(completions, now)
                    - bisect.bisect_right(completions, left))
        flow_limit = max(args.count_tolerance, offered*args.flow_tolerance)
        queue_limit = max(args.count_tolerance, offered*args.trend_tolerance/2)
        running_limit = max(args.count_tolerance,
                            sum(means['running'])/2*args.level_tolerance)
        checks = {
            'minimum_warmup': now >= max(args.warmup, args.window),
            'window_coverage': min(covered) >= args.window/2*.98,
            'enough_requests': offered >= 3 and finished >= 3,
            'balanced_flow': abs(offered-finished) <= flow_limit,
            'running_level': abs(means['running'][1]-means['running'][0]) <= running_limit,
            'waiting_level': abs(means['waiting'][1]-means['waiting'][0]) <= queue_limit,
            'outstanding_level': abs(means['outstanding'][1]-means['outstanding'][0]) <= queue_limit,
        }
        if getattr(args, 'ingress_policy', 'caught-up') == 'bounded':
            checks['ingress_level'] = abs(means['ingress'][1]-means['ingress'][0]) <= queue_limit
        else:
            checks['ingress_caught_up'] = sum(means['ingress'])/2 <= args.max_ingress
        latency = {}
        # warmup 只是最早检查时刻；只有这些窗口检查通过，才叫动态稳定区间。
        if getattr(args, 'ttft_stability', False):
            ttft_events = list(ttft_events or ())
            halves = [[value for stamp, value in ttft_events if low <= stamp < high]
                      for low, high in ((left, middle), (middle, now))]
            averages = [sum(v)/len(v) if v else None for v in halves]
            values = halves[0]+halves[1]
            limit = max(args.ttft_absolute_tolerance_ms,
                        args.ttft_relative_tolerance*sum(values)/len(values)) if values else 0.
            checks['ttft_samples'] = min(map(len, halves)) >= 3
            checks['ttft_level'] = (all(v is not None for v in averages)
                                    and abs(averages[1]-averages[0]) <= limit)
            latency['ttft'] = dict(half_samples=list(map(len, halves)),
                                  half_means_ms=averages, difference_limit_ms=limit)
        return dict(time_s=now, stable=all(checks.values()), checks=checks,
                    offered=offered, completed=finished, covered_s=covered,
                    half_means=means, flow_limit=flow_limit, queue_limit=queue_limit,
                    running_limit=running_limit, **latency)

    def check(self, now, arrivals, completions, ttft_events=()):
        """周期调用 evaluate 并更新连续通过次数，返回附带 qualified 的检查结果。"""
        result = self.evaluate(now, arrivals, completions, ttft_events)
        self.streak = self.streak+1 if result['stable'] else 0
        result.update(consecutive_passes=self.streak,
                      qualified=self.streak >= self.args.stable_checks)
        return result


def running_snapshot(scheduler, scheduled, manifest, request_ids=None):
    """返回原生 running 队列与本轮实际调度量，不把两者混为一谈。

    scheduled 来自调度后、前向前；manifest 保存原始请求，request_ids 映射原生 ID。
    故障边界传 None，只读取当前完成进度；source_line 为一基行号。
    """
    allocation = scheduled.num_scheduled_tokens if scheduled is not None else {}
    request_ids = request_ids if request_ids is not None else {rid: rid for rid in manifest}
    records = []
    for request in scheduler.running:
        internal_id = request.request_id
        rid = request_ids[internal_id]
        row = manifest[rid]  # 映射缺失直接报错，避免把原生 ID 错当成数据集编号。
        tokens = allocation.get(internal_id, 0)
        # schedule 后的计数已经包含本轮计划量；不能把它误当作已经执行完前向的进度。
        before = request.num_computed_tokens-tokens
        records.append(dict(
            request_id=rid, scheduler_request_id=internal_id, source_index=row['source_index'],
            source_line=row['source_index']+1,
            dataset_id=row.get('id'), chat_id=row.get('chat_id'), turn=row.get('turn'),
            input_length=row['input_length'], output_length=row['output_length'],
            scheduled_tokens=tokens, computed_tokens_before_step=before,
            generated_tokens=len(request.output_token_ids),
            phase=('decode' if before >= request.num_prompt_tokens
                   and request.output_token_ids else 'prefill'),
            **({k: row[k] for k in ('is_recovery', 'is_forwarded', 'original_request_id', 'trace_file', 'trace_label') if k in row}),
        ))
    running_count, waiting = scheduler.get_request_counts()
    if running_count != len(records):
        raise RuntimeError('Scheduler running count does not match its request list')
    scheduled_ids = [request_ids[rid] for rid, count in allocation.items() if count > 0]
    if not set(scheduled_ids) <= {row['request_id'] for row in records}:
        raise RuntimeError('Scheduled IDs are missing from scheduler.running')
    return dict(running_count=running_count, waiting_count=waiting,
                running_ids=[r['request_id'] for r in records],
                source_indices=[r['source_index'] for r in records],
                source_lines=[r['source_line'] for r in records],
                scheduler_running_ids=[r['scheduler_request_id'] for r in records],
                scheduled_ids=scheduled_ids, requests=records)

