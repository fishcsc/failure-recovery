# 离线核对前缀复用上限与轮换 GPU 对照，入口 main/analyze_reuse/analyze_rotation。
# 使用 report 读取结果；理想复用假设写入输出，不能当成真实引擎命中率。
"""Audit fixed-window stability and prefix reuse without running inference."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from report import read_csv, write_csv, find_case, load_config


# 理想上限假设：缓存无限、每次查询后立即可用，忽略并发、淘汰和生成 KV。
# 下面原说明会写入 oracle_assumptions，保留原文以保持报告内容不变。
def ideal_prefix_reuse(rows, pools, start=0, block_tokens=512):
    """Unbounded prompt-only caches, immediately populated after each lookup.

    This ignores service time, concurrent in-flight prefills, eviction and
    generated tokens. Count only known complete trace prefix blocks.
    """
    caches = [set() for _ in range(pools)]
    total = hits = requests = 0
    for index, row in enumerate(rows):
        cache = caches[index % pools]
        prefix, paths, matched = (), [], 0
        for block in row['hash_ids'][:row['input_length'] // block_tokens]:
            prefix += (block,)
            paths.append(prefix)
            if prefix in cache and matched == len(paths) - 1:
                matched += 1
        if index >= start:
            requests += 1
            total += row['input_length']
            hits += matched * block_tokens
        cache.update(paths)
    return {'cache_pools': pools, 'requests_measured': requests,
            'input_tokens': total, 'reused_prefix_tokens': hits,
            'prefix_ratio': hits / total}


def analyze_reuse(root):
    config = load_config(root)
    instances = len(config['gpus'])
    block_tokens = config.get('trace_block_tokens', 512)
    assert len(config['qps']) == 1 and config['arrival'] == 'constant'
    assert not config['rotate_workers']
    summaries = read_csv(root / 'summary.csv')
    aggregates = read_csv(root / 'aggregate.csv')
    output = root / 'extracted'
    output.mkdir(exist_ok=True)
    metrics = []
    for cap in config['cpu_cache_gb']:
        selected = [r for r in aggregates if float(r['cpu_cache_gb']) == cap]
        assert len(selected) == config['repeats']
        def total(key):
            return sum(int(r[key]) for r in selected)
        gq, gh = total('gpu_prefix_queries'), total('gpu_prefix_hits')
        cq, ch = total('cpu_prefix_queries'), total('cpu_prefix_hits')
        if cap:
            assert gq - gh == cq, 'Cannot combine tiers with unmatched denominators'
        ttft = [float(r['avg_ttft_ms']) / 1000 for r in selected]
        metrics.append({'cpu_cache_gb': cap, 'repeats': len(selected),
                        'ttft_mean_s': statistics.mean(ttft),
                        'ttft_sample_std_s': statistics.stdev(ttft) if len(ttft) > 1 else '',
                        'gpu_hit_ratio': gh / gq,
                        'cpu_conditional_hit_ratio': ch / cq if cq else '',
                        'cpu_extra_hit_ratio_of_gpu_queries': ch / gq,
                        'combined_hit_ratio': (gh + ch) / gq,
                        'mean_cpu_occupied_fraction': statistics.mean(float(r['mean_cpu_cache_occupied_fraction']) for r in selected),
                        'mean_cpu_load_gb': statistics.mean(float(r['cpu_load_bytes']) / 1e9 for r in selected),
                        'mean_cpu_store_gb': statistics.mean(float(r['cpu_store_bytes']) / 1e9 for r in selected)})
    write_csv(output / 'benefit_metrics.csv', metrics)
    raw = [dict(json.loads(line), source_index=i)
           for i, line in enumerate(Path(config['trace']).read_text().splitlines()) if line.strip()]
    for row in raw:
        row.update(input_length=int(row['input_length']), output_length=int(row['output_length']),
                   timestamp=float(row['timestamp']))
    eligible = [r for r in raw if r['input_length'] >= 1 and r['output_length'] >= 1
                and r['input_length'] + r['output_length'] <= config['max_model_len']
                and len(r['hash_ids']) >= r['input_length'] // block_tokens]
    eligible.sort(key=lambda r: (r['timestamp'], r['source_index']))
    used = [json.loads(line) for line in (root / 'requests.jsonl').read_text().splitlines()]
    assert [r['source_index'] for r in used] == [r['source_index'] for r in eligible[:len(used)]]
    warmup_index = next(i for i in range(len(used)) if i / config['qps'][0] >= config['warmup'])
    oracle = []
    for scope, data, pools, start in [
        ('all_eligible_shared', eligible, 1, 0),
        ('all_eligible_round_robin', eligible, instances, 0),
        ('used_requests_shared', eligible[:len(used)], 1, 0),
        ('used_requests_round_robin', eligible[:len(used)], instances, 0),
        ('measurement_arrivals_shared', eligible[:len(used)], 1, warmup_index),
        ('measurement_arrivals_round_robin', eligible[:len(used)], instances, warmup_index),
    ]:
        oracle.append({'scope': scope, **ideal_prefix_reuse(data, pools, start, block_tokens)})
    write_csv(output / 'prefix_reuse_oracle.csv', oracle)
    checks = []
    for row in summaries:
        win = config['duration'] - config['warmup']
        half_tol = max(2, .05 * win * max(float(row['offered_qps']), .1))
        failures = []
        for field in ('waiting_slope', 'outstanding_slope', 'ingress_slope'):
            if abs(float(row[field])) > float(row['slope_tolerance']):
                failures.append(field)
        if abs(float(row['offered_qps']) - float(row['completion_qps'])) > float(row['flow_tolerance']):
            failures.append('input_completion_rate_gap')
        for field in ('half_waiting_delta', 'half_outstanding_delta'):
            if abs(float(row[field])) > half_tol:
                failures.append(field)
        assert bool(failures) == (row['looks_stable'] == 'False')
        checks.append({k: row[k] for k in ('repeat', 'cpu_cache_gb', 'gpu', 'looks_stable',
                       'waiting_slope', 'outstanding_slope', 'half_waiting_delta', 'half_outstanding_delta',
                       'offered_qps', 'completion_qps')} | {'failed_checks': ';'.join(failures)})
    write_csv(output / 'stability_checks.csv', checks)
    fingerprints = {}
    for name in ('run.py', 'worker.py', 'metrics.py'):
        saved, current = root / 'source' / name, Path(__file__).parent / name
        if not saved.exists():
            continue
        fingerprints[name] = {'saved_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
                              'matches_current': saved.read_bytes() == current.read_bytes()}
    (output / 'benefit_audit.json').write_text(json.dumps({
        'stable_counts': dict(Counter(r['looks_stable'] for r in summaries)),
        'metric_denominators_checked': True, 'source_fingerprints': fingerprints,
        'oracle_assumptions': ideal_prefix_reuse.__doc__,
    }, indent=2))
    lines = ['# CPU 缓存收益核查', '',
             f"已完成 {len(aggregates)} 组、{len(summaries)} 条 GPU 结果，稳定性标志：{dict(Counter(r['looks_stable'] for r in summaries))}。", '',
             f"每组使用新引擎，不继承其他组的 trace 缓存；初始化有短小模型热身请求。随后用 trace 固定预热 {config['warmup']:g} 秒，再统计到 {config['duration']:g} 秒。未按稳定性门控启动统计。",
             'TTFT 按首 token 出现在窗口内筛选，从计划到达到首 token；TPOT 按完成时间筛选；running/waiting 按调度步骤持续时间加权。',
             '缓存命中使用窗口内的原生增量计数。预热后缓存保留，只排除预热期的统计样本。',
             '所有 stable=False 不代表所有 rank 都持续超载；负的未完成数趋势、完成率高于输入率也会失败，表明窗口内仍有瞬态或负载变化。', '',
             '| CPU GB/实例 | TTFT 均值±样本标准差 秒 | GPU 命中% | CPU 条件命中% | GPU+CPU 合计命中% | CPU 累计加载 GB/组 |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        cpu = f"{100*r['cpu_conditional_hit_ratio']:.2f}" if r['cpu_conditional_hit_ratio'] != '' else '—'
        std = f"{r['ttft_sample_std_s']:.3f}" if r['ttft_sample_std_s'] != '' else '—'
        lines.append(f"| {r['cpu_cache_gb']:g} | {r['ttft_mean_s']:.3f} ± {std} | {100*r['gpu_hit_ratio']:.2f} | {cpu} | {100*r['combined_hit_ratio']:.2f} | {r['mean_cpu_load_gb']:.3f} |")
    lines += ['', 'CPU 条件命中的分母是 GPU 未命中的查询 token。已检查 GPU_queries - GPU_hits = CPU_queries，因此合计比例用 (GPU_hits + CPU_hits) / GPU_queries 计算，不能直接加两层的百分比。', '',
              '## 离线前缀复用对照', '',
              f'假设缓存无限、每个请求立即把全部已知完整 {block_tokens}-token prompt 块存入缓存；忽略执行时间、并发在途、淘汰和生成 token。它分析复用机会，不是实际服务性能。',
              'arrival 窗口与原生命中计数的 admission 事件窗口不同，不能将离线数值当成实际计数的严格上界。', '',
              '| 范围 | 缓存池数 | 计入请求数 | 可复用完整前缀 token 比例% |',
              '|---|---:|---:|---:|']
    for r in oracle:
        lines.append(f"| {r['scope']} | {r['cache_pools']} | {r['requests_measured']} | {100*r['prefix_ratio']:.2f} |")
    lines += ['', '## 原因与限制', '',
              '1. 固定 60 秒预热不能证明服务或缓存复用分布已稳定，当前结果是短窗口对比。',
              f"2. 每档回放前 {len(used)} 条 eligible 请求，全 trace 有 {len(eligible)} 条；各次重复重新初始化，不能等同于连续回放更长时间。",
              f'3. 请求轮询到 {instances} 个独立的 GPU+CPU 缓存池，同前缀跨 rank 不能复用。此处没有全局 CPU 缓存、共享索引或缓存感知路由。',
              '4. GPU 已满足大部分可见复用。增加 CPU 容量只减少未被 GPU 满足的那部分 prefill；占满 CPU 不代表新增条目会再次命中。',
              '5. 应逐卡核对 CPU 读回和写入量；写入不代表后续必然复用。传输/管理开销可能抵消收益，需另做关键路径归因。',
              f'6. 合成 token 保留已知完整 {block_tokens}-token 前缀块，未知尾部按请求私有，生成内容不构成真实多轮对话延续。', '',
              '## 官方比较', '',
              f'Mooncake FAST25 论文 Table 2 的 Tool&Agent Cache Ratio 为 59%，使用 23,608 条、约一小时的工作负载。其全局缓存池与本实验的 {instances} 个独立缓存池不同。',
              '论文：https://www.usenix.org/system/files/fast25-qin.pdf',
              '数据说明：https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md', '',
              '## 后续实验应控制', '',
              '先对完整 trace 或代表性长窗口做离线复用/复用距离分析；分别比较轮询与前缀亲和路由。',
              '缓存热度和服务队列需分别判定，连续多个固定窗口满足事先规定的容差后再采样；无法稳定的组明确报告 transient/overload，不挑选有利窗口。',
              '选择测量窗口到达的一组请求并追踪其完成，报告超时/未完成比例，同时保留窗口吞吐与 token ITL；不要仅依靠已完成请求均值。']
    (output / 'benefit_analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:35]))
    print('Audit outputs:', output)


def analyze_rotation(root):
    root = Path(root)
    config = load_config(root)
    warmup, duration = config['warmup'], config['duration']
    output = []
    for qps in config['qps']:
        for capacity in config['cpu_cache_gb']:
            directories = [find_case(root, capacity, repeat, qps) for repeat in (1, 2)]
            if not all((d/'summary.json').exists() for d in directories):
                continue
            summaries = [[r for r in json.loads((d/'summary.json').read_text())
                          if r['global_qps'] == qps] for d in directories]
            for shard in range(len(config['gpus'])):
                rows = [next(r for r in summary if r['trace_shard'] == shard)
                        for summary in summaries]
                records = []
                for directory, row in zip(directories, rows):
                    path = directory/f'requests_qps_{qps:g}_gpu{row["gpu"]}.jsonl'
                    records.append({r['request_id']: r for r in
                                    (json.loads(line) for line in path.read_text().splitlines())})
                assert records[0].keys() == records[1].keys()
                for rid in records[0]:
                    for field in ('input_length', 'output_length', 'arrival_s'):
                        assert records[0][rid][field] == records[1][rid][field], (rid, field)
                paired = sorted(set.intersection(*[
                    {rid for rid, r in rs.items() if r['first_s'] is not None
                     and warmup <= r['first_s'] < duration} for rs in records]))
                output.append({
                    'cpu_cache_gb': capacity, 'global_qps': qps, 'trace_shard': shard,
                    'gpu_repeat1': rows[0]['gpu'], 'gpu_repeat2': rows[1]['gpu'],
                    'matched_ttft_samples': len(paired),
                    'matched_ttft_repeat1_ms': (sum(records[0][rid]['ttft_ms'] for rid in paired)/len(paired)
                                               if paired else None),
                    'matched_ttft_repeat2_ms': (sum(records[1][rid]['ttft_ms'] for rid in paired)/len(paired)
                                               if paired else None),
                    'all_ttft_repeat1_ms': rows[0]['avg_ttft_ms'],
                    'all_ttft_repeat2_ms': rows[1]['avg_ttft_ms'],
                    'window_itl_repeat1_ms': rows[0]['window_avg_itl_ms'],
                    'window_itl_repeat2_ms': rows[1]['window_avg_itl_ms'],
                })
    if output:
        with (root/'rotation_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--mode', choices=['reuse', 'rotation'], default='reuse')
    args = parser.parse_args()
    root = args.results.resolve()
    if args.mode == 'rotation':
        print(json.dumps(analyze_rotation(root), indent=2))
    else:
        analyze_reuse(root)
