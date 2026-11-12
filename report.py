# 普通扫描的结果提取/匹配比较，兼容旧故障报告；入口 extract/main。
# 当前六文件故障框架使用 failure_recovery/report.py，不从这里控制恢复。
"""Recover completed trials, compare matching requests, and summarize repeats."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from metrics import percentile


METRICS = (
    'avg_ttft_ms', 'p95_ttft_ms', 'avg_tpot_ms', 'window_avg_itl_ms',
    'completed_token_weighted_tpot_ms', 'completion_qps',
    'generation_tokens_per_s', 'mean_running', 'mean_waiting',
    'mean_running_per_gpu', 'mean_waiting_per_gpu', 'cpu_load_bytes',
    'cpu_prefix_hit_rate', 'gpu_prefix_hit_rate', 'mean_gpu_kv_usage',
    'mean_cpu_cache_occupied_fraction',
)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


FAILURE_REPORT_FILES = ('before_failure.md', 'failure_schedule.md', 'after_recovery.md')
RECOVERY_STRATEGY_NAMES = {'equal':'按请求数均分','prefill-balanced':'按新增 prefill 负载均衡',
                           'kv-aware':'共享前缀分组并平衡 prefill'}


def _failure_number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def _failure_table(headers, rows):
    def cell(value):
        return str(value if value is not None else '—').replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(cell, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell, row))+' |' for row in rows])


def prefix_evidence(trials):
    """展示共享组及实际命中的恢复请求，组外命中同样可见。"""
    rows = []
    for label,records in trials:
        groups = defaultdict(list)
        for r in records:
            if r.get('recovery_group'):
                groups[(r['target_gpu'],r['recovery_group'])].append(r)
        grouped = {id(r) for items in groups.values() if len(items)>1 for r in items}
        for r in records:
            if id(r) not in grouped and (r.get('gpu_cached_tokens_first_schedule') or 0)<=0:
                continue
            if r.get('first_token_batch') is not None:
                rows.append([label,r['original_request_id'],r['target_gpu'],
                    f"{r['gpu_cached_tokens_first_schedule']} / {r['input_length']}",r['prefill_scheduled_tokens'],
                    _failure_number(r['queue_wait_ms']),f"{r['first_schedule_batch']} / {r['first_token_batch']}"])
    return ('## 共享前缀请求证据\n\n'+_failure_table(
        ['试验','请求','GPU','命中 / 输入 tokens','实际 prefill tokens','入队至调度 ms','调度 / 首 token 批'],rows)) if rows else ''


def cpu_backup_evidence(items):
    """展示旧故障入口的备份开销，区分申请预算与实际占用。"""
    rows = [[label, _failure_number(s['capacity_bytes']/1e9), _failure_number(s['peak_bytes']/1e9),
             _failure_number(s['registered_bytes']/1e9),_failure_number(s['completed_store_bytes']/1e9),
             _failure_number(s['store_ms']),s['pending_blocks'],s['allocation_misses']]
            for label,s in items if s]
    return ('CPU备份：仅源实例，完整16-token块异步增量复制；CPU RAM由父进程持有。'
            '故障时不等待或补拷未完成的DMA。\n\n'+_failure_table(
                ['轮次/GPU','预算GB','有效块峰值GB','注册RAM GB','完成D2H GB','D2H工作量ms','未完成块','容量不足次数'],rows)
            if rows else '')


def cpu_restore_evidence(items):
    rows = [[label,_failure_number(s['load_bytes']/1e9),_failure_number(s['load_work_ms']),
             s['backup_tokens'],s['loaded_tokens'],_failure_number(s['mean_prefill_to_first_tokens'])]
            for label,summary in items if (s := summary.get('cpu_kv'))]
    return ('## CPU KV 加载与尾部补算\n\n'
            'TTFT从目标提交开始，到故障前进度后的下一个新token，包含H2D、入队同步、排队和尾部补算。'
            'TPOT只统计本次剩余输出；H2D工作量为各卡DMA耗时之和。\n\n'+_failure_table(
                ['组合/轮次','H2D GB','H2D工作量ms','CPU可用tokens合计','实际搬入tokens','平均首token前补算tokens'],rows)
            if rows else '')


def failure_markdown(name, data):
    """渲染旧故障入口的三份简短 Markdown；详细请求仍保留在运行记录中。"""
    number, table = _failure_number, _failure_table
    titles = dict(zip(FAILURE_REPORT_FILES, ('故障前各 GPU 状态', '故障请求与调度', '调度后恢复结果')))
    parts = ['# '+titles[name]]
    if data.get('status') == 'error':
        reason = str(data.get('error', '未知错误')).strip().splitlines()[-1]
        return '\n\n'.join(parts+['运行失败。', '原因：'+reason[:500]])+'\n'
    if name == FAILURE_REPORT_FILES[0]:
        cfg = data.get('config', {})
        cpu_label = 'CPU前缀offload' if cfg.get('cpu_kv_backup_gb') else 'CPU KV'
        parts.append(f"模型：{Path(cfg.get('model', '—')).name}；总 QPS：{number(cfg.get('aggregate_offered_qps'))}；"
                     f"GPU：{'/'.join(map(str,cfg.get('gpus', [])))}；{cpu_label}：{number(cfg.get('cpu_cache_gb'),0)} GB。")
        if cfg.get('recovery_cache_policy')=='shared':
            parts.append('缓存：同实例内正常、转发、故障恢复请求共用无 Salt 的 GPU 前缀池。')
        if cfg.get('source_qps') is not None:
            parts.append(f"独立速率：源实例 {number(cfg['source_qps'])} QPS；每个背景实例 {number(cfg['survivor_qps'])} QPS。")
        if not data.get('gpus'):
            return '\n\n'.join(parts+['等待运行；尚无测量数据。'])+'\n'
        parts.append(f"显存比例：{number(cfg.get('gpu_memory_utilization'))}；max_num_seqs：{cfg.get('max_num_seqs','—')}；"
                     f"batch_tokens：{cfg.get('batch_tokens','—')}。")
        first = data['gpus'][0]
        parts.append(f"延迟均值窗口：{number(first.get('baseline_start_s'))}–{number(first.get('baseline_end_s'))} 秒；"
                     'running/waiting 为故障前快照。'+('故障前'+
                     ('源实例' if cfg.get('failure_stability_scope', 'all')=='source' else '所有实例')+
                     '通过连续滚动窗口稳定性校验。'
                     if cfg.get('require_stable_before_failure') else '固定预热触发，不保证窗口稳定。'))
        engines = {e['gpu']:e for e in cfg.get('engines', [])}
        rows = []
        for gpu in data['gpus']:
            trace = engines.get(gpu['gpu'], {}).get('trace') or (cfg.get('trace') if gpu['worker']==0
                       else cfg.get('survivor_trace') or cfg.get('trace'))
            label = next((v for v in ('traceA','traceB') if v in str(trace)), Path(trace).stem if trace else '—')
            stats = gpu.get('baseline', {})
            rows.append([gpu['gpu'], label, number(stats.get('avg_ttft_ms')), number(stats.get('avg_tpot_ms')),
                         gpu.get('running_count'), gpu.get('waiting_count')])
        parts.append(table(['GPU','trace','TTFT (ms)','TPOT (ms/token)','running','waiting'],rows))
        evidence = cpu_backup_evidence([(g['gpu'],g.get('cpu_backup')) for g in data['gpus']])
        if evidence:
            parts.append(evidence)
    elif name == FAILURE_REPORT_FILES[1]:
        if 'assignment' not in data:
            return '\n\n'.join(parts+['等待故障；尚无调度记录。'])+'\n'
        assigned, forwarded = data['assignment'], data.get('forwarding_assignment', [])
        strategy = data.get('recovery_scheduler','equal')
        waiting = data.get('source_waiting_forwarded', data.get('source_waiting_not_recovered', []))
        future = data.get('source_future_forwarded', data.get('source_future_not_submitted'))
        parts.append(f"故障 GPU：{data.get('failed_gpu')}；实际时刻：{number(data.get('failure_s'))} 秒；"
                     f"running：{len(assigned)}；源 waiting：{len(waiting)}；尚未提交：{future if future is not None else '—'}。")
        legacy = 'source_stream_stops' in data.get('future_traffic_policy', 'source_stream_stops')
        parts.append('历史方案：只均分故障 running 至队首，源 waiting 和后续流量未转发。' if legacy else
                     f"策略：{RECOVERY_STRATEGY_NAMES[strategy]}。恢复请求进入等待队首，原有 running 继续执行；后续流量仍按原到达时间均分转发。")
        if strategy=='kv-aware':
            parts.append(f"最小共享前缀：{data['kv_min_shared_tokens']} tokens；拆组系数：{data['kv_split_factor']:g}。")
        order = data.get('recovery_order','source')
        parts.append(f'目标实例内队列顺序：`{order}`；全部请求直接入队，无外层组内等待。')
        if data.get('recovery_cache_policy')=='shared':
            parts.append('缓存统一共享：组内、组外及正常请求可按真实前缀匹配；前缀组仅用于调度。')
        if order != 'source':
            parts.append(table(['目标 GPU','恢复队首 → 队尾'],[
                [gpu,' → '.join(r['original_request_id'] for r in assigned if r['target_gpu']==gpu)]
                for gpu in sorted({r['target_gpu'] for r in assigned})]))
        estimates = {r['gpu']:r for r in data.get('estimated_prefill_by_target',[])}
        targets = sorted(set(estimates)|{r['target_gpu'] for r in assigned+forwarded})
        if estimates:
            parts.append(table(['目标 GPU','故障请求','完整输入 tokens','预计 prefill tokens','waiting / 未提交转发'],[
                [gpu,sum(r['target_gpu']==gpu for r in assigned),estimates[gpu]['input_tokens'],
                 estimates[gpu]['estimated_prefill_tokens'],
                 f"{sum(r['target_gpu']==gpu and r['source_queue_at_failure']=='waiting' for r in forwarded)} / "
                 f"{sum(r['target_gpu']==gpu and r['source_queue_at_failure']=='future' for r in forwarded)}"]
                for gpu in targets]))
        else:
            parts.append(table(['目标 GPU','故障 running','源 waiting 转发','未提交请求转发'],[
            [gpu,sum(r['target_gpu']==gpu for r in assigned),
             '—' if legacy else sum(r['target_gpu']==gpu and r['source_queue_at_failure']=='waiting' for r in forwarded),
             '—' if legacy else sum(r['target_gpu']==gpu and r['source_queue_at_failure']=='future' for r in forwarded)]
            for gpu in targets]))
        parts.append('## 故障请求')
        parts.append(table(['请求编号','输入 tokens','已生成 / 总输出','故障阶段','目标 GPU']+
                           (['前缀组'] if strategy=='kv-aware' else []),[
            [r['original_request_id'],r.get('input_length'),
             f"{r.get('source_generated_tokens','—')} / {r.get('output_length','—')}",r.get('source_phase'),r['target_gpu']]+
             ([r.get('recovery_group')] if strategy=='kv-aware' else [])
            for r in sorted(assigned,key=lambda row:row['original_request_id'])]))
    else:
        if 'summary' not in data:
            return '\n\n'.join(parts+['等待运行结束；尚无恢复结果。'])+'\n'
        summary = data['summary']; fwd = summary.get('forwarding'); traffic = summary.get('traffic')
        strategy = summary.get('recovery_scheduler')
        if strategy:
            parts.append('策略：'+RECOVERY_STRATEGY_NAMES[strategy]+'。')
        if summary.get('recovery_cache_policy')=='shared':
            parts.append('缓存：与同实例正常、转发及组内/组外恢复请求统一共享，无 Salt 隔离。')
        counts = [f"故障请求：{summary.get('completed',0)}/{summary.get('requests',0)} 完成"]
        if fwd:
            counts.append(f"转发请求：{fwd.get('completed',0)}/{fwd.get('requests',0)} 完成")
        if traffic:
            counts.append(f"全量请求：{traffic.get('completed',0)}/{traffic.get('planned',0)} 完成")
        parts.append('；'.join(counts)+'。')
        if summary.get('latency_measurement') in ('resume_cpu_kv_next_token_v1','resume_next_token_v1'):
            parts.append('TTFT：目标提交至保留原输出后的下一个新token；TPOT统计剩余输出。'+
                         ('包含CPU KV加载与尾部补算。' if summary.get('cpu_kv') else '未加载CPU KV。'))
        elif summary.get('latency_measurement') != 'restart_first_token_v2':
            parts.append('**历史口径**：TTFT 计到超过故障前输出进度的 token，TPOT 仅统计追赶后的输出。')
        else:
            parts.append('TTFT：目标入队至重新 prefill 后首 token；TPOT：本次首 token 到完成的耗时 ÷（输出 tokens − 1）。')
        if strategy=='kv-aware' or summary.get('recovery_cache_policy')=='shared':
            parts.append(f"全部恢复请求直接进入原生等待队列。首次调度 GPU 前缀命中："
                         f"{summary.get('gpu_cached_prompt_tokens',0)} tokens，"
                         f"{summary.get('requests_with_gpu_cache_hits',0)}/{summary.get('cache_observed_requests',0)} 个请求命中。")
        rows = [[r['gpu'],f"{r.get('completed',0)}/{r.get('assigned',0)}",number(r.get('avg_ttft_ms')),
                 number(r.get('avg_tpot_ms'))] for r in summary.get('by_target', [])]
        rows.append(['全部',f"{summary.get('completed',0)}/{summary.get('requests',0)}",
                     number(summary.get('avg_ttft_ms')),number(summary.get('avg_tpot_ms'))])
        parts.append(table(['恢复 GPU','完成数','平均 TTFT (ms)','平均 TPOT (ms/token)'],rows))
        evidence = cpu_restore_evidence([('本次',summary)])
        if evidence:
            parts.append(evidence)
        parts.append('## 故障后正常推理')
        parts.append('统计故障至到达流截止，不含排空阶段；running/waiting 为时间加权均值。')
        parts.append(table(['GPU','正常 TTFT (ms)','正常 TPOT (ms/token)','平均 running','平均 waiting','有 waiting 的时间'],[
            [r['gpu'],number(r.get('normal',{}).get('avg_ttft_ms')),number(r.get('normal',{}).get('avg_tpot_ms')),
             number(r.get('post_failure',{}).get('mean_running')),number(r.get('post_failure',{}).get('mean_waiting')),
             '—' if r.get('post_failure',{}).get('waiting_positive_time_fraction') is None else
             number(r['post_failure']['waiting_positive_time_fraction']*100)+'%'] for r in data.get('gpus', [])]))
        parts.append('## 逐故障请求')
        cpu_mode = 'cpu_kv' in summary
        parts.append(table(['请求编号','目标 GPU','TTFT (ms)','TPOT (ms/token)','结果']+
                           (['CPU可用tokens','首次调度缓存tokens','首token前补算tokens'] if cpu_mode else []),[
            [r['original_request_id'],r['target_gpu'],number(r.get('ttft_ms')),number(r.get('tpot_ms')),
             '完成' if r.get('status')=='completed' else '未完成']+
            ([r['cpu_backup_tokens'],r.get('gpu_cached_tokens_first_schedule'),r.get('prefill_to_first_tokens')]
             if cpu_mode else [])
            for r in sorted(data.get('recovery_requests', []),key=lambda row:row['original_request_id'])]))
        evidence = prefix_evidence([(data.get('summary',{}).get('recovery_order','source'),data.get('recovery_requests',[]))])
        if evidence:
            parts.append(evidence)
    return '\n\n'.join(parts)+'\n'


def write_failure_report(path, data):
    path.write_text(failure_markdown(path.name, data), encoding='utf-8')


def write_failure_comparison(output, results, config):
    """同一故障状态的三策略对照；始终只有三份 Markdown，不保留中间轮次文件。"""
    if (config.get('comparison_repeats',1)>1 or len(config.get('comparison_orders',[]))>1
            or config.get('comparison_cases') or config.get('recovery_mode')=='resume'):
        return write_failure_repetitions(output,results,config)
    number,table = _failure_number,_failure_table
    strategies = ('equal','prefill-balanced','kv-aware')
    by_strategy = {r['strategy']:r for r in results}
    progress = f'已完成 {len(results)}/3 种策略；顺序为 equal → prefill-balanced → kv-aware。'
    if not results:
        for name,title in zip(FAILURE_REPORT_FILES,('故障前各 GPU 状态','故障请求与调度','调度后恢复结果')):
            (output/name).write_text(f'# {title}\n\n{progress}\n\n等待运行。\n',encoding='utf-8')
        return
    first = results[0]
    before = failure_markdown(FAILURE_REPORT_FILES[0],first['before']).rstrip()
    before += '\n\n严格对照：第一轮记录，后两轮重放相同提交和前向步骤；下表和以上状态使用共同的逻辑故障时间。'
    before += '\n\n'+progress+'\n\n'+table(['策略','故障时刻 (s)','四实例状态校验','故障 running 数'],[
        [r['strategy'],number(r['plan']['failure_s']),
         '通过' if all(g.get('prefailure_verified') for g in r['before']['gpus']) else '未通过',
         len(r['plan']['assignment'])] for r in results])+'\n'
    (output/FAILURE_REPORT_FILES[0]).write_text(before,encoding='utf-8')
    schedule = ['# 故障请求与调度',progress,
        f"共同故障 GPU：{first['plan']['failed_gpu']}；故障 running：{len(first['plan']['assignment'])}。"
        '故障请求从原始 prompt 重新生成；原有 running 继续推理，源 waiting/未来流量继续均分转发。',
        f"目标实例内顺序：`{config.get('recovery_order','source')}`。",
        table(['策略','目标 GPU','故障请求','完整输入 tokens','预计 prefill tokens'],[
            [r['strategy'],e['gpu'],sum(a['target_gpu']==e['gpu'] for a in r['plan']['assignment']),
             e['input_tokens'],e['estimated_prefill_tokens']]
            for r in results for e in r['plan']['estimated_prefill_by_target']])]
    if config.get('recovery_cache_policy')=='shared':
        schedule.append('三策略均使用无 Salt 的实例本地共享前缀池：正常、转发、组内及组外恢复请求都可复用。')
    kv = by_strategy.get('kv-aware',{}).get('plan',{})
    if kv:
        schedule.append(f"KV 最小共享前缀：{kv['kv_min_shared_tokens']} tokens；拆组系数：{kv['kv_split_factor']:g}。")
    if first['plan'].get('recovery_execution')=='source_order_native_queue_v3':
        schedule.append('全部故障请求按所选顺序直接进入原生等待队首；共享前缀参与分配和顺序估算，实际命中由 vLLM 判断。')
    maps = {s:{a['original_request_id']:a for a in r['plan']['assignment']} for s,r in by_strategy.items()}
    rows = []
    for a in sorted(first['plan']['assignment'],key=lambda a:a['original_request_id']):
        cells = []
        for strategy in strategies:
            item = maps.get(strategy,{}).get(a['original_request_id'])
            cells.append('—' if not item else str(item['target_gpu'])+
                         (' / '+item['recovery_group'] if strategy=='kv-aware' else ''))
        rows.append([a['original_request_id'],a['input_length'],
                     f"{a['source_generated_tokens']} / {a['output_length']}",a['source_phase'],*cells])
    schedule += ['## 同一批故障请求',table(['请求','输入 tokens','已生成 / 总输出','阶段',
                                           'equal GPU','prefill GPU','KV GPU / 组'],rows)]
    (output/FAILURE_REPORT_FILES[1]).write_text('\n\n'.join(schedule)+'\n',encoding='utf-8')
    after = ['# 调度后恢复结果',progress,
        '恢复 TTFT：目标登记至首输出 token，包含原生排队和 prefill。全部故障请求直接入原生队列。TPOT：本次首 token 至完成耗时 ÷（输出 tokens − 1）。单位为 ms 和 ms/token。',
        table(['策略','恢复完成','平均 TTFT','平均 TPOT','全量完成','实际前缀命中 tokens'],[
            [r['strategy'],f"{s['completed']}/{s['requests']}",number(s['avg_ttft_ms']),number(s['avg_tpot_ms']),
             f"{s['traffic']['completed']}/{s['traffic']['planned']}",s['gpu_cached_prompt_tokens']]
            for r in results for s in [r['after']['summary']]]),
        '## 每 GPU 恢复指标',table(['策略','GPU','恢复完成','平均 TTFT','平均 TPOT'],[
            [r['strategy'],g['gpu'],f"{g['completed']}/{g['assigned']}",number(g['avg_ttft_ms']),number(g['avg_tpot_ms'])]
            for r in results for g in r['after']['summary']['by_target']]),
        '## 故障后正常推理',
        '故障至到达流截止，不含排空阶段；running/waiting 为时间加权均值。',
        table(['策略','GPU','正常 TTFT','正常 TPOT','running','waiting'],[
            [r['strategy'],g['gpu'],number(g['normal']['avg_ttft_ms']),number(g['normal']['avg_tpot_ms']),
             number(g['post_failure']['mean_running']),number(g['post_failure']['mean_waiting'])]
            for r in results for g in r['after']['gpus']])]
    metrics = {s:{a['original_request_id']:a for a in r['after']['recovery_requests']} for s,r in by_strategy.items()}
    rows = []
    for a in sorted(first['plan']['assignment'],key=lambda a:a['original_request_id']):
        cells = []
        for strategy in strategies:
            item = metrics.get(strategy,{}).get(a['original_request_id'])
            cells.append('—' if not item else number(item['ttft_ms'])+' / '+number(item['tpot_ms'])+
                         ('（未完成）' if item['status']!='completed' else ''))
        rows.append([a['original_request_id'],*cells])
    if all('ttft_breakdown' in r['after']['summary'] for r in results):
        after += ['## 平均 TTFT 分解（ms）',
            '两项之和为平均 TTFT；入队阶段包含提交开销；最后一项包含 chunked prefill 的多轮调度及其间的 decode。',
            table(['策略','入队至首次调度','首次调度至首 token'],[
                [r['strategy'],*[number(r['after']['summary']['ttft_breakdown'][key]) for key in
                                ('queue_wait_ms','scheduled_to_first_ms')]] for r in results])]
    after += ['## 逐故障请求：TTFT / TPOT',table(['请求','equal','prefill-balanced','kv-aware'],rows)]
    evidence = prefix_evidence([(r['strategy'],r['after']['recovery_requests']) for r in results])
    if evidence:
        after.append(evidence)
    (output/FAILURE_REPORT_FILES[2]).write_text('\n\n'.join(after)+'\n',encoding='utf-8')


def write_failure_repetitions(output, results, config):
    """策略×队列顺序共用故障基线；只汇总全部组合完成的轮次。"""
    strategies = ('equal','prefill-balanced','kv-aware')
    orders = config.get('comparison_orders') or [config.get('recovery_order','source')]
    variants = ([tuple(case) for case in config['comparison_cases']] if config.get('comparison_cases') else
                [(strategy,order) for strategy in strategies for order in orders])
    strategies = tuple(dict.fromkeys(strategy for strategy,_ in variants))
    def label(variant):
        strategy,order = variant
        return strategy if len(orders)==1 else strategy+' / '+('shortest' if order=='prefix-shortest' else order)
    def queue_cell(result,gpu):
        assigned = [a for a in result['plan']['assignment'] if a['target_gpu']==gpu]
        tokens = sum(a['estimated_prefill_tokens'] for a in assigned)
        return f"{tokens} tokens；"+' → '.join(a['original_request_id'] for a in assigned) if assigned else '—'
    repeats = config.get('comparison_repeats',1)
    by_round = defaultdict(dict)
    for result in results:
        variant = (result['strategy'],result.get('order',result['plan'].get('recovery_order',orders[0])))
        if variant in by_round[result['repeat']]:
            raise ValueError('Duplicate scheduler/order in one comparison round')
        by_round[result['repeat']][variant] = result
    complete = [n for n,items in sorted(by_round.items()) if set(items)==set(variants) and all(
        r['after']['summary']['status']=='complete' and not r['after']['summary']['traffic']['unfinished']
        and r['after']['summary']['forwarding']['status']=='complete' for r in items.values())]
    progress = f'已完成 {len(results)}/{len(variants)*repeats} 个策略试验，完整配对轮次 {len(complete)}/{repeats}。'
    number,table = _failure_number,_failure_table
    if not results:
        for name,title in zip(FAILURE_REPORT_FILES,('故障前各 GPU 状态','故障请求与调度','调度后恢复结果')):
            planned = '\n\n'+table(['轮次','计划故障 (s)'],[
                [n,number(t)] for n,t in enumerate(config.get('comparison_failure_times_s',[]),1)]) if name==FAILURE_REPORT_FILES[0] else ''
            (output/name).write_text(f'# {title}\n\n{progress}{planned}\n\n'
                f'{len(strategies)} 种分配、共 {len(variants)} 种组合；缓存统一 shared，全部请求直接进入原生等待队首。\n\n等待运行。\n',encoding='utf-8')
        return
    selected = {variant:[by_round[n][variant] for n in complete] for variant in variants}

    def average(values):
        samples = [v for v in values if v is not None]
        return statistics.mean(samples) if samples else None

    def mean_sd(values):
        samples = [v for v in values if v is not None]
        return (number(average(samples))+' ± '+number(statistics.stdev(samples) if len(samples)>1 else None)
                if samples else '—')

    rounds = [(n,items,next(iter(items.values()))) for n,items in sorted(by_round.items())]
    cpu_label = 'CPU前缀offload' if config.get('cpu_kv_backup_gb') else 'CPU KV'
    before = ['# 故障前各 GPU 状态',progress,
        f"模型：{Path(config.get('model','—')).name}；总 QPS：{number(config.get('aggregate_offered_qps'))}；"
        f"GPU：{'/'.join(map(str,config['gpus']))}；{cpu_label}：{number(config.get('cpu_cache_gb'),0)} GB。",
        f"源 trace：{Path(config.get('trace','—')).name}；背景 trace：{Path(config.get('survivor_trace') or config.get('trace','—')).name}。",
        f'每轮独立采样故障时刻并新建基线；轮内 {len(variants)} 种组合严格重放相同队列、计算进度和输出 token。'+
        ('故障前'+('源实例' if config.get('failure_stability_scope', 'all')=='source' else '所有实例')+
         '通过连续滚动窗口稳定性校验；不达标即停止。' if config.get('require_stable_before_failure')
         else '固定预热触发，不保证统计窗口稳定。'),
        *(['随机时间到达后，等待源实例下一个非空 running 边界触发并通知其余实例；计划/实际时刻分别记录，不按恢复结果筛选。']
          if config.get('failure_when_busy') else []),
        '缓存统一 shared：正常、转发及组内/组外恢复请求共用实例本地前缀池。全部故障请求直接入原生队首。',
        table(['轮次','计划故障 (s)','实际故障 (s)','延迟统计窗口 (s)','故障 running','轮内校验'],[
            [n,number(first['plan'].get('planned_failure_s')),number(first['plan']['failure_s']),
             ' / '.join(number(first['before']['gpus'][0].get(k)) for k in ('baseline_start_s','baseline_end_s')),
             len(first['plan']['assignment']),
             ('通过' if all(g.get('prefailure_verified') for r in items.values() for g in r['before']['gpus']) else '未通过')+
             f'（{len(items)}/{len(variants)}）'] for n,items,first in rounds]),
        table(['轮次','GPU','TTFT (ms)','TPOT (ms/token)','running','waiting'],[
            [n,g['gpu'],number(g.get('baseline',{}).get('avg_ttft_ms')),
             number(g.get('baseline',{}).get('avg_tpot_ms')),g.get('running_count'),g.get('waiting_count')]
            for n,_,first in rounds for g in first['before']['gpus']])]
    if config.get('source_qps') is not None:
        before.insert(3, f"独立速率：源实例 {number(config['source_qps'])} QPS；每个背景实例 {number(config['survivor_qps'])} QPS。")
    evidence = cpu_backup_evidence([(n,first['before']['gpus'][0].get('cpu_backup')) for n,_,first in rounds])
    if evidence:
        before.append(evidence+'\n\n同一轮后续组合重用第一组合冻结的CPU检查点，并重放GPU引用操作；故障前开销使用第一组合基线。')
    (output/FAILURE_REPORT_FILES[0]).write_text('\n\n'.join(before)+'\n',encoding='utf-8')
    schedule = ['# 故障请求与调度',progress,
        '各轮故障状态独立，轮内所有组合处理同一批请求及相同生成进度。同一分配策略的不同顺序保持目标 GPU 一致。'+
        ('恢复请求保留原输出、加载CPU KV并续接，直接进入原生队首；' if config.get('cpu_kv_backup_gb')
         else '恢复请求保留原输出、重算上下文后续接，直接进入原生队首；' if config.get('recovery_mode')=='resume'
         else '恢复请求从原始 prompt 重新生成并直接入原生队首；')+'原有 running 和后续流量继续处理。',
        '队列顺序：'+', '.join('shortest（prefix-shortest）' if order=='prefix-shortest' else order for order in orders)+'。']
    if config.get('recovery_cache_policy')=='shared':
        schedule.append('所有组合均使用无 Salt 的实例本地共享前缀池：正常、转发、组内及组外恢复请求都可复用。')
    for n,items,first in rounds:
        maps = {}
        for (strategy,_),result in items.items():
            maps.setdefault(strategy,{a['original_request_id']:a for a in result['plan']['assignment']})
        rows = []
        for a in sorted(first['plan']['assignment'],key=lambda a:a['original_request_id']):
            cells = []
            for strategy in strategies:
                item = maps.get(strategy,{}).get(a['original_request_id'])
                cells.append('—' if not item else str(item['target_gpu'])+
                             (' / '+item['recovery_group'] if strategy=='kv-aware' else ''))
            rows.append([a['original_request_id'],a['input_length'],
                         f"{a['source_generated_tokens']} / {a['output_length']}",a['source_phase'],
                         *([a.get('cpu_backup_tokens',0)] if config.get('cpu_kv_backup_gb') else []),*cells])
        schedule += [f"## 第 {n} 轮：故障于 {number(first['plan']['failure_s'])} 秒",
            table(['请求','输入 tokens','已生成 / 总输出','阶段']+
                  (['CPU可用KV tokens'] if config.get('cpu_kv_backup_gb') else [])+
                  [{'equal':'equal GPU','prefill-balanced':'prefill GPU','kv-aware':'KV GPU / 组'}[s] for s in strategies],rows)]
        if len(orders)>1:
            schedule.append('各 GPU：预计新增 prefill tokens；实际队首 → 队尾。')
            schedule.append(table(['策略 / 排队']+[f'GPU {g}' for g in config['gpus'][1:]],[
                [label(variant)]+[queue_cell(items[variant],gpu) for gpu in config['gpus'][1:]]
                for variant in variants if variant in items]))
    (output/FAILURE_REPORT_FILES[1]).write_text('\n\n'.join(schedule)+'\n',encoding='utf-8')

    after = ['# 三次重复的恢复结果' if repeats==3 else '# 多次重复的恢复结果',progress,
        '缓存统一 shared，无外层组内等待。每轮采样新的故障时刻，轮内所有组合共用故障状态。只用完整配对轮次计算均值和样本标准差，各轮等权，不按请求数混合；n=1 时标准差为“—”。TTFT 单位 ms，包含排队和 prefill；TPOT 单位 ms/token。',
        table(['策略 / 排队' if len(orders)>1 else '策略','完整轮数','TTFT 均值 ± 标准差','TPOT 均值 ± 标准差','平均命中 tokens'],[
            [label(variant),len(samples),mean_sd([r['after']['summary']['avg_ttft_ms'] for r in samples]),
             mean_sd([r['after']['summary']['avg_tpot_ms'] for r in samples]),
             number(average([r['after']['summary']['gpu_cached_prompt_tokens'] for r in samples]),0)]
            for variant,samples in selected.items()]),
        '## 逐轮实测',table(['轮次','策略 / 排队','恢复完成','TTFT','TPOT','全量完成','命中 tokens'],[
            [r['repeat'],label((r['strategy'],r.get('order',r['plan'].get('recovery_order',orders[0])))),
             f"{v['completed']}/{v['requests']}",number(v['avg_ttft_ms']),number(v['avg_tpot_ms']),
             f"{v['traffic']['completed']}/{v['traffic']['planned']}",v['gpu_cached_prompt_tokens']]
            for r in results for v in [r['after']['summary']]])]
    evidence = cpu_restore_evidence([(f"{r['repeat']} / "+label((r['strategy'],r['order'])),
                                     r['after']['summary']) for r in results])
    if evidence:
        after.append(evidence)
    gpu_rows = []
    for variant,items in selected.items():
        for gpu in config['gpus'][1:]:
            recovery = [g for r in items for g in r['after']['summary']['by_target'] if g['gpu']==gpu]
            normal = [g for r in items for g in r['after']['gpus'] if g['gpu']==gpu]
            gpu_rows.append([label(variant),gpu,
                number(average([g['avg_ttft_ms'] for g in recovery]))+' / '+number(average([g['avg_tpot_ms'] for g in recovery])),
                number(average([g['normal']['avg_ttft_ms'] for g in normal]))+' / '+number(average([g['normal']['avg_tpot_ms'] for g in normal])),
                number(average([g['post_failure']['mean_running'] for g in normal]))+' / '+number(average([g['post_failure']['mean_waiting'] for g in normal]))])
    after += ['## 每 GPU 重复间均值',table(['策略','GPU','恢复 TTFT / TPOT','正常 TTFT / TPOT','running / waiting'],gpu_rows)]
    if all('ttft_breakdown' in r['after']['summary'] for r in results):
        after += ['## 平均 TTFT 分解（ms）',
            '按完整轮次等权，两项之和为平均 TTFT；入队阶段包含提交开销，首次调度后仍可能经历多轮分块 prefill。',
            table(['策略','入队至首次调度','首次调度至首 token'],[
                [label(variant),*[number(average([r['after']['summary']['ttft_breakdown'][key] for r in selected[variant]]))
                     for key in ('queue_wait_ms','scheduled_to_first_ms')]] for variant in variants])]
    if config.get('comparison_cases'):
        after += ['## 调度开销计入后的故障至首 token',
            '目标入队 TTFT 不含分配算法执行时间；下表额外统计从故障时刻开始的实际耗时。',
            table(['组合','平均故障至首 token (ms)'],[
                [label(variant),number(average([average([q.get('failure_to_first_ms')
                    for q in r['after']['recovery_requests']]) for r in samples]))]
                for variant,samples in selected.items()])]
        group_rows = []
        for n,items,first in rounds:
            canonical = next((r for (s,_),r in items.items() if s=='kv-aware'),None)
            if canonical is None:
                continue
            groups = defaultdict(list)
            for row in canonical['plan']['assignment']:
                if row.get('recovery_group'):
                    groups[row['recovery_group']].append(row['original_request_id'])
            for ids in groups.values():
                if len(ids)<2:
                    continue
                cells = []
                for variant in variants:
                    measured = {r['original_request_id']:r for r in items.get(variant,{}).get('after',{}).get('recovery_requests',[])}
                    members = [measured[rid] for rid in sorted(ids) if rid in measured]
                    cells.append('—' if not members else number(average([r['ttft_ms'] for r in members]))+'；'+
                        str(sum(r.get('gpu_cached_tokens_first_schedule') or 0 for r in members))+'；'+
                        '/'.join(str(r.get('first_token_batch') or '—') for r in members))
                group_rows.append([n,', '.join(sorted(ids)),*cells])
        if group_rows:
            after += ['## 同一共享组的首 token',
                '各格：组内平均 TTFT (ms)；组内命中 tokens 合计；各成员首 token 批次（按请求编号，批次仅在各自 GPU 内有效）。',
                table(['轮次','共享请求',*[label(v) for v in variants]],group_rows)]
    request_rows = []
    for n,items,first in rounds:
        metrics = {s:{a['original_request_id']:a for a in r['after']['recovery_requests']} for s,r in items.items()}
        for rid in sorted(a['original_request_id'] for a in first['plan']['assignment']):
            cells = []
            for variant in variants:
                item = metrics.get(variant,{}).get(rid)
                cells.append('—' if not item else number(item['ttft_ms'])+' / '+number(item['tpot_ms'])+
                             ('；'+str(item.get('gpu_cached_tokens_first_schedule','—'))+' / '+
                              str(item.get('prefill_to_first_tokens','—')) if config.get('cpu_kv_backup_gb') else '')+
                             ('（未完成）' if item['status']!='completed' else ''))
            request_rows.append([n,rid,*cells])
    after += ['## 逐轮故障请求：TTFT / TPOT',
              *(['CPU模式每格：TTFT(ms) / TPOT(ms/token)；首次调度缓存tokens / 首token前实际补算tokens。']
                if config.get('cpu_kv_backup_gb') else []),
              table(['轮次','请求',*[label(v) for v in variants]],request_rows)]
    evidence = prefix_evidence([(f"{r['repeat']} / {r['strategy']}",r['after']['recovery_requests']) for r in results]) if len(orders)==1 else ''
    if evidence:
        after.append(evidence)
    (output/FAILURE_REPORT_FILES[2]).write_text('\n\n'.join(after)+'\n',encoding='utf-8')


def key(row):
    return int(row['repeat']), float(row['cpu_cache_gb']), float(row['global_qps'])


def load_config(root):
    """读取一组实验或历史独立 qps_* 容器目录的配置。"""
    path = root / 'config.json'
    if path.exists():
        return json.loads(path.read_text())
    configs = [json.loads(p.read_text()) for p in sorted(root.glob('qps_*/config.json'))]
    if not configs:
        raise FileNotFoundError(f'No experiment config found in {root}')
    config = configs[0].copy()
    for other in configs[1:]:
        for field in ('gpus', 'model', 'trace', 'trace_block_tokens', 'trace_time_unit',
                      'max_model_len', 'dtype', 'cpu_cache_gb', 'repeats', 'warmup', 'duration', 'arrival'):
            if other.get(field) != config.get(field):
                raise ValueError(f'Cannot combine runs with different {field}')
    config['qps'] = sorted({qps for c in configs for qps in c['qps']})
    return config


def find_case(root, capacity, repeat, qps):
    base = Path(f'cpu_gb_{capacity:g}/repeat_{repeat}')
    candidates = [root/base/f'qps_{qps:g}', root/base, root/f'qps_{qps:g}'/base]
    for path in candidates:
        if ((path/'summary.json').exists() or (path/'summary.csv').exists()) and any(
                path.glob(f'requests_qps_{qps:g}_gpu*.jsonl')):
            return path
    return candidates[0]


def summarize(rows, per_gpu=False):
    groups = defaultdict(list)
    for row in rows:
        group = float(row['global_qps']), float(row['cpu_cache_gb'])
        if per_gpu:
            group += (int(row['gpu']),)
        groups[group].append(row)
    result = []
    for group, values in sorted(groups.items()):
        item = {'global_qps': group[0], 'cpu_cache_gb': group[1]}
        if per_gpu:
            item['gpu'] = group[2]
        repeats = sorted({int(row['repeat']) for row in values})
        assert len(repeats) == len(values), 'Duplicate repeat in group'
        item.update(repeats_completed=len(repeats), repeats=','.join(map(str, repeats)))
        for metric in METRICS:
            samples = [float(row[metric]) for row in values if row.get(metric) not in (None, '')]
            if samples:
                item[metric + '_mean'] = statistics.mean(samples)
                # 只有一个样本时无法计算样本标准差，保留缺失值。
                item[metric + '_std'] = statistics.stdev(samples) if len(samples) > 1 else ''
        result.append(item)
    return result


def write_matched_comparison(output, target=None):
    """按同一卡、轮次、QPS 匹配请求，比较不同 CPU 容量，避免换了样本仍直接比较均值。"""
    output = Path(output)
    target = Path(target) if target is not None else output
    config = load_config(output)
    warmup, duration = config['warmup'], config['duration']
    rows, memberships = [], []
    for repeat in range(1, config['repeats']+1):
        for qps in config['qps']:
            cases = {}
            for capacity in config['cpu_cache_gb']:
                directory = find_case(output, capacity, repeat, qps)
                if not (directory/'summary.json').exists():
                    break
                records = {}
                for path in directory.glob(f'requests_qps_{qps:g}_gpu*.jsonl'):
                    for line in path.read_text().splitlines():
                        record = json.loads(line)
                        records[record['request_id']] = record
                cases[capacity] = records
            if len(cases) != len(config['cpu_cache_gb']):
                continue
            ttft_ids, tpot_ids = [], []
            for records in cases.values():
                ttft_ids.append({rid for rid, r in records.items()
                                 if r['first_s'] is not None and warmup <= r['first_s'] < duration})
                tpot_ids.append({rid for rid, r in records.items()
                                 if r['completed_s'] is not None
                                 and warmup <= r['completed_s'] < duration and r['tpot_ms'] is not None})
            common_ttft = sorted(set.intersection(*ttft_ids))
            common_tpot = sorted(set.intersection(*tpot_ids))
            memberships.append({'repeat': repeat, 'global_qps': qps,
                                'ttft_request_ids': common_ttft, 'tpot_request_ids': common_tpot})
            for capacity, records in cases.items():
                ttft = [records[rid]['ttft_ms'] for rid in common_ttft]
                tpot = [records[rid]['tpot_ms'] for rid in common_tpot]
                intervals = sum(records[rid]['output_length']-1 for rid in common_tpot)
                decode_ms = sum(records[rid]['tpot_ms']*(records[rid]['output_length']-1)
                                for rid in common_tpot)
                rows.append({
                    'cpu_cache_gb': capacity, 'repeat': repeat, 'global_qps': qps,
                    'matched_ttft_samples': len(ttft), 'matched_tpot_samples': len(tpot),
                    'matched_avg_ttft_ms': sum(ttft)/len(ttft) if ttft else None,
                    'matched_p95_ttft_ms': percentile(ttft, 95),
                    'matched_avg_tpot_ms': sum(tpot)/len(tpot) if tpot else None,
                    'matched_p50_tpot_ms': percentile(tpot, 50),
                    'matched_p95_tpot_ms': percentile(tpot, 95),
                    'matched_token_weighted_tpot_ms': decode_ms/intervals if intervals else None,
                })
    (target/'matched_requests.json').write_text(json.dumps(memberships, indent=2))
    if rows:
        with (target/'matched_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def extract(root, target=None):
    """从已结束的 case 提取逐卡、容量和重复轮次汇总，不重新执行推理。"""
    root = Path(root).resolve()
    config = load_config(root)
    target = Path(target) if target is not None else root / 'extracted'
    target.mkdir(parents=True, exist_ok=True)
    summaries, aggregates, sources = [], [], []
    seen_gpu, seen_aggregate = set(), set()
    directories = sorted({p.parent for p in root.rglob('summary.csv')
                          if any(part.startswith('cpu_gb_') for part in p.relative_to(root).parts)
                          and any(part.startswith('repeat_') for part in p.relative_to(root).parts)})
    for directory in directories:
        local = read_csv(directory / 'summary.csv')
        groups = defaultdict(list)
        for row in local:
            identity = key(row) + (int(row['gpu']),)
            assert identity not in seen_gpu, f'Duplicate row: {identity}'
            seen_gpu.add(identity)
            groups[key(row)].append(row)
        for row in read_csv(directory / 'aggregate.csv'):
            identity = key(row)
            assert identity not in seen_aggregate, f'Duplicate aggregate: {identity}'
            worker_rows = groups[identity]
            assert {int(r['gpu']) for r in worker_rows} == set(config['gpus'])
            assert len(worker_rows) == len(config['gpus'])
            assert abs(sum(float(r['completion_qps']) for r in worker_rows)
                       - float(row['completion_qps'])) < 1e-8
            for worker in worker_rows:
                raw = directory / f"requests_qps_{float(row['global_qps']):g}_gpu{worker['gpu']}.jsonl"
                assert raw.exists(), f'Missing raw requests: {raw}'
                with raw.open() as stream:
                    count = sum(bool(line.strip()) for line in stream)
                assert count == int(worker['planned']), f'Incomplete requests: {raw}'
            seen_aggregate.add(identity)
            summaries.extend(worker_rows)
            aggregates.append(row)
        if local:
            sources.append(str(directory.relative_to(root)))
    summaries.sort(key=lambda r: (float(r['global_qps']), float(r['cpu_cache_gb']),
                                  int(r['gpu']), int(r['repeat'])))
    aggregates.sort(key=lambda r: (float(r['global_qps']), float(r['cpu_cache_gb']), int(r['repeat'])))
    coverage = []
    for repeat in range(1, config['repeats'] + 1):
        for capacity in config['cpu_cache_gb']:
            for qps in config['qps']:
                complete = (repeat, float(capacity), float(qps)) in seen_aggregate
                coverage.append({'repeat': repeat, 'cpu_cache_gb': capacity,
                                 'global_qps': qps, 'status': 'completed' if complete else 'missing'})
    overall_stats = summarize(aggregates)
    gpu_stats = summarize(summaries, per_gpu=True)
    for name, rows in [('per_gpu_results.csv', summaries), ('overall_results.csv', aggregates),
                       ('per_gpu_statistics.csv', gpu_stats), ('overall_statistics.csv', overall_stats),
                       ('coverage.csv', coverage)]:
        write_csv(target / name, rows)

    status_path = root.with_suffix('.status')
    status = status_path.read_text().strip() if status_path.exists() else 'unknown'
    root_count = len(read_csv(root / 'aggregate.csv'))
    metadata = {'planned_trials': len(coverage), 'completed_trials': len(aggregates),
                'missing_trials': len(coverage) - len(aggregates), 'per_gpu_rows': len(summaries),
                'root_aggregate_rows': root_count, 'recovered_trial_count': len(aggregates) - root_count,
                'run_status': status, 'source_directories': sources,
                'standard_deviation': 'sample standard deviation; blank when n < 2',
                'validation': 'unique keys, complete GPU sets, summed completion QPS, raw request counts'}
    (target / 'extraction.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = [
        '# 实验结果提取', '',
        f"模型：{config.get('model', '未记录')}；精度设置：{config.get('dtype', 'legacy bfloat16')}。",
        f"Trace：{config.get('trace', '未记录')}；hash 块：{config.get('trace_block_tokens', 512)} tokens；时间戳单位：{config.get('trace_time_unit', 'ms')}。",
        f"计划 {len(coverage)} 组，已保存完整结果 {len(aggregates)} 组，缺失 {len(coverage)-len(aggregates)} 组。",
        f"每 GPU 共 {len(summaries)} 条结果。根目录原总表有 {root_count} 组，从分目录补提取 {len(aggregates)-root_count} 组。",
        '', '运行状态：' + status.replace('\n', '; '),
        f"配置每个组合重复 {config['repeats']} 次。统计表按实际完成次数计算均值和样本标准差；n=1 的标准差留空，缺失结果不计作零。",
        '每 GPU 统计见 per_gpu_statistics.csv，全部实例合计统计见 overall_statistics.csv；*_mean 为重复间均值，*_std 为样本标准差。',
        'P95 的重复间均值不是合并所有重复样本的 P95。统计窗口固定，不保证系统已达到稳态。',
        '提取只读取已有结果，没有重跑实验，也没有修改原始结果或根目录总表。', '',
        f"## {len(config['gpus'])} 卡合计", '',
        '| QPS | CPU GB/实例 | 重复数 | TTFT 秒 | P95 TTFT 秒 | TPOT ms | ITL ms | 完成 QPS | 平均 waiting/卡 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    by_setting = {(float(r['global_qps']), float(r['cpu_cache_gb'])): r for r in overall_stats}
    for qps in config['qps']:
        for capacity in config['cpu_cache_gb']:
            row = by_setting.get((float(qps), float(capacity)))
            if row is None:
                report.append(f'| {qps:g} | {capacity:g} | 0 | — | — | — | — | — | — |')
                continue
            report.append(f"| {qps:g} | {capacity:g} | {row['repeats_completed']} | "
                          f"{row['avg_ttft_ms_mean']/1000:.2f} | {row['p95_ttft_ms_mean']/1000:.2f} | "
                          f"{row['avg_tpot_ms_mean']:.2f} | {row['window_avg_itl_ms_mean']:.2f} | "
                          f"{row['completion_qps_mean']:.3f} | {row['mean_waiting_per_gpu_mean']:.2f} |")
    report += ['', '## 每 GPU 明细', '',
               'TTFT 包含请求排队；TPOT 是每个已完成请求的 TPOT 再平均；ITL 为测量窗口内逐 token 间隔。',
               'CPU_load_GB 是测量窗口内从 CPU 累计加载的 KV 字节数除以 1e9，不是缓存驻留量。',
               f"每组总时长 {config['duration']:g} 秒，前 {config['warmup']:g} 秒预热，统计窗口 {config['warmup']:g}–{config['duration']:g} 秒。QPS 为 {len(config['gpus'])} 卡合计输入速率。"]
    for qps in config['qps']:
        report += ['', f'### QPS={qps:g}', '',
                   '| GPU | CPU GB | 轮次 | TTFT 秒 | TPOT ms | ITL ms | running | waiting | 完成 QPS | CPU_load_GB |',
                   '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in sorted((r for r in summaries if float(r['global_qps']) == float(qps)),
                          key=lambda r: (int(r['gpu']), float(r['cpu_cache_gb']), int(r['repeat']))):
            report.append(f"| {row['gpu']} | {float(row['cpu_cache_gb']):g} | {row['repeat']} | "
                          f"{float(row['avg_ttft_ms'])/1000:.2f} | {float(row['avg_tpot_ms']):.2f} | "
                          f"{float(row['window_avg_itl_ms']):.2f} | {float(row['mean_running']):.2f} | "
                          f"{float(row['mean_waiting']):.2f} | {float(row['completion_qps']):.3f} | "
                          f"{float(row['cpu_load_bytes'])/1e9:.3f} |")
    (target / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    write_matched_comparison(root, target)
    print(json.dumps(metadata, indent=2))
    print('\n'.join(report[:30]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, help='Report directory; default: RESULTS/extracted')
    args = parser.parse_args()
    extract(args.results, args.output)
