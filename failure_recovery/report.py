"""故障实验的统计与三份 Markdown；不控制 GPU 调度和 KV 复制。

engine 用 update_latency 更新请求记录，experiment 用统计和写出函数生成报告。
离线空间估计入口是 summarize_recovery_placement，不能把估计值当实际 H2D 流量。
"""
from collections import defaultdict
import json
import hashlib
import math
import os
from pathlib import Path
import statistics

from metrics import percentile, latency_summary, StabilityGate


def metric_summary(records, lower=0., upper=math.inf):
    """在 [lower, upper) 内按事件汇总请求均值和分位数。

    TTFT 按首 token 时间筛选，TPOT 按完成时间筛选；未完成但已出首 token 的请求仍计 TTFT。
    """
    result = {}
    for name, event in [('ttft', 'first_s'), ('tpot', 'completed_s')]:
        values = [r[f'{name}_ms'] for r in records if r.get(event) is not None
                  and lower <= r[event] < upper and r.get(f'{name}_ms') is not None]
        result.update({f'{name}_requests': len(values),
                       f'avg_{name}_ms': sum(values)/len(values) if values else None,
                       f'p50_{name}_ms': percentile(values, 50),
                       f'p95_{name}_ms': percentile(values, 95)})
    return result


def update_latency(state, tokens, now, finished):
    """由本次累计输出 tokens 更新 state，返回本轮是否首次看到新输出。

    普通/转发 TTFT 从原计划到达算，恢复请求 arrival 已设为目标登记时间；
    恢复 TTFO 另从 source_failure_s 算，TPOT 只覆盖本次生成且至少两个 token 才有值。
    """
    first = tokens > state['last_tokens'] and state['first_s'] is None
    if tokens > state['last_tokens']:
        if first:
            state.update(first_s=now, ttft_ms=1000*(now-state['arrival_s']))
            if state.get('is_forwarded'):
                state['target_queue_to_first_ms'] = 1000*(now-state['admitted_s'])
            if state.get('is_recovery'):
                scheduled = state['first_scheduled_s']
                # TTFT=目标入队→本地观测首新 token；TTFO=源停止基准→同一 token，单位均为 ms。
                # 此处是 engine.step 的输出观测，不含 HTTP/网络传输时间。
                state.update(failure_to_first_ms=1000*(now-state['failure_s']),
                             ttfo_ms=1000*(now-state.get('source_failure_s',state['failure_s'])),
                             queue_wait_ms=1000*(scheduled-state['enqueued_s']),
                             scheduled_to_first_ms=1000*(now-scheduled),
                             first_token_index=1, first_token_observed_count=tokens,
                             interruption_ms=(1000*(now-state['source_last_token_s'])
                                              if state.get('source_last_token_s') is not None else None))
        state.update(last_tokens=tokens, last_token_s=now)
        if state.get('is_recovery'):
            # 字段名沿用旧命名；resume 时也只表示本次续接新增的输出数。
            state['generated_tokens_after_restart'] = tokens
    if finished:
        if state['first_s'] is None:
            raise RuntimeError('Completion without an observed first token')
        state.update(completed_s=now, status='completed',
                     tpot_ms=1000*(now-state['first_s'])/(tokens-1) if tokens > 1 else None)
    return first


def compact_record(row):
    """删去大 token 数组和内部 ID，仅缩短展示，不改变原始恢复记录。"""
    omitted = {'cache_salt', 'prompt_token_ids', 'generated_token_ids', 'tokens', 'hash_ids',
               'scheduler_request_id', 'arrival_epoch', 'first_epoch', 'completion_epoch',
               'continuation_token_ids'}
    return {k: v for k, v in row.items() if k not in omitted}


def observation_summary(loads, batches, lower=0., upper=math.inf):
    """按负载持续时长加权统计 running/waiting，再按 Step 统计预算与混合批次。"""
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
    """展示源 CPU 真实占用和备份开销，区分预算、当前 READY 和累计拷贝量。"""
    rows = [[label, _failure_number(s['capacity_bytes']/1e9), _failure_number(s['peak_bytes']/1e9),
             _failure_number(s['registered_bytes']/1e9),_failure_number(s['completed_store_bytes']/1e9),
             _failure_number(s['store_ms']),s['pending_blocks'],s['allocation_misses']]
            for label,s in items if s]
    text = ('CPU备份：仅源实例，完整16-token块异步增量复制；CPU RAM由父进程持有。'
            '故障时不等待或补拷未完成的DMA。\n\n'+_failure_table(
                ['轮次/GPU','预算GB','有效块峰值GB','注册RAM GB','完成D2H GB','D2H工作量ms','未完成块','容量不足次数'],rows)
            if rows else '')
    counts=[[label,s.get('capacity_blocks'),s.get('allocated_blocks'),s.get('ready_unique_blocks'),
             s.get('pending_blocks'),s.get('free_blocks'),s.get('fault_running_ready_unique_blocks')]
            for label,s in items if s and 'capacity_blocks' in s]
    if counts:
        text+='\n\n'+_failure_table(['轮次/GPU','物理Slot容量','当前占用','READY唯一块','PENDING块',
            '空闲Slot','故障running引用的READY唯一块'],counts)
        text+='\n\n累计完成D2H字节是历史传输量，当前READY字节是仍驻留的有效内容；两者不要求相等。'
    predictions=[[label,s.get('queue_prediction_calls',0),_failure_number(s.get('queue_prediction_ms',0))]
                 for label,s in items if s and s.get('queue_prediction_calls')]
    if predictions:
        text+='\n\n'+_failure_table(['轮次/GPU','在线恢复队列预测次数','预测CPU耗时ms'],predictions)
    reclaim=[[label,*[s['allocation_reclaim'][key] for key in
        ('checks','released_blocks','freed_gpu_blocks','retries','retry_successes')]]
        for label,s in items if s and 'allocation_reclaim' in s]
    if reclaim:
        text+='\n\n'+_failure_table(['轮次/GPU','失败后回收检查','释放备份引用块次','立即释放GPU块次',
            '分配重试','重试成功'],reclaim)
        text+='\n\n归还活跃块上的备份引用可能不立即增加空闲块，但能让后续原生抢占正常释放这些块。'
    return text


def cpu_restore_evidence(items):
    """展示真实导入与已有 GPU 复用量，避免把可用备份都算成 H2D。"""
    rows = [[label,_failure_number(s['load_bytes']/1e9),_failure_number(s['load_work_ms']),
             s['backup_tokens'],s['loaded_tokens'],_failure_number(s['mean_prefill_to_first_tokens'])]
            for label,summary in items if (s := summary.get('cpu_kv'))]
    return ('## CPU KV 加载与尾部补算\n\n'
            'TTFT从目标提交开始，到故障前进度后的下一个新token，包含H2D、入队同步、排队和尾部补算。'
            'TPOT只统计本次剩余输出；H2D工作量为各卡DMA耗时之和。\n\n'+_failure_table(
                ['组合/轮次','H2D GB','H2D工作量ms','CPU可用tokens合计','实际搬入tokens','平均首token前补算tokens'],rows)
            if rows else '')


def backup_range_text(ranges):
    """按原 prompt 加已确认输出的位置展示左闭右开区间。"""
    return ', '.join(f'[{start}, {end})' for start,end in ranges) if ranges else '无'


def batch_ranges(values):
    """无损压缩连续批次编号；编号只在各自 GPU 内有意义。"""
    spans=[]
    for value in sorted(set(values)):
        if spans and value==spans[-1][1]+1:
            spans[-1][1]=value
        else:
            spans.append([value,value])
    return ','.join(str(a) if a==b else f'{a}–{b}' for a,b in spans) or '—'


def placement_evidence(data):
    """展示前缀组和各目标覆盖空间，与源 CPU 全局去重容量分开统计。"""
    summary=data.get('placement_summary')
    if not summary:
        return ''
    gpus={r['worker']:r['gpu'] for r in data.get('estimated_prefill_by_target',[])}
    gpus.update({r['target_worker']:r['target_gpu'] for r in data.get('assignment',[])})
    rows=[[gpus.get(row['worker'],row['worker']),row['requests'],row['backup_unique_blocks'],
           _failure_number(row['backup_bytes']/1e9) if row['backup_bytes'] is not None else '—']
          for row in summary['by_target']]
    text='## 幸存目标的备份覆盖空间（统计）\n\n'
    text+='本轮硬预算是源实例共享CPU池，按全局Hash去重。此表按(目标实例, Block Hash)分别统计：同一目标内相同块一份，跨目标分别计数后相加，合计允许大于CPU池预算。仅计选中READY备份，不计源GPU运行KV、缺口重算KV和普通请求；实际H2D另行记录。\n\n'
    text+=_failure_table(['目标 GPU','故障请求数','备份KV唯一块','空间GB'],rows)
    total=summary['external_backup_bytes']
    text+=f"\n\n外部备份总计：{summary['external_backup_blocks']} 块 / "
    text+=(_failure_number(total/1e9)+' GB' if total is not None else '字节数待布局确认')
    text+=f"；共享CPU载体物理并集：{summary['cpu_cohort_unique_blocks']} 块；跨目标重复份数：{summary['cross_target_duplicate_blocks']} 块。"
    group_rows=[]
    for group in summary['groups']:
        targets=sorted({gpus.get(r['worker'],r['worker']) for r in group['members']})
        members=', '.join(f"{r['request_id']}(GPU{gpus.get(r['worker'],r['worker'])}:{r['queue_rank']})"
                          for r in group['members'])
        group_rows.append([group['group'],group['requests'],group['shared_prefix_tokens'],
                           '/'.join(map(str,targets)),members])
    text+='\n\n## 前缀组信息\n\n'
    text+='共同前缀按实际恢复输入的token序列计算并按16 tokens对齐；单请求组记0。成员括号内为目标GPU及队列位置，位置从0开始。组号只用于调度，不隔离缓存，组外相同Hash在同卡也可共享。\n\n'
    return text+_failure_table(['前缀组','请求数','共同前缀tokens','目标GPU','成员（GPU:队列位置）'],group_rows)


def failure_markdown(name, data):
    """把故障前、调度计划或恢复结果字典转换成对应的一份 Markdown。"""
    number, table = _failure_number, _failure_table
    titles = dict(zip(FAILURE_REPORT_FILES, ('故障前各 GPU 状态', '故障请求与调度', '调度后恢复结果')))
    parts = ['# '+titles[name]]
    if data.get('status') == 'error':
        reason = str(data.get('error', '未知错误')).strip().splitlines()[-1]
        return '\n\n'.join(parts+['运行失败。', '原因：'+reason[:500]])+'\n'
    if name == FAILURE_REPORT_FILES[0]:
        cfg = data.get('config', {})
        cpu_label = '源CPU KV备份' if cfg.get('cpu_kv_backup_gb') else 'CPU KV'
        parts.append(f"模型：{Path(cfg.get('model', '—')).name}；总 QPS：{number(cfg.get('aggregate_offered_qps'))}；"
                     f"GPU：{'/'.join(map(str,cfg.get('gpus', [])))}；{cpu_label}：{number(cfg.get('cpu_kv_backup_gb',0),3)} GB。")
        parts.append(f"备份策略：{cfg.get('backup_policy','full' if cfg.get('cpu_kv_backup_gb') else 'none')}；"
                     f"批次不变性：{cfg.get('batch_invariant',False)}；恢复方式：{cfg.get('recovery_mode','—')}。")
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
        if 'recovery_plan_reused' in data:
            method = (f"复用故障前预测计划（版本 {data['recovery_plan_version']}）"
                      if data['recovery_plan_reused'] else '故障时计算恢复计划')
            parts.append(f"计划来源：{method}；故障时校验/计算耗时："
                         f"{number(data['recovery_scheduling_ms'])} ms。")
        if data.get('recovery_cache_policy')=='shared':
            parts.append('缓存统一共享：组内、组外及正常请求可按真实前缀匹配；前缀组仅用于调度。')
        if order != 'source':
            parts.append(table(['目标 GPU','恢复队首 → 队尾'],[
                [gpu,' → '.join(r['original_request_id'] for r in
                    sorted((r for r in assigned if r['target_gpu']==gpu),key=lambda r:r.get('queue_rank',0)))]
                for gpu in sorted({r['target_gpu'] for r in assigned})]))
        evidence=placement_evidence(data)
        if evidence:
            parts.append(evidence)
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
        cpu_mode=any('cpu_backup_tokens' in r for r in assigned)
        if cpu_mode:
            parts.append('备份区间为故障时已确认 READY 的 KV，按“原 prompt + 已确认输出”从 0 编号，'
                         '使用左闭右开区间；共享区间按请求分别列出，物理内存只计一份。')
        parts.append(table(['请求编号','输入 tokens','已生成 / 总输出','已计算tokens','故障阶段','目标 GPU','队列位置']+
                           (['前缀组'] if strategy=='kv-aware' else [])+
                           (['READY备份区间','备份tokens'] if cpu_mode else []),[
            [r['original_request_id'],r.get('input_length'),
             f"{r.get('source_generated_tokens','—')} / {r.get('output_length','—')}",r.get('source_computed_tokens'),
             r.get('source_phase'),r['target_gpu'],r.get('queue_rank','—')]+
             ([r.get('recovery_group')] if strategy=='kv-aware' else [])+
             ([backup_range_text(r.get('cpu_backup_ranges',[])),r.get('cpu_backup_tokens',0)] if cpu_mode else [])
            for r in sorted(assigned,key=lambda row:row['original_request_id'])]))
    else:
        if 'summary' not in data:
            return '\n\n'.join(parts+['等待运行结束；尚无恢复结果。'])+'\n'
        summary = data['summary']
        fwd = summary.get('forwarding')
        traffic = summary.get('traffic')
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
        if summary.get('avg_ttfo_ms') is not None:
            parts.append('TTFO从源实例实际停止的Step边界计到故障进度后的首个新token，包含协调、排队、H2D与补算。'+
                f"均值 {number(summary['avg_ttfo_ms'])} ms；p50 {number(summary['p50_ttfo_ms'])} ms；p95 {number(summary['p95_ttfo_ms'])} ms。")
        cpu=summary.get('cpu_kv',{})
        if cpu.get('restore_protocol')=='eager-preload':
            overlap=cpu.get('kv_restore_mode')=='suffix-overlap'
            parts.append('恢复顺序：整批入队和GPU预留 → 批量H2D'+
                ('与真实缺失前缀计算并行' if overlap else '完成后才计算缺口')+
                '。三卡均确认加载完成后才统一允许拼接续接。后缀不再次H2D，不伪造计算进度。')
            parts.append(table(['恢复 GPU','H2D GB','准备+加载/头部计算墙钟 ms','GPU H2D ms','放行前头部tokens','H2D/前向包络交集 ms'],[
                [t['gpu'],number(t['load_bytes']/1e9),number(t['total_wall_ms']),number(t['h2d_gpu_ms']),
                 t['head_tokens'],number(t['copy_forward_envelope_overlap_ms'])]
                for t in cpu.get('by_target_transfer',[])]))
            parts.append('GPU H2D事件时间是加载阶段的子区间；三卡并发，不能合计成屏障耗时。'
                '交集取CUDA复制事件与含恢复前缀批次的前向事件包络，包含包络内主机间隙，不等于纯计算内核重叠。')
            if 'join_wait_deferrals' in cpu:
                parts.append(f"后缀门控请求临时跳过 {cpu['join_wait_deferrals']} 次；本轮继续扫描其他可执行请求，"
                    '调度结束后恢复原优先级。全部不可执行时仍正常结束本轮，计数不等于请求数。')
                parts.append(table(['GPU','批次','调度tokens / 预算','恢复请求数','waiting'],[
                    [target['gpu'],batch['batch_id'],f"{batch['total_scheduled_tokens']} / {batch['token_budget']}",
                     sum(r['cohort']=='recovery' for r in batch['scheduled_requests']),batch['waiting']]
                    for target in cpu.get('initial_recovery_batches',[]) for batch in target['batches']]))
        if cpu.get('restore_protocol')=='on-demand':
            parts.append('按需恢复：全部请求登记后即允许推进，不等待整批KV加载。每个前缀组轮到时去重加载；'
                'Prefix先加载再补后段；Suffix串行先补前段再加载，Suffix并行在前段调度时发起异步加载。'
                '后缀在真实前段与H2D均完成后接入原生块表，小尾部仍由原生Scheduler混批执行。')
            parts.append(table(['GPU','按需加载组数','H2D GB','GPU H2D累计ms','重叠包络累计ms'],[
                [t['gpu'],len(t.get('groups',[])),number(t['load_bytes']/1e9),number(t['h2d_gpu_ms']),
                 number(t['copy_forward_envelope_overlap_ms'])] for t in cpu.get('by_target_transfer',[])]))
            parts.append('各组加载区间可能互相重叠，累计区间耗时不是TTFO关键路径墙钟；'
                'CUDA包络交集也不等于纯内核重叠。所有加载均在请求入队后，计入TTFT/TTFO。')
        if data.get('records_path'):
            parts.append(f"故障输入、生成进度和最终Token IDs保存在 `{data['records_path']}`；此文件不包含KV张量字节。")
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
        parts.append('批次编号各GPU独立计数；首调度/首token批、全部prefill批和完整调度批范围均分别记录。')
        cpu_mode = 'cpu_kv' in summary
        parts.append(table(['请求编号','目标 GPU','TTFT (ms)','TPOT (ms/token)','结果']+
                           (['READY备份区间','CPU可用tokens','首次调度缓存tokens','首token前补算tokens'] if cpu_mode else [])+
                           ['首调度 / 首token批','prefill批','全部调度批'],[
            [r['original_request_id'],r['target_gpu'],number(r.get('ttft_ms')),number(r.get('tpot_ms')),
             '完成' if r.get('status')=='completed' else '未完成']+
            ([backup_range_text(r.get('cpu_backup_ranges',[])),r['cpu_backup_tokens'],
              r.get('gpu_cached_tokens_first_schedule'),r.get('prefill_to_first_tokens')]
             if cpu_mode else [])+
            [f"{r.get('first_schedule_batch','—')} / {r.get('first_token_batch','—')}",
             batch_ranges(r.get('prefill_batch_ids',[])),batch_ranges(r.get('scheduled_batch_ids',[]))]
            for r in sorted(data.get('recovery_requests', []),key=lambda row:row['original_request_id'])]))
        evidence = prefix_evidence([(data.get('summary',{}).get('recovery_order','source'),data.get('recovery_requests',[]))])
        if evidence:
            parts.append(evidence)
    return '\n\n'.join(parts)+'\n'


def write_failure_report(path, data):
    """写出一份 Markdown；path 决定使用哪种报告模板。"""
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


def initialize_results(output, config):
    """运行前建立三份占位报告，失败或尚未故障时也有明确状态。"""
    for name in FAILURE_REPORT_FILES:
        write_failure_report(output/name, dict(status='pending',
            **({'config': config} if name == FAILURE_REPORT_FILES[0] else {})))


def write_selective_suite(output, runs, error=None):
    """汇总选择性备份对照，只对共同故障现场展示恢复结果和校验状态。"""
    output=Path(output)
    number,table=_failure_number,_failure_table
    def fingerprint(record):
        rows=sorted((c['request_id'],list(c['prompt_token_ids']),c['generated_token_ids'],
                     c['source_computed_tokens']) for c in record['checkpoints'])
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    baseline=fingerprint(runs[0]['records']) if runs else None
    strict=bool(runs and all(r['records']['config'].get('inference_replay') for r in runs))
    before=['# 四组在线备份实验',
        'Qwen3-8B、SQuALITY；源1.5 QPS，背景各0.5 QPS；四实例TP=1。固定kv-aware/group-first、resume、'
        'batch=8192、显存比例0.90、预热60秒、seed=20261022。所有组开启VLLM_BATCH_INVARIANT=1。',
        '参考历史实验使用restart；本次统一计故障进度后的新token，不能与历史TTFT直接混算。',
        table(['组别','源停止秒数','running','waiting','池容量Slots','READY唯一块','PENDING','与E0故障状态完全相同'],[
            [run['case'],number(run['records']['plan']['source_stopped_forward_s']),
             len(run['records']['checkpoints']),len(run['records']['plan']['source_waiting_forwarded']),
             run['records']['summary'].get('cpu_kv',{}).get('source_backup',{}).get('capacity_blocks',0),
             run['records']['summary'].get('cpu_kv',{}).get('source_backup',{}).get('ready_unique_blocks',0),
             run['records']['summary'].get('cpu_kv',{}).get('source_backup',{}).get('pending_blocks',0),
             '是' if fingerprint(run['records'])==baseline else '否'] for run in runs]),
        '不同策略各自真实运行在线D2H并使用独立CPU池。在线故障集合若不同，下表仅为端到端观测，不能将均值差直接归因于备份位置。']
    if strict:
        before[0]='# 四组故障现场严格回放'
        before[-1]='E0记录提交游标、每步真实前向与输出；E1/E2/E3逐步重算并核验。各组独立进行在线D2H、淘汰与READY发布，不重放其他策略的CPU操作或Manifest。'
        before+=['## 物理运行时间与幸存GPU缓存',
            '严格固定推理工作量和故障状态；故障前使用参考逻辑时钟，实际耗时可因备份开销不同。正常下一次前向前等待上一批D2H并释放引用，以保持原生块释放顺序；故障边界不等待Pending。此控制限制备份/下一步计算交叠，不能用逻辑QPS/TTFT评价自然备份吞吐开销。',
            table(['组别','源故障前物理耗时秒','四卡推理状态核验','三幸存GPU缓存/引用/LRU核验'],[
                [run['case'],number(run['records']['plan']['prefailure'][0]['wall_s']),
                 '通过' if all(r['verified'] for r in run['records']['plan']['prefailure']) else '失败',
                 '通过' if all(r['gpu_cache_signature'] for r in run['records']['plan']['prefailure'][1:]) else '失败']
                for run in runs])]
    schedule=['# 故障请求、分配与备份区间',
        '请求分配和排队使用授权旧版scheduling.py，未修改算法。全部故障请求直接进入原生队首。',
        table(['组别','目标GPU','故障请求数','CPU实际加载tokens','首token前实际prefill tokens'],[
            [run['case'],gpu,len(rows),sum(r.get('cpu_loaded_tokens',0) for r in rows),
             sum(r['prefill_to_first_tokens'] for r in rows)]
            for run in runs for gpu in sorted({r['target_gpu'] for r in run['records']['recovery_requests']})
            for rows in [[r for r in run['records']['recovery_requests'] if r['target_gpu']==gpu]]]),
        '完整故障前各GPU状态、逐请求编号/生成进度/分配和区间见各组同名三份Markdown；原始Token IDs与Manifest路径记录在各组恢复文档。']
    after=['# 四组恢复结果',
        table(['组别','恢复完成','平均TTFO ms','平均TPOT ms/token','当前READY GB','H2D GB','平均prefill tokens','独立E0逐token校验'],[
            [run['case'],f"{run['records']['summary']['completed']}/{run['records']['summary']['requests']}",
             number(run['records']['summary']['avg_ttfo_ms']),number(run['records']['summary']['avg_tpot_ms']),
             number(run['records']['summary'].get('cpu_kv',{}).get('source_backup',{}).get('ready_unique_bytes',0)/1e9),
             number(sum(r.get('cpu_load_bytes',0) for r in run['records']['recovery_requests'])/1e9),
             number(statistics.mean(r['prefill_to_first_tokens'] for r in run['records']['recovery_requests'])),
             '全重算基准' if run['case']=='E0' else
             ('通过' if run.get('verified') else '待校验')]
            for run in runs]),
        'TTFO从源实例实际停止的Step边界计到第一个新token，包括协调、排队、H2D和补算。'
        '每个备份组另用其自身故障进度做独立E0重计算，逐token校验，不假设它与在线E0的故障集合相同。']
    if strict:
        after[-1]='四组的故障请求、已生成Token IDs、已计算上下文、running/waiting、幸存GPU缓存与LRU均已严格核验。备份组与同现场E0的完整实际恢复输出逐token比较。TTFO用源实际停止的物理时刻计时，包含协调、排队、H2D及补算；故障前推理为受控回放。'
    controlled=[]
    for label in ('controlled_E1','controlled_E2','controlled_E3'):
        path=Path('/tmp')/(output.name+'_'+label+'_records.json')
        if path.exists():
            controlled.append(json.loads(path.read_text()))
    if controlled:
        signatures=[hashlib.sha256(json.dumps(sorted((rid,c['prompt'],c['progress'],c['computed_tokens'])
            for rid,c in run['source']['checkpoints'].items())).encode()).hexdigest() for run in controlled]
        matched=len(set(signatures))==1
        samples=[r for r in controlled[0]['results'] if r['case']=='E0']
        groups=[('E0',samples,0)]+[(rows['results'][-1]['case'],
            [r for r in rows['results'] if r['case']!='E0'],rows['source']['source_stats']['capacity_blocks']) for rows in controlled]
        after+=['## 同一故障现场的三卡微基准',
            f'请求、输入、生成进度与已计算上下文匹配：{"通过" if matched else "未通过，不可配对"}。'
            '每个备份策略均重新运行源端真实D2H；固定3请求、1024-token内容片段，非在线负载性能排行。',
            table(['组别','全局Slot容量','请求数','平均TTFO ms','平均CPU加载tokens','平均实际prefill tokens'],[
                [label,capacity,len(rows),number(statistics.mean(r['ttfo_ms'] for r in rows)),
                 number(statistics.mean(r['cpu_loaded_tokens'] for r in rows)),
                 number(statistics.mean(r['prefill_tokens'] for r in rows))] for label,rows,capacity in groups])]
    after+=['## 详细记录',*[
        f"- {run['case']}：[故障前]({run['directory']}/before_failure.md)、"
        f"[请求与调度]({run['directory']}/failure_schedule.md)、[恢复]({run['directory']}/after_recovery.md)"
        for run in runs]]
    if error:
        after+=['## 未完成原因',str(error)]
    for name,parts in zip(FAILURE_REPORT_FILES,(before,schedule,after)):
        (output/name).write_text('\n\n'.join(parts)+'\n')


def write_async_profile_suite(output,runs):
    """并列展示异步备份的物理耗时和首次回放分歧，不把失败组当成通过。"""
    output=Path(output)
    table=_failure_table
    labels={'engine_step':'引擎执行/输出取回','native_scheduler':'原生调度',
        'schedule_observation':'调度观测包装','candidate_scan':'候选/hash遍历',
        'selection_and_eviction':'排序/选择/淘汰','host_registration':'内存注册/页固定',
        'dma_descriptors':'DMA描述符构造','dma_enqueue':'DMA提交CPU开销',
        'backup_submit':'备份其他管理','backup_poll':'非阻塞poll/引用释放',
        'allocation_reclaim':'分配失败后非阻塞回收','allocation_retry':'原生分配重试',
        'metadata_build':'Manifest目录维护','metadata_events':'Manifest变化标记','metadata_publish':'元数据发布管理',
        'ipc_serialization':'IPC序列化','ipc_transport_wait':'IPC发送/阻塞',
        'replay_verification':'回放校验/诊断','replay_pacing':'参考节拍等待',
        'request_admission':'请求准入','output_bookkeeping':'输出记账',
        'batch_bookkeeping':'批次记账','loop_bookkeeping':'负载统计',
        'backup_setup':'备份初始化','replay_setup':'参考加载','idle':'空闲等待',
        'unattributed':'未归因'}
    before=['# 异步备份与严格回放校验',
        '沿用原E0的四卡参考，不重采样。各组真实执行自己的异步D2H；正常Step间无D2H同步，故障只冻结此前确认READY的内容。',
        table(['组别','结果','源已核验Step','实际停止秒','比较边界','同边界E0秒','同边界本组秒','增加秒'],[
            [r['case'],r['status'],r['source'].get('verified_steps'),
             f"{r['source']['clock']['wall_ms']/1000:.2f}",r['boundary'],
             f"{r['baseline_cut']['wall_ms']/1000:.2f}",f"{r['actual_cut']['wall_ms']/1000:.2f}",
             f"{(r['actual_cut']['wall_ms']-r['baseline_cut']['wall_ms'])/1000:+.2f}"] for r in runs]),
        '失败组只比较首次分歧前的相同推理工作量，不与E0完整185秒相减。源实际停止时间不含退出时的DMA清理等待。']
    diagnosis=['# 首次分歧与故障状态',
        '逐Step校验以及故障Running/Waiting、计算/生成进度、幸存GPU缓存校验均保留；分歧组不产生有效恢复对照。']
    after=['# 物理耗时分解',
        '下表为主线程互斥墙钟区间，可与未归因项合计为总时间；GPU DMA和Coordinator是并发维度，单独报告，不能与主线程耗时相加。']
    for r in runs:
        a,b=r['actual_cut'],r['baseline_cut']
        keys=set(a['exclusive_ms'])|set(b['exclusive_ms'])|{'unattributed'}
        def value(c,k):
            return c['unattributed_ms'] if k=='unattributed' else c['exclusive_ms'].get(k,0.)
        rows=sorted(((k,value(b,k),value(a,k)) for k in keys),key=lambda x:(abs(x[2]-x[1]),x[2]),reverse=True)
        shown=rows[:10]
        if rows[10:]:
            shown.append(('其他阶段合计',sum(v0 for _,v0,_ in rows[10:]),sum(v1 for _,_,v1 in rows[10:])))
        after+=[f"## {r['case']}：{r['boundary']}",table(['阶段','E0秒','本组秒','差值秒'],[
            [labels.get(k,k),f'{v0/1000:.3f}',f'{v1/1000:.3f}',f'{(v1-v0)/1000:+.3f}']
            for k,v0,v1 in shown]),
            f"全部阶段合计：E0 {b['wall_ms']/1000:.3f}s，本组 {a['wall_ms']/1000:.3f}s；"
            f"同边界已确认D2H事件累计 {a['counters'].get('d2h_gpu_ms',0)/1000:.3f}s；"
            f"发送元数据 {a['counters'].get('ipc_bytes',0)/1e6:.2f}MB。完整小项保存在原始profile。"]
        cpu=r['source'].get('cpu_backup')
        if cpu:
            after.append(f"实际停止时CPU READY={cpu['ready_unique_blocks']}，PENDING={cpu['pending_blocks']}，"
            f"Step间D2H显式等待={cpu['replay_copy_wait_ms']:.3f}ms。")
        parent=r.get('coordinator')
        if parent:
            after.append('Coordinator独立累计：接收/反序列化 '
            f"{parent['exclusive_ms'].get('ipc_receive_deserialize',0)/1000:.3f}s，元数据应用 "
            f"{parent['exclusive_ms'].get('coordinator_metadata_apply',0)/1000:.3f}s；这些时间不与源线程相加。")
        divergence=r.get('divergence')
        diagnosis.append(f"## {r['case']}：{r['status']}")
        if divergence:
            diagnosis.append(f"首次分歧：worker {divergence['worker']}，Step索引 {divergence['step']}（从0计）。")
            expected=divergence.get('reference_detail')
            actual=divergence.get('actual_detail')
            if expected and actual:
                e={x[0]:x[1:] for x in expected['scheduled']}
                p={x[0]:x[1:] for x in actual['scheduled']}
                changed=[key for key in e.keys()|p.keys() if e.get(key)!=p.get(key)]
                diagnosis.append(table(['请求','E0 调度tokens/已算tokens','本组 调度tokens/已算tokens'],[
                    [key,str(e.get(key,'未调度')),str(p.get(key,'未调度'))] for key in sorted(changed)[:10]]))
                diagnosis.append('E0分配器：`'+json.dumps(expected.get('allocator',{}))+'`；本组：`'+json.dumps(actual.get('allocator',{}))+'`。')
        else:
            diagnosis.append('本组参考推理与最终现场核验通过。'+r.get('verification',''))
        for parts in (before,after):
            parts.append(f"{r['case']}原始计时：[目录]({os.path.relpath(r['profile_dir'],output)})；[运行日志]({os.path.relpath(r['log'],output)})。")
    for name,parts in zip(FAILURE_REPORT_FILES,(before,diagnosis,after)):
        (output/name).write_text('\n\n'.join(parts)+'\n')


def summarize_recovery_placement(assignments, mode='resume', block_bytes=None):
    """结束后按目标 GPU 分别去重选中 Hash，再汇总覆盖空间和前缀组。

    输入为分配表；返回报告统计，同 Hash 跨 GPU 各算一份，但源 CPU 池仍只存一份。
    """
    from .scheduling import _common_length, KV_BLOCK_TOKENS
    if mode not in ('restart','resume'):
        raise ValueError('Unknown recovery mode')
    if block_bytes is not None and block_bytes<=0:
        raise ValueError('KV block bytes must be positive')
    targets,all_keys,all_slots,groups=[],set(),set(),defaultdict(list)
    def size(count):
        return count*block_bytes if block_bytes is not None else (0 if not count else None)
    for worker,rows in sorted(assignments.items()):
        # 同目标跨组也能共享；跨 GPU 则需要不同物理副本，不能全局只算一份目标空间。
        keys=set()
        for index,row in enumerate(rows):
            rid=row.get('request_id',row.get('original_request_id'))
            tokens=tuple(row.get('prompt_token_ids',()))
            if mode=='resume':
                tokens+=tuple(row.get('generated_token_ids',()))
            groups[row.get('recovery_group') or 'single:'+rid].append(
                (worker,row.get('queue_rank',index),rid,tokens))
            keys.update(ref['block_key'] for ref in row.get('cpu_kv',{}).get('blocks',[]))
            all_slots.update((ref['cpu_slot'],ref['generation']) if 'cpu_slot' in ref else ('hash',ref['block_key'])
                             for ref in row.get('cpu_kv',{}).get('blocks',[]))
        all_keys.update(keys)
        targets.append(dict(worker=worker,requests=len(rows),backup_unique_blocks=len(keys),
                            backup_bytes=size(len(keys))))
    summaries=[]
    for group,members in sorted(groups.items()):
        sequences=[r[3] for r in members]
        shared=(_common_length(min(sequences),max(sequences))//KV_BLOCK_TOKENS*KV_BLOCK_TOKENS
                if len(sequences)>1 else 0)
        summaries.append(dict(group=group,requests=len(members),shared_prefix_tokens=shared,
            members=[dict(worker=w,queue_rank=rank,request_id=rid)
                     for w,rank,rid,_ in sorted(members,key=lambda r:(r[0],r[1]))]))
    total=sum(row['backup_unique_blocks'] for row in targets)
    return dict(block_bytes=block_bytes,by_target=targets,groups=summaries,
        external_backup_blocks=total,external_backup_bytes=size(total),
        cpu_cohort_unique_blocks=len(all_slots),cpu_cohort_unique_bytes=size(len(all_slots)),
        unique_native_hashes=len(all_keys),
        cross_target_duplicate_blocks=total-len(all_keys))
