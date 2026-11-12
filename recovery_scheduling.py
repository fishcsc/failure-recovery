"""旧 failure_sampling 的三种故障请求调度，保留历史算法与接口。

入口 assign_recovery_requests/order_recovery_requests，仅估计新增 Prefill。
当前实验对应 failure_recovery/scheduling.py，两者不互相调用。
"""
from collections import defaultdict, deque

KV_BLOCK_TOKENS = 16  # 与当前 EngineArgs(block_size=16) 保持一致。


def _prepare(checkpoints, survivors, mode):
    if not survivors or len(set(survivors)) != len(survivors):
        raise ValueError('Recovery needs distinct surviving instances')
    if mode not in ('restart', 'resume'):
        raise ValueError('Unknown recovery mode')
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        tokens = tuple(checkpoint.get('prompt_token_ids', ()))
        if mode == 'resume':
            tokens += tuple(checkpoint.get('generated_token_ids', ()))
        length = len(tokens) or checkpoint.get('input_length', 0)
        rows.append(dict(checkpoint, recovery_id=f"REC0_{checkpoint['request_id']}_{index:05d}",
                         prefill_tokens=length, _tokens=tokens, _index=index))
    return rows, {w: [] for w in survivors}


def _place(row, worker, cost=None, **metadata):
    return dict({k:v for k,v in row.items() if not k.startswith('_')}, target_worker=worker,
                estimated_prefill_tokens=row['prefill_tokens'] if cost is None else cost, **metadata)


def assign_equal_count(checkpoints, survivors, mode='restart'):
    """基线：保持源 running 顺序，按请求数 round-robin 均分。"""
    rows, assigned = _prepare(checkpoints, survivors, mode)
    for index, row in enumerate(rows):
        worker = survivors[index % len(survivors)]
        assigned[worker].append(_place(row, worker))
    return assigned


def assign_prefill_balanced(checkpoints, survivors, mode='restart'):
    """LPT 只决定目标实例；分配结束后恢复源 running 顺序，避免长请求插队。"""
    rows, assigned = _prepare(checkpoints, survivors, mode)
    loads = dict.fromkeys(survivors, 0)
    for row in sorted(rows, key=lambda r:(-r['prefill_tokens'],r['_index'])):
        worker = min(survivors, key=lambda w:(loads[w],len(assigned[w]),survivors.index(w)))
        assigned[worker].append(_place(row, worker))
        loads[worker] += row['prefill_tokens']
    rank = {r['recovery_id']:r['_index'] for r in rows}
    for items in assigned.values():
        items.sort(key=lambda r:rank[r['recovery_id']])
    return assigned


def _common_length(left, right):
    for index, (a,b) in enumerate(zip(left,right)):
        if a != b:
            return index
    return min(len(left),len(right))


def _shared_blocks(left, right):
    # 估算 left 可提供的完整前缀块；当前 right 至少重算最后一个 prompt token。
    # 短 prompt 是长 prompt 的完整前缀时，left 最后一块也可供 right 复用。
    length = min(_common_length(left,right),len(right)-1)
    return max(0,length//KV_BLOCK_TOKENS*KV_BLOCK_TOKENS)


def _group_cost(group):
    # group 按 token 字典序排列；相邻 LCP 可计算前缀树去重后的总计算量。
    return sum(len(row['_tokens'])-(0 if index==0 else
               _shared_blocks(group[index-1]['_tokens'],row['_tokens']))
               for index,row in enumerate(group))


def _split_group(group):
    """优先按更深的前缀分支拆组；完全相同的 prompt 过重时再按数量二分。"""
    depth = _common_length(group[0]['_tokens'],group[-1]['_tokens'])//KV_BLOCK_TOKENS*KV_BLOCK_TOKENS
    children = defaultdict(list)
    for row in group:
        children[row['_tokens'][depth:depth+KV_BLOCK_TOKENS]].append(row)
    if len(children)>1:
        return list(children.values())
    middle = len(group)//2
    return [group[:middle],group[middle:]]


def assign_kv_aware(checkpoints, survivors, mode='restart', min_shared_tokens=16, split_factor=1.2):
    """真实 token 前缀分组 → 去重估算 → 按组 LPT；放不平衡的重组继续拆分。

    预算为 max(最长单请求, 当前去重总量/实例数) × split_factor。
    拆组会重复计算公共前缀，故每次拆分后重算预算；单个请求不拆。
    这是前缀可复用时的计算量估计；实际执行顺序由原生队列决定，不保证命中。
    """
    if min_shared_tokens<16 or min_shared_tokens%KV_BLOCK_TOKENS:
        raise ValueError('min_shared_tokens must be a positive multiple of 16')
    if not 1 <= split_factor < float('inf'):
        raise ValueError('split_factor must be finite and at least 1')
    rows, assigned = _prepare(checkpoints,survivors,mode)
    buckets = defaultdict(list)
    for row in rows:
        if not row['_tokens']:
            raise ValueError('KV-aware scheduling requires actual prompt token IDs')
        # 小于一个有效共享前缀的请求单独成组，避免误记为可共享。
        key = (('prefix',row['_tokens'][:min_shared_tokens]) if len(row['_tokens'])>min_shared_tokens
               else ('single',row['_index']))
        buckets[key].append(row)
    pending = [sorted(group,key=lambda r:(r['_tokens'],r['_index'])) for group in buckets.values()]
    loads = dict.fromkeys(survivors,0)
    largest = max((r['prefill_tokens'] for r in rows),default=0)
    group_number = 0
    while pending:
        pending.sort(key=lambda g:(-_group_cost(g),min(r['_index'] for r in g)))
        total = sum(loads.values())+sum(_group_cost(g) for g in pending)
        limit = max(largest,total/len(survivors))*split_factor
        group = pending.pop(0)
        cost = _group_cost(group)
        worker = min(survivors,key=lambda w:(loads[w],len(assigned[w]),survivors.index(w)))
        if len(group)>1 and loads[worker]+cost>limit:
            pending.extend(_split_group(group))
            continue
        group_id = f'G{group_number:04d}'
        group_number += 1
        for index,row in enumerate(group):
            shared = 0 if index==0 else _shared_blocks(group[index-1]['_tokens'],row['_tokens'])
            # 字典序仅用于估算共享量；所有请求均直接入队，不建立放行依赖。
            assigned[worker].append(_place(row,worker,len(row['_tokens'])-shared,
                recovery_group=group_id,group_position=index,shared_prefix_tokens=shared))
        loads[worker] += cost
    rank = {r['recovery_id']:r['_index'] for r in rows}
    for items in assigned.values():
        items.sort(key=lambda r:rank[r['recovery_id']])
    return assigned


def assign_recovery_requests(checkpoints,survivors,strategy='equal',mode='restart',
                             min_shared_tokens=16,split_factor=1.2):
    """唯一策略入口；后续流量的普通 round-robin 转发不受此选项影响。"""
    if strategy=='equal':
        return assign_equal_count(checkpoints,survivors,mode)
    if strategy=='prefill-balanced':
        return assign_prefill_balanced(checkpoints,survivors,mode)
    if strategy=='kv-aware':
        return assign_kv_aware(checkpoints,survivors,mode,min_shared_tokens,split_factor)
    raise ValueError(f'Unknown recovery scheduler: {strategy}')


def _order_prefix_bundles(rows, tokens, rank, shared_first):
    """整组排队、不串行放行；按组新增计算量/成员数估算平均首 token 收益。

    shared_first=True 时强制多成员组先于所有单请求；False 时允许短单请求
    与共享组竞争，避免一个较重共享组让许多短请求一起延迟。
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('recovery_group') or row['recovery_id']].append(row)
    ordered,seen = [],[]
    def estimate(items):
        pending,prefixes,sequence,total = list(items),list(seen),[],0
        while pending:
            def cost(row):
                prompt = tokens[row['recovery_id']]
                return len(prompt)-max((_shared_blocks(prior,prompt) for prior in prefixes),default=0)
            index = min(range(len(pending)),key=lambda i:(cost(pending[i]),rank[pending[i]['recovery_id']]))
            row = pending.pop(index)
            total += cost(row)
            prefixes.append(tokens[row['recovery_id']])
            sequence.append(row)
        return total,sequence
    while groups:
        candidates = [key for key,items in groups.items() if len(items)>1] if shared_first else []
        estimates = {key:estimate(groups[key]) for key in candidates or groups}
        key = min(estimates,key=lambda key:(estimates[key][0]/len(groups[key]),estimates[key][0],
                                           min(rank[r['recovery_id']] for r in groups[key])))
        sequence = estimates[key][1]
        ordered.extend(sequence)
        seen.extend(tokens[r['recovery_id']] for r in sequence)
        del groups[key]
    return ordered


def order_recovery_requests(items, policy='source', mode='restart'):
    """只调整目标实例内的恢复顺序，不改变分配、不延迟入队。

    interleave 优先展开成员最多的组，用其他组成员隔开；prefix-shortest
    每次选择预计新增 prefill 最少的请求，已排在前面的任意请求均可提供前缀，
    不用组号隔离估算；目标已有正常请求的 KV 是否命中由原生调度器判断。
    估算仍不是命中保证；真实批次、命中和计算量由原生调度器决定。
    """
    if policy not in ('source','interleave','prefix-shortest','group-first','group-ratio'):
        raise ValueError('Unknown recovery queue order')
    rows = [dict(row) for row in items]
    rank = {r['recovery_id']:i for i,r in enumerate(rows)}
    tokens = {r['recovery_id']:r['prompt_token_ids']+
              (r['generated_token_ids'] if mode=='resume' else []) for r in rows}
    def group(row):
        return row.get('recovery_group') or row['recovery_id']
    if policy in ('group-first','group-ratio'):
        rows = _order_prefix_bundles(rows,tokens,rank,shared_first=policy=='group-first')
    if policy=='interleave':
        groups = defaultdict(deque)
        for row in rows:
            groups[group(row)].append(row)
        priority = {key:(-len(values),rank[values[0]['recovery_id']]) for key,values in groups.items()}
        rows,previous = [],None
        while groups:
            candidates = [key for key in groups if key!=previous] or list(groups)
            key = min(candidates,key=priority.get)
            rows.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            previous = key
    ordered,seen = [],[]
    def cached(row):
        return max((_shared_blocks(prior,tokens[row['recovery_id']]) for prior in seen),default=0)
    while rows:
        index = min(range(len(rows)),key=lambda i:(len(tokens[rows[i]['recovery_id']])-cached(rows[i]),
                    rank[rows[i]['recovery_id']])) if policy=='prefix-shortest' else 0
        row = rows.pop(index)
        shared = cached(row)
        row.update(queue_rank=len(ordered),shared_prefix_tokens=shared,
                   estimated_prefill_tokens=len(tokens[row['recovery_id']])-shared)
        ordered.append(row)
        seen.append(tokens[row['recovery_id']])
    return ordered
