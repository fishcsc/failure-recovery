"""故障请求的纯调度算法：先选目标实例，再排各实例的恢复队列。

experiment 在故障时调用；cpu_kv 在备份时也调用同一算法预测当前队列。
入口是 assign_recovery_requests 和 order_recovery_requests，不执行 GPU 操作。
"""
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import heapq
import pickle
import struct

KV_BLOCK_TOKENS = 16  # 与当前 EngineArgs(block_size=16) 保持一致。
_PREFIX_CACHE = ContextVar('recovery_prefix_comparisons',default=None)


@contextmanager
def cache_prefix_comparisons():
    """在一次计划计算内复用精确 LCP；嵌套调用共用缓存，退出即释放引用。"""
    if _PREFIX_CACHE.get() is not None:
        yield
        return
    token=_PREFIX_CACHE.set({})
    try:
        yield
    finally:
        _PREFIX_CACHE.reset(token)


def _prepare(checkpoints, survivors, mode):
    """把检查点转为可分配请求，并保留源 running 顺序作为稳定的并列次序。"""
    if not survivors or len(set(survivors)) != len(survivors):
        raise ValueError('Recovery needs distinct surviving instances')
    if mode not in ('restart', 'resume'):
        raise ValueError('Unknown recovery mode')
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        tokens = _input_tokens(checkpoint,mode)
        length = len(tokens) or checkpoint.get('input_length', 0)
        rows.append(dict(checkpoint, recovery_id=f"REC0_{checkpoint['request_id']}_{index:05d}",
                         prefill_tokens=length, _tokens=tokens, _index=index))
    return rows, {w: [] for w in survivors}


def _input_tokens(row,mode):
    """分配和队列排序共用同一个不可变输入对象，使同一对输入的 LCP 也可复用。"""
    prompt = row.get('prompt_token_ids',())
    generated = row.get('generated_token_ids',()) if mode=='resume' else ()
    cache = _PREFIX_CACHE.get()
    key = ('input',id(prompt),id(generated))
    if cache is not None and key in cache:
        return cache[key][2]
    tokens = tuple(prompt)+tuple(generated)
    if cache is not None:
        cache[key] = (prompt,generated,tokens)
    return tokens


def _place(row, worker, cost=None, **metadata):
    return dict({k:v for k,v in row.items() if not k.startswith('_')}, target_worker=worker,
                estimated_prefill_tokens=row['prefill_tokens'] if cost is None else cost, **metadata)


def assign_equal_count(checkpoints, survivors, mode='restart'):
    """基线：按源 running 顺序轮询均分请求数，返回 worker → 恢复请求列表。"""
    rows, assigned = _prepare(checkpoints, survivors, mode)
    for index, row in enumerate(rows):
        worker = survivors[index % len(survivors)]
        assigned[worker].append(_place(row, worker))
    return assigned


def assign_prefill_balanced(checkpoints, survivors, mode='restart'):
    """按恢复输入长度从长到短分配给新增负载最小的实例，返回分配表。

    长度优先只决定目标，分配完成后恢复源顺序，避免把长请求强制排到目标队首。
    """
    rows, assigned = _prepare(checkpoints, survivors, mode)
    # 只平衡新增恢复输入；将幸存实例原有负载都视为相同，而非实时读取 GPU 使用率。
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
    cache=_PREFIX_CACHE.get()
    key=(id(left),id(right))
    if cache is not None and key in cache:
        return cache[key][2]
    length=_scan_common_length(left,right)
    if cache is not None:
        # 缓存同时持有 token 对象，防止对象 ID 复用误命中旧比较结果。
        cache[key]=(left,right,length)
        cache[(key[1],key[0])]=(right,left,length)
    return length


def _scan_common_length(left, right):
    # 在线预测会重复比较长输入，先用切片成块比较；
    # 只有首个不同块逐 token 检查，减少 Python 扫描开销；
    # 仍保留精确公共前缀长度，最后不足一块也不能丢。
    offset=0
    limit=min(len(left),len(right))
    if type(left) is type(right) and isinstance(left,(list,tuple)):
        while offset+256<=limit and left[offset:offset+256]==right[offset:offset+256]:
            offset+=256
        end=min(offset+256,limit)
    else:
        end=limit
    for index, (a,b) in enumerate(zip(left[offset:end],right[offset:end]),offset):
        if a != b:
            return index
    return limit


def _shared_blocks(left, right):
    # 估算 left 可提供的完整前缀块；当前 right 至少重算最后一个 prompt token。
    # 短 prompt 是长 prompt 的完整前缀时，left 最后一块也可供 right 复用。
    length = min(_common_length(left,right),len(right)-1)
    return max(0,length//KV_BLOCK_TOKENS*KV_BLOCK_TOKENS)


def _group_cost(group):
    # group 按 token 字典序排列；相邻 LCP 可计算前缀树去重后的总计算量。
    # 如两条 100-token 输入共享 64 tokens，估计为 100+36；真实命中仍看原生执行。
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
    """按真实 token 前缀分组，估计组内去重 Prefill，再按组从重到轻均衡。

    输入为检查点及幸存 worker，返回 worker → 请求列表。过重的多成员组继续拆分；
    上限为 max（最长请求，当前估计总量/实例数）× split_factor，拆组后重新估算。
    此处只估计计算量，不计幸存实例原负载或 CPU 备份覆盖，不保证实际缓存命中。
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
    pending = []
    def enqueue(group):
        # 组内内容不变时成本也不变，只在初建或拆组时计算一次。
        cost = _group_cost(group)
        heapq.heappush(pending,(-cost,min(r['_index'] for r in group),group))
        return cost
    for group in buckets.values():
        enqueue(sorted(group,key=lambda r:(r['_tokens'],r['_index'])))
    pending_cost = sum(-entry[0] for entry in pending)
    loads = dict.fromkeys(survivors,0)
    largest = max((r['prefill_tokens'] for r in rows),default=0)
    group_number = 0
    while pending:
        total = sum(loads.values())+pending_cost
        limit = max(largest,total/len(survivors))*split_factor
        negative_cost,_,group = heapq.heappop(pending)
        cost = -negative_cost
        pending_cost -= cost
        worker = min(survivors,key=lambda w:(loads[w],len(assigned[w]),survivors.index(w)))
        # 组太重才拆；拆到不同 GPU 会重复计算共享前段，因此每次需重新估算上限。
        if len(group)>1 and loads[worker]+cost>limit:
            for child in _split_group(group):
                pending_cost += enqueue(child)
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


@cache_prefix_comparisons()
def assign_recovery_requests(checkpoints,survivors,strategy='equal',mode='restart',
                             min_shared_tokens=16,split_factor=1.2):
    """统一分配入口：按 strategy 返回 worker → 恢复请求列表。

    mode 决定是否把已生成 token 计入恢复输入；后续 waiting/未来流量单独转发。
    """
    if strategy=='equal':
        return assign_equal_count(checkpoints,survivors,mode)
    if strategy=='prefill-balanced':
        return assign_prefill_balanced(checkpoints,survivors,mode)
    if strategy=='kv-aware':
        return assign_kv_aware(checkpoints,survivors,mode,min_shared_tokens,split_factor)
    raise ValueError(f'Unknown recovery scheduler: {strategy}')


def _order_prefix_bundles(rows, tokens, rank, shared_first):
    """按组新增 Prefill/成员数选择下一组，估计平均首 token 收益。

    shared_first=True 强制共享组先于单请求；False 让短单请求也参与比较。
    整组仅表示队列相邻，所有请求直接入队，不等待组首完成才放行。
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('recovery_group') or row['recovery_id']].append(row)
    ordered = []
    # base 只由已确定的全局队列更新；组内试排使用副本，不污染其他候选组。
    base = {row['recovery_id']:0 for row in rows}
    estimates = {}
    def estimate(items):
        pending,sequence,total = list(items),[],0
        best = {row['recovery_id']:base[row['recovery_id']] for row in items}
        while pending:
            index = min(range(len(pending)),key=lambda i:(
                len(tokens[pending[i]['recovery_id']])-best[pending[i]['recovery_id']],
                rank[pending[i]['recovery_id']]))
            row = pending.pop(index)
            rid = row['recovery_id']
            total += len(tokens[rid])-best[rid]
            for other in pending:
                oid = other['recovery_id']
                best[oid] = max(best[oid],_shared_blocks(tokens[rid],tokens[oid]))
            sequence.append(row)
        return total,sequence
    while groups:
        candidates = [key for key,items in groups.items() if len(items)>1] if shared_first else []
        candidates = candidates or list(groups)
        for key in candidates:
            if key not in estimates:
                estimates[key] = estimate(groups[key])
        # group-ratio 比较的是预计新增计算量/组员数，不是 CPU 备份占比。
        key = min(candidates,key=lambda key:(estimates[key][0]/len(groups[key]),estimates[key][0],
                                           min(rank[r['recovery_id']] for r in groups[key])))
        sequence = estimates[key][1]
        ordered.extend(sequence)
        del groups[key]
        del estimates[key]
        for other,items in groups.items():
            changed = False
            for row in items:
                rid = row['recovery_id']
                shared = max(base[rid],max(_shared_blocks(tokens[r['recovery_id']],tokens[rid])
                                           for r in sequence))
                if shared>base[rid]:
                    base[rid] = shared
                    changed = True
            # 没有新增可复用前缀时，该组的试排结果仍然完全相同。
            if changed:
                estimates.pop(other,None)
    return ordered


@cache_prefix_comparisons()
def order_recovery_requests(items, policy='source', mode='restart'):
    """对一个目标的 items 排序并返回新列表，不改变目标分配或延迟入队。

    source 保持源顺序；prefix-shortest 逐次选择预计新增 Prefill 最小的请求；
    interleave 隔开同组成员；group-first/group-ratio 按组选择。
    best 增量保存已排输入可提供的最大共享量，代表潜在复用，不是实际 GPU 命中。
    """
    if policy not in ('source','interleave','prefix-shortest','group-first','group-ratio'):
        raise ValueError('Unknown recovery queue order')
    rows = [dict(row) for row in items]
    rank = {r['recovery_id']:i for i,r in enumerate(rows)}
    tokens = {r['recovery_id']:_input_tokens(r,mode) for r in rows}
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
    ordered = []
    best = dict.fromkeys(tokens,0)
    while rows:
        index = min(range(len(rows)),key=lambda i:(len(tokens[rows[i]['recovery_id']])-best[rows[i]['recovery_id']],
                    rank[rows[i]['recovery_id']])) if policy=='prefix-shortest' else 0
        row = rows.pop(index)
        rid = row['recovery_id']
        shared = best[rid]
        row.update(queue_rank=len(ordered),shared_prefix_tokens=shared,
                   estimated_prefill_tokens=len(tokens[row['recovery_id']])-shared)
        ordered.append(row)
        # 新确定一个队列位置，仅比较它与剩余请求；不再重扫整个已排前缀集合。
        for other in rows:
            oid = other['recovery_id']
            best[oid] = max(best[oid],_shared_blocks(tokens[rid],tokens[oid]))
    return ordered


def recovery_plan_config(survivors,strategy,mode,min_shared_tokens,split_factor,order):
    return dict(survivors=list(survivors),strategy=strategy,mode=mode,
                min_shared_tokens=min_shared_tokens,split_factor=split_factor,order=order)


def recovery_input_digest(checkpoints,mode):
    """对有序恢复输入生成稳定摘要；不包含与调度无关的 CPU 槽位和计算进度。"""
    digest = hashlib.sha256(struct.pack('<Q',len(checkpoints)))
    for row in checkpoints:
        # 固定协议编码整数元组，避免逐 token 的 Python 哈希循环。
        values = pickle.dumps(_input_tokens(row,mode),protocol=4)
        digest.update(struct.pack('<Q',len(values)))
        digest.update(values)
    return digest.hexdigest()


_PLAN_FIELDS = ('prefill_tokens','estimated_prefill_tokens','recovery_group',
                'group_position','shared_prefix_tokens','queue_rank')


def make_recovery_plan(checkpoints,queues,**config):
    """只保存输入标识和队列元数据，不保存旧检查点、旧槽位或旧 recovery_id。"""
    indices = {row['request_id']:i for i,row in enumerate(checkpoints)}
    return dict(schema_version=1,config=config,
        request_ids=[row['request_id'] for row in checkpoints],
        input_digest=recovery_input_digest(checkpoints,config['mode']),
        queues={w:[dict(cohort_index=indices[row['request_id']],
                       **{k:row[k] for k in _PLAN_FIELDS if k in row}) for row in rows]
                for w,rows in queues.items()})


def reuse_recovery_plan(plan,checkpoints,**config):
    """配置和有序输入匹配才复用；始终绑定故障时的新检查点，否则返回 None 重算。"""
    if not plan or plan.get('schema_version')!=1 or plan.get('config')!=config:
        return None
    if (plan.get('request_ids')!=[row['request_id'] for row in checkpoints]
            or plan.get('input_digest')!=recovery_input_digest(checkpoints,config['mode'])):
        return None
    queues = plan.get('queues',{})
    if set(queues)!=set(config['survivors']):
        return None
    assigned = {w:[] for w in config['survivors']}
    seen = set()
    for worker,items in assigned.items():
        for rank,metadata in enumerate(queues[worker]):
            index = metadata.get('cohort_index')
            if type(index) is not int or not 0<=index<len(checkpoints) or index in seen:
                return None
            seen.add(index)
            checkpoint = checkpoints[index]
            length = len(checkpoint.get('prompt_token_ids',()))
            if config['mode']=='resume':
                length += len(checkpoint.get('generated_token_ids',()))
            length = length or checkpoint.get('input_length',0)
            shared = metadata.get('shared_prefix_tokens')
            if (metadata.get('queue_rank')!=rank or metadata.get('prefill_tokens')!=length
                    or type(shared) is not int or not 0<=shared<=length
                    or metadata.get('estimated_prefill_tokens')!=length-shared):
                return None
            items.append(dict(checkpoint,
                recovery_id=f"REC0_{checkpoint['request_id']}_{index:05d}",target_worker=worker,
                **{k:metadata[k] for k in _PLAN_FIELDS if k in metadata}))
    return assigned if len(seen)==len(checkpoints) else None


def forwarding_assignment(waiting, future, survivors):
    """按原到达时间转发源 waiting 和未来流量，返回 worker → 转发请求列表。

    它们不是故障 running，需独立编号并走普通提交路径，避免重复恢复和重复计数。
    """
    assignments = {worker: [] for worker in survivors}
    for index, (kind, row) in enumerate(
            [('waiting', r) for r in waiting]+[('future', r) for r in future]):
        target = survivors[index % len(survivors)]
        assignments[target].append(dict(
            row, request_id=f"FWD0_{row['request_id']}_{index:05d}",
            original_request_id=row['request_id'], is_forwarded=True,
            source_queue_at_failure=kind, target_worker=target))
    return assignments
