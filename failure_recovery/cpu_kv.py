"""在线 CPU KV 备份和故障导入；父进程持有 RAM，源退出后字节仍可访问。

engine 调用 SourceKVBackup 的 submit/poll/freeze，目标用 KVTransfer.restore_cohort 加载。
以不可变的完整 16-token 块为单位，经真实 D2H/H2D 与原生 APC 恢复，
不伪造计算进度；当前只支持 TP=1、一个完整注意力 KV 组。
"""
from collections import deque
from contextlib import contextmanager
import ctypes
from functools import wraps
import hashlib
import json
import mmap
from multiprocessing.reduction import DupFd
import os
import time


class WallProfile:
    """分阶段累计 CPU 墙钟时间；嵌套阶段去重计时，CUDA Event 时间另计。"""
    def __init__(self, enabled=False, start=None):
        self.enabled=enabled
        self.start=time.perf_counter() if start is None else start
        self.end=None
        self.stack=[]
        self.exclusive={}
        self.inclusive={}
        self.calls={}
        self.counters={}

    @contextmanager
    def stage(self, name):
        if not self.enabled or self.end is not None:
            yield
            return
        frame=[name,time.perf_counter(),0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed=time.perf_counter()-frame[1]
            self.stack.pop()
            self.inclusive[name]=self.inclusive.get(name,0.)+elapsed
            self.exclusive[name]=self.exclusive.get(name,0.)+elapsed-frame[2]
            self.calls[name]=self.calls.get(name,0)+1
            if self.stack:
                self.stack[-1][2]+=elapsed

    def add(self, name, value):
        if self.enabled and self.end is None:
            self.counters[name]=self.counters.get(name,0)+value

    def stop(self):
        if self.end is None:
            self.end=time.perf_counter()

    def snapshot(self):
        """返回毫秒计时快照，扣除子阶段避免把同一段耗时重复相加。"""
        now=time.perf_counter() if self.end is None else self.end
        exclusive=dict(self.exclusive)
        inclusive=dict(self.inclusive)
        for i,(name,start,children) in enumerate(self.stack):
            elapsed=now-start
            active_child=now-self.stack[i+1][1] if i+1<len(self.stack) else 0.
            exclusive[name]=exclusive.get(name,0.)+elapsed-children-active_child
            inclusive[name]=inclusive.get(name,0.)+elapsed
        wall=1000*(now-self.start)
        amounts={k:1000*v for k,v in exclusive.items()}
        return dict(wall_ms=wall,exclusive_ms=amounts,inclusive_ms={k:1000*v for k,v in inclusive.items()},
            unattributed_ms=wall-sum(amounts.values()),calls=dict(self.calls),counters=dict(self.counters))


def profile_phase(name):
    """给关键方法添加同名耗时阶段，不改变其输入、返回值或异常。"""
    def decorate(function):
        @wraps(function)
        def measured(self,*args,**kwargs):
            profiler=getattr(self,'profiler',None)
            if profiler is None:
                return function(self,*args,**kwargs)
            with profiler.stage(name):
                return function(self,*args,**kwargs)
        return measured
    return decorate


class SharedKVPool:
    """父进程持有的匿名 RAM 文件描述符，容量使用十进制 GB，不受 /dev/shm 配额限制。"""
    def __init__(self, gigabytes, size_bytes=None):
        self.size = int(gigabytes * 1_000_000_000) if size_bytes is None else int(size_bytes)
        with open('/proc/meminfo') as stream:
            available = next(int(line.split()[1])*1024 for line in stream if line.startswith('MemAvailable:'))
        if self.size <= 0 or self.size > available-8*1024**3:
            raise ValueError('CPU KV budget must be positive and leave 8 GiB host RAM available')
        self.fd = os.memfd_create('failure_cpu_kv', os.MFD_CLOEXEC)
        self.epoch = os.urandom(12).hex()
        self.metadata = {}  # 父进程同时持有请求元数据，源退出后仍可核对备份进度。
        try:
            os.ftruncate(self.fd, self.size)
        except BaseException:
            self.close()
            raise

    def share(self):
        """向子进程传可复制的 FD、容量和池标识；不复制整份备份数据。"""
        return dict(fd=DupFd(self.fd), size=self.size, epoch=self.epoch)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.metadata.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CheckpointIndex:
    """全局 Hash → CPU 槽位/READY 状态的索引，PENDING 也占容量。

    相同 Hash 只占一槽；槽位复用递增 generation，避免旧引用误指向新字节。
    """
    def __init__(self, capacity):
        self.capacity, self.entries, self.free = capacity, {}, []
        self.next_slot = self.peak = self.misses = 0
        self.generations = {}
        self.evictions = 0

    def reserve(self, key):
        """为尚未备份的 Hash 预留 PENDING 槽位；重复 Hash 或容量不足返回 None。"""
        if key in self.entries:
            return None
        if self.free:
            slot = self.free.pop()
        elif self.next_slot < self.capacity:
            slot = self.next_slot
            self.next_slot += 1
        else:
            self.misses += 1
            return None
        self.entries[key] = [slot, False]
        # CPU 物理槽位会复用；代际变化让父进程能识别同槽位上的新内容。
        self.generations[slot] = self.generations.get(slot,0)+1
        self.peak = max(self.peak, len(self.entries))
        return slot

    def complete(self, keys):
        """仅在对应 D2H Event 已完成后，把这些 Hash 标为 READY。"""
        for key in keys:
            self.entries[key][1] = True

    def retain(self, active_keys):
        """回收已 READY 且不再需要的槽位，返回被移除的 Hash。

        PENDING 仍有 DMA 写入，不能提前回收供下一次复制使用。
        """
        evicted=[]
        for key, (slot, ready) in list(self.entries.items()):
            if ready and key not in active_keys:
                evicted.append(key)
                del self.entries[key]
                self.free.append(slot)
                self.evictions+=1
        return evicted

    def ref(self, key):
        """只返回 READY 槽位的代际引用；PENDING 不对恢复方可见。"""
        entry=self.entries.get(key)
        if entry is None or not entry[1]:
            return None
        return dict(cpu_slot=entry[0],generation=self.generations[entry[0]],block_key=key)

    def directory(self):
        return {str(ref['cpu_slot']):ref for key in self.entries if (ref:=self.ref(key)) is not None}

    def prefix(self, keys):
        slots = []
        for key in keys:
            entry = self.entries.get(key)
            if entry is None or not entry[1]:
                break
            slots.append(entry[0])
        return slots


def _key(value):
    return value.hex() if isinstance(value, bytes) else repr(value)


def apply_manifest_delta(entry, delta):
    """父进程按版本应用目录增量，原地更新 entry，不重建整份列表。

    先检查池/布局、版本连续和区间完整，再修改位置映射，防止部分应用或旧引用混入。
    """
    version=entry.get('version',0)
    initial=delta.get('initial',False)
    if (delta['base_version']!=version or (initial and version)
            or delta['version']<1 or (not initial and delta['version']!=version+1)):
        raise RuntimeError('Non-contiguous CPU manifest version')
    if version and (entry.get('epoch')!=delta['epoch'] or entry.get('signature')!=delta['signature']):
        raise RuntimeError('CPU manifest epoch/layout changed')
    current=entry.get('blocks_by_position',{})
    removed=set(delta['remove'])
    added={ref['logical_block']:ref for ref in delta['add']}
    start,end=delta['start_block'],delta['end_block']
    old_start,old_end=entry.get('start_block',0),entry.get('end_block',0)
    if (len(removed)!=len(delta['remove']) or len(added)!=len(delta['add']) or start<0 or end<start
            or not removed<=current.keys()):
        raise RuntimeError('Invalid CPU manifest removals/range')
    outside=old_end-old_start-max(0,min(old_end,end)-max(old_start,start))
    if sum(not start<=p<end for p in removed)!=outside:
        raise RuntimeError('CPU manifest did not revoke the old interval')
    if any(start<=p<end and p not in added for p in removed):
        raise RuntimeError('CPU manifest has an interior hole')
    for position,ref in added.items():
        if (not start<=position<end or (position in current and position not in removed)
                or ref['generation']<1 or ref['cpu_slot']<0 or not ref['block_key']):
            raise RuntimeError('Invalid CPU manifest addition')
    if len(current)-len(removed)+len(added)!=end-start:
        raise RuntimeError('CPU manifest interval is not contiguous')
    for position in removed:
        del current[position]
    current.update(added)
    entry.update(blocks_by_position=current,version=delta['version'],epoch=delta['epoch'],
        signature=delta['signature'],start_block=start,end_block=end,
        unpublished_ready_blocks=delta['unpublished_ready_blocks'])


def materialize_manifest(entry):
    """故障校验/恢复时才把位置字典转为有序 blocks；slots 仅表示从零开始的前缀。"""
    blocks=sorted(entry['blocks_by_position'].values(),key=lambda ref:ref['logical_block'])
    return dict(blocks=blocks,slots=[r['cpu_slot'] for r in blocks] if blocks and blocks[0]['logical_block']==0 else [],
        **{k:entry[k] for k in ('version','epoch','signature','unpublished_ready_blocks')})


def select_request_fifo(candidates, capacity, request_order, selected=None, owners=None):
    """按 request_order 逐请求补齐完整前缀，返回 selected 和 owners。

    selected 按 Hash 去重计容量，owners 记录各请求覆盖位置；容量满后仍可引用已选共享块。
    遇到缺口即停止该请求，不能跨过缺块拼出不可用的前缀。
    """
    ranks={rid:index for index,rid in enumerate(request_order)}
    if any(row[0] not in ranks for row in candidates):
        raise ValueError('FIFO backup order must include every candidate request')
    selected={} if selected is None else selected
    owners={} if owners is None else owners
    blocked=set()
    for rid,position,_,key,block in sorted(candidates,key=lambda c:(ranks[c[0]],c[1])):
        if rid in blocked:
            continue
        positions=owners.setdefault(rid,set())
        if position in positions:
            continue
        if position!=len(positions) or (key not in selected and len(selected)>=capacity):
            blocked.add(rid)
            continue
        selected.setdefault(key,block)
        positions.add(position)
    return selected,owners


def select_group_prefix_queue(candidates, capacity, request_order, groups):
    """先覆盖各共享组的公共前缀，再按预测目标队列补齐请求，返回选块和位置表。

    先按前缀深度轮转各组，避免一个长前缀独占预算；单成员组在第二阶段竞争。
    只用当前 running 的已计算块，不预知未来故障现场，Hash 全局去重。
    """
    by_request={}
    for row in candidates:
        by_request.setdefault(row[0],{})[row[1]]=row
    heads=[]
    for members in groups.values():
        if len(members)<2:
            continue
        available=[by_request.get(rid,{}) for rid in members]
        for position in range(min(map(len,available))):
            rows=[blocks.get(position) for blocks in available]
            if any(row is None for row in rows) or len({row[3] for row in rows})!=1:
                break
            heads.append((position,len(heads),rows))
    selected,owners={},{}
    for position,_,rows in sorted(heads,key=lambda h:h[:2]):
        key=rows[0][3]
        if key not in selected and len(selected)>=capacity:
            continue
        if any(len(owners.get(row[0],()))!=position for row in rows):
            continue
        selected.setdefault(key,rows[0][4])
        for rid,_,_,_,_ in rows:
            owners.setdefault(rid,set()).add(position)
    return select_request_fifo(candidates,capacity,request_order,selected,owners)


def predict_backup_queues(requests, survivors, min_shared_tokens=16, split_factor=1.2,
                          mode='resume', return_plan=False):
    """用当前 running 的 prompt/已输出 token 模拟 kv-aware/group-ratio。

    返回跨目标按队列位置轮转的请求顺序和前缀组；return_plan 额外返回可复用计划。
    不真的迁移请求，也不查看未来流量；恢复输入模式与故障后配置一致。
    """
    from .scheduling import (assign_recovery_requests,order_recovery_requests,cache_prefix_comparisons,
                             make_recovery_plan,recovery_plan_config)
    checkpoints=[dict(request_id=r.request_id,prompt_token_ids=list(r.prompt_token_ids),
                      generated_token_ids=list(r.output_token_ids)) for r in requests]
    with cache_prefix_comparisons():
        assigned=assign_recovery_requests(checkpoints,survivors,'kv-aware',mode,
                                          min_shared_tokens,split_factor)
        queues={worker:order_recovery_requests(rows,'group-ratio',mode)
                for worker,rows in assigned.items()}
        if return_plan:
            config=recovery_plan_config(survivors,'kv-aware',mode,min_shared_tokens,split_factor,'group-ratio')
            plan=make_recovery_plan(checkpoints,queues,**config)
    order=[queues[w][position]['request_id']
           for position in range(max(map(len,queues.values()),default=0))
           for w in survivors if position<len(queues[w])]
    groups={}
    for rows in queues.values():
        for row in rows:
            groups.setdefault(row['recovery_group'],[]).append(row['request_id'])
    if return_plan:
        return order,groups,plan
    return order,groups


def select_blocks(candidates, capacity, policy, request_order=None, groups=None):
    """按 policy 和物理槽位预算选择块，返回 Hash → GPU 块及请求 → 位置集合。

    candidates 每项为（请求、逻辑块号、可用块数、Hash、GPU 块）；
    前缀/后缀从相应一端扩展，遇到预算缺口停止该请求，始终只发布连续区间。
    """
    if policy in ('request-fifo','group-prefix-queue'):
        order=request_order if request_order is not None else list(dict.fromkeys(c[0] for c in candidates))
        if policy=='group-prefix-queue':
            if groups is None:
                raise ValueError('Group-prefix backup needs predicted live groups')
            return select_group_prefix_queue(candidates,capacity,order,groups)
        return select_request_fifo(candidates,capacity,order)
    if policy not in ('prefix-first','suffix-first'):
        raise ValueError('Unknown selective backup policy')
    ranked=sorted(candidates,key=lambda c:((c[1] if policy=='prefix-first' else c[2]-1-c[1]),c[0],c[3]))
    selected,owners,blocked={},{},set()
    for rid,position,end,key,block in ranked:
        if rid in blocked:
            continue
        positions=owners.setdefault(rid,set())
        expected=len(positions) if policy=='prefix-first' else end-1-len(positions)
        if position!=expected:
            blocked.add(rid)
            continue
        if key not in selected and len(selected)>=capacity:
            blocked.add(rid)
            continue
        selected.setdefault(key,block)
        positions.add(position)
    return selected,owners


class KVTransfer:
    """管理真实 GPU KV 视图、共享 CPU 映射和复制流，供源备份和目标导入使用。"""
    REGISTER_CHUNK = 64 * 1024**2

    def __init__(self, engine, scheduler, handle, profiler=None):
        """从引擎取得各层真实 KV 物理页，建立 CPU 槽位布局和复制流。

        handle 来自父进程共享池；signature 绑定模型、精度和张量布局，避免跨布局误加载。
        """
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, group_kernel_blocks
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        self.torch, self.scheduler = torch, scheduler
        self.profiler=profiler or WallProfile()
        config = scheduler.kv_cache_config
        if len(config.kv_cache_groups) != 1:
            raise ValueError('CPU KV recovery currently requires a single full-attention KV group')
        group = config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        if not isinstance(spec, FullAttentionSpec) or spec.block_size != 16:
            raise ValueError('CPU KV recovery requires full attention and 16-token blocks')
        runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        context = runner.compilation_config.static_forward_context
        self.views, self.layout, self.offsets = [], [], []
        # 一槽存同一逻辑块在所有层的 KV；offsets 是各层在该 CPU 槽内的字节偏移。
        self.block_bytes = 0
        for name in sorted(group.layer_names):
            tensor = group_kernel_blocks(context[name].kv_cache, config.num_blocks)
            stride = tensor.stride(0)*tensor.element_size()
            page = spec.page_size_bytes
            if stride != page:
                raise ValueError('Packed/aliased KV layouts are not supported by CPU KV recovery')
            self.views.append(tensor)
            self.offsets.append(self.block_bytes)
            self.block_bytes += page
            self.layout.append([name, page, str(tensor.dtype), list(tensor.shape[1:]), list(tensor.stride()[1:])])
        self.signature = hashlib.sha256(json.dumps([
            engine.model_config.model, str(engine.model_config.dtype), self.layout]).encode()).hexdigest()
        self.fd = handle['fd'].detach()
        self.epoch = handle.get('epoch')
        self.size = handle['size']
        if self.size < self.block_bytes:
            os.close(self.fd)
            raise ValueError('CPU KV budget cannot hold one full KV block')
        self.memory = mmap.mmap(self.fd, self.size)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.memory))
        self.registered = set()
        self.imports = {}  # 临时持有每个请求的导入块，直到原生块表正式接管。
        self.replay_refs = {}
        # pending 只表示尚未完成拼接的计划，不等于源端 D2H 的 PENDING 槽位。
        self.pending = {}
        self.cudart = CudaRTLibrary()
        self.stream = torch.cuda.Stream()

    @profile_phase('host_registration')
    def pin_slots(self, slots):
        """按较大内存片注册 CPU 页，避免每个小 KV 块单独做锁页注册。"""
        for slot in slots:
            begin, end = slot*self.block_bytes, (slot+1)*self.block_bytes
            for chunk in range(begin//self.REGISTER_CHUNK, (end-1)//self.REGISTER_CHUNK+1):
                if chunk in self.registered:
                    continue
                offset = chunk*self.REGISTER_CHUNK
                result = self.cudart.cudaHostRegister(self.base+offset, min(self.REGISTER_CHUNK,self.size-offset))
                if result:
                    raise RuntimeError(f'CPU KV cudaHostRegister failed: {result}')
                self.registered.add(chunk)
                self.profiler.add('host_registered_bytes',min(self.REGISTER_CHUNK,self.size-offset))

    @profile_phase('dma_enqueue')
    def copy(self, block_ids, slots, to_cpu):
        """按配对的 GPU 块号和 CPU 槽位异步复制所有层，返回 Event 和描述符。

        to_cpu=True 为 D2H，否则 H2D；复制流先等待当前计算流，调用本身不等待完成。
        调用方必须保留描述符及相关页引用，直到 end Event 完成。
        """
        from vllm import _custom_ops
        torch = self.torch
        self.pin_slots(slots)
        with self.profiler.stage('dma_descriptors'):
            src, dst, sizes = [], [], []
            for tensor, offset, layout in zip(self.views, self.offsets, self.layout):
                page = layout[1]
                for bid, slot in zip(block_ids, slots):
                    gpu = tensor.data_ptr()+bid*tensor.stride(0)*tensor.element_size()
                    cpu = self.base+slot*self.block_bytes+offset
                    src.append(gpu if to_cpu else cpu)
                    dst.append(cpu if to_cpu else gpu)
                    sizes.append(page)
            descriptors = [torch.tensor(values,dtype=torch.int64,pin_memory=True) for values in (src,dst,sizes)]
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 先等已有前向写完页，再在独立复制流搬运；完整块不会被后续 Decode 继续写。
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            begin.record()
            _custom_ops.swap_blocks_batch(*descriptors,is_src_access_order_any=not to_cpu)
            end.record()
        return dict(begin=begin,end=end,descriptors=descriptors,bytes=len(slots)*self.block_bytes)

    def restore(self, request, checkpoint, record=None):
        """单请求兼容入口，返回该请求的导入统计；正式恢复使用 restore_cohort。"""
        record=record if record is not None else {}
        self.restore_cohort([(request,checkpoint,record)])
        return record

    def _prepare_restore(self, request, checkpoint, record, reserved):
        """为一个请求校验范围、复用已有块并预留缺失页，返回计划/新块/CPU 槽位。

        reserved 让同批尚未发布到 APC 的共享块也只分配一次；此阶段不启动 H2D。
        """
        if checkpoint['signature']!=self.signature:
            raise RuntimeError('Source/target KV tensor layouts differ')
        self.validate_manifest(request,checkpoint)
        # 至少留最后一个输入位置供真实前向产生 logits；完整上下文命中也不能省掉这一步。
        refs=checkpoint.get('blocks',[
            dict(logical_block=i,cpu_slot=slot) for i,slot in enumerate(checkpoint['slots'])])
        refs=[ref for ref in refs if ref['logical_block']<(request.num_tokens-1)//16]
        start=refs[0]['logical_block'] if refs else 0
        end=refs[-1]['logical_block']+1 if refs else 0
        manager=self.scheduler.kv_cache_manager
        pool=manager.block_pool
        blocks={}
        for ref in refs:
            i=ref['logical_block']
            key=request.block_hashes[i]
            hit=pool.get_cached_block(key,[0])
            block=hit[0] if hit else reserved.get(key)
            if block is not None:
                blocks[i]=block
        pinned=list(blocks.values())
        pool.touch(pinned)
        self.imports[request.request_id]=list(pinned)
        missing=[ref for ref in refs if ref['logical_block'] not in blocks]
        # 导入占页之外还要留重算空间，否则 KV 全加载完也可能无法安排下一轮计算。
        reserve=max(16,min(pool.num_gpu_blocks//8,self.scheduler.max_num_running_reqs+512),
                    (request.num_tokens+15)//16+manager.watermark_blocks)
        evicted=self.make_import_room(len(missing),reserve)
        new=pool.get_new_blocks(len(missing)) if missing else []
        self.imports[request.request_id].extend(new)
        for ref,block in zip(missing,new):
            i=ref['logical_block']
            blocks[i]=block
            reserved[request.block_hashes[i]]=block
        logical=[pool.null_block]*start+[blocks[i] for i in range(start,end)]
        # 后缀前的 null 只保持逻辑块号，不代表已经拥有前段 KV；此时不能作为完整块表推理。
        ranges=[]
        for ref in missing:
            i=ref['logical_block']
            if ranges and ranges[-1][1]==i:
                ranges[-1][1]=i+1
            else:
                ranges.append([i,i+1])
        record.update(cpu_loaded_tokens=len(new)*16,cpu_load_bytes=len(new)*self.block_bytes,
            cpu_load_ms=None,cpu_available_tokens=len(refs)*16,
            gpu_reused_selected_tokens=len(pinned)*16,gpu_evicted_running_requests=evicted,
            load_begin_token=start*16,load_end_token=end*16,preload_complete=False,
            native_apc_tokens_after_import=None,native_apc_tokens_after_join=None,
            recovery_requeues=0,gpu_space_waits=0,join_wait_deferrals=0,join_ms=0.,preload_wall_ms=None)
        if record['cpu_loaded_tokens']+record['gpu_reused_selected_tokens']!=record['cpu_available_tokens']:
            raise RuntimeError('Imported KV accounting does not cover the selected CPU interval')
        plan=dict(request=request,start=start*16,end=end*16,blocks=logical[start:],
                  logical=logical,ranges=ranges,metrics=record)
        # 即使区间从 0 开始，也先阻止整批加载完成前提前输出；
        # 单独的后缀不能充当完整注意力需要的连续上下文。
        self.pending[request.request_id]=plan
        return plan,new,[ref['cpu_slot'] for ref in missing]

    def restore_cohort(self, items, mode='serial', step=None):
        """将 items 中整批所选 KV 去重分配、批量 H2D 并发布，返回分阶段耗时。

        每项为（原生请求、CPU 检查点、结果记录）；serial 等加载完成，
        suffix-overlap 在复制期间通过 step 执行真实前段重算。
        两者都要先完成全部导入才 ACK，此阶段不允许拼接后提前输出新 token。
        """
        if mode not in ('serial','suffix-overlap') or (mode=='suffix-overlap' and step is None):
            raise ValueError('suffix-overlap requires a native inference step callback')
        started=time.perf_counter()
        plans=[]
        ids=[]
        slots=[]
        reserved={}
        self.hold_joins=True
        self.restore_job=None
        try:
            for request,checkpoint,record in items:
                plan,new,positions=self._prepare_restore(request,checkpoint,record,reserved)
                plans.append(plan)
                ids.extend(block.block_id for block in new)
                slots.extend(positions)
            prepared=time.perf_counter()
            origin=self.torch.cuda.Event(enable_timing=True) if mode=='suffix-overlap' else None
            if origin:
                origin.record()
            job=self.copy(ids,slots,False) if ids else None
            self.restore_job=job  # 异常时也保留锁页描述符，直到复制流安全结束。
            head_events=[]
            head_wall=0.
            head_tokens=0
            head_steps=0
            if job and mode=='serial':
                job['end'].synchronize()
            elif job:
                while not job['end'].query():
                    begin=self.torch.cuda.Event(enable_timing=True)
                    end=self.torch.cuda.Event(enable_timing=True)
                    begin.record()
                    before=time.perf_counter()
                    observed=step()
                    end.record()
                    if observed.get('head_tokens',0):
                        head_events.append((begin,end))
                        head_wall+=1000*(time.perf_counter()-before)
                        head_tokens+=observed['head_tokens']
                        head_steps+=1
                    if not observed.get('scheduled_tokens',0):
                        time.sleep(.001)
                # 前段前向也必须真的执行完成，不能仅根据调度器
                # 提前推进的计数就拼接仍在计算中的 KV。
                if head_events:
                    head_events[-1][1].synchronize()
            copied=time.perf_counter()
            pool=self.scheduler.kv_cache_manager.block_pool
            # 字节确实到达后才登记 Hash；后缀仍需补齐前段，APC 才能识别为连续前缀。
            for plan in plans:
                request=plan['request']
                record=plan['metrics']
                for begin,end in plan['ranges']:
                    pool.cache_full_blocks(request,plan['logical'],begin,end,16,0)
                cached=self.scheduler.kv_cache_manager.get_computed_blocks(request)[1]
                record.update(native_apc_tokens_after_import=cached,preload_complete=True,
                    restore_protocol='eager-preload',kv_restore_mode=mode)
                if not (plan['start'] and (cached<plan['end'] or request in self.scheduler.running)):
                    del self.pending[request.request_id]
            copy_ms=job['begin'].elapsed_time(job['end']) if job else 0.
            overlap=forward=0.
            if job and head_events:
                copy_begin=origin.elapsed_time(job['begin'])
                copy_end=origin.elapsed_time(job['end'])
                for begin,end in head_events:
                    a=origin.elapsed_time(begin)
                    b=origin.elapsed_time(end)
                    forward+=b-a
                    overlap+=max(0.,min(b,copy_end)-max(a,copy_begin))
            elapsed=1000*(time.perf_counter()-started)
            # 整批只测得一个 GPU 复制时长，不能人为拆成每请求 DMA 时间；
            # 只有单请求兼容路径才填写该请求的独立复制耗时。
            if len(plans)==1:
                plans[0]['metrics'].update(cpu_load_ms=copy_ms,preload_wall_ms=elapsed)
            self.preload_timing=dict(mode=mode,prepare_wall_ms=1000*(prepared-started),
                loading_and_heads_wall_ms=1000*(copied-prepared),total_wall_ms=elapsed,
                h2d_gpu_ms=copy_ms,load_bytes=len(slots)*self.block_bytes,
                head_forward_wall_ms=head_wall,head_forward_cuda_envelope_ms=forward,
                copy_forward_envelope_overlap_ms=overlap,head_steps=head_steps,head_tokens=head_tokens)
            self.restore_job=None
            return self.preload_timing
        finally:
            self.hold_joins=False

    def register_demand(self, items, mode, clock_start):
        """只登记组和 CPU 区间，不分配 GPU 页或搬运；收到 resume 后才按队列触发。

        组内一次去重加载，组外继续使用原生 APC。serial 后缀先算前段再加载；
        suffix-overlap 在该组首次得到调度时发起 H2D，同时允许真实前段计算。
        """
        self.demand_mode=mode
        self.demand_clock_start=clock_start
        self.demand_groups={}
        self.demand_members={}
        self.demand_deferred={}
        self.demand_head_events=[]
        self.demand_origin=self.torch.cuda.Event(enable_timing=True)
        self.demand_origin.record()
        self.demand_timing=dict(mode=mode,load_policy='demand',prepare_wall_ms=0.,
            loading_and_heads_wall_ms=0.,total_wall_ms=0.,h2d_gpu_ms=0.,load_bytes=0,
            head_forward_wall_ms=0.,head_forward_cuda_envelope_ms=0.,
            copy_forward_envelope_overlap_ms=0.,head_steps=0,head_tokens=0,groups=[])
        for request,checkpoint,record in items:
            self.validate_manifest(request,checkpoint)
            if checkpoint['signature']!=self.signature:
                raise RuntimeError('Source/target KV tensor layouts differ')
            refs=[ref for ref in checkpoint.get('blocks',[
                dict(logical_block=i,cpu_slot=slot) for i,slot in enumerate(checkpoint['slots'])])
                if ref['logical_block']<(request.num_tokens-1)//16]
            start=refs[0]['logical_block']*16 if refs else 0
            end=(refs[-1]['logical_block']+1)*16 if refs else 0
            key=record.get('recovery_group') or request.request_id
            group=self.demand_groups.setdefault(key,dict(key=key,state='registered',items=[]))
            group['items'].append((request,checkpoint,record))
            self.demand_members[request.request_id]=key
            record.update(cpu_loaded_tokens=0,cpu_load_bytes=0,cpu_load_ms=None,
                cpu_available_tokens=len(refs)*16,gpu_reused_selected_tokens=0,
                gpu_evicted_running_requests=0,load_begin_token=start,load_end_token=end,
                preload_complete=False,native_apc_tokens_after_import=None,
                native_apc_tokens_after_join=None,recovery_requeues=0,gpu_space_waits=0,
                join_wait_deferrals=0,join_ms=0.,preload_wall_ms=None,
                restore_protocol='on-demand',kv_restore_mode=mode,kv_load_policy='demand',
                load_started_s=None,preload_complete_s=None)
            self.pending[request.request_id]=dict(request=request,checkpoint=checkpoint,
                start=start,end=end,metrics=record,demand_group=key)
        return self.demand_timing

    def load_demand(self, request, inside_schedule=False):
        """轮到本组时搬运；调度器内部空间不足则延后到本次前向结束再抢占腾页。

        原生 Scheduler 可能已经为 running 分配了本批工作，不能中途抢占这些请求。
        后缀 serial 只在前段已经真实完成的 advance_restores 中调用此函数。
        """
        if not hasattr(self,'demand_members') or request.request_id not in self.demand_members:
            return
        self.poll_demand()
        group=self.demand_groups[self.demand_members[request.request_id]]
        if group['state']!='registered':
            return
        plan=self.pending[request.request_id]
        if self.demand_mode=='serial' and plan['start']:
            cached=self.scheduler.kv_cache_manager.get_computed_blocks(request)[1]
            if max(request.num_computed_tokens,cached)<plan['start']:
                return
        if inside_schedule:
            pool=self.scheduler.kv_cache_manager.block_pool
            needed=set()
            cached_free=set()
            reserve=0
            for req,checkpoint,_ in group['items']:
                reserve=max(reserve,16,min(pool.num_gpu_blocks//8,
                    self.scheduler.max_num_running_reqs+512),
                    (req.num_tokens+15)//16+self.scheduler.kv_cache_manager.watermark_blocks)
                for ref in checkpoint.get('blocks',[]):
                    i=ref['logical_block']
                    if i>=(req.num_tokens-1)//16:
                        continue
                    key=req.block_hashes[i]
                    hit=pool.get_cached_block(key,[0])
                    if hit:
                        if hit[0].ref_cnt==0:
                            cached_free.add(hit[0].block_id)
                    else:
                        needed.add(key)
            if pool.get_num_free_blocks()<len(needed)+len(cached_free)+reserve:
                self.demand_deferred[group['key']]=None
                return
        self.demand_deferred.pop(group['key'],None)
        group['started']=time.perf_counter()
        plans=[]
        ids=[]
        slots=[]
        reserved={}
        for req,checkpoint,record in group['items']:
            plan,new,positions=self._prepare_restore(req,checkpoint,record,reserved)
            plan['demand_group']=group['key']
            plans.append(plan)
            ids.extend(block.block_id for block in new)
            slots.extend(positions)
            record.update(restore_protocol='on-demand',kv_load_policy='demand',
                kv_restore_mode=self.demand_mode,
                load_started_s=group['started']-self.demand_clock_start)
        group.update(plans=plans,prepared=time.perf_counter(),state='loading',
            job=self.copy(ids,slots,False) if ids else None)
        # serial 只等待当前组，不等待后面的组；并行只发起复制，下一轮非阻塞发布 READY。
        if group['job'] and self.demand_mode=='serial':
            group['job']['end'].synchronize()
        self.poll_demand()

    def poll_demand(self):
        """只发布已完成 H2D 的组；Event 和页引用在复制结束前一直存活。"""
        for group in getattr(self,'demand_groups',{}).values():
            if group['state']!='loading':
                continue
            job=group['job']
            if job and not job['end'].query():
                continue
            now=time.perf_counter()
            for plan in group['plans']:
                request=plan['request']
                for begin,end in plan['ranges']:
                    self.scheduler.kv_cache_manager.block_pool.cache_full_blocks(
                        request,plan['logical'],begin,end,16,0)
                cached=self.scheduler.kv_cache_manager.get_computed_blocks(request)[1]
                plan['metrics'].update(preload_complete=True,native_apc_tokens_after_import=cached,
                    preload_complete_s=now-self.demand_clock_start)
                if not plan['start']:
                    self.pending.pop(request.request_id,None)
            group.update(state='ready',finished=now)
            if job:
                job.pop('descriptors',None)
            timing=self.demand_timing
            timing['prepare_wall_ms']+=1000*(group['prepared']-group['started'])
            timing['loading_and_heads_wall_ms']+=1000*(now-group['prepared'])
            timing['total_wall_ms']+=1000*(now-group['started'])
            timing['h2d_gpu_ms']+=job['begin'].elapsed_time(job['end']) if job else 0.
            timing['load_bytes']+=job['bytes'] if job else 0

    def demand_step_timing(self, begin, end, wall_ms, head_tokens):
        """保存含前段重算的真实 Step 事件，结束时统一计算与各组 H2D 的交集。"""
        if head_tokens:
            self.demand_head_events.append((begin,end))
            self.demand_timing['head_forward_wall_ms']+=wall_ms
            self.demand_timing['head_steps']+=1
            self.demand_timing['head_tokens']+=head_tokens

    def demand_summary(self):
        """恢复结束后汇总事件；组区间可重叠，累计量不冒充恢复关键路径墙钟。"""
        self.stream.synchronize()
        self.torch.cuda.synchronize()
        self.poll_demand()
        origin=self.demand_origin
        spans=[(origin.elapsed_time(a),origin.elapsed_time(b)) for a,b in self.demand_head_events]
        self.demand_timing['head_forward_cuda_envelope_ms']=sum(b-a for a,b in spans)
        overlap=0.
        rows=[]
        for group in self.demand_groups.values():
            if group['state']!='ready':
                continue
            job=group['job']
            common=0.
            if job:
                lo,hi=origin.elapsed_time(job['begin']),origin.elapsed_time(job['end'])
                common=sum(max(0.,min(b,hi)-max(a,lo)) for a,b in spans)
            overlap+=common
            rows.append(dict(group=group['key'],requests=len(group['items']),
                started_s=group['started']-self.demand_clock_start,
                finished_s=group['finished']-self.demand_clock_start,
                load_bytes=job['bytes'] if job else 0,
                h2d_gpu_ms=job['begin'].elapsed_time(job['end']) if job else 0.,
                copy_forward_envelope_overlap_ms=common))
        self.demand_timing.update(groups=rows,copy_forward_envelope_overlap_ms=overlap)
        return self.demand_timing

    def make_import_room(self, count, reserve):
        """给所选块数 count 和重算余量 reserve 留空间，返回普通请求抢占数。

        无引用缓存可由原生分配淘汰；仍不足才抢占普通 running，保护本批恢复引用。
        整批放不下就报错，不静默减少加载量或改为延迟加载。
        """
        pool=self.scheduler.kv_cache_manager.block_pool
        evicted=0
        for victim in reversed(list(self.scheduler.running)):
            if pool.get_num_free_blocks()>=count+reserve:
                break
            if victim.request_id in self.imports:
                continue
            self.scheduler.running.remove(victim)
            self.scheduler._preempt_request(victim,time.monotonic())
            evicted+=1
        if pool.get_num_free_blocks()<count+reserve:
            raise RuntimeError('Selected cohort KV plus recompute headroom cannot fit on target GPU; '
                               'eager preload never truncates or switches to delayed loading')
        return evicted

    def validate_manifest(self, request, checkpoint):
        """加载前校验池标识、逻辑位置、Hash 和槽位范围，只接受一段连续区间。"""
        if checkpoint.get('epoch') is not None and checkpoint['epoch']!=self.epoch:
            raise RuntimeError('CPU manifest belongs to another backup pool')
        if 'blocks' not in checkpoint:
            return
        keys=SourceKVBackup.request_keys(request)
        positions=[]
        for ref in checkpoint['blocks']:
            position=ref['logical_block']
            positions.append(position)
            if (position<0 or position>=len(keys) or keys[position]!=ref['block_key']
                    or ref['generation']<1 or not 0<=ref['cpu_slot']<self.size//self.block_bytes):
                raise RuntimeError('CPU manifest hash/position/slot/generation mismatch')
        if positions!=sorted(set(positions)):
            raise RuntimeError('CPU manifest positions are duplicated or unordered')
        if positions and positions!=list(range(positions[0],positions[-1]+1)):
            raise RuntimeError('This phase supports one contiguous backup interval per request')

    def clip_prefill(self, request, computed, count):
        """把原生计算量裁到已加载后缀起点，避免重算覆盖后缀或在拼接前提前生成。"""
        plan=self.pending.get(request.request_id)
        if plan and plan.get('demand_group') in getattr(self,'demand_deferred',{}):
            return 0
        return min(count,max(0,plan['start']-computed)) if plan else count

    def waiting_for_join(self, request):
        """判断该 waiter 是否已到后缀拼接边界，会导致本轮调度量为零。

        同一轮前面请求可能刚安排共享前段；先绕过此 waiter，不能据此提前认定前向已完成。
        """
        plan=self.pending.get(request.request_id)
        if plan is None:
            return False
        if plan.get('demand_group') in getattr(self,'demand_deferred',{}):
            return True
        if 'demand_group' in plan and not plan['start'] and not plan['metrics']['preload_complete']:
            return False  # 等轮到该请求时加载前缀，不能当作已到拼接边界而永久跳过。
        if request.num_computed_tokens>=plan['start']:
            return True
        cached=self.scheduler.kv_cache_manager.get_computed_blocks(request)[1]
        return cached>=plan['start']

    def advance_restores(self):
        """调度前把真实已算完的前段与已加载后缀通过原生 APC 接起来。

        这里只同步并登记/校验块，不再次 H2D；必要时原生抢占后重入队，让块表重新采用连续链，
        不直接修改 num_computed_tokens 来跳过计算。
        """
        if getattr(self,'hold_joins',False):
            return
        self.poll_demand()
        scheduler=self.scheduler
        manager=scheduler.kv_cache_manager
        pool=manager.block_pool
        for rid,plan in list(self.pending.items()):
            request=plan['request']
            if 'demand_group' in plan:
                if not plan['start']:
                    continue
                cached=manager.get_computed_blocks(request)[1]
                if max(request.num_computed_tokens,cached)<plan['start']:
                    continue
                self.load_demand(request)
                plan=self.pending.get(rid)
                if plan is None or not plan['metrics']['preload_complete']:
                    continue
            start=plan['start']//16
            end=plan['end']//16
            hits,cached,_=manager.get_computed_blocks(request)
            computed=request.num_computed_tokens
            if computed<plan['start'] and cached<plan['start']:
                continue
            # 调度计数可能先于内核完成；拼接前必须确认前段 KV 已真实写好。
            started=time.perf_counter()
            self.torch.cuda.synchronize()
            if cached>=plan['start']:
                local=min(cached//16,end)
                prefix=list(hits.blocks[0][:local])
            else:
                local=start
                prefix=list(manager.get_blocks(rid).blocks[0][:start])
            if len(prefix)!=local:
                raise RuntimeError('Computed prefix blocks are missing at suffix join')
            pool.touch(prefix)
            self.imports[rid].extend(prefix)
            # 前段字节已真实算完，只补登记原生管理器尚未发布的完整块；
            # 不通过改写计算计数伪造已经恢复的进度。
            expected=SourceKVBackup.request_keys(request)
            for index,block in enumerate(prefix):
                boundary=(index+1)*16
                if block.block_hash is None or (block.block_hash_num_tokens or 0)<boundary:
                    pool.cache_full_blocks(request,prefix,index,index+1,16,0)
                elif _key(block.block_hash)!=expected[index]:
                    raise RuntimeError('Computed prefix hash mismatch at suffix join')
            # 后缀在整个前段重算期间持有引用，防止被原生分配淘汰。
            for i,block in enumerate(plan['blocks'],start):
                if _key(block.block_hash)!=expected[i] or block.ref_cnt<1:
                    raise RuntimeError('Preloaded suffix was evicted or changed before join')
            actual=manager.get_computed_blocks(request)[1]
            if actual<plan['end']:
                raise RuntimeError('Native APC cannot see the joined suffix')
            metrics=plan['metrics']
            metrics.update(native_apc_tokens_after_join=actual,
                join_ms=metrics['join_ms']+1000*(time.perf_counter()-started))
            if request in scheduler.running:
                # 用原生抢占/重入队重建块表；已完成的前段和导入后缀仍由 imports 保护。
                scheduler.running.remove(request)
                scheduler._preempt_request(request,time.monotonic())
                metrics['recovery_requeues']+=1
            del self.pending[rid]

    def release_imports(self):
        # waiting 尚未通过原生块表接管导入块，不能提前释放；
        # 后缀拼接后的重新入队也遵循同一接管规则。
        """原生块表接管后才释放临时导入引用；未入队采用或尚待拼接的页继续受保护。"""
        manager=self.scheduler.kv_cache_manager
        for rid in list(self.imports):
            if rid not in self.pending and manager.get_blocks(rid).blocks[0]:
                manager.block_pool.free_blocks(self.imports.pop(rid))

    def replay_operations(self, operations):
        """旧对照路径重放 GPU 引用增加/释放顺序；推理独立备份对照不使用它。"""
        pool = self.scheduler.kv_cache_manager.block_pool
        for kind, ids in operations:
            blocks = [pool.blocks[bid] for bid in ids]
            if kind == 'touch':
                pool.touch(blocks)
                for bid in ids:
                    self.replay_refs[bid] = self.replay_refs.get(bid,0)+1
            elif kind == 'free':
                pool.free_blocks(blocks)
                for bid in ids:
                    self.replay_refs[bid] -= 1
                    if not self.replay_refs[bid]:
                        del self.replay_refs[bid]
            else:
                raise ValueError('Unknown checkpoint GPU reference operation')

    def close(self):
        """先等复制流结束，再释放页引用、解除 CPU 锁页注册、关闭映射和 FD。"""
        self.stream.synchronize()
        self.restore_job=None
        self.release_imports()
        pool = self.scheduler.kv_cache_manager.block_pool
        for blocks in self.imports.values():
            pool.free_blocks(blocks)
        self.imports.clear()
        self.pending.clear()
        pool.free_blocks([pool.blocks[bid] for bid,n in self.replay_refs.items() for _ in range(n)])
        self.replay_refs.clear()
        for chunk in self.registered:
            result = self.cudart.cudaHostUnregister(self.base+chunk*self.REGISTER_CHUNK)
            if result:
                raise RuntimeError(f'CPU KV cudaHostUnregister failed: {result}')
        self.memory.close()
        os.close(self.fd)


class SourceKVBackup:
    """源实例的增量备份状态：选择完整块、提交 D2H、发布 READY 目录并冻结。

    临时持有 GPU 引用防止 DMA 期间页被复用，真正容量由 CheckpointIndex 控制。
    """
    def __init__(self, transfer, policy='full', manifest_mode='delta', debug_oracle=False, queue_config=None):
        self.transfer = transfer
        self.profiler=getattr(transfer,'profiler',None) or WallProfile()
        self.index = CheckpointIndex(transfer.size//transfer.block_bytes)
        # jobs 保留复制 Event、描述符和额外 GPU 引用；结束事件完成前三者都不能丢。
        self.jobs = deque()
        self.submitted_bytes = self.completed_bytes = self.copy_ms = self.submit_ms = 0.
        self.frozen = False
        self.operations = []
        self.published, self.dirty = {}, True
        self.policy=policy
        self.queue_config=queue_config
        if policy=='group-prefix-queue' and (not queue_config or not queue_config.get('survivors')):
            raise ValueError('Group-prefix backup requires surviving instance IDs')
        self.prediction_calls=0
        self.prediction_ms=0.
        self.recovery_plan=None
        self.published_plan_version=0
        self.selected,self.owners={},{}
        # selected: Hash→物理 GPU 块；owners: 请求→逻辑位置集合。共享块只计一份容量。
        self.versions={}
        self.selection_calls=self.deferred_additions=0
        self.events=deque(maxlen=5)
        self.publish_invalidations=None
        self.replay_copy_wait_ms=0.
        self.manifest_mode,self.debug_oracle=manifest_mode,debug_oracle
        # 下面的反向索引只加速目录发布，不缓存候选块收集或改变选块；
        # 避免目录优化同时改变备份策略，影响实验可比性。
        self.locations,self.dirty_positions={},{}
        # locations: Hash→引用它的请求/位置；dirty_positions 只通知真正受变化影响的位置。
        self.oracle={}
        self.manifest_freeze_ms=0.
        self.allocation_reclaim=dict(checks=0,released_blocks=0,freed_gpu_blocks=0,retries=0,retry_successes=0)

    def manifest(self, request, rid=None):
        """生成一个请求当前可恢复的连续 READY 区间，返回带布局/版本的块引用。

        前缀从 0 开始，后缀只取末端连续段；存在但无法连成该区间的 READY 块另行计数。
        """
        rid=request.request_id if rid is None else rid
        keys=self.request_keys(request)
        allowed=range(len(keys)) if self.policy=='full' else self.owners.get(rid,())
        blocks=[dict(logical_block=i,**ref) for i in sorted(allowed)
                if i<len(keys) and (ref:=self.index.ref(keys[i])) is not None]
        omitted=0
        if self.policy=='suffix-first' and blocks:
            # 在线更新也只发布最右侧连续 READY 区间；
            # 其他共享 READY 块仍占物理容量，但不能跨缺口假装可恢复。
            start=len(blocks)-1
            while start and blocks[start-1]['logical_block']+1==blocks[start]['logical_block']:
                start-=1
            omitted=start
            blocks=blocks[start:]
        elif blocks:
            length=0
            while length<len(blocks) and blocks[length]['logical_block']==length:
                length+=1
            omitted=len(blocks)-length
            blocks=blocks[:length]
        slots=[b['cpu_slot'] for b in blocks] if blocks and blocks[0]['logical_block']==0 else []
        return dict(slots=slots,blocks=blocks,signature=self.transfer.signature,
            epoch=getattr(self.transfer,'epoch',None),version=self.versions.get(rid,0),
            unpublished_ready_blocks=omitted)

    @staticmethod
    def request_keys(request):
        """生成带 KV 组号的原生 Hash，最多覆盖还需生成下一 token 之前的完整块。"""
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        return [_key(make_block_hash_with_group_id(h,0))
                for h in request.block_hashes[:(request.num_tokens-1)//16]]

    def legacy_updates(self, ids):
        """旧发布控制：重建请求目录，返回发生变化的完整 manifest。"""
        scheduler = self.transfer.scheduler
        result = []
        for rid,request in scheduler.requests.items():
            if not self.dirty and rid in self.published:
                continue
            manifest=self.manifest(request,rid)
            self.profiler.add('manifest_rebuilt_requests',1)
            previous=self.published.get(rid)
            if previous is not None:
                manifest['version']=previous['version']
            if previous!=manifest:
                self.versions[rid]=self.versions.get(rid,0)+1
                manifest['version']=self.versions[rid]
                result.append(dict(request_id=ids[rid],**manifest))
                self.profiler.add('manifest_full_entries',len(manifest['blocks']))
                self.published[rid]=manifest
        self.published = {rid: manifest for rid,manifest in self.published.items() if rid in scheduler.requests}
        self.dirty = False
        return result

    @profile_phase('metadata_events')
    def mark_keys(self, keys):
        """把 Hash 的 READY/撤销变化标到所有引用它的请求位置，返回是否影响已发布范围。"""
        if self.manifest_mode!='delta':
            return False
        visible=False
        for key in keys:
            for rid,position in self.locations.get(key,()):
                self.dirty_positions.setdefault(rid,set()).add(position)
                state=self.published[rid]
                visible|=(rid in self.transfer.scheduler.requests
                          and state['start_block']<=position<state['end_block'])
        return visible

    @profile_phase('metadata_events')
    def mark_owners(self, owners):
        """选择策略改变覆盖位置时，只把有变化的位置标脏，供目录增量更新。"""
        if self.manifest_mode!='delta':
            return
        for rid in self.owners.keys()|owners.keys():
            changed=self.owners.get(rid,set())^owners.get(rid,set())
            if changed:
                self.dirty_positions.setdefault(rid,set()).update(changed)

    def sync_publication_requests(self):
        """在请求加入、增长或退出时维护 Hash → 请求位置反向索引。

        READY 事件之后只通知相关位置，避免每轮重扫所有请求的整份目录。
        """
        requests=self.transfer.scheduler.requests
        for rid in self.published.keys()-requests.keys():
            state=self.published.pop(rid)
            for position,key in enumerate(state['keys']):
                self.locations[key].discard((rid,position))
                if not self.locations[key]:
                    del self.locations[key]
            self.dirty_positions.pop(rid,None)
        for rid,request in requests.items():
            state=self.published.get(rid)
            length=min(len(request.block_hashes),(request.num_tokens-1)//16)
            if state is not None and state['request'] is request and len(state['keys'])==length:
                continue
            keys=self.request_keys(request)  # 沿用原生键转换，不另外缓存候选 Hash。
            if state is None:
                state=dict(request=request,keys=[],ready={},start_block=0,end_block=0,
                    unpublished_ready_blocks=0,version=0)
                self.published[rid]=state
            old=state['keys']
            positions=self.dirty_positions.setdefault(rid,set())
            for position in range(max(len(old),len(keys))):
                before=old[position] if position<len(old) else None
                after=keys[position] if position<len(keys) else None
                if before==after:
                    continue
                if before is not None:
                    self.locations[before].discard((rid,position))
                    if not self.locations[before]:
                        del self.locations[before]
                if after is not None:
                    self.locations.setdefault(after,set()).add((rid,position))
                positions.add(position)
            state.update(request=request,keys=keys)

    def publication_delta(self, rid, positions):
        """按 positions 编辑引用，返回本请求的增删量；没有可见变化时返回 None。

        ready 可有缺口，公开范围必须连续；前缀取零起点，后缀取最右连续段。
        """
        state=self.published[rid]
        ready=state['ready']
        keys=state['keys']
        self.profiler.add('manifest_positions_checked',len(positions))
        old_start,old_end=state['start_block'],state['end_block']
        changed,missing=set(),set()
        allowed=self.owners.get(rid,())
        for position in positions:
            ref=(self.index.ref(keys[position]) if position<len(keys)
                 and (self.policy=='full' or position in allowed) else None)
            before=ready.get(position)
            if ref is not None:
                ref=dict(logical_block=position,**ref)
            if before==ref:
                continue
            changed.add(position)
            if ref is None:
                ready.pop(position,None)
                missing.add(position)
            else:
                ready[position]=ref
        if self.policy=='suffix-first' and ready:
            right=max((p for p in changed if p in ready),default=old_end-1)
            if old_end-1 in ready:
                right=max(right,old_end-1)
            elif right<old_end:
                right=max(ready)  # 最右端被移除时，重新定位实际剩余的右边界。
            end=right+1
            start=end
            old_surviving=max((p+1 for p in missing if old_start<=p<old_end),default=old_start)
            while start-1 in ready:
                start=old_surviving if start==old_end else start-1
        elif self.policy!='suffix-first':
            start=0
            end=min((p for p in missing if p<old_end),default=old_end)
            while end in ready:
                end+=1
        else:
            start=end=0
        omitted=len(ready)-(end-start)
        removed=set(range(old_start,min(old_end,start)))|set(range(max(old_start,end),old_end))
        # 新旧区间重叠部分只替换已变化位置，避免重发整段目录。
        replaced={p for p in changed if old_start<=p<old_end and start<=p<end}
        removed.update(replaced)
        added=set(range(start,min(end,old_start)))|set(range(max(start,old_end),end))|replaced
        if state['version'] and not removed and not added and omitted==state['unpublished_ready_blocks']:
            return None
        version=self.versions.get(rid,0)+1
        delta=dict(base_version=state['version'],version=version,epoch=getattr(self.transfer,'epoch',None),
            signature=self.transfer.signature,remove=sorted(removed),add=[ready[p] for p in sorted(added)],
            start_block=start,end_block=end,unpublished_ready_blocks=omitted,initial=not state['version'])
        state.update(start_block=start,end_block=end,unpublished_ready_blocks=omitted,version=version)
        self.versions[rid]=version
        self.profiler.add('manifest_delta_additions',len(added))
        self.profiler.add('manifest_delta_removals',len(removed))
        self.profiler.add('manifest_full_entries_equivalent',end-start)
        return delta

    @profile_phase('metadata_build')
    def updates(self, ids):
        """返回本轮可发布的目录增量，ids 将原生请求 ID 转回数据集 ID。

        可选 oracle 仅核对增量与旧目录结果，正式计时无需开启。
        """
        if self.manifest_mode=='legacy':
            return self.legacy_updates(ids)
        dirty=self.dirty
        self.sync_publication_requests()
        updates=[]
        for rid,request in self.transfer.scheduler.requests.items():
            positions=self.dirty_positions.pop(rid,None)
            state=self.published[rid]
            if positions is not None or not state['version']:
                self.profiler.add('manifest_edited_requests',1)
                delta=self.publication_delta(rid,positions or ())
                if delta is not None:
                    updates.append(dict(request_id=ids[rid],manifest_delta=delta))
            if self.debug_oracle:
                previous=self.oracle.get(rid)
                if dirty or previous is None:
                    expected=self.manifest(request,rid)
                    expected['version']=previous['version'] if previous else state['version']-1
                    if previous!=expected:
                        expected['version']+=1
                    self.oracle[rid]=expected
                actual=self.published_manifest(rid)
                if actual!=self.oracle[rid]:
                    differences={k:(actual.get(k),self.oracle[rid].get(k)) for k in actual
                                 if actual.get(k)!=self.oracle[rid].get(k)}
                    raise RuntimeError(f'Manifest delta oracle mismatch: {rid}: {differences}')
        self.oracle={rid:value for rid,value in self.oracle.items() if rid in self.published}
        self.dirty=False
        return updates

    def published_manifest(self, rid):
        """读取已发布区间的完整视图，供故障冻结核对，不查询 DMA 完成状态。"""
        if self.manifest_mode=='legacy':
            return self.published[rid]
        state=self.published[rid]
        return materialize_manifest(dict(state,blocks_by_position={p:state['ready'][p]
            for p in range(state['start_block'],state['end_block'])},
            signature=self.transfer.signature,epoch=getattr(self.transfer,'epoch',None)))

    def take_operations(self):
        operations, self.operations = self.operations, []
        return operations

    def take_recovery_plan(self, ids):
        """每个版本只发布一次；把原生 ID 转为父进程/故障检查点使用的公开 ID。"""
        if self.recovery_plan is None or self.recovery_plan['version']==self.published_plan_version:
            return None
        plan=dict(self.recovery_plan,request_ids=[ids[rid] for rid in self.recovery_plan['request_ids']])
        self.published_plan_version=plan['version']
        return plan

    @profile_phase('backup_poll')
    def poll(self):
        """非阻塞回收复制流队首已完成任务：标 READY、通知目录、释放临时 GPU 引用。

        任务在同一复制流上有序执行，遇到未完成队首就停止，不等待后续任务。
        """
        while self.jobs and self.jobs[0]['end'].query():
            job = self.jobs.popleft()
            self.index.complete(job['keys'])
            self.dirty = True
            self.mark_keys(job['keys'])
            self.completed_bytes += job['bytes']
            duration=job['begin'].elapsed_time(job['end'])
            self.copy_ms += duration
            self.profiler.add('d2h_gpu_ms',duration)
            self.profiler.add('d2h_completed_bytes',job['bytes'])
            self.transfer.scheduler.kv_cache_manager.block_pool.free_blocks(job['blocks'])
            self.operations.append(['free',[b.block_id for b in job['blocks']]])
        if self.policy!='full':
            retired=self.index.retain(self.selected)
            if retired:
                self.dirty=True
                self.mark_keys(retired)

    def before_replayed_step(self):
        """显式 step-fenced 控制：在正常前向前等待 D2H 并释放额外引用。

        只在故障检查之后调用，故障冻结不走此等待；默认 async 不调用它。
        """
        begin=time.perf_counter()
        for job in self.jobs:
            job['end'].synchronize()
        self.replay_copy_wait_ms+=1000*(time.perf_counter()-begin)
        self.poll()
        self.take_operations()  # 清空引用操作；此模式不拿它替代另一组的真实备份。

    @profile_phase('backup_submit')
    def submit(self):
        """真实前向结束后收集当前 running 的已计算完整块，选择并异步提交 D2H。

        先预留 CPU 槽位再保护 GPU 页；返回时备份可能仍是 PENDING，不代表故障时可恢复。
        """
        if self.frozen:
            return
        begin = time.perf_counter()
        scheduler = self.transfer.scheduler
        # GPU 原生 Hash 带 KV 组号，源选块和目标识别必须使用相同键。
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        with self.profiler.stage('candidate_scan'):
            active = {_key(make_block_hash_with_group_id(h,0)) for request in scheduler.requests.values()
                      for h in request.block_hashes}
            retired=self.index.retain(active)
            if retired:
                self.dirty=True
                visible=self.mark_keys(retired)
                if visible and self.publish_invalidations is not None:
                    self.publish_invalidations()
            blocks, keys, slots = [], [], []
            candidates=[]
            for request in scheduler.running:
                request_keys=self.request_keys(request)
                available=min(request.num_computed_tokens//16,len(request_keys))
                for position,block in enumerate(scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0][:available]):
                    if block.is_null or block.block_hash is None:
                        continue
                    key = _key(block.block_hash)
                    if self.policy!='full':
                        if key!=request_keys[position]:
                            raise RuntimeError('Candidate KV hash/position mismatch')
                        candidates.append((request.request_id,position,available,key,block))
                        continue
                    slot = self.index.reserve(key)
                    if slot is not None:
                        blocks.append(block)
                        keys.append(key)
                        slots.append(slot)
        if self.policy!='full':
            self.select_candidates(candidates,blocks,keys,slots)
        if blocks:
            # 临时引用阻止页在 D2H 中途被释放/复用；async 下也可能影响原生淘汰顺序。
            scheduler.kv_cache_manager.block_pool.touch(blocks)
            self.operations.append(['touch',[b.block_id for b in blocks]])
            job = self.transfer.copy([block.block_id for block in blocks],slots,True)
            job.update(blocks=blocks,keys=keys)
            self.jobs.append(job)
            self.submitted_bytes += job['bytes']
            self.profiler.add('d2h_submitted_bytes',job['bytes'])
        self.submit_ms += 1000*(time.perf_counter()-begin)

    @profile_phase('selection_and_eviction')
    def select_candidates(self,candidates,blocks,keys,slots):
        """按策略从 candidates 选块，更新覆盖位置并把新增拷贝追加到 blocks/keys/slots。

        复用槽位前先撤销旧目录；PENDING 无法回收时暂缓新增，不突破源 CPU 池预算。
        """
        self.selection_calls+=1
        ends={}
        for rid,position,_,_,_ in candidates:
            ends[rid]=max(ends.get(rid,0),position+1)
        candidates=[(rid,i,ends[rid],key,block) for rid,i,_,key,block in candidates]
        order,groups=None,None
        if self.policy in ('request-fifo','group-prefix-queue'):
            # FIFO 按到达顺序排，不能受抢占造成的 running 顺序变化影响；
            # 相同到达时间用登记顺序打破并列，不用请求编号排序。
            requests=list(self.transfer.scheduler.requests.values())
            requests=[r for _,r in sorted(enumerate(requests),
                key=lambda item:(getattr(item[1],'arrival_time',0),item[0]))]
            order=[r.request_id for r in requests]
            if self.policy=='group-prefix-queue':
                # 只模拟当前原生 running，包含尚无完整备份块的成员；
                # 不使用未来故障现场、waiting 或未来请求作先验。
                begin=time.perf_counter()
                with self.profiler.stage('backup_queue_prediction'):
                    order,groups,plan=predict_backup_queues(list(self.transfer.scheduler.running),
                                                           **self.queue_config,return_plan=True)
                self.prediction_calls+=1
                self.recovery_plan=dict(plan,version=self.prediction_calls)
                self.prediction_ms+=1000*(time.perf_counter()-begin)
        selected,owners=select_blocks(candidates,self.index.capacity,self.policy,order,groups)
        if owners!=self.owners:
            self.dirty=True
        self.mark_owners(owners)
        self.selected,self.owners=selected,owners
        evicted=self.index.retain(selected)
        if evicted:
            self.dirty=True
            self.mark_keys(evicted)
            # 复用槽位前先沿同一有序管道撤销旧逻辑引用；
            # 故障时还会核对池标识和槽位代际，
            # 防止把已经被新字节覆盖的旧目录发给恢复目标。
            if self.publish_invalidations is not None:
                self.publish_invalidations()
        for key,block in selected.items():
            if key in self.index.entries:
                continue
            if not self.index.free and self.index.next_slot>=self.index.capacity:
                self.deferred_additions+=1
                continue
            slot=self.index.reserve(key)
            blocks.append(block)
            keys.append(key)
            slots.append(slot)
        assert all(key in selected for key in keys)
        self.events.append(dict(selected=len(selected),new=len(keys),evicted=len(evicted),
            sample_new=keys[:2],sample_evicted=evicted[:2]))

    def freeze(self, request_ids):
        """故障时返回请求检查点和槽位目录，只使用已发布 READY 数据。

        不 poll、不等待、不紧急补拷；随后清理完成的 DMA 不会追加到该冻结现场。
        """
        started=time.perf_counter()
        self.frozen = True
        scheduler = self.transfer.scheduler
        requests = {}
        for rid in request_ids:
            request = scheduler.requests[rid]
            requests[rid] = self.manifest(request,rid)
            if self.manifest_mode=='delta' and rid in self.published:
                if requests[rid]!=self.published_manifest(rid):
                    raise RuntimeError(f'Frozen/published CPU manifest mismatch: {rid}')
        fault_slots={block['cpu_slot'] for checkpoint in requests.values() for block in checkpoint['blocks']}
        self.manifest_freeze_ms=1000*(time.perf_counter()-started)
        return dict(requests=requests,signature=self.transfer.signature,epoch=getattr(self.transfer,'epoch',None),
            slot_directory=self.index.directory(),
            stats=self.stats(len(fault_slots)))

    def stats(self, fault_unique=None):
        """只读记录容量、READY/PENDING、累计流量和耗时，不改变故障时可用备份。"""
        ready=sum(bool(value[1]) for value in self.index.entries.values())
        allocated=len(self.index.entries)
        return dict(capacity_bytes=self.transfer.size,block_bytes=self.transfer.block_bytes,
                manifest_mode=self.manifest_mode,manifest_debug_oracle=self.debug_oracle,
                manifest_freeze_ms=self.manifest_freeze_ms,
                allocation_reclaim=dict(self.allocation_reclaim),
                backup_policy=self.policy,evictions=self.index.evictions,selection_calls=self.selection_calls,
                queue_prediction_calls=self.prediction_calls,queue_prediction_ms=self.prediction_ms,
                replay_copy_wait_ms=self.replay_copy_wait_ms,
                deferred_additions=self.deferred_additions,selection_events=list(self.events),
                capacity_blocks=self.index.capacity,allocated_blocks=allocated,
                ready_unique_blocks=ready,ready_unique_bytes=ready*self.transfer.block_bytes,
                allocated_bytes=allocated*self.transfer.block_bytes,free_blocks=self.index.capacity-allocated,
                fault_running_ready_unique_blocks=fault_unique,
                peak_bytes=self.index.peak*self.transfer.block_bytes,completed_store_bytes=self.completed_bytes,
                registered_bytes=sum(min(self.transfer.REGISTER_CHUNK,
                    self.transfer.size-chunk*self.transfer.REGISTER_CHUNK) for chunk in self.transfer.registered),
                submitted_store_bytes=self.submitted_bytes,pending_blocks=sum(len(j['keys']) for j in self.jobs),
                store_ms=self.copy_ms,submit_ms=self.submit_ms,allocation_misses=self.index.misses)

    def close(self):
        # 等待 DMA 仅为安全清理，不把后来完成的块补进冻结目录。
        """仅为安全清理等待在途 D2H、释放 GPU 引用；不更新冻结目录。"""
        for job in self.jobs:
            job['end'].synchronize()
            self.transfer.scheduler.kv_cache_manager.block_pool.free_blocks(job['blocks'])
        self.jobs.clear()

