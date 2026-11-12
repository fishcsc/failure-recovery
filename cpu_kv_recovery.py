"""旧采样故障入口的 CPU KV 实现，供 failure_sampling 历史回归使用。

SharedKVPool 由父进程持有，SourceKVBackup 备份完整块，KVTransfer 导入原生 APC。
当前六文件故障实验使用 failure_recovery/cpu_kv.py，不经过本文件。
"""
from collections import deque
import ctypes
import hashlib
import json
import mmap
from multiprocessing.reduction import DupFd
import os
import time


class SharedKVPool:
    """父进程持有的匿名 RAM，源进程退出后备份仍在，不受 /dev/shm 配额限制。"""
    def __init__(self, gigabytes):
        self.size = int(gigabytes * 1_000_000_000)
        with open('/proc/meminfo') as stream:
            available = next(int(line.split()[1])*1024 for line in stream if line.startswith('MemAvailable:'))
        if self.size <= 0 or self.size > available-8*1024**3:
            raise ValueError('CPU KV budget must be positive and leave 8 GiB host RAM available')
        self.fd = os.memfd_create('failure_cpu_kv', os.MFD_CLOEXEC)
        self.metadata = {}  # 父进程同时持有请求元数据，源退出后仍可核对备份进度。
        try:
            os.ftruncate(self.fd, self.size)
        except BaseException:
            self.close()
            raise

    def share(self):
        return dict(fd=DupFd(self.fd), size=self.size)

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
    """旧版有限槽位索引：复制完成才可见，PENDING 不可用于恢复。"""
    def __init__(self, capacity):
        self.capacity, self.entries, self.free = capacity, {}, []
        self.next_slot = self.peak = self.misses = 0

    def reserve(self, key):
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
        self.peak = max(self.peak, len(self.entries))
        return slot

    def complete(self, keys):
        for key in keys:
            self.entries[key][1] = True

    def retain(self, active_keys):
        for key, (slot, ready) in list(self.entries.items()):
            if ready and key not in active_keys:
                del self.entries[key]
                self.free.append(slot)

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


class KVTransfer:
    """旧版真实 KV 页视图、共享 CPU 页注册和 DMA，保留历史恢复语义。"""
    REGISTER_CHUNK = 64 * 1024**2

    def __init__(self, engine, scheduler, handle):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, group_kernel_blocks
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

        self.torch, self.scheduler = torch, scheduler
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
        self.size = handle['size']
        if self.size < self.block_bytes:
            os.close(self.fd)
            raise ValueError('CPU KV budget cannot hold one full KV block')
        self.memory = mmap.mmap(self.fd, self.size)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.memory))
        self.registered, self.held = set(), []
        self.replay_refs = {}
        self.cudart = CudaRTLibrary()
        self.stream = torch.cuda.Stream()

    def pin_slots(self, slots):
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

    def copy(self, block_ids, slots, to_cpu):
        from vllm import _custom_ops
        torch = self.torch
        self.pin_slots(slots)
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
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            begin.record()
            _custom_ops.swap_blocks_batch(*descriptors,is_src_access_order_any=not to_cpu)
            end.record()
        return dict(begin=begin,end=end,descriptors=descriptors,bytes=len(slots)*self.block_bytes)

    def restore(self, request, checkpoint):
        """原生 Scheduler 运行前先把 CPU 字节加载并发布到目标 APC。"""
        if checkpoint['signature'] != self.signature:
            raise RuntimeError('Source/target KV tensor layouts differ')
        slots = checkpoint['slots'][:(request.num_tokens-1)//16]
        manager = self.scheduler.kv_cache_manager
        hits, tokens, _ = manager.get_computed_blocks(request)
        local = min(tokens//16, len(slots))
        if local == len(slots):
            return dict(cpu_loaded_tokens=0,cpu_load_bytes=0,cpu_load_ms=0.,cpu_available_tokens=len(slots)*16)
        pool = manager.block_pool
        prefix = list(hits.blocks[0][:local])
        pool.touch(prefix)
        self.held.extend(prefix)
        reserve = max(16, min(pool.num_gpu_blocks//8, self.scheduler.max_num_running_reqs+512))
        count = min(len(slots)-local, max(0,pool.get_num_free_blocks()-reserve))
        if not count:
            return dict(cpu_loaded_tokens=0,cpu_load_bytes=0,cpu_load_ms=0.,cpu_available_tokens=len(slots)*16)
        new = pool.get_new_blocks(count)
        self.held.extend(new)
        job = self.copy([block.block_id for block in new], slots[local:local+count],False)
        job['end'].synchronize()
        # H2D 完成后才发布，Hash 取自目标请求，供原生 APC 识别。
        pool.cache_full_blocks(request,prefix+new,local,local+count,16,0)
        return dict(cpu_loaded_tokens=count*16,cpu_load_bytes=job['bytes'],
                    cpu_load_ms=job['begin'].elapsed_time(job['end']),cpu_available_tokens=len(slots)*16)

    def release_imports(self):
        if self.held:
            self.scheduler.kv_cache_manager.block_pool.free_blocks(self.held)
            self.held = []

    def replay_operations(self, operations):
        """重放复制期间的 GPU 引用增加/释放顺序，使旧原生分配路径可对照。"""
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
        self.stream.synchronize()
        self.release_imports()
        pool = self.scheduler.kv_cache_manager.block_pool
        pool.free_blocks([pool.blocks[bid] for bid,n in self.replay_refs.items() for _ in range(n)])
        self.replay_refs.clear()
        for chunk in self.registered:
            result = self.cudart.cudaHostUnregister(self.base+chunk*self.REGISTER_CHUNK)
            if result:
                raise RuntimeError(f'CPU KV cudaHostUnregister failed: {result}')
        self.memory.close()
        os.close(self.fd)


class SourceKVBackup:
    """旧版完整块增量备份，持有 GPU 引用保护在途 DMA。"""
    def __init__(self, transfer):
        self.transfer = transfer
        self.index = CheckpointIndex(transfer.size//transfer.block_bytes)
        self.jobs = deque()
        self.submitted_bytes = self.completed_bytes = self.copy_ms = self.submit_ms = 0.
        self.frozen = False
        self.operations = []
        self.published, self.dirty = {}, True

    @staticmethod
    def request_keys(request):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        return [_key(make_block_hash_with_group_id(h,0))
                for h in request.block_hashes[:(request.num_tokens-1)//16]]

    def updates(self, ids):
        """故障前把变化的 READY 前缀元数据发布给父进程。"""
        scheduler = self.transfer.scheduler
        result = []
        for rid,request in scheduler.requests.items():
            if not self.dirty and rid in self.published:
                continue
            slots = self.index.prefix(self.request_keys(request))
            if self.published.get(rid) != slots:
                result.append(dict(request_id=ids[rid],slots=slots,signature=self.transfer.signature))
                self.published[rid] = slots
        self.published = {rid: slots for rid,slots in self.published.items() if rid in scheduler.requests}
        self.dirty = False
        return result

    def take_operations(self):
        operations, self.operations = self.operations, []
        return operations

    def poll(self):
        while self.jobs and self.jobs[0]['end'].query():
            job = self.jobs.popleft()
            self.index.complete(job['keys'])
            self.dirty = True
            self.completed_bytes += job['bytes']
            self.copy_ms += job['begin'].elapsed_time(job['end'])
            self.transfer.scheduler.kv_cache_manager.block_pool.free_blocks(job['blocks'])
            self.operations.append(['free',[b.block_id for b in job['blocks']]])

    def submit(self):
        if self.frozen:
            return
        begin = time.perf_counter()
        scheduler = self.transfer.scheduler
        # GPU 原生 Hash 带 KV 组号，源选块和目标识别必须使用相同键。
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        active = {_key(make_block_hash_with_group_id(h,0)) for request in scheduler.requests.values()
                  for h in request.block_hashes}
        self.index.retain(active)
        blocks, keys, slots = [], [], []
        for request in scheduler.running:
            available = request.num_computed_tokens//16
            for block in scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0][:available]:
                if block.is_null or block.block_hash is None:
                    continue
                key = _key(block.block_hash)
                slot = self.index.reserve(key)
                if slot is not None:
                    blocks.append(block);keys.append(key);slots.append(slot)
        if blocks:
            scheduler.kv_cache_manager.block_pool.touch(blocks)
            self.operations.append(['touch',[b.block_id for b in blocks]])
            job = self.transfer.copy([block.block_id for block in blocks],slots,True)
            job.update(blocks=blocks,keys=keys)
            self.jobs.append(job)
            self.submitted_bytes += job['bytes']
        self.submit_ms += 1000*(time.perf_counter()-begin)

    def freeze(self, request_ids):
        """故障边界仅冻结已就绪数据，不 poll、不等待、不临时补拷。"""
        self.frozen = True
        scheduler = self.transfer.scheduler
        requests = {}
        for rid in request_ids:
            request = scheduler.requests[rid]
            keys = self.request_keys(request)
            requests[rid] = dict(slots=self.index.prefix(keys),signature=self.transfer.signature)
        return dict(requests=requests,signature=self.transfer.signature,
            stats=dict(capacity_bytes=self.transfer.size,block_bytes=self.transfer.block_bytes,
                peak_bytes=self.index.peak*self.transfer.block_bytes,completed_store_bytes=self.completed_bytes,
                registered_bytes=sum(min(self.transfer.REGISTER_CHUNK,
                    self.transfer.size-chunk*self.transfer.REGISTER_CHUNK) for chunk in self.transfer.registered),
                submitted_store_bytes=self.submitted_bytes,pending_blocks=sum(len(j['keys']) for j in self.jobs),
                store_ms=self.copy_ms,submit_ms=self.submit_ms,allocation_misses=self.index.misses))

    def close(self):
        # 等待 DMA 仅为安全清理，不把后来完成的块补进冻结目录。
        for job in self.jobs:
            job['end'].synchronize()
            self.transfer.scheduler.kv_cache_manager.block_pool.free_blocks(job['blocks'])
        self.jobs.clear()

