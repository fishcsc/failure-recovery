"""普通 offload 的私有锁页内存后端，由 metrics.transfer_config 指定给 vLLM。

入口 PrivateCPUOffloadingSpec；只支持独立 TP=1，不承担故障检查点备份。
"""
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


class PrivateCPUOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config):
        if config.parallel.world_size != 1 or config.canonical_layout:
            raise ValueError('Private CPU KV storage requires TP=1 and non-canonical layout')
        super().__init__(config)

    def _uses_shared_region(self):
        # 由原生 create_worker() 创建私有锁页 CPU 张量，避免占用 /dev/shm。
        return False
