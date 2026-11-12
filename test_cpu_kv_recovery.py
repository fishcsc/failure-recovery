"""Checkpoint durability, visibility and native-cache publication (CPU-only)."""
import contextlib
import io
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from cpu_kv_recovery import CheckpointIndex, SharedKVPool, SourceKVBackup, KVTransfer
from sample_running import parse_args
from failure_sampling import FailureCoordinator
from report import failure_markdown


def write_pool(handle):
    fd = handle['fd'].detach()
    memory = mmap.mmap(fd,handle['size'])
    memory[:9] = b'committed'
    memory.close()
    os.close(fd)


class CPUKVTests(unittest.TestCase):
    def test_pending_bytes_are_not_visible_and_prefix_stops_at_first_gap(self):
        index = CheckpointIndex(3)
        slots = [index.reserve(k) for k in ('a','b','c')]
        index.complete(['b','c'])
        self.assertEqual(index.prefix(['a','b','c']),[])
        index.complete(['a'])
        self.assertEqual(index.prefix(['a','b','c']),slots)
        self.assertEqual(index.prefix(['a','missing','c']),slots[:1])
        self.assertIsNone(index.reserve('a'))

    def test_capacity_never_evicts_live_or_inflight_blocks(self):
        index = CheckpointIndex(2)
        a,b = index.reserve('a'),index.reserve('b')
        index.complete(['a'])
        index.retain({'a'})
        self.assertIsNone(index.reserve('c'))  # b is still in flight.
        index.complete(['b'])
        index.retain({'a'})
        self.assertEqual(index.reserve('c'),b)
        self.assertEqual(index.prefix(['a']),[a])
        self.assertEqual(index.misses,1)

    def test_parent_ram_survives_source_process_exit(self):
        with SharedKVPool(.001) as pool:
            pool.metadata['r'] = dict(slots=[0])
            process = mp.get_context('spawn').Process(target=write_pool,args=(pool.share(),))
            process.start();process.join(15)
            self.assertEqual(process.exitcode,0)
            with mmap.mmap(pool.fd,pool.size) as memory:
                self.assertEqual(memory[:9],b'committed')
        self.assertIsNone(pool.fd)
        self.assertEqual(pool.metadata,{})

    def test_fault_freeze_never_flushes_or_publishes_pending_dma(self):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        keys = [make_block_hash_with_group_id(k,0).hex() for k in (b'a',b'b')]
        blocks = [SimpleNamespace(block_hash=k) for k in (b'a',b'b')]
        manager = MagicMock()
        manager.get_blocks.return_value.blocks = [blocks]
        transfer = SimpleNamespace(size=128,block_bytes=16,signature='layout',registered=set(),REGISTER_CHUNK=64,
            scheduler=SimpleNamespace(requests={'r':SimpleNamespace(num_tokens=33,block_hashes=[b'a',b'b'])},kv_cache_manager=manager))
        backup = SourceKVBackup(transfer)
        a,b = backup.index.reserve(keys[0]),backup.index.reserve(keys[1])
        backup.index.complete(keys[:1])
        event = MagicMock()
        backup.jobs.append(dict(end=event,keys=keys[1:],blocks=blocks[1:]))
        checkpoint = backup.freeze(['r'])
        self.assertEqual(checkpoint['requests']['r']['slots'],[a])
        event.query.assert_not_called()
        event.synchronize.assert_not_called()
        backup.close()
        event.synchronize.assert_called_once()
        self.assertEqual(backup.index.prefix(keys),[a])

    def test_restore_publishes_real_bytes_before_native_lookup_without_faking_progress(self):
        transfer = KVTransfer.__new__(KVTransfer)
        transfer.signature = 'same';transfer.held=[]
        request = SimpleNamespace(num_tokens=65,num_computed_tokens=0)
        pool = MagicMock(num_gpu_blocks=128)
        new = [SimpleNamespace(block_id=i) for i in range(3)]
        pool.get_new_blocks.return_value = new
        pool.get_num_free_blocks.return_value = 100
        manager = SimpleNamespace(block_pool=pool,get_computed_blocks=lambda r:(SimpleNamespace(blocks=[[]]),0,0))
        transfer.scheduler = SimpleNamespace(kv_cache_manager=manager,max_num_running_reqs=4)
        event = MagicMock();ready=[]
        event.synchronize.side_effect = lambda: ready.append(True)
        begin = MagicMock();begin.elapsed_time.return_value = 2.
        transfer.copy = MagicMock(return_value=dict(end=event,begin=begin,bytes=123))
        pool.cache_full_blocks.side_effect = lambda *args: self.assertEqual(ready,[True])
        result = transfer.restore(request,dict(signature='same',slots=[10,11,12]))
        self.assertEqual(result['cpu_loaded_tokens'],48)
        self.assertEqual(request.num_computed_tokens,0)
        pool.cache_full_blocks.assert_called_once_with(request,new,0,3,16,0)
        transfer.release_imports()
        pool.free_blocks.assert_called_once_with(new)

    def test_restore_rejects_layout_mismatch_before_transfer(self):
        transfer = KVTransfer.__new__(KVTransfer);transfer.signature='target'
        with self.assertRaisesRegex(RuntimeError,'layouts differ'):
            transfer.restore(SimpleNamespace(),dict(signature='source',slots=[0]))

    def test_published_progress_and_slot_metadata_are_owned_by_parent(self):
        args = parse_args(['--inject-failure','--cpu-kv-backup-gb','.001'])
        metadata = {}
        with tempfile.TemporaryDirectory() as directory:
            coordinator = FailureCoordinator(args,[],Path(directory),0,cpu_metadata=metadata)
            for update in [dict(request_id='r',prompt_token_ids=[3,4],output_length=4),
                           dict(request_id='r',new_tokens=[5,6]),
                           dict(request_id='r',slots=[7],signature='layout')]:
                coordinator.handle(dict(kind='cpu_kv_metadata',worker=0,updates=[update]))
            del coordinator
            self.assertEqual(metadata['r']['generated_token_ids'],[5,6])
            self.assertEqual(metadata['r']['slots'],[7])
            replay = FailureCoordinator(args,[],Path(directory),0,cpu_metadata=metadata)
            self.assertIs(replay.cpu_metadata,metadata)
            with self.assertRaisesRegex(RuntimeError,'publisher'):
                replay.handle(dict(kind='cpu_kv_metadata',worker=1,updates=[]))
            replay.handle(dict(kind='cpu_kv_metadata',worker=0,
                               updates=[dict(request_id='r',completed=True)]))
            self.assertNotIn('r',metadata)

    def test_cpu_backup_defaults_to_resume_and_keeps_old_restart_default(self):
        self.assertEqual(parse_args([]).recovery_mode,'restart')
        args = parse_args(['--inject-failure','--cpu-kv-backup-gb','.5'])
        self.assertEqual(args.recovery_mode,'resume')
        self.assertEqual(args.latency_measurement,'resume_cpu_kv_next_token_v1')
        self.assertEqual(args.cpu_cache_gb,0)
        for flags in [[],['--inject-failure','--recovery-mode','restart']]:
            with self.subTest(flags=flags),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['--cpu-kv-backup-gb','.5']+flags)

    def test_resume_reports_do_not_use_historical_catchup_definition(self):
        summary = dict(requests=0,completed=0,latency_measurement='resume_next_token_v1')
        text = failure_markdown('after_recovery.md',dict(summary=summary))
        self.assertIn('下一个新token',text)
        self.assertNotIn('历史口径',text)
        summary.update(latency_measurement='resume_cpu_kv_next_token_v1',cpu_kv=dict(
            load_bytes=1024,load_work_ms=2,backup_tokens=32,loaded_tokens=16,mean_prefill_to_first_tokens=4))
        text = failure_markdown('after_recovery.md',dict(summary=summary))
        self.assertIn('CPU KV 加载与尾部补算',text)
        self.assertNotIn('历史口径',text)
        self.assertEqual(parse_args(['--recovery-mode','resume']).latency_measurement,'resume_next_token_v1')


if __name__=='__main__':
    unittest.main()
