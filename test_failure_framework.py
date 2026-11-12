"""Independent failure experiment: boundaries, native queues and strict replay."""
import ast
import contextlib
import copy
import io
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from failure_recovery import engine, experiment, main, report, scheduling
from failure_recovery.cpu_kv import CheckpointIndex, KVTransfer, SharedKVPool, SourceKVBackup, WallProfile
from test_failure_sampling import Engine, Queue


class FailureFrameworkTests(unittest.TestCase):
    def demand_fixture(self, suffix=False, mode='serial'):
        transfer=KVTransfer.__new__(KVTransfer)
        transfer.signature='same';transfer.block_bytes=32;transfer.imports={};transfer.pending={}
        transfer.validate_manifest=MagicMock()
        transfer.torch=SimpleNamespace(cuda=SimpleNamespace(Event=MagicMock()))
        transfer.stream=MagicMock()
        cache={};free=[100];ready=[False]
        pool=MagicMock(num_gpu_blocks=128)
        pool.get_num_free_blocks.side_effect=lambda:free[0]
        pool.get_cached_block.side_effect=lambda key,groups:[cache[key]] if key in cache else None
        next_id=[0]
        def allocate(n):
            result=[SimpleNamespace(block_id=i,ref_cnt=1) for i in range(next_id[0],next_id[0]+n)]
            next_id[0]+=n
            return result
        pool.get_new_blocks.side_effect=allocate
        def publish(request,logical,begin,end,*args):
            for i in range(begin,end):cache[request.block_hashes[i]]=logical[i]
        pool.cache_full_blocks.side_effect=publish
        def computed(request):
            n=0
            for key in request.block_hashes:
                if key not in cache:break
                n+=1
            return None,n*16,0
        transfer.scheduler=SimpleNamespace(max_num_running_reqs=4,running=[],
            kv_cache_manager=SimpleNamespace(block_pool=pool,watermark_blocks=0,get_computed_blocks=computed))
        event=MagicMock();event.query.side_effect=lambda:ready[0]
        event.synchronize.side_effect=lambda:ready.__setitem__(0,True)
        begin=MagicMock();begin.elapsed_time.return_value=1.
        transfer.copy=MagicMock(side_effect=lambda ids,slots,to_cpu:
            dict(begin=begin,end=event,descriptors=[],bytes=len(slots)*32))
        requests=[SimpleNamespace(request_id=str(i),num_tokens=65,num_computed_tokens=0,
            block_hashes=['a','b','c','d'] if i<2 else ['x','y','z','w']) for i in range(3)]
        records=[dict(recovery_group='G0' if i<2 else 'G1') for i in range(3)]
        positions=[2,3] if suffix else [0,1]
        checkpoint=dict(signature='same',slots=[0,1],blocks=[dict(logical_block=i,cpu_slot=j)
            for j,i in enumerate(positions)])
        transfer.register_demand([(req,checkpoint,rec) for req,rec in zip(requests,records)],mode,0)
        return transfer,requests,records,pool,free,ready,event

    def test_demand_registration_never_allocates_or_copies_before_resume(self):
        transfer,requests,records,pool,*_=self.demand_fixture()
        pool.get_new_blocks.assert_not_called();transfer.copy.assert_not_called()
        self.assertFalse(transfer.waiting_for_join(requests[0]))
        self.assertTrue(all(not record['preload_complete'] for record in records))

    def test_demand_prefix_loads_only_current_group_and_deduplicates_members(self):
        transfer,requests,records,pool,*_=self.demand_fixture()
        transfer.load_demand(requests[0],inside_schedule=True)
        transfer.copy.assert_called_once()
        self.assertEqual(records[0]['cpu_load_bytes'],64)
        self.assertEqual(records[1]['cpu_load_bytes'],0)
        self.assertTrue(records[0]['preload_complete'] and records[1]['preload_complete'])
        self.assertFalse(records[2]['preload_complete'])
        self.assertNotIn('0',transfer.pending)
        transfer.load_demand(requests[1],inside_schedule=True)
        self.assertEqual(transfer.copy.call_count,1)

    def test_demand_serial_suffix_waits_for_computed_head_before_loading(self):
        transfer,requests,records,pool,*_=self.demand_fixture(suffix=True)
        transfer.load_demand(requests[0],inside_schedule=True)
        transfer.copy.assert_not_called()
        self.assertEqual(transfer.clip_prefill(requests[0],0,65),32)
        requests[0].num_computed_tokens=32
        transfer.load_demand(requests[0])
        transfer.copy.assert_called_once()
        self.assertTrue(records[0]['preload_complete'])
        self.assertIn('0',transfer.pending)  # 真实拼接仍由下一次 advance_restores 完成。

    def test_demand_overlap_does_not_publish_or_wait_for_pending_dma(self):
        transfer,requests,records,pool,free,ready,event=self.demand_fixture(True,'suffix-overlap')
        transfer.load_demand(requests[0],inside_schedule=True)
        event.synchronize.assert_not_called();pool.cache_full_blocks.assert_not_called()
        self.assertEqual(transfer.clip_prefill(requests[0],0,65),32)
        ready[0]=True;transfer.poll_demand()
        self.assertTrue(records[0]['preload_complete'])
        self.assertEqual(transfer.demand_timing['load_bytes'],64)

    def test_demand_memory_pressure_defers_preemption_until_after_current_forward(self):
        transfer,requests,records,pool,free,*_=self.demand_fixture()
        free[0]=0;transfer.load_demand(requests[0],inside_schedule=True)
        pool.get_new_blocks.assert_not_called();transfer.copy.assert_not_called()
        self.assertTrue(transfer.waiting_for_join(requests[0]))
        self.assertEqual(transfer.clip_prefill(requests[0],0,65),0)
        free[0]=100;transfer.load_demand(requests[0])
        self.assertTrue(records[0]['preload_complete'])
        self.assertFalse(transfer.demand_deferred)

    def test_two_instance_rates_and_demand_ack_release_without_preloading(self):
        args=main.parse_args(['--gpus','2','5','--source-qps','1.5','--survivor-qps','.5',
            '--survivor-trace','data/qwen_traceB_blksz_16.jsonl','--cpu-kv-backup-gb','1',
            '--kv-load-policy','demand'])
        self.assertEqual(args.instances,2);self.assertEqual(args.aggregate_offered_qps,2.)
        coordinator=experiment.FailureCoordinator.__new__(experiment.FailureCoordinator)
        coordinator.args=args;coordinator.assignments={1:[dict(recovery_id='r1')]}
        coordinator.acks={};coordinator.plan={};coordinator.start=0
        coordinator.connections=[MagicMock(),MagicMock()];coordinator.output=Path('/tmp/unused-demand-test')
        message=dict(kind='recovery_enqueued',worker=1,recovery_ids=['r1'],admitted_recovery_ids=['r1'],
            queue_ids=dict(waiting=['r1'],skipped_waiting=[]),running_recovery_ids=[],
            kv_restore_mode='serial',restore_protocol='on-demand',preload_complete=False)
        with patch.object(experiment,'write_result'):
            self.assertTrue(coordinator.handle_ack(message))
        coordinator.connections[1].send.assert_called_once_with(dict(action='resume'))

    def test_four_instance_demand_waits_for_three_plan_acks_without_loading(self):
        coordinator=experiment.FailureCoordinator.__new__(experiment.FailureCoordinator)
        coordinator.args=main.parse_args(['--gpus','2','5','6','7','--cpu-kv-backup-gb','1',
                                         '--kv-load-policy','demand'])
        coordinator.acks={};coordinator.plan={};coordinator.start=0
        coordinator.assignments={i:[dict(recovery_id=f'r{i}')] for i in (1,2,3)}
        coordinator.connections=[MagicMock() for _ in range(4)];coordinator.output=Path('/tmp/unused-demand-test')
        with patch.object(experiment,'write_result'):
            for i in (1,2,3):
                coordinator.handle_ack(dict(kind='recovery_enqueued',worker=i,recovery_ids=[f'r{i}'],
                    admitted_recovery_ids=[f'r{i}'],queue_ids=dict(waiting=[f'r{i}'],skipped_waiting=[]),
                    running_recovery_ids=[],kv_restore_mode='serial',restore_protocol='on-demand',
                    preload_complete=False))
                self.assertEqual(sum(c.send.call_count for c in coordinator.connections),3 if i==3 else 0)

    def test_restore_space_deduplicates_across_groups_on_one_gpu(self):
        rows=[dict(request_id=rid,recovery_group=group,queue_rank=i,prompt_token_ids=[7]*32,
                   generated_token_ids=[],cpu_kv=dict(blocks=[dict(block_key='shared'),dict(block_key=rid)]))
              for i,(rid,group) in enumerate((('A','G0'),('B','G1')))]
        result=report.summarize_recovery_placement({1:rows},block_bytes=100)
        self.assertEqual(result['external_backup_bytes'],300)
        self.assertEqual(result['cpu_cohort_unique_blocks'],3)
        self.assertEqual(result['cross_target_duplicate_blocks'],0)

    def test_restore_space_counts_same_hash_once_per_target_gpu(self):
        def row(rid):
            return dict(request_id=rid,recovery_group='G0',queue_rank=0,prompt_token_ids=[7]*32,
                generated_token_ids=[8],cpu_kv=dict(blocks=[dict(block_key='shared'),dict(block_key=rid)]))
        result=report.summarize_recovery_placement({1:[row('A')],2:[row('B')]},block_bytes=100)
        self.assertEqual(result['external_backup_bytes'],400)
        self.assertEqual(result['cpu_cohort_unique_bytes'],300)
        self.assertEqual(result['cross_target_duplicate_blocks'],1)
        self.assertEqual(result['groups'][0]['shared_prefix_tokens'],32)
        self.assertEqual({r['worker'] for r in result['groups'][0]['members']},{1,2})

    def test_placement_report_contains_space_and_prefix_group_members(self):
        row=dict(request_id='A',recovery_group='G0',queue_rank=2,prompt_token_ids=[7]*32,
            generated_token_ids=[],cpu_kv=dict(blocks=[dict(block_key='shared')]))
        summary=report.summarize_recovery_placement({1:[row]},block_bytes=100)
        text=report.placement_evidence(dict(placement_summary=summary,
            assignment=[dict(target_worker=1,target_gpu=6)]))
        for value in ('目标实例','跨目标','前缀组信息','G0','A(GPU6:2)','6'):
            self.assertIn(value,text)

    def hol_fixture(self):
        requests={rid:SimpleNamespace(request_id=rid,num_computed_tokens=0) for rid in ('A','B','C','normal')}
        cached={rid:0 for rid in requests}
        scheduler=SimpleNamespace(waiting=Queue(requests.values()),skipped_waiting=Queue(),
            kv_cache_manager=SimpleNamespace(get_computed_blocks=lambda r:(None,cached[r.request_id],0)))
        transfer=KVTransfer.__new__(KVTransfer);transfer.scheduler=scheduler
        transfer.pending={rid:dict(start=16,metrics={}) for rid in ('A','B','C')}
        return scheduler,transfer,requests,cached

    def test_suffix_zero_token_waiter_is_skipped_before_other_group_and_ordinary_work(self):
        for fixed in (False,True):
            with self.subTest(fixed=fixed):
                scheduler,transfer,requests,cached=self.hol_fixture();deferred=[];scheduled=[]
                try:
                    while scheduler.waiting or scheduler.skipped_waiting:
                        queue=engine.recovery_queue(scheduler,['A','B','C'],lambda:scheduler.waiting,
                            transfer if fixed else None,deferred)
                        request=queue.peek_request()
                        count=transfer.clip_prefill(request,cached[request.request_id],32)
                        if not count:break  # The native waiting-loop behavior.
                        scheduled.append(queue.pop_request().request_id)
                        if request.request_id=='A':cached['B']=16
                finally:engine.restore_deferred_recoveries(scheduler,['A','B','C'],deferred)
                self.assertEqual(scheduled,['A','C','normal'] if fixed else ['A'])
                self.assertEqual([r.request_id for r in scheduler.waiting],['B'] if fixed else ['B','C','normal'])
                self.assertIn('B',transfer.pending)  # No early join or fabricated progress.
                self.assertEqual(requests['B'].num_computed_tokens,0)
                if fixed:self.assertEqual(transfer.pending['B']['metrics']['join_wait_deferrals'],1)

    def test_all_suffix_waiters_blocked_keep_native_exit_and_restore_both_queues(self):
        scheduler,transfer,requests,cached=self.hol_fixture()
        scheduler.waiting=Queue([requests['B']]);scheduler.skipped_waiting=Queue([requests['A']])
        cached.update(A=16,B=16);deferred=[]
        queue=engine.recovery_queue(scheduler,['A','B'],lambda:scheduler.waiting,transfer,deferred)
        self.assertTrue(queue)
        self.assertEqual(transfer.clip_prefill(queue.peek_request(),16,32),0)
        self.assertEqual(len(scheduler.waiting)+len(scheduler.skipped_waiting),1)
        engine.restore_deferred_recoveries(scheduler,['A','B'],deferred)
        self.assertEqual([r.request_id for r in scheduler.skipped_waiting],['A'])
        self.assertEqual([r.request_id for r in scheduler.waiting],['B'])
        self.assertEqual(deferred,[])

    def test_native_schedule_exception_restores_temporarily_deferred_request(self):
        scheduler,transfer,requests,cached=self.hol_fixture()
        scheduler.waiting=Queue([requests['B'],requests['C']]);cached['B']=16
        transfer.advance_restores=MagicMock()
        runner=engine.FailureRunner.__new__(engine.FailureRunner)
        runner.scheduler=scheduler;runner.cpu_transfer=transfer;runner.worker=1;runner.failure_time=1.
        runner.recovery_order=['B','C'];runner.join_deferred=[];runner.profiler=WallProfile()
        runner.start=time.perf_counter()
        def native():
            selected=engine.recovery_queue(scheduler,runner.recovery_order,lambda:scheduler.waiting,
                transfer,runner.join_deferred)
            self.assertEqual(selected.peek_request().request_id,'C')
            raise RuntimeError('native failure')
        runner.original_schedule=native
        with self.assertRaisesRegex(RuntimeError,'native failure'):runner.schedule()
        self.assertEqual([r.request_id for r in scheduler.waiting],['B','C'])
        self.assertEqual(runner.join_deferred,[])

    def test_join_wait_detection_does_not_defer_ready_heads_or_ordinary_requests(self):
        scheduler,transfer,requests,cached=self.hol_fixture()
        self.assertFalse(transfer.waiting_for_join(requests['A']))
        cached['A']=16;self.assertTrue(transfer.waiting_for_join(requests['A']))
        cached['A']=0;requests['A'].num_computed_tokens=16
        self.assertTrue(transfer.waiting_for_join(requests['A']))
        cached['normal']=100;self.assertFalse(transfer.waiting_for_join(requests['normal']))

    def reclaim_fixture(self, policy='full', ready=True, active=False):
        block=SimpleNamespace(block_id=1,ref_cnt=2 if active else 1)
        free=[0]
        def release(blocks):
            for b in blocks:
                b.ref_cnt-=1
                if b.ref_cnt==0:free[0]+=1
        pool=SimpleNamespace(get_num_free_blocks=lambda:free[0],free_blocks=release)
        manager=SimpleNamespace(block_pool=pool)
        transfer=SimpleNamespace(size=1,block_bytes=1,scheduler=SimpleNamespace(
            requests={},kv_cache_manager=manager))
        backup=SourceKVBackup(transfer,policy,
            queue_config=dict(survivors=[1,2,3]) if policy=='group-prefix-queue' else None)
        backup.selected={'key':block};backup.index.reserve('key')
        event=MagicMock();event.query.return_value=ready
        event.synchronize.side_effect=AssertionError('CPU waited for DMA')
        backup.jobs.append(dict(keys=['key'],blocks=[block],bytes=1,end=event,
            begin=SimpleNamespace(elapsed_time=lambda end:1.)))
        runner=engine.FailureRunner.__new__(engine.FailureRunner)
        runner.cpu_backup=backup;runner.scheduler=SimpleNamespace(kv_cache_manager=manager)
        runner.profiler=WallProfile(True)
        return runner,backup,block,pool,event

    def test_all_policies_reclaim_completed_copy_and_retry_same_allocation_once(self):
        for policy in ('full','prefix-first','suffix-first','request-fifo','group-prefix-queue'):
            with self.subTest(policy=policy):
                runner,backup,block,pool,event=self.reclaim_fixture(policy)
                result=object();calls=[]
                def allocate(*a,**k):
                    calls.append((a,k));return result if pool.get_num_free_blocks() else None
                runner.original_allocate=allocate
                self.assertIs(runner.allocate_slots('request',1,num_lookahead_tokens=0),result)
                self.assertEqual(calls,[(('request',1),dict(num_lookahead_tokens=0))]*2)
                self.assertEqual(backup.allocation_reclaim,dict(checks=1,released_blocks=1,
                    freed_gpu_blocks=1,retries=1,retry_successes=1))
                self.assertIsNotNone(backup.index.ref('key'))
                self.assertEqual(block.ref_cnt,0);event.synchronize.assert_not_called()

    def test_pending_dma_keeps_its_reference_and_never_spins_or_retries(self):
        runner,backup,block,pool,event=self.reclaim_fixture(ready=False)
        runner.original_allocate=MagicMock(return_value=None)
        self.assertIsNone(runner.allocate_slots('request',1))
        runner.original_allocate.assert_called_once_with('request',1)
        event.query.assert_called_once();event.synchronize.assert_not_called()
        self.assertEqual(block.ref_cnt,1);self.assertEqual(len(backup.jobs),1)
        self.assertIsNone(backup.index.ref('key'));self.assertEqual(backup.allocation_reclaim['retries'],0)

    def test_reclaim_active_copy_ref_enables_future_native_victim_free(self):
        runner,backup,block,pool,event=self.reclaim_fixture(active=True)
        runner.original_allocate=MagicMock(return_value=None)
        self.assertIsNone(runner.allocate_slots('request',1))
        self.assertEqual(runner.original_allocate.call_count,2)
        self.assertEqual(block.ref_cnt,1);self.assertEqual(pool.get_num_free_blocks(),0)
        self.assertEqual(backup.allocation_reclaim['released_blocks'],1)
        self.assertEqual(backup.allocation_reclaim['freed_gpu_blocks'],0)
        self.assertIsNotNone(backup.index.ref('key'))  # Backed content remains READY.
        pool.free_blocks([block]);self.assertEqual(pool.get_num_free_blocks(),1)
        event.synchronize.assert_not_called()

    def test_successful_empty_allocation_and_frozen_backup_never_poll(self):
        runner,backup,block,pool,event=self.reclaim_fixture()
        valid_empty=[];runner.original_allocate=MagicMock(return_value=valid_empty)
        self.assertIs(runner.allocate_slots('request',1),valid_empty)
        event.query.assert_not_called()
        runner.original_allocate=MagicMock(return_value=None);backup.frozen=True
        self.assertIsNone(runner.allocate_slots('request',1));event.query.assert_not_called()
        self.assertEqual(block.ref_cnt,1)

    def test_six_modules_share_only_cpu_data_and_metrics_without_upward_imports(self):
        root = Path(main.__file__).parent
        self.assertEqual({p.stem for p in root.glob('*.py')},
                         {'main', 'experiment', 'engine', 'scheduling', 'cpu_kv', 'report'})
        forbidden = {'sample_running', 'failure_sampling', 'worker',
                     'recovery_scheduling', 'cpu_kv_recovery'}
        for path in root.glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, forbidden)
                    if node.level:
                        self.assertNotEqual(node.module, 'main')
                        if path.stem not in ('main', 'experiment'):
                            self.assertNotEqual(node.module, 'experiment')
        code = ('import sys; import failure_recovery.main; '
                'assert not ({"torch", "vllm", "sample_running", "failure_sampling"} '
                '& sys.modules.keys())')
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

    def test_all_entries_use_one_input_and_observation_implementation(self):
        import metrics
        import sample_running
        import worker
        import workload
        for namespace, names in (
                (main, ('configure_inputs', 'qa_request', 'load_trace')),
                (engine, ('make_prompt', 'request_prompt', 'workload_identity')),
                (experiment, ('worker_workload', 'validate_replicas')),
                (sample_running, ('worker_workload', 'validate_replicas', 'workload_identity')),
                (worker, ('make_prompt', 'request_prompt'))):
            for name in names:
                self.assertIs(getattr(namespace, name), getattr(workload, name))
        self.assertIs(main.load_inputs, workload.load_probe_traces)
        for namespace in (engine, report, sample_running):
            for name in ('StabilityGate', 'latency_summary'):
                self.assertIs(getattr(namespace, name), getattr(metrics, name))
        for namespace in (engine, sample_running):
            self.assertIs(namespace.running_snapshot, metrics.running_snapshot)

    def test_original_squality_configuration_and_random_fault_times(self):
        from sample_running import parse_args as old_parse
        flags = ['--gpus', '2', '3', '5', '6', '--source-qps', '1.5',
            '--survivor-qps', '.5', '--survivor-trace', 'traceB',
            '--compare-recovery-schedulers', '--comparison-cases',
            'prefill-balanced/shortest', 'kv-aware/shortest', 'kv-aware/group-first',
            'kv-aware/group-ratio', '--comparison-repeats', '3', '--warmup', '60',
            '--failure-delay', '10', '180', '--max-duration', '300', '--seed', '20261022']
        expected = old_parse(flags+['--inject-failure'])
        actual = main.parse_args(flags)
        for field in ('instance_qps', 'aggregate_offered_qps', 'comparison_cases',
                      'comparison_failure_times_s', 'recovery_mode', 'latency_measurement'):
            self.assertEqual(getattr(actual, field), getattr(expected, field))
        self.assertFalse(any(key.startswith('_') for key in vars(actual)))
        self.assertEqual(len(set(actual.comparison_failure_times_s)), 3)

    def test_resume_defaults_and_invalid_configurations(self):
        self.assertEqual(main.parse_args([]).recovery_mode, 'restart')
        self.assertEqual(main.parse_args(['--cpu-kv-backup-gb', '1']).recovery_mode, 'resume')
        for flags in (['--gpus', '2'], ['--cpu-kv-backup-gb', '1', '--recovery-mode', 'restart'],
                      ['--recovery-order', 'group-first'], ['--samples', '5']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(flags)

    def test_cpu_checkpoint_commits_only_contiguous_ready_blocks(self):
        index = CheckpointIndex(3)
        slots = [index.reserve(key) for key in ('a', 'b', 'c')]
        index.complete(['b', 'c'])
        self.assertEqual(index.prefix(['a', 'b', 'c']), [])
        index.complete(['a'])
        self.assertEqual(index.prefix(['a', 'b', 'c']), slots)

    def test_cpu_freeze_ignores_inflight_copy_even_during_cleanup(self):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        hashes = [b'a', b'b']
        keys = [make_block_hash_with_group_id(value, 0).hex() for value in hashes]
        request = SimpleNamespace(num_tokens=33, block_hashes=hashes)
        transfer = SimpleNamespace(size=128, block_bytes=16, signature='layout',
            registered=set(), REGISTER_CHUNK=64,
            scheduler=SimpleNamespace(requests={'r': request}, kv_cache_manager=MagicMock()))
        backup = SourceKVBackup(transfer)
        first = backup.index.reserve(keys[0]); backup.index.reserve(keys[1])
        backup.index.complete(keys[:1])
        event = MagicMock()
        backup.jobs.append(dict(end=event, keys=keys[1:], blocks=[]))
        checkpoint = backup.freeze(['r'])
        self.assertEqual(checkpoint['requests']['r']['slots'], [first])
        event.query.assert_not_called(); event.synchronize.assert_not_called()
        backup.close()
        self.assertEqual(checkpoint['requests']['r']['slots'], [first])
        self.assertEqual(backup.index.prefix(keys), [first])

    def test_imported_kv_is_published_only_after_real_copy_completion(self):
        transfer = KVTransfer.__new__(KVTransfer)
        transfer.signature = 'same'; transfer.held = []
        transfer.block_bytes=41
        transfer.imports={};transfer.pending={}
        request = SimpleNamespace(request_id='r',num_tokens=65, num_computed_tokens=0,
                                  block_hashes=['a','b','c','d'])
        block_pool = MagicMock(num_gpu_blocks=128)
        block_pool.get_cached_block.return_value=None
        blocks = [SimpleNamespace(block_id=i) for i in range(3)]
        block_pool.get_new_blocks.return_value = blocks
        block_pool.get_num_free_blocks.return_value = 100
        native=[]
        transfer.scheduler = SimpleNamespace(max_num_running_reqs=4,running=[],
            kv_cache_manager=SimpleNamespace(block_pool=block_pool,watermark_blocks=0,
                get_blocks=lambda _:SimpleNamespace(blocks=[native]),
                get_computed_blocks=lambda _: (SimpleNamespace(blocks=[[]]), 0, 0)))
        completed = []
        event = MagicMock(); event.synchronize.side_effect = lambda: completed.append(True)
        begin = MagicMock(); begin.elapsed_time.return_value = 2.
        transfer.copy = MagicMock(return_value=dict(end=event, begin=begin, bytes=123))
        block_pool.cache_full_blocks.side_effect = lambda *args: self.assertEqual(completed, [True])
        result = transfer.restore(request, dict(signature='same', slots=[10, 11, 12]))
        self.assertEqual(result['cpu_loaded_tokens'], 48)
        self.assertEqual(result['gpu_reused_selected_tokens'],0)
        self.assertEqual(request.num_computed_tokens, 0)
        transfer.release_imports()
        block_pool.free_blocks.assert_not_called()  # Waiting has not adopted KV.
        native.extend(blocks)
        transfer.release_imports()
        block_pool.free_blocks.assert_called_once_with(blocks)

    def test_suffix_is_copied_before_resume_and_join_never_copies_or_fakes_progress(self):
        transfer=KVTransfer.__new__(KVTransfer)
        transfer.signature='same';transfer.held=[];transfer.imports={};transfer.pending={}
        transfer.block_bytes=16
        transfer.validate_manifest=MagicMock()
        request=SimpleNamespace(request_id='r',num_tokens=65,num_computed_tokens=0,
                                block_hashes=['a','b','c','d'])
        pages=[SimpleNamespace(block_id=i,block_hash=k,ref_cnt=1,
                              block_hash_num_tokens=(i+1)*16) for i,k in enumerate(request.block_hashes)]
        published={};native=[]
        pool=MagicMock(num_gpu_blocks=128)
        pool.get_num_free_blocks.return_value=100
        pool.get_cached_block.side_effect=lambda key,groups:([published[key]] if key in published else None)
        pool.get_new_blocks.return_value=pages[2:]
        def publish(req,blocks,a,b,*args):
            for i in range(a,b):published[req.block_hashes[i]]=blocks[i]
        pool.cache_full_blocks.side_effect=publish
        def lookup(req):
            hit=[]
            for key in req.block_hashes:
                if key not in published:break
                hit.append(published[key])
            return SimpleNamespace(blocks=[hit]),len(hit)*16,0
        manager=SimpleNamespace(block_pool=pool,watermark_blocks=0,
                                get_computed_blocks=lookup,get_blocks=lambda _:SimpleNamespace(blocks=[native]))
        transfer.scheduler=SimpleNamespace(max_num_running_reqs=4,running=[],kv_cache_manager=manager,
                                           _preempt_request=MagicMock())
        transfer.torch=MagicMock()
        event=MagicMock();begin=MagicMock();begin.elapsed_time.return_value=1.
        transfer.copy=MagicMock(return_value=dict(end=event,begin=begin,bytes=32))
        info=transfer.restore(request,dict(signature='same',slots=[0,1],blocks=[
            dict(logical_block=2,cpu_slot=0),dict(logical_block=3,cpu_slot=1)]))
        event.synchronize.assert_called_once()
        self.assertTrue(info['preload_complete'])
        self.assertEqual(info['native_apc_tokens_after_import'],0)
        self.assertEqual(info['cpu_loaded_tokens'],32)
        self.assertEqual(info['gpu_reused_selected_tokens'],0)
        self.assertEqual(request.num_computed_tokens,0)
        self.assertEqual(transfer.clip_prefill(request,0,65),32)
        self.assertIs(pool.cache_full_blocks.call_args.args[1][0],pool.null_block)
        transfer.release_imports();pool.free_blocks.assert_not_called()
        self.assertIn('r',transfer.imports)
        transfer.advance_restores();self.assertIn('r',transfer.pending)
        # The head has now genuinely executed; join the existing tail via APC.
        published.update(a=pages[0],b=pages[1]);native.extend(pages[:2])
        request.num_computed_tokens=32;transfer.scheduler.running.append(request)
        with patch.object(SourceKVBackup,'request_keys',return_value=[repr(k) for k in request.block_hashes]):
            transfer.advance_restores()
        transfer.copy.assert_called_once()  # No second H2D at the join.
        self.assertEqual(info['native_apc_tokens_after_join'],64)
        self.assertEqual(request.num_computed_tokens,32)
        self.assertNotIn('r',transfer.pending)
        transfer.scheduler._preempt_request.assert_called_once()

    def test_import_accounting_with_cold_mixed_and_fully_cached_intervals(self):
        for cached in (0,1,3):
            with self.subTest(existing_blocks=cached):
                transfer=KVTransfer.__new__(KVTransfer)
                transfer.signature='same';transfer.held=[];transfer.imports={};transfer.pending={}
                transfer.block_bytes=32
                request=SimpleNamespace(request_id='r',num_tokens=65,num_computed_tokens=0,
                                        block_hashes=['a','b','c','d'])
                pages=[SimpleNamespace(block_id=i) for i in range(3)]
                pool=MagicMock(num_gpu_blocks=128)
                pool.get_num_free_blocks.return_value=100
                pool.get_cached_block.side_effect=lambda key,groups:(
                    [pages[request.block_hashes.index(key)]] if request.block_hashes.index(key)<cached else None)
                pool.get_new_blocks.return_value=pages[cached:]
                native=[]
                transfer.scheduler=SimpleNamespace(max_num_running_reqs=4,running=[],
                    kv_cache_manager=SimpleNamespace(block_pool=pool,watermark_blocks=0,
                        get_blocks=lambda _:SimpleNamespace(blocks=[native]),
                        get_computed_blocks=lambda _: (SimpleNamespace(blocks=[pages[:cached]]),cached*16,0)))
                event=MagicMock();begin=MagicMock();begin.elapsed_time.return_value=1.
                transfer.copy=MagicMock(return_value=dict(end=event,begin=begin,bytes=(3-cached)*32))
                result=transfer.restore(request,dict(signature='same',slots=[10,11,12]))
                self.assertEqual(result['cpu_loaded_tokens'],(3-cached)*16)
                self.assertEqual(result['gpu_reused_selected_tokens'],cached*16)
                self.assertEqual(result['cpu_available_tokens'],48)
                self.assertEqual(result['cpu_load_bytes'],(3-cached)*32)
                if cached==3:transfer.copy.assert_not_called()
                else:transfer.copy.assert_called_once_with(list(range(cached,3)),list(range(10+cached,13)),False)
                native.extend(pages);transfer.release_imports()
                pool.free_blocks.assert_called_once_with(pages)

    def test_eager_preload_memory_rule_preempts_only_ordinary_work_or_fails(self):
        transfer=KVTransfer.__new__(KVTransfer);transfer.imports={'recover':[]}
        free=[0];ordinary=SimpleNamespace(request_id='ordinary');recover=SimpleNamespace(request_id='recover')
        pool=SimpleNamespace(get_num_free_blocks=lambda:free[0])
        transfer.scheduler=SimpleNamespace(running=[ordinary,recover],
            kv_cache_manager=SimpleNamespace(block_pool=pool),
            _preempt_request=lambda req,now:free.__setitem__(0,50))
        self.assertEqual(transfer.make_import_room(20,16),1)
        self.assertEqual(transfer.scheduler.running,[recover])
        with self.assertRaisesRegex(RuntimeError,'never truncates'):
            transfer.make_import_room(60,16)

    def test_overlap_runs_real_head_callback_without_publishing_or_waiting_for_h2d(self):
        transfer=KVTransfer.__new__(KVTransfer)
        transfer.signature='same';transfer.block_bytes=16;transfer.imports={};transfer.pending={}
        transfer.validate_manifest=MagicMock()
        request=SimpleNamespace(request_id='r',num_tokens=65,num_computed_tokens=0,
                                block_hashes=['a','b','c','d'])
        pool=MagicMock(num_gpu_blocks=128)
        pool.get_num_free_blocks.return_value=100;pool.get_cached_block.return_value=None
        pool.get_new_blocks.return_value=[SimpleNamespace(block_id=2),SimpleNamespace(block_id=3)]
        transfer.scheduler=SimpleNamespace(max_num_running_reqs=4,running=[],
            kv_cache_manager=SimpleNamespace(block_pool=pool,watermark_blocks=0,
                get_computed_blocks=lambda _:(SimpleNamespace(blocks=[[]]),0,0)))
        class Event:
            def __init__(self,t):self.t=t;self.synchronize=MagicMock();self.record=MagicMock()
            def elapsed_time(self,end):return end.t-self.t
        origin,head_begin,head_end=Event(0),Event(15),Event(45)
        transfer.torch=SimpleNamespace(cuda=SimpleNamespace(Event=MagicMock(side_effect=[origin,head_begin,head_end])))
        copied_begin,copied_end=Event(10),Event(30)
        copied_end.query=MagicMock(side_effect=[False,True])
        transfer.copy=MagicMock(return_value=dict(begin=copied_begin,end=copied_end,bytes=32))
        info={}
        def real_head():
            self.assertFalse(info['preload_complete'])
            self.assertEqual(transfer.clip_prefill(request,0,65),32)
            pool.cache_full_blocks.assert_not_called()
            transfer.advance_restores();pool.cache_full_blocks.assert_not_called()
            request.num_computed_tokens=32
            return dict(head_tokens=32,scheduled_tokens=32)
        checkpoint=dict(signature='same',slots=[0,1],blocks=[dict(logical_block=2,cpu_slot=0),dict(logical_block=3,cpu_slot=1)])
        timing=transfer.restore_cohort([(request,checkpoint,info)],'suffix-overlap',real_head)
        copied_end.synchronize.assert_not_called()
        head_end.synchronize.assert_called_once()
        self.assertTrue(info['preload_complete'])
        self.assertEqual(timing['head_tokens'],32)
        self.assertEqual(timing['copy_forward_envelope_overlap_ms'],15)
        self.assertEqual(request.num_computed_tokens,32)
        transfer.copy.assert_called_once_with([2,3],[0,1],False)

    def test_cohort_deduplicates_shared_suffix_before_any_h2d_is_published(self):
        transfer=KVTransfer.__new__(KVTransfer)
        transfer.signature='same';transfer.block_bytes=16;transfer.imports={};transfer.pending={}
        transfer.validate_manifest=MagicMock()
        pool=MagicMock(num_gpu_blocks=128)
        pool.get_num_free_blocks.return_value=100;pool.get_cached_block.return_value=None
        pages=[SimpleNamespace(block_id=2),SimpleNamespace(block_id=3)]
        pool.get_new_blocks.return_value=pages
        transfer.scheduler=SimpleNamespace(max_num_running_reqs=4,running=[],
            kv_cache_manager=SimpleNamespace(block_pool=pool,watermark_blocks=0,
                get_computed_blocks=lambda _:(SimpleNamespace(blocks=[[]]),0,0)))
        begin=MagicMock();begin.elapsed_time.return_value=1.
        transfer.copy=MagicMock(return_value=dict(begin=begin,end=MagicMock(),bytes=32))
        records=[{},{}]
        requests=[SimpleNamespace(request_id=str(i),num_tokens=65,num_computed_tokens=0,
                                  block_hashes=['a','b','c','d']) for i in range(2)]
        checkpoint=dict(signature='same',slots=[0,1],blocks=[dict(logical_block=2,cpu_slot=0),dict(logical_block=3,cpu_slot=1)])
        timing=transfer.restore_cohort([(r,checkpoint,info) for r,info in zip(requests,records)])
        transfer.copy.assert_called_once_with([2,3],[0,1],False)
        self.assertEqual([(r['cpu_loaded_tokens'],r['gpu_reused_selected_tokens']) for r in records],[(32,0),(0,32)])
        self.assertEqual(timing['load_bytes'],32)
        self.assertIsNone(records[0]['cpu_load_ms'])  # One batch, no invented per-request DMA split.

    def test_parallel_mode_is_explicit_and_only_valid_for_suffix_backup(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):main.parse_args(['--kv-restore-mode','suffix-overlap'])
        args=main.parse_args(['--cpu-kv-backup-gb','1','--backup-policy','suffix-first',
                              '--kv-restore-mode','suffix-overlap'])
        self.assertEqual(args.kv_restore_mode,'suffix-overlap')

    def test_resume_barrier_accepts_computed_heads_only_in_explicit_parallel_mode(self):
        coordinator=experiment.FailureCoordinator.__new__(experiment.FailureCoordinator)
        coordinator.args=SimpleNamespace(cpu_kv_backup_gb=1,instances=4,kv_restore_mode='serial')
        coordinator.acks={};coordinator.assignments={i:[dict(recovery_id=f'r{i}')] for i in (1,2,3)}
        message=dict(kind='recovery_enqueued',worker=1,recovery_ids=['r1'],admitted_recovery_ids=['r1'],
            queue_ids=dict(waiting=[],skipped_waiting=[]),running_recovery_ids=['r1'],
            kv_restore_mode='suffix-overlap',restore_protocol='eager-preload',preload_complete=True)
        with self.assertRaisesRegex(RuntimeError,'Only suffix-overlap'):coordinator.handle_ack(message)
        coordinator.args.kv_restore_mode='suffix-overlap'
        self.assertTrue(coordinator.handle_ack(message))
        self.assertEqual(set(coordinator.acks),{1})  # One target cannot release the barrier.

    def test_parent_owns_checkpoint_memory_and_progress_metadata(self):
        with tempfile.TemporaryDirectory() as directory, SharedKVPool(.001) as pool:
            coordinator = experiment.FailureCoordinator(main.parse_args(['--cpu-kv-backup-gb', '.001']),
                [], Path(directory), 0, cpu_metadata=pool.metadata)
            for update in (dict(request_id='r', prompt_token_ids=[3, 4], output_length=4),
                           dict(request_id='r', new_tokens=[5, 6]),
                           dict(request_id='r', slots=[0], signature='layout')):
                coordinator.handle(dict(kind='cpu_kv_metadata', worker=0, updates=[update]))
            del coordinator
            self.assertEqual(pool.metadata['r']['generated_token_ids'], [5, 6])
            self.assertEqual(pool.metadata['r']['slots'], [0])
        self.assertIsNone(pool.fd)
        self.assertFalse(pool.metadata)

    def test_replay_divergence_fails_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = [dict(scheduled_requests=[dict(request_id='r', scheduled_tokens=3)])]
            outputs = [SimpleNamespace(request_id='r', finished=False,
                                       outputs=[SimpleNamespace(token_ids=[1])])]
            recorded = engine.PrefailureReplay(engine.ReplayContext('record', directory), 0)
            recorded.step(1, .1, .2, batch, outputs)
            native = SimpleNamespace(running=[], waiting=[], skipped_waiting=[], requests={})
            recorded.finish(native, {}, 1, .3)
            replay = engine.PrefailureReplay(engine.ReplayContext('replay', directory), 0)
            outputs[0].outputs[0].token_ids = [2]
            with self.assertRaisesRegex(RuntimeError, 'diverged at forward 0'):
                replay.step(1, .1, .2, batch, outputs)
            self.assertEqual(replay.index, 0)

    def run_protocol(self, mode='restart', strategy='kv-aware', order='group-first', context=None):
        args = main.parse_args(['--warmup', '.02', '--failure-delay', '.03', '.03',
            '--max-duration', '.30', '--recovery-drain', '.5', '--survivor-trace', 'traceB',
            '--recovery-scheduler', strategy, '--recovery-order', order, '--recovery-mode', mode])
        args.per_instance_qps = 1.
        context = context or engine.ReplayContext()
        requests = [dict(request_id=f'TA{i:05d}', source_index=i, input_length=256,
                         output_length=20, tokens=[7]*240+[8+i]*16) for i in range(7)]
        arrivals = [0., .002, .004, .006, .008, .075, .29]
        engines = [Engine() for _ in range(4)]
        pairs = [mp.Pipe() for _ in range(4)]
        errors = []
        start = time.perf_counter()+.03
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,
                {'vllm': SimpleNamespace(SamplingParams=SimpleNamespace)}):
            output = Path(temporary)
            coordinator = experiment.FailureCoordinator(args, [p[0] for p in pairs],
                                                        output, start, context=context)
            def work(worker):
                try:
                    time.sleep(max(0, start-time.perf_counter()))
                    local = [dict(row, request_id=('TB' if worker else 'TA')+row['request_id'][2:])
                             for row in requests]
                    native = engines[worker]
                    result = engine.FailureRunner(native, native.scheduler, local, arrivals,
                        args, worker, args.gpus[worker], output, pairs[worker][1], start, context).run()
                    pairs[worker][1].send(dict(kind='result', **result))
                except BaseException as exc:
                    errors.append(exc)
                    pairs[worker][1].send(dict(kind='error', error=repr(exc)))
            threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(4)]
            for thread in threads:
                thread.start()
            pending = {p[0] for p in pairs}
            try:
                deadline = time.perf_counter()+5
                while pending and time.perf_counter()<deadline:
                    for connection in wait(pending, timeout=.1):
                        message = connection.recv()
                        self.assertNotEqual(message['kind'], 'error', message)
                        if not coordinator.handle(message):
                            self.assertEqual(message['kind'], 'result')
                            pending.remove(connection)
                self.assertFalse(pending)
                self.assertFalse(errors)
                result = coordinator.finalize()
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['traffic'], dict(planned=28, completed=28, unfinished=0))
                self.assertEqual({p.name for p in output.iterdir()}, set(report.FAILURE_REPORT_FILES))
                for worker, items in coordinator.assignments.items():
                    ack = coordinator.acks[worker]
                    self.assertEqual(ack['queue_ids']['waiting'][:len(items)], ack['recovery_ids'])
                    for row in items:
                        expected, _ = engine.recovery_input(row, mode)
                        self.assertEqual(engines[worker].prompts[row['recovery_id']], expected)
                for row in coordinator.after_report['recovery_requests']:
                    self.assertGreater(row['ttft_ms'], 0)
                    self.assertGreater(row['tpot_ms'], 0)
                    self.assertAlmostEqual(row['ttft_ms'], row['queue_wait_ms']+row['scheduled_to_first_ms'])
                    self.assertGreaterEqual(row['ttfo_ms'],row['ttft_ms'])
                    self.assertIsNotNone(row['first_new_token_id'])
                    self.assertEqual(len(row['continuation_sha256']),64)
                    self.assertIn('prefill_to_first_tokens',row)
                    batches=coordinator.results[row['target_worker']]['recovery_batches']
                    scheduled=[b['batch_id'] for b in batches
                        if any(r['request_id']==row['request_id'] for r in b['scheduled_requests'])]
                    prefill=[b['batch_id'] for b in batches
                        if any(r['request_id']==row['request_id'] and r['prefill_tokens']
                               for r in b['scheduled_requests'])]
                    self.assertEqual(row['scheduled_batch_ids'],scheduled)
                    self.assertEqual(row['prefill_batch_ids'],prefill)
                    self.assertEqual(row['first_schedule_batch'],scheduled[0])
                    self.assertIn(row['first_token_batch'],scheduled)
                    self.assertGreater(len(scheduled),1)  # Decode batches must not be truncated.
                    if mode == 'resume':
                        self.assertEqual(row['continued_output_index'], row['source_generated_tokens']+1)
                return copy.deepcopy(dict(before=coordinator.before_report, plan=coordinator.plan,
                                          after=coordinator.after_report))
            finally:
                for thread in threads:
                    thread.join(1)
                for parent, child in pairs:
                    parent.close(); child.close()

    def test_native_queues_mixed_traffic_and_resume(self):
        for mode in ('restart', 'resume'):
            with self.subTest(mode=mode):
                self.run_protocol(mode=mode)

    def test_strict_replay_uses_explicit_context_and_same_fault_state(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.run_protocol(context=engine.ReplayContext('record', directory))
            replay = self.run_protocol(order='shortest', context=engine.ReplayContext('replay', directory))
            self.assertEqual(baseline['plan']['original_running_ids'], replay['plan']['original_running_ids'])
            self.assertEqual([r['prefailure_state_signature'] for r in baseline['before']['gpus']],
                             [r['prefailure_state_signature'] for r in replay['before']['gpus']])

    def test_main_dry_run_outputs_three_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root/'model'; model.mkdir()
            (model/'config.json').write_text(json.dumps(dict(max_position_embeddings=128, vocab_size=32)))
            for name, token in [('source', 3), ('background', 4)]:
                row = dict(input_length=3, output_length=4, prompt_token_ids=[token]*3)
                (root/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for _ in range(10)))
            output = root/'result'
            flags = ['--model', str(model), '--trace', str(root/'source.jsonl'),
                '--survivor-trace', str(root/'background.jsonl'), '--source-qps', '1',
                '--survivor-qps', '1', '--warmup', '.1', '--failure-delay', '.2', '.2',
                '--max-duration', '1', '--output', str(output), '--dry-run']
            with contextlib.redirect_stdout(io.StringIO()):
                main.main(flags)
            self.assertEqual({p.name for p in output.iterdir()}, set(report.FAILURE_REPORT_FILES))
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                main.main(flags)


if __name__ == '__main__':
    unittest.main()
