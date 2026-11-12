"""CPU regression gates for incremental ordering and pre-failure plan reuse."""
import copy
import multiprocessing as mp
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# This historical implementation preserves the original decision rules and does
# not import the optimized scheduler. Compare complete results, including ties.
import recovery_scheduling as reference
from failure_recovery import engine, experiment, main, report, scheduling
from failure_recovery.cpu_kv import SourceKVBackup, WallProfile, predict_backup_queues


def checkpoints(count=12):
    return [dict(request_id=f'R{i}',source_index=i,input_length=65+i,output_length=30,
                 prompt_token_ids=[i//4]*48+[i]*17+[9]*i,generated_token_ids=[100+i])
            for i in range(count)]


def queues(module,rows,survivors=(1,2,3),strategy='kv-aware',mode='resume',
           minimum=16,split=1.2,order='group-ratio'):
    assigned=module.assign_recovery_requests(rows,list(survivors),strategy,mode,minimum,split)
    return {w:module.order_recovery_requests(items,order,mode) for w,items in assigned.items()}


class SchedulingOptimizationTests(unittest.TestCase):
    def test_randomized_placement_and_all_order_metadata_match_reference(self):
        rng=random.Random(20261104)
        for case in range(32):
            rows=[]
            for index in range(rng.randrange(25)):
                # Nested prefixes, identical inputs, short requests and 16-token boundaries.
                tokens=([rng.randrange(3)]*32+[rng.randrange(3)]*32+
                        [rng.randrange(4) for _ in range(rng.randrange(65))])
                tokens=tokens[:rng.choice((1,15,16,17,32,48,64,65,96,129))]
                rows.append(dict(request_id=f'R{index}',prompt_token_ids=tokens,
                    generated_token_ids=[rng.randrange(5) for _ in range(rng.randrange(5))]))
            original=copy.deepcopy(rows)
            for mode in ('resume','restart'):
                for strategy in ('equal','prefill-balanced','kv-aware'):
                    for order in ('source','interleave','prefix-shortest','group-first','group-ratio'):
                        options=dict(survivors=[3,1,2][:1+case%3],strategy=strategy,mode=mode,
                                     minimum=(16,32,64)[case%3],split=(1.,1.2,2.)[case%3],order=order)
                        with self.subTest(case=case,mode=mode,strategy=strategy,order=order):
                            self.assertEqual(queues(scheduling,rows,**options),queues(reference,rows,**options))
            self.assertEqual(rows,original)

    def test_group_forecasts_match_with_prefixes_shared_across_group_labels(self):
        rng=random.Random(31)
        for case in range(35):
            rows=checkpoints(18)
            for row in rows:
                row['prompt_token_ids']=[7]*32+[rng.randrange(3)]*32+[rng.randrange(3)]*rng.randrange(1,65)
            assigned=scheduling.assign_equal_count(rows,[1],'resume')[1]
            for row in assigned:
                row['recovery_group']=f'G{rng.randrange(7)}'
            for order in ('group-first','group-ratio'):
                with self.subTest(case=case,order=order):
                    self.assertEqual(scheduling.order_recovery_requests(assigned,order,'resume'),
                                     reference.order_recovery_requests(assigned,order,'resume'))

    def test_heavy_identical_and_nested_groups_keep_split_decisions(self):
        for identical in (False,True):
            rows=checkpoints(48)
            for i,row in enumerate(rows):
                row['prompt_token_ids']=[7]*1024+([] if identical else [i//8]*64+[i]*16)
                row['generated_token_ids']=[]
            for split in (1.,1.2,2.):
                for order in ('prefix-shortest','group-first','group-ratio'):
                    with self.subTest(identical=identical,split=split,order=order):
                        self.assertEqual(queues(scheduling,rows,split=split,order=order),
                                         queues(reference,rows,split=split,order=order))

    def test_prefix_shortest_compares_each_pair_once(self):
        rows=scheduling.assign_equal_count(checkpoints(32),[1],'resume')[1]
        with patch.object(scheduling,'_shared_blocks',wraps=scheduling._shared_blocks) as shared:
            scheduling.order_recovery_requests(rows,'prefix-shortest','resume')
        self.assertEqual(shared.call_count,32*31//2)

    def test_group_forecasts_scan_each_pair_at_most_once_and_release_cache(self):
        rows=scheduling.assign_equal_count(checkpoints(24),[1],'resume')[1]
        for i,row in enumerate(rows):
            row['recovery_group']=f'G{i%7}'
        with patch.object(scheduling,'_scan_common_length',wraps=scheduling._scan_common_length) as scan:
            scheduling.order_recovery_requests(rows,'group-ratio','resume')
        self.assertLessEqual(scan.call_count,24*23//2)
        self.assertIsNone(scheduling._PREFIX_CACHE.get())

    def test_placement_evaluates_each_unchanged_group_cost_once(self):
        rows=checkpoints(40)
        for i,row in enumerate(rows):
            row['prompt_token_ids']=[i]*32
        with patch.object(scheduling,'_group_cost',wraps=scheduling._group_cost) as cost:
            scheduling.assign_recovery_requests(rows,[1,2,3],'kv-aware','resume')
        self.assertEqual(cost.call_count,len(rows))

    def test_shared_block_estimate_remains_directional(self):
        self.assertEqual(scheduling._shared_blocks([7]*16,[7]*32),16)
        self.assertEqual(scheduling._shared_blocks([7]*32,[7]*16),0)


class RecoveryPlanReuseTests(unittest.TestCase):
    def setUp(self):
        self.rows=checkpoints()
        self.config=scheduling.recovery_plan_config([1,2,3],'kv-aware','resume',16,1.2,'group-ratio')
        self.expected=queues(scheduling,self.rows)
        self.plan=scheduling.make_recovery_plan(self.rows,self.expected,**self.config)
        self.plan['version']=1

    def test_matching_plan_skips_algorithms_and_binds_fresh_checkpoint(self):
        fresh=copy.deepcopy(self.rows)
        for row in fresh:
            row.update(cpu_kv={'epoch':'new','slots':[91]},source_computed_tokens=999)
        expected=queues(scheduling,fresh)
        with (patch.object(scheduling,'assign_recovery_requests',side_effect=AssertionError('recomputed')),
              patch.object(scheduling,'order_recovery_requests',side_effect=AssertionError('reordered'))):
            actual=scheduling.reuse_recovery_plan(self.plan,fresh,**self.config)
        self.assertEqual(actual,expected)
        self.assertTrue(all(row['cpu_kv'] is fresh[int(row['request_id'][1:])]['cpu_kv']
                            for rows in actual.values() for row in rows))
        self.assertNotIn('cpu_kv',self.plan['queues'][1][0])

    def test_changed_inputs_ids_membership_or_source_order_reject_plan(self):
        for kind in ('prompt','output','id','remove','append','order'):
            rows=copy.deepcopy(self.rows)
            if kind=='prompt':rows[0]['prompt_token_ids'][0]+=1
            elif kind=='output':rows[0]['generated_token_ids'][0]+=1
            elif kind=='id':rows[0]['request_id']='different'
            elif kind=='remove':rows.pop()
            elif kind=='append':rows.append(dict(rows[-1],request_id='added'))
            else:rows.reverse()
            with self.subTest(kind=kind):
                self.assertIsNone(scheduling.reuse_recovery_plan(self.plan,rows,**self.config))

    def test_every_scheduling_config_field_is_validated(self):
        for key,value in dict(survivors=[3,2,1],strategy='equal',mode='restart',
                              min_shared_tokens=32,split_factor=1.3,order='prefix-shortest').items():
            with self.subTest(key=key):
                self.assertIsNone(scheduling.reuse_recovery_plan(self.plan,self.rows,
                                                               **dict(self.config,**{key:value})))

    def test_corrupt_coverage_or_queue_metadata_rejects_plan(self):
        for kind in ('duplicate','missing','rank','cost','schema'):
            plan=copy.deepcopy(self.plan)
            if kind=='duplicate':plan['queues'][1].append(plan['queues'][1][0])
            elif kind=='missing':plan['queues'][1].pop()
            elif kind=='rank':plan['queues'][1][0]['queue_rank']=99
            elif kind=='cost':plan['queues'][1][0]['estimated_prefill_tokens']+=1
            else:plan['schema_version']=99
            with self.subTest(kind=kind):
                self.assertIsNone(scheduling.reuse_recovery_plan(plan,self.rows,**self.config))

    def test_empty_cohort_is_a_valid_reusable_plan(self):
        plan=scheduling.make_recovery_plan([],{1:[],2:[],3:[]},**self.config)
        self.assertEqual(scheduling.reuse_recovery_plan(plan,[],**self.config),{1:[],2:[],3:[]})

    def test_predictor_preserves_backup_order_and_supports_restart(self):
        native=[SimpleNamespace(request_id=row['request_id'],prompt_token_ids=row['prompt_token_ids'],
                                output_token_ids=row['generated_token_ids']) for row in self.rows]
        for mode in ('resume','restart'):
            order,groups,plan=predict_backup_queues(native,[1,2,3],mode=mode,return_plan=True)
            expected=queues(scheduling,self.rows,mode=mode)
            config=dict(self.config,mode=mode)
            self.assertEqual(scheduling.reuse_recovery_plan(plan,self.rows,**config),expected)
            self.assertEqual((order,groups),predict_backup_queues(native,[1,2,3],mode=mode))
            self.assertEqual(set(order),{row['request_id'] for row in self.rows})

    def coordinator(self,directory,plan=None,changed=False):
        args=main.parse_args(['--cpu-kv-backup-gb','.001','--backup-policy','group-prefix-queue',
                             '--recovery-scheduler','kv-aware','--recovery-order','group-ratio'])
        coordinator=experiment.FailureCoordinator(args,[MagicMock() for _ in range(4)],
                                                 Path(directory),time.perf_counter())
        rows=copy.deepcopy(self.rows)
        if changed:rows[0]['generated_token_ids'].append(123)
        for row in rows:
            row['cpu_kv']=dict(slots=[],blocks=[],signature='layout',epoch='pool',version=0)
            coordinator.cpu_metadata[row['request_id']]=dict(row['cpu_kv'],
                prompt_token_ids=row['prompt_token_ids'],generated_token_ids=row['generated_token_ids'])
        coordinator.snapshots={w:dict(time_s=1.,latencies=[]) for w in range(4)}
        coordinator.snapshots[0].update(checkpoints=rows,running_ids=[r['request_id'] for r in rows],
            waiting_requests=[],future_requests=[],waiting_ids=[],future_not_submitted=0,
            cpu_epoch='pool',cpu_directory={})
        coordinator.recovery_plan=plan
        return coordinator,rows

    def test_source_publication_parent_storage_and_fault_dispatch_reuse(self):
        native=[SimpleNamespace(request_id=f'native{i}',arrival_time=i,
            prompt_token_ids=row['prompt_token_ids'],output_token_ids=row['generated_token_ids'])
            for i,row in enumerate(self.rows)]
        transfer=SimpleNamespace(size=1,block_bytes=1,
            scheduler=SimpleNamespace(running=native,requests={r.request_id:r for r in native}))
        backup=SourceKVBackup(transfer,'group-prefix-queue',queue_config={'survivors':[1,2,3]})
        backup.select_candidates([],[],[],[])  # Includes members without any complete KV block.
        ids={r.request_id:row['request_id'] for r,row in zip(native,self.rows)}
        runner=engine.FailureRunner.__new__(engine.FailureRunner)
        runner.cpu_backup,runner.internal_ids=backup,ids
        runner.profiler=WallProfile();runner.worker=0;runner.gpu=0
        receiver,sender=mp.Pipe();runner.connection=sender
        try:
            with tempfile.TemporaryDirectory() as directory:
                coordinator,rows=self.coordinator(directory)
                # A plan must be published even when no KV-directory delta is ready.
                with patch.object(backup,'updates',return_value=[]):runner.publish_cpu_updates()
                coordinator.handle(receiver.recv())
                self.assertEqual(coordinator.recovery_plan['request_ids'],[r['request_id'] for r in rows])
                self.assertIsNone(backup.take_recovery_plan(ids))
                # The next completed step replaces the parent's plan before the fault.
                native[0].output_token_ids.append(123)
                rows[0]['generated_token_ids'].append(123)
                backup.select_candidates([],[],[],[])
                with patch.object(backup,'updates',return_value=[]):runner.publish_cpu_updates()
                coordinator.handle(receiver.recv())
                self.assertEqual(coordinator.recovery_plan['version'],2)
                with (patch.object(experiment,'recovery_assignment',side_effect=AssertionError('recomputed')),
                      patch.object(experiment,'order_recovery_requests',side_effect=AssertionError('reordered')),
                      patch.object(coordinator,'save_before')):
                    coordinator.dispatch()
                self.assertEqual(coordinator.assignments,queues(scheduling,rows))
                self.assertTrue(coordinator.plan['recovery_plan_reused'])
                self.assertIn('复用故障前预测计划',(Path(directory)/report.FAILURE_REPORT_FILES[1]).read_text())
                for worker in (1,2,3):
                    self.assertEqual(coordinator.connections[worker].send.call_args.args[0]['checkpoints'],
                                     coordinator.assignments[worker])
                with self.assertRaisesRegex(RuntimeError,'Out-of-order recovery plan'):
                    coordinator.handle(dict(kind='cpu_kv_metadata',worker=0,updates=[],
                                            recovery_plan=coordinator.recovery_plan))
        finally:
            receiver.close();sender.close()

    def test_missing_or_stale_plan_recomputes_once_at_fault(self):
        for plan,changed in ((None,False),(self.plan,True)):
            with self.subTest(missing=plan is None),tempfile.TemporaryDirectory() as directory:
                coordinator,rows=self.coordinator(directory,plan,changed)
                with (patch.object(experiment,'recovery_assignment',wraps=experiment.recovery_assignment) as assign,
                      patch.object(experiment,'order_recovery_requests',wraps=experiment.order_recovery_requests) as order,
                      patch.object(coordinator,'save_before')):
                    coordinator.dispatch()
                self.assertEqual(assign.call_count,1)
                self.assertEqual(order.call_count,3)
                self.assertFalse(coordinator.plan['recovery_plan_reused'])
                self.assertEqual(coordinator.assignments,queues(scheduling,rows))


if __name__=='__main__':
    unittest.main()
