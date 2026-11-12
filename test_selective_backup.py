"""Real-GPU E0/E1 correctness gate for the restored six-module baseline."""
import argparse
import contextlib
import io
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from failure_recovery.cpu_kv import (SharedKVPool,CheckpointIndex,SourceKVBackup,select_blocks,WallProfile,
    apply_manifest_delta,materialize_manifest,predict_backup_queues)

ROOT=Path(__file__).resolve().parent


class SelectiveIndexTests(unittest.TestCase):
    def test_physical_profile_nested_phases_do_not_double_count(self):
        now=[0.]
        with patch('failure_recovery.cpu_kv.time.perf_counter',side_effect=lambda:now[0]):
            clock=WallProfile(True,0.)
            with clock.stage('outer'):
                now[0]=1.
                with clock.stage('inner'):
                    now[0]=3.
                now[0]=4.
            now[0]=5.;clock.stop();snapshot=clock.snapshot()
            self.assertEqual(snapshot['exclusive_ms'],dict(inner=2000.,outer=2000.))
            self.assertEqual(snapshot['inclusive_ms']['outer'],4000.)
            self.assertEqual(snapshot['unattributed_ms'],1000.)
            now[0]=6.
            with clock.stage('cleanup'):now[0]=9.
            self.assertEqual(clock.snapshot(),snapshot)

    def test_first_divergence_keeps_same_boundary_physical_profiles(self):
        from failure_recovery.engine import PrefailureReplay,ReplayContext
        import tempfile
        batches=[dict(scheduled_requests=[dict(request_id='R',scheduled_tokens=16,computed_tokens_before_step=0)])]
        output=[SimpleNamespace(request_id='R',finished=False,outputs=[SimpleNamespace(token_ids=[7])])]
        native=SimpleNamespace(running=[],waiting=[],skipped_waiting=[],requests={})
        with tempfile.TemporaryDirectory() as directory:
            context=ReplayContext('record',directory,inference_only=True)
            baseline=PrefailureReplay(context,0)
            baseline.step(1,.1,.2,batches,output,diagnostic=dict(before_forward={'wall_ms':100.}))
            baseline.finish(native,{},1,.3);baseline.close()
            Path(directory,'worker_0_steps.json').write_text(json.dumps(baseline.details))
            replay=PrefailureReplay(ReplayContext('replay',directory,inference_only=True,
                diagnostics_directory=directory),0)
            batches[0]['scheduled_requests'][0]['scheduled_tokens']=8
            with self.assertRaisesRegex(RuntimeError,'diverged at forward 0'):
                replay.step(1,.1,.2,batches,output,diagnostic=dict(before_forward={'wall_ms':150.}))
            self.assertEqual(replay.index,0)
            self.assertEqual(replay.divergence['reference_detail']['before_forward']['wall_ms'],100.)
            self.assertEqual(replay.divergence['actual_detail']['before_forward']['wall_ms'],150.)
            self.assertEqual(replay.divergence['actual_detail']['scheduled'][0][1],8)

    def test_async_replay_is_default_and_fence_is_explicit(self):
        from failure_recovery.main import parse_args
        self.assertEqual(parse_args([]).replay_backup_mode,'async')
        self.assertEqual(parse_args(['--replay-backup-mode','step-fenced']).replay_backup_mode,'step-fenced')

    def test_pending_budget_and_reused_generation(self):
        index=CheckpointIndex(1)
        self.assertEqual(index.reserve('old'),0)
        self.assertEqual(index.retain(set()),[])
        self.assertIsNone(index.reserve('new'))
        self.assertIsNone(index.ref('old'))
        index.complete(['old']);old=index.ref('old')
        index.retain(set());self.assertEqual(index.reserve('new'),0)
        index.complete(['new']);new=index.ref('new')
        self.assertGreater(new['generation'],old['generation'])
        self.assertNotEqual(index.directory()['0'],old)
        self.assertEqual(index.peak,1)

    def test_shared_physical_budget_and_contiguous_owner_ranges(self):
        candidates=[]
        for rid in ('R0','R1','R2'):
            for i in range(4):
                key='shared'+str(i) if i<2 else rid+str(i)
                candidates.append((rid,i,4,key,object()))
        for policy in ('prefix-first','suffix-first'):
            selected,owners=select_blocks(candidates,4,policy)
            self.assertLessEqual(len(selected),4)
            for positions in owners.values():
                if positions:self.assertEqual(sorted(positions),list(range(min(positions),max(positions)+1)))
            if policy=='prefix-first':
                self.assertEqual(set.intersection(*[set(v) for v in owners.values()]),{0,1})
            else:
                self.assertTrue(all(3 in positions for positions in owners.values()))

    def test_ready_tail_manifest_is_not_a_fake_prefix(self):
        transfer=SimpleNamespace(size=8,block_bytes=1,signature='layout',epoch='pool')
        backup=SourceKVBackup(transfer,'suffix-first')
        keys=[str(i) for i in range(7)]
        for key in keys:backup.index.reserve(key)
        backup.index.complete(['0','1','4','5','6'])
        backup.owners={'R':set(range(7))}
        with patch.object(SourceKVBackup,'request_keys',return_value=keys):
            manifest=backup.manifest(SimpleNamespace(request_id='R'))
        self.assertEqual(manifest['slots'],[])
        self.assertEqual([b['logical_block'] for b in manifest['blocks']],[4,5,6])
        self.assertEqual(manifest['unpublished_ready_blocks'],2)
        self.assertTrue(all(b['generation']==1 for b in manifest['blocks']))

    def test_fifo_finishes_the_earliest_request_before_the_next(self):
        candidates=[(rid,i,4,f'{rid}{i}',object()) for rid in ('A','Z') for i in range(4)]
        selected,owners=select_blocks(candidates,6,'request-fifo',['Z','A'])
        self.assertEqual(owners,{'Z':{0,1,2,3},'A':{0,1}})
        self.assertEqual(list(selected),['Z0','Z1','Z2','Z3','A0','A1'])

    def test_fifo_full_pool_still_exposes_shared_prefix_to_later_requests(self):
        candidates=[(rid,i,4,'shared'+str(i) if i<2 else rid+str(i),object())
                    for rid in ('A','B') for i in range(4)]
        selected,owners=select_blocks(candidates,4,'request-fifo',['A','B'])
        self.assertEqual(len(selected),4)
        self.assertEqual(owners,{'A':{0,1,2,3},'B':{0,1}})

    def test_fifo_never_jumps_over_a_missing_head_block(self):
        candidates=[('A',0,3,'a0',None),('A',2,3,'a2',None)]
        _,owners=select_blocks(candidates,5,'request-fifo',['A'])
        self.assertEqual(owners,{'A':{0}})

    def test_fifo_cli_enables_resume(self):
        from failure_recovery.main import parse_args
        args=parse_args(['--backup-policy','request-fifo','--cpu-kv-backup-bytes','1024'])
        self.assertEqual(args.backup_policy,'request-fifo')
        self.assertEqual(args.recovery_mode,'resume')

    def test_fifo_uses_arrival_order_instead_of_id_or_current_running_order(self):
        a=SimpleNamespace(request_id='A',arrival_time=9.)
        z=SimpleNamespace(request_id='Z',arrival_time=1.)
        scheduler=SimpleNamespace(requests={'A':a,'Z':z},running=[a,z])
        backup=SourceKVBackup(SimpleNamespace(size=1,block_bytes=1,scheduler=scheduler),'request-fifo')
        blocks,keys,slots=[],[],[]
        backup.select_candidates([('A',0,1,'a',object()),('Z',0,1,'z',object())],blocks,keys,slots)
        self.assertEqual(keys,['z'])
        self.assertEqual(backup.owners,{'Z':{0},'A':set()})

    def test_group_prefixes_precede_first_request_private_tail(self):
        candidates=[(rid,i,3,('ab' if rid in 'AB' else 'cd')+str(i) if i==0 else rid+str(i),None)
                    for rid in ('A','B','C','D') for i in range(3)]
        selected,owners=select_blocks(candidates,3,'group-prefix-queue',['A','B','C','D'],
                                     {'G0':['A','B'],'G1':['C','D']})
        self.assertEqual(list(selected),['ab0','cd0','A1'])
        self.assertEqual(owners,{'A':{0,1},'B':{0},'C':{0},'D':{0}})

    def test_group_head_overflow_is_breadth_first_and_never_exceeds_budget(self):
        candidates=[(rid,i,3,('ab' if rid in 'AB' else 'cd')+str(i),None)
                    for rid in ('A','B','C','D') for i in range(3)]
        selected,owners=select_blocks(candidates,3,'group-prefix-queue',['A','B','C','D'],
                                     {'G0':['A','B'],'G1':['C','D']})
        self.assertEqual(list(selected),['ab0','cd0','ab1'])
        self.assertEqual(owners,{'A':{0,1},'B':{0,1},'C':{0},'D':{0}})

    def test_group_prefix_common_to_all_members_and_singletons_use_second_phase(self):
        candidates=[(rid,i,3,'shared' if i==0 and rid!='D' else rid+str(i),None)
                    for rid in ('A','B','C','D') for i in range(3)]
        selected,owners=select_blocks(candidates,2,'group-prefix-queue',['D','A','B','C'],
                                     {'G0':['A','B','C'],'G1':['D']})
        self.assertEqual(list(selected),['shared','D0'])
        self.assertEqual(owners,{'A':{0},'B':{0},'C':{0},'D':{0}})

    def test_group_backup_requires_matching_recovery_configuration(self):
        from failure_recovery.main import parse_args
        flags=['--backup-policy','group-prefix-queue','--cpu-kv-backup-bytes','1024']
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(flags)
        args=parse_args(flags+['--recovery-scheduler','kv-aware','--recovery-order','group-ratio'])
        self.assertEqual(args.backup_policy,'group-prefix-queue')

    def test_predicted_order_round_robins_target_queue_positions(self):
        rows=[SimpleNamespace(request_id=rid,prompt_token_ids=[i]*48,output_token_ids=[99])
              for i,rid in enumerate(('A','B','C','D','E','F'))]
        assigned={1:[dict(request_id='A',recovery_group='G0'),dict(request_id='B',recovery_group='G0')],
                  2:[dict(request_id='C',recovery_group='G1')],
                  3:[dict(request_id='D',recovery_group='G2'),dict(request_id='E',recovery_group='G2'),
                     dict(request_id='F',recovery_group='G2')]}
        with patch('failure_recovery.scheduling.assign_recovery_requests',return_value=assigned) as place, \
             patch('failure_recovery.scheduling.order_recovery_requests',side_effect=lambda items,*args:items):
            order,groups=predict_backup_queues(rows,[1,2,3])
        self.assertEqual(order,['A','C','D','B','E','F'])
        self.assertEqual(groups,{'G0':['A','B'],'G1':['C'],'G2':['D','E','F']})
        self.assertEqual([r['request_id'] for r in place.call_args.args[0]],list('ABCDEF'))
        self.assertEqual(place.call_args.args[0][0]['generated_token_ids'],[99])

    def test_chunked_common_prefix_matches_token_scan_at_boundaries(self):
        from failure_recovery.scheduling import _common_length
        for kind in (list,tuple):
            for length in (0,1,15,16,255,256,257,511,512,513,1024):
                left=kind([7]*length+[1,2]);right=kind([7]*length+[3,4])
                self.assertEqual(_common_length(left,right),length)
                self.assertEqual(_common_length(left,left),len(left))
                self.assertEqual(_common_length(left,left[:-1]),len(left)-1)
        self.assertEqual(_common_length([7,8,9],(7,8,10)),2)

    def test_chunked_prefix_keeps_placement_and_queue_order_exact(self):
        import random
        from failure_recovery import scheduling
        rng=random.Random(7)
        checkpoints=[dict(request_id=f'R{i}',prompt_token_ids=[i//4]*1024+
                         [rng.randrange(9) for _ in range(30+i)],generated_token_ids=[i]*i)
                     for i in range(24)]
        def plan():
            groups=scheduling.assign_kv_aware(checkpoints,[1,2,3],'resume')
            return {w:scheduling.order_recovery_requests(rows,'group-ratio','resume')
                    for w,rows in groups.items()}
        actual=plan()
        def slow(left,right):
            return next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
        with patch.object(scheduling,'_common_length',side_effect=slow):expected=plan()
        self.assertEqual(actual,expected)

    def test_prediction_cache_is_local_and_releases_references_even_on_error(self):
        from failure_recovery import scheduling
        left=[7]*700+[1];right=[7]*700+[2]
        with patch.object(scheduling,'_scan_common_length',wraps=scheduling._scan_common_length) as scan:
            with self.assertRaisesRegex(RuntimeError,'abort'):
                with scheduling.cache_prefix_comparisons():
                    self.assertEqual(scheduling._common_length(left,right),700)
                    self.assertEqual(scheduling._common_length(right,left),700)
                    self.assertEqual(scan.call_count,1)
                    raise RuntimeError('abort')
            self.assertIsNone(scheduling._PREFIX_CACHE.get())
            with scheduling.cache_prefix_comparisons():scheduling._common_length(left,right)
            self.assertEqual(scan.call_count,2)

    def test_exact_budget_cli_and_legacy_defaults(self):
        from failure_recovery.main import parse_args
        args=parse_args(['--backup-policy','prefix-first','--cpu-kv-backup-bytes','123456789'])
        self.assertEqual(args.cpu_kv_backup_bytes,123456789)
        self.assertEqual(args.recovery_mode,'resume')
        self.assertEqual(parse_args([]).backup_policy,'none')
        self.assertEqual(parse_args(['--cpu-kv-backup-gb','1']).backup_policy,'full')

    def test_eviction_is_published_before_slot_overwrite(self):
        a=SimpleNamespace(block_id=1,is_null=False,block_hash=b'a')
        b=SimpleNamespace(block_id=2,is_null=False,block_hash=b'b')
        request=SimpleNamespace(request_id='R',num_computed_tokens=32,num_tokens=33,block_hashes=[b'a',b'b'])
        events=[]
        manager=SimpleNamespace(get_blocks=lambda rid:SimpleNamespace(blocks=[[a,b]]),
            block_pool=SimpleNamespace(touch=lambda blocks:events.append('touch')))
        transfer=SimpleNamespace(size=1,block_bytes=1,signature='layout',epoch='pool',
            scheduler=SimpleNamespace(running=[request],requests={'R':request},kv_cache_manager=manager))
        backup=SourceKVBackup(transfer,'suffix-first')
        backup.index.reserve('61');backup.index.complete(['61'])
        backup.selected={'61':a};backup.owners={'R':{0}}
        def publish():
            events.append('invalidate');backup.updates({'R':'R'})
            self.assertEqual(backup.published_manifest('R')['blocks'],[])
        def copy(blocks,slots,to_cpu):
            self.assertEqual(events[0],'invalidate')
            self.assertEqual(blocks,[2]);self.assertEqual(slots,[0])
            self.assertEqual(backup.index.generations[0],2)
            events.append('copy');return dict(bytes=1,begin=None,end=None)
        transfer.copy=copy;backup.publish_invalidations=publish
        fake=SimpleNamespace(make_block_hash_with_group_id=lambda value,group:value)
        with patch.object(SourceKVBackup,'request_keys',return_value=['61','62']), \
             patch.dict(sys.modules,{'vllm.v1.core.kv_cache_utils':fake}):
            backup.updates({'R':'R'})
            backup.submit()
        self.assertEqual(events,['invalidate','touch','copy'])
        self.assertEqual(len(backup.index.entries),1)

    def test_inference_replay_does_not_reuse_cpu_operations_or_snapshot(self):
        import tempfile
        from failure_recovery.engine import PrefailureReplay,ReplayContext
        batch=[dict(scheduled_requests=[dict(request_id='R',scheduled_tokens=16)])]
        outputs=[SimpleNamespace(request_id='R',finished=False,outputs=[SimpleNamespace(token_ids=[7])])]
        native=SimpleNamespace(running=[],waiting=[],skipped_waiting=[],requests={})
        with tempfile.TemporaryDirectory() as directory:
            record=PrefailureReplay(ReplayContext('record',directory,inference_only=True),0)
            record.step(1,.1,.2,batch,outputs,cpu_ops=[['touch',[99]]])
            record.finish(native,{},1,.3,cpu_snapshot={'epoch':'old','requests':{}})
            self.assertFalse((Path(directory)/'worker_0.json').exists())
            record.close()
            saved=json.loads((Path(directory)/'worker_0.json').read_text())
            self.assertNotIn('cpu_ops',saved['steps'][0])
            self.assertNotIn('cpu_kv_snapshot',saved)
            replay=PrefailureReplay(ReplayContext('replay',directory,inference_only=True),0)
            replay.step(1,.1,.2,batch,outputs,cpu_ops=[['free',[123]]])
            self.assertTrue(replay.finish(native,{},1,.3,cpu_snapshot={'epoch':'new'})['prefailure_verified'])
            with self.assertRaisesRegex(ValueError,'cannot be mixed'):
                PrefailureReplay(ReplayContext('replay',directory),0)

    def test_inference_replay_checks_survivor_apc_and_eviction_state(self):
        import tempfile
        from failure_recovery.engine import PrefailureReplay,ReplayContext
        block=SimpleNamespace(block_id=0,block_hash=b'actual',block_hash_num_tokens=16,ref_cnt=0)
        pool=SimpleNamespace(blocks=[block],free_block_queue=SimpleNamespace(get_all_free_blocks=lambda:[block]))
        native=SimpleNamespace(running=[],waiting=[],skipped_waiting=[],requests={},
            kv_cache_manager=SimpleNamespace(block_pool=pool))
        with tempfile.TemporaryDirectory() as directory:
            record=PrefailureReplay(ReplayContext('record',directory,inference_only=True),1)
            record.finish(native,{},0,1.);record.close()
            block.block_hash=b'different'
            with self.assertRaisesRegex(RuntimeError,'survivor GPU cache'):
                PrefailureReplay(ReplayContext('replay',directory,inference_only=True),1).finish(native,{},0,1.)

    def test_inference_replay_cli_requires_complete_explicit_configuration(self):
        from failure_recovery.main import parse_args
        with patch.dict(os.environ,{'VLLM_BATCH_INVARIANT':'1'}):
            args=parse_args(['--inference-replay','record','--inference-replay-dir','/tmp/test-reference',
                             '--recovery-mode','resume'])
            self.assertEqual(args.inference_replay,'record')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(['--inference-replay','record','--recovery-mode','resume'])

    def test_replayed_step_releases_completed_dma_refs_before_native_forward(self):
        calls=[];ready=[False]
        event=SimpleNamespace(synchronize=lambda:(calls.append('wait'),ready.__setitem__(0,True)),
                              query=lambda:ready[0])
        block=SimpleNamespace(block_id=7)
        pool=SimpleNamespace(free_blocks=lambda blocks:calls.append('release'))
        transfer=SimpleNamespace(size=1,block_bytes=1,
            scheduler=SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=pool)))
        backup=SourceKVBackup(transfer);backup.index.reserve('key')
        backup.jobs.append(dict(keys=['key'],blocks=[block],bytes=1,end=event,
            begin=SimpleNamespace(elapsed_time=lambda end:1.)))
        backup.before_replayed_step()
        self.assertEqual(calls,['wait','release'])
        self.assertEqual(backup.index.ref('key')['cpu_slot'],0)
        self.assertFalse(backup.jobs)
        self.assertFalse(backup.operations)


class ManifestDeltaTests(unittest.TestCase):
    def make_backup(self, policy='suffix-first', capacity=128, mode='delta', oracle=True):
        scheduler=SimpleNamespace(requests={},running=[],kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(touch=lambda blocks:None,free_blocks=lambda blocks:None),
            get_blocks=lambda rid:SimpleNamespace(blocks=[scheduler.requests[rid].gpu_blocks])))
        transfer=SimpleNamespace(size=capacity,block_bytes=1,signature='layout',epoch='epoch',
            registered=set(),REGISTER_CHUNK=64,scheduler=scheduler,events=[],profiler=WallProfile(True))
        def copy(block_ids,slots,to_cpu):
            event=SimpleNamespace(ready=False,queries=0)
            def query():event.queries+=1;return event.ready
            event.query=query;event.synchronize=lambda:None
            transfer.events.append(event)
            return dict(bytes=len(slots),begin=SimpleNamespace(elapsed_time=lambda end:1.),end=event)
        transfer.copy=copy
        backup=SourceKVBackup(transfer,policy,mode,oracle,
            queue_config=dict(survivors=[1,2,3]) if policy=='group-prefix-queue' else None)
        wire={}
        def publish():
            for update in backup.updates({rid:rid for rid in scheduler.requests}):
                rid=update['request_id'];entry=wire.setdefault(rid,{})
                if 'manifest_delta' in update:apply_manifest_delta(entry,update['manifest_delta'])
                else:entry.update({k:v for k,v in update.items() if k!='request_id'})
            for rid in wire.keys()-scheduler.requests.keys():del wire[rid]
        backup.publish_invalidations=publish
        return backup,wire,publish

    @staticmethod
    def request(rid, count, shared=2):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        hashes=[(f'shared_{i}' if i<shared else f'{rid}_{i}').encode() for i in range(count)]
        return SimpleNamespace(request_id=rid,num_tokens=count*16+1,num_computed_tokens=count*16,
            prompt_token_ids=[(i if i<shared else 1000+sum(map(ord,rid))*100+i)
                              for i in range(count) for _ in range(16)],output_token_ids=[99],
            block_hashes=hashes,gpu_blocks=[SimpleNamespace(block_id=i,is_null=False,
                block_hash=make_block_hash_with_group_id(h,0)) for i,h in enumerate(hashes)])

    def test_suffix_slide_sends_one_add_one_remove_without_full_rebuild(self):
        backup,wire,publish=self.make_backup(oracle=False)
        request=self.request('R',101);backup.transfer.scheduler.requests={'R':request}
        keys=backup.request_keys(request)
        for key in keys:backup.index.reserve(key)
        backup.index.complete(keys);backup.owners={'R':set(range(100))};publish()
        new={'R':set(range(1,101))};backup.mark_owners(new);backup.owners=new;backup.dirty=True
        with patch.object(backup,'manifest',side_effect=AssertionError('full rebuild')):
            updates=backup.updates({'R':'R'})
        delta=updates[0]['manifest_delta']
        self.assertEqual(delta['remove'],[0]);self.assertEqual(len(delta['add']),1)
        self.assertEqual(delta['add'][0]['logical_block'],100)
        apply_manifest_delta(wire['R'],delta)
        self.assertEqual(materialize_manifest(wire['R']),backup.manifest(request))

    def test_delta_rejects_gap_epoch_and_duplicate_before_mutating_directory(self):
        backup,wire,publish=self.make_backup(oracle=False)
        request=self.request('R',3);backup.transfer.scheduler.requests={'R':request};publish()
        entry=wire['R'];before=dict(entry,blocks_by_position=dict(entry['blocks_by_position']))
        base=dict(base_version=1,version=2,epoch='epoch',signature='layout',remove=[],add=[],
            start_block=0,end_block=0,unpublished_ready_blocks=0)
        for change in [dict(version=3),dict(base_version=0),dict(epoch='foreign'),
                       dict(remove=[0]),dict(end_block=1),dict(start_block=-1)]:
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                apply_manifest_delta(entry,dict(base,**change))
            self.assertEqual(entry,before)

    def test_ready_holes_and_shared_hash_notify_each_owner(self):
        for policy in ('full','prefix-first','suffix-first','request-fifo','group-prefix-queue'):
            backup,wire,publish=self.make_backup(policy)
            requests={rid:self.request(rid,5) for rid in ('A','B')}
            backup.transfer.scheduler.requests=requests
            backup.owners={rid:set(range(5)) for rid in requests};publish()
            all_keys=set(key for r in requests.values() for key in backup.request_keys(r))
            for key in sorted(all_keys):backup.index.reserve(key)
            for rid in requests:
                keys=backup.request_keys(requests[rid])
                backup.index.complete([keys[0],keys[3],keys[4]])
                backup.mark_keys([keys[0],keys[3],keys[4]])
            backup.dirty=True;publish()
            for rid,r in requests.items():self.assertEqual(materialize_manifest(wire[rid]),backup.manifest(r))
            self.assertEqual(len(backup.locations[backup.request_keys(requests['A'])[0]]),2)
            for r in requests.values():
                keys=backup.request_keys(r);backup.index.complete(keys[1:3]);backup.mark_keys(keys[1:3])
            backup.dirty=True;publish()
            for rid,r in requests.items():self.assertEqual(materialize_manifest(wire[rid]),backup.manifest(r))

    def test_fixed_event_replay_matches_legacy_slots_manifests_and_pending_fault(self):
        import random
        for policy in ('full','prefix-first','suffix-first','request-fifo','group-prefix-queue'):
            rng=random.Random(17)
            delta,wire,publish=self.make_backup(policy,capacity=19)
            legacy,old_wire,old_publish=self.make_backup(policy,capacity=19,mode='legacy')
            for step in range(160):
                requests={rid:self.request(rid,3+step//11+index) for index,rid in enumerate(('A','B','C'))}
                if step%13==0:requests.pop('B')
                for r in requests.values():r.num_computed_tokens=max(0,len(r.block_hashes)-rng.randrange(3))*16
                running=[r for r in requests.values() if rng.random()>.2]
                for backup,pub in ((legacy,old_publish),(delta,publish)):
                    backup.transfer.scheduler.requests=requests
                    backup.transfer.scheduler.running=running
                    for event in backup.transfer.events[:-1] if step%3 else backup.transfer.events:event.ready=True
                    backup.poll();pub();backup.submit()
                self.assertEqual(delta.index.entries,legacy.index.entries,(policy,step))
                self.assertEqual(delta.index.generations,legacy.index.generations)
                self.assertEqual(delta.index.free,legacy.index.free)
                self.assertEqual(delta.owners,legacy.owners)
                self.assertEqual(delta.operations,legacy.operations)
                self.assertEqual(delta.submitted_bytes,legacy.submitted_bytes)
                for rid in requests:self.assertEqual(materialize_manifest(wire[rid]),old_wire[rid],(policy,step,rid))
            publish();old_publish()
            queries=[event.queries for event in delta.transfer.events]
            checkpoint=delta.freeze(list(requests));old=legacy.freeze(list(requests))
            self.assertEqual(checkpoint['requests'],old['requests'])
            self.assertEqual(checkpoint['slot_directory'],old['slot_directory'])
            self.assertEqual(queries,[event.queries for event in delta.transfer.events])
            self.assertLessEqual(len(delta.index.entries),19)


def reference_worker(gpu,config,checkpoints,expected,connection):
    """Independent E0 recomputation for an online run's exact fault progress."""
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_BATCH_INVARIANT='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_LOGGING_LEVEL='ERROR',
        VLLM_NO_USAGE_STATS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for name in list(os.environ):
        if name.startswith(('KV_P1_','KV_P2_','KV_P3_','KV_PREFILL_','P1_KV_')):del os.environ[name]
    engine=None
    try:
        from vllm import EngineArgs,SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        engine=LLMEngine.from_engine_args(EngineArgs(model=config['model'],dtype=config['dtype'],
            max_model_len=config['max_model_len'],max_num_seqs=config['max_num_seqs'],
            max_num_batched_tokens=config['batch_tokens'],gpu_memory_utilization=.9,
            block_size=16,enable_prefix_caching=True,enable_chunked_prefill=True,
            async_scheduling=False,seed=20261019),enable_multiprocessing=False)
        for checkpoint in checkpoints:
            rid=checkpoint['request_id'];tokens=checkpoint['prompt_token_ids']+checkpoint['generated_token_ids']
            engine.add_request(rid,{'prompt_token_ids':tokens},SamplingParams(temperature=0,
                max_tokens=len(expected[rid]),ignore_eos=True,detokenize=False))
        completed={}
        while engine.has_unfinished_requests():
            with contextlib.redirect_stdout(io.StringIO()):outputs=engine.step()
            for output in outputs:
                if output.finished:completed[output.request_id]=list(output.outputs[0].token_ids)
        rows=[]
        for rid,tokens in completed.items():
            mismatch=next((i for i,(a,b) in enumerate(zip(tokens,expected[rid])) if a!=b),None)
            same=tokens==expected[rid]
            rows.append(dict(request_id=rid,matches=same,first_difference=mismatch,
                reference_token_ids=tokens,reference_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
        connection.send(dict(gpu=gpu,rows=rows))
    except BaseException:
        connection.send(dict(error=traceback.format_exc()));raise
    finally:
        if engine is not None:engine.engine_core.shutdown()
        connection.close()


def verify_records(args):
    saved=json.loads(Path(args.verify_records).read_text())
    assert saved['config']['batch_invariant'],'Exact-token verification requires the same batch-invariant configuration'
    by_id={c['request_id']:c for c in saved['checkpoints']}
    expected={r['original_request_id']:r['continuation_token_ids'] for r in saved['recovery_requests']}
    partitions=[[] for _ in args.gpus]
    for row in saved['recovery_requests']:
        partitions[(row['target_worker']-1)%len(partitions)].append(by_id[row['original_request_id']])
    ctx=mp.get_context('spawn');processes=[];connections=[];results=[]
    try:
        for gpu,checkpoints in zip(args.gpus,partitions):
            if not checkpoints:continue
            parent,child=ctx.Pipe()
            process=ctx.Process(target=reference_worker,args=(gpu,saved['config'],checkpoints,expected,child))
            process.start();child.close();processes.append(process);connections.append(parent)
        for process,connection in zip(processes,connections):
            if not connection.poll(1200):raise TimeoutError('E0 reference recomputation timed out')
            result=connection.recv()
            if result.get('error'):raise RuntimeError(result['error'])
            results.extend(result['rows']);process.join(60);assert process.exitcode==0
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(15)
        for connection in connections:connection.close()
    assert len(results)==len(expected)
    Path(args.output).write_text(json.dumps(dict(source=args.verify_records,requests=len(results),
        matched=sum(r['matches'] for r in results),rows=results)))
    if not all(r['matches'] for r in results):
        raise AssertionError('Online restore differs from independent E0 reference: '+
            str([(r['request_id'],r['first_difference']) for r in results if not r['matches']]))
    print(f'Independent E0 reference: {len(results)}/{len(results)} exact-token continuations PASS',flush=True)


def online_arguments(gpus):
    return ['--gpus',*map(str,gpus),'--model','/data1/jhcheng/models/scchen/Qwen3-8B',
        '--trace','data/SQuALITY/processed_all/squality_requests.jsonl',
        '--survivor-trace','data/qwen_traceB_blksz_16.jsonl','--source-qps','1.5','--survivor-qps','.5',
        '--qps','3','--arrival','constant','--trace-order','file','--recovery-mode','resume',
        '--recovery-scheduler','kv-aware','--recovery-order','group-first','--warmup','60',
        '--failure-delay','10','180','--window','60','--max-duration','300','--recovery-drain','600',
        '--gpu-memory-utilization','.90','--max-num-seqs','256','--batch-tokens','8192',
        '--ingress-policy','bounded','--seed','20261022','--record-recovery-tokens']


def async_profile_suite(args):
    """One attempt per async policy; mismatches are results, never paired winners."""
    from failure_recovery.report import write_async_profile_suite
    if not args.reference_dir or not args.full_budget_bytes or not args.half_budget_bytes:
        raise ValueError('Async profiling requires an existing reference and exact full/half byte budgets')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use a new output directory')
    reference=Path(args.reference_dir).resolve()
    if len(args.gpus)!=4 or len(list(reference.glob('worker_*.json')))!=4:
        raise ValueError('Four GPUs and the original four inference journals are required')
    private=ROOT/'backups'/(output.name+'_profiles');private.mkdir()
    env=dict(os.environ,VLLM_BATCH_INVARIANT='1');runs=[];baseline=None;baseline_details=None
    cases=[('E0_none','none',0),('E1_full','full',args.full_budget_bytes),
           ('E2_prefix','prefix-first',args.half_budget_bytes),('E3_suffix','suffix-first',args.half_budget_bytes)]
    for case,policy,budget in cases:
        profile_dir=private/case;log=private/(case+'.log')
        command=[sys.executable,'-u','-m','failure_recovery.main',*online_arguments(args.gpus),
            '--backup-policy',policy,'--cpu-kv-backup-bytes',str(budget),'--output',str(output/case),
            '--inference-replay','replay','--inference-replay-dir',str(reference),
            '--replay-backup-mode','async','--profile-backup-overhead','--profile-dir',str(profile_dir)]
        command+=['--manifest-mode',args.manifest_mode]
        if args.manifest_debug_oracle:command+=['--manifest-debug-oracle']
        if baseline is not None:command+=['--replay-diagnostics-dir',str(private/'E0_none')]
        print('ASYNC PROFILE',case,'log',log,flush=True)
        with log.open('w') as stream:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        profiles=[json.loads(p.read_text()) for p in sorted(profile_dir.glob('worker_?.json'))]
        source=next((p for p in profiles if p['worker']==0),None)
        if source is None:raise RuntimeError(f'{case} has no source profile; inspect {log}')
        divergence=next((p['divergence'] for p in profiles if p.get('divergence')),None)
        passed=code==0 and len(profiles)==4 and all(p['status']=='fault_verified' for p in profiles)
        raw=Path('/tmp')/(output.name+'_'+case+'_records.json')
        record=json.loads(raw.read_text()) if passed else None
        if passed:shutil.copy2(raw,private/(case+'_records.json'))
        if case=='E0_none':
            if not passed:raise RuntimeError('The original E0 reference failed; no valid profiling baseline')
            baseline=record;baseline_details=json.loads((profile_dir/'worker_0_steps.json').read_text())
            actual=base=source['clock'];boundary='完整参考故障现场'
        elif passed:
            assert [(s['worker'],s['state_signature'],s['gpu_cache_signature']) for s in record['plan']['prefailure']]==[
                (s['worker'],s['state_signature'],s['gpu_cache_signature']) for s in baseline['plan']['prefailure']]
            expected={r['original_request_id']:r['continuation_token_ids'] for r in baseline['recovery_requests']}
            assert all(r['continuation_token_ids']==expected[r['original_request_id']] for r in record['recovery_requests'])
            actual=source['clock'];base=runs[0]['source']['clock'];boundary='完整参考故障现场'
        else:
            details=json.loads((profile_dir/'worker_0_steps.json').read_text())
            own=source.get('divergence')
            detail=(own.get('actual_detail') if own and own.get('kind')=='forward' else
                    details[min(source['verified_steps'],len(details))-1] if details else None)
            if detail is None:raise RuntimeError(f'{case} failed before a comparable native boundary; inspect {log}')
            index=detail['step'];actual=detail['before_forward'];base=baseline_details[index]['before_forward']
            boundary=f'第{index}步前（此前{index}步匹配）'
        cpu=source.get('cpu_backup')
        if cpu:assert cpu['replay_copy_wait_ms']==0.,'Async case accidentally waited for D2H'
        parent=profile_dir/'coordinator.json'
        run=dict(case=case.split('_')[0],status='通过' if passed else '分歧' if divergence else '异常',
            source=source,profile_dir=str(profile_dir),log=str(log),actual_cut=actual,baseline_cut=base,
            boundary=boundary,divergence=divergence,
            coordinator=json.loads(parent.read_text()) if parent.exists() else None,
            verification=(f"{record['summary']['completed']}/{record['summary']['requests']} 完整恢复输出与E0一致。" if passed else ''))
        runs.append(run);write_async_profile_suite(output,runs)
        print('RESULT',case,run['status'],'verified source steps',source['verified_steps'],
              'physical seconds',round(source['clock']['wall_ms']/1000,3),flush=True)
        if code and not divergence:raise RuntimeError(f'{case} stopped for a non-replay error; inspect {log}')
    (private/'suite.json').write_text(json.dumps(runs,ensure_ascii=False))
    print('Async profiling suite complete:',output,flush=True)


def online_suite(args):
    """Each policy gets a fresh online run/pool; verification gates the next case."""
    from failure_recovery.report import write_selective_suite
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if len(args.gpus)!=4 or not args.full_budget_bytes or not args.half_budget_bytes:
        raise ValueError('Online suite requires four GPUs and exact full/half budgets')
    cases=[('E0','none',0,'E0_none'),('E1','full',args.full_budget_bytes,'E1_full'),
           ('E2','prefix-first',args.half_budget_bytes,'E2_prefix'),
           ('E3','suffix-first',args.half_budget_bytes,'E3_suffix')]
    common=online_arguments(args.gpus)
    env=dict(os.environ,VLLM_BATCH_INVARIANT='1');runs=[]
    strict=getattr(args,'strict_replay',False)
    reference=ROOT/'backups'/(output.name+'_inference_replay')
    if strict:
        reference.mkdir(parents=True,exist_ok=True)
        print('STRICT inference reference:',reference,'; CPU pools/DMA remain independent',flush=True)
    inference_state=None;cohort_state=None
    try:
        for case,policy,budget,directory in cases:
            path=output/directory
            raw=Path('/tmp')/(output.name+'_'+directory+'_records.json')
            log=Path('/tmp')/(output.name+'_'+directory+'.log')
            if not (args.resume_suite and raw.exists()):
                command=[sys.executable,'-u','-m','failure_recovery.main',*common,
                    '--backup-policy',policy,'--cpu-kv-backup-bytes',str(budget),'--output',str(path)]
                if strict:command+=['--inference-replay','record' if case=='E0' else 'replay',
                                    '--inference-replay-dir',str(reference)]
                print('ONLINE',case,'budget_bytes',budget,'log',log,flush=True)
                with log.open('w') as stream:
                    code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                if code:raise RuntimeError(f'{case} online run failed; see {log}')
            record=json.loads(raw.read_text());summary=record['summary'];config=record['config']
            assert summary['status']=='complete' and not summary['traffic']['unfinished']
            assert config['gpus']==args.gpus and config['batch_invariant'] and config['recovery_mode']=='resume'
            assert config['recovery_scheduler']=='kv-aware' and config['recovery_order']=='group-first'
            assert abs(config['cpu_kv_backup_gb']*1e9-budget)<1
            if strict:
                assert config.get('inference_replay')==('record' if case=='E0' else 'replay')
                before=record['plan']['prefailure']
                assert len(before)==4 and all(row['verified'] for row in before)
                state=[(r['worker'],r['steps'],r['state_signature'],r['gpu_cache_signature']) for r in before]
                cohort=sorted((r['request_id'],r['source_computed_tokens'],r['generated_token_ids'])
                              for r in record['checkpoints'])
                if inference_state is None:inference_state,cohort_state=state,cohort
                assert state==inference_state and cohort==cohort_state,'Backup policies changed the fault/background state'
            if case!='E0':
                source=summary['cpu_kv']['source_backup'];block_bytes=source['block_bytes']
                assert args.full_budget_bytes%block_bytes==0
                assert args.half_budget_bytes==(args.full_budget_bytes//block_bytes//2)*block_bytes
                assert source['capacity_bytes']==budget and source['allocated_bytes']<=budget
                if case=='E1':assert source['allocation_misses']==0,'B_full was insufficient'
                else:assert source['selection_calls']>0
            run=dict(case=case,directory=directory,records=record,verified=case=='E0')
            runs.append(run);write_selective_suite(output,runs)
            if case!='E0':
                verified=Path('/tmp')/(output.name+'_'+directory+'_verification.json')
                verify_log=Path('/tmp')/(output.name+'_'+directory+'_verification.log')
                if strict:
                    expected={r['original_request_id']:r['continuation_token_ids']
                              for r in runs[0]['records']['recovery_requests']}
                    checks=[dict(request_id=r['original_request_id'],
                        matches=r['continuation_token_ids']==expected[r['original_request_id']],
                        reference_token_ids=expected[r['original_request_id']]) for r in record['recovery_requests']]
                    check=dict(source=str(output/'E0_none'),requests=len(checks),
                        matched=sum(r['matches'] for r in checks),rows=checks)
                    verified.write_text(json.dumps(check))
                    verify_log.write_text(f"Strict same-scene E0 reference: {check['matched']}/{check['requests']} exact continuations\n")
                    assert all(r['matches'] for r in checks),'Strict backup recovery differs from same-scene E0'
                elif not(args.resume_suite and verified.exists()):
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--gpus',*map(str,args.gpus[:3]),
                        '--verify-records',str(raw),'--output',str(verified)]
                    print('VERIFY',case,'log',verify_log,flush=True)
                    with verify_log.open('w') as stream:
                        code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                    if code:raise RuntimeError(f'{case} differs from E0 reference; see {verify_log}')
                check=json.loads(verified.read_text())
                assert check['matched']==check['requests']==summary['requests']
                run['verified']=True
                target=path/'after_recovery.md'
                if '## 独立E0全重算校验' not in target.read_text():
                    with target.open('a') as stream:stream.write(
                        f"\n## 独立E0全重算校验\n\n{check['matched']}/{check['requests']} 个请求的全部续接Token IDs一致。"
                        + ('四组故障与幸存缓存状态均已严格核验，直接与同现场E0的实际恢复输出比较。'
                           if strict else '使用本组各请求自己的故障进度，未假设与在线E0的故障集合相同。')
                        + f"记录：`{verified}`。\n")
            write_selective_suite(output,runs)
            print('PASS',case,'requests',summary['requests'],'TTFO',round(summary['avg_ttfo_ms'],2),flush=True)
        print('Online E0/E1/E2/E3 suite complete:',output,flush=True)
    except BaseException as exc:
        write_selective_suite(output,runs,error=str(exc));raise


def check_physical_pages(transfer, blocks, slots):
    """Compare physical page order, independent of the logical KV view strides."""
    import torch
    capacity=transfer.size//transfer.block_bytes
    raw=torch.frombuffer(transfer.memory,dtype=torch.uint8,count=capacity*transfer.block_bytes)
    raw=raw.reshape(capacity,transfer.block_bytes)
    gpu_ids=torch.tensor([b.block_id for b in blocks],device='cuda',dtype=torch.int64)
    cpu_ids=torch.tensor(slots,dtype=torch.int64)
    for tensor,offset,layout in zip(transfer.views,transfer.offsets,transfer.layout):
        physical=tensor.as_strided((tensor.shape[0],layout[1]//tensor.element_size()),(tensor.stride(0),1))
        actual=physical.index_select(0,gpu_ids).contiguous().view(torch.uint8).cpu()
        expected=raw[:,offset:offset+layout[1]].index_select(0,cpu_ids)
        if not torch.equal(actual,expected):
            raise AssertionError(f'KV physical bytes differ: {layout[0]}')
    return len(slots)*transfer.block_bytes


def gpu_worker(gpu,handle,connection,rows,length,data=None,policy='full',fault_generated=8,check_parallel=False,trace_dir=None):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_BATCH_INVARIANT='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_LOGGING_LEVEL='ERROR',
        VLLM_NO_USAGE_STATS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for name in list(os.environ):
        if name.startswith(('KV_P1_','KV_P2_','KV_P3_','KV_PREFILL_','P1_KV_')):del os.environ[name]
    engine=transfer=backup=None
    try:
        import torch
        from vllm import EngineArgs,SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        from failure_recovery.cpu_kv import KVTransfer,SourceKVBackup
        engine=LLMEngine.from_engine_args(EngineArgs(model='/data1/jhcheng/models/scchen/Qwen3-8B',
            max_model_len=12000,max_num_seqs=8,max_num_batched_tokens=8192,
            num_gpu_blocks_override=800,block_size=16,gpu_memory_utilization=.5,
            enable_prefix_caching=True,enable_chunked_prefill=True,async_scheduling=False,
            seed=20261019),enable_multiprocessing=False)
        scheduler=engine.engine_core.engine_core.scheduler
        runner=engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        params=lambda n:SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,detokenize=False)
        def step():
            with contextlib.redirect_stdout(io.StringIO()):return engine.step()
        def drain():
            while engine.has_unfinished_requests():step()
        def reset():
            assert engine.engine_core.reset_prefix_cache(reset_connector=True)
        engine.add_request('initialization',{'prompt_token_ids':[10]*128},params(2));drain();reset()
        transfer=KVTransfer(engine,scheduler,handle)
        if data is None:
            backup=SourceKVBackup(transfer,policy,
                queue_config=dict(survivors=[1,2,3]) if policy=='group-prefix-queue' else None)
            common=rows[0]['prompt_token_ids'][:32]
            prompts={r['request_id']:common+r['prompt_token_ids'][32:length-64]+r['prompt_token_ids'][-64:] for r in rows}
            internal={rid:engine.add_request(rid,{'prompt_token_ids':tokens},params(64)) for rid,tokens in prompts.items()}
            outputs={rid:[] for rid in prompts};snapshots=None;source_checks={};step_id=0
            while engine.has_unfinished_requests():
                if not backup.frozen:
                    backup.poll()
                    if all(len(tokens)>=fault_generated for tokens in outputs.values()):
                        frozen=backup.freeze(list(internal.values()))
                        snapshots={rid:dict(frozen['requests'][iid],prompt=prompts[rid],
                            progress=list(outputs[rid]),computed_tokens=scheduler.requests[iid].num_computed_tokens)
                            for rid,iid in internal.items()}
                        for rid,iid in internal.items():
                            refs=snapshots[rid]['blocks'];slots=[b['cpu_slot'] for b in refs]
                            owned=scheduler.kv_cache_manager.get_blocks(iid).blocks[0]
                            blocks=[owned[b['logical_block']] for b in refs]
                            source_checks[rid]=check_physical_pages(transfer,blocks,slots) if slots else 0
                        freeze_step=step_id
                for output in step():
                    if output.outputs:outputs[output.request_id]=list(output.outputs[0].token_ids)
                step_id+=1
                if not backup.frozen:backup.submit()
            assert snapshots is not None
            stats=frozen['stats']
            assert stats['allocation_misses']==0
            connection.send(dict(checkpoints=snapshots,reference=outputs,source_checks=source_checks,
                source_stats=stats,freeze_step=freeze_step,block_bytes=transfer.block_bytes,length=length,
                slot_directory=frozen['slot_directory'],backup_policy=policy))
        else:
            rid=rows[0]['request_id'];checkpoint=data['checkpoints'][rid]
            prompt=checkpoint['prompt']+checkpoint['progress']
            expected=data['reference'][rid][len(checkpoint['progress']):]
            original_schedule=scheduler.schedule
            original_clip=scheduler._reserve_prefill_lookahead
            results=[]
            recovery_case={'full':'E1','prefix-first':'E2','suffix-first':'E3'}[policy]
            cases=('E0',recovery_case,'E3_parallel') if check_parallel else ('E0',recovery_case)
            for case in cases:
                # Warm the exact compute shape without CPU imports, then clear
                # APC. CPU registration/copy preparation stays in measured E1.
                reset()
                refs=checkpoint['blocks']
                begin=refs[0]['logical_block']*16 if refs else 0
                end=(refs[-1]['logical_block']+1)*16 if refs else 0
                if case!='E0' and begin:
                    engine.add_request('gap_warmup',{'prompt_token_ids':prompt[:begin]},params(1));drain();reset()
                if case!='E0' and refs:
                    seed=prompt[:end]
                    engine.add_request('shape_seed',{'prompt_token_ids':seed},params(1));drain()
                engine.add_request('shape_warmup',{'prompt_token_ids':prompt},params(2));drain();reset()
                connection.send(dict(kind='case_ready',case=case,request_id=rid))
                command=connection.recv()
                start=command['start'];time.sleep(max(0.,start-time.perf_counter()))
                iid=engine.add_request('REC_'+case+'_'+rid,{'prompt_token_ids':prompt},params(len(expected)))
                info=dict(cpu_loaded_tokens=0,cpu_load_bytes=0,cpu_load_ms=0.)
                steps=[]
                def clip(request,computed,count):
                    return transfer.clip_prefill(request,computed,original_clip(request,computed,count))
                def schedule(*args,**kwargs):
                    transfer.advance_restores()
                    scheduled=original_schedule(*args,**kwargs)
                    if iid in scheduled.num_scheduled_tokens:
                        count=scheduled.num_scheduled_tokens[iid]
                        before=scheduler.requests[iid].num_computed_tokens-count
                        steps.append(dict(before=before,count=count,prefill=min(count,max(0,len(prompt)-before))))
                    transfer.release_imports()
                    return scheduled
                scheduler.schedule=schedule
                scheduler._reserve_prefill_lookahead=clip
                timing={}
                if case=='E3_parallel':
                    def head_step():
                        before=len(steps);out=step()
                        assert all(not r.outputs or not r.outputs[0].token_ids for r in out)
                        return dict(head_tokens=sum(r['prefill'] for r in steps[before:]),
                                    scheduled_tokens=sum(r['count'] for r in steps[before:]))
                    if trace_dir:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                            timing=transfer.restore_cohort([(scheduler.requests[iid],checkpoint,info)],'suffix-overlap',head_step)
                        prof.export_chrome_trace(str(Path(trace_dir)/f'overlap_gpu_{gpu}.json'))
                    else:timing=transfer.restore_cohort([(scheduler.requests[iid],checkpoint,info)],'suffix-overlap',head_step)
                elif case!='E0':
                    info.update(transfer.restore(scheduler.requests[iid],checkpoint,record=info))
                    timing=transfer.preload_timing
                if case!='E0' and refs:
                    request=scheduler.requests[iid];pool=scheduler.kv_cache_manager.block_pool
                    detached=[pool.get_cached_block(request.block_hashes[r['logical_block']],[0])[0] for r in refs]
                    assert info['preload_complete']
                    assert info['cpu_loaded_tokens']+info['gpu_reused_selected_tokens']==len(refs)*16
                    assert check_physical_pages(transfer,detached,[r['cpu_slot'] for r in refs])==info['cpu_load_bytes']
                    assert request.num_computed_tokens<=begin if case=='E3_parallel' else request.num_computed_tokens==0
                hits,cached,_=scheduler.kv_cache_manager.get_computed_blocks(scheduler.requests[iid])
                loaded=list(hits.blocks[0])
                generated=[];first=None;gpu_table_ok=False
                while engine.has_unfinished_requests():
                    actual_outputs=step();observed=time.perf_counter()
                    for output in actual_outputs:
                        if output.outputs:
                            generated=list(output.outputs[0].token_ids)
                            if generated and first is None:
                                first=observed
                                if case!='E0' and refs:
                                    all_blocks=scheduler.kv_cache_manager.get_blocks(iid).blocks[0]
                                    loaded=list(all_blocks[begin//16:end//16])
                                    gpu_table=runner.block_tables.input_block_tables[0][0,begin//16:end//16].cpu().tolist()
                                    assert gpu_table==[b.block_id for b in loaded]
                                gpu_table_ok=True
                scheduler.schedule=original_schedule
                scheduler._reserve_prefill_lookahead=original_clip
                assert generated==expected,(case,rid,'Continuation differs',next((i for i,(a,b) in enumerate(zip(generated,expected)) if a!=b),None))
                prefill=sum(s['prefill'] for s in steps)
                assert prefill==len(prompt)-info['cpu_loaded_tokens'],(case,rid,prefill,cached,len(prompt),info)
                if case!='E0' and refs:
                    assert info['cpu_loaded_tokens']==len(refs)*16
                    assert check_physical_pages(transfer,loaded,[b['cpu_slot'] for b in refs])==info['cpu_load_bytes']
                actual_apc=info.get('native_apc_tokens_after_join') or cached
                result=dict(case=case,request_id=rid,gpu=gpu,ttfo_ms=1000*(first-start),
                    native_apc_tokens=actual_apc,first_schedule_apc_tokens=cached,
                    prefill_tokens=prefill,**info,transfer_timing=timing,
                    continued_token_ids=generated,continuation_matches=True,physical_bytes_match=True,
                    gpu_table_matches=gpu_table_ok,steps=[s for s in steps if s['prefill']])
                results.append(result)
                print(case,rid,'TTFO',round(result['ttfo_ms'],2),'APC',actual_apc,'prefill',prefill,'PASS',flush=True)
                connection.send(dict(kind='case_result',result=result))
            connection.send(dict(kind='done',results=results))
    except BaseException:
        connection.send(dict(error=traceback.format_exc()));raise
    finally:
        if backup is not None:backup.close()
        if transfer is not None:transfer.close()
        if engine is not None:engine.engine_core.shutdown()
        connection.close()


def validate(args):
    from failure_recovery.report import _failure_table as table
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing to overwrite existing reports')
    output.mkdir(parents=True,exist_ok=True)
    rows=[];seen_docs=set()
    with (ROOT/'data/SQuALITY/processed_all/squality_requests.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if len(row['prompt_token_ids'])>=args.length and row['doc_id'] not in seen_docs:
                rows.append(row);seen_docs.add(row['doc_id'])
            if len(rows)==args.requests:break
    assert len(rows)==args.requests and len(args.gpus)>=args.requests+1
    ctx=mp.get_context('spawn');children=[];connections=[]
    with SharedKVPool(2,args.budget_bytes) as pool:
        parent,child=ctx.Pipe()
        source=ctx.Process(target=gpu_worker,args=(args.gpus[0],pool.share(),child,rows,args.length,
            None,args.backup_policy,args.fault_generated))
        source.start();child.close()
        try:
            if not parent.poll(900):raise TimeoutError('Source validation timed out')
            data=parent.recv();source.join(60)
            if data.get('error'):raise RuntimeError(data['error'])
            assert source.exitcode==0
        finally:
            if source.is_alive():source.terminate();source.join(15)
            parent.close()
        try:
            from failure_recovery.scheduling import assign_recovery_requests,order_recovery_requests
            checkpoints=[dict(request_id=rid,source_index=i,input_length=len(c['prompt']),
                output_length=64,prompt_token_ids=c['prompt'],generated_token_ids=c['progress'])
                for i,(rid,c) in enumerate(data['checkpoints'].items())]
            assignment=assign_recovery_requests(checkpoints,list(range(1,args.requests+1)),
                'kv-aware','resume',16,1.2)
            assignment={w:order_recovery_requests(items,'group-first','resume') for w,items in assignment.items()}
            by_id={r['request_id']:r for r in rows}
            for worker in range(1,args.requests+1):
                assert len(assignment[worker])==1,('Fixture must distribute one request per survivor',
                    {w:[r['request_id'] for r in items] for w,items in assignment.items()})
                gpu=args.gpus[worker];row=by_id[assignment[worker][0]['request_id']]
                parent,child=ctx.Pipe()
                target=ctx.Process(target=gpu_worker,args=(gpu,pool.share(),child,[row],args.length,data,
                    args.backup_policy,args.fault_generated,args.check_suffix_parallel,
                    str(output) if args.check_suffix_parallel and worker==1 else None))
                target.start();child.close();children.append(target);connections.append(parent)
            results=[]
            recovery_case={'full':'E1','prefix-first':'E2','suffix-first':'E3'}[args.backup_policy]
            cases=('E0',recovery_case,'E3_parallel') if args.check_suffix_parallel else ('E0',recovery_case)
            for case in cases:
                for connection in connections:
                    if not connection.poll(900):raise TimeoutError('Target startup timed out')
                    ready=connection.recv()
                    if ready.get('error'):raise RuntimeError(ready['error'])
                    assert ready['kind']=='case_ready' and ready['case']==case
                start=time.perf_counter()+.1
                for connection in connections:connection.send(dict(start=start))
                for connection in connections:
                    if not connection.poll(900):raise TimeoutError('Recovery validation timed out')
                    result=connection.recv()
                    if result.get('error'):raise RuntimeError(result['error'])
                    results.append(result['result'])
            for connection,target in zip(connections,children):
                assert connection.recv()['kind']=='done'
                target.join(60);assert target.exitcode==0
        finally:
            for target in children:
                if target.is_alive():target.terminate();target.join(15)
            for connection in connections:connection.close()
    raw=Path('/tmp')/(output.parent.name+'_'+output.name+'_records.json')
    raw.write_text(json.dumps(dict(source=data,results=results)))
    before='# 固定故障现场：E0全重算与CPU备份恢复\n\nQwen3-8B、TP=1、16-token块、batch=8192、VLLM_BATCH_INVARIANT=1。源退出后CPU RAM由父进程保留。取不同SQuALITY文档片段及问题，统一前32 tokens检验共享去重。固定生成进度，非在线QPS性能结果。\n\n'+table(
        ['请求','原prompt','故障已生成','已计算KV','CPU READY tokens'],[
        [rid,len(c['prompt']),len(c['progress']),c['computed_tokens'],len(c['blocks'])*16] for rid,c in data['checkpoints'].items()])
    source_stats=data['source_stats']
    before+=f"\n\n在线策略：{args.backup_policy}；池容量 {source_stats['capacity_blocks']} Slots / {source_stats['capacity_bytes']} bytes；"
    before+=f"READY唯一块 {source_stats['ready_unique_blocks']}；PENDING {source_stats['pending_blocks']}；淘汰 {source_stats['evictions']}。"
    schedule='# 恢复与加载\n\n使用原有kv-aware/group-first分配和排序；该小型现场将请求分散到幸存GPU。两种模式使用相同输入、生成进度和CPU快照。E0只作为全重算参考，不导入KV；E1沿用旧版真实H2D/APC。核验物理字节和GPU前向块表，不拷贝故障后的源KV用于恢复。\n\n'+table(
        ['请求','目标GPU','E1备份区间','实际H2D字节'],[
        [r['request_id'],r['gpu'],(f"[{data['checkpoints'][r['request_id']]['blocks'][0]['logical_block']*16},"
          f"{(data['checkpoints'][r['request_id']]['blocks'][-1]['logical_block']+1)*16})"
          if data['checkpoints'][r['request_id']]['blocks'] else '空'),r['cpu_load_bytes']]
          for r in results if r['case']!='E0'])
    after='# E0与备份恢复验收\n\n'+table(['组别','请求','TTFO ms','实际APC tokens','真实prefill tokens','输出/字节/块表校验'],[
        [r['case'],r['request_id'],f"{r['ttfo_ms']:.2f}",r['native_apc_tokens'],r['prefill_tokens'],'通过'] for r in results])
    after+='\n\nTTFO从协调端释放固定恢复现场计到首个新token；模型形状预热后清空APC。字节读回校验在生成完成后执行，不计入TTFO。\n\n'
    after+='后缀组的Join后APC连续长度包含本轮真实补算的前段，不等于CPU加载量；CPU加载量见H2D和备份区间。\n\n'
    after+=f'完整Token IDs、CPU统计与逐步prefill记录：`{raw}`。'
    for name,text in zip(('before_failure.md','failure_schedule.md','after_recovery.md'),(before,schedule,after)):
        (output/name).write_text(text+'\n')
    print('Validation complete:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,required=True)
    parser.add_argument('--requests',type=int,default=3)
    parser.add_argument('--length',type=int,default=1024)
    parser.add_argument('--output',required=True)
    parser.add_argument('--verify-records',help='Recompute an online run from its exact checkpoints and compare all token IDs')
    parser.add_argument('--backup-policy',choices=['full','prefix-first','suffix-first'],default='full')
    parser.add_argument('--budget-bytes',type=int)
    parser.add_argument('--fault-generated',type=int,default=8)
    parser.add_argument('--check-suffix-parallel',action='store_true',help='Compare serial/parallel using the same frozen CPU KV, and trace one GPU')
    parser.add_argument('--run-online-suite',action='store_true')
    parser.add_argument('--run-async-profile-suite',action='store_true')
    parser.add_argument('--reference-dir',help='Existing E0 journals; async profiling never records a new fault scene')
    parser.add_argument('--manifest-mode',choices=['delta','legacy'],default='delta')
    parser.add_argument('--manifest-debug-oracle',action='store_true')
    parser.add_argument('--resume-suite',action='store_true')
    parser.add_argument('--strict-replay',action='store_true',
                        help='Record E0 native inference once, independently back up and verify the same scene in E1/E2/E3')
    parser.add_argument('--full-budget-bytes',type=int)
    parser.add_argument('--half-budget-bytes',type=int)
    args=parser.parse_args()
    if not 128<=args.length<=11872 or args.length%16 or args.requests<1:
        parser.error('Use block-aligned length in [128,11872] and requests>=1')
    if not 2<=args.fault_generated<48:parser.error('Use fault progress in [2,48)')
    if args.check_suffix_parallel and args.backup_policy!='suffix-first':parser.error('Parallel gate requires suffix-first')
    if args.backup_policy!='full' and not args.budget_bytes:parser.error('Selective backup requires an exact budget')
    if args.run_async_profile_suite:async_profile_suite(args)
    elif args.run_online_suite:online_suite(args)
    elif args.verify_records:verify_records(args)
    else:validate(args)
