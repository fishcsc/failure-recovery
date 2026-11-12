"""CPU-only protocol tests, including four concurrent replicas and native-style queues."""
from collections import deque
import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from failure_sampling import (FailureCoordinator, PrefailureReplay, compare_recovery_schedulers, metric_summary, promote_recoveries,
                              recovery_assignment, recovery_input, recovery_queue,
                              replay_failure, update_latency,
                              forwarding_assignment, merge_pending, RESULT_FILES, observation_summary,
                              failure_stability)
from sample_running import parse_args, worker_workload, validate_replicas, main as sampling_main
from workload import load_probe_traces
from running_report import request_groups
from report import failure_markdown,write_failure_comparison
from recovery_scheduling import assign_prefill_balanced, assign_kv_aware, _shared_blocks, order_recovery_requests


class Queue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, request):
        self.appendleft(request)

    def remove_request(self, request):
        self.remove(request)


class Scheduler:
    def __init__(self):
        self.running, self.waiting, self.skipped_waiting, self.requests = [], Queue(), Queue(), {}
        self.prefill_chunk = None

    def get_request_counts(self):
        return len(self.running), len(self.waiting)+len(self.skipped_waiting)

    def _select_waiting_queue_for_scheduling(self):
        return self.skipped_waiting or self.waiting or None

    def schedule(self):
        while len(self.running) < 3:
            queue = self._select_waiting_queue_for_scheduling()
            if not queue:
                break
            self.running.append(queue.pop_request())
        allocation = {}
        for request in self.running:
            remaining = max(0,request.num_prompt_tokens-request.num_computed_tokens)
            allocation[request.request_id] = min(remaining,self.prefill_chunk or remaining) if remaining else 1
            request.num_computed_tokens += allocation[request.request_id]
        return SimpleNamespace(num_scheduled_tokens=allocation)


class Engine:
    def __init__(self):
        self.scheduler, self.prompts, self.salts = Scheduler(), {}, {}
        self.recovery_chunk = None

    def add_request(self, rid, prompt, params):
        if rid.startswith('REC') and self.recovery_chunk:
            self.scheduler.prefill_chunk = self.recovery_chunk
        internal = rid+'-native'
        request = SimpleNamespace(request_id=internal, external_id=rid,
                                  cache_salt=prompt.get('cache_salt'),
                                  num_prompt_tokens=len(prompt['prompt_token_ids']),
                                  num_computed_tokens=0, output_token_ids=[], max_tokens=params.max_tokens)
        self.prompts[rid] = list(prompt['prompt_token_ids'])
        self.salts[rid] = prompt.get('cache_salt')
        self.scheduler.requests[internal] = request
        self.scheduler.waiting.append(request)
        return internal

    def has_unfinished_requests(self):
        return bool(self.scheduler.requests)

    def step(self):
        scheduled = self.scheduler.schedule()
        time.sleep(.005)
        outputs = []
        for rid in scheduled.num_scheduled_tokens:
            request = self.scheduler.requests[rid]
            if request.num_computed_tokens<request.num_prompt_tokens:
                continue
            request.output_token_ids.append(len(request.output_token_ids)+1)
            finished = len(request.output_token_ids) == request.max_tokens
            outputs.append(SimpleNamespace(
                request_id=request.external_id, finished=finished,
                outputs=[SimpleNamespace(token_ids=list(request.output_token_ids),
                                         finish_reason='length' if finished else None)]))
            if finished:
                self.scheduler.running.remove(request)
                del self.scheduler.requests[rid]
        return outputs

    def abort_request(self, ids):
        self.scheduler.requests.clear()
        self.scheduler.running.clear()
        self.scheduler.waiting.clear()
        self.scheduler.skipped_waiting.clear()


class FailureTests(unittest.TestCase):
    def test_failure_requires_consecutive_stable_windows_and_rejects_queue_growth(self):
        args = parse_args(['--inject-failure', '--require-stable-before-failure'])
        arrivals = list(range(120))
        records = [dict(completed_s=i+.5) for i in arrivals]
        load = [dict(start_s=i, end_s=i+1, running=1, waiting=0, outstanding=1, ingress=0)
                for i in range(120)]
        self.assertTrue(failure_stability(args, load, arrivals, records, 120)['qualified'])
        growing = [dict(r, outstanding=r['start_s']/4) for r in load]
        proof = failure_stability(args, growing, arrivals, records, 120)
        self.assertFalse(proof['qualified'])
        self.assertFalse(proof['checks'][-1]['checks']['outstanding_level'])
        recovered = [dict(r, running=8 if r['start_s']<60 else 1) for r in load]
        proof = failure_stability(args, recovered, arrivals, records, 120)
        self.assertTrue(proof['checks'][-1]['stable'])
        self.assertFalse(proof['qualified'])
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parse_args(['--require-stable-before-failure'])

    def test_unstable_fault_aborts_before_dispatch_and_cleans_up_requests(self):
        args = parse_args(['--inject-failure', '--require-stable-before-failure',
            '--warmup', '.02', '--failure-delay', '.03', '.03', '--max-duration', '.30'])
        engine = Engine()
        requests = [dict(request_id='TA00000', source_index=0, input_length=3,
                         output_length=20, tokens=[7, 8, 9])]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,
                {'vllm': SimpleNamespace(SamplingParams=SimpleNamespace)}):
            with self.assertRaisesRegex(RuntimeError, 'not stable at failure'):
                replay_failure(engine, engine.scheduler, requests, [0.], args, 0,
                    args.gpus[0], Path(temporary), None, time.perf_counter())
        self.assertFalse(engine.has_unfinished_requests())

    def test_busy_failure_waits_for_running_then_signals_survivors(self):
        args = parse_args(['--inject-failure', '--failure-when-busy', '--warmup', '.02',
            '--failure-delay', '.03', '.03', '--max-duration', '.30'])
        args._failure_event = threading.Event()
        engine, captured = Engine(), {}
        connection = SimpleNamespace(send=lambda row: captured.update(row), poll=lambda _: True,
            recv=lambda: dict(action='fail', failure_s=captured['time_s']))
        requests = [dict(request_id='TA00000', source_index=0, input_length=3,
                         output_length=20, tokens=[7, 8, 9])]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,
                {'vllm': SimpleNamespace(SamplingParams=SimpleNamespace)}):
            replay_failure(engine, engine.scheduler, requests, [.08], args, 0,
                args.gpus[0], Path(temporary), connection, time.perf_counter())
        self.assertGreaterEqual(captured['time_s'], .08)
        self.assertEqual(captured['target_time_s'], .05)
        self.assertEqual(captured['running_ids'], ['TA00000'])
        self.assertTrue(args._failure_event.is_set())

    def test_background_stability_does_not_block_source_scoped_experiment(self):
        for scope in ('source','all'):
            args = parse_args(['--inject-failure', '--require-stable-before-failure',
                '--failure-stability-scope',scope,'--warmup','.02',
                '--failure-delay','.03','.03','--max-duration','.30'])
            captured = []
            commands = iter([dict(action='recover',failure_s=.05,checkpoints=[],forwarded=[]),
                             dict(action='resume')])
            connection = SimpleNamespace(send=captured.append,poll=lambda _:True,recv=lambda:next(commands))
            engine = Engine()
            requests = [dict(request_id='TB00000',source_index=0,input_length=3,
                             output_length=20,tokens=[7,8,9])]
            with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,
                    {'vllm':SimpleNamespace(SamplingParams=SimpleNamespace)}):
                if scope=='source':
                    replay_failure(engine,engine.scheduler,requests,[0.],args,1,args.gpus[1],
                                   Path(temporary),connection,time.perf_counter())
                    self.assertNotIn('stability',captured[0])
                else:
                    with self.assertRaisesRegex(RuntimeError,'not stable at failure'):
                        replay_failure(engine,engine.scheduler,requests,[0.],args,1,args.gpus[1],
                                       Path(temporary),connection,time.perf_counter())

    def test_queue_orders_preserve_placement_and_recompute_prefix_cost(self):
        prompts = [[1]*64+[2]*32,[1]*64+[3]*16,[1]*64+[4]*8,[5]*40,[6]*56]
        rows = [dict(recovery_id=str(i),request_id=str(i),target_worker=3,
                     prompt_token_ids=tokens,generated_token_ids=[],recovery_group='A' if i<3 else str(i))
                for i,tokens in enumerate(prompts)]
        interleaved = order_recovery_requests(rows,'interleave')
        self.assertEqual([r['recovery_id'] for r in interleaved],['0','3','1','4','2'])
        shortest = order_recovery_requests(rows,'prefix-shortest')
        self.assertEqual([r['recovery_id'] for r in shortest],['3','4','2','1','0'])
        self.assertEqual([r['estimated_prefill_tokens'] for r in shortest],[40,56,72,16,32])
        self.assertTrue(all(r['target_worker']==3 for r in shortest))
        self.assertEqual([r['queue_rank'] for r in shortest],list(range(5)))
        # 组外和没有组号的请求也在同一前缀池，估算不得按组隔离。
        for ungrouped in ([dict(r,recovery_group=r['recovery_id']) for r in rows],
                          [{k:v for k,v in r.items() if k!='recovery_group'} for r in rows]):
            shared = order_recovery_requests(ungrouped,'prefix-shortest')
            self.assertEqual([r['estimated_prefill_tokens'] for r in shared],[40,56,72,16,32])
        self.assertEqual(rows[0].get('queue_rank'),None)
        self.assertEqual([r['recovery_id'] for r in order_recovery_requests(rows[:3],'interleave')],['0','1','2'])
        first = order_recovery_requests(rows,'group-first')
        self.assertEqual([r['recovery_id'] for r in first],['2','1','0','3','4'])
        ratio = order_recovery_requests(rows,'group-ratio')
        self.assertEqual([r['recovery_id'] for r in ratio],['3','2','1','0','4'])
        self.assertTrue(all(r['target_worker']==3 for r in first+ratio))
        self.assertEqual(sum(r['estimated_prefill_tokens'] for r in first),216)
        with self.assertRaises(ValueError):
            order_recovery_requests(rows,'unknown')

    @staticmethod
    def checkpoints(prompts):
        return [dict(request_id=f'TA{i}', prompt_token_ids=tokens,
                     input_length=len(tokens), generated_token_ids=[9]*7, output_length=1000-i)
                for i,tokens in enumerate(prompts)]

    def test_prefill_lpt_balances_added_input_independently_of_decode(self):
        rows = self.checkpoints([[i]*n for i,n in enumerate([100,90,80,70,60,50])])
        assigned = assign_prefill_balanced(rows,[1,2,3])
        self.assertEqual([sum(r['estimated_prefill_tokens'] for r in assigned[w]) for w in assigned],
                         [150,150,150])
        self.assertEqual(assigned,assign_prefill_balanced(rows,[1,2,3]))
        self.assertEqual(sum(r['prefill_tokens'] for items in
            assign_prefill_balanced(rows,[1,2,3],mode='resume').values() for r in items),492)
        self.assertEqual([sum(r['estimated_prefill_tokens'] for r in items) for items in
                          recovery_assignment(rows,[1,2,3]).values()],[170,150,130])
        shuffled = self.checkpoints([[i]*n for i,n in enumerate([50,100,60,90,70,80])])
        for items in assign_prefill_balanced(shuffled,[1,2,3]).values():
            self.assertEqual([int(r['request_id'][2:]) for r in items],
                             sorted(int(r['request_id'][2:]) for r in items))

    def test_kv_group_uses_only_full_prefix_blocks_and_recomputes_last_token(self):
        self.assertEqual(_shared_blocks([1]*32,[1]*32),16)
        self.assertEqual(_shared_blocks([1]*33,[1]*33),32)
        self.assertEqual(_shared_blocks([1]*32,[1]*32+[2]*16),32)
        self.assertEqual(_shared_blocks([1]*32+[2]*16,[1]*32),16)
        self.assertEqual(_shared_blocks([1]*31+[2]*10,[1]*31+[3]*10),16)
        self.assertEqual(_shared_blocks([2]+[1]*64,[3]+[1]*64),0)
        rows = self.checkpoints([[1]*64+[2]*16,[1]*64+[3]*16])
        assigned = assign_kv_aware(rows,[1,2,3])
        group = assigned[1]
        self.assertEqual([len(v) for v in assigned.values()],[2,0,0])
        self.assertEqual([r['estimated_prefill_tokens'] for r in group],[80,16])
        self.assertEqual(group[1]['shared_prefix_tokens'],64)
        self.assertEqual(group[0]['recovery_group'],group[1]['recovery_group'])
        three = assign_kv_aware(self.checkpoints([[1]*240+[i]*16 for i in (2,3,4)]),[1,2,3])[1]
        self.assertEqual([r['shared_prefix_tokens'] for r in three],[0,240,240])
        self.assertTrue(all('prefill_after' not in r for r in three))

    def test_kv_overweight_groups_split_and_preserve_all_requests(self):
        # Identical prompts still recompute their final block. A large group must split.
        rows = self.checkpoints([[1]*64]*30)
        assigned = assign_kv_aware(rows,[1,2,3])
        self.assertTrue(all(assigned.values()))
        self.assertEqual(assigned,assign_kv_aware(rows,[1,2,3]))
        placed = [r for items in assigned.values() for r in items]
        self.assertEqual(len({r['recovery_id'] for r in placed}),30)
        self.assertEqual({r['request_id'] for r in placed},{r['request_id'] for r in rows})
        groups = {}
        for r in placed:
            groups.setdefault(r['recovery_group'],set()).add(r['target_worker'])
        self.assertTrue(all(len(workers)==1 for workers in groups.values()))
        # Branching after one common block must split into independent prefix groups.
        branches = self.checkpoints([[1]*16+[i]*64 for i in range(12)])
        split = assign_kv_aware(branches,[1,2,3])
        self.assertEqual([len(v) for v in split.values()],[4,4,4])

    def test_kv_singletons_empty_inputs_and_invalid_settings(self):
        self.assertEqual(assign_kv_aware([],[1,2,3]),{1:[],2:[],3:[]})
        rows = self.checkpoints([[i]*80 for i in range(3)])
        self.assertEqual([len(v) for v in assign_kv_aware(rows,[1,2,3]).values()],[1,1,1])
        for kwargs in [dict(min_shared_tokens=17),dict(split_factor=float('nan')),
                       dict(split_factor=float('inf')),dict(split_factor=.9)]:
            with self.assertRaises(ValueError):
                assign_kv_aware(rows,[1,2,3],**kwargs)
        for survivors in [[],[1,1]]:
            with self.assertRaises(ValueError):
                assign_prefill_balanced(rows,survivors)
        with self.assertRaises(ValueError):
            recovery_assignment(rows,[1,2,3],strategy='unknown')
        with self.assertRaisesRegex(ValueError,'actual prompt'):
            assign_kv_aware([dict(request_id='empty',input_length=80)],[1,2,3])

    def test_scheduler_cli_uses_one_shared_cache_policy_for_all_strategies(self):
        args = parse_args(['--inject-failure','--recovery-scheduler','kv-aware'])
        self.assertEqual(args.recovery_cache_policy,'shared')
        self.assertEqual(args.recovery_order,'prefix-shortest')
        self.assertEqual(parse_args(['--inject-failure','--recovery-order','source']).recovery_order,'source')
        self.assertEqual(parse_args(['--inject-failure','--recovery-order','interleave']).recovery_order,'interleave')
        matrix = parse_args(['--inject-failure','--compare-recovery-schedulers',
                             '--comparison-orders','source','shortest'])
        self.assertEqual(matrix.comparison_orders,['source','prefix-shortest'])
        self.assertEqual(parse_args(['--inject-failure','--recovery-order','shortest']).recovery_order,'prefix-shortest')
        selected = parse_args(['--inject-failure','--compare-recovery-schedulers','--comparison-cases',
                              'prefill-balanced/shortest','kv-aware/shortest','kv-aware/group-first','kv-aware/group-ratio'])
        self.assertEqual(selected.comparison_cases,[('prefill-balanced','prefix-shortest'),('kv-aware','prefix-shortest'),
                                                   ('kv-aware','group-first'),('kv-aware','group-ratio')])
        for strategy in ('equal','prefill-balanced','kv-aware'):
            for trace in ([],['--survivor-trace','trace-B']):
                self.assertEqual(parse_args(['--inject-failure','--recovery-scheduler',strategy]+trace
                                            ).recovery_cache_policy,'shared')
        with patch('sys.stderr'):
            for argv in [['--recovery-scheduler','kv-aware'],
                         ['--inject-failure','--comparison-repeats','3'],
                         ['--inject-failure','--compare-recovery-schedulers','--comparison-repeats','0'],
                         ['--inject-failure','--compare-recovery-schedulers','--comparison-repeats','3',
                          '--failure-delay','50','50'],
                         ['--inject-failure','--recovery-scheduler','kv-aware','--recovery-cache-policy','cold'],
                         ['--inject-failure','--recovery-cache-policy','group'],
                         ['--inject-failure','--recovery-scheduler','prefill-balanced','--recovery-cache-policy','inherit'],
                         ['--inject-failure','--comparison-orders','source','shortest'],
                         ['--inject-failure','--compare-recovery-schedulers',
                          '--comparison-orders','shortest','prefix-shortest'],
                         ['--inject-failure','--comparison-cases','kv-aware/shortest'],
                         ['--inject-failure','--compare-recovery-schedulers','--comparison-cases','bad/shortest'],
                         ['--inject-failure','--compare-recovery-schedulers','--comparison-cases','kv-aware/group-first',
                          '--comparison-orders','source'],
                         ['--inject-failure','--recovery-order','group-first'],
                         ['--inject-failure','--compare-recovery-schedulers','--comparison-orders','group-ratio']]:
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_markdown_shows_fault_metrics_without_dumping_forwarded_records(self):
        report = dict(summary=dict(latency_measurement='restart_first_token_v2',requests=2,completed=1,
            avg_ttft_ms=0.,avg_tpot_ms=None,forwarding=dict(requests=123,completed=100)),
            recovery_requests=[dict(original_request_id='TA00001',target_gpu=3,ttft_ms=1234.567,
                tpot_ms=None,status='aborted_at_end'),dict(original_request_id='TA00002',target_gpu=5,
                ttft_ms=0.,tpot_ms=1.236,status='completed')],
            forwarded_requests=[dict(request_id='FWD0_verbose_not_for_report')],gpus=[])
        rendered = failure_markdown('after_recovery.md',report)
        self.assertIn('| TA00001 | 3 | 1234.57 | — | 未完成 |',rendered)
        self.assertIn('| TA00002 | 5 | 0.00 | 1.24 | 完成 |',rendered)
        self.assertIn('100/123',rendered)
        self.assertNotIn('FWD0_verbose',rendered)
        self.assertLess(len(rendered.splitlines()),40)
        report['summary']['latency_measurement']='restart_catchup_v1'
        self.assertIn('历史口径',failure_markdown('after_recovery.md',report))

    def test_shared_cache_hits_are_visible_for_every_strategy_without_group_labels(self):
        for strategy in ('equal','prefill-balanced','kv-aware'):
            report = dict(summary=dict(recovery_scheduler=strategy,recovery_cache_policy='shared',
                latency_measurement='restart_first_token_v2',requests=1,completed=1,
                gpu_cached_prompt_tokens=64,requests_with_gpu_cache_hits=1,cache_observed_requests=1),
                recovery_requests=[dict(original_request_id='TA',target_gpu=3,input_length=96,
                    gpu_cached_tokens_first_schedule=64,prefill_scheduled_tokens=32,queue_wait_ms=1.,
                    first_schedule_batch=1,first_token_batch=2,ttft_ms=3.,tpot_ms=1.,status='completed')])
            rendered = failure_markdown('after_recovery.md',report)
            self.assertIn('首次调度 GPU 前缀命中：64 tokens',rendered)
            self.assertIn('共享前缀请求证据',rendered)
            self.assertIn('| 64 / 96 | 32 |',rendered)

    def test_failure_dry_run_only_keeps_three_files_without_input_or_source_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings":128,"vocab_size":1000}')
            source,background=root/'traceA_blksz_16.jsonl',root/'traceB_blksz_16.jsonl'
            for trace,length in [(source,8),(background,16)]:
                trace.write_text(''.join(json.dumps(dict(timestamp=i,input_length=32,output_length=length,
                    hash_ids=[0,1]))+'\n' for i in range(4)))
            output=root/'results'
            argv=['sample_running.py','--inject-failure','--dry-run','--model',str(model),'--trace',str(source),
                  '--survivor-trace',str(background),'--warmup','.02','--failure-delay','.03','.03',
                  '--max-duration','.3','--output',str(output)]
            with patch('sys.argv',argv),patch('sys.stdout'):
                sampling_main()
            self.assertEqual({p.name for p in output.iterdir()},set(RESULT_FILES))
            before=(output/'before_failure.md').read_text()
            self.assertIn('尚无测量数据',before)
            self.assertIn(model.name,before)

    def test_startup_error_is_saved_in_third_file_without_extra_error_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'results'
            argv=['sample_running.py','--inject-failure','--model',str(Path(temporary)/'missing'),
                  '--output',str(output)]
            with patch('sys.argv',argv),self.assertRaises(FileNotFoundError):
                sampling_main()
            self.assertEqual({p.name for p in output.iterdir()},set(RESULT_FILES))
            after=(output/'after_recovery.md').read_text()
            self.assertIn('运行失败',after)
            self.assertIn('FileNotFoundError',after)

    def test_compact_observations_clip_intervals_and_do_not_confuse_waiting_with_full_batches(self):
        loads = [dict(start_s=0.,end_s=2.,running=2,waiting=1,ingress=0,outstanding=3),
                 dict(start_s=2.,end_s=5.,running=4,waiting=0,ingress=0,outstanding=4)]
        batches = [dict(time_s=1.,tokens=8192,full=True,mixed=True,kv_usage=.5),
                   dict(time_s=3.,tokens=4,full=False,mixed=False,kv_usage=1.)]
        result=observation_summary(loads,batches,1.,4.)
        self.assertEqual(result['recorded_seconds'],3.)
        self.assertAlmostEqual(result['mean_running'],10/3)
        self.assertAlmostEqual(result['mean_waiting'],1/3)
        self.assertAlmostEqual(result['waiting_positive_time_fraction'],1/3)
        self.assertEqual(result['max_waiting'],1)
        self.assertEqual(result['full_schedules'],1)
        self.assertEqual(result['normal_recovery_mixed_schedules'],1)
        self.assertEqual(result['peak_gpu_kv_usage'],1.)

    def test_mixed_trace_loader_keeps_ids_and_schedules_but_shares_cache_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings":128,"vocab_size":1000}')
            source, background = root/'traceA_blksz_16.jsonl', root/'traceB_blksz_16.jsonl'
            for path, output in [(source, 8), (background, 16)]:
                path.write_text(''.join(json.dumps(dict(timestamp=i, input_length=32,
                    output_length=output, hash_ids=[0, 1]))+'\n' for i in range(50)))
            args = parse_args(['--inject-failure', '--model', str(model), '--trace', str(source),
                               '--survivor-trace', str(background), '--qps', '4', '--warmup', '5',
                               '--failure-delay', '5', '5', '--max-duration', '40'])
            output = root/'output'
            output.mkdir()
            requests, arrivals = load_probe_traces(args, output)
            self.assertEqual(args.recovery_cache_policy, 'shared')
            self.assertEqual(len(requests), 80)
            identities = []
            for worker, gpu in enumerate(args.gpus):
                rows, times = worker_workload(args, requests, arrivals, worker)
                self.assertEqual(len(rows), 40)
                self.assertEqual(times, list(range(40)))
                self.assertTrue(all(r['request_id'].startswith('TA' if worker == 0 else 'TB') for r in rows))
                identities.append(rows[0]['cache_salt'])
                (output/f'latencies_gpu{gpu}.jsonl').write_text('')
            self.assertEqual(identities,[None]*4)
            # trace 校验仍区分输入文件，但不用于隔离缓存。
            self.assertNotEqual(args.trace_workloads[0]['trace_sha256'],args.trace_workloads[1]['trace_sha256'])
            self.assertTrue(all(r['cache_salt'] is None for r in requests))
            (output/'config.json').write_text(json.dumps(vars(args)))
            groups, _ = request_groups(output, 50)
            self.assertEqual([r['planned_requests'] for r in groups], [40]*4)
            self.assertEqual(groups[0]['first_request_id'], 'TA00000')
            self.assertEqual(groups[1]['first_request_id'], 'TB00000')
            # 不同速率下，两条流分别按各自到达时间构造，源速率不再除以实例数。
            args = parse_args(['--model', str(model), '--trace', str(source),
                '--survivor-trace', str(background), '--source-qps', '2', '--survivor-qps', '.5',
                '--warmup', '5', '--window', '5', '--check-every', '1', '--max-duration', '20'])
            separate = root/'separate';separate.mkdir()
            requests, arrivals = load_probe_traces(args, separate)
            self.assertEqual(args.instance_qps, [2., .5, .5, .5])
            self.assertEqual(args.aggregate_offered_qps, 3.5)
            self.assertIsNone(args.per_instance_qps)
            self.assertEqual(args.failure_stability_scope, 'source')
            for worker in range(4):
                rows, times = worker_workload(args, requests, arrivals, worker)
                rate = 2 if worker==0 else .5
                self.assertEqual(len(rows), int(20*rate))
                self.assertEqual(times, [i/rate for i in range(int(20*rate))])
            self.assertEqual([r['per_instance_qps'] for r in args.trace_workloads], [2., .5])

    def test_mixed_replica_identity_checks_only_compare_the_three_backgrounds(self):
        args = parse_args(['--inject-failure', '--survivor-trace', 'traceB'])
        ready = [dict(worker=w, identity=dict(request_sequence_sha256='A' if w == 0 else 'B',
                                             arrivals_sha256='same',
                                             prompt_tokens_sha256='A' if w == 0 else 'B')) for w in range(4)]
        validate_replicas(args, ready)
        ready[2]['identity']['prompt_tokens_sha256'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'prompt_tokens_sha256'):
            validate_replicas(args, ready)

    def test_failure_time_is_reproducible_after_warmup_and_needs_four_instances(self):
        arguments = ['--inject-failure', '--warmup', '60', '--failure-delay', '10', '30']
        a, b = parse_args(arguments), parse_args(arguments)
        self.assertEqual(a.failure_time_s, b.failure_time_s)
        self.assertTrue(70 <= a.failure_time_s <= 90)
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(arguments+['--gpus', '2', '5'])
            with self.assertRaises(SystemExit):
                parse_args(arguments+['--max-duration', '80'])

    def test_repeated_fault_times_are_independently_sampled_once_per_round(self):
        arguments = ['--inject-failure','--compare-recovery-schedulers','--comparison-repeats','3',
                     '--warmup','60','--failure-delay','10','180','--max-duration','300','--seed','20261022']
        a,b = parse_args(arguments),parse_args(arguments)
        self.assertEqual(a.comparison_failure_times_s,b.comparison_failure_times_s)
        self.assertEqual(len(set(a.comparison_failure_times_s)),3)
        self.assertTrue(all(70<=t<=240 for t in a.comparison_failure_times_s))
        self.assertEqual(a.failure_time_s,a.comparison_failure_times_s[0])
        # 更改重复数不改变首轮采样；后续策略不重新抽样。
        single = parse_args(arguments+['--comparison-repeats','1'])
        self.assertEqual(a.failure_time_s,single.failure_time_s)

    def test_equal_assignment_handles_remainders_and_uses_distinct_attempt_ids(self):
        rows = [dict(request_id=f'TA{i}') for i in range(8)]
        assigned = recovery_assignment(rows, [1, 2, 3])
        self.assertEqual([len(assigned[w]) for w in (1, 2, 3)], [3, 3, 2])
        all_rows = [r for values in assigned.values() for r in values]
        self.assertEqual({r['request_id'] for r in all_rows}, {r['request_id'] for r in rows})
        self.assertEqual(len({r['recovery_id'] for r in all_rows}), 8)
        self.assertFalse({r['recovery_id'] for r in all_rows} & {r['request_id'] for r in rows})
        self.assertEqual(recovery_assignment([], [1, 2, 3]), {1: [], 2: [], 3: []})

    def test_resume_uses_saved_tokens_and_generates_only_remaining_output(self):
        row = dict(prompt_token_ids=[1, 2, 3], generated_token_ids=[4, 5], output_length=6)
        self.assertEqual(recovery_input(row, 'resume'), ([1, 2, 3, 4, 5], 4))
        self.assertEqual(recovery_input(row, 'restart'), ([1, 2, 3], 6))
        with self.assertRaises(ValueError):
            recovery_input(dict(row, output_length=2), 'resume')

    def test_forward_waiting_and_future_once_with_original_arrivals_and_separate_ids(self):
        waiting = [dict(request_id='TA0', arrival_s=1., source_generated_tokens=3)]
        future = [dict(request_id=f'TA{i}', arrival_s=float(i+10)) for i in range(1, 6)]
        assigned = forwarding_assignment(waiting, future, [1, 2, 3])
        self.assertEqual([len(assigned[w]) for w in (1, 2, 3)], [2, 2, 2])
        rows = [r for items in assigned.values() for r in items]
        self.assertEqual({r['original_request_id'] for r in rows}, {'TA'+str(i) for i in range(6)})
        self.assertEqual(len({r['request_id'] for r in rows}), 6)
        self.assertTrue(all(r['request_id'].startswith('FWD0_') for r in rows))
        self.assertEqual(assigned[1][0]['source_queue_at_failure'], 'waiting')
        self.assertEqual(assigned[1][0]['source_generated_tokens'], 3)
        self.assertEqual(assigned[1][0]['arrival_s'], 1.)
        self.assertEqual(future[0]['request_id'], 'TA1')

    def test_forwarding_merges_pending_ingress_without_reordering_admitted_requests(self):
        original = [dict(request_id=str(i)) for i in range(3)]
        forwarded = [dict(request_id='waiting', arrival_s=.5), dict(request_id='future', arrival_s=3.)]
        rows, arrivals = merge_pending(original, [0., 2., 4.], 1, forwarded)
        self.assertEqual([r['request_id'] for r in rows], ['0', 'waiting', '1', 'future', '2'])
        self.assertEqual(arrivals, [0., .5, 2., 3., 4.])

    def test_queue_head_across_skipped_queue_without_preempting_running(self):
        scheduler = Scheduler()
        running = SimpleNamespace(request_id='running')
        ordinary = SimpleNamespace(request_id='ordinary')
        a, b = SimpleNamespace(request_id='recover-a'), SimpleNamespace(request_id='recover-b')
        scheduler.running.append(running)
        scheduler.waiting.extend([ordinary, a, b])
        scheduler.skipped_waiting.append(SimpleNamespace(request_id='skipped-ordinary'))
        order = ['recover-a', 'recover-b']
        promote_recoveries(scheduler, order)
        self.assertEqual(list(scheduler.waiting), [a, b, ordinary])
        chosen = recovery_queue(scheduler, order, scheduler._select_waiting_queue_for_scheduling)
        self.assertIs(chosen, scheduler.waiting)
        self.assertEqual(scheduler.running, [running])

    def test_recovery_latency_includes_queue_and_prefill_and_keeps_one_token_tpot_null(self):
        state = dict(is_recovery=True, last_tokens=0, first_s=None, arrival_s=10., enqueued_s=10.,admitted_s=11.,
                     first_scheduled_s=12., failure_s=9., source_last_token_s=8.,
                     status='unfinished', ttft_ms=None, tpot_ms=None, completed_s=None)
        self.assertTrue(update_latency(state, 1, 13., False))
        self.assertEqual(state['ttft_ms'], 3000)
        self.assertEqual(state['queue_wait_ms'], 2000)
        self.assertEqual(state['scheduled_to_first_ms'], 1000)
        self.assertNotIn('group_wait_ms',state)
        self.assertEqual(state['failure_to_first_ms'], 4000)
        self.assertEqual(state['interruption_ms'], 5000)
        update_latency(state, 3, 13.2, True)
        self.assertAlmostEqual(state['tpot_ms'], 100.)
        single = dict(state, last_tokens=0, first_s=None)
        update_latency(single, 1, 13., True)
        self.assertIsNone(single['tpot_ms'])

    def test_restart_ttft_uses_token_one_and_tpot_covers_entire_restart(self):
        state = dict(is_recovery=True, last_tokens=0, first_s=None, arrival_s=10., enqueued_s=10.,admitted_s=10.,
                     first_scheduled_s=12., failure_s=9., source_last_token_s=8.,
                     source_generated_tokens=3, status='unfinished', ttft_ms=None,
                     tpot_ms=None, completed_s=None)
        self.assertTrue(update_latency(state, 1, 13., False))
        self.assertEqual(state['ttft_ms'], 3000.)
        self.assertFalse(update_latency(state, 3, 13.2, False))
        self.assertFalse(update_latency(state, 4, 13.3, False))
        self.assertEqual(state['first_s'], 13.)
        update_latency(state, 6, 13.5, True)
        self.assertAlmostEqual(state['tpot_ms'], 100.)
        self.assertEqual(state['generated_tokens_after_restart'], 6)
        self.assertEqual(state['first_token_index'], 1)
        almost_finished_source = dict(state, last_tokens=0, first_s=None, source_generated_tokens=5)
        update_latency(almost_finished_source, 1, 13., False)
        update_latency(almost_finished_source, 6, 13.5, True)
        self.assertAlmostEqual(almost_finished_source['tpot_ms'], 100.)

    def test_pre_failure_metrics_do_not_count_future_completions_or_zero_missing_tpot(self):
        rows = [dict(first_s=65., completed_s=70., ttft_ms=10., tpot_ms=20.),
                dict(first_s=69., completed_s=100., ttft_ms=30., tpot_ms=40.),
                dict(first_s=None, completed_s=None, ttft_ms=None, tpot_ms=None)]
        summary = metric_summary(rows, 60., 80.)
        self.assertEqual(summary['avg_ttft_ms'], 20.)
        self.assertEqual(summary['ttft_requests'], 2)
        self.assertEqual(summary['avg_tpot_ms'], 20.)
        self.assertEqual(summary['tpot_requests'], 1)

    def test_unfinished_restart_contributes_first_token_ttft_without_complete_tpot(self):
        state = dict(is_recovery=True, last_tokens=0, first_s=None, arrival_s=10., enqueued_s=10.,admitted_s=10.,
                     first_scheduled_s=12., failure_s=9., source_last_token_s=8.,
                     source_generated_tokens=5, status='unfinished', ttft_ms=None,
                     tpot_ms=None, completed_s=None)
        update_latency(state, 3, 13., False)
        self.assertEqual(state['first_s'], 13.)
        self.assertEqual(state['generated_tokens_after_restart'], 3)
        summary = metric_summary([state])
        self.assertEqual(summary['ttft_requests'], 1)
        self.assertEqual(summary['avg_ttft_ms'], 3000.)
        self.assertEqual(summary['tpot_requests'], 0)

    def test_four_replica_failure_protocol_and_duplicate_trace_ids_end_to_end(self):
        self.run_four_replicas(mixed=False)

    def test_mixed_traces_and_shared_recovery_protocol_end_to_end(self):
        self.run_four_replicas(mixed=True)

    def test_forwarded_traffic_is_accounted_when_drain_deadline_expires(self):
        self.run_four_replicas(mixed=True, drain=0.)

    def test_prefill_balanced_protocol_end_to_end(self):
        self.run_four_replicas(mixed=True,strategy='prefill-balanced')

    def test_busy_fault_signal_preserves_four_instance_recovery_protocol(self):
        self.run_four_replicas(mixed=True,busy=True)

    def test_kv_group_admits_every_member_to_native_waiting_before_resume(self):
        self.run_four_replicas(mixed=True,strategy='kv-aware')

    def test_kv_source_order_is_preserved_when_lexical_leader_is_later_and_prefill_is_chunked(self):
        self.run_four_replicas(mixed=True,strategy='kv-aware',chunked=True)

    def test_prefix_shortest_order_reaches_native_queue_without_admission_gates(self):
        captured = self.run_four_replicas(mixed=True,strategy='kv-aware',order='prefix-shortest')
        assigned = captured['plan']['assignment']
        self.assertEqual([r['original_request_id'] for r in assigned],['TA00002','TA00001','TA00000'])
        self.assertEqual([r['estimated_prefill_tokens'] for r in assigned],[240,24,32])
        self.assertEqual(len({r['target_worker'] for r in assigned}),1)
        self.assertEqual(captured['after']['summary']['recovery_order'],'prefix-shortest')

    def test_group_queue_orders_keep_all_members_in_native_queue_before_resume(self):
        for order in ('group-first','group-ratio'):
            captured=self.run_four_replicas(mixed=True,strategy='kv-aware',order=order)
            self.assertEqual(captured['after']['summary']['recovery_order'],order)

    def test_strict_replay_matches_all_four_queues_tokens_and_comparison_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = []
            for index,strategy in enumerate(('equal','prefill-balanced','kv-aware')):
                captured = self.run_four_replicas(mixed=True,strategy=strategy,journal=temporary,
                                                  journal_mode='replay' if index else 'record')
                results.append(dict(strategy=strategy,**captured))
            signatures = [[g['prefailure_state_signature'] for g in r['before']['gpus']] for r in results]
            self.assertEqual(signatures[0],signatures[1])
            self.assertEqual(signatures[0],signatures[2])
            self.assertEqual([r['plan']['failure_s'] for r in results],[results[0]['plan']['failure_s']]*3)
            self.assertEqual([r['original_request_id'] for r in results[0]['plan']['assignment']],
                             [r['original_request_id'] for r in results[1]['plan']['assignment']])
            output = Path(temporary)/'reports'
            output.mkdir()
            write_failure_comparison(output,results,{})
            self.assertEqual({p.name for p in output.iterdir()},set(RESULT_FILES))
            for name in RESULT_FILES:
                content = (output/name).read_text()
                self.assertTrue(all(strategy in content for strategy in ('equal','prefill-balanced','kv-aware')))
            self.assertIn('通过',(output/'before_failure.md').read_text())

            # 每轮新采样并记录，轮内重放两次；聚合以完整三策略配对为单位。
            args = parse_args(['--inject-failure','--compare-recovery-schedulers','--comparison-repeats','3',
                               '--warmup','.02','--failure-delay','.03','.12','--max-duration','.3'])
            calls,seen_cases,case_count = [],[],3
            def fake_workers(local,requests,arrivals,stage,capture):
                self.assertEqual(local.recovery_cache_policy,'shared')
                repeat = len(calls)//case_count+1
                calls.append((local._prefailure_mode,local._prefailure_dir,local.failure_time_s))
                seen_cases.append((local.recovery_scheduler,local.recovery_order))
                fixture = next(r for r in results if r['strategy']==local.recovery_scheduler)
                capture.update(copy.deepcopy({k:fixture[k] for k in ('before','plan','after')}))
                capture['plan'].update(planned_failure_s=local.failure_time_s,failure_s=local.failure_time_s,
                                       recovery_order=local.recovery_order)
                for row in capture['plan']['assignment']+capture['after']['recovery_requests']:
                    row['original_request_id'] = f"R{repeat}_"+row['original_request_id']
                summary = capture['after']['summary']
                scale = .5 if case_count==6 and local.recovery_order=='prefix-shortest' else 1.
                if local.recovery_order in ('group-first','group-ratio'):
                    scale = .6 if local.recovery_order=='group-first' else .4
                summary.update(avg_ttft_ms=repeat*1000.*scale,avg_tpot_ms=repeat*10.,recovery_order=local.recovery_order)
                for row in capture['after']['recovery_requests']:
                    row.update(ttft_ms=repeat*1000.*scale,tpot_ms=repeat*10.)
                return summary
            repeated_output = Path(temporary)/'repetitions'
            repeated_output.mkdir()
            with patch('sample_running.run_workers',side_effect=fake_workers),patch('sys.stdout'):
                repeated = compare_recovery_schedulers(args,[],[],repeated_output)
            self.assertEqual([r['repeat'] for r in repeated],[1,1,1,2,2,2,3,3,3])
            self.assertEqual([mode for mode,_,_ in calls],['record','replay','replay']*3)
            self.assertEqual(len({directory for _,directory,_ in calls}),3)
            self.assertEqual([t for _,_,t in calls],[t for t in args.comparison_failure_times_s for _ in range(3)])
            for offset in (0,3,6):
                self.assertEqual(len({directory for _,directory,_ in calls[offset:offset+3]}),1)
            self.assertEqual({p.name for p in repeated_output.iterdir()},set(RESULT_FILES))
            rendered = (repeated_output/'after_recovery.md').read_text()
            self.assertIn('| equal | 3 | 2000.00 ± 1000.00 | 20.00 ± 10.00 |',rendered)
            for repeat in (1,2,3):
                self.assertIn(f'| {repeat} | R{repeat}_',rendered)
                self.assertIn(f'R{repeat}_',(repeated_output/'failure_schedule.md').read_text())
                self.assertIn(f'## 第 {repeat} 轮',(repeated_output/'failure_schedule.md').read_text())
            self.assertIn('每轮独立采样',(repeated_output/'before_failure.md').read_text())
            write_failure_comparison(repeated_output,repeated[:4],vars(args))
            partial = (repeated_output/'after_recovery.md').read_text()
            self.assertIn('完整配对轮次 1/3',partial)
            self.assertIn('| equal | 1 | 1000.00 ± — | 10.00 ± — |',partial)
            repeated[2]['after']['summary']['traffic']['unfinished'] = 1
            write_failure_comparison(repeated_output,repeated,vars(args))
            incomplete = (repeated_output/'after_recovery.md').read_text()
            self.assertIn('完整配对轮次 2/3',incomplete)
            self.assertIn('| equal | 2 | 2500.00 ± 707.11 | 25.00 ± 7.07 |',incomplete)

            # 两种排队顺序必须共用每轮的唯一基线，共 3 次记录、15 次重放。
            args.comparison_orders=['source','prefix-shortest']
            calls.clear();seen_cases.clear();case_count=6
            matrix_output=Path(temporary)/'matrix';matrix_output.mkdir()
            with patch('sample_running.run_workers',side_effect=fake_workers),patch('sys.stdout'):
                matrix = compare_recovery_schedulers(args,[],[],matrix_output)
            self.assertEqual(len(matrix),18)
            self.assertEqual([mode for mode,_,_ in calls],(['record']+['replay']*5)*3)
            self.assertEqual(len({directory for _,directory,_ in calls}),3)
            expected=[(strategy,order) for strategy in ('equal','prefill-balanced','kv-aware')
                      for order in args.comparison_orders]
            self.assertEqual(seen_cases,expected*3)
            self.assertEqual([t for _,_,t in calls],[t for t in args.comparison_failure_times_s for _ in range(6)])
            self.assertEqual({p.name for p in matrix_output.iterdir()},set(RESULT_FILES))
            matrix_report=(matrix_output/'after_recovery.md').read_text()
            self.assertIn('已完成 18/18',matrix_report)
            self.assertIn('| equal / source | 3 | 2000.00 ± 1000.00 |',matrix_report)
            self.assertIn('| equal / shortest | 3 | 1000.00 ± 500.00 |',matrix_report)
            self.assertIn('| kv-aware / shortest | 3 | 1000.00 ± 500.00 |',matrix_report)
            self.assertIn('equal GPU',(matrix_output/'failure_schedule.md').read_text())
            self.assertIn('（6/6）',(matrix_output/'before_failure.md').read_text())
            write_failure_comparison(matrix_output,matrix[:5],vars(args))
            self.assertIn('完整配对轮次 0/3',(matrix_output/'after_recovery.md').read_text())
            write_failure_comparison(matrix_output,matrix[:7],vars(args))
            self.assertIn('完整配对轮次 1/3',(matrix_output/'after_recovery.md').read_text())

            calls.clear();seen_cases.clear()
            def wrong_placement(local,requests,arrivals,stage,capture):
                summary=fake_workers(local,requests,arrivals,stage,capture)
                if len(calls)==2:
                    row=capture['plan']['assignment'][0]
                    row['target_worker']=1+row['target_worker']%3
                return summary
            with patch('sample_running.run_workers',side_effect=wrong_placement),patch('sys.stdout'):
                with self.assertRaisesRegex(RuntimeError,'GPU placement'):
                    compare_recovery_schedulers(args,[],[],matrix_output)
            self.assertEqual(len(calls),2)  # 不重试凑相同分配。

            args.comparison_cases=[('prefill-balanced','prefix-shortest'),('kv-aware','prefix-shortest'),
                                   ('kv-aware','group-first'),('kv-aware','group-ratio')]
            args.comparison_orders=['prefix-shortest','group-first','group-ratio']
            calls.clear();seen_cases.clear();case_count=4
            focused_output=Path(temporary)/'group_orders';focused_output.mkdir()
            with patch('sample_running.run_workers',side_effect=fake_workers),patch('sys.stdout'):
                focused=compare_recovery_schedulers(args,[],[],focused_output)
            self.assertEqual(len(focused),12)
            self.assertEqual(seen_cases,args.comparison_cases*3)
            self.assertEqual([mode for mode,_,_ in calls],(['record']+['replay']*3)*3)
            content=(focused_output/'after_recovery.md').read_text()
            self.assertIn('已完成 12/12',content)
            self.assertIn('| kv-aware / group-first | 3 | 1200.00 ± 600.00 |',content)
            self.assertIn('| kv-aware / group-ratio | 3 | 800.00 ± 400.00 |',content)
            self.assertIn('同一共享组的首 token',content)
            self.assertIn('故障至首 token',content)
            self.assertNotIn('equal GPU',(focused_output/'failure_schedule.md').read_text())

    def test_strict_replay_rejects_different_output_tokens_and_snapshot_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(_prefailure_dir=temporary,_prefailure_mode='record')
            journal = PrefailureReplay(args,0)
            batch = [dict(scheduled_requests=[dict(request_id='TA',scheduled_tokens=3)])]
            output = [SimpleNamespace(request_id='TA',finished=False,
                                      outputs=[SimpleNamespace(token_ids=[7])])]
            journal.step(1,.1,.2,batch,output)
            engine = Engine()
            internal = engine.add_request('TA',dict(prompt_token_ids=[1,2,3]),SimpleNamespace(max_tokens=20))
            proof = dict(qualified=True, checks=[dict(stable=True)])
            journal.finish(engine.scheduler,{internal:'TA'},1,.3,stability=proof)
            args._prefailure_mode = 'replay'
            changed = PrefailureReplay(args,0)
            self.assertEqual(changed.data['stability'], proof)
            output[0].outputs[0].token_ids = [8]
            with self.assertRaisesRegex(RuntimeError,'diverged at forward'):
                changed.step(1,.1,.2,batch,output)
            changed = PrefailureReplay(args,0)
            changed.index = 1
            engine.scheduler.requests[internal].num_computed_tokens += 1
            with self.assertRaisesRegex(RuntimeError,'snapshot mismatch'):
                changed.finish(engine.scheduler,{internal:'TA'},1,.3)

    def run_four_replicas(self, mixed, drain=.5, strategy='equal',journal=None,journal_mode=None,chunked=False,order='source',busy=False):
        args = parse_args(['--inject-failure', '--warmup', '.02', '--failure-delay', '.03', '.03',
                           '--max-duration', '.30', '--recovery-drain', str(drain),
                           '--recovery-scheduler',strategy,'--recovery-order',order]+
                          (['--survivor-trace', 'trace-B'] if mixed else []))
        args.per_instance_qps = 1.
        if busy:
            args.failure_when_busy = True
            args._failure_event = threading.Event()
        if journal:
            args._prefailure_dir,args._prefailure_mode = journal,journal_mode
        requests = [dict(request_id=f'TA{i:05d}', source_index=i, input_length=3,
                         output_length=20, tokens=[7, 8, 9]) for i in range(7)]
        if strategy=='kv-aware' or journal:
            requests = [dict(r,input_length=256,tokens=[7]*240+[8+r['source_index']]*16) for r in requests]
        if order=='prefix-shortest':
            requests = [dict(r,input_length=256-8*i,tokens=[7]*min(224,256-8*i)+
                        [8+i]*max(0,32-8*i)) for i,r in enumerate(requests)]
        if chunked:
            args.kv_split_factor = 3
            # 字典序末尾的源请求最先出现；过去的依赖放行会让它等待后来的请求。
            requests = [dict(r,input_length=512,tokens=[7]*64+[12-r['source_index']]*448) for r in requests]
        arrivals = [0., .002, .004, .006, .008, .075, .29]
        engines = [Engine() for _ in range(4)]
        if chunked:
            for engine in engines:
                engine.recovery_chunk = 64
        pairs = [mp.Pipe() for _ in range(4)]
        start = time.perf_counter()+.03
        errors, summaries = [], []
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, {'vllm': SimpleNamespace(SamplingParams=SimpleNamespace)}):
            output = Path(temporary)
            coordinator = FailureCoordinator(args, [p[0] for p in pairs], output, start)

            def run(worker):
                try:
                    time.sleep(max(0., start-time.perf_counter()))
                    engine = engines[worker]
                    local = requests
                    if mixed:
                        local = [dict(r, request_id=('TB' if worker else 'TA')+r['request_id'][2:],
                                      trace_file='trace-B' if worker else args.trace,
                                      cache_salt='trace-B' if worker else 'trace-A') for r in requests]
                    summary = replay_failure(engine, engine.scheduler, local, arrivals, args,
                                             worker, args.gpus[worker], output, pairs[worker][1], start)
                    pairs[worker][1].send(dict(kind='result', **summary))
                except BaseException as exc:
                    errors.append(exc)
                    pairs[worker][1].send(dict(kind='error', error=repr(exc)))

            threads = [threading.Thread(target=run, args=(w,), daemon=True) for w in range(4)]
            for thread in threads:
                thread.start()
            pending = {p[0] for p in pairs}
            deadline = time.perf_counter()+5
            try:
                while pending and time.perf_counter() < deadline:
                    for connection in wait(pending, timeout=.1):
                        message = connection.recv()
                        self.assertNotEqual(message['kind'], 'error', message)
                        if not coordinator.handle(message):
                            self.assertEqual(message['kind'], 'result')
                            summaries.append(message)
                            pending.remove(connection)
                self.assertFalse(pending, 'Protocol did not finish')
                self.assertFalse(errors)
                summary = coordinator.finalize()
                self.assertEqual(summary['status'], 'complete')
                self.assertGreater(summary['requests'], 0)
                self.assertEqual(summary['requests'], summary['completed'])
                self.assertEqual(len(summaries), 4)
                self.assertTrue(all(s['time_s'] <= coordinator.plan['failure_s']
                                    for s in coordinator.snapshots.values()))
                source = coordinator.snapshots[0]
                self.assertEqual(set(coordinator.plan['original_running_ids']),
                                 {r['request_id'] for r in source['checkpoints']})
                self.assertTrue(source['waiting_ids'])
                self.assertEqual(len(source['future_requests']), 2)
                self.assertFalse(set(source['waiting_ids']) & set(coordinator.plan['original_running_ids']))
                failed = next(s for s in summaries if s['worker'] == 0)
                self.assertEqual(failed['status'], 'failed_instance')
                for worker, items in coordinator.assignments.items():
                    ack = coordinator.acks[worker]
                    admitted = ack['admitted_recovery_ids']
                    self.assertEqual(ack['queue_ids']['waiting'][:len(admitted)],admitted)
                    self.assertEqual(admitted,[r['recovery_id'] for r in items])
                    self.assertNotIn('pending_recovery_ids',ack)
                    for item in items:
                        self.assertEqual(engines[worker].prompts[item['recovery_id']],
                                         item['prompt_token_ids'])
                        self.assertIsNone(engines[worker].salts[item['recovery_id']])
                self.assertTrue(all(salt is None for engine in engines for salt in engine.salts.values()))
                self.assertEqual({p.name for p in output.iterdir()}, set(RESULT_FILES))
                after = coordinator.after_report
                records = after['recovery_requests']
                if strategy=='kv-aware':
                    self.assertTrue(any(r['shared_prefix_tokens'] for r in records))
                    for r in records:
                        self.assertNotIn('prefill_after',r)
                        # 所有组成员在恢复屏障前入队，早于任何恢复 prefill 调度。
                        target_records=[a for a in records if a['target_worker']==r['target_worker']]
                        self.assertLess(r['admitted_s'],min(a['first_scheduled_s'] for a in target_records))
                        self.assertAlmostEqual(r['ttft_ms'],1000*(r['first_s']-r['enqueued_s']))
                    self.assertIn('前缀组',(output/'failure_schedule.md').read_text())
                    self.assertIn('直接进入原生等待队列',(output/'after_recovery.md').read_text())
                self.assertEqual(len(records), summary['requests'])
                self.assertEqual(summary['recovery_execution'],'source_order_native_queue_v3')
                self.assertTrue(all(r['prefill_scheduled_tokens']==r['input_length']-r['gpu_cached_tokens_first_schedule']
                                    for r in records))
                self.assertTrue(all(r['prefill_batch_ids'][-1]==r['first_token_batch']
                                    and r['first_schedule_batch']<=r['first_token_batch'] for r in records))
                self.assertTrue(all(abs(r['ttft_ms']-r['queue_wait_ms']-
                                        r['scheduled_to_first_ms'])<1e-7 for r in records))
                self.assertTrue(all(r['ttft_ms'] > 0 and r['tpot_ms'] > 0 for r in records))
                self.assertTrue(all(r['first_token_index'] == r['first_token_observed_count'] == 1
                                    and r['latency_measurement'] == 'restart_first_token_v2' for r in records))
                self.assertTrue(all(abs(r['tpot_ms']-(r['completed_s']-r['first_s'])*1000/
                                       (r['output_length']-1)) < 1e-8 for r in records))
                self.assertTrue(all(r['source_status'] == 'RUNNING' and r['source_phase'] == 'decode'
                                    and r['source_remaining_tokens'] ==
                                    r['original_output_length']-r['source_generated_tokens'] for r in records))
                self.assertTrue(all(r['gpu_cached_tokens_first_schedule'] == 0 for r in records))
                forwarded = after['forwarded_requests']
                expected_forwarded = set(source['waiting_ids']) | {
                    r['request_id'] for r in source['future_requests']}
                self.assertEqual(len(forwarded), len(expected_forwarded))
                self.assertEqual({r['original_request_id'] for r in forwarded}, expected_forwarded)
                self.assertEqual({r['request_id'] for r in forwarded}, {
                    r['request_id'] for items in coordinator.forwarding.values() for r in items})
                self.assertTrue(all(r['arrival_s'] == arrivals[r['source_index']] for r in forwarded))
                self.assertTrue(all(r['admitted_s'] is None or r['admitted_s'] >= r['arrival_s'] for r in forwarded))
                self.assertTrue(all(r['source_queue_at_failure'] in ('waiting', 'future') for r in forwarded))
                if drain:
                    self.assertTrue(all(r['status'] == 'completed' for r in forwarded))
                    self.assertEqual(summary['forwarding']['status'], 'complete')
                    self.assertTrue(all(r['last_tokens'] == r['output_length'] for r in forwarded))
                else:
                    self.assertGreater(summary['forwarding']['unfinished'], 0)
                    self.assertEqual(summary['forwarding']['status'], 'incomplete_forwarding')
                examples = [batch for gpu in after['gpus'] for batch in gpu['recovery_prefill_batches']]
                self.assertTrue(any({'normal', 'recovery'} <= {r['cohort'] for r in batch['scheduled_requests']}
                                    for batch in examples))
                self.assertTrue(all(sum(r['scheduled_tokens'] for r in batch['scheduled_requests']) ==
                                    batch['total_scheduled_tokens'] for batch in examples))
                self.assertTrue(all(r['prefill_tokens']+r['decode_tokens']==r['scheduled_tokens']
                                    for batch in examples for r in batch['scheduled_requests']))
                before = coordinator.before_report
                plan = coordinator.plan
                rendered = (output/'after_recovery.md').read_text()
                self.assertIn('| 请求编号 |',rendered)
                self.assertTrue(all(r['original_request_id'] in rendered for r in records))
                self.assertTrue(all(r['request_id'] not in rendered for r in forwarded))
                self.assertEqual(len(before['gpus']),4)
                self.assertEqual(len(plan['assignment']), len(records))
                self.assertEqual(len(plan['forwarding_assignment']), len(forwarded))
                if drain:
                    self.assertEqual(summary['traffic'],dict(planned=28,completed=28,unfinished=0))
                self.assertTrue(all('prompt_token_ids' not in r and 'cache_salt' not in r for r in records))
                if mixed:
                    self.assertTrue(all(r['target_trace'] == 'trace-B' for r in records))
                    self.assertEqual(summary['requests_with_gpu_cache_hits'], 0)
                return dict(before=before,plan=plan,after=after)
            finally:
                for thread in threads:
                    thread.join(timeout=1)
                for parent, child in pairs:
                    parent.close()
                    child.close()


if __name__ == '__main__':
    unittest.main()
