"""CPU-only checks for experiment comparability and metric accounting."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from metrics import CacheMetrics, percentile, transfer_config, interval_summary
from report import extract, read_csv, write_csv, write_matched_comparison
from run import (aggregate_results, build_plan, check_cpu_capacity, load_trace,
                 run_configuration, trace_shard)
from worker import make_prompt, request_prompt
from audit import ideal_prefix_reuse
from sample_running import (StabilityGate, latency_summary, parse_args as sampling_args,
                            running_snapshot, worker_workload, workload_identity)
from running_report import request_groups, time_series, stable_probe_summary


class RunningSamplingTests(unittest.TestCase):
    def test_untimed_prepared_tokens_keep_exact_prompts_lengths_and_dataset_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 64, "vocab_size": 128}')
            trace = root/'prepared.jsonl'
            rows = [dict(request_id=f'train-0001-q{i+1}', doc_id='train-0001',
                         prompt_token_ids=[1, 2, 3, 4+i], input_length=4, output_length=5+i)
                    for i in range(2)]
            trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[.5],
                arrival='constant', duration=4, instances=1, trace_order='file')
            with patch('transformers.AutoTokenizer.from_pretrained') as tokenizer:
                requests, schedules = load_trace(args, root)
            tokenizer.assert_not_called()
            self.assertEqual(args.trace_format, 'tokens')
            self.assertEqual(args.trace_block_tokens, 16)
            self.assertEqual(schedules[.5], [0., 2.])
            self.assertEqual([r['id'] for r in requests], [r['request_id'] for r in rows])
            self.assertEqual([r['request_id'] for r in requests], ['TA00000', 'TA00001'])
            self.assertEqual([r['output_length'] for r in requests], [5, 6])
            self.assertEqual([request_prompt(r, (10, 20)) for r in requests],
                             [r['prompt_token_ids'] for r in rows])
            self.assertTrue(all(r['timestamp_kind']=='source_order' for r in requests))
            args.arrival = 'trace'
            with self.assertRaisesRegex(ValueError, 'no arrival timestamps'):
                load_trace(args, root)

    def test_bounded_ingress_allows_a_fixed_buffer_but_rejects_growth(self):
        args = sampling_args(['--ingress-policy','bounded'])
        gate = StabilityGate(args)
        gate.add(0,60,running=1,waiting=0,outstanding=3,ingress=2)
        arrivals = list(range(61))
        proof = gate.evaluate(60,arrivals,arrivals)
        self.assertTrue(proof['checks']['ingress_level'])
        self.assertTrue(proof['stable'])
        args.ingress_policy = 'caught-up'
        self.assertFalse(gate.evaluate(60,arrivals,arrivals)['checks']['ingress_caught_up'])
        args.ingress_policy = 'bounded'
        gate = StabilityGate(args)
        gate.add(0,30,running=1,waiting=0,outstanding=3,ingress=2)
        gate.add(30,60,running=1,waiting=0,outstanding=3,ingress=10)
        self.assertFalse(gate.evaluate(60,arrivals,arrivals)['checks']['ingress_level'])

    def test_stable_probe_report_uses_same_period_for_latency_and_weighted_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(gpus=[2], warmup=0, max_duration=30, measurement_duration=10)
            (root/'config.json').write_text(json.dumps(config))
            summary = dict(per_instance_qps=2, status='complete',
                stable_periods=[dict(start_s=10,end_s=20)], latency=dict(unfinished=0))
            (root/'summary.json').write_text(json.dumps([summary]))
            load = [dict(start_s=a,end_s=b,running=n,waiting=w) for a,b,n,w in
                    [(0,10,8,3),(10,15,1,0),(15,20,3,2),(20,30,8,3)]]
            (root/'load_gpu2.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in load))
            records = [dict(arrival_s=0,first_s=t,completed_s=t,ttft_ms=v,tpot_ms=11)
                       for t,v in [(9,9000),(12,120),(19,180),(21,9000)]]
            for name in ['latencies_gpu2.jsonl','requests.jsonl']:
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
            (root/'stability_gpu2.jsonl').write_text(json.dumps(dict(reason='periodic',qualified=True,checks={}))+'\n')
            result = stable_probe_summary(root)
            self.assertTrue(result['qualified'])
            self.assertEqual(result['stable']['seconds'],10)
            self.assertEqual(result['stable']['mean_running'],2)
            self.assertEqual(result['stable']['p95_running'],3)
            self.assertEqual(result['stable']['mean_waiting'],1)
            self.assertEqual(result['stable']['avg_ttft_ms'],150)
            self.assertEqual(result['stable']['ttft_samples'],2)
            self.assertEqual(result['post_warmup']['mean_running'],6)

    def test_ttft_stability_rejects_latency_growth_even_with_flat_queue(self):
        args = sampling_args(['--ttft-stability'])
        gate = StabilityGate(args)
        gate.add(0, 60, running=1, waiting=0, outstanding=1, ingress=0)
        arrivals = list(range(61))
        flat = [(i+.5, 1000.) for i in range(60)]
        self.assertTrue(gate.evaluate(60, arrivals, arrivals, flat)['stable'])
        rising = [(t, 100. if t<30 else 3000.) for t, _ in flat]
        result = gate.evaluate(60, arrivals, arrivals, rising)
        self.assertTrue(result['checks']['running_level'])
        self.assertFalse(result['checks']['ttft_level'])
        self.assertFalse(result['stable'])
        self.assertFalse(gate.evaluate(60, arrivals, arrivals, [])['checks']['ttft_samples'])

    def test_raw_qa_preserves_real_shared_prefix_answers_and_source_ids(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kw: [1]+[
            ord(c) % 128 for c in messages[0]['content']]+[2]
        tokenizer.encode.side_effect = lambda text, **kw: [ord(c) % 128 for c in text]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 64, "vocab_size": 128}')
            trace = root/'qa.jsonl'
            rows = [dict(id=str(i), doc_id='doc', context=context, question=question, answer=answer)
                    for i, (context, question, answer) in enumerate([
                        ('abc', 'X', 'yes'), ('abc', 'Y', 'no'), ('z'*90, 'Z', 'long')])]
            trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[1],
                arrival='constant', duration=2, instances=1, trace_order='file')
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
                requests, schedules = load_trace(args, root)
            self.assertEqual(args.trace_format, 'qa')
            self.assertEqual(args.trace_block_tokens, 16)
            self.assertEqual(schedules[1], [0., 1.])
            self.assertEqual([r['id'] for r in requests], ['0', '1'])
            self.assertEqual([r['output_length'] for r in requests], [3, 2])
            a, b = [request_prompt(r, (3, 4)) for r in requests]
            self.assertEqual(a, requests[0]['prompt_token_ids'])
            prefix = 1+len('Context:\nabc\n\nQuestion:\n')
            self.assertEqual(a[:prefix], b[:prefix])
            self.assertNotEqual(a, b)
            self.assertEqual(json.loads((root/'excluded.json').read_text())[0]['source_index'], 2)
            self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])
            args.arrival = 'trace'
            with self.assertRaisesRegex(ValueError, 'no arrival timestamps'):
                load_trace(args, root)

    def test_request_groups_keep_missing_and_unfinished_requests_in_same_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(gpus=[2, 5], request_routing='replicate', warmup=10)
            (root/'config.json').write_text(json.dumps(config))
            manifest = [dict(request_id=f'r{i}', input_length=32, output_length=8) for i in range(5)]
            (root/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
            records = [dict(request_id='r0', arrival_s=0, status='completed', ttft_ms=100, tpot_ms=10),
                       dict(request_id='r1', arrival_s=1, status='completed', ttft_ms=200, tpot_ms=20),
                       dict(request_id='r2', arrival_s=12, status='completed', ttft_ms=300, tpot_ms=30),
                       dict(request_id='r3', arrival_s=13, status='aborted_at_end', ttft_ms=400, tpot_ms=None)]
            for gpu in config['gpus']:
                (root/f'latencies_gpu{gpu}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            rows, _ = request_groups(root, 2)
            a, b, c = rows[:3]
            self.assertEqual(a['avg_ttft_ms'], 150)
            self.assertEqual(a['avg_tpot_ms'], 15)
            self.assertEqual(a['fully_post_warmup'], 0)
            self.assertEqual(b['fully_post_warmup'], 1)
            self.assertEqual(b['fully_completed'], 0)
            self.assertEqual(b['unfinished_requests'], 1)
            self.assertEqual(b['tpot_requests'], 1)
            self.assertEqual(c['planned_requests'], 1)
            self.assertEqual(c['not_submitted_requests'], 1)
            self.assertIsNone(c['avg_ttft_ms'])
            self.assertEqual(rows[3]['first_request_id'], a['first_request_id'])
            self.assertEqual(rows[4]['last_request_id'], b['last_request_id'])

    def test_replicas_receive_identical_full_order_times_and_prompts(self):
        args = sampling_args(['--gpus', '4', '5', '6', '7', '--qps', '4', '--qps-scope', 'per-instance'])
        requests = [dict(request_id=f'TA{i:05d}', input_length=3, output_length=2,
                         tokens=[i, i+1, i+2]) for i in range(12)]
        arrivals = [i/4 for i in range(12)]
        identities = []
        for worker in range(4):
            rows, times = worker_workload(args, requests, arrivals, worker)
            self.assertEqual([r['request_id'] for r in rows], [r['request_id'] for r in requests])
            self.assertEqual(times, arrivals)
            identities.append(workload_identity(rows, times))
        self.assertTrue(all(i == identities[0] for i in identities))
        self.assertEqual(args.per_instance_qps, 4)
        self.assertEqual(args.aggregate_offered_qps, 16)
        changed = [dict(r) for r in requests]
        changed[2] = dict(changed[2], tokens=[9, 9, 9])
        self.assertNotEqual(workload_identity(changed, arrivals)['prompt_tokens_sha256'],
                            identities[0]['prompt_tokens_sha256'])
        args.request_routing = 'round-robin'
        for worker in range(4):
            rows, times = worker_workload(args, requests, arrivals, worker)
            self.assertEqual(rows, requests[worker::4])
            self.assertEqual(times, arrivals[worker::4])

    def test_default_total_qps_counts_all_four_replicas(self):
        for total, per_instance in [(4, 1), (5, 1.25)]:
            args = sampling_args(['--gpus', '2', '5', '6', '7', '--qps', str(total)])
            self.assertEqual(args.qps_scope, 'global')
            self.assertEqual(args.aggregate_offered_qps, total)
            self.assertEqual(args.per_instance_qps, per_instance)
            self.assertEqual(args.logical_qps, per_instance)
            arrivals = [i/args.logical_qps for i in range(10)]
            requests = [dict(request_id=f'TA{i:05d}') for i in range(10)]
            for worker in range(4):
                rows, times = worker_workload(args, requests, arrivals, worker)
                self.assertEqual(len(rows), 10)
                self.assertAlmostEqual(times[1]-times[0], 1/per_instance)

    def test_replicated_manifest_preserves_file_order_and_targets_all_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 128, "vocab_size": 32}')
            row = dict(input_length=32, output_length=8, hash_ids=[1, 2])
            trace = root/'trace_blksz_16.jsonl'
            trace.write_text(''.join(json.dumps(dict(row, timestamp=t))+'\n' for t in (2, 0, 1)))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[1],
                                      arrival='constant', duration=3, instances=4,
                                      request_routing='replicate', trace_order='file')
            requests, _ = load_trace(args, root)
            records = [json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['source_index'] for r in requests], [0, 1, 2])
            self.assertTrue(all(r['workers'] == [0, 1, 2, 3] and 'worker' not in r for r in records))
            args.arrival = 'trace'
            with self.assertRaisesRegex(ValueError, 'nondecreasing timestamps'):
                load_trace(args, root)

    def test_time_series_weights_load_by_time_and_bins_latency_by_event_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text(json.dumps(dict(gpus=[4], max_duration=20)))
            load = [dict(start_s=0, end_s=12, running=2, waiting=1, outstanding=3),
                    dict(start_s=12, end_s=20, running=6, waiting=3, outstanding=9)]
            events = [dict(event='first_token', time_s=9, ttft_ms=1000),
                      dict(event='first_token', time_s=11, ttft_ms=2000),
                      dict(event='completed', time_s=19, tpot_ms=50)]
            for name, rows in [('load', load), ('events', events)]:
                (root/f'{name}_gpu4.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            rows, _ = time_series(root, 10)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[1]['mean_running'], 5.2)
            self.assertAlmostEqual(rows[1]['mean_waiting'], 2.6)
            self.assertEqual(rows[0]['avg_ttft_ms'], 1000)
            self.assertEqual(rows[1]['avg_ttft_ms'], 2000)
            self.assertIsNone(rows[0]['avg_tpot_ms'])
            self.assertEqual(rows[1]['avg_tpot_ms'], 50)

    def test_latency_cohorts_keep_unfinished_ttft_and_exclude_epoch_crossing_tpot(self):
        base = dict(ttft_ms=100., tpot_ms=20., arrival_s=70., first_s=70.1,
                    arrival_epoch=1, first_epoch=1, completion_epoch=1, status='completed')
        rows = [base, dict(base, ttft_ms=200., tpot_ms=None, completion_epoch=None,
                           status='aborted_at_end'),
                dict(base, ttft_ms=300., arrival_epoch=None, first_epoch=1, completion_epoch=2),
                dict(base, ttft_ms=None, tpot_ms=None, first_s=None, first_epoch=None,
                     completion_epoch=None, status='aborted_at_end')]
        stats = latency_summary(rows, 60)
        self.assertEqual(stats['stable']['ttft_requests'], 2)
        self.assertEqual(stats['stable']['avg_ttft_ms'], 150.)
        self.assertEqual(stats['stable']['tpot_requests'], 1)
        self.assertEqual(stats['stable']['avg_tpot_ms'], 20.)
        self.assertEqual(stats['full_run']['ttft_requests'], 3)
        self.assertEqual(stats['unfinished'], 2)
        self.assertEqual(stats['stable_arrivals_without_first_token'], 1)

    def gate(self):
        args = sampling_args(['--window', '60', '--warmup', '60', '--stable-checks', '3'])
        return StabilityGate(args)

    def test_stability_requires_complete_window_and_consecutive_passes(self):
        gate = self.gate()
        arrivals = list(range(121))
        completions = [t+.5 for t in arrivals]
        gate.add(0, 30, running=2, waiting=0, outstanding=2, ingress=0)
        self.assertFalse(gate.check(30, arrivals, completions)['stable'])
        gate.add(30, 60, running=2, waiting=0, outstanding=2, ingress=0)
        first = gate.check(60, arrivals, completions)
        self.assertTrue(first['stable'])
        self.assertFalse(first['qualified'])
        gate.add(60, 70, running=2, waiting=0, outstanding=2, ingress=0)
        self.assertFalse(gate.check(70, arrivals, completions)['qualified'])
        gate.add(70, 80, running=2, waiting=0, outstanding=2, ingress=0)
        self.assertTrue(gate.check(80, arrivals, completions)['qualified'])

    def test_saturated_running_does_not_hide_growing_queue(self):
        gate = self.gate()
        for second in range(60):
            gate.add(second, second+1, running=20, waiting=second,
                     outstanding=20+second, ingress=0)
        result = gate.check(60, list(range(121)), [t/0.4 for t in range(30)])
        self.assertFalse(result['stable'])
        self.assertFalse(result['checks']['balanced_flow'])
        self.assertFalse(result['checks']['waiting_level'])
        self.assertEqual(result['consecutive_passes'], 0)

    def test_sampler_revalidates_and_rejects_ingress_backlog(self):
        gate = self.gate()
        gate.add(0, 60, running=1, waiting=0, outstanding=1, ingress=0)
        arrivals = list(range(121))
        for _ in range(3):
            qualified = gate.check(60, arrivals, arrivals)
        self.assertTrue(qualified['qualified'])
        gate.add(60, 90, running=1, waiting=0, outstanding=20, ingress=10)
        proof = gate.evaluate(90, arrivals, arrivals)
        self.assertFalse(proof['stable'])
        self.assertFalse(proof['checks']['ingress_caught_up'])

    def test_running_ids_map_original_lines_and_distinguish_scheduled_subset(self):
        requests = [SimpleNamespace(request_id='TA00007', num_computed_tokens=40,
                                    num_prompt_tokens=32, output_token_ids=[1, 2]),
                    SimpleNamespace(request_id='TA00011', num_computed_tokens=16,
                                    num_prompt_tokens=64, output_token_ids=[])]
        scheduler = SimpleNamespace(running=requests, get_request_counts=lambda: (2, 3))
        manifest = {'TA00007': dict(source_index=7, input_length=32, output_length=20),
                    'TA00011': dict(source_index=11, input_length=64, output_length=8,
                                    chat_id='conversation-2', turn=4)}
        snapshot = running_snapshot(scheduler, SimpleNamespace(num_scheduled_tokens={'TA00007': 1}), manifest)
        self.assertEqual(snapshot['source_indices'], [7, 11])
        self.assertEqual(snapshot['source_lines'], [8, 12])
        self.assertEqual(snapshot['scheduled_ids'], ['TA00007'])
        self.assertEqual(snapshot['requests'][0]['computed_tokens_before_step'], 39)
        self.assertEqual(snapshot['requests'][0]['phase'], 'decode')
        self.assertEqual(snapshot['requests'][1]['phase'], 'prefill')
        self.assertEqual(snapshot['waiting_count'], 3)
        with self.assertRaises(KeyError):
            running_snapshot(scheduler, None, {})

    def test_empty_running_is_recorded_without_busy_time_bias(self):
        scheduler = SimpleNamespace(running=[], get_request_counts=lambda: (0, 0))
        snapshot = running_snapshot(scheduler, None, {})
        self.assertEqual(snapshot['running_ids'], [])
        self.assertEqual(snapshot['scheduled_ids'], [])

    def test_native_randomized_request_id_uses_add_request_mapping(self):
        request = SimpleNamespace(request_id='TA00007-a1b2c3d4', num_computed_tokens=16,
                                  num_prompt_tokens=32, output_token_ids=[])
        scheduler = SimpleNamespace(running=[request], get_request_counts=lambda: (1, 0))
        manifest = {'TA00007': dict(source_index=7, input_length=32, output_length=8)}
        scheduled = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        snapshot = running_snapshot(scheduler, scheduled, manifest,
                                    {request.request_id: 'TA00007'})
        self.assertEqual(snapshot['running_ids'], ['TA00007'])
        self.assertEqual(snapshot['scheduled_ids'], ['TA00007'])
        self.assertEqual(snapshot['scheduler_running_ids'], [request.request_id])
        self.assertEqual(snapshot['requests'][0]['computed_tokens_before_step'], 0)

    def test_shared_manifest_respects_instance_count_and_original_source_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 128, "vocab_size": 32}')
            # Invalid first row and blank second line must not renumber eligible rows.
            row = dict(timestamp=0, input_length=32, output_length=8, hash_ids=[1, 2])
            trace = root/'trace_blksz_16.jsonl'
            trace.write_text(json.dumps(dict(row, input_length=0))+'\n\n'+
                             ''.join(json.dumps(dict(row, timestamp=i))+'\n' for i in range(10)))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[1],
                                      arrival='constant', duration=6, instances=2)
            requests, _ = load_trace(args, root)
            records = [json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['worker'] for r in records], [0, 1, 0, 1, 0, 1])
            self.assertEqual(requests[0]['request_id'], 'TA00002')
            self.assertEqual(requests[0]['source_index'], 2)


class ExperimentTests(unittest.TestCase):
    def test_two_gpu_dispatch_preserves_all_requests_and_fractional_arrivals(self):
        args = argparse.Namespace(gpus=[2, 5], qps=[.5], routing_offset=1, duration=40)
        requests = list(range(20))
        arrivals = [i/.5 for i in requests]
        context = MagicMock()
        pipes = [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())]
        context.Pipe.side_effect = pipes
        context.Process.return_value.is_alive.return_value = False
        with tempfile.TemporaryDirectory() as temporary, \
                patch('run.mp.get_context', return_value=context), \
                patch('run.GPUUtilSampler'), \
                patch('run.receive_all', side_effect=[[], RuntimeError('stop after dispatch')]):
            with self.assertRaisesRegex(RuntimeError, 'stop after dispatch'):
                run_configuration(args, requests, {.5: arrivals}, Path(temporary))
        dispatched = []
        for worker, call in enumerate(context.Process.call_args_list):
            assigned = call.kwargs['args'][3]
            command = pipes[worker][0].send.call_args_list[0].args[0]
            self.assertEqual(assigned, requests[(worker-1) % 2::2])
            self.assertEqual(command['arrivals'], [i/.5 for i in assigned])
            dispatched.extend(assigned)
        self.assertEqual(sorted(dispatched), requests)

    def test_two_gpu_plan_uses_two_shards_and_rotates_without_invalid_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(gpus=[2, 5], qps=[.5, 1, 2],
                                      cpu_cache_gb=[0, 10, 20, 30], repeats=3,
                                      rotate_workers=True, cpu_cache_storage='private')
            plan = build_plan(args, Path(temporary))
            self.assertEqual(len(plan), 36)
            for case, directory in plan:
                self.assertEqual(case.instances, 2)
                self.assertEqual(case.routing_offset, (case.repeat-1) % 2)
                routing = json.loads((directory/'routing.json').read_text())
                self.assertEqual(sorted(r['trace_shard'] for r in routing), [0, 1])

    def test_capacity_preflight_uses_selected_instance_count(self):
        with patch('run.available_host_bytes', return_value=70_000_000_000):
            check_cpu_capacity(30, instances=2)
            with self.assertRaisesRegex(RuntimeError, '4 instances x 30 GB'):
                check_cpu_capacity(30, instances=4)

    def test_qwen_trace_auto_uses_16_token_blocks_and_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 4096, "vocab_size": 32}')
            records = [dict(timestamp=i*.25, input_length=48, output_length=8,
                            hash_ids=[1, 2, i+3], chat_id=i, parent_chat_id=-1,
                            turn=1, type='text') for i in range(180)]
            trace = root/'qwen_traceA_blksz_16.jsonl'
            trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[3],
                                      arrival='trace', duration=40)
            output = root/'seconds'
            output.mkdir()
            requests, seconds = load_trace(args, output)
            self.assertEqual(args.trace_block_tokens, 16)
            self.assertEqual(args.trace_time_unit, 's')
            self.assertEqual(requests[0]['chat_id'], 0)
            self.assertEqual(requests[0]['type'], 'text')
            self.assertEqual(len(requests), 120)
            self.assertAlmostEqual(seconds[3][1], 1/3)
            # Explicit milliseconds override must replay the same physical times.
            for row in records:
                row['timestamp'] *= 1000
            trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
            args.trace_time_unit = 'ms'
            output = root/'milliseconds'
            output.mkdir()
            _, milliseconds = load_trace(args, output)
            for a, b in zip(seconds[3], milliseconds[3]):
                self.assertAlmostEqual(a, b)

    def test_model_context_cap_and_explicit_trace_block_size_filter_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'other_model'
            model.mkdir()
            (model/'config.json').write_text(json.dumps(dict(text_config=dict(
                max_position_embeddings=128, vocab_size=32))))
            records = [dict(timestamp=i, input_length=48, output_length=8,
                            hash_ids=[1, 2, 3]) for i in range(60)]
            records += [dict(timestamp=61, input_length=64, output_length=8, hash_ids=[1, 2, 3, 4]),
                        dict(timestamp=62, input_length=48, output_length=8, hash_ids=[1])]
            trace = root/'custom.jsonl'
            trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[1],
                                      arrival='constant', duration=40, max_model_len=64,
                                      trace_block_tokens=16, trace_time_unit='s', dtype='float16')
            output = root/'result'
            output.mkdir()
            requests, _ = load_trace(args, output)
            self.assertEqual(len(requests), 40)
            self.assertEqual(args.max_model_len, 64)
            self.assertEqual(args.model_vocab_size, 32)
            self.assertEqual(args.dtype, 'float16')
            self.assertEqual(len(json.loads((output/'excluded.json').read_text())), 2)
            args.max_model_len = 129
            with self.assertRaisesRegex(ValueError, 'model config limit 128'):
                load_trace(args, output)

    def test_16_token_prefixes_are_shared_but_changed_blocks_and_tails_are_private(self):
        allowed = tuple(range(1000))
        a = make_prompt('a', 50, [1, 2, 3], allowed, 16)
        b = make_prompt('b', 50, [1, 2, 4], allowed, 16)
        self.assertEqual(len(a), 50)
        self.assertEqual(a[:32], b[:32])
        self.assertNotEqual(a[32:48], b[32:48])
        self.assertNotEqual(a[48:], b[48:])
        rows = [dict(input_length=48, hash_ids=[1, 2, 3]),
                dict(input_length=48, hash_ids=[1, 2, 4])]
        self.assertEqual(ideal_prefix_reuse(rows, 1, block_tokens=16)['reused_prefix_tokens'], 32)

    def test_each_qps_gets_an_independent_case_and_single_trial_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(gpus=[2, 0, 4, 5], qps=[1.5, 2],
                                      cpu_cache_gb=[0, 10], repeats=2,
                                      rotate_workers=False, cpu_cache_storage='private')
            plan = build_plan(args, Path(temporary))
            self.assertEqual(len(plan), 8)
            self.assertEqual(len({directory for _, directory in plan}), 8)
            self.assertEqual(args.qps, [1.5, 2])
            self.assertEqual([case.cpu_cache_gb for case, _ in plan], [0, 0, 10, 10, 10, 10, 0, 0])
            for case, directory in plan:
                self.assertEqual(len(case.qps), 1)
                self.assertEqual(directory.name, f'qps_{case.qps[0]:g}')
                self.assertEqual(case.routing_offset, 0)
                self.assertEqual([r['gpu'] for r in json.loads((directory/'routing.json').read_text())],
                                 [2, 0, 4, 5])

    def test_report_recovers_completed_case_in_new_layout_without_zero_filling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(gpus=[2, 0, 4, 5], qps=[1, 2], cpu_cache_gb=[0, 10],
                          repeats=3, warmup=10, duration=40)
            (root/'config.json').write_text(json.dumps(config))
            case = root/'cpu_gb_0/repeat_1/qps_1'
            case.mkdir(parents=True)
            summaries = []
            for gpu in config['gpus']:
                summaries.append(dict(global_qps=1, cpu_cache_gb=0, repeat=1, gpu=gpu,
                                      completion_qps=.25, planned=1, avg_ttft_ms=100,
                                      avg_tpot_ms=10, window_avg_itl_ms=10, mean_running=2,
                                      mean_waiting=0, cpu_load_bytes=0))
                record = dict(request_id=f'r{gpu}', first_s=11, completed_s=20,
                              ttft_ms=100, tpot_ms=10, output_length=3)
                (case/f'requests_qps_1_gpu{gpu}.jsonl').write_text(json.dumps(record)+'\n')
            write_csv(case/'summary.csv', summaries)
            (case/'summary.json').write_text(json.dumps(summaries))
            write_csv(case/'aggregate.csv', [dict(global_qps=1, cpu_cache_gb=0, repeat=1,
                      completion_qps=1, avg_ttft_ms=100, p95_ttft_ms=100, avg_tpot_ms=10,
                      window_avg_itl_ms=10, mean_waiting_per_gpu=0)])
            with patch('builtins.print'):
                extract(root)
            coverage = read_csv(root/'extracted/coverage.csv')
            self.assertEqual(len(coverage), 12)
            self.assertEqual(sum(r['status'] == 'completed' for r in coverage), 1)
            stats = read_csv(root/'extracted/per_gpu_statistics.csv')
            self.assertEqual(len(stats), 4)
            self.assertTrue(all(r['repeats_completed'] == '1' for r in stats))
            self.assertTrue(all(r['avg_ttft_ms_mean'] == '100.0' for r in stats))
            self.assertTrue(all(r['avg_ttft_ms_std'] == '' for r in stats))

    def test_rotation_preserves_each_shard_and_its_arrival_phase(self):
        requests = list(range(12))
        arrivals = [i/2.5 for i in requests]
        original = {w: list(zip(requests[w::4], arrivals[w::4])) for w in range(4)}
        rotated = {w: list(zip(requests[trace_shard(w, 1)::4],
                               arrivals[trace_shard(w, 1)::4])) for w in range(4)}
        for worker in range(4):
            self.assertEqual(rotated[(worker+1) % 4], original[worker])
        self.assertEqual(sorted(r for shard in rotated.values() for r, _ in shard), requests)

    def test_interval_window_covers_tokens_without_requiring_completion(self):
        events = [
            {'time_s': 9, 'new_tokens': 1, 'interval_ms': 900},
            {'time_s': 10, 'new_tokens': 1, 'interval_ms': None},
            {'time_s': 11, 'new_tokens': 1, 'interval_ms': 100},
            {'time_s': 39, 'new_tokens': 2, 'interval_ms': 600},
            {'time_s': 40, 'new_tokens': 1, 'interval_ms': 999},
        ]
        result = interval_summary(events, 10, 40)
        self.assertEqual(result['window_itl_tokens'], 3)
        self.assertAlmostEqual(result['window_avg_itl_ms'], 700/3)
        self.assertEqual(result['window_max_itl_ms'], 300)
        self.assertEqual(result['window_coalesced_token_events'], 1)

    def test_matched_comparison_uses_same_completed_request_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text(json.dumps(dict(
                repeats=1, qps=[3, 4], cpu_cache_gb=[0, 10], warmup=10, duration=40)))
            for capacity in (0, 10):
                directory = root/f'cpu_gb_{capacity}/repeat_1'
                directory.mkdir(parents=True)
                (directory/'summary.json').write_text('[]')
                records = [dict(request_id='a', first_s=11, completed_s=20,
                                ttft_ms=100, tpot_ms=100+capacity, output_length=3),
                           dict(request_id='b', first_s=12,
                                completed_s=21 if capacity == 0 else None,
                                ttft_ms=200, tpot_ms=1000 if capacity == 0 else None,
                                output_length=3)]
                (directory/'requests_qps_3_gpu2.jsonl').write_text(
                    ''.join(json.dumps(r)+'\n' for r in records))
            result = write_matched_comparison(root)
            self.assertEqual([r['global_qps'] for r in result], [3, 3])
            self.assertEqual([r['matched_tpot_samples'] for r in result], [1, 1])
            self.assertEqual([r['matched_avg_tpot_ms'] for r in result], [100, 110])
            self.assertEqual([r['matched_ttft_samples'] for r in result], [2, 2])

    def test_capacity_is_per_instance_in_decimal_gb(self):
        self.assertIsNone(transfer_config(0))
        config = transfer_config(25)
        self.assertEqual(config['kv_connector'], 'OffloadingConnector')
        self.assertEqual(config['kv_role'], 'kv_both')
        self.assertEqual(config['kv_connector_extra_config']['cpu_bytes_to_use'], 25_000_000_000)
        self.assertEqual(config['kv_connector_extra_config']['block_size'], 16)
        self.assertEqual(config['kv_connector_extra_config']['spec_name'], 'PrivateCPUOffloadingSpec')
        self.assertEqual(transfer_config(25, 'shm')['kv_connector_extra_config']['spec_name'],
                         'CPUOffloadingSpec')

    def test_private_storage_accepts_reported_ram_despite_small_shm(self):
        with patch('run.available_host_bytes', return_value=298_200_000_000), \
             patch('run.shutil.disk_usage', return_value=SimpleNamespace(free=70_600_000_000)):
            check_cpu_capacity(50, storage='private')
            with self.assertRaisesRegex(RuntimeError, '/dev/shm=70.6 GB'):
                check_cpu_capacity(50, storage='shm')

    def test_private_storage_still_rejects_insufficient_ram(self):
        with patch('run.available_host_bytes', return_value=200_000_000_000):
            with self.assertRaisesRegex(RuntimeError, 'reserve 8 GB'):
                check_cpu_capacity(50, storage='private')

    def test_native_delta_stats_respect_window_and_separate_denominators(self):
        monitor = CacheMetrics(10, 40)
        stats = SimpleNamespace(
            prefix_cache_stats=SimpleNamespace(queries=100, hits=40),
            connector_prefix_cache_stats=SimpleNamespace(queries=60, hits=30))
        connector = SimpleNamespace(data={
            'types': {'vllm:kv_offload_load_bytes': 'counter'},
            'data': {'vllm:kv_offload_load_bytes': {(): 1000}}})
        for elapsed in (9, 10, 39, 40):
            monitor.record(elapsed, stats, connector)
        result = monitor.summary()
        self.assertEqual(monitor.full['cpu_load_bytes'], 4000)
        self.assertEqual(result['cpu_load_bytes'], 2000)
        self.assertEqual(result['gpu_prefix_queries'], 200)
        self.assertEqual(result['gpu_prefix_hit_rate'], 0.4)
        self.assertEqual(result['cpu_prefix_queries'], 120)
        self.assertEqual(result['cpu_prefix_hit_rate'], 0.5)
        self.assertEqual(monitor.native['measurement']['vllm:kv_offload_load_bytes']['sum'], 2000)

    def test_no_samples_are_missing_rather_than_zero_latency(self):
        self.assertIsNone(percentile([], 95))
        self.assertIsNone(CacheMetrics(0, 40).summary()['cpu_prefix_hit_rate'])

    def test_identical_trace_and_routing_across_capacities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{"max_position_embeddings": 40960}')
            trace = root/'trace.jsonl'
            trace.write_text(''.join(json.dumps({
                'timestamp': i*250, 'input_length': 512, 'output_length': 10,
                'hash_ids': [i//4],
            })+'\n' for i in range(600)))
            args = argparse.Namespace(model=str(model), trace=str(trace), qps=[3],
                                      arrival='constant', duration=40)
            cases = []
            for capacity in (0, 10, 25, 50):
                output = root/str(capacity)
                output.mkdir()
                requests, schedule = load_trace(args, output)
                cases.append((requests, schedule, (output/'requests.jsonl').read_text()))
            self.assertTrue(all(case == cases[0] for case in cases))
            self.assertEqual(args.trace_block_tokens, 512)
            self.assertEqual(args.trace_time_unit, 'ms')
            self.assertEqual(len(cases[0][0]), 120)
            self.assertEqual(cases[0][1][3][:4], [0, 1/3, 2/3, 1])
            for worker in range(4):
                self.assertEqual(len(cases[0][1][3][worker::4]), 30)
            # Trace mode is intentionally rescaled, not forced to exactly 3 QPS.
            args.arrival = 'trace'
            output = root/'trace_case'
            output.mkdir()
            _, schedule = load_trace(args, output)
            self.assertEqual(len(schedule[3]), 120)

    def test_aggregate_pools_samples_and_ignores_unfinished_tpot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for gpu, ttfts in enumerate(([1, 2, 3], [100])):
                row = dict(gpu=gpu, looks_stable=True)
                for key in ('offered_qps', 'completion_qps', 'generation_tokens_per_s',
                            'mean_running', 'mean_waiting', 'mean_gpu_kv_usage',
                            'mean_cpu_cache_occupied_fraction', 'preemptions', 'planned',
                            'completed_total', 'unfinished', 'not_admitted', 'cpu_load_bytes',
                            'cpu_store_bytes', 'cpu_store_allocation_failures',
                            'gpu_prefix_queries', 'gpu_prefix_hits',
                            'cpu_prefix_queries', 'cpu_prefix_hits'):
                    row[key] = 0
                rows.append(row)
                records = [dict(first_s=11, completed_s=None, ttft_ms=value,
                                tpot_ms=None, e2e_ms=None) for value in ttfts]
                (root/f'requests_qps_3_gpu{gpu}.jsonl').write_text(
                    ''.join(json.dumps(record)+'\n' for record in records))
            args = argparse.Namespace(cpu_cache_gb=10, repeat=1, arrival='constant',
                                      warmup=10, duration=40)
            result = aggregate_results(rows, root, args, 3)
            self.assertEqual(result['ttft_samples'], 4)
            self.assertEqual(result['avg_ttft_ms'], 26.5)
            self.assertAlmostEqual(result['p95_ttft_ms'], 85.45)
            self.assertIsNone(result['avg_tpot_ms'])


if __name__ == '__main__':
    unittest.main()
