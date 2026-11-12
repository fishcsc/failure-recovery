"""父进程协调：管理四实例、故障屏障和同现场策略对照。

入口是 run_workers；与 engine 交换控制消息，调用 scheduling 分配请求，
由 report 生成结果，并持有 cpu_kv 的共享 RAM，确保源进程退出后备份仍在。
"""
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import tempfile
import time

from workload import worker_workload, validate_replicas

from .cpu_kv import SharedKVPool, WallProfile, profile_phase, apply_manifest_delta, materialize_manifest
from .engine import ReplayContext, worker_main
from .report import (FAILURE_REPORT_FILES as RESULT_FILES,
    write_failure_report as write_result, write_failure_comparison,
    initialize_results, compact_record, metric_summary, observation_summary,
    _failure_number, percentile)
from .scheduling import (assign_recovery_requests as recovery_assignment,
                        order_recovery_requests, forwarding_assignment,cache_prefix_comparisons,
                        recovery_plan_config,reuse_recovery_plan)


def experiment_config(args, engines=()):
    """提取实验配置和引擎身份，供三份结果文档记录同一组设定。"""
    keys = ('model', 'trace', 'survivor_trace', 'gpus', 'aggregate_offered_qps', 'per_instance_qps',
            'source_qps', 'survivor_qps', 'instance_qps', 'failure_stability_scope', 'cpu_kv_backup_gb',
            'cpu_kv_backup_bytes','backup_policy','kv_restore_mode','kv_load_policy','inference_replay','inference_replay_dir',
            'replay_backup_mode','profile_backup_overhead','profile_dir','replay_diagnostics_dir',
            'manifest_mode','manifest_debug_oracle',
            'ttft_stability', 'ttft_relative_tolerance', 'ttft_absolute_tolerance_ms',
            'ingress_policy',
            'qps_scope', 'arrival', 'request_routing', 'trace_order', 'cpu_cache_gb',
            'gpu_memory_utilization', 'max_num_seqs', 'batch_tokens', 'max_model_len', 'dtype',
            'warmup', 'window', 'max_duration', 'recovery_drain', 'recovery_mode', 'require_stable_before_failure', 'failure_when_busy',
            'recovery_cache_policy', 'recovery_scheduler', 'recovery_execution', 'recovery_order', 'comparison_orders', 'comparison_cases', 'comparison_repeats', 'kv_min_shared_tokens', 'kv_split_factor',
            'seed', 'failure_time_s', 'failure_delay', 'comparison_failure_times_s', 'latency_measurement', 'trace_workloads')
    return dict({k:getattr(args,k) for k in keys if hasattr(args,k)},
                backup_pool_definition='source_shared_cpu_hash',
                batch_invariant=os.environ.get('VLLM_BATCH_INVARIANT','0')=='1',
                engines=[dict(worker=e['worker'],gpu=e['gpu'],dtype=e['dtype'],kv_blocks=e['kv_blocks'],
                    device_name=e['identity']['device_name'],trace=e['identity']['trace']) for e in engines])


class FailureCoordinator:
    """保存一次实验的快照、分配方案、加载确认和最终结果。

    只协调进程和元数据，不执行模型前向，也不通过管道传输 KV 张量。
    """
    def __init__(self, args, connections, output, start, engines=None, cpu_metadata=None, context=None):
        self.args, self.connections, self.output, self.start = args, connections, output, start
        self.context = context or ReplayContext()
        self.profiler=WallProfile(getattr(args,'profile_backup_overhead',False),start)
        # 三类消息分别按逻辑 worker 存储，只有收齐当前阶段才进入下一阶段。
        self.snapshots, self.acks, self.results = {}, {}, {}
        self.assignments, self.forwarding = None, None
        self.config = experiment_config(args, engines or [])
        self.cpu_metadata = cpu_metadata if cpu_metadata is not None else {}
        self.recovery_plan = None
        self.manifest_materialize_ms=0.
        initialize_results(output, self.config)

    @staticmethod
    def public_snapshot(snapshot):
        omitted = {'checkpoints', 'waiting_requests', 'future_requests', 'latencies', 'scheduler_running_ids',
                   'source_indices', 'source_lines', 'scheduled_ids', 'kind','cpu_directory','stopped_monotonic_s'}
        row = {k: v for k, v in snapshot.items() if k not in omitted}
        row['requests'] = [compact_record(r) for r in row.get('requests', [])]
        return row

    def save_before(self):
        self.before_report = dict(status='recorded', config=self.config,
                                  failure_s=self.plan['failure_s'], gpus=[self.public_snapshot(s)
                                  for _, s in sorted(self.snapshots.items())])
        write_result(self.output/RESULT_FILES[0], self.before_report)

    def handle(self, message):
        """分派 worker 消息；四份故障快照到齐后触发恢复。

        返回 True 表示该 worker 仍在运行；最终结果返回 False，供父进程移出等待集合。
        """
        kind, worker = message['kind'], message['worker']
        if kind == 'cpu_kv_metadata':
            return self.apply_cpu_metadata(message)
        if kind == 'pre_failure':
            if worker in self.snapshots:
                raise RuntimeError('Duplicate failure snapshot')
            self.snapshots[worker] = message
            if len(self.snapshots) == self.args.instances:
                self.profiler.stop()
                self.dispatch()
            return True
        if kind == 'result':
            self.results[worker] = message.pop('failure_results')
            return False
        return self.handle_ack(message)

    @profile_phase('coordinator_metadata_apply')
    def apply_cpu_metadata(self,message):
        """按源进程消息顺序维护父进程的请求进度和 CPU 块目录。

        返回 True；这里接收的是 token 增量和槽位引用，KV 字节已在共享 RAM 中。
        """
        worker=message['worker']
        if worker != 0 or not getattr(self.args, 'cpu_kv_backup_gb', 0):
            raise RuntimeError('Unexpected CPU checkpoint metadata publisher')
        for update in message['updates']:
            rid = update['request_id']
            if update.get('completed'):
                self.cpu_metadata.pop(rid,None)
                continue
            entry = self.cpu_metadata.setdefault(rid,dict(generated_token_ids=[],slots=[]))
            if 'manifest_delta' in update:
                with self.profiler.stage('manifest_delta_apply'):
                    apply_manifest_delta(entry,update['manifest_delta'])
                continue
            if 'version' in update and update['version']<=entry.get('version',0):
                raise RuntimeError('Out-of-order CPU manifest version')
            entry['generated_token_ids'].extend(update.get('new_tokens',[]))
            entry.update({k:v for k,v in update.items() if k not in ('new_tokens','request_id')})
        if (plan:=message.get('recovery_plan')) is not None:
            if self.recovery_plan is not None and plan['version']<=self.recovery_plan['version']:
                raise RuntimeError('Out-of-order recovery plan version')
            self.recovery_plan=plan
        return True
    def handle_ack(self,message):
        """检查目标已入队且所选 KV 已加载；三台目标全确认后统一 resume。

        返回是否识别该消息，防止某张卡提前推理而改变恢复计时边界。
        """
        kind,worker=message['kind'],message['worker']
        if kind != 'recovery_enqueued':
            return False
        if worker in self.acks or self.assignments is None or worker not in self.assignments:
            raise RuntimeError('Unexpected recovery enqueue acknowledgement')
        demand=getattr(self.args,'kv_load_policy','eager')=='demand'
        if getattr(self.args,'cpu_kv_backup_gb',0):
            if demand:
                if message.get('restore_protocol')!='on-demand' or message.get('preload_complete'):
                    raise RuntimeError('Demand ACK must register plans without cohort H2D')
            elif message.get('restore_protocol')!='eager-preload' or not message.get('preload_complete'):
                raise RuntimeError('All targets must finish selected KV preload before global resume')
        expected = [r['recovery_id'] for r in self.assignments[worker]]
        if message['recovery_ids'] != expected:
            raise RuntimeError('Recovery assignment/admission mismatch')
        if message['admitted_recovery_ids'] != expected:
            raise RuntimeError('All recovery requests must enter native queues before resuming')
        queued={rid for ids in message['queue_ids'].values() for rid in ids}
        running=set(message.get('running_recovery_ids',[]))
        if not running<=set(expected):
            raise RuntimeError('Unexpected running recovery IDs')
        if running and getattr(self.args,'kv_restore_mode','serial')!='suffix-overlap':
            raise RuntimeError('Only suffix-overlap can compute recovery heads before global resume')
        if message.get('kv_restore_mode','serial')!=getattr(self.args,'kv_restore_mode','serial'):
            raise RuntimeError('Recovery restore modes differ between coordinator and target')
        if not set(expected) <= queued|running:
            raise RuntimeError('Admitted recovery is missing from native queues')
        for ids in message['queue_ids'].values():
            ordinary_seen = False
            for rid in ids:
                if rid in expected and ordinary_seen:
                    raise RuntimeError('Recovery request is behind ordinary waiting requests')
                ordinary_seen |= rid not in expected
        self.acks[worker] = message
        if len(self.acks) == self.args.instances-1:
            # 放行由最慢目标决定；不能把三卡 H2D 时间之和当成这段墙钟等待。
            for target in self.assignments:
                self.connections[target].send(dict(action='resume'))
            self.plan.update(resume_sent_s=time.perf_counter()-self.start,
                             enqueue_acknowledgements=list(self.acks.values()))
            write_result(self.output/RESULT_FILES[1], self.plan)
        return True

    def dispatch(self):
        """四份快照齐备后，校验 CPU 检查点并分配故障请求。

        优先复用配置及输入匹配的预测计划，否则分配并排序；发送任务并保存调度报告。
        """
        source = self.snapshots[0]
        failure_s = max(s['time_s'] for s in self.snapshots.values())
        checkpoints = source['checkpoints']
        if [c['request_id'] for c in checkpoints] != source['running_ids']:
            raise RuntimeError('Failure cohort must equal the native source running queue')
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            # 单一管道保证增量顺序；再核对冻结目录，排除槽位复用后的过期引用。
            for checkpoint in checkpoints:
                saved = self.cpu_metadata[checkpoint['request_id']]
                if 'blocks_by_position' in saved:
                    began=time.perf_counter()
                    saved.update(materialize_manifest(saved))
                    self.manifest_materialize_ms+=1000*(time.perf_counter()-began)
                if (saved['slots'] != checkpoint['cpu_kv']['slots']
                        or saved['generated_token_ids'] != checkpoint['generated_token_ids']
                        or saved['prompt_token_ids'] != checkpoint['prompt_token_ids']
                        or saved.get('signature') != checkpoint['cpu_kv']['signature']):
                    raise RuntimeError('Parent-owned CPU checkpoint/progress differs from fault snapshot')
                manifest=checkpoint['cpu_kv']
                if (saved.get('blocks')!=manifest.get('blocks') or saved.get('version')!=manifest.get('version')
                        or saved.get('epoch')!=manifest.get('epoch')):
                    raise RuntimeError('Published CPU logical-block manifest differs from frozen checkpoint')
                if manifest.get('epoch')!=source.get('cpu_epoch'):
                    raise RuntimeError('Frozen CPU pool epoch mismatch')
                for ref in manifest.get('blocks',[]):
                    expected={k:ref[k] for k in ('cpu_slot','generation','block_key')}
                    if source.get('cpu_directory',{}).get(str(ref['cpu_slot']))!=expected:
                        raise RuntimeError('Stale CPU slot/generation/key in frozen manifest')
        timeline = {}
        if self.context.mode:
            # 所有实例重建完成后统一时钟，故障后的原到达流仍严格保持全局 QPS。
            timeline['timeline_start'] = time.perf_counter()-failure_s
            self.start = timeline['timeline_start']
        # 推理独立回放用实际停止时钟计算 TTFO；旧策略对照沿用逻辑故障时间。
        # 两者都换算到目标后续使用的同一时间轴，不能直接拿故障前 wall_s 相减。
        source_failure_s = (source['stopped_monotonic_s']-self.start
                            if self.context.inference_only else source['time_s'])
        schedule_started=time.perf_counter()
        config=recovery_plan_config(list(range(1,self.args.instances)),self.args.recovery_scheduler,
            self.args.recovery_mode,self.args.kv_min_shared_tokens,self.args.kv_split_factor,self.args.recovery_order)
        self.assignments=reuse_recovery_plan(self.recovery_plan,checkpoints,**config)
        plan_reused=self.assignments is not None
        if not plan_reused:
            # 没有预测或输入已改变才重算；CPU 备份覆盖不参与分配/排序。
            with cache_prefix_comparisons():
                self.assignments = recovery_assignment(checkpoints, config['survivors'],
                    self.args.recovery_scheduler,self.args.recovery_mode,self.args.kv_min_shared_tokens,self.args.kv_split_factor)
                self.assignments = {w:order_recovery_requests(items,self.args.recovery_order,
                    self.args.recovery_mode) for w,items in self.assignments.items()}
        scheduling_ms=1000*(time.perf_counter()-schedule_started)
        self.forwarding = forwarding_assignment(source['waiting_requests'], source['future_requests'], list(self.assignments))
        if {r['original_request_id'] for items in self.forwarding.values() for r in items} & set(source['running_ids']):
            raise RuntimeError('Forwarded traffic overlaps the running recovery cohort')
        self.plan = dict(status='scheduled', failed_worker=0, failed_gpu=self.args.gpus[0],
            recovery_plan_reused=plan_reused,recovery_scheduling_ms=scheduling_ms,
            recovery_plan_version=self.recovery_plan['version'] if plan_reused else None,
            recovery_protocol='running_priority_and_forward_all_v3', latency_measurement=self.args.latency_measurement,
            source_trace=self.args.trace, survivor_trace=self.args.survivor_trace,
            recovery_cache_policy=self.args.recovery_cache_policy, recovery_mode=self.args.recovery_mode,
            recovery_scheduler=self.args.recovery_scheduler,
            recovery_execution=self.args.recovery_execution,
            recovery_order=self.args.recovery_order,
            kv_min_shared_tokens=self.args.kv_min_shared_tokens,kv_split_factor=self.args.kv_split_factor,
            estimated_prefill_by_target=[dict(worker=w,gpu=self.args.gpus[w],
                input_tokens=sum(r['prefill_tokens'] for r in items),
                estimated_prefill_tokens=sum(r['estimated_prefill_tokens'] for r in items),
                groups=len({r['recovery_group'] for r in items if r.get('recovery_group')}))
                for w,items in self.assignments.items()],
            planned_failure_s=self.args.failure_time_s, failure_s=failure_s,
            source_stopped_forward_s=source['time_s'], dispatch_started_s=time.perf_counter()-self.start,
            source_failure_physical_s=source_failure_s,
            prefailure=[dict(worker=w,verified=s.get('prefailure_verified',False),
                steps=s.get('prefailure_steps'),state_signature=s.get('prefailure_state_signature'),
                gpu_cache_signature=s.get('prefailure_gpu_cache_signature'),wall_s=s.get('prefailure_wall_s'))
                for w,s in sorted(self.snapshots.items())] if self.context.mode else [],
            policy=self.args.recovery_scheduler+'_queue_head', snapshots_s={str(w):s['time_s'] for w,s in self.snapshots.items()},
            snapshot_skew_s=failure_s-min(s['time_s'] for s in self.snapshots.values()),
            original_running_ids=source['running_ids'], source_waiting_forwarded=source['waiting_ids'],
            source_future_forwarded=source['future_not_submitted'],
            future_traffic_policy='survivors_keep_original_stream; source_waiting_and_future_round_robin',
            assignment=[dict(original_request_id=r['request_id'], source_index=r['source_index'],
                source_line=r['source_index']+1, recovery_id=r['recovery_id'], target_worker=w,
                target_gpu=self.args.gpus[w], source_generated_tokens=len(r['generated_token_ids']),
                source_status=r.get('source_status'), source_phase=r.get('source_phase'),
                source_computed_tokens=r.get('source_computed_tokens'),
                source_remaining_tokens=r['output_length']-len(r['generated_token_ids']),
                estimated_prefill_tokens=r['estimated_prefill_tokens'],recovery_group=r.get('recovery_group'),
                group_position=r.get('group_position'),queue_rank=r.get('queue_rank'),
                input_length=r['input_length'], output_length=r['output_length'])
                for w, items in self.assignments.items() for r in items],
            forwarding_assignment=[dict(original_request_id=r['original_request_id'], request_id=r['request_id'],
                source_line=r['source_index']+1, target_worker=w, target_gpu=self.args.gpus[w],
                source_queue_at_failure=r['source_queue_at_failure'], arrival_s=r['arrival_s'],
                source_generated_tokens=r.get('source_generated_tokens', 0),
                input_length=r['input_length'], output_length=r['output_length'])
                for w, items in self.forwarding.items() for r in items])
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            for entry in self.plan['assignment']:
                checkpoint = next(r for items in self.assignments.values() for r in items
                                  if r['recovery_id']==entry['recovery_id'])
                manifest=checkpoint['cpu_kv']
                entry['cpu_backup_tokens'] = len(manifest.get('blocks',manifest['slots']))*16
                refs=manifest.get('blocks',[])
                entry['cpu_backup_ranges'] = ([(refs[0]['logical_block']*16,
                    (refs[-1]['logical_block']+1)*16)] if refs else [])
        for snapshot in self.snapshots.values():
            lower = max(self.args.warmup, failure_s-self.args.window)
            snapshot.update(baseline_start_s=lower, baseline_end_s=failure_s,
                            baseline=metric_summary(snapshot['latencies'], lower, failure_s))
        self.save_before()
        write_result(self.output/RESULT_FILES[1], self.plan)
        self.connections[0].send(dict(action='fail', failure_s=failure_s,**timeline))
        for worker, items in self.assignments.items():
            self.connections[worker].send(dict(action='recover', failure_s=failure_s,
                source_failure_s=source_failure_s,
                checkpoints=items, forwarded=self.forwarding[worker],**timeline))
        print(f"Failure GPU={self.args.gpus[0]} t={failure_s:.3f}s: running={len(checkpoints)}, "
              f"waiting={len(source['waiting_ids'])}, future={source['future_not_submitted']}", flush=True)

    def finalize(self):
        """所有 worker 结束后核对请求去向，汇总延迟、批次和备份数据。

        返回 summary，并保存最终报告；故障 running 与转发流量分开计数，避免重复统计。
        """
        if self.assignments is None or len(self.acks) != self.args.instances-1 or len(self.results) != self.args.instances:
            raise RuntimeError('Failure protocol did not complete')
        recovered = [r for data in self.results.values() for r in data['recovery']]
        forwarded = [r for data in self.results.values() for r in data['forwarded']]
        expected = {(r['recovery_id'], w) for w, items in self.assignments.items() for r in items}
        if {(r['request_id'],r['target_worker']) for r in recovered} != expected or len(recovered) != len(expected):
            raise RuntimeError('Recovery request accounting mismatch')
        expected_fwd = {(r['request_id'], w) for w, items in self.forwarding.items() for r in items}
        if {(r['request_id'],r['target_worker']) for r in forwarded} != expected_fwd or len(forwarded) != len(expected_fwd):
            raise RuntimeError('Source waiting/future traffic accounting mismatch')
        completed = sum(r['status']=='completed' for r in recovered)
        fwd_completed = sum(r['status']=='completed' for r in forwarded)
        summary = dict(status='no_running_requests' if not expected else
            'complete' if completed == len(expected) else 'incomplete_recovery',
            failed_worker=0, failed_gpu=self.args.gpus[0], failure_s=self.plan['failure_s'],
            recovery_protocol=self.plan['recovery_protocol'], latency_measurement=self.args.latency_measurement,
            recovery_scheduler=self.args.recovery_scheduler,
            recovery_execution=self.args.recovery_execution,
            recovery_order=self.args.recovery_order,
            requests=len(recovered), completed=completed, unfinished=len(recovered)-completed,
            **metric_summary(recovered), recovery_cache_policy=self.args.recovery_cache_policy,
            cache_observed_requests=sum(r['gpu_cached_tokens_first_schedule'] is not None for r in recovered),
            requests_with_gpu_cache_hits=sum((r['gpu_cached_tokens_first_schedule'] or 0)>0 for r in recovered),
            gpu_cached_prompt_tokens=sum(r['gpu_cached_tokens_first_schedule'] or 0 for r in recovered),
            forwarding=dict(status='complete' if fwd_completed==len(forwarded) else 'incomplete_forwarding',
                requests=len(forwarded), completed=fwd_completed, unfinished=len(forwarded)-fwd_completed,
                waiting=sum(r['source_queue_at_failure']=='waiting' for r in forwarded),
                future=sum(r['source_queue_at_failure']=='future' for r in forwarded), **metric_summary(forwarded)),
            by_target=[dict(worker=w,gpu=self.args.gpus[w],assigned=len(self.assignments[w]),
                completed=sum(r['status']=='completed' for r in recovered if r['target_worker']==w),
                **metric_summary([r for r in recovered if r['target_worker']==w])) for w in self.assignments])
        planned = sum(d['planned_original_requests'] for d in self.results.values())
        # 原正常完成 + running 恢复完成 + waiting/未来转发完成，三部分互不重叠。
        original_completed = sum(r['status']=='completed' and not r.get('is_forwarded')
                                 for d in self.results.values() for r in d['normal'])
        summary['traffic'] = dict(planned=planned, completed=original_completed+completed+fwd_completed,
                                 unfinished=planned-original_completed-completed-fwd_completed)
        last = max((r['completed_s'] for r in recovered if r['completed_s'] is not None), default=None)
        summary.update(recovery_completed_s=last,
                       recovery_elapsed_s=last-self.plan['failure_s'] if last is not None else None)
        measured = [r for r in recovered if r['ttft_ms'] is not None]
        ttfo=[r['ttfo_ms'] for r in measured if r.get('ttfo_ms') is not None]
        summary.update(avg_ttfo_ms=sum(ttfo)/len(ttfo) if ttfo else None,
                       p50_ttfo_ms=percentile(ttfo,50), p95_ttfo_ms=percentile(ttfo,95))
        summary['ttft_breakdown'] = {key:sum(r[key] for r in measured)/len(measured) if measured else None
                                    for key in ('queue_wait_ms','scheduled_to_first_ms')}
        if getattr(self.args, 'cpu_kv_backup_gb', 0):
            transfers=[dict(worker=w,gpu=self.args.gpus[w],**data['timing']['restore'])
                       for w,data in self.results.items() if data['timing'].get('restore') is not None]
            summary['cpu_kv'] = dict(
                source_backup=self.snapshots[0]['cpu_backup'],
                backup_tokens=sum(r['cpu_backup_tokens'] for r in recovered),
                loaded_tokens=sum(r['cpu_loaded_tokens'] for r in recovered),
                load_bytes=sum(r['cpu_load_bytes'] for r in recovered),
                load_work_ms=sum(t['h2d_gpu_ms'] for t in transfers),
                by_target_transfer=transfers,kv_restore_mode=getattr(self.args,'kv_restore_mode','serial'),
                join_wait_deferrals=sum(r.get('join_wait_deferrals',0) for r in recovered),
                initial_recovery_batches=[dict(worker=w,gpu=self.args.gpus[w],
                    batches=data['recovery_prefill_batches'][:3]) for w,data in sorted(self.results.items()) if w!=0],
                restore_protocol=('on-demand' if getattr(self.args,'kv_load_policy','eager')=='demand'
                                  else 'eager-preload'),
                preload_complete=all(r.get('preload_complete') for r in recovered),
                preload_wall_sum_ms=sum(t['total_wall_ms'] for t in transfers),
                join_work_sum_ms=sum(r['join_ms'] for r in recovered),
                evicted_running_requests=sum(r['gpu_evicted_running_requests'] for r in recovered),
                mean_prefill_to_first_tokens=(sum(r.get('prefill_to_first_tokens',0) for r in recovered)
                                              /len(recovered) if recovered else None))
        gpus = []
        failure_s = self.plan['failure_s']
        for worker, data in sorted(self.results.items()):
            before = self.snapshots[worker]
            lower = max(self.args.warmup, failure_s-self.args.window)
            before['load'] = observation_summary(data['load_rows'],data['batch_rows'],lower,failure_s)
            if worker == 0:
                continue
            normal = [r for r in data['normal'] if not r.get('is_forwarded')]
            windows = []
            begin = failure_s
            while begin < self.args.max_duration:
                end = min(begin+self.args.window, self.args.max_duration)
                windows.append(dict(start_s=begin,end_s=end,normal=metric_summary(normal,begin,end),
                    load=observation_summary(data['load_rows'],data['batch_rows'],begin,end)))
                begin = end
            gpus.append(dict(worker=worker,gpu=self.args.gpus[worker],timing=data['timing'],
                normal=metric_summary(normal,failure_s,self.args.max_duration),
                normal_planned=data['planned_original_requests'], normal_completed=sum(r['status']=='completed' for r in normal),
                post_failure=observation_summary(data['load_rows'],data['batch_rows'],failure_s,self.args.max_duration),
                windows=windows,
                recovery_prefill_batches=data['recovery_prefill_batches']))
        self.save_before()
        status = (summary['status'] if summary['status'] != 'complete' else
                  'complete' if not summary['traffic']['unfinished'] else 'incomplete')
        self.after_report = dict(status=status,summary=summary,gpus=gpus,
            metric_definitions=dict(ttft='(first_s - enqueued_s) * 1000; first output token of this attempt',
                tpot='(completed_s - first_s) * 1000 / (output_tokens - 1)',
                forwarded_ttft='(first_s - original arrival_s) * 1000'),
            recovery_requests=[compact_record(r) for r in recovered],
            forwarded_requests=[compact_record(r) for r in forwarded])
        if getattr(self.args,'record_recovery_tokens',False):
            path=Path('/tmp')/(self.output.parent.name+'_'+self.output.name+'_records.json')
            checkpoints=[dict(c,prompt_token_ids=list(c['prompt_token_ids']))
                         for c in self.snapshots[0]['checkpoints']]
            path.write_text(json.dumps(dict(config=self.config,plan=self.plan,
                checkpoints=checkpoints,recovery_requests=recovered,summary=summary,
                before_gpus=self.before_report['gpus'],
                normal_by_target=[{k:r[k] for k in ('worker','gpu','normal','post_failure','windows')} for r in gpus],
                recovery_batches=[dict(worker=w,gpu=self.args.gpus[w],batches=data.get('recovery_batches',[]))
                    for w,data in sorted(self.results.items()) if w!=0])))
            self.after_report['records_path']=str(path)
        write_result(self.output/RESULT_FILES[2], self.after_report)
        return summary


def run_workers(args, requests, arrivals, output, capture=None, context=None, pool=None):
    # 父进程持有匿名 RAM，源实例 0 退出后备份仍在；
    # 旧策略对照每轮共用一池，让不同排队策略读取同一检查点。
    """用 requests 与 arrivals 启动一组四实例，返回恢复 summary。

    父进程持有共享池并处理 IPC；capture 可接收三份报告数据，context 控制回放。
    外部传入的 pool 由调用方释放，本函数只释放自己创建的池。
    """
    owned_pool = None
    context = context or ReplayContext()
    if getattr(args, 'cpu_kv_backup_gb', 0) and pool is None:
        owned_pool = pool = SharedKVPool(args.cpu_kv_backup_gb,getattr(args,'cpu_kv_backup_bytes',None))
    ctx = mp.get_context('spawn')
    if getattr(args, 'failure_when_busy', False):
        context = replace(context, failure_event=ctx.Event())
    processes, connections, ready = [], [], []
    failure=None
    failed=False
    original_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for worker, gpu in enumerate(args.gpus):
            parent, child = ctx.Pipe()
            worker_requests, worker_arrivals = worker_workload(args, requests, arrivals, worker)
            local = copy.copy(args)
            worker_context = replace(context, cpu_handle=pool.share() if pool else None)
            local.per_instance_qps = args.instance_qps[worker]
            process = ctx.Process(target=worker_main, args=(
                worker, gpu, local, worker_requests, worker_arrivals,
                str(output), child, worker_context))
            # spawn 前就设置设备掩码，避免子进程早期导入看到其他 GPU。
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        pending = set(connections)
        deadline = time.perf_counter()+args.startup_timeout
        while pending:
            readable = wait(pending, timeout=max(0., deadline-time.perf_counter()))
            if not readable:
                raise TimeoutError('Engine startup timed out')
            for connection in readable:
                message = connection.recv()
                if message['kind'] != 'ready':
                    raise RuntimeError(str(message))
                ready.append(message)
                pending.remove(connection)
        validate_replicas(args, ready)
        start = time.perf_counter()+2
        failure = FailureCoordinator(args, connections, output, start, engines=ready,
            cpu_metadata=pool.metadata if pool is not None else None, context=context)
        for connection in connections:
            connection.send(dict(start=start))
        pending = set(connections)
        deadline = start+args.max_duration+args.recovery_drain+120
        while pending:
            with failure.profiler.stage('coordinator_wait'):
                readable = wait(pending, timeout=min(5., max(0., deadline-time.perf_counter())))
            if not readable:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Failure experiment timed out')
                continue
            for connection in readable:
                with failure.profiler.stage('ipc_receive_deserialize'):
                    message = connection.recv()
                kind = message['kind']
                if kind == 'error':
                    failed=True
                    raise RuntimeError(f"Worker {message['worker']} failed:\n"+message['traceback'])
                if failure.handle(message):
                    continue
                if kind == 'result':
                    pending.remove(connection)
                else:
                    raise RuntimeError(f'Unexpected worker message: {kind}')
        summary = failure.finalize()
        if capture is not None:
            capture.update(before=failure.before_report,plan=failure.plan,after=failure.after_report)
        return summary
    finally:
        if failure is not None and failure.profiler.enabled:
            failure.profiler.stop()
            Path(args.profile_dir,'coordinator.json').write_text(json.dumps(dict(failure.profiler.snapshot(),
                manifest_materialize_ms=failure.manifest_materialize_ms)))
        if original_visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_visible
        # 正常退出由 worker 关闭引擎；中断时先给子进程时间清理，再强制回收。
        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in connections:
            connection.close()
        if owned_pool is not None:
            owned_pool.close()


def compare_recovery_schedulers(args, requests, arrivals, output):
    """每轮采样新故障时刻，轮内首组合记录现场，其余组合严格回放。

    返回各组合报告列表；同时校验故障进度及同策略跨排队方式的目标分配一致。
    """

    results = []
    cases = args.comparison_cases or [(strategy,order) for strategy in ('equal','prefill-balanced','kv-aware')
                                     for order in args.comparison_orders]
    write_failure_comparison(output,results,vars(args))
    with tempfile.TemporaryDirectory(prefix='failure_comparison_') as temporary, ExitStack() as pools:
        for repeat,failure_s in enumerate(args.comparison_failure_times_s,1):
            cpu_pool = None
            if getattr(args, 'cpu_kv_backup_gb', 0):
                cpu_pool = SharedKVPool(args.cpu_kv_backup_gb)
                pools.callback(cpu_pool.close)
            baseline = Path(temporary)/f'repeat_{repeat}'
            baseline.mkdir()
            reference,placements = None,{}
            for index,(strategy,order) in enumerate(cases):
                local = copy.copy(args)
                local.failure_time_s = failure_s
                local.recovery_scheduler = strategy
                local.recovery_order = order
                context = ReplayContext(mode='replay' if index else 'record', directory=str(baseline))
                stage = baseline/f'{strategy}_{order}'
                stage.mkdir()
                captured = {}
                print(f'Repeat {repeat}/{args.comparison_repeats}; comparison {index+1}/{len(cases)}: {strategy}/{order}; '
                      f'planned failure={failure_s:.3f}s; pre-failure={context.mode}',flush=True)
                summary = run_workers(local,requests,arrivals,stage,capture=captured, context=context, pool=cpu_pool)
                # 故障前状态完全相同；同策略的不同排队顺序还必须保持目标分配相同。
                state = ([g['prefailure_state_signature'] for g in captured['before']['gpus']],
                         sorted((r['original_request_id'],r['source_generated_tokens'],r['source_computed_tokens'])
                                for r in captured['plan']['assignment']))
                if cpu_pool is not None:
                    state += (sorted((r['original_request_id'],r['cpu_backup_tokens'])
                                     for r in captured['plan']['assignment']),)
                if not all(g.get('prefailure_verified') for g in captured['before']['gpus']):
                    raise RuntimeError('Comparison requires verified pre-failure replay on all four instances')
                reference = state if reference is None else reference
                if state!=reference:
                    raise RuntimeError('Comparison changed the fault cohort, source progress or background state')
                placed = sorted((r['original_request_id'],r['target_worker'],r.get('recovery_group'))
                                for r in captured['plan']['assignment'])
                if placed!=placements.setdefault(strategy,placed):
                    raise RuntimeError('Queue order comparison unexpectedly changed GPU placement')
                results.append(dict(repeat=repeat,strategy=strategy,order=order,**captured))
                write_failure_comparison(output,results,vars(args))
                if (summary['status']!='complete' or summary['traffic']['unfinished']
                        or summary['forwarding']['status']!='complete'):
                    raise RuntimeError(f'{strategy}: incomplete recovery/traffic; comparison stopped')
                print(f'Repeat {repeat}: {strategy}/{order}: recovery={summary["completed"]}/{summary["requests"]}; '
                      f'TTFT={_failure_number(summary["avg_ttft_ms"])}ms; '
                      f'TPOT={_failure_number(summary["avg_tpot_ms"])}ms/token',flush=True)
            if cpu_pool is not None:
                cpu_pool.close()
    return results
