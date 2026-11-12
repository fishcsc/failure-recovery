"""所有实验共用的 CPU 输入准备，不创建推理引擎。

入口 load_trace/load_probe_traces 读取数据及到达表，worker 再调用 request_prompt。
run、sample_running 和故障 main 共用本文件，确保编号、顺序和合成 token 一致。
"""
from array import array
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import random

SEED = 20261019


def configure_inputs(args):
    """从模型配置和首条数据识别上下文、格式、Hash 粒度及时间单位，原地补全 args。"""
    args.model = str(Path(args.model).expanduser().resolve())
    args.trace = str(Path(args.trace).expanduser().resolve())
    cfg = json.loads((Path(args.model)/'config.json').read_text())
    text = cfg.get('text_config') or cfg
    model_limit = next((int(text[k]) for k in
                       ('max_position_embeddings', 'n_positions', 'max_sequence_length', 'seq_length')
                       if text.get(k)), None)
    requested = getattr(args, 'max_model_len', None)
    args.max_model_len = requested if requested is not None else model_limit
    if not args.max_model_len or args.max_model_len <= 0:
        raise ValueError('Model context length is unknown; set --max-model-len')
    if model_limit is not None and args.max_model_len > model_limit:
        raise ValueError(f'--max-model-len exceeds model config limit {model_limit}')
    args.model_vocab_size = int(text['vocab_size']) if text.get('vocab_size') else None
    args.dtype = getattr(args, 'dtype', 'auto')
    with Path(args.trace).open(encoding='utf-8') as source:
        first = next((json.loads(line) for line in source if line.strip()), None)
    if not isinstance(first, dict):
        raise ValueError('Trace must be a nonempty JSONL file of request objects')
    qa_format = {'context', 'question', 'answer'} <= first.keys() and 'input_length' not in first
    token_format = {'prompt_token_ids', 'input_length', 'output_length'} <= first.keys() and 'timestamp' not in first
    args.trace_format = 'qa' if qa_format else 'tokens' if token_format else 'trace'
    if (qa_format or token_format) and args.arrival != 'constant':
        label = 'Raw QA' if qa_format else 'Prepared token input'
        raise ValueError(f'{label} has no arrival timestamps; use --arrival constant')
    qwen_format = {'chat_id', 'parent_chat_id', 'turn'} <= first.keys()
    name = re.search(r'(?:blksz|block_size)[_-]?(\d+)', Path(args.trace).stem, re.I)
    explicit_block = getattr(args, 'trace_block_tokens', None)
    args.trace_block_tokens = (explicit_block if explicit_block is not None else
                              first.get('block_size') or
                              (int(name.group(1)) if name else 16 if qwen_format or qa_format or token_format else 512))
    if not isinstance(args.trace_block_tokens, int) or args.trace_block_tokens <= 0:
        raise ValueError('Trace block size must be a positive integer')
    unit = getattr(args, 'trace_time_unit', 'auto')
    if unit == 'auto':
        unit = first.get('timestamp_unit', 's' if qwen_format or qa_format or token_format else 'ms')
    if unit not in ('s', 'ms'):
        raise ValueError('Trace timestamps must use s or ms; set --trace-time-unit')
    args.trace_time_unit = unit


def qa_request(row, tokenizer, source_index):
    """把 QA 变成真实 tokenizer 输入，返回保留原数据标识的请求字典。

    把文档放在问题之前，使同文档不同问题保留真实可共享前缀。
    """
    if any(not isinstance(row.get(k), str) for k in ('context', 'question', 'answer')):
        raise ValueError('QA context, question and answer must be strings')
    content = f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\nAnswer:"
    tokens = tokenizer.apply_chat_template([dict(role='user', content=content)],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    return dict({k: row[k] for k in ('id', 'doc_id', 'task', 'title', 'question') if k in row},
        prompt_token_ids=tokens, input_length=len(tokens),
        output_length=max(1, len(tokenizer.encode(row['answer'], add_special_tokens=False))),
        timestamp=source_index, hash_ids=[], prompt_format='qa_context_first_no_thinking')


def load_trace(args, output):
    """读取 args.trace，过滤超长/无效输入，返回 requests 与 QPS → 到达秒数表。

    source_index 始终是原始零基行号；constant 用 i/QPS，trace 按原间隔缩放。
    output 接收输入清单；故障入口会用临时目录接收它，不增加公开结果文件。
    """
    configure_inputs(args)
    tokenizer = None
    if args.trace_format == 'qa':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    rows, excluded = [], []
    with Path(args.trace).open(encoding='utf-8') as source:
        for source_index, line in enumerate(source):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if tokenizer is not None:
                    r = qa_request(r, tokenizer, source_index)
                if args.trace_format == 'tokens':
                    # 这里只标记文件顺序，真实 constant 到达时间由下面的 i/QPS 决定。
                    r = dict(r, timestamp=source_index, timestamp_kind='source_order')
                if 'request_id' in r:
                    r.setdefault('id', r['request_id'])
                n, out = int(r['input_length']), int(r['output_length'])
                hashes, stamp = r.get('hash_ids', []), float(r['timestamp'])
                if not isinstance(hashes, list) or not math.isfinite(stamp):
                    raise ValueError('hash_ids must be a list and timestamp must be finite')
                tokens = r.get('prompt_token_ids')
                if tokens is not None and (not isinstance(tokens, list) or len(tokens) != n
                        or any(type(t) is not int or t < 0 or
                               (args.model_vocab_size and t >= args.model_vocab_size) for t in tokens)):
                    raise ValueError('prompt_token_ids must match input_length and the model vocabulary')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid trace request on line {source_index+1}: {error}') from error
            if n < 1 or out < 1 or n+out > args.max_model_len or (
                    tokens is None and len(hashes) < n//args.trace_block_tokens):
                excluded.append({'source_index': source_index,
                                 'input_length': n, 'output_length': out,
                                 'reason': 'invalid_length_or_hashes'})
                continue
            rows.append({
                **r,
                'request_id': f'TA{source_index:05d}',
                # 实验编号由原始行号生成；数据集自身的 id 仍留在 r 中，两者不是同一个字段。
                'source_index': source_index,
                'timestamp': stamp,
                'input_length': n, 'output_length': out,
                'hash_ids': hashes,
            })
    if not rows:
        raise ValueError('No valid requests in trace')

    # 普通扫描可按时间排序；复制流实验可保留数据集原文件顺序。
    if getattr(args, 'trace_order', 'timestamp') == 'timestamp':
        rows.sort(key=lambda r: (r['timestamp'], r['source_index']))
    elif args.arrival == 'trace' and any(a['timestamp'] > b['timestamp'] for a, b in zip(rows, rows[1:])):
        raise ValueError('File-order trace arrivals require nondecreasing timestamps; use constant arrival')
    time_scale = 1 if args.trace_time_unit == 's' else .001
    rel_times = [(r['timestamp']-rows[0]['timestamp'])*time_scale for r in rows]
    base_rate = ((len(rows)-1)/rel_times[-1]
                 if args.trace_format == 'trace' and rel_times[-1] > 0 else None)
    if args.arrival == 'trace' and base_rate is None:
        raise ValueError('Trace arrival mode requires a positive timestamp span')

    schedules = {}
    for qps in args.qps:
        if args.arrival == 'constant':
            required = math.ceil(qps*args.duration)
            if required > len(rows):
                raise ValueError(f'Trace has {len(rows)} eligible requests, need {required} for QPS={qps:g}')
            arrivals = [i/qps for i in range(required) if i/qps < args.duration]
        else:
            # 保留原 trace 的突发间隔形状，仅按目标平均 QPS 缩放时间。
            scale = base_rate/qps
            n = bisect.bisect_left(rel_times, args.duration/scale)
            arrivals = [x*scale for x in rel_times[:n]]
        schedules[qps] = arrivals

    n_needed = max(len(s) for s in schedules.values())
    if not n_needed:
        raise ValueError('No arrivals generated')
    requests = rows[:n_needed]
    with (output/'requests.jsonl').open('w', encoding='utf-8') as f:
        for i, r in enumerate(requests):
            instances = getattr(args, 'instances', 4)
            routing = (dict(workers=list(range(instances)))
                       if getattr(args, 'request_routing', 'round-robin') == 'replicate'
                       else dict(worker=i % instances))
            f.write(json.dumps(dict(r, **routing), ensure_ascii=False)+'\n')
    (output/'excluded.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    rate = f'{base_rate:.3f}' if base_rate is not None else 'unknown'
    print(f'Trace loaded: {len(rows)} eligible, {len(excluded)} excluded; '
          f'prepared {n_needed} requests; original eligible trace rate {rate} QPS; '
          f'hash block={args.trace_block_tokens} tokens, timestamp unit={args.trace_time_unit}',
          flush=True)
    return requests, schedules


def load_probe_traces(args, output):
    """为采样/故障实验准备请求和到达时间，返回两份一一对应的列表。

    独立源/背景模式给请求标注 workers 和 TA/TB 编号，各流按自己的速率生成到达表。
    """
    base = copy.copy(args)
    base.qps, base.duration = [args.logical_qps], args.max_duration
    fields = ('model', 'trace', 'max_model_len', 'model_vocab_size',
              'trace_block_tokens', 'trace_time_unit', 'trace_format')
    if not args.survivor_trace:
        requests, schedules = load_trace(base, output)
        for field in fields:
            setattr(args, field, getattr(base, field))
        return requests, schedules[args.logical_qps]

    requests, arrivals, inputs, excluded = [], [], [], []
    for label, trace, workers, prefix in [
            ('traceA', args.trace, [0], 'TA'),
            ('traceB', args.survivor_trace, list(range(1, args.instances)), 'TB')]:
        local = copy.copy(base)
        local.trace = trace
        rate = args.instance_qps[workers[0]] if hasattr(args, 'instance_qps') else args.logical_qps
        local.qps = [rate]
        # 同一背景流会被多个 worker 复制，不能因此把每实例 QPS 再除以实例数。
        directory = output/'input_traces'/label
        directory.mkdir(parents=True)
        rows, schedules = load_trace(local, directory)
        excluded.extend(dict(row, trace_label=label) for row in
                        json.loads((directory/'excluded.json').read_text()))
        checksum = hashlib.sha256()
        with Path(local.trace).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                checksum.update(chunk)
        tagged = [dict(row, request_id=f"{prefix}{row['source_index']:05d}",
                       workers=workers, trace_label=label, trace_file=local.trace,
                       trace_block_tokens=local.trace_block_tokens, cache_salt=None) for row in rows]
        with (directory/'requests.jsonl').open('w') as stream:
            for row in tagged:
                stream.write(json.dumps(row)+'\n')
        metadata = dict(trace=local.trace, trace_sha256=checksum.hexdigest(),
                        workers=workers, cache_salt=None, trace_block_tokens=local.trace_block_tokens,
                        trace_time_unit=local.trace_time_unit, planned_requests=len(tagged),
                        trace_format=local.trace_format,
                        per_instance_qps=rate)
        (directory/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
        inputs.append(metadata)
        requests.extend(tagged)
        arrivals.extend(schedules[rate])
        if label == 'traceA':
            for field in fields:
                setattr(args, field, getattr(local, field))
        else:
            args.survivor_trace = local.trace
    if inputs[0]['trace_sha256'] == inputs[1]['trace_sha256']:
        raise ValueError('--survivor-trace must differ from the source trace')
    args.trace_workloads = inputs
    (output/'excluded.json').write_text(json.dumps(excluded, indent=2)+'\n')
    with (output/'requests.jsonl').open('w') as stream:
        for row in requests:
            stream.write(json.dumps(row)+'\n')
    return requests, arrivals


# 各实例输入流与确定性 token 构造，供三个实验入口共用。

def make_prompt(request_id, length, hashes, allowed, block_tokens=512):
    """由 trace Hash 确定性构造 token，返回长度为 length 的输入列表。

    相同前缀 Hash 产生相同 token；不同请求的未满块尾部单独生成，不把 Hash 当模型输入。
    """
    def block(key, size):
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
        return [rng.choice(allowed) for _ in range(size)]

    parent = hashlib.sha256(f'toolagent-token-v1:{SEED}'.encode()).digest()
    tokens = []
    full, tail = divmod(length, block_tokens)
    for depth in range(full):
        part = json.dumps(hashes[depth], sort_keys=True).encode()
        parent = hashlib.sha256(parent + b'/' + part).digest()
        tokens.extend(block(parent, block_tokens))
    if tail:
        tokens.extend(block(parent + b'/private-tail/' + request_id.encode(), tail))
    return tokens


def request_prompt(row, allowed, block_tokens=512):
    """有真实 prompt_token_ids 时原样使用，否则按 trace 块粒度生成确定性输入。"""
    if 'prompt_token_ids' in row:
        return row['prompt_token_ids']
    return make_prompt(row['request_id'], row['input_length'], row['hash_ids'], allowed,
                       row.get('trace_block_tokens', block_tokens))


def worker_workload(args, requests, arrivals, worker):
    """按 worker 取对应请求/到达表：独立 trace、完整复制或历史轮询分片。"""
    if getattr(args, 'survivor_trace', None):
        selected = [(row, stamp) for row, stamp in zip(requests, arrivals) if worker in row['workers']]
        return [row for row, _ in selected], [stamp for _, stamp in selected]
    if args.request_routing == 'replicate':
        return requests, arrivals
    return requests[worker::args.instances], arrivals[worker::args.instances]


def validate_replicas(args, ready):
    """ready 后核对同一复制组的输入指纹；源/背景本来不同，不能跨组比较。"""
    if args.request_routing != 'replicate':
        return
    groups = ([ready] if not args.survivor_trace else
              [[r for r in ready if r['worker'] == 0], [r for r in ready if r['worker'] != 0]])
    for engines in groups:
        for key in ('request_sequence_sha256', 'arrivals_sha256', 'prompt_tokens_sha256'):
            if len({engine['identity'][key] for engine in engines}) != 1:
                raise RuntimeError(f'Replicated instances disagree on {key}')


def workload_identity(requests, arrivals):
    """返回请求顺序、计划到达和实际输入 token 的指纹，供启动时确认副本一致。"""
    token_hash = hashlib.sha256()
    for row in requests:
        token_hash.update(json.dumps([row['request_id'], row['input_length'], row['output_length']]).encode())
        token_hash.update(array('I', row['tokens']).tobytes())
    return dict(planned_requests=len(requests),
                first_request_ids=[r['request_id'] for r in requests[:10]],
                request_sequence_sha256=hashlib.sha256(
                    json.dumps([r['request_id'] for r in requests]).encode()).hexdigest(),
                arrivals_sha256=hashlib.sha256(json.dumps(arrivals).encode()).hexdigest(),
                prompt_tokens_sha256=token_hash.hexdigest())


