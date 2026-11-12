#!/usr/bin/env python3
"""Measure position-dependent prefill cost vs. fixed-size KV H2D load cost.

Outputs THREE figures, each in PNG/PDF/SVG formats:
  prefill_chunk_latency.png   : Chunk Index vs. Resuming Time for Compute (ms)
  h2d_kv_chunk_latency.png    : Chunk Index vs. Resuming Time for Load (ms)
  compute_vs_load.png         : Compute + zoomed Load (broken left y-axis)

Also saves summary CSVs, raw-sample CSVs, and run_info.json.

Example:
  CUDA_VISIBLE_DEVICES=0 python prefill_chunk_position_benchmark_paper.py \
    --model /data1/jhcheng/models/scchen/Qwen3-8B \
    --context-length 16384 --chunk-size 512 --repeats 10 \
    --warmup-runs 2 --h2d-repeats 20 \
    --output-dir ./prefill_position_8b

To redraw figures from existing CSVs without a GPU, add --plot-only.

Important: the plotted time is the time of *each fixed-size chunk*, not the
full end-to-end request recovery time. Prefill uses HF DynamicCache, whose KV
concatenation/copy overhead is included. H2D uses a pinned, contiguous KV-sized
buffer and does not include vLLM paged-KV allocation / APC registration.
"""

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', default='/data1/jhcheng/models/scchen/Qwen3-8B')
    parser.add_argument('--context-length', type=int, default=16384)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=10,
                        help='Recorded full-context model passes')
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--attn-impl', choices=['sdpa', 'flash_attention_2'],
                        default='sdpa')
    parser.add_argument('--prompt-file',
                        help='UTF-8 text prompt (repeated if shorter than context length)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h2d-repeats', type=int, default=20)
    parser.add_argument('--output-dir', default='./prefill_chunk_position_results')
    parser.add_argument('--plot-only', action='store_true',
                        help='Redraw figures from existing CSV files, without GPU inference')

    # Backward compatibility with the previous script's example command.
    # Both experiments always run; the standalone attention experiment is removed.
    parser.add_argument('--measure-h2d', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attention-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attention-repeats', type=int, default=20, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.context_length <= 0 or args.chunk_size <= 0:
        parser.error('context-length and chunk-size must be positive')
    if args.context_length % args.chunk_size:
        parser.error('context-length must be divisible by chunk-size')
    if args.repeats < 1 or args.warmup_runs < 0 or args.h2d_repeats < 1:
        parser.error('Invalid repeat count')
    if args.attention_only:
        print('Note: --attention-only is ignored; only Compute and Load are now measured.', flush=True)
    return args


def timer_once(torch, fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def summarize(per_chunk, chunk_size):
    rows = []
    for idx, measurements in enumerate(per_chunk):
        rows.append({
            'chunk_index': idx + 1,                  # 1-based, matches figure X axis
            'start_token': idx * chunk_size,         # token offset remains 0-based
            'end_token': (idx + 1) * chunk_size,
            'previous_context_tokens': idx * chunk_size,
            'mean_ms': round(statistics.mean(measurements), 4),
            'std_ms': round(statistics.pstdev(measurements), 4),
            'median_ms': round(statistics.median(measurements), 4),
            'min_ms': round(min(measurements), 4),
            'max_ms': round(max(measurements), 4),
            'samples': len(measurements),
        })
    return rows


def save_raw_csv(path, samples, chunk_size):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_index', 'start_token', 'run_index', 'gpu_ms'])
        for idx, measurements in enumerate(samples):
            for rep, ms in enumerate(measurements):
                writer.writerow([idx + 1, idx * chunk_size, rep + 1, round(ms, 6)])


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Research-paper figure settings.  Colour and text are centralised here.
COMPUTE_COLOR = '#0B3C8C'  # dark blue
LOAD_COLOR = '#C82423'     # red
AXIS_LABEL_SIZE = 12
TICK_SIZE = 10.5
LEGEND_SIZE = 10.5


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,   # editable TrueType text, no fonts bundled
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.labelsize': AXIS_LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'axes.linewidth': 1.0,
        'savefig.facecolor': 'white',
    })
    return plt


def _save_paper_fig(fig, png_path):
    png_path = Path(png_path)
    for extension in ('.png', '.pdf', '.svg'):
        kwargs = {'dpi': 350} if extension == '.png' else {}
        fig.savefig(png_path.with_suffix(extension), bbox_inches='tight', **kwargs)


def plot_curves(path, curves, ylabel, title):
    """Original mean-only single-curve plot with publication-resolution exports."""
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    single_curve = len(curves) == 1
    for rows, label in curves:
        x = [int(r['chunk_index']) for r in rows]
        y = [float(r['mean_ms']) for r in rows]
        color = COMPUTE_COLOR if 'Compute' in label else LOAD_COLOR
        ax.plot(x, y, color=color, marker='o', markersize=3.7,
                linewidth=2.0, label='Mean' if single_curve else label)
    ax.set(xlabel='Chunk Index', ylabel=ylabel, title=title)
    ax.grid(alpha=0.19)
    ax.legend(frameon=False)
    _save_paper_fig(fig, path)
    plt.close(fig)


def plot_combined_broken_axis(path, compute_rows, load_rows):
    """Paper plot: two left-side y scales linked by explicit break marks.

    Crucially both y scales are in milliseconds. The bottom panel zooms in on
    Load around 3 ms; the upper shows Compute. Break marks make the differing
    scales obvious. A dashed 3-ms guide meets the *left* y-axis at tick 3.
    """
    import numpy as np
    plt = _plot_setup()
    xc = np.array([int(r['chunk_index']) for r in compute_rows])
    yc = np.array([float(r['mean_ms']) for r in compute_rows])
    xl = np.array([int(r['chunk_index']) for r in load_rows])
    yl = np.array([float(r['mean_ms']) for r in load_rows])
    if not len(xc) or not len(xl):
        raise ValueError('Need non-empty Compute and Load CSV inputs')

    fig = plt.figure(figsize=(7.0, 4.7))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.075,
                           left=0.15, right=0.97, top=0.95, bottom=0.14)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)

    line_top, = top.plot(xc, yc, color=COMPUTE_COLOR, marker='o',
                         markersize=3.7, linewidth=2.15, label='Compute (Prefill)')
    line_bottom, = bottom.plot(xl, yl, color=LOAD_COLOR, marker='o',
                              markersize=3.7, linewidth=2.15, label='Load (H2D)')

    # Keep the 3 ms line legible in the lower (magnified) panel.
    # It is a reference, not a fabricated measurement.
    lower_min = min(float(np.min(yl)), 3.0)
    lower_max = max(float(np.max(yl)), 3.0)
    margin = max(0.12, (lower_max - lower_min) * 0.35)
    bottom.set_ylim(lower_min - margin, lower_max + margin)
    upper_min, upper_max = float(np.min(yc)), float(np.max(yc))
    upper_span = max(upper_max - upper_min, 5.0)
    top.set_ylim(upper_min - 0.09 * upper_span, upper_max + 0.10 * upper_span)

    # A subtle dashed reference at exactly 3.0 ms reaches the left y-axis.
    bottom.axhline(3.0, color=LOAD_COLOR, linewidth=1.05,
                   linestyle=(0, (4.0, 3.5)), alpha=0.72, zorder=1)
    # Guarantee a numerical '3' printed as a genuine *left-side* y tick.
    lo, hi = bottom.get_ylim()
    tick_values = [3.0]
    if lo <= 2.5 <= hi and abs(lo - 2.5) > 0.04:
        tick_values.insert(0, 2.5)
    if lo <= 3.5 <= hi and abs(hi - 3.5) > 0.04:
        tick_values.append(3.5)
    bottom.set_yticks(tick_values)
    bottom.set_yticklabels([f'{v:g}' for v in tick_values])

    # Shared x-axis with a visible break on BOTH sides of the two panels.
    top.spines['bottom'].set_visible(False)
    bottom.spines['top'].set_visible(False)
    top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    bottom.tick_params(axis='x', top=False)
    d = 0.010
    kw = dict(color='#454545', clip_on=False, linewidth=1.3)
    top.plot((-d, +d), (-d, +d), transform=top.transAxes, **kw)
    top.plot((1-d, 1+d), (-d, +d), transform=top.transAxes, **kw)
    bottom.plot((-d, +d), (1-d, 1+d), transform=bottom.transAxes, **kw)
    bottom.plot((1-d, 1+d), (1-d, 1+d), transform=bottom.transAxes, **kw)

    top.set_xlim(min(xc.min(), xl.min()) - 0.35, max(xc.max(), xl.max()) + 0.35)
    bottom.set_xlabel('Chunk Index')
    fig.supylabel('Resuming Time (ms)', x=0.025, fontsize=AXIS_LABEL_SIZE)
    top.grid(alpha=0.18)
    bottom.grid(alpha=0.16)
    # Figure-level legend doesn't hide data or place load near bottom of plot.
    top.legend(handles=[line_top, line_bottom], loc='upper left',
               frameon=False, handlelength=2.3)
    _save_paper_fig(fig, path)
    plt.close(fig)


def plot_outputs(out, compute_rows, load_rows, chunk_size, nbytes):
    plot_curves(out / 'prefill_chunk_latency.png',
                [(compute_rows, 'Compute (Prefill)')],
                'Resuming Time for Compute (ms)',
                f'Qwen3 chunked prefill ({chunk_size} tokens/chunk)')
    plot_curves(out / 'h2d_kv_chunk_latency.png',
                [(load_rows, 'Load (H2D)')],
                'Resuming Time for Load (ms)',
                f'Fixed KV-sized copy ({nbytes/1024**2:.1f} MiB/chunk)')
    plot_combined_broken_axis(out / 'compute_vs_load.png', compute_rows, load_rows)


def load_summary_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def make_input(torch, args, tokenizer_factory, vocab_size):
    if args.prompt_file:
        content = Path(args.prompt_file).read_text(encoding='utf-8')
        tokenizer = tokenizer_factory.from_pretrained(args.model, trust_remote_code=False)
        encoded = tokenizer.encode(content, add_special_tokens=False)
        if not encoded:
            raise ValueError('Prompt file tokenizes to an empty sequence')
        repeats = (args.context_length + len(encoded) - 1) // len(encoded)
        ids = (encoded * repeats)[:args.context_length]
        return torch.tensor(ids, dtype=torch.long, device='cuda')[None, :]
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    return torch.randint(0, vocab_size, (1, args.context_length),
                         generator=generator, dtype=torch.long).to('cuda')


def run_model_pass(torch, backbone, token_ids, args, record):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    steps = args.context_length // args.chunk_size
    latencies = []
    with torch.inference_mode():
        for idx in range(steps):
            a, b = idx * args.chunk_size, (idx + 1) * args.chunk_size
            chunk = token_ids[:, a:b]
            positions = torch.arange(a, b, device='cuda', dtype=torch.long)

            def forward():
                # The backbone excludes lm_head; produces the KV for this chunk.
                return backbone(input_ids=chunk, past_key_values=cache,
                                use_cache=True, cache_position=positions,
                                return_dict=False)

            if record:
                ms, result = timer_once(torch, forward)
                latencies.append(ms)
            else:
                result = forward()
                torch.cuda.synchronize()
            del result

    if cache.get_seq_length() != args.context_length:
        raise RuntimeError(f'KV length {cache.get_seq_length()} != {args.context_length}')
    del cache
    return latencies


def measure_h2d(torch, args, num_layers, num_kv_heads, head_dim, dtype):
    """Fixed-size contiguous pinned CPU -> GPU copy per chunk index."""
    shape = (2, num_layers, num_kv_heads, args.chunk_size, head_dim)
    try:
        source = torch.empty(shape, dtype=dtype, pin_memory=True)
    except (RuntimeError, OSError) as exc:
        raise RuntimeError('Pinned host memory allocation failed') from exc
    source.zero_()
    target = torch.empty(shape, dtype=dtype, device='cuda')

    def copy():
        target.copy_(source, non_blocking=True)

    # Warm the pinned-memory -> GPU DMA path before recording Chunk 1.
    # This reduces cold-start bias without discarding recorded samples.
    with torch.inference_mode():
        for _ in range(100):
            copy()
        torch.cuda.synchronize()

    by_chunk = []
    with torch.inference_mode():
        for _ in range(args.context_length // args.chunk_size):
            for _ in range(3):
                copy()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.h2d_repeats):
                ms, _ = timer_once(torch, copy)
                samples.append(ms)
            by_chunk.append(samples)

    nbytes = source.numel() * source.element_size()
    del source, target
    return by_chunk, nbytes


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if args.plot_only:
        compute_rows = load_summary_csv(out / 'prefill_chunk_latency.csv')
        load_rows = load_summary_csv(out / 'h2d_kv_chunk_latency.csv')
        info_path = out / 'run_info.json'
        if not info_path.exists():
            raise SystemExit('Missing run_info.json for H2D size; run a measurement first')
        info = json.loads(info_path.read_text(encoding='utf-8'))
        plot_outputs(out, compute_rows, load_rows, args.chunk_size,
                     int(info['h2d_bytes_per_chunk']))
        print(f'Redrawn 3 figures as PNG/PDF/SVG in: {out.resolve()}')
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required')
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install transformers (Qwen3 support) and accelerate') from exc

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16
    device_info = torch.cuda.get_device_name(0)
    print(f'GPU: {device_info}, model: {args.model}, '
          f'context: {args.context_length}, chunk: {args.chunk_size}', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl,
        device_map={'': 'cuda:0'}, low_cpu_mem_usage=True, trust_remote_code=False,
    ).eval()
    if not hasattr(model, 'model'):
        raise RuntimeError('Expected model.model backbone: ' + type(model).__name__)
    cfg = model.config
    backbone = model.model
    ids = make_input(torch, args, AutoTokenizer, cfg.vocab_size)

    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup_runs):
        print(f'Prefill warmup {i + 1}/{args.warmup_runs}', flush=True)
        run_model_pass(torch, backbone, ids, args, record=False)
        gc.collect()

    compute_samples = [[] for _ in range(args.context_length // args.chunk_size)]
    for i in range(args.repeats):
        print(f'Prefill pass {i + 1}/{args.repeats}', flush=True)
        times = run_model_pass(torch, backbone, ids, args, record=True)
        for j, ms in enumerate(times):
            compute_samples[j].append(ms)
        gc.collect()

    print('Measuring H2D (fixed KV bytes per chunk)', flush=True)
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    load_samples, nbytes = measure_h2d(
        torch, args, cfg.num_hidden_layers, kv_heads, head_dim, dtype
    )
    compute_rows = summarize(compute_samples, args.chunk_size)
    load_rows = summarize(load_samples, args.chunk_size)

    save_csv(out / 'prefill_chunk_latency.csv', compute_rows)
    save_raw_csv(out / 'prefill_chunk_latency_raw.csv', compute_samples, args.chunk_size)
    save_csv(out / 'h2d_kv_chunk_latency.csv', load_rows)
    save_raw_csv(out / 'h2d_kv_chunk_latency_raw.csv', load_samples, args.chunk_size)

    plot_outputs(out, compute_rows, load_rows, args.chunk_size, nbytes)

    info = {
        'model': args.model, 'gpu': device_info, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'attention_impl': args.attn_impl,
        'cache_type': 'DynamicCache', 'context_length': args.context_length,
        'chunk_size': args.chunk_size, 'num_chunks': len(compute_rows),
        'repeats': args.repeats, 'warmup_runs': args.warmup_runs,
        'h2d_repeats': args.h2d_repeats, 'h2d_bytes_per_chunk': nbytes,
        'h2d_mib_per_chunk': round(nbytes / 1024**2, 4),
        'peak_gpu_allocated_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'compute_note': 'Chunk-level Transformer time including HF DynamicCache append/copy.',
        'load_note': 'Contiguous pinned H2D, excluding paged-KV restore and APC.',
    }
    (out / 'run_info.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'Compute: first {compute_rows[0]["mean_ms"]:.2f} ms, '
          f'last {compute_rows[-1]["mean_ms"]:.2f} ms', flush=True)
    print(f'Load: fixed {nbytes / 1024**2:.1f} MiB/chunk, '
          f'mean {statistics.mean(row["mean_ms"] for row in load_rows):.2f} ms', flush=True)
    print('Saved: prefill_chunk_latency.png, h2d_kv_chunk_latency.png, '
          'compute_vs_load.png (also PDF/SVG, plus CSV and run_info.json)', flush=True)
    print(f'Output: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
