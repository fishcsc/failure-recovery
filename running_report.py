# 离线读取 sample_running 的观测，输出稳定区间、时间曲线和固定请求数组统计。
# 入口 main，使用 metrics/report；只分析已有结果，不启动推理。
"""Export time-windowed TTFT/TPOT and time-weighted running/waiting from a probe."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

from metrics import percentile
from report import write_csv


def read_rows(path):
    # 读取运行中日志时，最后一行可能尚未写完，只接收完整行。
    with path.open() as stream:
        for line in stream:
            if line.endswith('\n'):
                yield json.loads(line)


def stable_probe_summary(root):
    """单实例扫描只在同一批合格区间计算延迟和时长加权负载，返回稳定区间摘要。"""
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    summary, = json.loads((root/'summary.json').read_text())
    gpu, = config['gpus']
    records = list(read_rows(root/f'latencies_gpu{gpu}.jsonl'))
    load = list(read_rows(root/f'load_gpu{gpu}.jsonl'))
    periods = [(p['start_s'], min(p['end_s'], config['max_duration']))
               for p in summary['stable_periods']]

    def measure(intervals):
        seconds, running, waiting, queue_seconds = 0., 0., 0., 0.
        weights = defaultdict(float)
        for row in load:
            duration = sum(max(0., min(row['end_s'], end)-max(row['start_s'], begin))
                           for begin, end in intervals)
            seconds += duration
            running += duration*row['running'];waiting += duration*row['waiting']
            queue_seconds += duration*(row['waiting']>0)
            weights[row['running']] += duration
        p95, cumulative = None, 0.
        for count, duration in sorted(weights.items()):
            cumulative += duration
            if seconds and cumulative >= .95*seconds:
                p95 = count;break
        result = dict(seconds=seconds, mean_running=running/seconds if seconds else None,
            p95_running=p95, mean_waiting=waiting/seconds if seconds else None,
            waiting_fraction=queue_seconds/seconds if seconds else None)
        for name, event in [('ttft', 'first_s'), ('tpot', 'completed_s')]:
            values = [r[f'{name}_ms'] for r in records if r.get(event) is not None
                      and r.get(f'{name}_ms') is not None
                      and any(begin<=r[event]<end for begin,end in intervals)]
            result.update({f'{name}_samples':len(values),
                f'avg_{name}_ms':sum(values)/len(values) if values else None,
                f'p95_{name}_ms':percentile(values,95)})
        return result

    checks = [r for r in read_rows(root/f'stability_gpu{gpu}.jsonl') if r['reason']=='periodic']
    stable = measure(periods)
    return dict(qps=summary['per_instance_qps'], gpu=gpu, status=summary['status'],
        qualified=(summary['status']=='complete' and checks[-1]['qualified']
                   and stable['seconds']>=config['measurement_duration']),
        stable=stable, stable_periods=periods, final_checks=checks[-1]['checks'],
        post_warmup=measure([(config['warmup'],config['max_duration'])]),
        overdue_without_first=sum(r['first_s'] is None and
            r['arrival_s']<=config['max_duration']-10 for r in records),
        unfinished=summary['latency']['unfinished'],
        planned=len(list(read_rows(root/'requests.jsonl'))))


def time_series(root, width):
    """把负载区间按秒数切进时间桶，延迟事件按发生时间入桶，返回逐卡曲线数据。"""
    config = json.loads((root/'config.json').read_text())
    results = []
    for gpu in config['gpus']:
        bins = defaultdict(lambda: dict(covered=0., running=0., waiting=0., outstanding=0.,
                                       ttft=[], tpot=[], admitted=0, completed=0))
        end = 0.
        for row in read_rows(root/f'load_gpu{gpu}.jsonl'):
            begin, end = row['start_s'], min(row['end_s'], config['max_duration'])
            for index in range(int(begin//width), math.ceil(end/width)):
                seconds = max(0., min(end, (index+1)*width)-max(begin, index*width))
                bucket = bins[index]
                bucket['covered'] += seconds
                for key in ('running', 'waiting', 'outstanding'):
                    bucket[key] += seconds*row[key]
        for event in read_rows(root/f'events_gpu{gpu}.jsonl'):
            if event['time_s'] >= end:
                continue
            bucket = bins[int(event['time_s']//width)]
            if event['event'] == 'admitted':
                bucket['admitted'] += 1
            elif event['event'] == 'first_token':
                bucket['ttft'].append(event['ttft_ms'])
            elif event['event'] == 'completed':
                bucket['completed'] += 1
                if event['tpot_ms'] is not None:
                    bucket['tpot'].append(event['tpot_ms'])
        for index, bucket in sorted(bins.items()):
            seconds = bucket['covered']
            if not seconds:
                continue
            row = dict(gpu=gpu, start_s=index*width, end_s=min((index+1)*width, end),
                       covered_s=seconds, admitted=bucket['admitted'], completed=bucket['completed'])
            for key in ('running', 'waiting', 'outstanding'):
                row[f'mean_{key}'] = bucket[key]/seconds
            for key in ('ttft', 'tpot'):
                values = bucket[key]
                row[f'{key}_requests'] = len(values)
                row[f'avg_{key}_ms'] = sum(values)/len(values) if values else None
                row[f'p95_{key}_ms'] = percentile(values, 95)
            results.append(row)
    return results, config


def request_groups(root, size):
    """按到达顺序组成固定请求数的批组，保留未完成和未提交样本，避免只看成功快请求。"""
    config = json.loads((root/'config.json').read_text())
    manifest = list(read_rows(root/'requests.jsonl'))
    results = []
    for worker, gpu in enumerate(config['gpus']):
        planned = ([r for r in manifest if worker in r['workers']] if manifest and 'workers' in manifest[0] else
                   manifest if config.get('request_routing') == 'replicate' else
                   [r for i, r in enumerate(manifest) if r.get('worker', i % len(config['gpus'])) == worker])
        actual = {r['request_id']: r for r in read_rows(root/f'latencies_gpu{gpu}.jsonl')}
        for start in range(0, len(planned), size):
            cohort = planned[start:start+size]
            records = [actual[r['request_id']] for r in cohort if r['request_id'] in actual]
            completed = sum(r['status'] == 'completed' for r in records)
            row = dict(gpu=gpu, group=start//size+1, start_request_index=start,
                       first_request_id=cohort[0]['request_id'], last_request_id=cohort[-1]['request_id'],
                       planned_requests=len(cohort), submitted_requests=len(records),
                       completed_requests=completed, unfinished_requests=len(records)-completed,
                       not_submitted_requests=len(cohort)-len(records),
                       fully_completed=int(completed == len(cohort)),
                       fully_post_warmup=int(len(records) == len(cohort) and
                                             all(r['arrival_s'] >= config['warmup'] for r in records)),
                       arrival_start_s=min((r['arrival_s'] for r in records), default=None),
                       arrival_end_s=max((r['arrival_s'] for r in records), default=None),
                       mean_input_length=sum(r['input_length'] for r in cohort)/len(cohort),
                       mean_output_length=sum(r['output_length'] for r in cohort)/len(cohort))
            for key in ('ttft', 'tpot'):
                values = [r[f'{key}_ms'] for r in records if r[f'{key}_ms'] is not None]
                row[f'{key}_requests'] = len(values)
                row[f'avg_{key}_ms'] = sum(values)/len(values) if values else None
                row[f'p50_{key}_ms'] = percentile(values, 50)
                row[f'p95_{key}_ms'] = percentile(values, 95)
            results.append(row)
    return results, config


def plot(rows, config, path, group_size=None):
    # 调用已有 R 绘图，不为离线画图修改推理环境依赖。
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1]); stem <- args[2]; heading <- args[3]; warmup <- as.numeric(args[4])
group_mode <- as.numeric(args[5]) > 0
gpus <- sort(unique(d$gpu)); colors <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7")
keys <- c("avg_ttft_ms", "avg_tpot_ms", "mean_running", "mean_waiting")
labels <- c("TTFT (seconds)", "TPOT (ms/token)", "Running requests", "Waiting requests")
if (group_mode) {
  keys <- c("avg_ttft_ms", "avg_tpot_ms", "mean_input_length", "mean_output_length")
  labels <- c("TTFT (seconds)", "TPOT (ms/token)", "Input tokens / request", "Output tokens / request")
  d$avg_ttft_ms[d$ttft_requests < d$planned_requests] <- NA
  d$avg_tpot_ms[d$fully_completed == 0] <- NA
  xlimits <- range(d$group)
  xtext <- paste("Arrival-ordered request group (",args[5]," requests/group)",sep="")
  clean <- d$group[d$fully_post_warmup == 1]
  warmup <- if (length(clean)) min(clean)-.5 else NA
} else {
  xlimits <- range(c(d$start_s,d$end_s)); xtext <- "Replay time (seconds)"
}
for (kind in c("png", "svg")) {
  if (kind == "png") png(paste0(stem,".png"), width=1600, height=1000, res=130, type="cairo")
  else svg(paste0(stem,".svg"), width=12.3, height=7.7)
  par(mfrow=c(2,2), mar=c(4.1,4.4,1.2,1), oma=c(2.5,0,3.2,0))
  for (i in seq_along(keys)) {
    scale <- if (i == 1) .001 else 1
    values <- d[[keys[i]]]*scale
    ymax <- if (all(is.na(values))) 1 else max(1, values, na.rm=TRUE)*1.07
    plot(NA, xlim=xlimits, ylim=c(0,ymax), xlab=xtext, ylab=labels[i])
    grid(col="#E5E5E5"); if (!is.na(warmup)) abline(v=warmup, lty=2, col="#999999")
    for (j in seq_along(gpus)) {
      z <- d[d$gpu == gpus[j],]
      xpos <- if (group_mode) z$group else (z$start_s+z$end_s)/2
      lines(xpos, z[[keys[i]]]*scale, col=colors[(j-1)%%4+1], lwd=1.8)
    }
    legend("topleft", legend=paste("GPU",gpus), col=rep(colors,length.out=length(gpus)),
           lty=1, lwd=1.8, bty="n", cex=.8)
  }
  mtext(heading, outer=TRUE, side=3, line=1, cex=1.15)
  caption <- if (group_mode) "Same arrival-ordered request cohorts for TTFT and TPOT; incomplete cohorts hidden from TPOT plot."
             else "TTFT: first-token windows; TPOT: completion windows; load: time-weighted."
  boundary <- if (group_mode) "Dashed line: first wholly post-warmup group; groups are not stability labels."
              else "Dashed line: minimum warmup, not a stability boundary."
  mtext(caption, outer=TRUE, side=1, line=.3, cex=.8)
  mtext(boundary, outer=TRUE, side=1, line=1.4, cex=.8)
  dev.off()
}
'''
    scope = 'per instance' if config.get('qps_scope') in ('per_instance', 'per-instance') else 'global'
    title = f"QPS={config['qps']:g} ({scope}), {Path(config['model']).name}, {Path(config['trace']).name}"
    subprocess.run(['Rscript', '--vanilla', '-e', code, str(path)+'.csv', str(path),
                    title, str(config['warmup']), str(group_size or 0)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--bin-seconds', type=float, default=10)
    parser.add_argument('--requests-per-group', type=int,
                        help='Group by dataset arrival order instead of time, preserving incomplete cohorts')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0:
        parser.error('--bin-seconds must be positive and finite')
    if args.requests_per_group is not None and args.requests_per_group <= 0:
        parser.error('--requests-per-group must be positive')
    root = args.directory.resolve()
    rows, config = (request_groups(root, args.requests_per_group) if args.requests_per_group
                    else time_series(root, args.bin_seconds))
    output = root/'time_series'
    output.mkdir(exist_ok=True)
    stem = output/(f'request_groups_{args.requests_per_group}' if args.requests_per_group
                   else f'time_series_{args.bin_seconds:g}s')
    csv_path = Path(str(stem)+'.csv')
    write_csv(csv_path, rows)
    if args.plot:
        plot(rows, config, stem, args.requests_per_group)
    print(f'{len(rows)} GPU/windows saved to {csv_path}')


if __name__ == '__main__':
    main()
