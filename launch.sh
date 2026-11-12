#!/usr/bin/env bash
# Usage: bash launch.sh SESSION OUTPUT [run.py arguments...]
set -u
set -o pipefail
worker=0
if [[ "${1:-}" == --worker ]]; then worker=1; shift; fi
if [[ $# -lt 2 ]]; then
    printf 'Usage: bash launch.sh SESSION OUTPUT [run.py arguments...]\n' >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit 1
session=$1
output=$(realpath -m -- "$2")
shift 2
python_bin=${BENCH_PYTHON:-/data1/scchen/envs/vllm_dev/bin/python}
if [[ "$worker" -eq 0 ]]; then
    if [[ ! "$session" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        printf 'Session name must contain only letters, digits, _ and -.\n' >&2
        exit 2
    fi
    for path in "$output" "$output.log" "$output.status"; do
        if [[ -e "$path" ]]; then printf 'Already exists: %s\n' "$path" >&2; exit 1; fi
    done
    if tmux has-session -t "=$session" 2>/dev/null; then
        printf 'Session already exists: %s\n' "$session" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "$output")" || exit 1
    printf -v command '%q ' env "BENCH_PYTHON=$python_bin" bash "$script_dir/launch.sh" --worker "$session" "$output" "$@"
    tmux new-session -d -s "$session" -c "$script_dir" "$command" || exit 1
    tmux set-option -w -t "$session:0" remain-on-exit on
    printf 'Started: %s\nLog: %s.log\nStatus: %s.status\n' "$session" "$output" "$output"
    exit 0
fi

printf 'RUNNING\n' > "$output.status"
"$python_bin" -u run.py "$@" --output "$output" 2>&1 | tee "$output.log"
codes=("${PIPESTATUS[@]}")
printf 'python_exit_code=%s\nlog_exit_code=%s\n' "${codes[0]}" "${codes[1]}" > "$output.status"
summary_code=0
if [[ -f "$output/config.json" ]]; then
    "$python_bin" report.py "$output" 2>&1 | tee -a "$output.log"
    summary_code=$?
    printf 'summary_exit_code=%s\n' "$summary_code" >> "$output.status"
fi
if [[ "${codes[0]}" -ne 0 ]]; then exit "${codes[0]}"; fi
if [[ "${codes[1]}" -ne 0 ]]; then exit "${codes[1]}"; fi
exit "$summary_code"
