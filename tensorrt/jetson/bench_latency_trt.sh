#!/bin/bash
# Host wrapper for tensorrt/bench_latency_trt.py: records tegrastats (200 ms) on the host
# for the whole run, runs the benchmark inside the jetson-llm container, then summarizes
# with the llama.cpp side's own summarizer (scripts/bench_latency.py summarize).
#
# usage: bash tensorrt/jetson/bench_latency_trt.sh <run name> [extra bench args]
#   e.g. bash tensorrt/jetson/bench_latency_trt.sh full
#        bash tensorrt/jetson/bench_latency_trt.sh test --rounds 1 --limit-per-dataset 3 --min-cooldown 5
# Run it inside tmux, with the desktop stopped (sudo systemctl stop display-manager).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="$1"; shift
OUT="results_latency_trt/$NAME"
cd "$REPO"
if [ -e "$OUT/requests.jsonl" ]; then echo "ERROR: $OUT already has requests.jsonl"; exit 1; fi
mkdir -p "$OUT"
{ date; uname -a; nvpmodel -q 2>/dev/null; systemctl is-active display-manager; swapon --show; free -m; } > "$OUT/board_state.txt" 2>&1
tegrastats --interval 200 --logfile "$OUT/tegrastats.log" &
TP=$!
bash tensorrt/jetson/run_in_container.sh python3 tensorrt/bench_latency_trt.py run --out "$OUT" "$@" 2>&1 | tee "$OUT/console.log"
RC=${PIPESTATUS[0]}
kill $TP 2>/dev/null; sleep 1
bash tensorrt/jetson/run_in_container.sh python3 scripts/bench_latency.py summarize "$OUT" --tegrastats "$OUT/tegrastats.log" 2>&1 \
  | grep -v -i -E "copyright|idiap|caffe|google inc|yangqing|rights reserved|license|SHMEM|insufficient|docker run|^$" \
  | tee "$OUT/summary.txt"
exit $RC
