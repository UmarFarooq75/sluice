#!/bin/bash
# Gap-2 harness (D10): n>=3 repeats of the headline configs with medians, so
# published numbers carry bands instead of single-run point estimates.
# Runs the pair back-to-back per round so thermal drift hits both equally.
# USAGE: bash scripts/bench_repeat.sh [N_ROUNDS]   (default 3; ~15 min/round)
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
N=${1:-3}
for i in $(seq 1 "$N"); do
  pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
  [ "${pct:-0}" -lt 40 ] && { echo "ABORT round $i: memory pressure"; exit 1; }
  pgrep -f "stream_run.*gguf" && { echo "ABORT: engine busy"; exit 1; }
  for cfg in "m0_s8 0" "m025_s8 0.25" "m125_s8 1.25"; do
    set -- $cfg  # bash: intentional word split
    echo "== round $i $1 =="
    env LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=$2 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > "results/rep_${1}_r${i}.txt" 2>&1
    grep -E 'decode:|logits_hash' "results/rep_${1}_r${i}.txt"
  done
done
echo "== medians =="
for c in m0_s8 m025_s8 m125_s8; do
  vals=$(grep -h 'decode:' results/rep_${c}_r*.txt | grep -oE '\(([0-9.]+)' | tr -d '(' | sort -n)
  echo "$c: [$(echo "$vals" | tr '\n' ' ')] median=$(echo "$vals" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
done
echo "REPEAT BENCH DONE"
