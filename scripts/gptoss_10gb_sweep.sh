#!/bin/bash
# E30: what does a 10GB budget buy on gpt-oss-120b, and where does the warm
# asymptote land? slots {12,16} x margin {0, 0.25}, GEN=512 (long runs so the
# LRU reaches steady state - short benches under-measure the warm hit rate;
# Umar's question). slots8 anchors are banked (rep_*, GEN=64).
# PREDICTION (pre-registered): s16 m0 ~2.5-3.5 tok/s (hit ~.70-.75 warm),
# s16 m0.25 ~3-4.5; warm hit visibly above the 64-token runs' hit; phys
# footprint <= ~10GB with the guard idle.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
for cfg in "s12_m0 12 0" "s12_m025 12 0.25" "s16_m0 16 0" "s16_m025 16 0.25"; do
  set -- $cfg
  pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
  [ "${pct:-0}" -lt 35 ] && { echo "ABORT $1: memory pressure"; exit 1; }
  pgrep -f "stream_run.*gguf" >/dev/null && { echo "ABORT: engine busy"; exit 1; }
  echo "== $1 =="
  env LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFILL_SLOTS=1 \
      ./csrc/stream_run "$MODEL" 512 "$PROMPT" 128 > "results/e30_${1}_g512.txt" 2>&1
  grep -E "decode:|prefill:|hit|mem:" "results/e30_${1}_g512.txt" | head -4
done
echo "E30 SWEEP DONE"
