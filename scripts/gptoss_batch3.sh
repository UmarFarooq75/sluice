#!/bin/bash
# E23 trace + E20 slow-start rerun + combined product-default candidate.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
[ "${pct:-0}" -lt 40 ] && { echo "ABORT: memory pressure (free ${pct}%)"; exit 1; }
pgrep -f stream_run && { echo "ABORT: stray stream_run"; exit 1; }

echo "== trace m0 s8 (routing ground truth for belady sim) =="
env LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=0 LLMSTREAM_PREFETCH=0 LLMSTREAM_PRINT_IDS=999999 \
    ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > results/trace_m0_s8.txt 2>&1
grep -cE '^ids ' results/trace_m0_s8.txt | sed 's/^/id lines: /'

echo "== adpt_t93 slow-start =="
env LLMSTREAM_SLOTS=8 LLMSTREAM_AGREE_TARGET=0.93 LLMSTREAM_PREFETCH=0 \
    ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > results/adpt_t93_ss.txt 2>&1
grep -E 'decode:|hit |router_agreement|adaptive_margin' results/adpt_t93_ss.txt

echo "== product-default candidate: adaptive t93 + auto prefetch =="
env LLMSTREAM_SLOTS=8 LLMSTREAM_AGREE_TARGET=0.93 \
    ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > results/adpt_t93_pf.txt 2>&1
grep -E 'decode:|hit |router_agreement|adaptive_margin|prefetch_issued' results/adpt_t93_pf.txt
echo "BATCH3 DONE"
