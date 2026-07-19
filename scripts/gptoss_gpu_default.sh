#!/bin/bash
# E24b: GPU at the product default (m0.25, auto-prefetch, s8). Prediction:
# miss-bound regime -> GPU ~= CPU (~1.5-1.6); GPU pays only at high hit.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
[ "${pct:-0}" -lt 40 ] && { echo "ABORT: memory pressure"; exit 1; }
pgrep -f stream_run && { echo "ABORT: stray stream_run"; exit 1; }
echo "== gpu default m025 s8 +pf =="
env LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=0.25 \
    LLMSTREAM_SLOT_DEV=gpu LLMSTREAM_NGL=99 \
    ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > results/gpu_default_m025_s8.txt 2>&1
grep -E 'decode:|prefill:|hit |router_agreement|prefetch_issued|mem:' results/gpu_default_m025_s8.txt
echo "GPU DEFAULT DONE"
