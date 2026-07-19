#!/bin/bash
# E24: GPU compute ceiling for gpt-oss. CPU ceiling measured 10.47-11.12 tok/s
# at hit .996 (m2 s12). Same config on Metal: does the GPU lift the ceiling,
# and by how much? Hardware bound: ~100GB/s unified / 2.7GB active = ~37 tok/s.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
[ "${pct:-0}" -lt 40 ] && { echo "ABORT: memory pressure (free ${pct}%)"; exit 1; }
pgrep -f stream_run && { echo "ABORT: stray stream_run"; exit 1; }
echo "== gpu ceiling m2 s12 =="
env LLMSTREAM_SLOTS=12 LLMSTREAM_MARGIN=2 LLMSTREAM_PREFETCH=0 \
    LLMSTREAM_SLOT_DEV=gpu LLMSTREAM_NGL=99 \
    ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > results/gpu_ceiling_m2_s12.txt 2>&1
grep -E 'decode:|prefill:|hit |logits_hash|mem:|guard' results/gpu_ceiling_m2_s12.txt
echo "GPU CEILING DONE"
