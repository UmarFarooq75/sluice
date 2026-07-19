#!/bin/bash
# E20 result rungs (adaptive margin) + E22 (gpt-oss true prefetch recall).
# E20 prediction (logged pre-run): target .93 settles margin 0.25-0.6, decode
# between fixed m0.25 and m0.5 rates; target .95 settles <=0.25.
# E22 question: OLMoE lookahead true recall measured 92% (prefetch_used
# counter); E17 called gpt-oss prefetch 60% waste with the OLD counter.
# What is gpt-oss's TRUE used/issued - is lookahead wrong here, or was the
# metric wrong?
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."

pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
[ "${pct:-0}" -lt 40 ] && { echo "ABORT: memory pressure (free ${pct}%)"; exit 1; }
pgrep -f stream_run && { echo "ABORT: stray stream_run"; exit 1; }

r() { # label env...
    local label=$1; shift
    echo "== $label =="
    env LLMSTREAM_SLOTS=8 "$@" \
        ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > "results/$label.txt" 2>&1
    grep -E 'decode:|hit |router_agreement|adaptive_margin|prefetch_issued|mem:' "results/$label.txt"
    echo
}

r adpt_t93  LLMSTREAM_AGREE_TARGET=0.93 LLMSTREAM_PREFETCH=0
r adpt_t95  LLMSTREAM_AGREE_TARGET=0.95 LLMSTREAM_PREFETCH=0
r pf_m125   LLMSTREAM_MARGIN=1.25 LLMSTREAM_PREFETCH=1 LLMSTREAM_PREFETCH_FORCE=1
r pf_m025   LLMSTREAM_MARGIN=0.25 LLMSTREAM_PREFETCH=1 LLMSTREAM_PREFETCH_FORCE=1
echo "ADPT+PF RUNS DONE"
