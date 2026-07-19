#!/bin/bash
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
NLLTEXT="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
nll() {
    pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
    echo "== nll_$1 (SLOTS=$2 MARGIN=$3) mem_free=${pct}% =="
    LLMSTREAM_NLL=1 LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 0 "$NLLTEXT" 1 > "results/gptoss_nll_$1.txt" 2>&1
    echo "rung exit: $?"
    grep -E 'avg_nll|hit|router_agreement' "results/gptoss_nll_$1.txt"
    echo
}
nll m1_slots8    8 1.0
nll m125_slots8  8 1.25
nll m15_slots8   8 1.5
nll m2_slots8    8 2.0
echo "nll sweep complete"
