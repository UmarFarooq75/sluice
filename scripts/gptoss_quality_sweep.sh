#!/bin/bash
# gpt-oss quality sweep: router-agreement (generation) + teacher-forced NLL
# per margin, under a clean-swap protocol (refuse to start a rung while the
# system is still under memory pressure - the slots12/16 contamination lesson).
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
# fixed public-domain passage, ~130 tokens; NLL over it is deterministic
NLLTEXT="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."

guard() {
    for i in 1 2 3 4 5; do
        pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
        echo "mem_free=${pct}%"
        [ "${pct:-0}" -ge 40 ] && return 0
        echo "pressure high, cooling down 60s"; sleep 60
    done
    echo "WARN: starting rung under pressure (free=${pct}%)"
}

agree() { # label slots margin
    guard
    echo "== agree_$1 (SLOTS=$2 MARGIN=$3) =="
    LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > "results/gptoss_agree_$1.txt" 2>&1
    echo "rung exit: $?"
    grep -E 'decode:|hit|router_agreement|read_work' "results/gptoss_agree_$1.txt"
    echo
}

nll() { # label slots margin
    guard
    echo "== nll_$1 (SLOTS=$2 MARGIN=$3) =="
    LLMSTREAM_NLL=1 LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 0 "$NLLTEXT" 1 > "results/gptoss_nll_$1.txt" 2>&1
    echo "rung exit: $?"
    grep -E 'avg_nll|hit|router_agreement' "results/gptoss_nll_$1.txt"
    echo
}

# generation runs: fidelity + speed with the new counters
agree m1_slots8     8 1.0
agree m125_slots8   8 1.25
agree m15_slots8    8 1.5
agree m125_slots10 10 1.25
# NLL runs: m=0 anchor then the margin curve, all at slots8
nll m0_slots8       8 0
nll m1_slots8       8 1.0
nll m125_slots8     8 1.25
nll m15_slots8      8 1.5
nll m2_slots8       8 2.0
echo "sweep complete"
