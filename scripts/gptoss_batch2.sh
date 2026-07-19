#!/bin/bash
# batch 2: low-margin NLL points (find the at-anchor operating point) with
# warm-split; prefetch-forced runs; clean re-run of a contaminated rung;
# prompt battery for hit-rate robustness. Memory-guarded per rung.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
NLLTEXT="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
guard() {
    pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
    echo "mem_free=${pct}%"
    [ "${pct:-0}" -lt 40 ] && sleep 60 || true
}
nll() { guard; echo "== nll_$1 (SLOTS=$2 MARGIN=$3) =="
    LLMSTREAM_NLL=1 LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=0 \
        ./csrc/stream_run "$MODEL" 0 "$NLLTEXT" 1 > "results/gptoss_nll_$1.txt" 2>&1
    echo "exit $?"; grep -E 'avg_nll|router_agreement' "results/gptoss_nll_$1.txt"; echo; }
gen() { guard; echo "== gen_$1 (SLOTS=$2 MARGIN=$3 PF=$4) =="
    LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=$4 ${5:+LLMSTREAM_PREFETCH_FORCE=1} \
        ./csrc/stream_run "$MODEL" 64 "Explain why the sky is blue in two sentences." 1 > "results/gptoss_gen_$1.txt" 2>&1
    echo "exit $?"; grep -E 'decode:|hit |prefetch_issued|router_agreement' "results/gptoss_gen_$1.txt"; echo; }

# 1) low-margin NLL: where does quality return to the anchor?
nll m025_slots8  8 0.25
nll m05_slots8   8 0.5
# 2) re-run the m0 anchor + m1/m125 with warm-split (same binary for comparability)
nll m0_slots8b   8 0
nll m1_slots8b   8 1.0
nll m125_slots8b 8 1.25
# 3) prefetch experiments at slots8
gen pf_m05    8 0.5  1 force
gen pf_m125   8 1.25 1 force
# 4) clean re-run of a swap-contaminated rung
gen m1_slots12_clean 12 1.0 0
echo "batch2 complete"
