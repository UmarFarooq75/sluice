#!/bin/bash
# E27 TTFT bench, gpt-oss-120b, slots8 m0 (exact), prefetch default.
# Rungs: baseline ubatch=1 @128-tok prompt (the D2 status quo), pf @128, pf @512.
# PREDICTION (pre-registered, docs/prefill-design.md): pf prefill >= 8 tok/s
# (baseline 0.7-2.9), 512-tok TTFT < 75 s vs ~5-8 min extrapolated baseline.
# Baseline at 512 is skipped to spare ~10 min of thrash; linear physics and
# the banked chat logs are its measurement.
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf

# ~128-token and ~512-token prompts: paragraph repeated (fixed literal, so the
# artifact reproduces; routing sees varied positions via context, not text)
P1="The history of computing spans mechanical calculators, vacuum tubes, transistors, and integrated circuits. Each generation multiplied speed while shrinking cost and size, enabling applications the prior generation could not imagine. Programming languages evolved alongside, from raw machine code through assembly, Fortran and Lisp, to structured, object-oriented, and functional styles. Networks then connected these machines, and the resulting internet reshaped commerce, science, and daily life in ways still unfolding today."
P128="$P1 $P1"
P512="$P1 $P1 $P1 $P1 $P1 $P1 $P1 $P1"

run() { # tag ubatch prompt
    pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
    [ "${pct:-0}" -lt 40 ] && { echo "ABORT $1: memory pressure"; exit 1; }
    pgrep -f "stream_run.*gguf" >/dev/null && { echo "ABORT: engine busy"; exit 1; }
    echo "== $1 =="
    env LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=0 $4 \
        ./csrc/stream_run "$MODEL" 8 "$3" "$2" > "results/pf_ttft_$1.txt" 2>&1
    grep -E "prompt_toks|prefill:|decode:|pf_calls|prefill pool|mem:" "results/pf_ttft_$1.txt" | sed 's/^/  /'
}

run base_u1_128 1   "$P128"
run pf_128      128 "$P128" "LLMSTREAM_PREFILL_SLOTS=1"
run pf_512      512 "$P512" "LLMSTREAM_PREFILL_SLOTS=1"
echo "PF TTFT BENCH DONE"
