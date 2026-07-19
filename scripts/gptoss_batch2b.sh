#!/bin/bash
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
guard() { pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}'); echo "mem_free=${pct}%"; [ "${pct:-0}" -lt 40 ] && sleep 60 || true; }
gen() { # label slots margin pf force
    guard; echo "== gen_$1 (SLOTS=$2 MARGIN=$3 PF=$4 FORCE=${5:-no}) =="
    env LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=$4 ${5:+LLMSTREAM_PREFETCH_FORCE=1} \
        ./csrc/stream_run "$MODEL" 64 "Explain why the sky is blue in two sentences." 1 > "results/gptoss_gen_$1.txt" 2>&1
    echo "exit $?"; grep -E 'decode:|hit |prefetch_issued|router_agreement' "results/gptoss_gen_$1.txt"; echo; }
gen pf_m05    8 0.5  1 force
gen pf_m125   8 1.25 1 force
gen pf_m025   8 0.25 1 force
gen m1_slots12_clean 12 1.0 0
echo "batch2b complete"
