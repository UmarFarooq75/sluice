#!/bin/bash
# E23 LFU-protected eviction measurement. Sim predicted (from real trace):
# s8 m0 hit .411(lru) -> static-oracle .489; online LFU-protected should land
# between: hit .43-.47, decode +5-15%. Quality bar: m0 hashes must be
# BIT-IDENTICAL to LRU runs (eviction changes retention, never logits).
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
pct=$(memory_pressure -Q 2>/dev/null | awk '/percentage/ {gsub("%","",$NF); print $NF}')
[ "${pct:-0}" -lt 40 ] && { echo "ABORT: memory pressure (free ${pct}%)"; exit 1; }
pgrep -f stream_run && { echo "ABORT: stray stream_run"; exit 1; }

r() { # label extra-env...
    local label=$1; shift
    echo "== $label =="
    env "$@" ./csrc/stream_run "$MODEL" 64 "$PROMPT" 1 > "results/$label.txt" 2>&1
    grep -E 'decode:|hit |logits_hash|prefetch_issued' "results/$label.txt"
    echo
}

# back-to-back LRU/LFU pairs so thermal drift cannot masquerade as policy
r m0_s8_lru   LLMSTREAM_SLOTS=8  LLMSTREAM_MARGIN=0 LLMSTREAM_PREFETCH=0
r m0_s8_lfu   LLMSTREAM_SLOTS=8  LLMSTREAM_MARGIN=0 LLMSTREAM_PREFETCH=0 LLMSTREAM_EVICT=lfu
r m0_s12_lfu  LLMSTREAM_SLOTS=12 LLMSTREAM_MARGIN=0 LLMSTREAM_PREFETCH=0 LLMSTREAM_EVICT=lfu
r m025_s8_lfu_pf LLMSTREAM_SLOTS=8 LLMSTREAM_MARGIN=0.25 LLMSTREAM_EVICT=lfu

h_lru=$(grep -o 'logits_hash=.*' results/m0_s8_lru.txt)
h_lfu=$(grep -o 'logits_hash=.*' results/m0_s8_lfu.txt)
[ "$h_lru" = "$h_lfu" ] && echo "EVICT QUALITY GATE PASS: m0 s8 lru==lfu ($h_lru)" \
    || { echo "EVICT QUALITY GATE FAIL: lru=$h_lru lfu=$h_lfu"; exit 1; }
echo "BATCH4 DONE"
