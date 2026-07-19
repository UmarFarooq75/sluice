#!/bin/bash
# gpt-oss-120b streamed benchmark ladder (M2 Air 16GB, CPU backend).
#
# Model: 63.4GB MXFP4, 36 layers x 128 experts, top-4, 5.1B active params.
# Expert set per layer = 6 tensors (gate/up/down weights + biases).
# Per-expert bytes ~= 13.2MB weights + ~35KB biases.
# Expert cache RAM ~= slots x 36 x 13.2MB: 8 slots ~3.8GB, 12 ~5.7GB, 16 ~7.6GB.
#
# Protocol notes:
# - sustained-thermal: n_gen=64 minimum; record decode over the full window,
#   not the first seconds (fanless chassis throttles).
# - no resident reference is possible on this machine (that is the point);
#   m=0 exact-routing mode is the quality anchor.
# - LLMSTREAM_PREFETCH=0 for first runs; big-model regime may re-enable later.

set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf
PROMPT="Explain why the sky is blue in two sentences."
GEN=${GEN:-64}

run() { # label slots margin
    [ -n "$2" ] && [ -n "$3" ] || { echo "ABORT: empty slots/margin for $1"; exit 1; }
    echo "== $1 (SLOTS=$2 MARGIN=$3) =="
    LLMSTREAM_SLOTS=$2 LLMSTREAM_MARGIN=$3 LLMSTREAM_PREFETCH=${PF:-0} \
        ./csrc/stream_run "$MODEL" "$GEN" "$PROMPT" 1 > "results/gptoss_$1.txt" 2>&1
    echo "rung $1 exit: $?"
    grep -E 'prefill:|decode:|hit|logits_hash' "results/gptoss_$1.txt"
    echo
}

# ladder: margin sweep in LOGIT units (gpt-oss routes on raw biased logits,
# so probability-scale margins like 0.02 are near-no-ops), then RAM curve.
# m0_slots12 (exact anchor) already measured: 0.75 tok/s, hit 0.526.
if [ "${LADDER2:-0}" != "1" ]; then
  run m002_slots12   12 0.02
  run m05_slots12    12 0.5
  run m1_slots12     12 1.0
  run m2_slots12     12 2.0
  run m1_slots16    16 1.0
  run m1_slots8      8 1.0
fi
# cliff hunt (added after first ladder): fine margins at the slots8 sweet spot
if [ "${LADDER2:-0}" = "1" ]; then
  run m125_slots8    8 1.25
  run m15_slots8     8 1.5
  run m1_slots10    10 1.0
  run m125_slots10  10 1.25
fi
