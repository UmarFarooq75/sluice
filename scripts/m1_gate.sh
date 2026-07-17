#!/bin/bash
# M1-C logit-equivalence gate.
# Compares FNV-1a hashes over every generated position's full logit vector
# across resident and streamed runs at identical n_ubatch (float accumulation
# order depends on ubatch size, so only same-ubatch runs are comparable).
# PASS iff all three hashes match and the generated text is identical.
set -e
cd "$(dirname "$0")/.."

GGUF=$(find hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots -name "*.gguf" | head -1)
[ -z "$GGUF" ] && { echo "GGUF not found"; exit 1; }
N_GEN=${N_GEN:-32}
PROMPT=${PROMPT:-"Write a Python function that checks whether a number is prime, then explain its complexity."}

# all gate runs disable weight repacking: repacked layouts use different gemm
# kernels (different float accumulation order) than the plain path slot tensors
# take, which shifts low-order logit bits with no semantic difference
export LLMSTREAM_NO_REPACK=1

run() { # name slots
  local name=$1 slots=$2
  if [ "$slots" = "0" ]; then
    ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" 1 > "results/m1_gate_$name.txt" 2>/dev/null
  else
    LLMSTREAM_SLOTS=$slots ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" 1 > "results/m1_gate_$name.txt" 2>/dev/null
  fi
  grep -E "logits_hash|decode:|io:" "results/m1_gate_$name.txt" | sed "s/^/  [$name] /"
}

echo "== M1 logit-equivalence gate (n_ubatch=1, n_gen=$N_GEN) =="
run resident 0
run slots32  32
run slots12  12

h0=$(grep logits_hash results/m1_gate_resident.txt | cut -d= -f2)
h1=$(grep logits_hash results/m1_gate_slots32.txt  | cut -d= -f2)
h2=$(grep logits_hash results/m1_gate_slots12.txt  | cut -d= -f2)
t0=$(grep "^text:" results/m1_gate_resident.txt)
t1=$(grep "^text:" results/m1_gate_slots32.txt)
t2=$(grep "^text:" results/m1_gate_slots12.txt)

if [ "$h0" = "$h1" ] && [ "$h0" = "$h2" ] && [ "$t0" = "$t1" ] && [ "$t0" = "$t2" ]; then
  echo "GATE PASS: all logit hashes bit-identical ($h0)"
else
  echo "GATE FAIL: resident=$h0 slots32=$h1 slots12=$h2"
  exit 1
fi
