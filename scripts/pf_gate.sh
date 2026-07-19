#!/bin/bash
# E27 prefill-pool gate. Three legs at identical n_ubatch=32 (hashes are only
# comparable at the same ubatch: float accumulation order differs across
# ubatch sizes):
#   resident@32 vs streamed+pf@32  -> must hash-match (prediction 1)
#   streamed WITHOUT pf @32        -> must D12-exit (negative control: the
#                                     pool is what makes ubatch>1 possible)
set -e
cd "$(dirname "$0")/.."

GGUF=$(find hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots -name "*.gguf" | head -1)
[ -z "$GGUF" ] && { echo "GGUF not found"; exit 1; }
N_GEN=32
UB=32
PROMPT="Write a Python function that checks whether a number is prime, then explain its complexity."
export LLMSTREAM_NO_REPACK=1

echo "== E27 pf gate (n_ubatch=$UB, n_gen=$N_GEN) =="
./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" "$UB" > results/pf_gate_resident.txt 2>/dev/null
grep -E "logits_hash|prefill:|decode:" results/pf_gate_resident.txt | sed 's/^/  [resident] /'

LLMSTREAM_SLOTS=12 LLMSTREAM_PREFILL_SLOTS=1 \
    ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" "$UB" > results/pf_gate_pf.txt 2>/dev/null
grep -E "logits_hash|prefill:|decode:|pf_calls|prefill pool" results/pf_gate_pf.txt | sed 's/^/  [pf] /'

echo "  [no-pf negative control - expecting union>slots exit]"
if LLMSTREAM_SLOTS=12 ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" "$UB" > results/pf_gate_nopf.txt 2>&1; then
  echo "GATE FAIL: no-pf ubatch=$UB run should have exited on union overflow"
  exit 1
else
  grep -m1 "union" results/pf_gate_nopf.txt | sed 's/^/  [no-pf] /'
fi

h0=$(grep logits_hash results/pf_gate_resident.txt | cut -d= -f2)
h1=$(grep logits_hash results/pf_gate_pf.txt       | cut -d= -f2)
t0=$(grep "^text:" results/pf_gate_resident.txt)
t1=$(grep "^text:" results/pf_gate_pf.txt)

if [ -n "$h0" ] && [ "$h0" = "$h1" ] && [ "$t0" = "$t1" ]; then
  echo "GATE PASS: resident@$UB == streamed+pf@$UB ($h0), no-pf control exited as designed"
else
  echo "GATE FAIL: resident=$h0 pf=$h1"
  exit 1
fi
