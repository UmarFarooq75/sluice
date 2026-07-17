#!/bin/bash
# The head-to-head: Qwen3.6-35B-A3B Q5_K_M (26.5GB) on a 16GB machine.
#   A) stock llama.cpp mmap (Ollama's engine underneath) - OS-managed paging
#   B) streamed, exact (m=0)      - bit-exact, managed expert cache
#   C) streamed, margin m=0.02    - the near-lossless traffic dial
# Run `sudo purge` before each mode for honest cold numbers (script pauses).
set -e
cd "$(dirname "$0")/.."

GGUF=$(find hf_home/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots -name "*.gguf" 2>/dev/null | head -1)
[ -z "$GGUF" ] && { echo "Qwen3.6 GGUF not found"; exit 1; }
N_GEN=${N_GEN:-64}
SLOTS=${SLOTS:-}   # default: computed for ~8GB expert cache after probe
PROMPT=${PROMPT:-"Write a Python function that merges two sorted lists in O(n), then explain why the naive approach is slower."}

pause_purge() { echo; echo ">>> run 'sudo purge' in another terminal for cold-cache honesty, then press enter"; read -r; }

if [ -z "$SLOTS" ]; then
  # probe layer/expert geometry with a tiny slot count, no decode
  probe=$(LLMSTREAM_SLOTS=16 ./csrc/stream_run "$GGUF" 0 "hi" 1 2>/dev/null | grep "^llmstream:")
  echo "probe: $probe"
  mb_per_exp=$(echo "$probe" | sed -E 's/.* ([0-9.]+) MB\/expert.*/\1/')
  layers=$(echo "$probe" | sed -E 's/llmstream: ([0-9]+) layers.*/\1/')
  SLOTS=$(python3 -c "print(max(16, int(8000 / ($layers * $mb_per_exp))))")
  echo "chosen slots/layer: $SLOTS (~8GB expert cache)"
fi

pause_purge
echo "=== A) stock mmap (Ollama engine) ==="
./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" 512 | tee results/qwen36_stock.txt | grep -E "prefill:|decode:"

pause_purge
echo "=== B) streamed exact m=0, slots=$SLOTS ==="
LLMSTREAM_SLOTS=$SLOTS ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" 1 | tee results/qwen36_stream_m0.txt | grep -E "prefill:|decode:|io:"

pause_purge
echo "=== C) streamed margin m=0.02, slots=$SLOTS ==="
LLMSTREAM_MARGIN=0.02 LLMSTREAM_SLOTS=$SLOTS ./csrc/stream_run "$GGUF" "$N_GEN" "$PROMPT" 1 | tee results/qwen36_stream_m002.txt | grep -E "prefill:|decode:|io:"

echo; echo "full outputs in results/qwen36_*.txt"
