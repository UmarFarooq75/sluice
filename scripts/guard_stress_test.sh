#!/bin/bash
# E10 live test: run gpt-oss streamed with SLOTS=auto, then mid-generation
# spawn a memory hog that pushes the system toward pressure. PASS criteria:
# (1) the run finishes with coherent output, (2) guard log lines show cap
# drops + evictions, (3) the machine never hangs (swap stays bounded).
set -e
cd "$(dirname "$0")/.."
MODEL=models/gpt-oss-120b-MXFP4.gguf

echo "== baseline: SLOTS=auto, no hog =="
LLMSTREAM_SLOTS=auto LLMSTREAM_MARGIN=1.25 LLMSTREAM_PREFETCH=0 \
    ./csrc/stream_run "$MODEL" 48 "Explain why the sky is blue in two sentences." 1 \
    > results/gptoss_guard_baseline.txt 2>&1
echo "exit: $?"
grep -E 'auto slots|decode:|guard|hit ' results/gptoss_guard_baseline.txt || true
sysctl vm.swapusage

echo "== stress: hog grabs ~5GB after 20s =="
( sleep 20
  python3 - << 'PY'
import time
# grab ~5GB in 256MB steps, hold 45s, release
chunks = []
for i in range(20):
    chunks.append(bytearray(256 * 1024 * 1024))
    for j in range(0, len(chunks[-1]), 16384):
        chunks[-1][j] = 1
    time.sleep(0.5)
time.sleep(45)
PY
) &
HOG=$!
LLMSTREAM_SLOTS=auto LLMSTREAM_MARGIN=1.25 LLMSTREAM_PREFETCH=0 \
    ./csrc/stream_run "$MODEL" 96 "Explain why the sky is blue in two sentences." 1 \
    > results/gptoss_guard_stress.txt 2>&1
echo "exit: $?"
wait $HOG 2>/dev/null || true
grep -E 'auto slots|pressure|decode:|guard|hit ' results/gptoss_guard_stress.txt || true
sysctl vm.swapusage
echo "guard stress test complete"
