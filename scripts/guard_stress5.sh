#!/bin/bash
# stress test 5: event-triggered attack. Hog fires only when the artifact
# proves decode is producing tokens, guaranteeing overlap.
cd "$(dirname "$0")/.."
ART=results/gptoss_guard_stress5.txt
rm -f "$ART"
LLMSTREAM_SLOTS=auto LLMSTREAM_MARGIN=1.25 LLMSTREAM_PREFETCH=0 \
    ./csrc/stream_run models/gpt-oss-120b-MXFP4.gguf 512 "Explain why the sky is blue in two sentences." 1 > "$ART" 2>&1 &
RUN=$!
# wait for evidence of token flow (dots), then let prefill clear
while [ "$(stat -f%z "$ART" 2>/dev/null || echo 0)" -lt 40 ]; do sleep 2; done
sleep 8
echo "hog: firing (artifact $(stat -f%z "$ART") bytes)"
.venv/bin/python3 - << 'PY'
import os, time
chunks = []
for i in range(32):
    chunks.append(bytearray(os.urandom(256 * 1024 * 1024)))
    time.sleep(0.1)
time.sleep(30)
PY
echo "hog: released"
wait $RUN
echo "run exit: $?"
grep -E "auto slots|pressure|guard|decode:|hit " "$ART"
sysctl vm.swapusage
