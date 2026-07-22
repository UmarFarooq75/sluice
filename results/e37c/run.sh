#!/bin/bash
# E37c — G1 close-out at matched N. Detached, self-guarding. ONE streamed leg.
#   gate: wait up to 2 min for avail >=8 GB (protocol #2), else abort+log.
#   leg: N=64, SLOTS=16, PREFILL_SLOTS=64, ubatch=128, guard ON, exact greedy, prompt A.
# Survives VS Code / terminal close (run under nohup + caffeinate from the launch cmd).
set -u
REPO=/Users/umarfarooq/Desktop/research
BIN=$REPO/csrc/stream_run
MODEL=$REPO/models/gpt-oss-20b-MXFP4.gguf
OUT=$REPO/results/e37c
mkdir -p "$OUT"
. "$REPO/scripts/lib/preflight.sh"
PROMPT="Write a Python function that merges two sorted lists into one sorted list without using sort()."
NGEN=64
GATE=8.0

rm -f "$OUT/DONE" "$OUT/ABORTED"

avail_gb() {
  vm_stat | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
    /Pages free/{f=$3} /Pages inactive/{iv=$3} /Pages speculative/{sp=$3} /Pages purgeable/{pu=$3}
    END { gsub(/\./,"",f); gsub(/\./,"",iv); gsub(/\./,"",sp); gsub(/\./,"",pu);
          printf "%.2f", (f+iv+sp+pu)*ps/1e9 }'
}

ok=0
for t in $(seq 0 10 120); do
  a=$(avail_gb)
  echo "[$(date +%H:%M:%S)] avail=${a} GB (need >=${GATE})" >> "$OUT/gate.log"
  if awk "BEGIN{exit !($a >= $GATE)}"; then ok=1; break; fi
  sleep 10
done
if [ "$ok" -ne 1 ]; then
  a=$(avail_gb)
  echo "ABORTED: avail=${a} GB < ${GATE} GB after 120s (protocol #2). No run performed." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

echo "[$(date +%H:%M:%S)] streamed leg start (avail=$(avail_gb) GB)" >> "$OUT/gate.log"
/usr/bin/time -l env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=16 LLMSTREAM_PREFILL_SLOTS=64 \
rc=$(( ${rc:-0} + $? ))
  "$BIN" "$MODEL" "$NGEN" "$PROMPT" 128 \
  > "$OUT/streamed.out" 2> "$OUT/streamed.err"

# summarize from on-disk artifacts
L=streamed
hit=$(grep -oE '\(hit [0-9.]+\)' "$OUT/$L.out" | head -1 | grep -oE '[0-9.]+')
tps=$(grep -E '^decode:' "$OUT/$L.out" | grep -oE '\([0-9.]+ tok' | grep -oE '[0-9.]+')
dec=$(grep -E '^decode:' "$OUT/$L.out" | grep -oE '[0-9.]+ s' | head -1 | grep -oE '[0-9.]+')
hash=$(grep -oE 'logits_hash=[0-9a-f]+' "$OUT/$L.out")
stall=$(grep -oE 'stall=[0-9.]+ s' "$OUT/$L.out" | head -1 | grep -oE '[0-9.]+')
bw=$(grep -oE 'per_stream_bw=[0-9]+ MB/s' "$OUT/$L.out")
avgbw=$(grep -oE 'avg_bw=[0-9]+ MB/s' "$OUT/$L.out")
cap=$(grep -oE 'final_cap=[0-9]+' "$OUT/$L.out" || echo 'cap=full')
memline=$(grep -oE 'peak_rss=[0-9.]+ GB phys_footprint=[0-9.]+ GB' "$OUT/$L.out")
rss=$(grep -i 'maximum resident set size' "$OUT/$L.err" | grep -oE '[0-9]+' | head -1)
rssgb=$(awk "BEGIN{printf \"%.2f\", ${rss:-0}/1073741824}")
{
  echo "E37c results ($(date)) — CLEAN quiet box (avail>=${GATE}GB), N=${NGEN} SLOTS=16 prompt A exact"
  printf "streamed  hit=%s  tok/s=%s  decode=%ss  stall=%ss  RSS(time)=%sGB  %s  %s  %s  %s\n" \
    "${hit:-NA}" "${tps:-NA}" "${dec:-NA}" "${stall:-NA}" "$rssgb" "$memline" "$cap" "${avgbw:-}" "$hash"
} > "$OUT/summary.txt"

sl_finish "${rc:-1}" "$OUT"
