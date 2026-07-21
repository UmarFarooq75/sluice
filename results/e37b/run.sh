#!/bin/bash
# E37b — G1 close-out on a quiet box. Detached, self-guarding.
# Waits up to 2 min for avail >=12 GB (protocol #2), else aborts + logs.
# Then: resident leg (compute ceiling) -> streamed leg (guard-managed, SLOTS=16).
# Survives VS Code / terminal close (run under nohup + caffeinate).
set -u
REPO=/Users/umarfarooq/Desktop/research
BIN=$REPO/csrc/stream_run
MODEL=$REPO/models/gpt-oss-20b-MXFP4.gguf
OUT=$REPO/results/e37b
mkdir -p "$OUT"
PROMPT="Write a Python function that merges two sorted lists into one sorted list without using sort()."
NGEN=20

rm -f "$OUT/DONE" "$OUT/ABORTED"

avail_gb() {
  vm_stat | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
    /Pages free/{f=$3} /Pages inactive/{iv=$3} /Pages speculative/{sp=$3} /Pages purgeable/{pu=$3}
    END { gsub(/\./,"",f); gsub(/\./,"",iv); gsub(/\./,"",sp); gsub(/\./,"",pu);
          printf "%.2f", (f+iv+sp+pu)*ps/1e9 }'
}

# --- protocol #2 gate: wait up to 120s for avail >=12 GB ---
GATE=12.0
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

run_leg() {  # $1=label  $2=extra-env(space-sep KEY=VAL, may be empty)
  local label="$1"; shift
  /usr/bin/time -l env LLMSTREAM_CHAT=1 "$@" \
    "$BIN" "$MODEL" "$NGEN" "$PROMPT" 128 \
    > "$OUT/$label.out" 2> "$OUT/$label.err"
}

# --- resident leg: full model in RAM = compute ceiling (unset SLOTS = resident) ---
echo "[$(date +%H:%M:%S)] resident leg start (avail=$(avail_gb) GB)" >> "$OUT/gate.log"
run_leg resident

# --- streamed leg: guard-managed, SLOTS=16 (registry balanced), guard ON ---
echo "[$(date +%H:%M:%S)] streamed leg start (avail=$(avail_gb) GB)" >> "$OUT/gate.log"
run_leg streamed LLMSTREAM_SLOTS=16 LLMSTREAM_PREFILL_SLOTS=64

# --- summarize from on-disk artifacts ---
summ() {  # $1=label
  local L="$1"
  local hit tps hash rss stall dec bw avgbw cap
  hit=$(grep -oE '\(hit [0-9.]+\)' "$OUT/$L.out" | head -1 | grep -oE '[0-9.]+')
  tps=$(grep -E '^decode:' "$OUT/$L.out" | grep -oE '\([0-9.]+ tok' | grep -oE '[0-9.]+')
  dec=$(grep -E '^decode:' "$OUT/$L.out" | grep -oE '[0-9.]+ s' | head -1 | grep -oE '[0-9.]+')
  hash=$(grep -oE 'logits_hash=[0-9a-f]+' "$OUT/$L.out")
  stall=$(grep -oE 'stall=[0-9.]+ s' "$OUT/$L.out" | head -1 | grep -oE '[0-9.]+')
  bw=$(grep -oE 'per_stream_bw=[0-9]+ MB/s' "$OUT/$L.out")
  avgbw=$(grep -oE 'avg_bw=[0-9]+ MB/s' "$OUT/$L.out")
  cap=$(grep -oE 'final_cap=[0-9]+' "$OUT/$L.out" || echo 'cap=full')
  rss=$(grep -i 'maximum resident set size' "$OUT/$L.err" | grep -oE '[0-9]+' | head -1)
  local rssgb=$(awk "BEGIN{printf \"%.2f\", ${rss:-0}/1073741824}")
  printf "%-9s hit=%-6s tok/s=%-6s decode=%-6ss stall=%-6ss RSS=%-6sGB %-12s %-16s %-18s %s\n" \
    "$L" "${hit:-NA}" "${tps:-NA}" "${dec:-NA}" "${stall:-NA}" "$rssgb" "$cap" "${avgbw:-}" "${bw:-}" "${hash:-}"
}
{
  echo "E37b results ($(date))  — CLEAN (quiet box, avail>=12GB verified)"
  summ resident
  summ streamed
} > "$OUT/summary.txt"

echo "done" > "$OUT/DONE"
