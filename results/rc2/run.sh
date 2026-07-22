#!/bin/bash
# RC2 — bisect to the minimal failing reproducer. Detached, self-gating. LOCALIZE ONLY.
#   gate: long poll, 60 s cadence, 12 h ceiling, avail >=8 GB held for 3 consecutive
#         readings (a single passing reading catches a transient that collapses again).
#   legs: control (is DEBUG_HASH inert?) -> fresh -> reused-prefix -> E39 persist path.
#   Small by design: short prompts, n_gen=1, so the final forward pass is a 1-token
#   decode in every leg and its dbg lines align one-to-one across legs.
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/rc2
mkdir -p "$OUT"
GATE=8.0; POLL=60; MAX_S=43200; HOLD=3
rm -f "$OUT/DONE" "$OUT/ABORTED"
. "$REPO/scripts/lib/preflight.sh"

if sl_engine_running; then
  echo "ABORTED: a stream_run is already running (protocol #1)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi

ok=0; streak=0; start=$(date +%s)
echo "[$(date '+%m-%d %H:%M:%S')] RC2 long-poll armed: need ${HOLD} consecutive >=${GATE} GB" >> "$OUT/gate.log"
while :; do
  a=$(sl_avail_gb)
  if sl_ge "$a" "$GATE"; then streak=$((streak+1)); else streak=0; fi
  echo "[$(date '+%m-%d %H:%M:%S')] avail=${a} GB streak=${streak}/${HOLD}" >> "$OUT/gate.log"
  [ "$streak" -ge "$HOLD" ] && { ok=1; break; }
  [ $(( $(date +%s) - start )) -ge "$MAX_S" ] && break
  sleep "$POLL"
done
if [ "$ok" -ne 1 ]; then
  echo "ABORTED: avail never held >=${GATE} GB for ${HOLD} readings within 12h (last=$(sl_avail_gb) GB)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi
if sl_engine_running; then
  echo "ABORTED: a stream_run appeared while waiting (protocol #1)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi

echo "[$(date '+%m-%d %H:%M:%S')] legs start (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
/usr/bin/python3 "$OUT/bisect.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$rc (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
sl_finish "$rc" "$OUT"
