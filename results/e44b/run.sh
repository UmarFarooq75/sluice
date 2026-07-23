#!/bin/bash
# E44b — auto-pin retest at SLOTS=24. Detached, self-gating (s1v3 pattern).
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/e44b
mkdir -p "$OUT"
GATE=7.0; POLL=30; MAX_S=43200; HOLD=3   # lowered 8.0->7.0 for this run only (owner-directed, lablog E44b amendment 2)
rm -f "$OUT/DONE" "$OUT/ABORTED"
. "$REPO/scripts/lib/preflight.sh"

if sl_engine_running; then
  echo "ABORTED: a stream_run is already running (protocol #1)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi

ok=0; streak=0; start=$(date +%s)
echo "[$(date '+%m-%d %H:%M:%S')] E44b long-poll armed: need ${HOLD} consecutive >=${GATE} GB" >> "$OUT/gate.log"
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
/usr/bin/python3 "$OUT/gate.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$rc (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
sl_finish "$rc" "$OUT"
