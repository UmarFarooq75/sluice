#!/bin/bash
# G5-1 — KV persistence revisit. Detached, self-gating (s1v3 pattern).
#   gate: long poll, 60 s cadence, 12 h ceiling, avail >=8 GB held for 3 consecutive
#         readings; protocol #1 stray check before and after the wait.
#   legs: scripts/gate.sh FULL first (pre-reg gate 4: bit-exact fdf0f83dd70504c5 +
#         byte-identical-off vs the pre-change reference) — if that is not GREEN the
#         persistence legs never run. Then gate.py (gates 1,2,3,5).
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/g5kv
mkdir -p "$OUT"
GATE=8.0; POLL=60; MAX_S=43200; HOLD=3
rm -f "$OUT/DONE" "$OUT/ABORTED"
. "$REPO/scripts/lib/preflight.sh"

if sl_engine_running; then
  echo "ABORTED: a stream_run is already running (protocol #1)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi

ok=0; streak=0; start=$(date +%s)
echo "[$(date '+%m-%d %H:%M:%S')] G5-1 long-poll armed: need ${HOLD} consecutive >=${GATE} GB" >> "$OUT/gate.log"
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

echo "[$(date '+%m-%d %H:%M:%S')] gate 4 first: scripts/gate.sh full (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
bash "$REPO/scripts/gate.sh" > "$OUT/gate_full.log" 2>&1
rc_gate=$?
if [ "$rc_gate" -ne 0 ] || ! grep -q "GATE GREEN" "$OUT/gate_full.log"; then
  # PARTIAL is not green (its own words); a race that re-deferred the model legs
  # must void this run, not silently narrow it to gates 1/2/3/5.
  echo "ABORTED: scripts/gate.sh not GREEN (rc=$rc_gate) — see gate_full.log." > "$OUT/ABORTED"
  echo "[$(date '+%m-%d %H:%M:%S')] gate 4 not GREEN rc=$rc_gate — persistence legs skipped" >> "$OUT/gate.log"
  echo void > "$OUT/DONE"; exit 1
fi
echo "[$(date '+%m-%d %H:%M:%S')] gate 4 GREEN; persistence legs start" >> "$OUT/gate.log"

/usr/bin/python3 "$OUT/gate.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$rc (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
sl_finish "$rc" "$OUT"
