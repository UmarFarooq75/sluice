#!/bin/bash
# Portability-change gate runner: waits for the memory window, runs the FULL
# gate (bit-exact + byte-identical-off), leaves the receipt in gate_full.log.
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/portgate
. "$REPO/scripts/lib/preflight.sh"
rm -f "$OUT/DONE"
streak=0; start=$(date +%s)
while :; do
  a=$(sl_avail_gb)
  if sl_ge "$a" "8.0"; then streak=$((streak+1)); else streak=0; fi
  echo "[$(date '+%m-%d %H:%M:%S')] avail=$a streak=$streak/3" >> "$OUT/gate.log"
  [ "$streak" -ge 3 ] && break
  [ $(( $(date +%s) - start )) -ge 43200 ] && { echo timeout > "$OUT/DONE"; exit 0; }
  sleep 60
done
bash "$REPO/scripts/gate.sh" > "$OUT/gate_full.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "GATE GREEN" "$OUT/gate_full.log"; then echo green > "$OUT/DONE"; else echo "rc=$rc not-green" > "$OUT/DONE"; fi
