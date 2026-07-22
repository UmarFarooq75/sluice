#!/bin/bash
# R0 — batch-shape invariance probe. Detached, self-gating.
#   waits for results/e41b/DONE first: E41b owns the quiet window, and two model
#   processes at once is protocol #1's exact prohibition.
#   then the standard long-poll avail gate (>=8 GB, 3 consecutive readings).
#   9 legs, sequential, harness only — the engine binary is unchanged.
# Survives VS Code / terminal close (run under nohup + caffeinate).
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/r0
E41B=$REPO/results/e41b
mkdir -p "$OUT"
. "$REPO/scripts/lib/preflight.sh"
GATE=8.0
POLL=60
MAX_S=43200      # 12 h for the RAM gate, measured from when E41b releases
HOLD=3

rm -f "$OUT/DONE" "$OUT/ABORTED"

avail_gb() {
  vm_stat | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
    /Pages free/{f=$3} /Pages inactive/{iv=$3} /Pages speculative/{sp=$3} /Pages purgeable/{pu=$3}
    END { gsub(/\./,"",f); gsub(/\./,"",iv); gsub(/\./,"",sp); gsub(/\./,"",pu);
          printf "%.2f", (f+iv+sp+pu)*ps/1e9 }'
}

# ---- phase 1: wait for E41b to be finished, whatever its outcome -------------
# It may still be waiting on its own gate for up to 12 h, so this wait is open
# ended by design; the RAM budget below only starts once E41b is out of the way.
echo "[$(date '+%m-%d %H:%M:%S')] waiting for E41b to finish (results/e41b/DONE)" >> "$OUT/gate.log"
waited=0
while [ ! -f "$E41B/DONE" ]; do
  if ! pgrep -f "e41b/run.sh" > /dev/null 2>&1; then
    echo "ABORTED: E41b's launcher is gone and it never wrote DONE. Not starting R0 on an unknown state." > "$OUT/ABORTED"
    echo "aborted" > "$OUT/DONE"
    exit 0
  fi
  sleep 60
  waited=$((waited+60))
  [ $((waited % 1800)) -eq 0 ] && \
    echo "[$(date '+%m-%d %H:%M:%S')] still waiting on E41b (${waited}s)" >> "$OUT/gate.log"
done
echo "[$(date '+%m-%d %H:%M:%S')] E41b finished: $(cat "$E41B/DONE") — R0 may proceed" >> "$OUT/gate.log"

# never overlap with a still-live engine from E41b's own teardown
for i in $(seq 1 30); do
  pgrep -f "csrc/stream_run" > /dev/null 2>&1 || break
  sleep 10
done
if pgrep -f "csrc/stream_run" > /dev/null 2>&1; then
  echo "ABORTED: a stream_run is still running 5 min after E41b's DONE (protocol #1)." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

# ---- phase 2: RAM gate ------------------------------------------------------
ok=0; streak=0; start=$(date +%s)
while :; do
  a=$(avail_gb)
  if awk "BEGIN{exit !($a >= $GATE)}"; then streak=$((streak+1)); else streak=0; fi
  echo "[$(date '+%m-%d %H:%M:%S')] avail=${a} GB streak=${streak}/${HOLD} (need >=${GATE})" >> "$OUT/gate.log"
  if [ "$streak" -ge "$HOLD" ]; then ok=1; break; fi
  now=$(date +%s); [ $((now - start)) -ge "$MAX_S" ] && break
  sleep "$POLL"
done
if [ "$ok" -ne 1 ]; then
  echo "ABORTED: avail never held >=${GATE} GB for ${HOLD} consecutive readings within 12h (last=$(avail_gb) GB, protocol #2)." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

# ---- phase 3: the probe -----------------------------------------------------
echo "[$(date '+%m-%d %H:%M:%S')] legs start (avail=$(avail_gb) GB)" >> "$OUT/gate.log"
/usr/bin/python3 "$OUT/probe.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$rc (avail=$(avail_gb) GB)" >> "$OUT/gate.log"

sl_finish "$rc" "$OUT"
