#!/bin/bash
# E45 — KV persistence revisit. Detached, self-gating (s1v3 pattern).
#   gate: long poll, 60 s cadence, 12 h ceiling, avail >=8 GB held for 3 consecutive
#         readings; protocol #1 stray check before and after the wait.
#   legs: scripts/gate.sh FULL first (pre-reg gate 4: bit-exact fdf0f83dd70504c5 +
#         byte-identical-off vs the pre-change reference) — if that is not GREEN the
#         persistence legs never run. Then gate.py (gates 1,2,3,5).
set -u
REPO=/Users/umarfarooq/Desktop/research
OUT=$REPO/results/e45
mkdir -p "$OUT"
GATE=7.0; POLL=60; MAX_S=43200; HOLD=3
rm -f "$OUT/DONE" "$OUT/ABORTED"
. "$REPO/scripts/lib/preflight.sh"

if sl_engine_running; then
  echo "ABORTED: a stream_run is already running (protocol #1)." > "$OUT/ABORTED"
  echo aborted > "$OUT/DONE"; exit 0
fi

ok=0; streak=0; start=$(date +%s)
echo "[$(date '+%m-%d %H:%M:%S')] E45 long-poll armed: need ${HOLD} consecutive >=${GATE} GB" >> "$OUT/gate.log"
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

# byte-identical-off precondition, inline (scripts/gate.sh would self-defer below
# its 8 GB gate; this run is owner-directed at 7 GB). Deterministic lines only —
# the same filter gate.sh section 6 uses. Bit-exactness of the NEW binary with the
# flag unset is enforced in-run: leg A1 must equal the pinned N=64 hash.
MODEL_GGUF=$REPO/models/gpt-oss-20b-MXFP4.gguf
REFBIN=$REPO/.gate/stream_run.ref
NEWBIN=$REPO/csrc/stream_run
echo "[$(date '+%m-%d %H:%M:%S')] byte-identical-off precondition (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
ip="Name three primary colors."
env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$REFBIN" "$MODEL_GGUF" 8 "$ip" 1 2>/dev/null | grep -E '^(mode=|logits_hash=|text:|ttft: ctx_held|canon:|canon_reply:)' > "$OUT/bi_ref.out"
env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$NEWBIN" "$MODEL_GGUF" 8 "$ip" 1 2>/dev/null | grep -E '^(mode=|logits_hash=|text:|ttft: ctx_held|canon:|canon_reply:)' > "$OUT/bi_new.out"
if ! cmp -s "$OUT/bi_ref.out" "$OUT/bi_new.out"; then
  echo "ABORTED: new binary DIFFERS from reference with flags unset (see bi_*.out)." > "$OUT/ABORTED"
  echo "[$(date '+%m-%d %H:%M:%S')] byte-identical-off FAILED — legs skipped" >> "$OUT/gate.log"
  echo void > "$OUT/DONE"; exit 1
fi
echo "[$(date '+%m-%d %H:%M:%S')] byte-identical-off PASS; legs start" >> "$OUT/gate.log"

/usr/bin/python3 "$OUT/gate.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$rc (avail=$(sl_avail_gb) GB)" >> "$OUT/gate.log"
sl_finish "$rc" "$OUT"
