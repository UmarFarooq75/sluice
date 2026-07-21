#!/bin/bash
# E41b — KV canonicalization at end of turn. Detached, self-guarding.
#   gate: LONG POLL — every 60 s for up to 12 h, fire when avail >=8 GB holds for
#         3 consecutive readings (a single reading can catch a transient dip that
#         would collapse again mid-leg). Every reading logged. Abort+log at 12 h.
#   legs (sequential, one model process at a time — protocol #1):
#     A  canon ON,  2-turn chat  -> turn-2 reused/reprefill/ttft + logits_hash
#     B  canon OFF, 2-turn chat  -> CONTROL: identical rendering, stock reuse path
#     C  fresh single-shot       -> turn-2's rendering with reused=0 (reference hash)
#   then the byte-identical-off executable check and the bit-exact gate.
# Survives VS Code / terminal close (run under nohup + caffeinate from the launch cmd).
set -u
REPO=/Users/umarfarooq/Desktop/research
BIN=$REPO/csrc/stream_run
MODEL=$REPO/models/gpt-oss-20b-MXFP4.gguf
OUT=$REPO/results/e41b
PREV=/private/tmp/claude-501/-Users-umarfarooq-Desktop-research/ba7a8b96-b96b-4bdd-a879-d7a7b2ff0f3f/scratchpad/stream_run.pre_e41b
mkdir -p "$OUT"
GATE=8.0

rm -f "$OUT/DONE" "$OUT/ABORTED"

avail_gb() {
  vm_stat | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
    /Pages free/{f=$3} /Pages inactive/{iv=$3} /Pages speculative/{sp=$3} /Pages purgeable/{pu=$3}
    END { gsub(/\./,"",f); gsub(/\./,"",iv); gsub(/\./,"",sp); gsub(/\./,"",pu);
          printf "%.2f", (f+iv+sp+pu)*ps/1e9 }'
}

# RUN 2. Run 1 was VOID — see results/e41b/run1_void/VOID.md (a missing re.M made
# leg A render a different conversation). R0 now owns the window, so wait for it.
echo "[$(date '+%m-%d %H:%M:%S')] run 2 armed; waiting for R0 (results/r0/DONE)" >> "$OUT/gate.log"
while [ ! -f "$REPO/results/r0/DONE" ]; do
  if ! pgrep -f "r0/run.sh" > /dev/null 2>&1; then
    echo "ABORTED: R0's launcher is gone and it never wrote DONE. Not starting on an unknown state." > "$OUT/ABORTED"
    echo "aborted" > "$OUT/DONE"; exit 0
  fi
  sleep 60
done
echo "[$(date '+%m-%d %H:%M:%S')] R0 finished: $(cat "$REPO/results/r0/DONE")" >> "$OUT/gate.log"
for i in $(seq 1 30); do pgrep -f "csrc/stream_run" > /dev/null 2>&1 || break; sleep 10; done

# protocol #1: never start a second model process
if pgrep -f "csrc/stream_run" > /dev/null 2>&1; then
  echo "ABORTED: a stream_run is already running (protocol #1). No run performed." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

# long poll: 60 s cadence, 12 h ceiling, needs HOLD consecutive passing readings.
# One passing reading is not enough — avail spikes when an app closes a window and
# collapses again seconds later, and a dip mid-leg is worse than never starting.
POLL=60
MAX_S=43200            # 12 h
HOLD=3                 # consecutive readings >= GATE before we fire
ok=0
streak=0
start=$(date +%s)
echo "[$(date '+%m-%d %H:%M:%S')] long-poll armed: every ${POLL}s for up to ${MAX_S}s, need ${HOLD} consecutive >=${GATE} GB" >> "$OUT/gate.log"
while :; do
  a=$(avail_gb)
  if awk "BEGIN{exit !($a >= $GATE)}"; then streak=$((streak+1)); else streak=0; fi
  echo "[$(date '+%m-%d %H:%M:%S')] avail=${a} GB streak=${streak}/${HOLD} (need >=${GATE})" >> "$OUT/gate.log"
  if [ "$streak" -ge "$HOLD" ]; then ok=1; break; fi
  now=$(date +%s)
  if [ $((now - start)) -ge "$MAX_S" ]; then break; fi
  sleep "$POLL"
done
if [ "$ok" -ne 1 ]; then
  a=$(avail_gb)
  echo "ABORTED: avail never held >=${GATE} GB for ${HOLD} consecutive readings within 12h (last=${a} GB, protocol #2). No run performed." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

# 12 h may have passed since the first stray check — re-check before spending RAM
if pgrep -f "csrc/stream_run" > /dev/null 2>&1; then
  echo "ABORTED: a stream_run appeared while waiting (protocol #1). No run performed." > "$OUT/ABORTED"
  echo "aborted" > "$OUT/DONE"
  exit 0
fi

echo "[$(date +%H:%M:%S)] legs A/B/C start (avail=$(avail_gb) GB)" >> "$OUT/gate.log"
/usr/bin/python3 "$OUT/gate.py" > "$OUT/legs.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] legs finished rc=$rc (avail=$(avail_gb) GB)" >> "$OUT/gate.log"

# --- protocol #3: byte-identical with the flag unset (executable check) -------
# same short prompt through the pre-E41b binary and the new one, canon UNSET.
# stdout must match byte for byte, not just the hash.
echo "[$(date +%H:%M:%S)] inert check" >> "$OUT/gate.log"
IP="Name three primary colors."
for pair in "old:$PREV" "new:$BIN"; do
  tag=${pair%%:*}; exe=${pair#*:}
  [ -x "$exe" ] || { echo "inert check SKIPPED: $exe missing" >> "$OUT/gate.log"; continue; }
  env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$exe" "$MODEL" 8 "$IP" 1 \
    > "$OUT/inert_$tag.raw" 2> "$OUT/inert_$tag.err"
  # keep only lines that are DETERMINISTIC for a given binary+input. Timings,
  # prefetch race counters and peak_rss differ between two runs of the SAME
  # binary, so comparing them guarantees a false RED (run 1 hit exactly that).
  grep -E '^(mode=|logits_hash=|text:|ttft: ctx_held|canon:|canon_reply:)' "$OUT/inert_$tag.raw" > "$OUT/inert_$tag.out"
done

# --- protocol #4: bit-exact gate, canonical prompt A / N=8 / SLOTS=5 ---------
echo "[$(date +%H:%M:%S)] bit-exact gate" >> "$OUT/gate.log"
PROMPT_A="Write a Python function that merges two sorted lists into one sorted list without using sort()."
env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$BIN" "$MODEL" 8 "$PROMPT_A" 1 \
  > "$OUT/bitexact.out" 2> "$OUT/bitexact.err"

{
  echo "E41b gate checks ($(date))"
  echo "expected bit-exact hash: fdf0f83dd70504c5"
  grep -E '^logits_hash=' "$OUT/bitexact.out" || echo "logits_hash: MISSING"
  if [ -f "$OUT/inert_old.out" ] && [ -f "$OUT/inert_new.out" ]; then
    if cmp -s "$OUT/inert_old.out" "$OUT/inert_new.out"; then
      echo "inert (canon unset), old vs new stdout: IDENTICAL"
    else
      echo "inert (canon unset), old vs new stdout: DIFFERS"
      diff "$OUT/inert_old.out" "$OUT/inert_new.out" | head -20
    fi
  else
    echo "inert check: not run"
  fi
} > "$OUT/gates.txt" 2>&1

echo "done" > "$OUT/DONE"
