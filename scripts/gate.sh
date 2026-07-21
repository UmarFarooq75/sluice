#!/bin/bash
# The standing protocol as ONE command. Every future rung invokes this instead of
# hand-rolling its own gate — that hand-rolling is what let two different avail
# yardsticks into the tree at once (see scripts/lib/preflight.sh).
#
#   bash scripts/gate.sh            full gate; model legs self-defer if unable
#   bash scripts/gate.sh --static   static checks only, never touches the model
#   bash scripts/gate.sh --ref      snapshot the CURRENT binary as the byte-identical
#                                   reference. Run this BEFORE editing the engine.
#
# Exit 0 = everything that COULD run passed. Exit 1 = something that ran FAILED.
# A deferred model leg is NOT a failure — it is "not yet", and it says so loudly
# so it can never be mistaken for a pass.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
. scripts/lib/preflight.sh

MODEL="$ROOT/models/gpt-oss-20b-MXFP4.gguf"
BIN="$ROOT/csrc/stream_run"
REF="$ROOT/.gate/stream_run.ref"
GATE_AVAIL=8.0
# canonical bit-exact anchor: prompt A, N=8, SLOTS=5 (docs/lablog.md)
PROMPT_A="Write a Python function that merges two sorted lists into one sorted list without using sort()."
EXPECT_HASH="fdf0f83dd70504c5"

P="\033[32m✓\033[0m"; F="\033[31m✗\033[0m"; D="\033[33m·\033[0m"; B="\033[1m"; R="\033[0m"
fail=0
deferred=0
say()  { printf "  $1 %s\n" "$2"; }
bad()  { say "$F" "$1"; fail=1; }
defer(){ say "$D" "$1"; deferred=1; }

if [ "${1:-}" = "--ref" ]; then
  [ -x "$BIN" ] || { echo "no engine binary to snapshot — build first"; exit 1; }
  mkdir -p "$ROOT/.gate"; cp "$BIN" "$REF"
  echo "reference snapshot taken: .gate/stream_run.ref"
  echo "now make your engine change, then: bash scripts/gate.sh"
  exit 0
fi

printf "${B}sluice gate${R}\n\n${B}static${R}\n"

# --- 1. syntax of every shipped script --------------------------------------
for f in cli/sluice ui/app.py scripts/check_urls.py; do
  if /usr/bin/python3 -c "import ast,sys;ast.parse(open('$f').read())" 2>/dev/null; then
    say "$P" "syntax $f"
  else bad "syntax $f"; fi
done
for f in scripts/*.sh scripts/lib/*.sh results/*/run.sh; do
  [ -e "$f" ] || continue
  bash -n "$f" 2>/dev/null && say "$P" "syntax $f" || bad "syntax $f"
done

# --- 2. manifest is valid JSON ----------------------------------------------
if /usr/bin/python3 -c "import json;json.load(open('packaging/models.json'))" 2>/dev/null; then
  say "$P" "packaging/models.json parses"
else bad "packaging/models.json is not valid JSON"; fi

# --- 3. no undocumented engine flags ----------------------------------------
# Protocol #7's sibling: a flag nobody documented is a flag nobody can audit.
und=$(/usr/bin/python3 - <<'PY'
import re
from pathlib import Path
src = Path("csrc/stream_run.cpp").read_text(); rm = Path("README.md").read_text()
print(" ".join(v for v in sorted(set(re.findall(r'getenv\("(LLMSTREAM_[A-Z0-9_]+)"\)', src))) if v not in rm))
PY
)
if [ -z "$und" ]; then say "$P" "all engine LLMSTREAM_* flags documented in README"
else bad "undocumented engine flags: $und"; fi

# --- 4. the engine compiles --------------------------------------------------
# compiled to a temp path on purpose: a detached experiment may be armed against
# csrc/stream_run right now, and rewriting that file underneath it could hand the
# launcher a half-written binary.
tmpbin=$(mktemp -t sluice_gate_build)
if OUT="$tmpbin" bash scripts/build_driver.sh >/dev/null 2>&1; then say "$P" "engine builds (to a temp path; csrc/stream_run untouched)"
else bad "engine does NOT build (scripts/build_driver.sh)"; fi
rm -f "$tmpbin"

# --- model-dependent legs ----------------------------------------------------
printf "\n${B}model legs${R}\n"
av=$(sl_avail_gb); sw=$(sl_swap_gb)
can_run=1
if [ "${1:-}" = "--static" ]; then
  defer "SKIPPED by --static (${av} GB avail, ${sw:-0} GB swap)"; can_run=0
fi
if [ "$can_run" -eq 1 ] && [ ! -f "$MODEL" ]; then
  defer "PENDING MODEL WINDOW — $(basename "$MODEL") not on disk (sluice pull gpt-oss-20b)"; can_run=0
fi
if [ "$can_run" -eq 1 ] && sl_engine_running; then
  defer "PENDING MODEL WINDOW — another engine is running (protocol #1); this gate will not add a second"; can_run=0
fi
if [ "$can_run" -eq 1 ] && ! sl_ge "$av" "$GATE_AVAIL"; then
  defer "PENDING MODEL WINDOW — ${av} GB avail < ${GATE_AVAIL} GB (protocol #2); swap in use ${sw:-?} GB"; can_run=0
fi

if [ "$can_run" -eq 1 ]; then
  say "$P" "preflight ok (${av} GB avail, ${sw:-0} GB swap, no strays)"

  # --- 5. bit-exact gate (protocol #4) ---------------------------------------
  out=$(env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$BIN" "$MODEL" 8 "$PROMPT_A" 1 2>/dev/null)
  got=$(printf "%s" "$out" | sed -n 's/^logits_hash=\(.*\)$/\1/p')
  if [ "$got" = "$EXPECT_HASH" ]; then say "$P" "bit-exact gate $got"
  else bad "bit-exact gate: got '${got:-<none>}' expected $EXPECT_HASH"; fi

  # --- 6. byte-identical-off (protocol #3) -----------------------------------
  if [ -x "$REF" ]; then
    ip="Name three primary colors."
    # Compare only DETERMINISTIC lines. Wall-clock timings, prefetch race counters
    # and peak_rss differ between two runs of the SAME binary, so a full-stdout cmp
    # can never pass — E41b run 1 reported "inert: DIFFERS" on nothing but timing
    # noise while mode=, logits_hash= and text: were identical.
    env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$REF" "$MODEL" 8 "$ip" 1 2>/dev/null | grep -E '^(mode=|logits_hash=|text:|ttft: ctx_held|canon:|canon_reply:)' >/tmp/gate_ref.out
    env LLMSTREAM_CHAT=1 LLMSTREAM_SLOTS=5 "$BIN" "$MODEL" 8 "$ip" 1 2>/dev/null | grep -E '^(mode=|logits_hash=|text:|ttft: ctx_held|canon:|canon_reply:)' >/tmp/gate_new.out
    if cmp -s /tmp/gate_ref.out /tmp/gate_new.out; then say "$P" "byte-identical with flags unset (vs .gate reference)"
    else bad "stdout DIFFERS from the reference with all flags unset:
$(diff /tmp/gate_ref.out /tmp/gate_new.out | head -8)"; fi
  else
    defer "no reference binary — run 'bash scripts/gate.sh --ref' BEFORE an engine change"
  fi
fi

# --- 7. protocol #4, enforced mechanically -----------------------------------
# Added because I broke it: a careless `git add -A` staged an uncommitted engine
# change alongside unrelated tooling, and it took a `git show --stat` to notice.
# Protocol #4 says no engine commit before a green bit-exact gate; a rule that
# lives only in a document is a rule that gets swept up by a wildcard.
printf "\n${B}staging${R}\n"
staged_engine=$(git diff --cached --name-only 2>/dev/null | grep -E '^(csrc/|patches/)' || true)
if [ -n "$staged_engine" ]; then
  if [ "$can_run" -eq 1 ] && [ "$fail" -eq 0 ]; then
    say "$P" "engine change staged, and the bit-exact gate is green this run"
  else
    bad "ENGINE CHANGE STAGED without a green bit-exact gate (protocol #4):
$(printf '%s' "$staged_engine" | sed 's/^/      /')
      unstage it (git reset HEAD <file>) or re-run this gate when the window opens"
  fi
else
  say "$P" "no engine change staged"
fi

printf "\n"
if [ "$fail" -ne 0 ]; then
  printf "${B}GATE RED${R} — something that ran failed.\n"; exit 1
elif [ "$deferred" -ne 0 ]; then
  printf "${B}GATE PARTIAL${R} — static checks passed; model legs DEFERRED (see · above).\n"
  printf "This is NOT a green gate. Re-run when the window opens.\n"; exit 0
else
  printf "${B}GATE GREEN${R} — static and model legs all passed.\n"; exit 0
fi
