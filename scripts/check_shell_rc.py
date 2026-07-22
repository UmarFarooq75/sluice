#!/usr/bin/env python3
"""Refuse a launcher that can report success over a dead harness.

THIRD instance of "a missing value is not a data point", this time one layer up.
The Python layer got the rule (scripts/lib/harness.py) but the SHELL above it did
not, so a crashed harness still wrote DONE: done:

  e41b run 1  results/e41b/run1_void/       — silent default -> confident RED
  R0          results/r0/summary_AUTO_WRONG.txt — crash scored as divergence
  RC2         results/rc2/                  — bisect.py died on BrokenPipeError,
      run.sh logged rc=0 and wrote DONE: done. Cause:
          echo "[$(date ...)] legs finished rc=$?" ...
      bash expands left to right, so $(date) runs first and resets $? to date's
      status. rc was ALWAYS 0. The rule is only real if every layer obeys it.

Two checks per results/*/run.sh:
  1. no `$?` on a line that also contains a command substitution
  2. the DONE marker is written from a captured rc (sl_finish), not unconditionally
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems, checked = [], 0
    for f in sorted((ROOT / "results").glob("*/run.sh")):
        checked += 1
        for i, line in enumerate(f.read_text().splitlines(), 1):
            # $((...)) is ARITHMETIC expansion and does not run a command, so it
            # cannot clobber $?. Only $(...) command substitution does. Strip the
            # arithmetic form first or this flags the correct fix as the bug.
            probe = re.sub(r"\$\(\(.*?\)\)", "", line)
            if "$?" in probe and "$(" in probe:
                problems.append(f"{f.relative_to(ROOT)}:{i}: `$?` on a line with a command "
                                f"substitution — $? is clobbered before it is read: {line.strip()[:60]}")
        txt = f.read_text()
        # a DONE written unconditionally after the harness call cannot express failure
        for m in re.finditer(r'^\s*echo\s+"?done"?\s*>\s*"\$OUT/DONE"', txt, re.M):
            problems.append(f"{f.relative_to(ROOT)}: writes DONE unconditionally; "
                            f"use sl_finish \"$rc\" \"$OUT\" so a crashed harness writes void")
    for p in problems:
        print(p)
    if problems:
        return 1
    print(f"{checked} launcher(s): exit codes propagate, DONE reflects harness status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
