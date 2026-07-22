#!/usr/bin/env python3
"""Refuse any summary that compares against a VOID leg.

The rule: A MISSING VALUE IS NOT A DATA POINT. It has produced a confident, wrong,
publishable verdict twice, so it is checked mechanically rather than remembered:

  E41b run 1  results/e41b/run1_void/  — a regex without re.M missed `canon_reply:`,
      the harness substituted its default "", leg A rendered an EMPTY assistant turn
      (315 tokens vs the other legs' 421), and the summary printed
      "GATE 1 faithfulness A == C : RED" — a comparison between two different prompts.

  R0          results/r0/summary_AUTO_WRONG.txt — legs A8/A8_g1 hit D12 and exited
      rc=1 with no hash. The summary compared None against a hash and diffed an empty
      token list, printing "ARGMAX STABLE ACROSS SHAPES : NO". Every leg that ran had
      produced the identical hash; the truth was the exact opposite of the verdict.

Exit 1 if any summary pairs a declared VOID leg with a verdict token.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from harness import VOID_MARKER, VERDICT_TOKENS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    checked = 0
    for s in sorted((ROOT / "results").glob("*/summary*.txt")):
        if "AUTO_WRONG" in s.name:
            continue  # deliberately preserved bad artifact, kept as evidence
        txt = s.read_text()
        m = re.search(rf"^{re.escape(VOID_MARKER)}\s*(.*)$", txt, re.M)
        if not m:
            continue  # pre-dates the banner; not retroactively enforced
        checked += 1
        if m.group(1).strip() == "none":
            continue
        names = [n.split("(")[0].strip() for n in m.group(1).split(",") if n.strip()]
        for line in txt.splitlines():
            if line.startswith(VOID_MARKER) or "VOID" in line:
                continue
            for n in names:
                if not n or not re.search(rf"\b{re.escape(n)}\b", line):
                    continue
                if any(re.search(rf"\b{re.escape(v)}\b", line) for v in VERDICT_TOKENS):
                    problems.append(f"{s.relative_to(ROOT)}: VOID leg '{n}' carries a "
                                    f"verdict -> {line.strip()[:80]}")
    for p in problems:
        print(p)
    if problems:
        return 1
    print(f"{checked} summary file(s) declare VOID status; none compares against a VOID leg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
