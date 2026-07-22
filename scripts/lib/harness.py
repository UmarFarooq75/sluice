"""Shared harness primitives. The rule this file exists to enforce:

    A MISSING VALUE IS NOT A DATA POINT.

It has bitten twice, both times producing a confident, wrong, publishable verdict:

  1. E41b run 1 (results/e41b/run1_void/) — a regex without re.M missed the engine's
     `canon_reply:` line, so the harness silently substituted its default "". Leg A
     then rendered a conversation with an EMPTY assistant turn, 315 tokens against
     the other legs' 421, and the summary reported "GATE 1 faithfulness: RED" from a
     comparison between two different prompts.

  2. R0 (results/r0/summary_AUTO_WRONG.txt) — legs A8/A8_g1 hit D12 and exited rc=1
     with no hash. The summary compared None against a hash ("not equal") and diffed
     an empty token list ("diverges at step 0"), and printed
     "ARGMAX STABLE ACROSS SHAPES : NO". The truth was the exact opposite: every leg
     that ran produced the identical hash.

Both were local fixes at the time. This module makes it structural: a leg that did
not produce a result is VOID, and a VOID leg can never appear as a side of an
equality. It renders as VOID and taints the comparison, instead of silently
becoming "not equal to" whatever it is held against.

`scripts/gate.sh` refuses any summary that pairs a declared VOID leg with a verdict
token, so the rule is checked and not merely documented.
"""

VOID_MARKER = "VOID_LEGS:"
# tokens that assert an outcome. A VOID leg must never appear on a line with one.
VERDICT_TOKENS = ("yes", "NO", "GREEN", "RED", "MATCH", "DIFFER",
                  "identical", "IDENTICAL", "PASS", "FAIL", "equal")


class Leg:
    """One measurement. `void` is the only thing callers should branch on."""

    def __init__(self, label, rc=0, hash=None, toks=None, extra=None):
        self.label = label
        self.rc = rc
        self.hash = hash
        self.toks = toks if toks is not None else []
        self.extra = extra or {}

    @property
    def void(self):
        return self.rc != 0 or not self.hash

    @property
    def void_reason(self):
        if self.rc != 0:
            return f"rc={self.rc}, produced no result"
        if not self.hash:
            return "no logits_hash in output"
        return ""

    def __repr__(self):
        return f"<Leg {self.label} {'VOID' if self.void else self.hash}>"


def compare(a, b, what="hash"):
    """Compare two Legs. Returns (verdict, detail).

    verdict is one of: "EQUAL", "DIFFER", "VOID". A VOID verdict is NOT a failure
    and NOT a pass — it means the comparison could not be made, which is a different
    fact and must be reported as one.
    """
    voids = [x for x in (a, b) if x.void]
    if voids:
        return "VOID", "; ".join(f"{x.label}: {x.void_reason}" for x in voids)
    if what == "hash":
        same = a.hash == b.hash
        return ("EQUAL" if same else "DIFFER",
                f"{a.label}={a.hash} {b.label}={b.hash}")
    if what == "toks":
        for i, (x, y) in enumerate(zip(a.toks, b.toks)):
            if x != y:
                return "DIFFER", f"step {i}: {a.label} emitted {x}, {b.label} emitted {y}"
        if len(a.toks) != len(b.toks):
            return "DIFFER", f"length {len(a.toks)} vs {len(b.toks)}"
        return "EQUAL", f"{len(a.toks)} tokens identical"
    raise ValueError(f"unknown comparison: {what}")


def void_banner(legs):
    """The machine-readable declaration gate.sh keys off. Always emit it — an
    absent banner and 'no void legs' must not look the same to the checker."""
    v = [l for l in legs if l.void]
    if not v:
        return f"{VOID_MARKER} none"
    return f"{VOID_MARKER} " + ", ".join(f"{l.label} ({l.void_reason})" for l in v)


def verdict_line(label, verdict, detail=""):
    """Render one comparison. VOID never renders as a yes/no."""
    if verdict == "VOID":
        return f"  {label:26} VOID — comparison not possible ({detail})"
    return f"  {label:26} {verdict}" + (f"  {detail}" if detail else "")


def summarize_all(pairs):
    """pairs = [(label, verdict, detail)]. Returns (overall, lines).

    overall is "PASS" only if every comparison is EQUAL. If any is VOID the overall
    is "INCOMPLETE" — deliberately not "FAIL", because an unrun leg is not evidence
    of a defect, and deliberately not "PASS", because absence of evidence is not
    evidence of absence.
    """
    lines = [verdict_line(l, v, d) for l, v, d in pairs]
    if any(v == "VOID" for _, v, _ in pairs):
        return "INCOMPLETE", lines
    if all(v == "EQUAL" for _, v, _ in pairs):
        return "PASS", lines
    return "FAIL", lines
