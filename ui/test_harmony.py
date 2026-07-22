"""Parser fixtures for the harmony channel split. Run: python3 ui/test_harmony.py

Exists because two live-UI symptoms got past code review (S2, 2026-07-22):

  (a) channel internals reaching the user, rendered as stray punctuation. The UI
      streams a PARTIAL buffer, so it can end mid-marker ("<|chan"). Every strip
      regex required a closing "|>", so the fragment survived to st.markdown, which
      mangles tag-like text.
  (b) a reply with no final channel shown AS the answer. The model was still in the
      analysis channel when it stopped (token cap, or Fast derailing the structure);
      the UI presented that thinking as if it were the reply.

THE INVARIANT: channel internals must NEVER be visible, at any buffer boundary.
"""
import re
import sys
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "ui" / "app.py").read_text()
ns = {"re": re}
exec(src[src.index("def strip_harmony"):src.index("# ---------------- sidebar")], ns)
strip_harmony, split_harmony = ns["strip_harmony"], ns["split_harmony"]

A = "<|channel|>analysis<|message|>"
F = "<|start|>assistant<|channel|>final<|message|>"

CASES = [
    ("normal two-channel reply", A + "Reasoning here.<|end|>" + F + "The answer.<|return|>",
     "The answer.", "Reasoning here."),
    ("final channel only", F + "Direct.<|return|>", "Direct.", ""),
    ("analysis only — symptom (b)", A + "Still thinking.", "", "Still thinking."),
    ("commentary channel present",
     "<|channel|>commentary<|message|>tool<|end|>" + F + "A.<|return|>", "A.", "tool"),
    ("no markers at all", "Just plain text.", "Just plain text.", ""),
    ("empty", "", "", ""),
]

# symptom (a): EVERY truncation point of a normal stream. The UI renders the buffer
# on each chunk, so any prefix can be displayed — none may leak internals.
FULL = A + "Reasoning here.<|end|>" + F + "The answer.<|return|>"
PREFIXES = [FULL[:i] for i in range(1, len(FULL) + 1)]


def main():
    bad = 0
    for name, raw, want_final, want_think in CASES:
        f, t = split_harmony(raw)
        ok = f == want_final and t == want_think
        leak = "<|" in f or "<|" in t
        if not ok or leak:
            bad += 1
            print(f"FAIL  {name}\n      final={f!r} (want {want_final!r})"
                  f"\n      think={t!r} (want {want_think!r}) leak={leak}")
        else:
            print(f"pass  {name}")

    leaks = []
    for p in PREFIXES:
        f, t = split_harmony(p)
        if "<|" in f or "<|" in t:
            leaks.append(p[-18:])
    if leaks:
        bad += 1
        print(f"FAIL  streaming prefixes: {len(leaks)}/{len(PREFIXES)} leak internals")
        print(f"      e.g. …{leaks[0]!r}")
    else:
        print(f"pass  streaming prefixes ({len(PREFIXES)} truncation points, zero leaks)")

    print("\n" + ("FAILURES: %d" % bad if bad else "all harmony fixtures pass"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
