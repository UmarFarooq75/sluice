"""E44b — warmpack CLEAN leg (pre-reg: docs/lablog.md).

Prompt B is held out from the pack (gptossA.warmpack.pack was built from prompt A).
Legs: OFF, OFF repeat (noise bar), ON — at N in {8, 20}. SLOTS=24, exact greedy.

HARD GATE: ON hash == OFF hash per N (warm-start is logit-neutral). Prompt B has
no pinned hash; OFF is the in-run reference. OFF↔OFF hash mismatch VOIDS the run.
Decision rule (fixed pre-run): clean dHit @ N=8 > 2x OFF-OFF bar (floor 1.0 pt)
=> warmpack ships opt-in with receipts; otherwise closed for good.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e44b"; OUT.mkdir(parents=True, exist_ok=True)
PACK = str(ROOT / "results" / "warmpack_ab" / "gptossA.warmpack.pack")
PROMPT_B = "Explain how the Earth formed and why it can support life."

ARMS = [("off", {}), ("off2", {}), ("on", {"LLMSTREAM_WARMPACK": PACK})]


def run_leg(arm, extra, n):
    label = f"{arm}_n{n}"
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24"})
    e.pop("LLMSTREAM_WARMPACK", None); e.update(extra)
    p = subprocess.run([BIN, MODEL, str(n), PROMPT_B, "1"],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    hit = re.findall(r"decode_misses=\d+ \(hit ([\d.]+)\)", both)
    tps = re.findall(r"([\d.]+) tok/s", both)
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None,
               extra={"n": n, "arm": arm, "hit": hit[-1] if hit else None,
                      "tps": tps[-1] if tps else None})


legs = []
for n in (8, 20):
    for arm, extra in ARMS:
        print(f"leg {arm} N={n} ...", flush=True)
        legs.append(run_leg(arm, extra, n))

by = {(l.extra["arm"], l.extra["n"]): l for l in legs}
def get(arm, n):
    l = by.get((arm, n))
    return None if (l is None or l.void) else l

problems, rows = [], []
for n in (8, 20):
    off, off2, on = get("off", n), get("off2", n), get("on", n)
    if not (off and off2 and on):
        rows.append(f"| {n} | VOID leg present — no comparison |")
        continue
    if off.hash != off2.hash:
        problems.append(f"N={n}: OFF/OFF hash mismatch ({off.hash} vs {off2.hash}) — run VOID")
        continue
    neutral = "yes" if on.hash == off.hash else "HASH MISMATCH (HARD RED)"
    if on.hash != off.hash:
        problems.append(f"N={n}: ON hash {on.hash} != OFF {off.hash}")
    rows.append("| %d | %s | %s | %s | %s / %s / %s | %s |" % (
        n, off.extra["hit"], off2.extra["hit"], on.extra["hit"],
        off.extra["tps"], off2.extra["tps"], on.extra["tps"], neutral))

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if problems else "GREEN")
lines = [
    "E44b — warmpack clean leg (%s) | LIGHT LOAD (7.0 GB gate, owner-directed)" % time.strftime("%Y-%m-%d %H:%M"),
    "prompt B (held out from pack), SLOTS=24, exact greedy, ubatch=1 | pack=gptossA (CLEAN by construction)",
    void_banner(legs), "",
    "| N | OFF hit | OFF2 hit | ON hit | tok/s off/off2/on | logit-neutral |",
    "|---|---|---|---|---|---|",
    *rows, "",
    ("problems:\n  " + "\n  ".join(problems)) if problems else "no hash problems",
    "",
    "(decision vs the pre-registered bar is written by the reviewer, not this script)",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
