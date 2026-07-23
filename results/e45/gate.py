"""E45 — page-cache probe (pre-reg: docs/lablog.md).

Leg order IS the experiment: A1 (default, F_NOCACHE) -> B1 (pagecache) ->
B2 (pagecache, warmed) -> A2 (default again, after B warmed the cache).
HARD GATE: every leg's hash == pinned N=64 hash. Metrics: tok/s, stall, read MB,
avg bw — comparisons only between non-VOID legs; decision by reviewer.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e45"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT_A = "Write a Python function that merges two sorted lists into one sorted list without using sort()."
PINNED64 = "7fff2b7b9461da2a"
N = 64

LEGS = [("A1_default", {}), ("B1_pagecache", {"LLMSTREAM_PAGECACHE": "1"}),
        ("B2_pagecache", {"LLMSTREAM_PAGECACHE": "1"}), ("A2_default", {})]


def run_leg(label, extra):
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24"})
    e.pop("LLMSTREAM_PAGECACHE", None); e.update(extra)
    p = subprocess.run([BIN, MODEL, str(N), PROMPT_A, "1"],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    tps = re.findall(r"([\d.]+) tok/s", both)
    stall = re.findall(r"stall=([\d.]+) s", both)
    mb = re.findall(r"total_read=([\d.]+) MB", both)
    bw = re.findall(r"avg_bw=(\d+) MB/s", both)
    hit = re.findall(r"decode_misses=\d+ \(hit ([\d.]+)\)", both)
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None,
               extra={"tps": tps[-1] if tps else None, "stall": stall[-1] if stall else None,
                      "mb": mb[-1] if mb else None, "bw": bw[-1] if bw else None,
                      "hit": hit[-1] if hit else None})


legs = []
for label, extra in LEGS:
    print(f"leg {label} ...", flush=True)
    legs.append(run_leg(label, extra))

problems = [f"  {l.label}: got {l.hash} expected {PINNED64}"
            for l in legs if not l.void and l.hash != PINNED64]
rows = ["| leg | hit | tok/s | stall s | read MB | avg bw |", "|---|---|---|---|---|---|"]
for l in legs:
    if l.void:
        rows.append(f"| {l.label} | VOID ({l.void_reason}) |  |  |  |  |")
    else:
        x = l.extra
        rows.append("| %s | %s | %s | %s | %s | %s |" % (
            l.label, x["hit"], x["tps"], x["stall"], x["mb"], x["bw"]))

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if problems else "GREEN")
lines = [
    "E45 — page-cache probe (%s) | LIGHT LOAD (7.0 GB gate, owner-directed)" % time.strftime("%Y-%m-%d %H:%M"),
    "prompt A, N=64, SLOTS=24, streamed | order: A1 default -> B1 -> B2 pagecache -> A2 default",
    void_banner(legs), "",
    "HARD GATE — pinned N=64 hash %s:" % PINNED64,
    ("  all non-VOID legs reproduce it" if not problems else "  HASH MISMATCH:\n" + "\n".join(problems)),
    "", *rows, "",
    "(decision vs pre-registered rule by reviewer, not this script)",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
