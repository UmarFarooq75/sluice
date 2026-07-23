"""E47 — isolate the CPU-config delta found by E46 (pre-reg: docs/lablog.md).

A shipped(ub128+pool64+mmap) A2 | B ub1+pool64+mmap | C ub1+nopool+mmap |
D ub1+nopool+mmapOFF(SLOT_DEV fallback hack, labeled) D2. N=64, prompt A, SLOTS=24.
HARD GATE: every leg reproduces the pinned hash. Decision by reviewer vs A<->A2 bar.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e47"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT_A = "Write a Python function that merges two sorted lists into one sorted list without using sort()."
PINNED64 = "7fff2b7b9461da2a"
N = 64

LEGS = [
    ("A_shipped",  "128", {"LLMSTREAM_PREFILL_SLOTS": "64"}),
    ("A2_shipped", "128", {"LLMSTREAM_PREFILL_SLOTS": "64"}),
    ("B_ub1_pool", "1",   {"LLMSTREAM_PREFILL_SLOTS": "64"}),
    ("C_ub1_nopool", "1", {}),
    ("D_ub1_nopool_nommap",  "1", {"LLMSTREAM_SLOT_DEV": "gpu"}),  # falls back to CPU; side effect = use_mmap false
    ("D2_ub1_nopool_nommap", "1", {"LLMSTREAM_SLOT_DEV": "gpu"}),
]


def run_leg(label, ubatch, extra):
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24"})
    for k in ("LLMSTREAM_PREFILL_SLOTS", "LLMSTREAM_SLOT_DEV", "LLMSTREAM_NGL"):
        e.pop(k, None)
    e.update(extra)
    p = subprocess.run([BIN, MODEL, str(N), PROMPT_A, ubatch],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    pre = re.findall(r"prefill:\s+([\d.]+) s", p.stdout)
    tps = re.findall(r"decode:\s+[\d.]+ s \(([\d.]+) tok/s\)", p.stdout)
    hit = re.findall(r"decode_misses=\d+ \(hit ([\d.]+)\)", both)
    stall = re.findall(r"stall=([\d.]+) s", both)
    fell_back = "using CPU" in both
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None,
               extra={"prefill_s": pre[-1] if pre else None, "tps": tps[-1] if tps else None,
                      "hit": hit[-1] if hit else None, "stall": stall[-1] if stall else None,
                      "cpu_fallback": fell_back})


legs = []
for label, ub, extra in LEGS:
    print(f"leg {label} ...", flush=True)
    legs.append(run_leg(label, ub, extra))

problems = []
for l in legs:
    if not l.void and l.hash != PINNED64:
        problems.append(f"{l.label}: hash {l.hash} != pinned {PINNED64} (HARD RED)")
for l in legs:
    if not l.void and l.label.startswith("D") and not l.extra["cpu_fallback"]:
        problems.append(f"{l.label}: expected the SLOT_DEV CPU-fallback warning, not found — leg is not the intended config (VOID by meaning)")

rows = ["| leg | prefill s | decode tok/s | hit | stall s |", "|---|---|---|---|---|"]
for l in legs:
    if l.void:
        rows.append(f"| {l.label} | VOID ({l.void_reason}) |  |  |  |")
    else:
        rows.append("| %s | %s | %s | %s | %s |" % (
            l.label, l.extra["prefill_s"], l.extra["tps"], l.extra["hit"], l.extra["stall"]))

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if problems else "GREEN")
lines = [
    "E47 — isolate the CPU-config delta (%s) | LIGHT LOAD (7.0 GB gate, owner-directed)" % time.strftime("%Y-%m-%d %H:%M"),
    "N=%d prompt A SLOTS=24 CPU | steps: A ub128+pool+mmap -> B ub1 -> C -pool -> D -mmap (SLOT_DEV hack)" % N,
    void_banner(legs), "",
    ("problems:\n  " + "\n  ".join(problems)) if problems else "hash gate clean; D legs confirmed CPU-fallback config",
    "", *rows, "",
    "(step attribution and ship decision by reviewer vs the A<->A2 noise bar)",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
