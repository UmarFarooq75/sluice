"""E44 — auto-pin retest at the shipped regime (pre-reg: docs/lablog.md).

12 single-shot legs: {LRU, LRU repeat, lfru, warmpack} x N in {8, 20, 64}.
Prompt A, exact greedy, streamed, SLOTS=24, guard ON, ubatch=1.

HARD GATE: every leg's logits_hash must equal the pinned prompt-A hash for its N.
A residency policy that changes the math is a feature-breaking bug, whatever its
speed. Comparisons (hit rate, tok/s, TTFT) are made only between non-VOID legs.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = os.environ.get("E44_MODEL", str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf"))
OUT = ROOT / "results" / "e44"; OUT.mkdir(parents=True, exist_ok=True)
# the only real 24-layer gpt-oss pack; built FROM prompt A, so the warm arm is a
# CONTAMINATED upper bound (pre-run amendment in docs/lablog.md fixes the rule:
# no lift here kills warmpack for good; a lift requires a clean leg before claims)
PACK = str(ROOT / "results" / "warmpack_ab" / "gptossA.warmpack.pack")

PROMPT_A = "Write a Python function that merges two sorted lists into one sorted list without using sort()."
PINNED = {8: "fdf0f83dd70504c5", 20: "e3fa62923ee35254", 64: "7fff2b7b9461da2a"}

ARMS = [
    ("lru",      {}),
    ("lru2",     {}),                                   # noise bar: same config twice
    ("lfru",     {"LLMSTREAM_EVICT": "lfru"}),
    ("warmpack", {"LLMSTREAM_WARMPACK": PACK}),
]


def run_leg(arm, extra, n):
    label = f"{arm}_n{n}"
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24"})
    e.pop("LLMSTREAM_EVICT", None); e.pop("LLMSTREAM_WARMPACK", None)
    e.update(extra)
    t0 = time.time()
    p = subprocess.run([BIN, MODEL, str(n), PROMPT_A, "1"],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    wall = time.time() - t0
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    hit = re.findall(r"decode_misses=\d+ \(hit ([\d.]+)\)", both)
    tps = re.findall(r"([\d.]+) tok/s", both)
    ttft = re.findall(r"ttft: total=(\d+) ms", both)
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None,
               extra={"n": n, "arm": arm, "hit": hit[-1] if hit else None,
                      "tps": tps[-1] if tps else None,
                      "ttft": ttft[-1] if ttft else None, "wall": f"{wall:.0f}"})


legs = []
for n in (8, 20, 64):
    for arm, extra in ARMS:
        print(f"leg {arm} N={n} ...", flush=True)
        legs.append(run_leg(arm, extra, n))

hash_red = []
for l in legs:
    if not l.void and l.hash != PINNED[l.extra["n"]]:
        hash_red.append(f"  {l.label}: got {l.hash} expected {PINNED[l.extra['n']]}")

by = {(l.extra["arm"], l.extra["n"]): l for l in legs}
def field(arm, n, k):
    l = by.get((arm, n))
    return "VOID" if (l is None or l.void) else (l.extra.get(k) or "?")

rows = ["| N | LRU hit | LRU2 hit | lfru hit | warm hit | LRU tok/s | lfru tok/s | warm tok/s | LRU ttft | warm ttft |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for n in (8, 20, 64):
    rows.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        n, field("lru", n, "hit"), field("lru2", n, "hit"), field("lfru", n, "hit"),
        field("warmpack", n, "hit"), field("lru", n, "tps"), field("lfru", n, "tps"),
        field("warmpack", n, "tps"), field("lru", n, "ttft"), field("warmpack", n, "ttft")))

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if hash_red else "GREEN")

lines = [
    "E44 — auto-pin retest at SLOTS=24 (%s)" % time.strftime("%Y-%m-%d %H:%M"),
    "model=%s prompt A, exact greedy, guard ON, ubatch=1, streamed | pack=%s "
    "(warm arm CONTAMINATED upper bound: pack source = test prompt)" % (
        os.path.basename(MODEL), os.path.basename(PACK)),
    void_banner(legs), "",
    "HARD GATE — pinned hash per N (policy must not change the math):",
    ("  all non-VOID legs reproduce their pinned hash" if not hash_red else
     "  HASH MISMATCH:\n" + "\n".join(hash_red)), "",
    "measured (comparisons only between non-VOID legs; verdicts vs the",
    "pre-registered noise bar are written by the reviewer, not this script):",
    *rows, "",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
