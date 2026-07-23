"""E46 — backend A/B at the shipped regime (pre-reg: docs/lablog.md).

Legs: cpu, cpu2 (noise bar), gpu, gpu2 (determinism pair). N=64, prompt A,
SLOTS=24. Gates: cpu legs reproduce the pinned hash; gpu legs must equal EACH
OTHER (backend determinism) — never the CPU pin (different kernels, different
floats, RC4 logic); gpu-vs-cpu token ids compared, RC4-class judgment by reviewer.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e46"; OUT.mkdir(parents=True, exist_ok=True)
PROMPT_A = "Write a Python function that merges two sorted lists into one sorted list without using sort()."
PINNED64 = "7fff2b7b9461da2a"
N = 64

GPU_ENV = {"LLMSTREAM_SLOT_DEV": "gpu", "LLMSTREAM_NGL": "99"}
LEGS = [("cpu", {}), ("cpu2", {}), ("gpu", GPU_ENV), ("gpu2", GPU_ENV)]


def run_leg(label, extra):
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24", "LLMSTREAM_PRINT_TOKS": "1"})
    for k in ("LLMSTREAM_SLOT_DEV", "LLMSTREAM_NGL", "LLMSTREAM_PREFILL_SLOTS"):
        e.pop(k, None)
    ubatch = "1" if extra else "128"
    if not extra:
        e["LLMSTREAM_PREFILL_SLOTS"] = "64"
    e.update(extra)
    p = subprocess.run([BIN, MODEL, str(N), PROMPT_A, ubatch],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    toks = [int(x) for x in re.findall(r"^tok\s+(\d+) ", p.stdout, re.M)]
    pre = re.findall(r"prefill:\s+([\d.]+) s", p.stdout)
    tps = re.findall(r"decode:\s+[\d.]+ s \(([\d.]+) tok/s\)", p.stdout)
    hit = re.findall(r"decode_misses=\d+ \(hit ([\d.]+)\)", both)
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None, toks=toks,
               extra={"prefill_s": pre[-1] if pre else None,
                      "tps": tps[-1] if tps else None,
                      "hit": hit[-1] if hit else None})


legs = {}
for label, extra in LEGS:
    print(f"leg {label} ...", flush=True)
    legs[label] = run_leg(label, extra)

L = legs
problems, notes = [], []
for c in ("cpu", "cpu2"):
    if not L[c].void and L[c].hash != PINNED64:
        problems.append(f"{c} hash {L[c].hash} != pinned {PINNED64} (HARD RED)")
if not L["gpu"].void and not L["gpu2"].void:
    if L["gpu"].hash != L["gpu2"].hash:
        problems.append(f"gpu backend NOT deterministic: {L['gpu'].hash} vs {L['gpu2'].hash} (HARD RED)")
    else:
        notes.append("gpu backend deterministic (gpu == gpu2 hash %s)" % L["gpu"].hash)
if not L["cpu"].void and not L["gpu"].void:
    div = next((i for i, (x, y) in enumerate(zip(L["cpu"].toks, L["gpu"].toks)) if x != y),
               None if len(L["cpu"].toks) == len(L["gpu"].toks) else min(len(L["cpu"].toks), len(L["gpu"].toks)))
    notes.append("cpu-vs-gpu token ids: " + ("FULL MATCH (%d)" % len(L["cpu"].toks) if div is None
                 else "diverge@%d (cross-backend, RC4-class judgment by reviewer)" % div))

rows = ["| leg | prefill s | decode tok/s | hit | hash |", "|---|---|---|---|---|"]
for label, _ in LEGS:
    l = L[label]
    if l.void:
        rows.append(f"| {label} | VOID ({l.void_reason}) |  |  |  |")
    else:
        rows.append("| %s | %s | %s | %s | %s |" % (
            label, l.extra["prefill_s"], l.extra["tps"], l.extra["hit"], l.hash))

incomplete = any(l.void for l in L.values())
gate = "INCOMPLETE" if incomplete else ("RED" if problems else "GREEN")
lines = [
    "E46 — backend A/B at shipped regime (%s) | LIGHT LOAD (7.0 GB gate, owner-directed)" % time.strftime("%Y-%m-%d %H:%M"),
    "prompt A, N=%d, SLOTS=24 | cpu: ubatch=128 + prefill pool 64 | gpu: SLOT_DEV=gpu NGL=99 ubatch=1" % N,
    void_banner(list(L.values())), "",
    ("problems:\n  " + "\n  ".join(problems)) if problems else "hash gates clean",
    *["  " + x for x in notes], "",
    *rows, "",
    "(policy decision vs pre-registered falsifier by reviewer)",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
