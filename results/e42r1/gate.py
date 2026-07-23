"""E42-r1 — self-drafting speculative decoding (pre-reg: docs/lablog.md).

3 prompts x {off, spec K=2}. N=64, SLOTS=24, PREFILL_SLOTS=64 (batched verify
routes through the prefill pool; D12 protection), exact greedy, streamed.

Gates: (1) off leg on prompt A must equal pinned N=64 hash (off path of the new
binary is bit-exact; byte-identical-off vs ref runs in run.sh). (2) spec-on token
ids vs spec-off: full match expected; on divergence the script records the first
position and both texts — RC4-class judgment (isolated flip, coherent both sides)
belongs to the reviewer, systematic divergence is RED on its face. (3) accept
rate + tok/s + union telemetry recorded per leg.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, void_banner  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e42r1"; OUT.mkdir(parents=True, exist_ok=True)
N = 64
PINNED64_A = "7fff2b7b9461da2a"

PROMPTS = [
    ("code",  "Write a Python function that merges two sorted lists into one sorted list without using sort()."),
    ("prose", "Explain how the Earth formed and why it can support life."),
    ("reason", "What is the difference between TCP and UDP?"),
]


def run_leg(tag, prompt, spec):
    label = f"{tag}_{'spec2' if spec else 'off'}"
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "24",
              "LLMSTREAM_PREFILL_SLOTS": "64", "LLMSTREAM_PRINT_TOKS": "1"})
    e.pop("LLMSTREAM_SPEC", None)
    if spec: e["LLMSTREAM_SPEC"] = "2"
    p = subprocess.run([BIN, MODEL, str(N), prompt, "1"],
                       capture_output=True, text=True, errors="replace", env=e,
                       timeout=1800)
    (OUT / f"{label}.out").write_text(p.stdout)
    (OUT / f"{label}.err").write_text(p.stderr)
    both = p.stdout + "\n" + p.stderr
    h = re.findall(r"logits_hash=([0-9a-f]+)", p.stdout)
    toks = [int(x) for x in re.findall(r"^tok\s+(\d+) ", p.stdout, re.M)]
    tps = re.findall(r"decode:\s+[\d.]+ s \(([\d.]+) tok/s\)", p.stdout)
    spec_line = re.findall(r"^spec: (.*)$", p.stdout, re.M)
    text = re.findall(r"^text: (.*)$", p.stdout, re.M)
    return Leg(label, rc=p.returncode, hash=h[-1] if h else None, toks=toks,
               extra={"tps": tps[-1] if tps else None,
                      "spec": spec_line[-1] if spec_line else "",
                      "text": text[-1][:160] if text else ""})


legs, rows, problems = [], [], []
for tag, prompt in PROMPTS:
    print(f"{tag}: off ...", flush=True)
    off = run_leg(tag, prompt, False)
    print(f"{tag}: spec K=2 ...", flush=True)
    on = run_leg(tag, prompt, True)
    legs += [off, on]
    if tag == "code" and not off.void and off.hash != PINNED64_A:
        problems.append(f"off leg hash {off.hash} != pinned {PINNED64_A} (HARD RED)")
    if off.void or on.void:
        rows.append(f"| {tag} | VOID leg — no comparison |  |  |  |")
        continue
    div = next((i for i, (x, y) in enumerate(zip(off.toks, on.toks)) if x != y),
               None if len(off.toks) == len(on.toks) else min(len(off.toks), len(on.toks)))
    match = "FULL MATCH (%d toks)" % len(off.toks) if div is None else "diverge@%d" % div
    rows.append("| %s | %s | %s | %s | %s |" % (
        tag, match, off.extra["tps"], on.extra["tps"], on.extra["spec"]))
    if div is not None:
        rows.append("|  | off: %s |  |  |  |" % off.extra["text"].replace("|", "/"))
        rows.append("|  | on:  %s |  |  |  |" % on.extra["text"].replace("|", "/"))

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if problems else "GREEN")
lines = [
    "E42-r1 — self-drafting spec-dec, K=2 (%s) | LIGHT LOAD (7.0 GB gate, owner-directed)" % time.strftime("%Y-%m-%d %H:%M"),
    "N=%d SLOTS=24 PREFILL_SLOTS=64 exact greedy streamed | drafter: n-gram order 3->2 (pre-registered)" % N,
    void_banner(legs), "",
    ("problems:\n  " + "\n  ".join(problems)) if problems else "off-path hash: pinned hash reproduced on prompt A",
    "",
    "| prompt | token stream vs off | off tok/s | spec tok/s | spec telemetry |",
    "|---|---|---|---|---|",
    *rows, "",
    "(RC4-class divergence judgment and ship decision by reviewer per pre-registration)",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if gate in ("GREEN", "RED") else 1)
