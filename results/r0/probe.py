"""R0 — batch-shape invariance probe. Harness only; the engine is unchanged.

QUESTION: does the target's argmax (and its logits) depend on the SHAPE of the
batch a token is computed in? Three open threads hang on this — E39 (KV restore,
failed its hash gate for an unestablished reason), E41b (KV canon, gate pending),
and E42 speculative decoding (K-token verification vs 1-token decode).

HOW, without touching the engine: cparams.n_ubatch comes from argv[4], and the
prompt is submitted as ONE llama_batch that llama.cpp splits into n_ubatch pieces.
So varying n_ubatch varies the prefill batch shape and nothing else. Decode is
always 1 token per pass, so any difference downstream is inherited from prefill.

TWO FAMILIES, because the prefill pool is a second variable and must not be
confounded with batch shape:
  A (pool OFF)  ubatch 1,2,4,8   — pure batch-shape variation, one code path
  B (pool ON)   ubatch 1,128     — 128 is D12's path (per-layer union > slots,
                                   which is exactly what the pool exists for)
Pool logic only engages at ne[1] > 1, so A1 and B1 exercise identical code and
MUST produce identical hashes. That is an internal consistency check on this
harness, not a result.

VALIDITY PRECONDITION: leg B128 reproduces E37c's published hash 7fff2b7b9461da2a
(same prompt, N=64, SLOTS=16, PREFILL_SLOTS=64, ubatch=128). If it does not, this
harness is not measuring what E37c measured and the run is void.

That comparison rests on one claim, so it is stated rather than assumed: E37c ran
WITHOUT LLMSTREAM_PRINT_TOKS and these legs run with it. PRINT_TOKS is numerically
inert — it adds a printf and flips the `special` flag of llama_token_to_piece, which
changes only the rendered `text:` string, never a logit or a routing decision. If
B128 still matches E37c's hash, that inertness is also empirically confirmed by this
very run, which is part of why the check is worth having.

TWO DISTINCT QUESTIONS, deliberately reported separately:
  argmax stability  — token sequence identical? This is what SPEC-DEC needs. If
                      argmax is stable, speculative decoding emits identical text
                      regardless of logit bit-noise.
  bit equality      — logits_hash identical? This is what OUR GATE demands. It is
                      strictly stronger. E39 failed exactly here while its text
                      matched, so the two can and do come apart.

n_gen=1 legs answer "first-divergence position" directly: if prefill numerics
differ at all, the very first sampled logits already differ, so a divergence at
step 1 is the expected signature rather than something later in the run.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "r0"
OUT.mkdir(parents=True, exist_ok=True)

# prompt A — the canonical bit-exact-gate prompt, and the one E37c used
PROMPT = ("Write a Python function that merges two sorted lists into one sorted "
          "list without using sort().")
E37C_HASH = "7fff2b7b9461da2a"   # E37c: N=64 SLOTS=16 PREFILL_SLOTS=64 ubatch=128
NGEN = 64

BASE = {"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "16", "LLMSTREAM_PRINT_TOKS": "1"}
# greedy by construction: TEMP/REP_PEN unset => argmax, no sampler, reproducible


def avail_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, errors="replace").stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    p = sum(int(m.group(1)) for k in
            ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            if (m := re.search(rf"{k}:\s+(\d+)", out)))
    return p * ps / 1e9


def leg(label, ubatch, ngen, pool):
    """One run. Returns dict of parsed results; never raises on engine failure —
    a crash here is a FINDING (D12 predicts exit(1) when union > slots with no
    pool), not a harness error."""
    env = os.environ.copy()
    env.update(BASE)
    if pool:
        env["LLMSTREAM_PREFILL_SLOTS"] = "64"
    else:
        env.pop("LLMSTREAM_PREFILL_SLOTS", None)
    t0 = time.time()
    with open(OUT / f"{label}.out", "w") as fo, open(OUT / f"{label}.err", "w") as fe:
        rc = subprocess.call([BIN, MODEL, str(ngen), PROMPT, str(ubatch)],
                             stdout=fo, stderr=fe, env=env)
    txt = (OUT / f"{label}.out").read_text()
    h = re.search(r"logits_hash=(\w+)", txt)
    toks = re.findall(r"^tok\s+(\d+)\s+\|", txt, re.M)
    return {"label": label, "ubatch": ubatch, "ngen": ngen, "pool": pool, "rc": rc,
            "hash": h.group(1) if h else None, "toks": toks,
            "n_toks": len(toks), "secs": time.time() - t0,
            "hit": (m.group(1) if (m := re.search(r"\(hit ([\d.]+)\)", txt)) else None),
            "tps": (m.group(1) if (m := re.search(r"^decode:.*?\(([\d.]+) tok/s", txt, re.M)) else None)}


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return (None if len(a) == len(b) else min(len(a), len(b))), None, None


def main():
    a = avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    legs = [("A1", 1, NGEN, False), ("A2", 2, NGEN, False),
            ("A4", 4, NGEN, False), ("A8", 8, NGEN, False),
            ("B1", 1, NGEN, True), ("B128", 128, NGEN, True),
            # first-step legs: is the divergence present at the very first logits?
            ("A1_g1", 1, 1, False), ("A8_g1", 8, 1, False), ("B128_g1", 128, 1, True)]
    res = {}
    for label, ub, ng, pool in legs:
        res[label] = r = leg(label, ub, ng, pool)
        print(f"  {label:9} ubatch={ub:<4} n_gen={ng:<3} pool={int(pool)} "
              f"rc={r['rc']} hash={r['hash']} toks={r['n_toks']} {r['secs']:.0f}s", flush=True)

    L = [f"R0 — batch-shape invariance probe ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail at start {a:.2f} GB | SLOTS=16 greedy prompt A | CLEAN", ""]

    # --- validity precondition -------------------------------------------
    ok = res["B128"]["hash"] == E37C_HASH
    L += [f"VALIDITY: B128 hash {res['B128']['hash']} vs E37c {E37C_HASH} -> "
          f"{'MATCH (harness reproduces the published artifact)' if ok else 'MISMATCH — RESULTS VOID'}",
          f"CONSISTENCY: A1 vs B1 (pool inert at ne=1) -> "
          f"{'identical' if res['A1']['hash'] == res['B1']['hash'] else 'DIFFER — pool is not inert at ubatch=1'}",
          ""]

    L.append("legs")
    for k, r in res.items():
        L.append(f"  {k:9} ubatch={r['ubatch']:<4} n_gen={r['ngen']:<3} pool={int(r['pool'])} "
                 f"rc={r['rc']} hash={r['hash']} hit={r['hit']} tok/s={r['tps']} toks={r['n_toks']}")

    # --- the two questions, kept apart -----------------------------------
    ref = res["A1"]
    L += ["", "vs A1 (ubatch=1, pool off) — the reference shape"]
    L.append(f"  {'leg':9} {'bit-equal':>10} {'argmax-stable':>14}  first token divergence")
    argmax_all, bits_all = True, True
    crashed = [k for k, r in res.items() if r["rc"] != 0 or not r["hash"]]
    for k in ("A2", "A4", "A8", "B1", "B128"):
        r = res[k]
        if r["rc"] != 0 or not r["hash"]:
            L.append(f"  {k:9} {'CRASHED':>10} {'—':>14}  rc={r['rc']}, no hash produced — "
                     f"excluded from the verdict (a crash is not a divergence)")
            continue
        be = r["hash"] == ref["hash"]
        i, x, y = first_diff(ref["toks"], r["toks"])
        st = i is None
        argmax_all &= st
        bits_all &= be
        where = "-" if st else f"step {i}: A1 emitted {x}, {k} emitted {y}"
        L.append(f"  {k:9} {('yes' if be else 'NO'):>10} {('yes' if st else 'NO'):>14}  {where}")

    L += ["", "first-step legs (n_gen=1): does the very first sampled logit already differ?"]
    g1 = {k: res[k]["hash"] for k in ("A1_g1", "A8_g1", "B128_g1") if res[k]["rc"] == 0 and res[k]["hash"]}
    L.append(f"  A1_g1={g1['A1_g1']}  A8_g1={g1['A8_g1']}  B128_g1={g1['B128_g1']}")
    same_g1 = len(set(g1.values())) == 1
    L.append(f"  -> {'identical: prefill shape does NOT perturb the first logits'if same_g1 else 'DIFFER at step 1: prefill shape perturbs logits immediately, as expected if this is a batch-shape effect'}")

    if crashed:
        L += ["", f"LEGS THAT DID NOT RUN: {', '.join(crashed)} — excluded from the verdict below.",
              "  These are findings in their own right (see the .err files), not divergences."]
    L += ["", "=" * 72,
          f"ARGMAX STABLE ACROSS SHAPES : {'YES' if argmax_all else 'NO'}   (what spec-dec needs)",
          f"BIT-EQUAL ACROSS SHAPES     : {'YES' if bits_all else 'NO'}   (what our gate demands)",
          "=" * 72]
    if bits_all and argmax_all:
        L += ["", "CONSEQUENCE (pre-registered): batch shape is INVARIANT.",
              " - E39's hash failure is NOT explained by batch shape -> it is a real",
              "   restore bug and should be REOPENED as such.",
              " - E42 spec-dec keeps its Exact-tier claim; gate 1(b) is expected to pass.",
              " - E41b's pending gate has one fewer excuse available to it."]
    elif argmax_all and not bits_all:
        L += ["", "CONSEQUENCE (pre-registered): logits differ in the last bits, but ARGMAX",
              "does not flip. This is E39's exact signature (identical text, different hash).",
              " - Spec-dec would emit IDENTICAL TEXT while failing a bit-equality gate.",
              " - Forces an owner call on what 'Exact' means: identical output, or",
              "   identical logits? Our gate currently demands the stronger one.",
              " - E39/E41b/E42 re-scope together, or the gate's definition changes."]
    else:
        L += ["", "CONSEQUENCE (pre-registered): ARGMAX FLIPS with batch shape.",
              " - Bit-exact speculative decoding is IMPOSSIBLE on this backend.",
              " - E42 re-scopes to Balanced-tier pending owner call.",
              " - E39/E41b inherit the same verdict; one root cause covers all three."]

    L += ["", "INSTRUMENTATION GAP (stated, not worked around):",
          " max |delta logit| was requested and is NOT obtainable from this harness.",
          " The engine emits logits_hash (a digest) and token ids, never raw logits, so",
          " magnitude cannot be recovered without an engine change. What IS available",
          " and was not used here: LLMSTREAM_DEBUG_HASH prints a per-tensor FNV of every",
          " ffn_moe_* intermediate, which would localise a divergence to the first",
          " differing (layer, node) — arguably more actionable than a magnitude. It is",
          " left for a follow-up because enabling it changes which nodes the callback",
          " is asked about, and that is a perturbation this probe should not carry."]

    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
