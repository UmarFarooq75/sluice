"""RC1 — localize the reuse-path divergence to a (layer, node). LOCALIZE ONLY.

R0 exonerated batch shape. E41b runs 1 and 2 showed that an identical 421-token
rendering yields three different hashes down three KV paths (canon-reused 404,
stock-reused 296, fresh 0), reproducibly. So the fault lives in the reuse path:
llama_memory_seq_rm plus partial re-prefill.

INSTRUMENT: LLMSTREAM_DEBUG_HASH prints `dbg <tensor-name> <fnv>` for every
ffn_moe_* intermediate. Tensor names carry the layer index, so diffing two runs'
streams names the first differing (layer, node) — the propagation origin.

ALIGNMENT, which is the whole reason this works: the two legs' PREFILL passes have
different batch composition and therefore emit different numbers of dbg lines, so
they cannot be diffed positionally. But the last forward pass in each leg is a
1-TOKEN DECODE (n_gen=1), and those lines align one-to-one. This probe compares
only that final aligned block and says so, rather than diffing whole streams and
pretending the offsets mean something.

NO FIXES IN THIS RUNG.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, compare, void_banner, verdict_line  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "rc1"
OUT.mkdir(parents=True, exist_ok=True)

# smallest reproducer: short prompts, one generated token
P1 = "Name three primary colors."
P2 = "Which is warmest?"
SYSTEM = "You are a helpful assistant."
NGEN = 1

BASE = {"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "16", "LLMSTREAM_SYSTEM": SYSTEM,
        "LLMSTREAM_PHASE_TIMERS": "1"}


def avail_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    p = sum(int(m.group(1)) for k in
            ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            if (m := re.search(rf"{k}:\s+(\d+)", out)))
    return p * ps / 1e9


def hist(turns):
    return "\x1e".join(f"{r}\x1f{c}" for r, c in turns)


def run_single(label, prompt, dbg, extra=None):
    env = os.environ.copy(); env.update(BASE)
    if dbg:
        env["LLMSTREAM_DEBUG_HASH"] = "1"
    if extra:
        env.update(extra)
    with open(OUT / f"{label}.out", "w") as fo, open(OUT / f"{label}.err", "w") as fe:
        rc = subprocess.call([BIN, MODEL, str(NGEN), prompt, "1"],
                             stdout=fo, stderr=fe, env=env)
    return parse(label, rc)


def run_server(label, requests, dbg, extra=None):
    env = os.environ.copy(); env.update(BASE)
    env.update({"LLMSTREAM_SERVER": "1", "LLMSTREAM_IDLE_EXIT": "180"})
    if dbg:
        env["LLMSTREAM_DEBUG_HASH"] = "1"
    if extra:
        env.update(extra)
    fe = open(OUT / f"{label}.err", "w")
    p = subprocess.Popen([BIN, MODEL, str(NGEN), "SENTINEL", "1"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=fe,
                         env=env, text=True, bufsize=1)

    def until_ready():
        buf = []
        for line in p.stdout:
            if line.startswith("<<<READY>>>"):
                return "".join(buf)
            buf.append(line)
        return "".join(buf)

    until_ready()
    blocks = []
    for req in requests:
        p.stdin.write(req.replace("\n", "\\n") + "\n"); p.stdin.flush()
        blocks.append(until_ready())
    p.stdin.close()
    try:
        rc = p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill(); rc = -9
    fe.close()
    (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
    return parse(label, rc)


def parse(label, rc):
    txt = (OUT / f"{label}.out").read_text()
    h = re.findall(r"logits_hash=(\w+)", txt)
    return Leg(label, rc=rc, hash=(h[-1] if h else None),
               extra={"txt": txt,
                      "rendered": re.findall(r"rendered=(\d+)", txt),
                      "reused": re.findall(r"reused=(\d+)", txt)})


def final_pass_dbg(leg):
    """dbg lines of the LAST forward pass only — the 1-token decode, which is the
    only pass whose shape matches across legs. Split on layer-0 restarts."""
    lines = re.findall(r"^dbg\s+(\S+)\s+([0-9a-f]{16})$", leg.extra["txt"], re.M)
    if not lines:
        return []
    starts = [i for i, (n, _) in enumerate(lines) if re.search(r"-0$|_0$|\b0\b", n)]
    return lines[starts[-1]:] if starts else lines


def main():
    a = avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    turn2 = hist([("user", P1), ("assistant", "Red, blue and yellow."), ("user", P2)])

    print("RC1-C  control: is DEBUG_HASH numerically inert?", flush=True)
    C_off = run_single("RC1-C_dbgoff", turn2, dbg=False)
    C_on = run_single("RC1-F_fresh", turn2, dbg=True)
    inert, inert_detail = compare(C_off, C_on)

    print("RC1-R  reused-prefix continuation", flush=True)
    R = run_server("RC1-R_reused", [hist([("user", P1)]), turn2], dbg=True)

    kvfile = OUT / "rc1.kv"
    kvfile.unlink(missing_ok=True)
    print("RC1-E39  persist/restore path", flush=True)
    run_server("RC1-E39_save", [turn2], dbg=False, extra={"LLMSTREAM_KV_PERSIST": str(kvfile)})
    E39 = run_server("RC1-E39_restore", [turn2], dbg=True,
                     extra={"LLMSTREAM_KV_PERSIST": str(kvfile)})

    legs = [C_off, C_on, R, E39]
    L = [f"RC1 — reuse-path divergence localization ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail at start {a:.2f} GB | SLOTS=16 greedy | n_gen={NGEN} | CLEAN",
         "LOCALIZE ONLY — no fixes in this rung.", "",
         void_banner(legs), ""]

    L.append("CONTROL — DEBUG_HASH inertness (everything below depends on this)")
    L.append(verdict_line("dbg off vs dbg on", inert, inert_detail))
    if inert != "EQUAL":
        L.append("  *** DEBUG_HASH is NOT inert. Localization below is NOT trustworthy. ***")

    L += ["", "hashes"]
    for lg in legs:
        L.append(f"  {lg.label:20} rc={lg.rc} hash={lg.hash} "
                 f"rendered={lg.extra['rendered']} reused={lg.extra['reused']}")

    for name, lg in (("RC1-R (reused prefix)", R), ("RC1-E39 (restored KV)", E39)):
        v, d = compare(C_on, lg)
        L += ["", f"{name} vs RC1-F (fresh, identical rendering)",
              verdict_line("logits_hash", v, d)]
        if v != "DIFFER":
            L.append("  no divergence to localize in this leg")
            continue
        fa, fb = final_pass_dbg(C_on), final_pass_dbg(lg)
        if not fa or not fb:
            L.append("  VOID — no dbg lines captured; cannot localize")
            continue
        if len(fa) != len(fb):
            L.append(f"  VOID — final-pass dbg blocks differ in length ({len(fa)} vs {len(fb)}); "
                     f"they are not the same shape and must not be diffed positionally")
            continue
        first = next((i for i, (x, y) in enumerate(zip(fa, fb)) if x[1] != y[1]), None)
        if first is None:
            L.append(f"  ALL {len(fa)} ffn_moe_* tensors of the final pass are IDENTICAL, yet the "
                     f"hashes differ -> outcome 3: the fault is OUTSIDE the MoE path "
                     f"(attention / norm / output head). DEBUG_HASH coverage is the limit.")
        else:
            L.append(f"  FIRST DIVERGENT NODE: {fa[first][0]}  ({fa[first][1]} vs {fb[first][1]})")
            L.append(f"  position {first} of {len(fa)} in the final 1-token pass")
            L.append(f"  identical up to that node: {first} tensors")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
