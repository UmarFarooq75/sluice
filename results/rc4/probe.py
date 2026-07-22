"""RC4 — confound cleanup, length reconciliation, and the attribution question.

LOCALIZE ONLY. No fixes.

Three groups, one window.

G1 CROSS-SHAPE AT THREE LENGTHS, POOL OFF.
    RC3's "pool not required" rested on B_ub4(pool=0) vs A_ref(pool=1), which moves
    two things at once. Here the pool-off family (ub 1/2/4) is compared against
    ITSELF, so a split inside it needs no pool at all. ub=1 is pool-off by
    construction (the pool engages only at ne[1] > 1), so it is both the pool-off
    anchor and the gate-path-vs-UI-path leg the directive asked for.

    Three lengths reconcile R0 (23 tokens, everything EQUAL) with RC3 (245 tokens,
    splits). Pre-registered:
      H1 sliding-window threshold (=128): short EQUAL, mid EQUAL, long DIFFER
      H2 chunk-count threshold (~>10):    mid at ub=4 (25 chunks) DIFFERS
      H3 kernel-path switch by chunk size: tracks size alone, length-independent
    The MID length at ub=4 is the cell that separates H1 from H2.

G2 LOCALIZATION. DEBUG_HASH on the smallest failing config vs the reference,
    comparing the FIRST DECODE PASS only — 1 token in both, so those dbg blocks
    align; prefill blocks cannot align by construction and are excluded.

G3 ATTRIBUTION. csrc/stock_probe.cpp built against PRISTINE b10064, same alignment
    split, zero streaming. Answers whether this is inherited upstream numerics or
    ours. Skips itself (never fabricates) if the pristine build is absent.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, compare, void_banner, verdict_line  # noqa: E402

sys.argv = [sys.argv[0], "rc4"]
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("rc2b", ROOT / "results" / "rc2" / "bisect.py")
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)

OUT = ROOT / "results" / "rc4"
OUT.mkdir(parents=True, exist_ok=True)
SWA = 128
NGEN = 1  # prefill-dominated: we are asking about the prefill, not the generation

SHORT_SYS = "You are a helpful assistant."
MID_SYS = ("You are a helpful assistant. Answer carefully and concisely. Prefer "
           "concrete examples over abstractions. When a question is ambiguous, say "
           "which reading you chose. Do not speculate beyond the evidence given, and "
           "state plainly when something is unknown rather than guessing at it.")
PRISTINE = Path("/private/tmp/claude-501/-Users-umarfarooq-Desktop-research/"
                "9e0ac63a-c167-4b2c-8f7c-911a29143f78/scratchpad/pristine")


def prompts():
    """Three lengths. LONG reuses RC3's exact turn-2 rendering."""
    t1, b1 = B.server("RC4_t1", [B.hist([("user", B.P1)])], 200, SHORT_SYS, 1, False)
    rep = B.reply_of(b1[0])
    return [("short", SHORT_SYS, B.hist([("user", B.P1)])),
            ("mid", MID_SYS, B.hist([("user", B.P1)])),
            ("long", SHORT_SYS, B.hist([("user", B.P1), ("assistant", rep),
                                        ("user", B.P2)]))]


def final_pass_dbg(leg):
    """dbg lines of the LAST forward pass only - the 1-token decode, the only pass
    whose shape matches across legs. Split on the last layer-0 restart."""
    lines = re.findall(r"^dbg\s+(\S+)\s+([0-9a-f]{16})$", leg.extra.get("txt", ""), re.M)
    if not lines:
        return []
    starts = [i for i, (n, _) in enumerate(lines) if re.search(r"[-_]0$", n)]
    return lines[starts[-1]:] if starts else lines


def chunks(n, ub):
    return [min(ub, n - i * ub) for i in range((n + ub - 1) // ub)]


def g3_stock(text):
    """Attribution leg. Returns rows or a skip reason — never a fabricated number."""
    lib = PRISTINE / "build" / "bin"
    if not (lib / "libllama.dylib").exists():
        return None, f"pristine build absent at {lib} — leg SKIPPED, not inferred"
    exe = ROOT / "csrc" / "stock_probe_pristine"
    src = ROOT / "csrc" / "stock_probe.cpp"
    cc = ["clang++", "-O3", "-std=c++17", f"-I{PRISTINE}/include",
          f"-I{PRISTINE}/ggml/include", str(src), f"-L{lib}", "-lllama", "-lggml",
          "-lggml-base", f"-Wl,-rpath,{lib}", "-o", str(exe)]
    # errors="replace": the engine can emit a token piece that splits a multi-byte
    # UTF-8 char across a pipe read, which raises UnicodeDecodeError under strict
    # decoding (S1v3 VOID). Lossy DISPLAY decode is safe because the comparison
    # source of truth is the ASCII "tok <id> |piece|" lines from PRINT_TOKS.
    r = subprocess.run(cc, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        return None, f"stock_probe build failed: {r.stderr[-300:]}"
    pf = OUT / "stock_prompt.txt"; pf.write_text(text)
    rows = []
    for ub in (512, 64, 4):
        p = subprocess.run([str(exe), str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf"),
                            str(pf), str(ub)], capture_output=True, text=True, errors="replace")
        h = re.search(r"logits_hash=(\w+)", p.stdout)
        am = re.search(r"argmax=(\d+)", p.stdout)
        nt = re.search(r"n_tokens=(\d+)", p.stdout)
        rows.append((ub, h.group(1) if h else None, am.group(1) if am else None,
                     int(nt.group(1)) if nt else None, p.returncode))
        (OUT / f"stock_ub{ub}.out").write_text(p.stdout + "\n--stderr--\n" + p.stderr[-2000:])
    return rows, None


def main():
    a = B.avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    L = [f"RC4 — confound cleanup, length reconciliation, attribution "
         f"({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail at start {a:.2f} GB | SLOTS=16 greedy | n_gen={NGEN} | sliding_window={SWA}",
         "LOCALIZE ONLY.", ""]
    all_legs, g1 = [], {}
    PROMPTS = prompts()               # engine runs ONCE, not per reference
    LONG_PROMPT = PROMPTS[2][2]

    # ---- G1: cross-shape at three lengths, pool-off family vs itself ----------
    for lname, sysmsg, prompt in PROMPTS:
        legs = {}
        for ub, pool in ((512, True), (1, False), (2, False), (4, False)):
            lab = f"{lname}_ub{ub}{'_pool' if pool else ''}"
            legs[ub] = B.single(lab, prompt, NGEN, sysmsg, ub, pool)
            all_legs.append(legs[ub])
            n = legs[ub].extra["rendered"] or 0
            print(f"  {lab:22} rendered={n:<4} chunks={len(chunks(n, ub)):<4} "
                  f"hash={legs[ub].hash}", flush=True)
        g1[lname] = legs
    n_by_len = {k: (v[512].extra["rendered"] or 0) for k, v in g1.items()}

    L.append("G1 — cross-shape, POOL OFF family (ub 1/2/4) vs itself and vs pool-on ref")
    L.append(f"  {'length':8} {'tokens':>7} {'>SWA?':>6}  {'ub1 vs ub2':>12} {'ub1 vs ub4':>12} "
             f"{'ub2 vs ub4':>12} {'ub1 vs ref':>12}")
    g1v = {}
    for lname, legs in g1.items():
        n = n_by_len[lname]
        vs = [compare(legs[1], legs[2])[0], compare(legs[1], legs[4])[0],
              compare(legs[2], legs[4])[0], compare(legs[1], legs[512])[0]]
        g1v[lname] = vs
        L.append(f"  {lname:8} {n:>7} {('YES' if n > SWA else 'no'):>6}  "
                 + " ".join(f"{v:>12}" for v in vs))
    L += ["", "  (ub1/ub2/ub4 are all pool-OFF by construction — the pool engages only at",
          "   ne[1] > 1 — so any split among the first three columns needs no pool at all.)",
          "  (ub1 vs ref is the GATE-PATH vs UI-PATH comparison, never made before.)"]

    # hypothesis adjudication
    pooloff_split = {k: any(v != "EQUAL" for v in vs[:3]) for k, vs in g1v.items()}
    L += ["", "  hypothesis adjudication (pre-registered):"]
    mid_n = n_by_len.get("mid", 0)
    if not pooloff_split["short"] and not pooloff_split["mid"] and pooloff_split["long"]:
        L.append("    H1 (sliding-window threshold) SUPPORTED: splits appear only over 128.")
    elif pooloff_split["mid"] and mid_n <= SWA:
        L.append("    H2 (chunk-count threshold) SUPPORTED: mid splits while under the window.")
    elif not any(pooloff_split.values()):
        L.append("    NO pool-off split at any length -> RC3's 'pool not required' is NOT")
        L.append("    supported by a clean comparison; the pool IS implicated after all.")
    else:
        L.append("    MIXED — no pre-registered hypothesis fits; reporting, not forcing one.")

    # ---- G2: localization ----------------------------------------------------
    L += ["", "G2 - DEBUG_HASH localization, first decode pass only"]
    ref_leg, bad_leg = g1["long"][512], g1["long"][4]
    if compare(ref_leg, bad_leg)[0] != "DIFFER":
        L.append("  SKIPPED - the long ub=4 leg did not diverge from the reference in")
        L.append("  this run, so there is nothing to localize. Not manufacturing a target.")
    else:
        dbg = {}
        for tag, ub, pool in (("G2_ref", 512, True), ("G2_bad", 4, False)):
            dbg[tag] = B.single(tag, LONG_PROMPT, NGEN, SHORT_SYS, ub, pool,
                                extra={"LLMSTREAM_DEBUG_HASH": "1"})
            all_legs.append(dbg[tag])
        fa, fb = final_pass_dbg(dbg["G2_ref"]), final_pass_dbg(dbg["G2_bad"])
        if not fa or not fb:
            L.append("  VOID - no dbg lines captured; cannot localize")
        elif len(fa) != len(fb):
            L.append(f"  VOID - final-pass dbg blocks differ in length ({len(fa)} vs "
                     f"{len(fb)}); not the same shape, must not be diffed positionally")
        else:
            i = next((k for k, (x, y) in enumerate(zip(fa, fb)) if x[1] != y[1]), None)
            if i is None:
                L.append(f"  ALL {len(fa)} ffn_moe_* tensors of the final pass are IDENTICAL")
                L.append("  yet the hashes differ -> the fault is OUTSIDE the MoE path")
                L.append("  (attention / norm / output head). DEBUG_HASH coverage is the limit.")
            else:
                L.append(f"  FIRST DIVERGENT NODE: {fa[i][0]}  ({fa[i][1]} vs {fb[i][1]})")
                L.append(f"  position {i} of {len(fa)}; identical up to that node")

    # ---- G3: attribution ------------------------------------------------------
    L += ["", "G3 — ATTRIBUTION: pristine b10064, zero streaming, same alignment split"]
    rows, skip = g3_stock(LONG_PROMPT)
    if skip:
        L.append(f"  SKIPPED: {skip}")
    else:
        ref = rows[0]
        L.append(f"  {'ub':>5} {'n_tok':>6} {'rc':>3}  {'argmax':>7}  logits_hash")
        for ub, h, am, nt, rc in rows:
            L.append(f"  {ub:>5} {str(nt):>6} {rc:>3}  {str(am):>7}  {h}")
        diff = [r for r in rows[1:] if r[1] != ref[1]]
        amdiff = [r for r in rows[1:] if r[2] != ref[2]]
        L.append("")
        if diff:
            L.append("  PRISTINE b10064 SPLITS ON ALIGNMENT TOO -> INHERITED UPSTREAM NUMERICS.")
            L.append("  => E39 and E41b gate 1 reclassify as a BACKEND PROPERTY, not our bugs.")
            L.append("     Our bit-exact gates remain sound as CONFIG-PINNED guarantees.")
            L.append(f"     argmax {'ALSO flips' if amdiff else 'is stable'} across shapes.")
        else:
            L.append("  PRISTINE b10064 does NOT split -> the sensitivity is OURS.")
            L.append("  => localize within patches/llmstream.patch.")

    L = [x for x in L]
    L.insert(3, void_banner(all_legs))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
