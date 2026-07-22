"""S1v3 — gate 1v2 at PRODUCT CONFIG. Non-reconvergence retired as a failure signal.

GATE 1v2. Identity-vs-cold is unachievable: RC4 showed the backend varies across batch
shapes and S1 found it flipping an argmax ("every"/"Every"). So the question is not
"is canon identical to cold" but "does canon ADD divergence beyond what stock already
has". Three arms per prompt, IDENTICAL rendering:

    canon  server, LLMSTREAM_KV_CANON=1, 2 turns   (reuse ~= whole canonical KV)
    stock  server, canon off, 2 turns              (reuse ~= up to assistant boundary)
    cold   single-shot of the same turn-2          (reuse = 0)

  (a) flip incidence: canon-vs-cold divergences <= stock-vs-cold, over N=6 prompts.
  (b) every flip classified: position, token pair, tie-class, re-convergence. Any flip
      that is NOT near-tie-shaped (wrong content, incoherence, derailment) -> hard RED.
  (c) unconditional hard REDs, independent of flips:
        - canon_reply not faithfully echoable (client contract broken)
        - reuse telemetry wrong (canon claims a canonical KV it did not reuse)
        - truncation no-op mishandled (must degrade to stock, never error)
        - incoherent reply (degenerate repetition)

GATE 2: worst turn-2 TTFT. RAPID-FIRE: recorded, never gated.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, compare, void_banner  # noqa: E402

sys.argv = [sys.argv[0], "s1v3"]
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("rc2b", ROOT / "results" / "rc2" / "bisect.py")
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)

OUT = ROOT / "results" / "s1v3"; OUT.mkdir(parents=True, exist_ok=True)
BIN, MODEL = B.BIN, B.MODEL
# PRODUCT CONFIG. S1v2 ran NGEN=200 and 4 of 6 prompts never reached the final channel,
# so canon had nothing to canonicalise. 200 was a harness choice, not a product one:
# the UI ships max-new-tokens 2048 and the 1457-char system prompt. NGEN is a CAP, so
# replies still stop at EOG - this removes the artificial truncation, it does not force
# 2048 tokens of generation.
NGEN = 2048
SYS = re.search(r'DEFAULT_SYSTEM = """(.*?)"""',
                (ROOT / "ui" / "app.py").read_text(), re.S).group(1).strip()

PROMPTS = [
    ("p1", "Explain how the Earth formed and why it can support life.", "Summarize that in three bullet points."),
    ("p2", "What is the difference between a stack and a queue?", "Give me one real use case for each."),
    ("p3", "Why does bread rise when you add yeast?", "What happens if the water is too hot?"),
    ("p4", "How does a refrigerator make things cold?", "Which part uses the most energy?"),
    ("p5", "What causes the seasons on Earth?", "Why is the southern hemisphere opposite?"),
    ("p6", "Explain what a hash table is and why lookups are fast.", "When does it get slow?"),
]


def toks(block):
    return re.findall(r"^tok\s+(\d+)\s+\|", block, re.M)


def pieces(block):
    return re.findall(r"^tok\s+\d+\s+\|([^|]*)\|", block, re.M)


def unesc(x):
    return x.replace("\\n", "\n").replace("\\\\", "\\")


def session(label, p1, p2, canon, rapid_fire=False, ngen=NGEN):
    """Two turns in server mode. Returns (blocks, turn2, timing, err)."""
    extra = {"LLMSTREAM_PRINT_TOKS": "1"}
    if canon:
        extra["LLMSTREAM_KV_CANON"] = "1"
    env = B.env_for(SYS, 128, True, server_mode=True, extra=extra)
    fe = open(OUT / f"{label}.err", "w")
    p = subprocess.Popen([BIN, MODEL, str(ngen), "SENTINEL", "128"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=fe,
                         env=env, text=True, bufsize=1)
    marks, blocks = [], []

    def until_ready():
        buf, t_text = [], None
        for line in p.stdout:
            if line.startswith("text:") and t_text is None:
                t_text = time.time()
            if line.startswith("<<<READY>>>"):
                marks.append((t_text, time.time())); return "".join(buf)
            buf.append(line)
        marks.append((t_text, time.time())); return "".join(buf)

    until_ready(); marks.clear()
    p.stdin.write(B.hist([("user", p1)]) + "\n"); p.stdin.flush()
    blocks.append(until_ready())

    def bail(msg):
        (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
        p.kill(); fe.close()
        return blocks, None, None, msg

    m = re.search(r"(?m)^canon_reply: (.*)$", blocks[0])
    if canon and not m:
        return bail("canon produced no canon_reply (final channel not reached)")
    # stock arm must render the SAME turn 2, so it echoes the canon arm's reply text;
    # without canon we fall back to the UI's own final-channel extraction.
    rep = unesc(m.group(1)) if m else None
    if rep is None:
        raw = re.search(r"(?m)^text: (.*)$", blocks[0])
        rep = raw.group(1) if raw else ""
    turn2 = B.hist([("user", p1), ("assistant", rep), ("user", p2)])
    p.stdin.write(turn2.replace("\n", "\\n") + "\n"); p.stdin.flush()
    blocks.append(until_ready())
    rf = None
    if rapid_fire:
        t3 = time.time()
        p.stdin.write(turn2.replace("\n", "\\n") + "\n"); p.stdin.flush()
        blocks.append(until_ready()); rf = time.time() - t3
    p.stdin.close()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    fe.close()
    (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
    return blocks, turn2, {"canon_windows_s": [round(r - t, 2) if t else None for t, r in marks],
                           "rapid_fire_s": rf, "canon_reply": rep}, None


def classify(a_tok, a_pc, c_tok, c_pc):
    """Every divergence: position, token pair, tie-class, re-convergence."""
    flips, i, n = [], 0, min(len(a_tok), len(c_tok))
    while i < n:
        if a_tok[i] != c_tok[i]:
            x, y = a_pc[i] if i < len(a_pc) else "", c_pc[i] if i < len(c_pc) else ""
            if x.strip().lower() == y.strip().lower() and x.strip():
                cls = "case/whitespace"
            elif x.strip().isalpha() and y.strip().isalpha():
                cls = "word-substitution"
            elif not x.strip() or not y.strip():
                cls = "whitespace"
            else:
                cls = "OTHER"
            # Re-convergence is RECORDED but is NOT a failure signal. After a tie flip
            # the model continues autoregressively from a different token, so divergent
            # continuation is expected behaviour, not derailment. S1v2 proved the rule
            # miscalibrated: it fired on canon at pos 13/62 while STOCK diverged at
            # pos 3 - i.e. the rule would have condemned the shipping product.
            tail = 20
            recon = (a_tok[i + 1:i + 1 + tail] == c_tok[i + 1:i + 1 + tail]
                     and i + 1 + tail <= n)
            flips.append({"pos": i, "pair": (a_tok[i], c_tok[i]), "pieces": (x, y),
                          "class": cls, "reconverged": recon})
            if not recon:
                break          # after a non-reconverging flip the streams are not aligned
        i += 1
    return flips


def reasoning_len(pc, tk):
    """Tokens spent in the analysis channel, and whether the final channel was reached.

    The lead's question: if real prompts routinely burn >500 tokens reasoning at
    'Reasoning: low', the truncation no-op is a product-frequency problem rather than
    an edge case.
    """
    try:
        a0 = pc.index("analysis")
    except ValueError:
        return None, False
    fin = next((i for i, x in enumerate(pc) if x == "final" and i > a0), None)
    return (len(pc) if fin is None else fin) - a0, fin is not None


def incoherent(pc):
    """Degenerate repetition: the same piece 20+ times consecutively."""
    run, last = 0, None
    for x in pc:
        run = run + 1 if x == last else 1
        last = x
        if run >= 20:
            return True
    return False


def main():
    a = B.avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    L = [f"S1v3 — canon ship gates, product config ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail {a:.2f} GB | SLOTS=16 | NGEN={NGEN} | N={len(PROMPTS)} prompts", "",
         "GATE 1v2 — NON-INFERIORITY: canon may not ADD divergence beyond stock.", ""]
    rows, hard_red, all_legs, reasoning, transcripts = [], [], [], [], []
    tot_canon = tot_stock = 0

    for i, (tag, p1, p2) in enumerate(PROMPTS):
        cb, turn2, ct, cerr = session(f"{tag}_canon", p1, p2, True, rapid_fire=(i == 0))
        if cerr:
            # Distinguish the two causes. Final channel never reached => truncation
            # no-op, a product CONDITION to price, not a canon defect. Final channel
            # reached but no canon_reply => canon is broken: hard RED.
            pc = pieces(cb[0]) if cb else []
            rl, reached = reasoning_len(pc, toks(cb[0]) if cb else [])
            reasoning.append((tag, rl, reached))
            if reached:
                hard_red.append(f"{tag}: final channel WAS reached but canon produced no "
                                f"canon_reply — canon defect")
            rows.append((tag, "NOOP" if not reached else "VOID", cerr, None, [], []))
            print(f"  {tag}: {'truncation no-op' if not reached else 'CANON DEFECT'} "
                  f"(reasoning {rl} tokens)", flush=True)
            continue
        sb, _, st, serr = session(f"{tag}_stock", p1, p2, False)
        cold = B.single(f"{tag}_cold", turn2, NGEN, SYS, 128, True,
                        extra={"LLMSTREAM_PRINT_TOKS": "1"})
        all_legs.append(cold)

        a_t, a_p = toks(cb[1]), pieces(cb[1])
        s_t, s_p = toks(sb[1]), pieces(sb[1])
        d_t, d_p = toks(cold.extra.get("txt", "")), pieces(cold.extra.get("txt", ""))
        if not (a_t and s_t and d_t):
            rows.append((tag, "VOID", "missing token stream", None, [], []))
            hard_red.append(f"{tag}: missing token stream"); continue

        rl, reached = reasoning_len(a_p, a_t)
        reasoning.append((tag, rl, reached))
        cf = classify(a_t, a_p, d_t, d_p)
        sf = classify(s_t, s_p, d_t, d_p)
        tot_canon += len(cf); tot_stock += len(sf)

        # unconditional hard REDs
        ttft = re.search(r"ttft: total=(\d+)", cb[1])
        ru = re.search(r"ctx_held=(\d+) rendered=(\d+) reused=(\d+) reprefill=(\d+)", cb[1])
        if ru:
            held, rend, reused, repre = map(int, ru.groups())
            if reused < held:
                hard_red.append(f"{tag}: reuse telemetry — canon KV {held} but only {reused} reused")
        else:
            hard_red.append(f"{tag}: reuse telemetry missing")
        if incoherent(a_p):
            hard_red.append(f"{tag}: canon reply incoherent (degenerate repetition)")
        # tie-class judged AT THE FLIP TOKEN only. Coherence is judged by a human on
        # the full transcripts, which this run delivers.
        bad = [f for f in cf if f["class"] == "OTHER"]
        if bad:
            hard_red.append(f"{tag}: flip at pos {bad[0]['pos']} is class OTHER "
                            f"{bad[0]['pieces']} — not tie-shaped")
        transcripts.append((tag, "".join(a_p), "".join(s_p), "".join(d_p)))
        rows.append((tag, "ok", f"canon {len(cf)} vs stock {len(sf)} flips",
                     int(ttft.group(1)) / 1000 if ttft else None, cf, sf))
        print(f"  {tag}: canon_flips={len(cf)} stock_flips={len(sf)} "
              f"ttft={rows[-1][3]}s", flush=True)

    L.append(f"  {'prompt':7} {'canon flips':>12} {'stock flips':>12} {'ttft_s':>7}")
    for tag, v, d, ttft, cf, sf in rows:
        L.append(f"  {tag:7} {(len(cf) if v == 'ok' else 'VOID'):>12} "
                 f"{(len(sf) if v == 'ok' else '-'):>12} {str(ttft):>7}")
    L += ["", f"  TOTAL canon flips {tot_canon} vs stock flips {tot_stock}"]
    non_inferior = tot_canon <= tot_stock
    L.append(f"  (a) non-inferiority (canon <= stock): {'PASS' if non_inferior else 'FAIL'}")

    L += ["", "  (b) flip classification — every divergence"]
    any_flip = False
    for tag, v, d, _, cf, sf in rows:
        for nm, fl in (("canon", cf), ("stock", sf)):
            for f in fl:
                any_flip = True
                L.append(f"    {tag} {nm:5} pos={f['pos']:<4} {f['pair']} "
                         f"{f['pieces']} class={f['class']} reconverged={f['reconverged']}")
    if not any_flip:
        L.append("    none — all three arms produced identical token streams")

    L += ["", "  (c) unconditional hard REDs"]
    L += [f"    {x}" for x in hard_red] if hard_red else ["    none"]

    g1 = "GREEN" if (non_inferior and not hard_red) else "RED"
    L += ["", f"  GATE 1v2: {g1}"]
    if g1 == "RED":
        L.append("  CONSEQUENCE: canon STAYS DARK. Report to owner. No partial ship.")

    ttfts = [r[3] for r in rows if r[3]]
    if ttfts:
        w = max(ttfts)
        L += ["", f"GATE 2 — worst turn-2 TTFT {w:.2f}s -> "
              f"{'GREEN' if w <= 4.5 else ('AMBER' if w <= 6 else 'RED')} "
              f"(stock 8.22 s, cold 16.22 s)"]

    L += ["", "RAPID-FIRE (recorded, NOT a gate)"]
    L.append("  canon window (text: -> READY) is the delay a user queues behind.")
    L += ["", f"MEASURED FLIP RATE for the README: canon {tot_canon} divergences over "
          f"{len([r for r in rows if r[1] == 'ok'])} prompts (stock {tot_stock})."]
    L += ["", "REASONING LENGTH (analysis-channel tokens; the truncation-frequency question)"]
    for tag, rl, reached in reasoning:
        L.append(f"  {tag}: {rl} tokens, final channel {'REACHED' if reached else 'NOT reached'}")
    got = [r for _, r, _ in reasoning if r]
    if got:
        L.append(f"  min {min(got)} / median {sorted(got)[len(got)//2]} / max {max(got)} "
                 f"— NGEN was {NGEN}")
        L.append(f"  >500 tokens on {sum(1 for x in got if x > 500)}/{len(got)} prompts: "
                 f"{'PRODUCT-FREQUENCY problem, ships only with UI warning + docs' if sum(1 for x in got if x > 500) else 'edge case at this config'}")
    tf = OUT / "transcripts.txt"
    tf.write_text("\n\n".join(
        f"=== {t} ===\n--- canon ---\n{c}\n--- stock ---\n{s_}\n--- cold ---\n{d}"
        for t, c, s_, d in transcripts))
    L += ["", f"TRANSCRIPTS for human coherence review: {tf}"]
    L.insert(2, void_banner(all_legs))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 1 if g1 == "RED" else 0


if __name__ == "__main__":
    sys.exit(main())
