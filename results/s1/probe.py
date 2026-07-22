"""S1 — canon ship gates 1-2, rapid-fire pricing, and the rc5 precision pair.

Gate 1 (FAITHFULNESS): turn-2 token sequence with canon ON vs a fresh single-shot of
    the IDENTICAL rendering, on 3 distinct prompts. Tokens are the argmax sequence, so
    this tests exactly what a user sees. logits_hash equality is NOT required and NOT
    expected: RC4 showed cross-shape hash variance is an upstream property.
    PRE-REGISTERED CONSEQUENCE: any text mismatch on any prompt -> canon STAYS DARK,
    report to owner, no partial ship. Wired as a hard verdict, not a judgement call.

Gate 2 (TTFT): turn-2 TTFT ~3 s (accept <= 4.5 s). Falsifier > 6 s.

RAPID-FIRE (record, do NOT gate): canon runs after `text:` is printed and before
    <<<READY>>>. A user who sends the next message instantly queues behind it. That
    interval IS the queue delay, so it is measured directly rather than modelled.

RC5 (precision pair, folded in): pool-on vs pool-off at the SAME ubatch=4. Every
    previous comparison moved shape and pool together. ub=4 is the largest shape
    pool-off can take before D12 exits, so this is the only matched-shape pair.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, compare, void_banner, verdict_line  # noqa: E402

sys.argv = [sys.argv[0], "s1"]
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("rc2b", ROOT / "results" / "rc2" / "bisect.py")
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)

OUT = ROOT / "results" / "s1"; OUT.mkdir(parents=True, exist_ok=True)
BIN, MODEL = B.BIN, B.MODEL
NGEN = 200
# ROOT CAUSE of S1 run 1: SHORT_SYS has no "Reasoning:" line, so gpt-oss reasoned at
# default effort and burned all 200 tokens inside the analysis channel on p1/p3. The
# engine then correctly reported "canon skipped - no final-channel marker in this
# reply" - canon has nothing to canonicalise if the final channel was never reached.
# E41b run 2 worked because it used the UI's 1457-char system prompt, which ends with
# "Reasoning: low". Capping reasoning is what makes a 200-token budget sufficient.
SYS = B.SHORT_SYS + " Reasoning: low"

PROMPTS = [
    ("p1", "Explain how the Earth formed and why it can support life.",
     "Summarize that in three bullet points."),
    ("p2", "What is the difference between a stack and a queue?",
     "Give me one real use case for each."),
    ("p3", "Why does bread rise when you add yeast?",
     "What happens if the water is too hot?"),
]


def canon_session(label, p1, p2, rapid_fire=False):
    """Two (or three) turns with canon ON, timestamping the canon window.

    The canon cost sits between the `text:` line and <<<READY>>>, so timing those two
    events measures the exact interval a user would queue behind.
    """
    env = B.env_for(SYS, 1, True, server_mode=True,
                    extra={"LLMSTREAM_KV_CANON": "1", "LLMSTREAM_PRINT_TOKS": "1"})
    fe = open(OUT / f"{label}.err", "w")
    p = subprocess.Popen([BIN, MODEL, str(NGEN), "SENTINEL", "128"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=fe,
                         env=env, text=True, bufsize=1)
    marks, blocks = [], []

    def until_ready():
        buf, t_text = [], None
        for line in p.stdout:
            if line.startswith("text:") and t_text is None:
                t_text = time.time()
            if line.startswith("<<<READY>>>"):
                marks.append((t_text, time.time()))
                return "".join(buf)
            buf.append(line)
        marks.append((t_text, time.time()))
        return "".join(buf)

    until_ready()                                    # load
    marks.clear()
    t0 = time.time()
    p.stdin.write(B.hist([("user", p1)]) + "\n"); p.stdin.flush()
    blocks.append(until_ready())
    m = re.search(r"(?m)^canon_reply: (.*)$", blocks[0])
    if not m:
        # write the artifact BEFORE bailing: p1/p3 produced no .out at all in run 1,
        # so the evidence for why canon skipped lived only in stderr.
        (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
        p.kill(); fe.close()
        return None, None, None, "no canon_reply in turn 1 — canon did not run"
    rep = m.group(1).replace("\\n", "\n").replace("\\\\", "\\")
    turn2 = B.hist([("user", p1), ("assistant", rep), ("user", p2)])

    t_send2 = time.time()
    p.stdin.write(turn2.replace("\n", "\\n") + "\n"); p.stdin.flush()
    blocks.append(until_ready())
    rf = None
    if rapid_fire:
        # send turn 3 the instant turn 2's READY lands - i.e. the best case. The cost
        # a user actually feels is the text->READY interval measured below.
        m2 = re.search(r"(?m)^canon_reply: (.*)$", blocks[1])
        rep2 = m2.group(1).replace("\\n", "\n").replace("\\\\", "\\") if m2 else ""
        turn3 = B.hist([("user", p1), ("assistant", rep), ("user", p2),
                        ("assistant", rep2), ("user", "Thanks. One more sentence?")])
        t3 = time.time()
        p.stdin.write(turn3.replace("\n", "\\n") + "\n"); p.stdin.flush()
        blocks.append(until_ready())
        rf = time.time() - t3
    p.stdin.close()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    fe.close()
    (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
    # canon window per turn = text: -> READY
    canon_win = [round(r - t, 2) if t else None for t, r in marks]
    return blocks, turn2, {"canon_windows_s": canon_win, "rapid_fire_s": rf}, None


def toks_of(block):
    return re.findall(r"^tok\s+(\d+)\s+\|", block, re.M)


def main():
    a = B.avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    L = [f"S1 — canon ship gates ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail {a:.2f} GB | SLOTS=16 | NGEN={NGEN} | canon ON | CLEAN", ""]
    rows, all_legs, fails = [], [], []

    for i, (tag, p1, p2) in enumerate(PROMPTS):
        blocks, turn2, timing, err = canon_session(f"{tag}_canon", p1, p2,
                                                   rapid_fire=(i == 0))
        if err:
            rows.append((tag, "VOID", err, None, None, None)); fails.append(tag); continue
        fresh = B.single(f"{tag}_fresh", turn2, NGEN, SYS, 128, True,
                         extra={"LLMSTREAM_PRINT_TOKS": "1"})
        all_legs.append(fresh)
        a_toks, c_toks = toks_of(blocks[1]), fresh.extra.get("txt", "")
        c_toks = re.findall(r"^tok\s+(\d+)\s+\|", c_toks, re.M)
        if not a_toks or not c_toks:
            rows.append((tag, "VOID",
                         f"no tokens captured (canon {len(a_toks)}, fresh {len(c_toks)}) "
                         f"— PRINT_TOKS missing?", None, None, timing))
            fails.append(tag)
            print(f"  {tag}: VOID — no tokens captured", flush=True)
            continue
        same = a_toks == c_toks
        ttft = re.search(r"ttft: total=(\d+)", blocks[1])
        reused = re.search(r"reused=(\d+) reprefill=(\d+)", blocks[1])
        ttft_s = int(ttft.group(1)) / 1000 if ttft else None
        rows.append((tag, "MATCH" if same else "MISMATCH",
                     f"{len(a_toks)} vs {len(c_toks)} tokens", ttft_s,
                     reused.groups() if reused else None, timing))
        if not same:
            fails.append(tag)
        print(f"  {tag}: tokens {'MATCH' if same else 'MISMATCH'} "
              f"ttft={ttft_s}s canon_windows={timing['canon_windows_s']}", flush=True)

    L.append("GATE 1 — turn-2 token (argmax) sequence, canon ON vs fresh, identical rendering")
    L.append(f"  {'prompt':8} {'verdict':10} {'detail':22} {'ttft_s':>7}  reused/reprefill")
    for tag, v, d, ttft, ru, _ in rows:
        L.append(f"  {tag:8} {v:10} {d:22} {str(ttft):>7}  {ru}")
    g1 = "GREEN" if not fails else "RED"
    L += ["", f"  GATE 1: {g1}"]
    if fails:
        L += ["  PRE-REGISTERED CONSEQUENCE: canon STAYS DARK. Report to owner.",
              "  No partial ship. This is not a judgement call."]

    ttfts = [r[3] for r in rows if r[3] is not None]
    if ttfts:
        worst = max(ttfts)
        g2 = "GREEN" if worst <= 4.5 else ("AMBER" if worst <= 6.0 else "RED")
        L += ["", f"GATE 2 — turn-2 TTFT: worst {worst:.2f}s over {len(ttfts)} prompts -> {g2}",
              "  (target ~3 s, accept <=4.5, falsifier >6; stock baseline 8.22 s, cold 16.22 s)"]

    L += ["", "RAPID-FIRE (recorded, NOT a gate)"]
    for tag, _, _, _, _, t in rows:
        if not t:
            continue
        extra = ""
        if t["rapid_fire_s"] is not None:
            extra = "; turn-3 sent instantly completed in %.2f s" % t["rapid_fire_s"]
        L.append("  %s: canon window per turn (text: -> READY) = %s s%s"
                 % (tag, t["canon_windows_s"], extra))
    L.append("  The canon window IS the delay a user queues behind if they reply instantly.")

    # ---- RC5 precision pair -------------------------------------------------
    L += ["", "RC5 — pool isolated at MATCHED shape (ub=4 both arms)"]
    _, tp1, tp2 = PROMPTS[0]
    t1, b1 = B.server("rc5_t1", [B.hist([("user", tp1)])], NGEN, SYS, 1, False)
    rep = B.reply_of(b1[0])
    pr = B.hist([("user", tp1), ("assistant", rep), ("user", tp2)])
    on = B.single("rc5_ub4_poolon", pr, 1, SYS, 4, True)
    off = B.single("rc5_ub4_pooloff", pr, 1, SYS, 4, False)
    all_legs += [on, off]
    v, d = compare(on, off)
    L.append(verdict_line("pool ON vs pool OFF @ ub=4", v, d))
    # print the interpretation that MATCHES the result. Printing both branches around
    # one verdict reads as analysis but is just a template, and a reader can pick the
    # half they like.
    if v == "DIFFER":
        L.append("  => the pool changes numerics at matched shape: pool isolated.")
    elif v == "EQUAL":
        L.append("  => pool EXONERATED at this shape; RC4's ub1-vs-ref split attributes")
        L.append("     to SHAPE, not pool. Caveat: ub=4 is the only shape pool-off can")
        L.append("     take before D12 exits, so this is one shape, not a general claim.")
    else:
        L.append("  => VOID; no interpretation.")

    L.insert(2, void_banner(all_legs))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
