"""RC2 — bisect from E41b's known-failing config to RC1's known-passing one.

LOCALIZE ONLY. No fixes.

THE UNIT OF MEASUREMENT is a PAIR, and it is always the same comparison:
    reused  = server, turn 1, then turn 2 which reuses a KV prefix
    fresh   = single-shot of the IDENTICAL turn-2 rendering, reused=0
    diverged  <=>  hash(reused) != hash(fresh)
Both arms must render the same number of tokens or the pair is VOID — that premise
check is the lesson of E41b run 1, where a 315-vs-421 mismatch produced a confident
RED from two different prompts.

THE CONFOUND THIS SCRIPT EXISTS TO BREAK: gpt-oss's sliding_window is 128 and
E41b's n_ubatch was also 128. Any leg that moves both proves nothing. So:

  S1a / S1b cross the 128-token context boundary by TWO INDEPENDENT ROUTES —
    S1a varies turn-1 GENERATION length at fixed prompt,
    S1b varies PROMPT length at fixed short generation.
  Crossing 128 is the only factor common to both. If divergence tracks the crossing
  in both routes, it is the window and not generation-length or prompt-length.
  If it tracks only one route, that route's dimension is the mechanism instead.

  S2 varies the prefill pool at a fixed over-window config (candidate 2 / D14).
  S3 uses ubatch=64 at context ~300, so ubatch != sliding_window (candidate 3).

Every leg runs ubatch=1 and pool OFF unless it is specifically testing those, so the
pool is not silently active the way it was in E41b but not in RC1.
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
OUT = ROOT / "results" / "rc2"
OUT.mkdir(parents=True, exist_ok=True)
SWA = 128  # gpt-oss.attention.sliding_window, read from GGUF metadata

SHORT_SYS = "You are a helpful assistant."
LONG_SYS = re.search(r'DEFAULT_SYSTEM = """(.*?)"""',
                     (ROOT / "ui" / "app.py").read_text(), re.S).group(1).strip()
P1 = "Explain how the Earth formed and why it can support life."
P2 = "Summarize that in three bullet points."


def avail_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    p = sum(int(m.group(1)) for k in
            ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            if (m := re.search(rf"{k}:\s+(\d+)", out)))
    return p * ps / 1e9


def env_for(system, ubatch, pool, extra=None):
    e = os.environ.copy()
    e.update({"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "16",
              "LLMSTREAM_SYSTEM": system, "LLMSTREAM_PHASE_TIMERS": "1"})
    if pool:
        e["LLMSTREAM_PREFILL_SLOTS"] = "64"
    else:
        e.pop("LLMSTREAM_PREFILL_SLOTS", None)
    if extra:
        e.update(extra)
    return e


def hist(turns):
    return "\x1e".join(f"{r}\x1f{c}" for r, c in turns)


def parse(label, rc, text):
    (OUT / f"{label}.out").write_text(text)
    h = re.findall(r"logits_hash=(\w+)", text)
    rend = re.findall(r"rendered=(\d+)", text)
    reus = re.findall(r"ctx_held=\d+ rendered=\d+ reused=(\d+)", text)
    return Leg(label, rc=rc, hash=(h[-1] if h else None),
               extra={"rendered": int(rend[-1]) if rend else None,
                      "reused": int(reus[-1]) if reus else None,
                      "ttft": (m.group(1) if (m := re.search(r"ttft: total=(\d+)", text)) else None)})


def single(label, prompt, ngen, system, ubatch, pool):
    with open(OUT / f"{label}.err", "w") as fe:
        p = subprocess.run([BIN, MODEL, str(ngen), prompt, str(ubatch)],
                           stdout=subprocess.PIPE, stderr=fe, text=True,
                           env=env_for(system, ubatch, pool))
    return parse(label, p.returncode, p.stdout)


def server(label, requests, ngen, system, ubatch, pool):
    fe = open(OUT / f"{label}.err", "w")
    p = subprocess.Popen([BIN, MODEL, str(ngen), "SENTINEL", str(ubatch)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=fe,
                         env=env_for(system, ubatch, pool), text=True, bufsize=1)

    def until_ready():
        buf = []
        for line in p.stdout:
            if line.startswith("<<<READY>>>"):
                return "".join(buf)
            buf.append(line)
        return "".join(buf)

    until_ready()
    blocks = []
    for r in requests:
        p.stdin.write(r.replace("\n", "\\n") + "\n"); p.stdin.flush()
        blocks.append(until_ready())
    p.stdin.close()
    try:
        rc = p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill(); rc = -9
    fe.close()
    return parse(label, rc, "\n<<<TURN>>>\n".join(blocks)), blocks


def reply_of(block):
    m = re.search(r"(?m)^text: (.*)$", block)
    return m.group(1).strip() if m else ""


def pair(name, system, ngen, ubatch, pool):
    """One bisect point. Returns (verdict, detail, legs, facts)."""
    # two-step: turn 1 first, so turn 2 echoes its REAL reply. Anything else
    # renders a different conversation in the two arms — E41b run 1's exact defect.
    t1, b1 = server(f"{name}_t1", [hist([("user", P1)])], ngen, system, ubatch, pool)
    rep = reply_of(b1[0])
    turn2 = hist([("user", P1), ("assistant", rep), ("user", P2)])
    reused, _ = server(f"{name}_reused", [hist([("user", P1)]), turn2],
                       ngen, system, ubatch, pool)
    fresh = single(f"{name}_fresh", turn2, ngen, system, ubatch, pool)
    facts = {"rendered_reused": reused.extra["rendered"],
             "rendered_fresh": fresh.extra["rendered"],
             "reused": reused.extra["reused"],
             "ctx_turn1": (t1.extra["rendered"] or 0) + ngen,
             "ttft_reused": reused.extra["ttft"], "ttft_fresh": fresh.extra["ttft"]}
    # premise: both arms MUST render the same prompt (E41b run 1's lesson)
    if facts["rendered_reused"] != facts["rendered_fresh"]:
        return "VOID", (f"renderings differ ({facts['rendered_reused']} vs "
                        f"{facts['rendered_fresh']}) — not the same prompt"), [reused, fresh], facts
    v, d = compare(reused, fresh)
    return v, d, [reused, fresh], facts


def main():
    a = avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if subprocess.run(["pgrep", "-f", "csrc/stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    # name, system, ngen, ubatch, pool, expectation
    plan = [
        # S1a — cross 128 via GENERATION length, prompt fixed short
        ("S1a_under", SHORT_SYS, 40, 1, False, "PASS (ctx well under 128)"),
        ("S1a_over", SHORT_SYS, 200, 1, False, "DIVERGE (ctx far over 128)"),
        # S1b — cross 128 via PROMPT length, generation fixed short
        ("S1b_under", SHORT_SYS, 20, 1, False, "PASS (ctx under 128)"),
        ("S1b_over", LONG_SYS, 20, 1, False, "DIVERGE (long system pushes ctx over 128)"),
        # S2 — pool at a fixed over-window config (candidate 2 / D14)
        ("S2_pool_on", SHORT_SYS, 200, 128, True, "isolates the prefill pool"),
        # S3 — ubatch != sliding_window, breaking the 128/128 confound
        ("S3_ub64", SHORT_SYS, 200, 64, True, "ubatch 64 vs window 128"),
    ]
    rows, all_legs = [], []
    for name, sysmsg, ngen, ub, pool, expect in plan:
        t0 = time.time()
        v, d, legs, f = pair(name, sysmsg, ngen, ub, pool)
        all_legs += legs
        rows.append((name, ngen, ub, pool, v, d, f, expect))
        print(f"  {name:12} ngen={ngen:<4} ub={ub:<4} pool={int(pool)} -> {v}  "
              f"(ctx_t1={f['ctx_turn1']} rendered={f['rendered_reused']} "
              f"reused={f['reused']}) {time.time()-t0:.0f}s", flush=True)

    L = [f"RC2 — bisect to the minimal failing reproducer ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail at start {a:.2f} GB | SLOTS=16 greedy | sliding_window={SWA} | CLEAN",
         "LOCALIZE ONLY. Unit = (reused-prefix turn 2) vs (fresh single-shot of the",
         "IDENTICAL rendering). DIFFER = the bug reproduces at that config.", "",
         void_banner(all_legs), ""]
    L.append(f"  {'leg':12} {'ngen':>5} {'ub':>4} {'pool':>5} {'ctx_t1':>7} {'rend':>5} "
             f"{'reused':>7} {'crosses':>8}  verdict")
    for name, ngen, ub, pool, v, d, f, _ in rows:
        L.append(f"  {name:12} {ngen:>5} {ub:>4} {int(pool):>5} {f['ctx_turn1']:>7} "
                 f"{str(f['rendered_reused']):>5} {str(f['reused']):>7} "
                 f"{('YES' if (f['ctx_turn1'] or 0) > SWA else 'no'):>8}  {v}")
    L += ["", "expectations vs measured"]
    for name, _, _, _, v, d, _, expect in rows:
        L.append(f"  {name:12} expected {expect:<48} got {v}")
        if v == "VOID":
            L.append(f"               {d}")

    # verdict logic, stated rather than eyeballed
    def got(n):
        return next(v for nm, _, _, _, v, _, _, _ in rows if nm == n)
    s1a = (got("S1a_under"), got("S1a_over"))
    s1b = (got("S1b_under"), got("S1b_over"))
    L += ["", "=" * 72]
    if s1a == ("EQUAL", "DIFFER") and s1b == ("EQUAL", "DIFFER"):
        L.append("SWA CROSSING CONFIRMED via two independent routes (generation length and")
        L.append("prompt length). The only factor common to both is crossing the 128-token")
        L.append("sliding window. Pool and batch shape are exonerated (both ran ubatch=1,")
        L.append("pool off). Minimal failing reproducer = S1a_over / S1b_over.")
    elif s1a == ("EQUAL", "DIFFER") and s1b != ("EQUAL", "DIFFER"):
        L.append("Divergence tracks GENERATION LENGTH only, not the window: candidate 4")
        L.append("(KV written by decode vs prefill) outranks the SWA hypothesis.")
    elif s1b == ("EQUAL", "DIFFER") and s1a != ("EQUAL", "DIFFER"):
        L.append("Divergence tracks PROMPT LENGTH only: candidate 5 (length/position)")
        L.append("outranks the SWA hypothesis.")
    elif all(v == "EQUAL" for _, _, _, _, v, _, _, _ in rows):
        L.append("NO LEG REPRODUCED. Every dimension individually exonerated. The")
        L.append("divergence requires the full E41b combination — a legitimate outcome,")
        L.append("reported rather than forced into a winner. Next step is to add back")
        L.append("dimensions together rather than shrink further.")
    else:
        L.append("MIXED — no single dimension explains the split; see the table. Not")
        L.append("forcing a winner.")
    L.append("=" * 72)
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
