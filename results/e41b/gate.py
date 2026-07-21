"""E41b — KV canonicalization at end of turn. 3 legs, one process at a time.

A  canon ON,  2-turn chat   -> turn-2 reused/reprefill/ttft + logits_hash
B  canon OFF, 2-turn chat   -> CONTROL: identical rendering, stock partial-prefill path
C  fresh single-shot        -> turn-2's rendering with reused=0 (the reference hash)

Gate 1 (faithfulness): hash(A.turn2) == hash(C).
Control: hash(B.turn2) == hash(C)?  If B also differs, the divergence is intrinsic
to partial prefill (batch-shape FP), predates this feature, and is the E39 disease.

A runs first: its `canon_reply:` line is the assistant text B and C must echo, so
all three legs render byte-identically and only the KV path differs.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e41b"
OUT.mkdir(parents=True, exist_ok=True)
NGEN = 200  # same as E38, so turn-2 TTFT is comparable to its 20.5 s

m = re.search(r'DEFAULT_SYSTEM = """(.*?)"""', (ROOT / "ui" / "app.py").read_text(), re.S)
SYSTEM = m.group(1).strip() if m else "You are a helpful assistant."

P1 = "Explain how the Earth formed and why it can support life."
P2 = "Thanks — can you summarize that in three bullet points?"

BASE = {
    "LLMSTREAM_CHAT": "1",
    "LLMSTREAM_SLOTS": "16",
    "LLMSTREAM_PREFILL_SLOTS": "64",
    "LLMSTREAM_PHASE_TIMERS": "1",
    "LLMSTREAM_SYSTEM": SYSTEM,
}


def env_for(extra=None):
    e = os.environ.copy()
    e.update(BASE)
    if extra:
        e.update(extra)
    return e


def avail_gb():
    """Same four buckets and the same page size as run.sh's awk gate — a second
    opinion here is only useful if it uses the same yardstick, or the launcher
    passes its gate and this aborts anyway."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    p = 0
    for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        mm = re.search(rf"{k}:\s+(\d+)", out)
        if mm:
            p += int(mm.group(1))
    return p * ps / 1e9


def hist(turns):
    """turns = [(role, content), ...] -> the engine's history wire format."""
    return "\x1e".join(f"{r}\x1f{c}" for r, c in turns)


def server(label, requests, canon):
    """Run a server-mode session; `requests` is a list of wire-format prompts.

    Returns the per-turn stdout blocks. `requests` may contain callables, which
    are given the parsed turns so far and return the next wire prompt.
    """
    errf = open(OUT / f"{label}.err", "w")
    extra = {"LLMSTREAM_SERVER": "1", "LLMSTREAM_IDLE_EXIT": "300"}
    if canon:
        extra["LLMSTREAM_KV_CANON"] = "1"
    p = subprocess.Popen([BIN, MODEL, str(NGEN), "SERVER_SENTINEL", "128"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=errf, env=env_for(extra), text=True, bufsize=1)

    def until_ready():
        buf = []
        for line in p.stdout:
            if line.startswith("<<<READY>>>"):
                return "".join(buf)
            buf.append(line)
        return "".join(buf)

    until_ready()  # load
    blocks = []
    for req in requests:
        wire = req(blocks) if callable(req) else req
        p.stdin.write(wire.replace("\n", "\\n") + "\n")
        p.stdin.flush()
        blocks.append(until_ready())
    p.stdin.close()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    (OUT / f"{label}.out").write_text("\n<<<TURN>>>\n".join(blocks))
    return blocks


def field(block, pat, cast=str, default=None):
    mm = re.search(pat, block)
    return cast(mm.group(1)) if mm else default


def unesc(s):
    return s.replace("\\n", "\n").replace("\\\\", "\\")


def main():
    a = avail_gb()
    print(f"avail = {a:.2f} GB")
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB gate")
    if subprocess.run(["pgrep", "-f", "stream_run"], capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    t0 = time.time()
    # --- leg A: canon ON -------------------------------------------------
    A = server("A_canon_on", [
        hist([("user", P1)]),
        lambda b: hist([("user", P1),
                        ("assistant", unesc(field(b[0], r"^canon_reply: (.*)$", default=""))),
                        ("user", P2)]),
    ], canon=True)
    reply = unesc(field(A[0], r"(?m)^canon_reply: (.*)$", default=""))
    if not reply:
        sys.exit("ABORT: leg A produced no canon_reply — canonicalization did not run")
    (OUT / "canon_reply.txt").write_text(reply)

    turn2 = hist([("user", P1), ("assistant", reply), ("user", P2)])

    # --- leg B: canon OFF, identical rendering ---------------------------
    B = server("B_canon_off", [hist([("user", P1)]), turn2], canon=False)

    # --- leg C: fresh full re-prefill of turn-2's rendering ---------------
    C = server("C_fresh", [turn2], canon=False)

    def row(name, blk):
        return dict(
            leg=name,
            hash=field(blk, r"logits_hash=(\w+)"),
            reused=field(blk, r"reused=(\d+)", int),
            rendered=field(blk, r"rendered=(\d+)", int),
            reprefill=field(blk, r"reprefill=(\d+)", int),
            ttft_ms=field(blk, r"ttft: total=(\d+)", int),
            prefill_s=field(blk, r"prefill: ([\d.]+) s", float),
        )

    rA, rB, rC = row("A turn2 (canon ON)", A[1]), row("B turn2 (canon OFF)", B[1]), row("C fresh", C[0])
    canon_line = field(A[0], r"(?m)^(canon: .*)$", default="(none)")

    L = []
    L.append(f"E41b — KV canonicalization ({time.strftime('%Y-%m-%d %H:%M')})")
    L.append(f"avail at start {a:.2f} GB | NGEN={NGEN} SLOTS=16 guard ON | CLEAN")
    L.append(f"total wall {time.time() - t0:.0f} s")
    L.append("")
    L.append(f"turn-1 canonicalization: {canon_line}")
    L.append("")
    for r in (rA, rB, rC):
        L.append(f"{r['leg']:<22} hash={r['hash']} rendered={r['rendered']} "
                 f"reused={r['reused']} reprefill={r['reprefill']} "
                 f"ttft={r['ttft_ms']} ms prefill={r['prefill_s']} s")
    L.append("")
    L.append(f"GATE 1 faithfulness  A == C : {'GREEN' if rA['hash'] == rC['hash'] else 'RED'}")
    L.append(f"CONTROL              B == C : {'match' if rB['hash'] == rC['hash'] else 'DIFFER'}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
