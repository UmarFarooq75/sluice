"""E38 — first-token decomposition, 3 legs.

L1  baseline single-turn (short prompt, no reuse)      -> phase floor
L2  server 2-turn chat (the 47.6 s anomaly)            -> divergence evidence
L3  single-turn control, prompt length ~= L2 turn-2    -> isolates re-prefill cost

Runs legs sequentially (one model process at a time). Artifacts -> results/e38/.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
BIN = str(ROOT / "csrc" / "stream_run")
MODEL = str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e38"
OUT.mkdir(parents=True, exist_ok=True)
NGEN = 200

# reuse the UI's real system prompt so the reused-prefix length is comparable
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


def one_shot(label, prompt, ngen=NGEN):
    """Single-turn (non-server) run."""
    log = OUT / f"{label}.out"
    with open(log, "w") as f:
        subprocess.run([BIN, MODEL, str(ngen), prompt, "128"],
                       stdout=f, stderr=open(OUT / f"{label}.err", "w"), env=env_for())
    return log.read_text()


def server_two_turn(label):
    """Server mode: turn 1, then turn 2 carrying the re-rendered history."""
    log = open(OUT / f"{label}.out", "w")
    p = subprocess.Popen([BIN, MODEL, str(NGEN), "SERVER_SENTINEL", "128"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(OUT / f"{label}.err", "w"),
                         env=env_for({"LLMSTREAM_SERVER": "1", "LLMSTREAM_IDLE_EXIT": "180"}),
                         # errors="replace": the engine can emit a token piece that splits a multi-byte
                         # UTF-8 char across a pipe read, which raises UnicodeDecodeError under strict
                         # decoding (S1v3 VOID). Lossy DISPLAY decode is safe because the comparison
                         # source of truth is the ASCII "tok <id> |piece|" lines from PRINT_TOKS.
                         text=True, errors="replace", bufsize=1)

    def read_until_ready():
        buf = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            buf.append(line)
            log.write(line); log.flush()
            if line.strip() == "<<<READY>>>":
                break
        return "".join(buf)

    def send(req):
        p.stdin.write(req.replace("\n", "\\n") + "\n")
        p.stdin.flush()

    read_until_ready()                      # load-done
    send("user\x1f" + P1)
    t1 = read_until_ready()
    # the assistant text the UI would put back into history (specials stripped)
    mt = re.search(r"^text: (.*?)(?=\n<<<READY>>>)", t1, re.S | re.M)
    answer1 = mt.group(1).strip() if mt else ""

    send("user\x1f" + P1 + "\x1eassistant\x1f" + answer1 + "\x1euser\x1f" + P2)
    t2 = read_until_ready()

    try:
        send("exit")
        p.wait(timeout=30)
    except Exception:
        p.kill()
    log.close()
    return t1, t2, answer1


def phases(text):
    out = {}
    for key, pat in [
        ("ttft", r"ttft: total=(\d+) ms \| template=(\d+) tokenize=(\d+) kv_match=(\d+) prefill=(\d+) first_sample=(\d+)"),
        ("ctx", r"ttft: ctx_held=(\d+) rendered=(\d+) reused=(\d+) reprefill=(\d+) \(([\d.]+)%"),
        ("div", r"ttft: diverged_at=(\d+) kv_had=(-?\d+)\|(.*?)\| rerender_has=(-?\d+)\|(.*?)\|"),
    ]:
        mm = re.findall(pat, text)
        if mm:
            out[key] = mm[-1]
    mm = re.findall(r"^prefill: ([\d.]+) s \(([\d.]+) tok/s\)", text, re.M)
    if mm:
        out["prefill"] = mm[-1]
    mm = re.findall(r"^mode=\w+ prompt_toks=(\d+) reused=(\d+)", text, re.M)
    if mm:
        out["mode"] = mm[-1]
    return out


if __name__ == "__main__":
    report = []

    print("L1: baseline single-turn…", flush=True)
    l1 = one_shot("L1_baseline", P1)
    report.append(("L1 baseline (short prompt, no reuse)", phases(l1)))

    print("L2: server 2-turn (anomaly)…", flush=True)
    t1, t2, answer1 = server_two_turn("L2_twoturn")
    report.append(("L2 turn-1 (cold, short prompt)", phases(t1)))
    report.append(("L2 turn-2 (THE ANOMALY: re-rendered history)", phases(t2)))
    (OUT / "L2_answer1.txt").write_text(answer1)

    # L3: single-turn whose rendered length ~= L2 turn-2, to isolate re-prefill cost
    print("L3: length-matched control…", flush=True)
    l3 = one_shot("L3_control", P1 + "\n\n" + answer1 + "\n\n" + P2)
    report.append(("L3 control (same rendered length, no reuse path)", phases(l3)))

    lines = ["E38 — first-token decomposition (%s)" % time.strftime("%Y-%m-%d %H:%M"),
             "system prompt: %d chars | n_gen=%d | SLOTS=16 guard ON | CLEAN" % (len(SYSTEM), NGEN), ""]
    for name, ph in report:
        lines.append(f"## {name}")
        if "ttft" in ph:
            t = ph["ttft"]
            lines.append(f"   ttft total={t[0]} ms | template={t[1]} tokenize={t[2]} kv_match={t[3]} prefill={t[4]} first_sample={t[5]}")
        if "ctx" in ph:
            c = ph["ctx"]
            lines.append(f"   ctx_held={c[0]} rendered={c[1]} reused={c[2]} reprefill={c[3]} ({c[4]}% of rendered)")
        if "div" in ph:
            d = ph["div"]
            lines.append(f"   diverged_at={d[0]}  kv_had={d[1]}|{d[2]}|  rerender_has={d[3]}|{d[4]}|")
        if "prefill" in ph:
            lines.append(f"   prefill={ph['prefill'][0]} s ({ph['prefill'][1]} tok/s)")
        lines.append("")
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    (OUT / "DONE").write_text("done")
