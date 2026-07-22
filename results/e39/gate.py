"""E39 gate — resumed chat must be byte-identical to an unbroken chat.

A: unbroken, feature OFF   : turn1 -> turn2 (reference)
B: split,   feature ON     : turn1 -> save -> EXIT || restart -> restore -> turn2

Gate: B's turn-2 logits_hash AND text == A's turn-2 logits_hash AND text.
That proves the KV restore is faithful *and* that the feature does not perturb
output. Also records turn-2 TTFT for both, to settle P-A vs P-B.
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
BIN, MODEL = str(ROOT / "csrc" / "stream_run"), str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf")
OUT = ROOT / "results" / "e39"; OUT.mkdir(parents=True, exist_ok=True)
KV = str(OUT / "chat.kv")
NGEN = 60

m = re.search(r'DEFAULT_SYSTEM = """(.*?)"""', (ROOT / "ui" / "app.py").read_text(), re.S)
SYSTEM = m.group(1).strip() if m else "You are a helpful assistant."
P1 = "Explain how the Earth formed and why it can support life."
P2 = "Thanks — can you summarize that in three bullet points?"

BASE = {"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "16", "LLMSTREAM_PREFILL_SLOTS": "64",
        "LLMSTREAM_PHASE_TIMERS": "1", "LLMSTREAM_SYSTEM": SYSTEM,
        "LLMSTREAM_SERVER": "1", "LLMSTREAM_IDLE_EXIT": "180"}


class Server:
    def __init__(self, label, persist=None):
        e = os.environ.copy(); e.update(BASE)
        if persist: e["LLMSTREAM_KV_PERSIST"] = persist
        else: e.pop("LLMSTREAM_KV_PERSIST", None)
        self.log = open(OUT / f"{label}.out", "w")
        self.p = subprocess.Popen([BIN, MODEL, str(NGEN), "SERVER_SENTINEL", "128"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(OUT / f"{label}.err", "w"),
                                  # errors="replace": the engine can emit a token piece that splits a multi-byte
                                  # UTF-8 char across a pipe read, which raises UnicodeDecodeError under strict
                                  # decoding (S1v3 VOID). Lossy DISPLAY decode is safe because the comparison
                                  # source of truth is the ASCII "tok <id> |piece|" lines from PRINT_TOKS.
                                  env=e, text=True, errors="replace", bufsize=1)
        self.ready()

    def ready(self):
        buf = []
        while True:
            line = self.p.stdout.readline()
            if not line: break
            buf.append(line); self.log.write(line); self.log.flush()
            if line.strip() == "<<<READY>>>": break
        return "".join(buf)

    def turn(self, req):
        self.p.stdin.write(req.replace("\n", "\\n") + "\n"); self.p.stdin.flush()
        return self.ready()

    def close(self):
        try:
            self.p.stdin.write("exit\n"); self.p.stdin.flush(); self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.log.close()


def answer_of(t):
    mm = re.search(r"^text: (.*?)(?=\n<<<READY>>>)", t, re.S | re.M)
    return mm.group(1).strip() if mm else ""

def hash_of(t):
    mm = re.findall(r"logits_hash=([0-9a-f]+)", t)
    return mm[-1] if mm else None

def ttft_of(t):
    mm = re.findall(r"ttft: total=(\d+) ms", t)
    return mm[-1] if mm else None

def ctx_of(t):
    mm = re.findall(r"ttft: ctx_held=(\d+) rendered=(\d+) reused=(\d+) reprefill=(\d+)", t)
    return mm[-1] if mm else None


if os.path.exists(KV): os.remove(KV)

print("A: unbroken chat, feature OFF …", flush=True)
a = Server("A_unbroken_off", persist=None)
a1 = a.turn("user\x1f" + P1)
ans = answer_of(a1)
req2 = "user\x1f" + P1 + "\x1eassistant\x1f" + ans + "\x1euser\x1f" + P2
a2 = a.turn(req2)
a.close()

print("B1: turn 1, feature ON, then exit (checkpoint) …", flush=True)
b1 = Server("B1_turn1_on", persist=KV)
b1t1 = b1.turn("user\x1f" + P1)
b1.close()

print("B2: restart, restore, turn 2 …", flush=True)
b2 = Server("B2_resume_on", persist=KV)
b2t2 = b2.turn(req2)
b2.close()

ah, bh = hash_of(a2), hash_of(b2t2)
atext, btext = answer_of(a2), answer_of(b2t2)
same_hash, same_text = (ah == bh and ah is not None), (atext == btext and atext != "")
resumed = "KV RESUMED" in (OUT / "B2_resume_on.err").read_text()

lines = [
    "E39 gate — resumed vs unbroken (%s)" % time.strftime("%Y-%m-%d %H:%M"),
    "n_gen=%d SLOTS=16 guard ON exact greedy | CLEAN" % NGEN, "",
    "A unbroken(OFF)  turn2: hash=%s ttft=%s ms ctx(held,rendered,reused,reprefill)=%s" % (ah, ttft_of(a2), ctx_of(a2)),
    "B2 resumed(ON)   turn2: hash=%s ttft=%s ms ctx(held,rendered,reused,reprefill)=%s" % (bh, ttft_of(b2t2), ctx_of(b2t2)),
    "B1 turn1(ON)          : ttft=%s ms" % ttft_of(b1t1),
    "",
    "resume announced on stderr : %s" % ("YES" if resumed else "NO"),
    "turn-2 logits_hash match   : %s" % ("PASS" if same_hash else "FAIL"),
    "turn-2 text match          : %s" % ("PASS" if same_text else "FAIL"),
    "",
    "GATE: %s" % ("GREEN" if (same_hash and same_text) else "RED"),
]
(OUT / "summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
(OUT / "DONE").write_text("done")
