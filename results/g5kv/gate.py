"""G5-1 gate — KV persistence revisit under the honest contract (pre-reg: docs/lablog.md).

E39 re-run, widened to the pre-registered scope:
  gate 1  resumed turn-2 TOKEN IDS == unbroken turn-2 token ids, 3 prompt pairs. Hard RED.
  gate 2  matched-shape hash probe: EQUAL/DIFFER recorded per pair. NOT a gate —
          it decides the claim wording (byte-identical vs same-text-config-pinned).
  gate 3  resume announces loudly on stderr, with size+latency instrumentation.
  gate 4  runs in run.sh via scripts/gate.sh (bit-exact + byte-identical-off).
  gate 5  lifecycle: corrupt-file rejection, truncated-file rejection, atomic write
          (no .tmp survivor), bounded size on disk.

Config is pinned to E39's exactly (SLOTS=16, PREFILL_SLOTS=64, n_gen=60, n_ubatch=128)
so the two experiments compare row-for-row. The model is parameterized (G5KV_MODEL):
nothing here is gpt-oss-specific except the default path — the same gate must hold for
any GGUF MoE family sluice runs.

Source of truth for gate 1 is PRINT_TOKS token ids, not decoded text (S1v3: display
text can lose bytes at pipe boundaries; ids cannot).
"""
import os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import Leg, compare, void_banner, verdict_line  # noqa: E402

BIN = str(ROOT / "csrc" / "stream_run")
MODEL = os.environ.get("G5KV_MODEL", str(ROOT / "models" / "gpt-oss-20b-MXFP4.gguf"))
OUT = ROOT / "results" / "g5kv"; OUT.mkdir(parents=True, exist_ok=True)
KV = str(OUT / "chat.kv")
NGEN = 60

m = re.search(r'DEFAULT_SYSTEM = """(.*?)"""', (ROOT / "ui" / "app.py").read_text(), re.S)
SYSTEM = m.group(1).strip() if m else "You are a helpful assistant."

# three pairs, three registers: prose, code, reasoning
PAIRS = [
    ("Explain how the Earth formed and why it can support life.",
     "Thanks — can you summarize that in three bullet points?"),
    ("Write a Python function that merges two sorted lists into one sorted list without using sort().",
     "What is the time complexity of that function, briefly?"),
    ("What is the difference between TCP and UDP?",
     "Which one would you pick for a video call, in one sentence?"),
]

BASE = {"LLMSTREAM_CHAT": "1", "LLMSTREAM_SLOTS": "16", "LLMSTREAM_PREFILL_SLOTS": "64",
        "LLMSTREAM_PHASE_TIMERS": "1", "LLMSTREAM_SYSTEM": SYSTEM,
        "LLMSTREAM_SERVER": "1", "LLMSTREAM_IDLE_EXIT": "180",
        "LLMSTREAM_PRINT_TOKS": "1"}


class Server:
    def __init__(self, label, persist=None):
        e = os.environ.copy(); e.update(BASE)
        if persist: e["LLMSTREAM_KV_PERSIST"] = persist
        else: e.pop("LLMSTREAM_KV_PERSIST", None)
        self.label = label
        self.log = open(OUT / f"{label}.out", "w")
        # errors="replace": engine token pieces can split multi-byte UTF-8 across a
        # pipe read (S1v3 VOID). Ids from the ASCII tok lines are the comparison
        # source of truth, so lossy display decode is safe.
        self.p = subprocess.Popen([BIN, MODEL, str(NGEN), "SERVER_SENTINEL", "128"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(OUT / f"{label}.err", "w"),
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
        rc = None
        try:
            self.p.stdin.write("exit\n"); self.p.stdin.flush()
            rc = self.p.wait(timeout=120)
        except Exception:
            self.p.kill(); rc = -9
        self.log.close()
        return rc


def answer_of(t):
    mm = re.search(r"^text: (.*?)(?=\n<<<READY>>>)", t, re.S | re.M)
    return mm.group(1).strip() if mm else ""

def toks_of(t):
    return [int(x) for x in re.findall(r"^tok\s+(\d+) ", t, re.M)]

def hash_of(t):
    mm = re.findall(r"logits_hash=([0-9a-f]+)", t)
    return mm[-1] if mm else None

def ctx_of(t):
    mm = re.findall(r"ttft: ctx_held=(\d+) rendered=(\d+) reused=(\d+) reprefill=(\d+)", t)
    return mm[-1] if mm else None

def stderr_of(label):
    return (OUT / f"{label}.err").read_text(errors="replace")


if os.path.exists(KV): os.remove(KV)
legs, rows, probe_rows, tele_rows, life_rows = [], [], [], [], []
hard_red = False

for i, (p1, p2) in enumerate(PAIRS, 1):
    if os.path.exists(KV): os.remove(KV)

    print(f"pair {i} A: unbroken, OFF ...", flush=True)
    a = Server(f"p{i}_A_unbroken_off")
    a1 = a.turn("user\x1f" + p1)
    ans = answer_of(a1)
    if not ans:
        # E41b class: an empty scrape must never silently become an empty assistant
        # turn in req2 — that compares two different prompts. Void the pair loudly.
        a.close()
        legs.append(Leg(f"p{i}_A", rc=0, hash=None, extra={"note": "empty turn-1 answer scrape"}))
        rows.append((f"p{i} turn-2 token ids", "VOID", f"p{i}_A: empty turn-1 answer scrape"))
        continue
    req2 = "user\x1f" + p1 + "\x1eassistant\x1f" + ans + "\x1euser\x1f" + p2
    a2 = a.turn(req2)
    rc_a = a.close()
    A = Leg(f"p{i}_A", rc=rc_a or 0, hash=hash_of(a2), toks=toks_of(a2),
            extra={"text": answer_of(a2), "ctx": ctx_of(a2)})

    print(f"pair {i} B1: turn 1, ON, exit checkpoints ...", flush=True)
    b1 = Server(f"p{i}_B1_turn1_on", persist=KV)
    b1.turn("user\x1f" + p1)
    rc_b1 = b1.close()

    # lifecycle: checkpoint must exist, atomically (no .tmp survivor), bounded
    kv_ok = os.path.exists(KV)
    tmp_left = os.path.exists(KV + ".tmp")
    kv_mb = os.path.getsize(KV) / 1e6 if kv_ok else 0.0
    life_rows.append((f"p{i} checkpoint exists after exit", "PASS" if kv_ok and rc_b1 == 0 else "FAIL"))
    life_rows.append((f"p{i} no .tmp survivor (atomic)", "PASS" if not tmp_left else "FAIL"))
    life_rows.append((f"p{i} checkpoint size {kv_mb:.1f} MB (< 1024 MB bound)",
                      "PASS" if 0 < kv_mb < 1024 else "FAIL"))
    if not kv_ok or tmp_left or not (0 < kv_mb < 1024) or rc_b1 != 0:
        hard_red = True

    print(f"pair {i} B2: restart, resume, turn 2 ...", flush=True)
    b2 = Server(f"p{i}_B2_resume_on", persist=KV)
    b2t2 = b2.turn(req2)
    rc_b2 = b2.close()
    B2 = Leg(f"p{i}_B2", rc=rc_b2 or 0, hash=hash_of(b2t2), toks=toks_of(b2t2),
             extra={"text": answer_of(b2t2), "ctx": ctx_of(b2t2)})
    legs += [A, B2]

    # gate 1: token ids (hard), text recorded alongside
    v_tok, d_tok = compare(A, B2, "toks")
    rows.append((f"p{i} turn-2 token ids", v_tok, d_tok))
    if v_tok == "DIFFER": hard_red = True

    # gate 2 probe: hash at matched shape — recorded, never gated
    v_h, d_h = compare(A, B2, "hash")
    probe_rows.append((f"p{i} hash A_ctx={A.extra['ctx']} B2_ctx={B2.extra['ctx']}", v_h, d_h))

    # gate 3: loud resume + instrumentation present
    err = stderr_of(f"p{i}_B2_resume_on")
    resumed = "KV RESUMED" in err
    instr = re.search(r"KV RESUMED .* \(([\d.]+) MB in ([\d.]+) s\)", err)
    tele_rows.append((f"p{i} resume announced", "PASS" if resumed else "FAIL"))
    tele_rows.append((f"p{i} resume instrumented (MB, s)",
                      f"PASS ({instr.group(1)} MB, {instr.group(2)} s)" if instr else "FAIL"))
    if not resumed or not instr: hard_red = True

# gate 5b: corrupt + truncated files must both fall back to "starting fresh",
# and the server must still take a working turn (fail closed, never half-restore)
def lifecycle_reject(tag, mutate):
    mutate(KV)
    s = Server(f"life_{tag}_on", persist=KV)
    t = s.turn("user\x1fSay the word hello and nothing else.")
    rc = s.close()
    err = stderr_of(f"life_{tag}_on")
    fresh = "starting fresh" in err
    served = bool(toks_of(t)) and (rc or 0) == 0
    life_rows.append((f"{tag} file -> starting fresh", "PASS" if fresh else "FAIL"))
    life_rows.append((f"{tag} file -> turn still served", "PASS" if served else "FAIL"))
    return fresh and served

print("lifecycle: corrupt file ...", flush=True)
ok_corrupt = lifecycle_reject("corrupt", lambda p: Path(p).write_bytes(os.urandom(1 << 20)))
print("lifecycle: truncated file ...", flush=True)
# rebuild a valid checkpoint, then cut it in half
b = Server("life_mk_ckpt_on", persist=KV)
b.turn("user\x1f" + PAIRS[0][0]); b.close()
def _trunc(p):
    data = Path(p).read_bytes(); Path(p).write_bytes(data[:len(data)//2])
ok_trunc = lifecycle_reject("truncated", _trunc)
if not (ok_corrupt and ok_trunc): hard_red = True

probe_verdicts = [v for _, v, _ in probe_rows]
if all(v == "EQUAL" for v in probe_verdicts):
    wording = "hash matched at matched shape in all pairs -> claim MAY say byte-identical (config-pinned)"
elif any(v == "VOID" for v in probe_verdicts):
    wording = "probe incomplete -> claim stays: same text, config-pinned"
else:
    wording = "hash differs on resume (upstream cross-shape variance, RC4) -> claim stays: same text, config-pinned"

incomplete = any(l.void for l in legs)
gate = "INCOMPLETE" if incomplete else ("RED" if hard_red else "GREEN")

lines = [
    "G5-1 gate — KV persistence revisit (%s)" % time.strftime("%Y-%m-%d %H:%M"),
    "model=%s n_gen=%d SLOTS=16 PREFILL_SLOTS=64 n_ubatch=128 exact greedy | E39 config, pinned" % os.path.basename(MODEL),
    void_banner(legs), "",
    "gate 1 — resumed turn-2 == unbroken turn-2, token ids (HARD):",
    *[verdict_line(l, v, d) for l, v, d in rows], "",
    "gate 2 — matched-shape hash probe (recorded, decides wording, NOT a gate):",
    *[verdict_line(l, v, d) for l, v, d in probe_rows],
    "  wording: " + wording, "",
    "gate 3 — telemetry:",
    *["  %-46s %s" % r for r in tele_rows], "",
    "gate 5 — lifecycle:",
    *["  %-46s %s" % r for r in life_rows], "",
    "(gate 4 — bit-exact + byte-identical-off — runs in run.sh via scripts/gate.sh; see gate_full.log)",
    "",
    "GATE: %s" % gate,
]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
# A completed RED is a valid result, not a dead harness: DONE must read "done" for
# both GREEN and RED (e39 precedent). Only an INCOMPLETE run — void legs, no full
# comparison — exits nonzero so sl_finish marks it void.
sys.exit(0 if gate in ("GREEN", "RED") else 1)
