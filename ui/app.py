# sluice chat - the packaged dev/test UI (`sluice ui`).
#
# Chat over the PERSISTENT engine server (LLMSTREAM_SERVER=1): the model loads
# once and stays warm between messages; the expert cache carries over, and
# multi-turn requests reuse the KV prefix (reused=N) so old turns are never
# re-prefilled. E27 expert-major prefill is on (LLMSTREAM_PREFILL_SLOTS=1).
#
# Safety contract (the host machine is never collateral):
#   - ONE server process ever, shared across browser sessions
#   - a running benchmark blocks chat (one model at a time)
#   - visible Stop button; idle self-exit after 10 min (driver-side);
#     orphan sweep on render; config changes restart the engine cleanly
import atexit
import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "csrc" / "stream_run"
IDLE_EXIT_S = 600

st.set_page_config(page_title="sluice", page_icon=":material/stream:", layout="centered")

MODELS = {
    "gpt-oss-120b": {
        "path": ROOT / "models" / "gpt-oss-120b-MXFP4.gguf",
        "slots": 8, "chat": True, "size": "117B total · 5.1B active · 63.4 GB file",
        "note": "the flagship: 4× this machine's RAM. First message pays a one-time "
                "load (~1–2 min); later turns stay warm.",
    },
    "gpt-oss-20b": {
        "path": ROOT / "models" / "gpt-oss-20b-MXFP4.gguf",
        "slots": 16, "chat": True, "size": "21B total · 3.6B active · 12.1 GB file",
        "note": "the middle tier - much faster, still bigger than RAM comfort.",
    },
    "OLMoE-1B-7B": {
        "path": next((ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").glob("*/*.gguf"), None)
        if (ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").exists() else None,
        "slots": 32, "chat": True, "size": "7B total · 1.3B active · 4.3 GB file",
        "note": "small and snappy (~45-70 tok/s warm) - ideal for UI testing.",
    },
}

# Speed↔quality dial, calibrated by the measured battery - not raw knobs.
MODES = {
    "Exact": {"margin": 0.0, "desc": "bit-identical to the resident model, hash-gated"},
    "Balanced": {"margin": 0.25, "desc": "5-domain NLL battery: no measurable change · fidelity ≥ .91"},
    "Fast": {"margin": 1.25, "desc": "~3× decode speed · measured, documented quality cost"},
}

SUGGESTIONS = {
    ":blue[:material/lightbulb:] Explain something": "Explain why the sky is blue in two sentences.",
    ":green[:material/code:] Write code": "Write a Python function that checks whether a number is prime.",
    ":violet[:material/psychology:] Reason": "A train leaves at 9am at 40 mph; another at 11am at 60 mph on a parallel track. When does the second catch the first?",
}


@st.cache_resource
def _server_slot():
    # one engine per machine, shared by every browser session
    return {"proc": None, "key": None, "started": 0.0}


def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.stdin.write(b"exit\n")
        proc.stdin.flush()
        proc.wait(timeout=3)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


atexit.register(lambda: _terminate(_server_slot()["proc"]))


def cfg_key(cfg, sys_prompt, n_gen):
    return (str(cfg["path"]), cfg["slots"], cfg["margin"], cfg["backend"], sys_prompt, n_gen)


def ensure_server(cfg, sys_prompt, n_gen, status):
    slot = _server_slot()
    key = cfg_key(cfg, sys_prompt, n_gen)
    if slot["proc"] is not None and slot["proc"].poll() is None and slot["key"] == key:
        return slot["proc"]
    _terminate(slot["proc"])
    # one model process on this machine, EVER: a running benchmark (any
    # stream_run that is not our SERVER_SENTINEL) blocks chat - a second
    # 117B engine took the host to swap once already (2026-07-19)
    others = subprocess.run(["pgrep", "-f", "stream_run"], capture_output=True, text=True)
    if others.returncode == 0:
        for pid in others.stdout.split():
            cmdline = subprocess.run(["ps", "-p", pid, "-o", "command="],
                                     capture_output=True, text=True).stdout
            if "SERVER_SENTINEL" not in cmdline and "stream_run" in cmdline:
                raise RuntimeError(
                    "A benchmark run is using the engine right now - chat is "
                    "blocked until it finishes (one model at a time, "
                    "machine-safety rule).")
    subprocess.run(["pkill", "-f", "stream_run.*SERVER_SENTINEL"], capture_output=True)

    env = os.environ.copy()
    env.update({
        "LLMSTREAM_SLOTS": str(cfg["slots"]),
        "LLMSTREAM_MARGIN": str(cfg["margin"]),
        "LLMSTREAM_STREAM_OUT": "1",
        "LLMSTREAM_SERVER": "1",
        "LLMSTREAM_IDLE_EXIT": str(IDLE_EXIT_S),
        "LLMSTREAM_CHAT": "1",
        "LLMSTREAM_PREFILL_SLOTS": "1",  # E27: expert-major prefill (8× TTFT)
    })
    if sys_prompt.strip():
        env["LLMSTREAM_SYSTEM"] = sys_prompt.strip()
    if cfg["backend"] == "gpu":
        env["LLMSTREAM_SLOT_DEV"] = "gpu"
        env["LLMSTREAM_NGL"] = "99"
        env.pop("LLMSTREAM_PREFILL_SLOTS", None)  # pf pool is CPU-only (v1)
    status.update(label="Loading model - one-time, stays warm after this…", state="running")
    proc = subprocess.Popen(
        [str(ENGINE), str(cfg["path"]), str(n_gen), "SERVER_SENTINEL", "128"],
        env=env, cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.set_blocking(proc.stdout.fileno(), False)
    os.set_blocking(proc.stderr.fileno(), False)
    buf = b""
    t0 = time.time()
    while b"<<<READY>>>" not in buf:
        chunk = proc.stdout.read(4096)
        if chunk:
            buf += chunk
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace")[-1500:]
            raise RuntimeError(f"Engine died during load:\n{err}")
        if time.time() - t0 > 600:
            _terminate(proc)
            raise RuntimeError("Engine load timed out")
        time.sleep(0.1)
    slot.update(proc=proc, key=key, started=time.time())
    return proc


METRIC_PATTERNS = {
    "prefill": r"prefill:\s+([\d.]+) s \(([\d.]+) tok/s\)",
    "decode": r"decode:\s+([\d.]+) s \(([\d.]+) tok/s\)",
    "hit": r"hit ([\d.]+)",
    "agreement": r"router_agreement=([\d.]+)",
    "mem": r"peak_rss=([\d.]+) GB phys_footprint=([\d.]+) GB",
    "generated": r"generated=(\d+)",
    "reused": r"reused=(\d+)",
    "therm": r"therm: (\w+)",
}


def parse_metrics(tail):
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = re.search(pat, tail)
        if m:
            out[k] = m.groups()
    return out


def split_harmony(raw):
    """gpt-oss harmony format: analysis channel = thinking, final = answer."""
    m = re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S)
    if m:
        final = re.sub(r"<\|[^|]*\|>", "", m.group(1))
        thinking = re.sub(r"<\|[^|]*\|>", " ", raw[:m.start()]).strip()
    elif "<|" in raw:
        final, thinking = "", re.sub(r"<\|[^|]*\|>", " ", raw).strip()
    else:
        final, thinking = raw, ""
    return final, thinking


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### :material/water: sluice")
    st.caption("Virtual memory for LLMs - models bigger than your RAM, with quality receipts.")

    available = [k for k, v in MODELS.items() if v["path"] and Path(v["path"]).exists()]
    if not available:
        st.error(f"No model files found under {ROOT} - check models/ and hf_home/")
        st.stop()
    choice = st.selectbox("Model", available, help="Only models present on disk are listed")
    cfg = dict(MODELS[choice])
    st.caption(f":material/database: {cfg['size']}")
    st.caption(cfg["note"])

    mode = st.segmented_control("Mode", list(MODES.keys()), default="Balanced",
                                help="The speed↔quality dial, in battery-calibrated steps")
    mode = mode or "Balanced"
    cfg["margin"] = MODES[mode]["margin"]
    st.caption(f":material/verified: {MODES[mode]['desc']}")

    with st.expander("Advanced", icon=":material/tune:"):
        cfg["margin"] = st.slider("Margin (raw dial; 0 = bit-exact)", 0.0, 2.0,
                                  float(cfg["margin"]), 0.05)
        if cfg["margin"] > 0.5:
            st.warning("Margin > 0.5 is outside the validated quality band - "
                       "the model can derail. 0.25 is the battery default.")
        cfg["slots"] = st.select_slider("Slots/layer (expert cache)",
                                        [4, 8, 12, 16, 32, 48], value=cfg["slots"],
                                        help="More slots = more RAM, higher hit rate. "
                                             "12 is the measured 16GB-Mac ceiling for 120B.")
        cfg["backend"] = st.segmented_control("Backend", ["cpu", "gpu"], default="cpu",
                                              help="CPU loads much faster and decodes the "
                                                   "same below ~0.9 hit; GPU pays at high hit rates") or "cpu"
        n_gen = st.number_input("Max new tokens", min_value=16, max_value=1024, value=256, step=16)
        sys_prompt = st.text_area(
            "System prompt",
            value="You are a helpful, concise assistant. Answer the user's message directly. Reasoning: low",
            help="'Reasoning: low' keeps gpt-oss from very long thinking")

    st.space("small")

    # engine card
    slot = _server_slot()
    alive = slot["proc"] is not None and slot["proc"].poll() is None
    with st.container(border=True):
        if alive:
            st.badge("Engine warm", icon=":material/mode_heat:", color="green")
            try:
                rss = psutil.Process(slot["proc"].pid).memory_info().rss / 1e9
                st.caption(f"pid {slot['proc'].pid} · RSS {rss:.1f} GB · "
                           f"warm {int(time.time() - slot['started'])}s")
            except psutil.NoSuchProcess:
                pass
            if st.button("Stop engine", icon=":material/stop_circle:", type="primary",
                         width="stretch"):
                _terminate(slot["proc"])
                slot.update(proc=None, key=None)
                st.rerun()
        else:
            stray = subprocess.run(["pgrep", "-f", "SERVER_SENTINEL"], capture_output=True)
            if stray.returncode == 0:
                subprocess.run(["pkill", "-9", "-f", "SERVER_SENTINEL"], capture_output=True)
                st.warning("Found and killed an orphaned engine from a previous session")
            st.badge("Engine off", icon=":material/power_settings_new:", color="gray")
            st.caption("Starts on your first message")
        st.caption(f":material/timer: auto-stops after {IDLE_EXIT_S // 60} min idle "
                   "(driver-side - survives a UI crash)")

    # live machine stats: fragment reruns itself, not the whole app
    @st.fragment(run_every=2)
    def live_stats():
        vm = psutil.virtual_memory()
        c1, c2 = st.columns(2)
        c1.metric("System CPU", f"{psutil.cpu_percent(interval=None):.0f}%")
        c2.metric("RAM free", f"{vm.available / 1e9:.1f} GB")
        s = _server_slot()
        if s["proc"] is not None and s["proc"].poll() is None:
            try:
                p = psutil.Process(s["proc"].pid)
                with p.oneshot():
                    st.caption(f":material/memory: engine {p.cpu_percent(interval=None):.0f}% CPU · "
                               f"RSS {p.memory_info().rss / 1e9:.2f} GB")
            except psutil.NoSuchProcess:
                pass

    with st.container(border=True):
        live_stats()

    if st.button("Clear chat", icon=":material/mop:", width="stretch"):
        st.session_state.chat_log = []
        st.rerun()

# ---------------- chat ----------------
st.title("Chat", anchor=False)
st.caption(f"{choice} · {mode} mode · streaming from a {cfg['size'].split('·')[-1].strip()} "
           "file through the expert cache")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        if turn.get("thinking"):
            with st.expander("Thinking", icon=":material/psychology:"):
                st.text(turn["thinking"])
        st.markdown(turn["text"])
        if turn.get("timing"):
            st.caption(turn["timing"])

prompt = None
if not st.session_state.chat_log:
    st.space("medium")
    st.markdown(":material/stream: **Ask anything** - the model streams its experts "
                "from disk as it thinks.", text_alignment="center")
    picked = st.pills("Try:", list(SUGGESTIONS.keys()), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]

typed = st.chat_input(f"Message {choice}…", submit_mode="disable")
prompt = typed or prompt

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_log.append({"role": "user", "text": prompt})

    with st.chat_message("assistant"):
        status = st.status("Starting…", expanded=False)
        think_ph = st.empty()
        answer_ph = st.empty()
        foot_ph = st.empty()
        try:
            proc = ensure_server(cfg, sys_prompt, n_gen, status)
        except RuntimeError as e:
            status.update(label="Engine failed", state="error")
            st.error(str(e))
            st.stop()
        eng_ps = psutil.Process(proc.pid)

        t0 = time.time()

        def clean(t):
            return t.replace("\x1e", " ").replace("\x1f", " ").replace("\n", "\\n")

        turns = ["%s\x1f%s" % (t["role"], clean(t["text"]))
                 for t in st.session_state.chat_log]
        proc.stdin.write("\x1e".join(turns).encode() + b"\n")
        proc.stdin.flush()
        status.update(label="Prefilling prompt (expert-major)…", state="running")

        buf = b""
        stderr_tail = ""
        streaming = False
        done = False
        ttft = None
        raw = ""
        guard_seen = []
        while not done:
            chunk = proc.stdout.read(4096)
            echunk = proc.stderr.read(4096)
            if echunk:
                stderr_tail = (stderr_tail + echunk.decode(errors="replace"))[-4000:]
                guard_seen = [l for l in stderr_tail.splitlines()
                              if "pressure" in l or "guard" in l][-3:]
            if chunk:
                buf += chunk
                if not streaming and b"<<<STREAM>>>\n" in buf:
                    streaming = True
                    status.update(label="Generating…", state="running")
                    buf = buf.split(b"<<<STREAM>>>\n", 1)[1]
                if streaming:
                    end = buf.find(b"<<<END>>>")
                    raw = (buf[:end] if end >= 0 else buf).decode(errors="replace")
                    if ttft is None and raw.strip():
                        ttft = time.time() - t0
                    final, thinking = split_harmony(raw)
                    if thinking:
                        with think_ph.container():
                            with st.expander("Thinking", icon=":material/psychology:",
                                             expanded=(not final)):
                                st.text(thinking)
                    answer_ph.markdown((final + ("▌" if end < 0 else ""))
                                       if (final or not thinking) else "")
                if b"<<<READY>>>" in buf:
                    done = True
            if proc.poll() is not None:
                done = True
            if not chunk and not echunk:
                time.sleep(0.05)

        total = time.time() - t0
        tail = buf.decode(errors="replace")
        met = parse_metrics(tail)
        status.update(label="Done - engine stays warm", state="complete")

        timing_bits = [f"total {total:.1f}s"]
        if ttft:
            timing_bits.append(f"first token {ttft:.1f}s")
        if "decode" in met:
            timing_bits.append(f"{met['decode'][1]} tok/s")
        if "reused" in met and int(met["reused"][0]) > 0:
            timing_bits.append(f"reused {met['reused'][0]} ctx tokens")
        timing = " · ".join(timing_bits)

        with foot_ph.container():
            cap_col, pop_col = st.columns([5, 1])
            cap_col.caption(timing)
            with pop_col.popover("physics", icon=":material/query_stats:"):
                st.markdown("**Run physics**")
                if "prefill" in met:
                    st.caption(f"prefill {met['prefill'][1]} tok/s ({met['prefill'][0]}s)")
                if "decode" in met:
                    st.caption(f"decode {met['decode'][1]} tok/s")
                if "hit" in met:
                    st.caption(f"expert-cache hit {float(met['hit'][0]) * 100:.0f}%")
                if "agreement" in met:
                    st.caption(f"routing fidelity {float(met['agreement'][0]) * 100:.1f}%")
                if "mem" in met:
                    st.caption(f"peak RSS {met['mem'][0]} GB · footprint {met['mem'][1]} GB")
                if "therm" in met:
                    st.caption(f"thermal: {met['therm'][0]}")
                if guard_seen:
                    st.warning("memory guard was active:\n" + "\n".join(guard_seen))

        final_clean, _ = split_harmony(raw)
        final_clean = re.sub(r"<\|[^|]*\|>", "", final_clean).strip()
        a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
        if met.get("generated") == ("0",) or not (final_clean or (a and a.group(1).strip())):
            final_clean = final_clean or ("*(model produced no tokens - try rephrasing "
                                          "or adjusting the system prompt)*")
        st.session_state.chat_log.append(
            {"role": "assistant", "text": final_clean,
             "thinking": a.group(1).strip() if a else "", "timing": timing})
        st.rerun()
