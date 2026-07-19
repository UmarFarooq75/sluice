# Chat page over the PERSISTENT engine server (LLMSTREAM_SERVER=1).
#
# The model loads once and stays warm between messages - no per-message
# reload, and the expert cache carries over (second reply is faster than the
# first). Safety contract, because the host machine is never collateral:
#   - ONE server process ever (shared across all browser sessions)
#   - visible Stop button kills it instantly
#   - the server kills ITSELF after 10 minutes idle (driver-side, works even
#     if this UI dies)
#   - config changes restart it cleanly (old one terminated first)
# Multi-turn: the UI sends the whole conversation each turn; the engine
# reuses the KV prefix shared with the previous request (reused=N in the
# metrics) so old turns are not re-prefilled.
import atexit
import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "csrc" / "stream_run"
IDLE_EXIT_S = 600

st.set_page_config(page_title="llmstream chat", layout="wide")

MODELS = {
    "gpt-oss-120b (117B, MXFP4)": {
        "path": ROOT / "models" / "gpt-oss-120b-MXFP4.gguf",
        "slots": 8, "margin": 0.25, "chat": True,
        "note": "validated default ~1.6 tok/s; first message pays the one-time load (~1-2 min), later turns reuse context",
    },
    "OLMoE-1B-7B (fast)": {
        "path": next((ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").glob("*/*.gguf"), None)
        if (ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").exists() else None,
        "slots": 32, "margin": 0.0, "chat": True,
        "note": "small model: ~45 tok/s warm, snappy for UI testing",
    },
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
    # never two engines: also clear any stray from a crashed session
    subprocess.run(["pkill", "-f", "stream_run.*SERVER_SENTINEL"], capture_output=True)

    env = os.environ.copy()
    env.update({
        "LLMSTREAM_SLOTS": str(cfg["slots"]),
        "LLMSTREAM_MARGIN": str(cfg["margin"]),
        "LLMSTREAM_STREAM_OUT": "1",
        "LLMSTREAM_SERVER": "1",
        "LLMSTREAM_IDLE_EXIT": str(IDLE_EXIT_S),
        "LLMSTREAM_CHAT": "1",
    })
    if sys_prompt.strip():
        env["LLMSTREAM_SYSTEM"] = sys_prompt.strip()
    if cfg["backend"] == "gpu":
        env["LLMSTREAM_SLOT_DEV"] = "gpu"
        env["LLMSTREAM_NGL"] = "99"
    status.update(label="loading model (one-time; stays warm after this)…", state="running")
    proc = subprocess.Popen(
        [str(ENGINE), str(cfg["path"]), str(n_gen), "SERVER_SENTINEL", "1"],
        env=env, cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.set_blocking(proc.stdout.fileno(), False)
    os.set_blocking(proc.stderr.fileno(), False)
    # wait for load-complete READY
    buf = b""
    t0 = time.time()
    while b"<<<READY>>>" not in buf:
        chunk = proc.stdout.read(4096)
        if chunk:
            buf += chunk
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace")[-1500:]
            raise RuntimeError(f"engine died during load:\n{err}")
        if time.time() - t0 > 600:
            _terminate(proc)
            raise RuntimeError("engine load timed out")
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
}


def parse_metrics(tail):
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = re.search(pat, tail)
        if m:
            out[k] = m.groups()
    return out


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Model")
    available = [k for k, v in MODELS.items() if v["path"] and Path(v["path"]).exists()]
    if not available:
        st.error(f"no model files found under {ROOT} — check models/ and hf_home/")
        st.stop()
    choice = st.selectbox("model", available)
    cfg = dict(MODELS[choice])
    st.caption(cfg["note"])
    sys_prompt = st.text_area(
        "system prompt",
        value="You are a helpful, concise assistant. Answer the user's message directly. Reasoning: low",
        help="grounds the model; 'Reasoning: low' keeps gpt-oss from long thinking")
    cfg["margin"] = st.slider("margin (speed↔quality dial; 0 = bit-exact)", 0.0, 2.0, float(cfg["margin"]), 0.05)
    if cfg["margin"] > 0.5:
        st.warning("margin > 0.5 is OUTSIDE the validated quality band — the model can derail. "
                   "0.25 is the battery-validated default.")
    cfg["slots"] = st.select_slider(
        "slots/layer (expert cache)", ["auto", 4, 8, 12, 16, 32, 48], value=cfg["slots"],
        help="auto = engine sizes the cache from THIS machine's free memory "
             "(model metadata + available RAM + guard headroom) — the "
             "docker-style resource mode")
    cfg["backend"] = st.radio("backend", ["cpu", "gpu"], horizontal=True,
                              help="CPU loads much faster and decodes the same at default settings")
    n_gen = st.number_input("max new tokens", min_value=16, max_value=1024, value=128, step=16)

    st.divider()
    st.subheader("Engine")
    slot = _server_slot()
    alive = slot["proc"] is not None and slot["proc"].poll() is None
    if alive:
        try:
            rss = psutil.Process(slot["proc"].pid).memory_info().rss / 1e9
            st.success(f"running · pid {slot['proc'].pid} · RSS {rss:.1f} GB · "
                       f"warm {int(time.time() - slot['started'])}s")
        except psutil.NoSuchProcess:
            st.info("engine stopped")
        if st.button("⏹ Stop engine now", type="primary", use_container_width=True):
            _terminate(slot["proc"])
            slot.update(proc=None, key=None)
            st.rerun()
    else:
        stray = subprocess.run(["pgrep", "-f", "SERVER_SENTINEL"], capture_output=True)
        if stray.returncode == 0:
            subprocess.run(["pkill", "-9", "-f", "SERVER_SENTINEL"], capture_output=True)
            st.warning("found and killed an orphaned engine from a previous session")
        st.info("engine off — starts on your first message")
    st.caption(f"auto-stops after {IDLE_EXIT_S // 60} min idle (driver-side — "
               "survives even if this UI crashes). Config changes restart it.")

# ---------------- layout ----------------
chat_col, stats_col = st.columns([2.2, 1.0])

with stats_col:
    st.subheader("Live machine")
    cpu_ph = st.empty()
    ram_ph = st.empty()
    eng_ph = st.empty()
    st.subheader("Run metrics")
    met_ph = st.empty()
    guard_ph = st.empty()


def refresh_stats(eng_ps=None):
    vm = psutil.virtual_memory()
    cpu_ph.metric("system CPU", f"{psutil.cpu_percent(interval=None):.0f}%")
    ram_ph.metric("RAM used", f"{vm.used / 1e9:.1f} / {vm.total / 1e9:.0f} GB",
                  f"{vm.available / 1e9:.1f} GB free", delta_color="off")
    if eng_ps is not None and eng_ps.is_running():
        with eng_ps.oneshot():
            eng_ph.metric("engine", f"{eng_ps.cpu_percent(interval=None):.0f}% CPU",
                          f"RSS {eng_ps.memory_info().rss / 1e9:.2f} GB", delta_color="off")
    else:
        eng_ph.metric("engine", "idle")


refresh_stats(psutil.Process(slot["proc"].pid) if alive else None)

with chat_col:
    st.subheader("Chat")
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    for turn in st.session_state.chat_log:
        with st.chat_message(turn["role"]):
            if turn.get("thinking"):
                with st.expander("thinking (analysis channel)"):
                    st.text(turn["thinking"])
            st.markdown(turn["text"])
            if turn.get("timing"):
                st.caption(turn["timing"])
    prompt = st.chat_input("ask the model…")

if prompt:
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_log.append({"role": "user", "text": prompt})

        with st.chat_message("assistant"):
            status = st.status("starting…", expanded=False)
            think_ph = st.empty()
            answer_ph = st.empty()
            time_ph = st.empty()
            try:
                proc = ensure_server(cfg, sys_prompt, n_gen, status)
            except RuntimeError as e:
                status.update(label="engine failed", state="error")
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
            status.update(label="prefilling prompt…", state="running")

            buf = b""
            stderr_tail = ""
            streaming = False
            done = False
            ttft = None
            raw = ""
            last_stat = 0.0
            while not done:
                chunk = proc.stdout.read(4096)
                echunk = proc.stderr.read(4096)
                if echunk:
                    stderr_tail = (stderr_tail + echunk.decode(errors="replace"))[-4000:]
                if chunk:
                    buf += chunk
                    if not streaming and b"<<<STREAM>>>\n" in buf:
                        streaming = True
                        status.update(label="generating…", state="running")
                        buf = buf.split(b"<<<STREAM>>>\n", 1)[1]
                    if streaming:
                        end = buf.find(b"<<<END>>>")
                        raw = (buf[:end] if end >= 0 else buf).decode(errors="replace")
                        if ttft is None and raw.strip():
                            ttft = time.time() - t0
                        m = re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S)
                        if m:
                            final = re.sub(r"<\|[^|]*\|>", "", m.group(1))
                            thinking = re.sub(r"<\|[^|]*\|>", " ", raw[:m.start()])
                        elif "<|" in raw:
                            final = ""
                            thinking = re.sub(r"<\|[^|]*\|>", " ", raw)
                        else:
                            final = raw
                            thinking = ""
                        if thinking.strip():
                            with think_ph.container():
                                with st.expander("thinking (analysis channel)", expanded=(not final)):
                                    st.text(thinking)
                        answer_ph.markdown((final + ("▌" if end < 0 else "")) if (final or not thinking) else "")
                    if b"<<<READY>>>" in buf:
                        done = True
                if proc.poll() is not None:
                    done = True
                if time.time() - last_stat > 0.5:
                    last_stat = time.time()
                    try:
                        refresh_stats(eng_ps)
                    except psutil.NoSuchProcess:
                        pass
                    guard = [l for l in stderr_tail.splitlines() if "pressure" in l or "guard" in l]
                    if guard:
                        guard_ph.warning("memory guard active:\n" + "\n".join(guard[-3:]))
                if not chunk and not echunk:
                    time.sleep(0.05)

            total = time.time() - t0
            tail = buf.decode(errors="replace")
            met = parse_metrics(tail)
            status.update(label="done (engine stays warm)", state="complete")

            timing_bits = [f"total {total:.1f}s"]
            if ttft:
                timing_bits.append(f"first token {ttft:.1f}s")
            if "decode" in met:
                timing_bits.append(f"decode {met['decode'][1]} tok/s")
            timing = " · ".join(timing_bits)
            time_ph.caption(timing)

            rows = []
            if "reused" in met and int(met["reused"][0]) > 0:
                rows.append(f"context reused {met['reused'][0]} tokens (multi-turn KV)")
            if "prefill" in met:
                rows.append(f"prefill {met['prefill'][1]} tok/s")
            if "decode" in met:
                rows.append(f"decode {met['decode'][1]} tok/s")
            if "hit" in met:
                rows.append(f"cache hit {float(met['hit'][0]) * 100:.0f}%")
            if "agreement" in met:
                rows.append(f"routing fidelity {float(met['agreement'][0]) * 100:.1f}%")
            if "mem" in met:
                rows.append(f"peak RSS {met['mem'][0]} GB · footprint {met['mem'][1]} GB")
            met_ph.info("\n\n".join(rows) if rows else "no metrics parsed")

            m = re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S)
            final_clean = re.sub(r"<\|[^|]*\|>", "", m.group(1) if m else raw).strip()
            a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
            if met.get("generated") == ("0",) or not (final_clean or (a and a.group(1).strip())):
                final_clean = final_clean or "*(model produced no tokens — try rephrasing or adjusting the system prompt)*"
            st.session_state.chat_log.append(
                {"role": "assistant", "text": final_clean,
                 "thinking": a.group(1).strip() if a else "", "timing": timing})
