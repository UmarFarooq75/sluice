# Chat page: talk to a streamed model and watch the machine while it answers.
#
# Left: chat with live token streaming (harmony "analysis" thinking collapses
# into an expander; the final channel renders as the reply).
# Right: live stats sampled every ~0.5s while the engine runs — engine CPU%,
# RSS, system RAM, plus the engine's own end-of-run metrics (tok/s, hit rate,
# routing fidelity, footprint).
#
# Honest boundaries, stated in the UI:
# - v1 turns are stateless: each message is its own conversation (the driver
#   is one-shot; persistent-KV server mode is future work). On 120B each turn
#   re-prefills — the first tokens take a while.
# - GPU utilization needs `sudo powermetrics` on macOS; we show backend +
#   engine footprint instead of pretending.
import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / "csrc" / "stream_run"
REGISTRY_KEY = "llmstream_chat_proc"

st.set_page_config(page_title="llmstream chat", layout="wide")

MODELS = {
    "gpt-oss-120b (117B, MXFP4)": {
        "path": ROOT / "models" / "gpt-oss-120b-MXFP4.gguf",
        "slots": 8, "margin": 0.25, "chat": True, "backend": "cpu",
        "note": "product default: ~1.6 tok/s decode, prefill is slow (re-streams per turn)",
    },
    "OLMoE-1B-7B (fast)": {
        "path": next((ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").glob("*/*.gguf"), None)
        if (ROOT / "hf_home/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots").exists() else None,
        "slots": 32, "margin": 0.0, "chat": True, "backend": "cpu",
        "note": "small model: ~28 tok/s streamed, snappy for UI testing",
    },
}


def _registry():
    if REGISTRY_KEY not in st.session_state:
        st.session_state[REGISTRY_KEY] = None
    return st.session_state


def engine_running():
    return subprocess.run(["pgrep", "-f", "stream_run"], capture_output=True).returncode == 0


def build_cmd(cfg, prompt, n_gen):
    env = os.environ.copy()
    env.update({
        "LLMSTREAM_SLOTS": str(cfg["slots"]),
        "LLMSTREAM_MARGIN": str(cfg["margin"]),
        "LLMSTREAM_STREAM_OUT": "1",
    })
    if cfg["chat"]:
        env["LLMSTREAM_CHAT"] = "1"
    if cfg["backend"] == "gpu":
        env["LLMSTREAM_SLOT_DEV"] = "gpu"
        env["LLMSTREAM_NGL"] = "99"
    return [str(ENGINE), str(cfg["path"]), str(n_gen), prompt, "1"], env


METRIC_PATTERNS = {
    "prefill": r"prefill:\s+([\d.]+) s \(([\d.]+) tok/s\)",
    "decode": r"decode:\s+([\d.]+) s \(([\d.]+) tok/s\)",
    "hit": r"hit ([\d.]+)",
    "agreement": r"router_agreement=([\d.]+)",
    "mem": r"peak_rss=([\d.]+) GB phys_footprint=([\d.]+) GB",
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
    choice = st.selectbox("model", [k for k, v in MODELS.items() if v["path"] and Path(v["path"]).exists()])
    cfg = dict(MODELS[choice])
    st.caption(cfg["note"])
    cfg["margin"] = st.slider("margin (speed↔quality dial; 0 = bit-exact)", 0.0, 2.0, float(cfg["margin"]), 0.05)
    cfg["slots"] = st.select_slider("slots/layer (expert cache)", [4, 8, 12, 16, 32, 48], value=cfg["slots"])
    cfg["backend"] = st.radio("backend", ["cpu", "gpu"], horizontal=True,
                              help="GPU only pays above ~0.9 hit rate (measured); CPU is the default")
    n_gen = st.slider("max new tokens", 16, 512, 128, 16)
    st.divider()
    st.caption("v1 is stateless per turn — each message re-prefills. "
               "Quality/speed/compute numbers per mode are in the README table.")

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


def refresh_stats(eng_proc_ps=None):
    vm = psutil.virtual_memory()
    cpu_ph.metric("system CPU", f"{psutil.cpu_percent(interval=None):.0f}%")
    ram_ph.metric("RAM used", f"{vm.used / 1e9:.1f} / {vm.total / 1e9:.0f} GB",
                  f"{vm.available / 1e9:.1f} GB free", delta_color="off")
    if eng_proc_ps is not None and eng_proc_ps.is_running():
        with eng_proc_ps.oneshot():
            eng_ph.metric("engine",
                          f"{eng_proc_ps.cpu_percent(interval=None):.0f}% CPU",
                          f"RSS {eng_proc_ps.memory_info().rss / 1e9:.2f} GB", delta_color="off")
    else:
        eng_ph.metric("engine", "idle")


refresh_stats()

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

        if engine_running():
            st.error("another engine process is already running — one model at a time (machine-safety rule)")
            st.stop()

        cmd, env = build_cmd(cfg, prompt, n_gen)
        t0 = time.time()
        proc = subprocess.Popen(cmd, env=env, cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL)
        ps_proc = psutil.Process(proc.pid)
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        with st.chat_message("assistant"):
            status = st.status(f"loading {choice} — first tokens can take a while…", expanded=False)
            think_ph = st.empty()
            answer_ph = st.empty()
            time_ph = st.empty()

            buf = b""
            stderr_tail = ""
            streaming = False
            ttft = None
            raw = ""
            last_stat = 0.0
            while True:
                if proc.poll() is not None and not streaming and b"<<<STREAM>>>" not in buf:
                    break
                chunk = proc.stdout.read(4096) if proc.stdout else None
                echunk = proc.stderr.read(4096) if proc.stderr else None
                if echunk:
                    stderr_tail = (stderr_tail + echunk.decode(errors="replace"))[-4000:]
                if chunk:
                    buf += chunk
                    if not streaming and b"<<<STREAM>>>\n" in buf:
                        streaming = True
                        status.update(label="generating…", state="running")
                        buf = buf.split(b"<<<STREAM>>>\n", 1)[1]
                    if streaming:
                        if ttft is None and buf.strip():
                            ttft = time.time() - t0
                        end = buf.find(b"<<<END>>>")
                        raw = (buf[:end] if end >= 0 else buf).decode(errors="replace")
                        # harmony split: analysis channel = thinking, final = reply
                        final = raw
                        thinking = ""
                        m = re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S)
                        if m:
                            final = m.group(1)
                            a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
                            if a:
                                thinking = a.group(1)
                        final = re.sub(r"<\|[^|]*\|>", "", final)
                        if thinking:
                            with think_ph.container():
                                with st.expander("thinking (analysis channel)", expanded=False):
                                    st.text(thinking)
                        answer_ph.markdown(final + ("▌" if end < 0 else ""))
                        if end >= 0 and proc.poll() is not None:
                            break
                if time.time() - last_stat > 0.5:
                    last_stat = time.time()
                    try:
                        refresh_stats(ps_proc)
                    except psutil.NoSuchProcess:
                        pass
                    guard = [l for l in stderr_tail.splitlines() if "pressure" in l or "guard" in l]
                    if guard:
                        guard_ph.warning("memory guard active:\n" + "\n".join(guard[-3:]))
                if not chunk and not echunk:
                    if proc.poll() is not None and (not streaming or b"<<<END>>>" in buf):
                        break
                    time.sleep(0.05)

            rest = (proc.stdout.read() or b"") if proc.stdout else b""
            tail = (buf + rest).decode(errors="replace")
            total = time.time() - t0
            met = parse_metrics(tail)
            status.update(label="done", state="complete")

            timing_bits = [f"total {total:.1f}s"]
            if ttft:
                timing_bits.append(f"first token {ttft:.1f}s")
            if "decode" in met:
                timing_bits.append(f"decode {met['decode'][1]} tok/s")
            timing = " · ".join(timing_bits)
            time_ph.caption(timing)

            rows = []
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
            refresh_stats()

            final_clean = re.sub(r"<\|[^|]*\|>", "",
                                 re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S).group(1)
                                 if re.search(r"<\|channel\|>final<\|message\|>", raw) else raw).strip()
            think_txt = ""
            a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
            if a:
                think_txt = a.group(1).strip()
            st.session_state.chat_log.append(
                {"role": "assistant", "text": final_clean or "(no output)",
                 "thinking": think_txt, "timing": timing})
