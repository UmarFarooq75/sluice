"""llmstream probe -- a Streamlit test bench for the llmstream engine.

Drives the CLI binaries (csrc/stream_run for CPU, csrc/stream_run_metal for
Metal) as a subprocess:

    ./csrc/stream_run <model.gguf> <n_gen> <prompt> [n_ubatch]

with the LLMSTREAM_* environment knobs, streams the engine's stdout live into
the page, parses the metric lines it prints, and presents them as three pillar
groups: SPEED (how fast), QUALITY (how faithful), COMPUTE (how small).

Only one engine process is ever allowed at a time -- this is a 16 GB machine
and two model processes would swap. Starting a new run (from any browser tab)
terminates the previous process first.

Run with:
    .venv/bin/streamlit run examples/streamlit_probe/app.py
"""

import atexit
import os
import re
import select
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import psutil  # optional: enables the Peak RSS tile
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parents[2]  # .../Desktop/research
BACKENDS = {
    "CPU (stream_run)": ROOT / "csrc" / "stream_run",
    "Metal GPU (stream_run_metal)": ROOT / "csrc" / "stream_run_metal",
}
FALLBACK_MB_PER_EXPERT = 13.25  # gpt-oss-style estimate when the engine line is missing
LOG_TAIL_CHARS = 12_000         # how much of the log to render live
POLL_SECONDS = 0.75             # fragment auto-rerun interval while a run is active

st.set_page_config(page_title="llmstream probe", layout="wide")


# --------------------------------------------------------------------------- #
# process management: one engine subprocess, ever, across all sessions        #
# --------------------------------------------------------------------------- #

def _terminate(proc):
    """Politely then firmly stop an engine process."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


@st.cache_resource
def _registry():
    """Cross-session singleton tracking the single allowed engine process."""
    reg = {"lock": threading.Lock(), "proc": None}
    atexit.register(lambda: _terminate(reg["proc"]))  # no orphaned model processes
    return reg


def engine_cmd_env(cfg):
    """Build argv and the extra environment for a run config."""
    cmd = [cfg["binary"], cfg["model"], str(cfg["n_gen"]), cfg["prompt"]]
    if cfg["ubatch"]:
        cmd.append(str(cfg["ubatch"]))
    env = {
        "LLMSTREAM_SLOTS": cfg["slots"],
        "LLMSTREAM_MARGIN": f"{cfg['margin']:g}",
        "LLMSTREAM_PREFETCH": "1" if cfg["prefetch"] else "0",
    }
    if cfg["threads"]:
        env["LLMSTREAM_THREADS"] = str(cfg["threads"])
    if cfg["metal"]:
        env["LLMSTREAM_SLOT_DEV"] = "gpu"
        env["LLMSTREAM_NGL"] = str(cfg["ngl"])
    return cmd, env


def start_engine(cfg):
    """Kill whatever engine process exists anywhere, then launch a new one."""
    reg = _registry()
    with reg["lock"]:
        _terminate(reg["proc"])
        cmd, extra_env = engine_cmd_env(cfg)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, **extra_env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # loader noise and metrics in one stream
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        reg["proc"] = proc
    return proc


def drain_output(proc):
    """Non-blocking: move available engine output into session state.

    Returns True once the pipe hits EOF (the process is done).
    """
    fd = proc.stdout.fileno()
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return False
        chunk = os.read(fd, 65536)
        if not chunk:
            return True
        st.session_state.raw_out += chunk


def sample_rss(proc, run):
    """Track peak resident set size of the engine, if psutil is available."""
    if psutil is None or proc.poll() is not None:
        return
    try:
        rss = psutil.Process(proc.pid).memory_info().rss
        run["peak_rss"] = max(run.get("peak_rss") or 0, rss)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# parsing the engine's output                                                 #
# --------------------------------------------------------------------------- #

def parse_output(text):
    """Pull metrics out of the combined stdout/stderr of a run.

    Expected lines (see csrc/stream_run.cpp):
        llmstream: 36 layers, 8 slots/layer, 2 tensors/expert, 13.25 MB/expert, ...
        prefill: 13.76 s (0.73 tok/s)
        decode:  76.00 s (0.75 tok/s)
        io: ... decode_misses=... (hit 0.526) ...
        io: margin=... / io: router_agreement=0.9550 ...
        logits_hash=...
        text: <generated text, possibly multi-line>
    """
    def fnum(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    met = {
        "prefill_s":   fnum(r"prefill:\s+([\d.]+)\s*s"),
        "prefill_tps": fnum(r"prefill:\s+[\d.]+\s*s\s*\(([\d.]+)\s*tok/s\)"),
        "decode_s":    fnum(r"decode:\s+([\d.]+)\s*s"),
        "decode_tps":  fnum(r"decode:\s+[\d.]+\s*s\s*\(([\d.]+)\s*tok/s\)"),
        "hit":         fnum(r"\(hit\s+([\d.]+)\)"),
        "agree":       fnum(r"router_agreement=([\d.]+)"),
    }

    m = re.search(r"logits_hash=([0-9a-fA-F]+)", text)
    met["logits_hash"] = m.group(1) if m else None

    m = re.search(
        r"llmstream:\s+(\d+)\s+layers,\s+(\d+)\s+slots/layer"
        r"(?:,\s+\d+\s+tensors/expert,\s+([\d.]+)\s+MB/expert)?",
        text,
    )
    met["llmstream_line"] = m.group(0) if m else None
    met["layers"] = int(m.group(1)) if m else None
    met["slots_per_layer"] = int(m.group(2)) if m else None
    met["mb_per_expert"] = float(m.group(3)) if m and m.group(3) else None

    m = re.search(r"generated=(\d+)", text)
    met["generated"] = int(m.group(1)) if m else None

    # COMPUTE pillar: expert cache = slots x layers x MB-per-expert
    if met["layers"] and met["slots_per_layer"]:
        mb = met["mb_per_expert"] or FALLBACK_MB_PER_EXPERT
        met["cache_gb"] = met["layers"] * met["slots_per_layer"] * mb / 1024.0
    else:
        met["cache_gb"] = None

    # generated text: the final "text: ..." line (may span lines), searched
    # after logits_hash so text that itself contains "text:" cannot confuse us
    start = 0
    mh = re.search(r"logits_hash=[0-9a-fA-F]+\n?", text)
    if mh:
        start = mh.end()
    mt = re.search(r"(?m)^text: ", text[start:])
    met["gen_text"] = text[start + mt.end():].rstrip("\n") if mt else None
    return met


def finalize_run():
    """Parse the finished (or killed) run, file it into history, clear state."""
    ss = st.session_state
    proc, run = ss.proc, ss.run
    ss.proc = None
    ss.run = None
    if proc is None or run is None:
        return
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
    try:
        proc.stdout.close()
    except OSError:
        pass
    reg = _registry()
    with reg["lock"]:
        if reg["proc"] is proc:
            reg["proc"] = None

    met = parse_output(ss.raw_out.decode("utf-8", errors="replace"))
    status = "ok" if rc == 0 else ("killed" if rc < 0 else f"exit {rc}")
    result = {**run, **met, "status": status, "rc": rc,
              "duration": time.time() - run["t0"]}
    ss.prev = ss.last            # previous run, for metric deltas
    ss.last = result
    ss.history.append({
        "time": run["ts"],
        "model": run["model_file"],
        "backend": run["backend"],
        "slots": run["slots"],
        "margin": run["margin"],
        "n_gen": run["n_gen"],
        "decode tok/s": met["decode_tps"],
        "prefill tok/s": met["prefill_tps"],
        "hit %": round(met["hit"] * 100, 1) if met["hit"] is not None else None,
        "agree %": round(met["agree"] * 100, 2) if met["agree"] is not None else None,
        "cache GB": round(met["cache_gb"], 2) if met["cache_gb"] is not None else None,
        "peak GB": round(run["peak_rss"] / 2**30, 2) if run.get("peak_rss") else None,
        "status": status,
    })


# --------------------------------------------------------------------------- #
# small formatting helpers                                                    #
# --------------------------------------------------------------------------- #

def fmt_num(v, nd=2):
    return f"{v:.{nd}f}" if v is not None else "n/a"


def fmt_pct(v, nd=1):
    return f"{v * 100:.{nd}f}%" if v is not None else "n/a"


def delta_str(cur, prev, nd=2, scale=1.0):
    """Delta vs the previous run for st.metric; None hides the delta."""
    if cur is None or prev is None:
        return None
    return f"{(cur - prev) * scale:+.{nd}f}"


def log_tail():
    text = st.session_state.raw_out.decode("utf-8", errors="replace")
    if len(text) > LOG_TAIL_CHARS:
        text = "... (earlier output truncated) ...\n" + text[-LOG_TAIL_CHARS:]
    return text


@st.cache_data(show_spinner=False)
def find_models():
    """Scan models/ and hf_home/hub/ for .gguf files -> {label: path}."""
    paths = []
    d = ROOT / "models"
    if d.is_dir():
        paths += sorted(d.glob("*.gguf"))
    d = ROOT / "hf_home" / "hub"
    if d.is_dir():
        paths += sorted(d.rglob("*.gguf"))
    out = {}
    for p in paths:
        label = f"{p.name}  ({p.stat().st_size / 2**30:.1f} GB)"
        if label in out:  # duplicate basename: disambiguate with parent dir
            label = f"{p.parent.name}/{label}"
        out[label] = str(p)
    return out


# --------------------------------------------------------------------------- #
# session state                                                               #
# --------------------------------------------------------------------------- #

for key, default in (("proc", None), ("run", None), ("last", None),
                     ("prev", None), ("history", []), ("raw_out", b"")):
    st.session_state.setdefault(key, default)


# --------------------------------------------------------------------------- #
# sidebar: run configuration                                                  #
# --------------------------------------------------------------------------- #

st.sidebar.header("Configuration")

models = find_models()
if st.sidebar.button("Rescan model dirs"):
    find_models.clear()
    st.rerun()

model_label = None
if models:
    model_label = st.sidebar.selectbox("Model (.gguf)", list(models))
else:
    st.sidebar.warning("No .gguf files found under models/ or hf_home/hub/.")

available_backends = {
    name: path for name, path in BACKENDS.items()
    if path.is_file() and os.access(path, os.X_OK)
}
backend_label = None
if available_backends:
    backend_label = st.sidebar.selectbox("Backend", list(available_backends))
else:
    st.sidebar.error("No stream_run binary in csrc/ -- build the engine first.")
is_metal = bool(backend_label) and "metal" in available_backends[backend_label].name

slots_raw = st.sidebar.text_input(
    "Slots per layer (LLMSTREAM_SLOTS)", value="auto",
    help="Expert-cache slots per layer: an integer, 0 for fully resident "
         "(no streaming), or 'auto' to let the engine size the cache.")
slots = slots_raw.strip().lower()
slots_ok = slots == "auto" or slots.isdigit()
if not slots_ok:
    st.sidebar.error("Slots must be an integer or 'auto'.")

margin = st.sidebar.slider(
    "Margin (LLMSTREAM_MARGIN)", 0.0, 2.0, 0.0, 0.01,
    help="Router margin. 0 = bit-exact output; higher = faster (more cache "
         "hits) at a measured quality cost.")
st.sidebar.caption("Margin 0 is bit-exact; raising it trades output fidelity "
                   "for speed.")

n_gen = st.sidebar.number_input("Tokens to generate", 1, 4096, 32)
prompt = st.sidebar.text_area(
    "Prompt", height=120,
    value="The three most important ideas in computer architecture are")

with st.sidebar.expander("Advanced"):
    prefetch = st.checkbox("Prefetch experts (LLMSTREAM_PREFETCH)", value=True)
    threads = st.number_input("Threads (LLMSTREAM_THREADS, 0 = engine default)",
                              0, 32, 0)
    ubatch = st.number_input("n_ubatch (0 = engine default)", 0, 4096, 0)
    ngl = 999
    if is_metal:
        ngl = st.number_input("GPU layers (LLMSTREAM_NGL)", 0, 999, 999)

ready = bool(models and available_backends and slots_ok and prompt.strip())

cfg = None
if model_label and backend_label:
    cfg = {
        "binary": str(available_backends[backend_label]),
        "model": models[model_label],
        "model_file": Path(models[model_label]).name,
        "backend": "metal" if is_metal else "cpu",
        "metal": is_metal,
        "slots": slots if slots_ok else "auto",
        "margin": float(margin),
        "n_gen": int(n_gen),
        "ubatch": int(ubatch),
        "threads": int(threads),
        "prefetch": bool(prefetch),
        "ngl": int(ngl) if is_metal else None,
        "prompt": prompt,
    }
    cmd, extra_env = engine_cmd_env(cfg)
    preview = " ".join(
        [f"{k}={v}" for k, v in extra_env.items()]
        + [shlex.quote(os.path.relpath(c, ROOT) if os.path.isabs(c) else c)
           for c in cmd])
    with st.sidebar.expander("Command preview"):
        st.code(preview, language="bash")


# --------------------------------------------------------------------------- #
# main: title, run/kill controls                                              #
# --------------------------------------------------------------------------- #

st.title("llmstream probe")
st.caption("Test bench for the llmstream engine. Every run answers three "
           "questions: how fast (SPEED), how faithful (QUALITY), "
           "how small (COMPUTE).")

col_run, col_kill, _ = st.columns([1, 1, 4])
run_clicked = col_run.button("Run", type="primary", disabled=not ready)
kill_clicked = col_kill.button("Kill run")

if kill_clicked:
    reg = _registry()
    with reg["lock"]:
        target = st.session_state.proc or reg["proc"]
        _terminate(target)
        if reg["proc"] is target:
            reg["proc"] = None
    if target is None:
        st.toast("No engine process is running.")
    # a killed session process is drained and filed as 'killed' by the
    # live panel below

if run_clicked and cfg is not None:
    if st.session_state.proc is not None:
        # a run is still going: stop it and file it before starting fresh
        _terminate(st.session_state.proc)
        try:
            st.session_state.raw_out += st.session_state.proc.stdout.read() or b""
        except OSError:
            pass
        finalize_run()
    st.session_state.raw_out = b""
    st.session_state.run = {
        **cfg,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "t0": time.time(),
        "peak_rss": None,
    }
    st.session_state.proc = start_engine(cfg)


# --------------------------------------------------------------------------- #
# live panel: polls the subprocess without full-page reruns                   #
# --------------------------------------------------------------------------- #

run_active = st.session_state.proc is not None


@st.fragment(run_every=POLL_SECONDS if run_active else None)
def live_panel():
    ss = st.session_state
    proc, run = ss.proc, ss.run
    if proc is not None and run is not None:
        done = drain_output(proc)
        sample_rss(proc, run)
        elapsed = time.time() - run["t0"]
        label = (f"Running {run['model_file']} on {run['backend']} "
                 f"(slots={run['slots']}, margin={run['margin']:g}, "
                 f"n_gen={run['n_gen']}) -- {elapsed:.0f} s")
        with st.status(label, state="running", expanded=True):
            st.code(log_tail() or "(waiting for engine output)", language="text")
        if done:
            finalize_run()
            st.rerun()  # full-page rerun renders the results panel below
    else:
        last = ss.last
        if last is None:
            st.caption("No run yet. Pick a model in the sidebar and press Run.")
            return
        label = (f"Engine log -- {last['model_file']} finished with status "
                 f"'{last['status']}' in {last['duration']:.1f} s")
        state = "complete" if last["status"] == "ok" else "error"
        with st.status(label, state=state, expanded=False):
            st.code(log_tail() or "(no output captured)", language="text")


live_panel()


# --------------------------------------------------------------------------- #
# results panel: SPEED / QUALITY / COMPUTE pillars, output, history           #
# --------------------------------------------------------------------------- #

last = st.session_state.last
prev = st.session_state.prev or {}

if last is not None:
    st.subheader("Last run")
    st.caption(f"{last['model_file']} -- {last['backend']} -- "
               f"slots={last['slots']} margin={last['margin']:g} "
               f"n_gen={last['n_gen']} -- status: {last['status']}"
               + (f" -- generated {last['generated']} tokens"
                  if last.get("generated") is not None else ""))

    speed_col, quality_col, compute_col = st.columns(3, gap="medium")

    with speed_col.container(border=True):
        st.markdown("**SPEED** -- how fast")
        a, b = st.columns(2)
        a.metric("Decode tok/s", fmt_num(last["decode_tps"]),
                 delta=delta_str(last["decode_tps"], prev.get("decode_tps")),
                 help="Steady-state generation speed", border=True)
        b.metric("Prefill tok/s", fmt_num(last["prefill_tps"]),
                 delta=delta_str(last["prefill_tps"], prev.get("prefill_tps")),
                 help="Prompt-processing speed", border=True)
        st.caption(f"prefill {fmt_num(last['prefill_s'])} s + "
                   f"decode {fmt_num(last['decode_s'])} s wall time")

    with quality_col.container(border=True):
        st.markdown("**QUALITY** -- how faithful")
        a, b = st.columns(2)
        if last["agree"] is not None:
            agree_val = fmt_pct(last["agree"], 2)
        elif last["margin"] == 0:
            agree_val = "bit-exact"
        else:
            agree_val = "n/a"
        a.metric("Router agreement", agree_val,
                 delta=delta_str(last["agree"], prev.get("agree"), scale=100),
                 help="Share of expert-routing decisions identical to exact "
                      "(margin 0) routing", border=True)
        b.metric("Margin", f"{last['margin']:g}",
                 help="LLMSTREAM_MARGIN used for this run", border=True)
        st.caption("Margin 0 = bit-exact output; higher runs faster at a "
                   "measured quality cost."
                   + (f" logits_hash={last['logits_hash']}"
                      if last["logits_hash"] else ""))

    with compute_col.container(border=True):
        st.markdown("**COMPUTE** -- how small")
        a, b = st.columns(2)
        a.metric("Expert cache",
                 f"{last['cache_gb']:.2f} GB" if last["cache_gb"] is not None else "n/a",
                 delta=delta_str(last["cache_gb"], prev.get("cache_gb")),
                 delta_color="inverse",
                 help="slots x layers x MB/expert (engine-reported size; "
                      f"{FALLBACK_MB_PER_EXPERT} MB/expert fallback)",
                 border=True)
        b.metric("Cache hit rate", fmt_pct(last["hit"]),
                 delta=delta_str(last["hit"], prev.get("hit"), nd=1, scale=100),
                 help="Expert-cache hit rate during decode", border=True)
        c, d = st.columns(2)
        slots_val = (f"{last['slots_per_layer']}/layer"
                     if last["slots_per_layer"] is not None else str(last["slots"]))
        c.metric("Slots", slots_val,
                 help=f"Requested: {last['slots']}", border=True)
        if last.get("peak_rss"):
            d.metric("Peak RSS", f"{last['peak_rss'] / 2**30:.2f} GB",
                     delta=delta_str(
                         last["peak_rss"] / 2**30,
                         (prev.get("peak_rss") or 0) / 2**30 if prev.get("peak_rss") else None),
                     delta_color="inverse",
                     help="Peak resident memory of the engine process", border=True)
        st.caption((last["llmstream_line"] or "resident mode (no llmstream line)")
                   + ("" if psutil else " -- pip install psutil for peak RSS"))

    st.subheader("Generated text")
    with st.chat_message("user"):
        st.write(last["prompt"])
    with st.chat_message("assistant"):
        if last["gen_text"]:
            st.write(last["gen_text"])
        else:
            st.caption("(no generated text captured -- run killed or failed)")

if st.session_state.history:
    st.subheader(f"History -- {len(st.session_state.history)} run(s) this session")
    st.dataframe(
        list(reversed(st.session_state.history)),
        hide_index=True,
        column_config={
            "decode tok/s": st.column_config.NumberColumn(format="%.2f"),
            "prefill tok/s": st.column_config.NumberColumn(format="%.2f"),
            "margin": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.prev = None
        st.rerun()
