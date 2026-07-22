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

st.set_page_config(page_title="sluice", page_icon=":material/water:", layout="centered")

# Known-model metadata: nice descriptions + per-preset slot counts (measured
# footprints, E28/E30). Matched to discovered files by filename substring;
# anything else found on disk still runs, with safe auto defaults.
# "toks" = balanced-mode decode tok/s MEASURED on the reference machine (M2 Air,
# ~1.7 GB/s cold reads; docs/lablog.md); io-bound so it scales with YOUR disk.
REF_BW_GBS = 1.7
KNOWN = {
    "gpt-oss-120b": {
        "size": "117B total · 5.1B active · 63.4 GB", "toks": 2.4,
        "note": "the flagship: 4× this machine's RAM. First message pays a one-time load (~1–2 min); later turns stay warm.",
        "mem": {"Light": (8, "~5.7 GB"), "Balanced": (12, "~7.6 GB"), "Performance": (12, "~7.6 GB (16 trips the guard on 16GB)")},
    },
    "gpt-oss-20b": {
        "size": "21B total · 3.6B active · 12.1 GB", "toks": 3.5,
        "note": "the middle tier — ~3× the 120B, comfortable on 16 GB RAM.",
        "mem": {"Light": (12, "~5.5 GB"), "Balanced": (16, "~6.6 GB"), "Performance": (24, "~8.5 GB")},
    },
    "olmoe": {
        "size": "7B total · 1.3B active · 4.3 GB", "toks": 48.0,
        "note": "small and snappy — ideal for UI testing.",
        "mem": {"Light": (16, "~1.4 GB"), "Balanced": (32, "~2.6 GB"), "Performance": (48, "~3.5 GB")},
    },
}


@st.cache_data(show_spinner=False)
def machine_disk_bw():
    """Sample a model file with F_NOCACHE random reads (the streaming access
    pattern) → GB/s. Cached: runs once per session. None if no file to sample."""
    import fcntl, random as _rnd
    f = next((p for p in (ROOT / "models").glob("*.gguf") if p.stat().st_size > 1e9), None)
    if not f:
        return None
    span, n = 8 * 1024 * 1024, 24
    fd = os.open(str(f), os.O_RDONLY)
    try:
        try:
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE: honest cold reads
        except Exception:
            pass
        rng = _rnd.Random(42)
        size = f.stat().st_size
        t0 = time.time(); got = 0
        for _ in range(n):
            off = rng.randrange(0, max(1, size - span)) & ~4095
            got += len(os.pread(fd, span, off))
        return got / (time.time() - t0) / 1e9
    finally:
        os.close(fd)


def preload_estimate(choice, mode):
    """Expected decode tok/s for the selected model on THIS machine, before load.
    Scales the reference (io-bound) number by the measured disk bandwidth, and by
    the quality mode (higher fidelity floor = fewer skips = a touch slower)."""
    meta = model_meta(choice)
    ref = meta.get("toks")
    if not ref:
        return None
    bw = machine_disk_bw()
    scale = (bw / REF_BW_GBS) if bw else 1.0
    mode_mult = {"Exact": 0.75, "Balanced": 1.0, "Fast": 1.25}.get(mode, 1.0)
    return ref * scale * mode_mult, bw
DEFAULT_MEM = {"Light": (8, "smaller cache"), "Balanced": (16, "balanced cache"), "Performance": (32, "large cache")}


@st.cache_data(ttl=10)
def discover_models():
    """Scan disk for runnable GGUFs - models/ plus the hf_home cache. Newly
    pulled models appear automatically; nothing is hard-coded."""
    found = {}
    for gguf in sorted((ROOT / "models").glob("*.gguf")):
        found[gguf.stem.replace("-MXFP4", "").replace("-mxfp4", "")] = gguf
    hf = ROOT / "hf_home" / "hub"
    if hf.exists():
        for snap in hf.glob("models--*/snapshots/*/*.gguf"):
            name = snap.parts[snap.parts.index("hub") + 1].split("--")[-1]
            found.setdefault(name, snap)
    return {k: str(v) for k, v in found.items()}


def model_meta(name):
    key = next((k for k in KNOWN if k in name.lower()), None)
    return KNOWN[key] if key else {"size": "custom model", "note": "auto-detected on disk.", "mem": DEFAULT_MEM}

# Speed↔quality dial, calibrated by the measured battery - not raw knobs.
# Quality maps to a FAMILY-AGNOSTIC routing-fidelity floor (D3, closed E20):
# the fraction of the model's true top-k experts the router must keep. Same
# meaning on gpt-oss / OLMoE / Qwen / DeepSeek — the engine's adaptive
# controller finds the margin that holds it. Exact = bit-identical (agree off).
MODES = {
    "Exact": {"agree": 0.0, "margin": 0.0, "desc": "bit-identical to the resident model, hash-gated"},
    "Balanced": {"agree": 0.95, "margin": 0.0, "desc": "keep ≥95% of routing decisions · no measurable quality change (5-domain NLL) · recommended"},
    "Fast": {"agree": 0.90, "margin": 0.0, "desc": "keep ≥90% · faster · edge of the validated band"},
}

SUGGESTIONS = {
    ":blue[:material/lightbulb:] Explain something": "Explain why the sky is blue in two sentences.",
    ":green[:material/code:] Write code": "Write a Python function that checks whether a number is prime.",
    ":violet[:material/psychology:] Reason": "A train leaves at 9am at 40 mph; another at 11am at 60 mph on a parallel track. When does the second catch the first?",
}

DEFAULT_SYSTEM = """You are a knowledgeable, reliable, and practical AI assistant.

Your goals are to:
- Provide accurate, clear, and helpful answers.
- Ask brief clarifying questions when information is missing or ambiguous.
- Be concise by default, but provide more detail when requested or when it improves the answer.
- Explain complex topics in plain language without oversimplifying.
- Be honest about uncertainty. Never fabricate facts, sources, or capabilities.
- If you don't know something, say so and suggest how the user can find the answer.
- Follow the user's instructions as long as they are safe, legal, and consistent with your role.
- Correct mistakes when identified without being defensive.
- Structure responses for readability using headings or bullet points when appropriate.
- When solving problems, think through the task internally and present only the final reasoning needed for the user.
- For code, prioritize correctness, readability, security, and maintainability. Include comments only when they add value.
- When multiple valid approaches exist, briefly compare the main trade-offs and recommend one.
- Avoid unnecessary verbosity, repetition, and filler.

Communication style:
- Be friendly, professional, and direct.
- Match the user's level of technical knowledge when possible.
- Do not be overly apologetic or overly confident.
- Focus on actionable, useful responses.

Always prioritize being truthful, helpful, and clear.

Reasoning: low"""

# The DEFAULT_SYSTEM literal above must stay exactly as it is, trailing
# "Reasoning: low" included: results/e38/legs.py and results/e41b/gate.py scrape it
# out of this file with re.search(r'DEFAULT_SYSTEM = """(.*?)"""') so their runs use
# the UI's real system prompt. Both fall back SILENTLY to "You are a helpful
# assistant." on a miss, which would quietly destroy E38<->E41b comparability. So the
# reasoning level is stripped and re-appended at runtime rather than templated in.
REASONING_LEVELS = ["Low", "Medium", "High"]
_REASONING_RE = re.compile(r"\n*^Reasoning:[ \t]*\w+[ \t]*$", re.M)


def parse_canon_reply(text):
    """Extract the engine's `canon_reply:` line (E41b), or None if absent.

    Absent is the normal case: the line only exists when LLMSTREAM_KV_CANON is set,
    so None means "engine did not canonicalize" and the caller keeps its own text.
    The engine escapes newlines and backslashes to keep the line parseable; this
    reverses exactly that, in the order that makes \\\\n survive as a literal.
    """
    m = re.search(r"^canon_reply: (.*)$", text, re.M)
    if not m:
        return None
    out, s, i = [], m.group(1), 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append("\n" if s[i + 1] == "n" else s[i + 1]); i += 2
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def with_reasoning(sys_text, level):
    """Replace any trailing 'Reasoning: x' line with the chosen level.

    gpt-oss reads this line from the system prompt; it is not an engine flag.
    At Low this reproduces DEFAULT_SYSTEM byte for byte (asserted in the sidebar),
    so the default path is unchanged.
    """
    return _REASONING_RE.sub("", sys_text).rstrip() + f"\n\nReasoning: {level.lower()}"


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
    return (str(cfg["path"]), cfg["slots"], cfg["margin"], cfg.get("agree", 0.0), cfg["backend"],
            cfg.get("temp", 0.8), sys_prompt, n_gen)


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
        "LLMSTREAM_AGREE_TARGET": str(cfg.get("agree", 0.0)),
        "LLMSTREAM_STREAM_OUT": "1",
        "LLMSTREAM_SERVER": "1",
        "LLMSTREAM_IDLE_EXIT": str(IDLE_EXIT_S),
        "LLMSTREAM_CHAT": "1",
        "LLMSTREAM_PREFILL_SLOTS": "1",  # E27: expert-major prefill (8× TTFT)
        # chat sampling: greedy decoding loops ("X for Y for X for Y..."), so
        # use a repetition penalty + light temperature like every LLM runtime.
        # These match Ollama's defaults exactly (docs.ollama.com/modelfile).
        # (The bit-exact gates run WITHOUT these, staying greedy/deterministic.)
        "LLMSTREAM_TEMP": str(cfg.get("temp", 0.8)),
        "LLMSTREAM_TOP_P": "0.9",
        "LLMSTREAM_TOP_K": "40",
        "LLMSTREAM_REP_PEN": "1.1",
        "LLMSTREAM_REP_LAST": "64",
    })
    if sys_prompt.strip():
        env["LLMSTREAM_SYSTEM"] = sys_prompt.strip()
    # E41b canon: ON by default for chat (S3, after gate 5). Turn-2 TTFT drops from
    # ~8 s to ~3 s because the next turn reuses the whole canonical KV instead of
    # re-prefilling from the assistant boundary. LLMSTREAM_KV_CANON=0 disables it.
    #
    # The flip is CLIENT-side on purpose: the engine's own default stays OFF, so
    # every gate and harness keeps its pinned configuration without being touched,
    # and the later strict switch is a one-line change here rather than a revert.
    env["LLMSTREAM_KV_CANON"] = os.environ.get("LLMSTREAM_KV_CANON", "1")
    # ubatch drives batched (expert-major) prefill, which needs the CPU-only
    # prefill pool to absorb a batch's expert union. On GPU there is no pool,
    # so a batched prefill whose union exceeds the slot count hits the engine's
    # exit(1) boundary (D12) and the request dies with 0 tokens. GPU therefore
    # runs ubatch=1 (token-by-token prefill, always safe); CPU keeps 128 + pool.
    ubatch = "128"
    if cfg["backend"] == "gpu":
        env["LLMSTREAM_SLOT_DEV"] = "gpu"
        env["LLMSTREAM_NGL"] = "99"
        env.pop("LLMSTREAM_PREFILL_SLOTS", None)  # pf pool is CPU-only (v1)
        ubatch = "1"
    status.update(label="Loading model - one-time, stays warm after this…", state="running")
    proc = subprocess.Popen(
        [str(ENGINE), str(cfg["path"]), str(n_gen), "SERVER_SENTINEL", ubatch],
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


DEBUG_DIR = ROOT / "ui" / ".debug"
DEBUG_KEEP = 6          # turns retained
DEBUG_CAP = 200_000     # bytes per turn, so a runaway reply cannot fill the disk


def capture_turn(raw, stderr_tail, meta):
    """Persist the RAW stream (markers intact) + engine stderr for one turn.

    Exists because two live-UI symptoms could not be confirmed against the session
    that produced them: the UI held raw output and engine stderr only in memory, so
    when a reply rendered wrongly there was nothing left to diagnose. Rolling and
    capped; best-effort and never allowed to affect the turn.
    """
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (DEBUG_DIR / f"turn-{stamp}.txt").write_text(
            f"# {meta}\n\n=== RAW (markers intact) ===\n{raw[:DEBUG_CAP]}\n\n"
            f"=== ENGINE STDERR (tail) ===\n{stderr_tail[-8000:]}\n")
        old = sorted(DEBUG_DIR.glob("turn-*.txt"))[:-DEBUG_KEEP]
        for f in old:
            f.unlink(missing_ok=True)
    except Exception:
        pass    # diagnostics must never break a reply


def strip_harmony(t):
    """Remove ALL harmony control tokens from text, including the plain-text
    channel/role labels that follow a marker (`<|channel|>analysis`,
    `<|start|>assistant`) — otherwise the literal word 'analysis'/'final'
    leaks into the rendered text."""
    t = re.sub(r"<\|channel\|>\s*\w+", " ", t)   # channel marker + its name
    t = re.sub(r"<\|start\|>\s*\w+", " ", t)     # start marker + role
    t = re.sub(r"<\|[^|]*\|>", " ", t)           # any remaining COMPLETE markers
    # A partial marker at the buffer edge ("<|chan", "<|channel|") has no closing
    # "|>" yet, so nothing above matches it. Left alone it reaches st.markdown and
    # gets mangled into stray punctuation. Drop any trailing fragment.
    t = re.sub(r"<\|[^|]*\|?$", "", t)
    t = re.sub(r"<\|", "", t)                    # belt and braces: never show internals
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]+", " ", t)).strip()


def split_harmony(raw):
    """gpt-oss harmony format: analysis channel = thinking, final = answer."""
    m = re.search(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S)
    if m:
        final = strip_harmony(m.group(1))
        thinking = strip_harmony(raw[:m.start()])
    elif "<|" in raw:
        final, thinking = "", strip_harmony(raw)
    else:
        final, thinking = raw, ""
    return final, thinking


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### :material/water: sluice")
    st.caption("Virtual memory for LLMs - models bigger than your RAM, with quality receipts.")

    disk_models = discover_models()
    if not disk_models:
        st.error("No models found on disk. Pull one with `sluice pull gpt-oss-20b`, "
                 "then it appears here automatically.")
        st.stop()
    # default to gpt-oss-20b when present (best speed/quality balance on 16 GB).
    # note "20b" is a substring of "120b", so exclude the 120b explicitly.
    names = list(disk_models)
    default_ix = next((i for i, n in enumerate(names)
                       if "20b" in n.lower() and "120b" not in n.lower()), 0)
    choice = st.selectbox("Model", names, index=default_ix,
                          help="Every model on your disk — newly pulled ones show up automatically")
    meta = model_meta(choice)
    cfg = {"path": disk_models[choice]}
    st.caption(f":material/database: {meta['size']}")
    st.caption(meta["note"])

    # Memory: a plain-language budget, not "slots/layer". Each preset maps to a
    # measured RAM footprint for this model; the engine sizes its expert cache
    # to fit. "How much of your Mac to spend."
    mem_opts = meta["mem"]
    mem_choice = st.segmented_control(
        "Memory to use", list(mem_opts), default="Balanced",
        help="How much RAM to give the expert cache. More = faster (higher cache "
             "hit rate), up to what your Mac can spare without slowing down.")
    mem_choice = mem_choice or "Balanced"
    cfg["slots"], mem_gb = mem_opts[mem_choice]
    st.caption(f":material/memory: uses {mem_gb} of RAM")

    mode = st.segmented_control("Response quality", list(MODES.keys()), default="Balanced",
                                help="The speed↔quality dial, in battery-calibrated steps")
    mode = mode or "Balanced"
    cfg["margin"] = MODES[mode]["margin"]
    cfg["agree"] = MODES[mode]["agree"]
    st.caption(f":material/verified: {MODES[mode]['desc']}")

    reasoning = st.selectbox(
        "Reasoning effort", REASONING_LEVELS, index=0,
        help="How much the model thinks before answering. The honest trade-off: "
             "higher reasoning means MORE HIDDEN TOKENS generated before the first "
             "visible word, and at streamed speeds those tokens cost the same as "
             "visible ones. High can add tens of seconds to perceived first-token "
             "time even though tok/s is unchanged. Raise it for hard problems, not "
             "for chat.")
    reasoning = reasoning or "Low"
    show_thinking = st.toggle(
        "Show thinking", value=True,
        help="Reveal the model's internal reasoning in a collapsible panel. Hiding it "
             "does not make it cheaper — those tokens are still generated and still "
             "paid for; it only removes them from view.")
    if reasoning != "Low":
        st.caption(":material/timer: Slower first token — more hidden reasoning to generate.")

    cfg["backend"] = "cpu"
    cfg["temp"] = 0.8
    with st.expander("Advanced", icon=":material/tune:"):
        if cfg["agree"] > 0.0:
            cfg["agree"] = st.slider("Routing fidelity floor (keep ≥ of true experts)",
                                     0.85, 1.0, float(cfg["agree"]), 0.01,
                                     help="Family-agnostic quality dial (D3): the fraction of the "
                                          "model's true top-k experts the router must keep. Means the "
                                          "same thing on every model; lower = faster, more streaming skips.")
            if cfg["agree"] < 0.90:
                st.warning("Below 0.90 fidelity is outside the validated band — the model can dip.")
        else:
            st.caption(":material/lock: Exact: bit-identical routing (100% fidelity, reproducible).")
        cfg["slots"] = st.select_slider("Expert-cache slots/layer (raw)",
                                        [4, 8, 12, 16, 24, 32, 48], value=cfg["slots"],
                                        help="The raw cache size behind the Memory presets")
        # GPU is a footgun on Apple Silicon: it has no expert-major prefill pool,
        # so it prefills token-by-token (ubatch=1) and first-token can hit 100-300s
        # on a long prompt, while decoding no faster than CPU below ~0.9 hit. There
        # is no case on this hardware where GPU wins, so it is not selectable.
        cfg["backend"] = "cpu"
        st.caption(":material/bolt: Backend: CPU (expert-major prefill — fastest on Apple Silicon)")
        cfg["temp"] = st.slider("Temperature (creativity)", 0.0, 1.5, 0.8, 0.05,
                                help="0 = deterministic/focused, higher = more varied. "
                                     "Ollama's default is 0.8. Below ~0.3 can loop on long replies.")
        n_gen = st.number_input("Max new tokens", min_value=16, max_value=3584, value=2048, step=128,
                                help="Cap on reply length. Context window is 4096 tokens total "
                                     "(prompt + reply); very long chats truncate the oldest turns.")
        sys_prompt_base = st.text_area(
            "System prompt", value=_REASONING_RE.sub("", DEFAULT_SYSTEM).rstrip(), height=200,
            help="Grounds the model. The 'Reasoning:' line is no longer edited here — "
                 "the Reasoning effort dropdown above owns it and appends it for you. "
                 "Any 'Reasoning:' line you type is replaced by that setting.")

    sys_prompt = with_reasoning(sys_prompt_base, reasoning)
    # the default path must be byte-identical to what shipped before this control
    # existed, or every prior measurement stops describing the default UI
    assert with_reasoning(_REASONING_RE.sub("", DEFAULT_SYSTEM).rstrip(), "Low") == DEFAULT_SYSTEM

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
            est = preload_estimate(choice, mode)
            if est:
                toks, bw = est
                prov = f"scaled to your {bw:.1f} GB/s disk" if bw else "reference machine"
                st.caption(f":material/speed: expected ~**{toks:.0f} tok/s** ({mode.lower()}) · {prov}")
        st.caption(f":material/timer: auto-stops after {IDLE_EXIT_S // 60} min idle "
                   "(driver-side - survives a UI crash)")

    # Live machine stats. The fragment is DEFINED here but RENDERED at the very END
    # of the script — see the call after the chat block.
    #
    # Why: run_every=2 schedules an auto-rerun, and a rerun that fires while the main
    # script is blocked in the generation read loop PREEMPTS that run. The script
    # thread dies mid-turn, so the reply renders but its stats caption never gets
    # appended and the spinner sticks forever (S2 live defect; reproduced and
    # confirmed by disabling the fragment). Rendering last means a generating run ends
    # in st.rerun() before the fragment is ever created, so nothing is scheduled while
    # we are blocked. On an idle run we reach the end normally and the 2 s refresh
    # resumes. SLUICE_UI_LIVE_STATS=0 disables it entirely (kept as an escape hatch).
    _live = os.environ.get("SLUICE_UI_LIVE_STATS", "1") != "0"

    @st.fragment(run_every=2 if _live else None)
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

    stats_ph = st.empty()

    if st.button("Clear chat", icon=":material/mop:", width="stretch"):
        st.session_state.chat_log = []
        st.rerun()

# ---------------- chat ----------------
st.title("Chat", anchor=False)
st.caption(f"{choice} · {mem_choice} memory · {mode} quality · "
           f"streaming from a {meta['size'].split('·')[-1].strip()} file")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        if turn.get("thinking") and show_thinking:
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

# submit_mode="stop": while a reply streams, the send button becomes a Stop
# button that halts the script mid-generation (the script is blocked in the
# read loop, so this is the only way to interrupt from the browser).
typed = st.chat_input(f"Message {choice}…", submit_mode="stop")
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
        # if the previous reply was stopped mid-stream, the warm engine still
        # has an unfinished generation in its pipe - it's dirty, so restart it
        # clean before this request rather than reading stale tokens
        if st.session_state.get("gen_active"):
            _terminate(_server_slot()["proc"])
            _server_slot().update(proc=None, key=None)
            st.session_state.gen_active = False
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

        # E41b client contract: when KV canonicalization is on, the engine rewrites
        # its KV into the template's canonical form for the turn and tells us the
        # exact assistant text that form contains, via `canon_reply:`. We MUST echo
        # that string back, not our own extraction — anything else stops the KV from
        # being a prefix of the next render and the reuse silently evaporates (no
        # error, just a slow turn). `text` stays what the USER sees; `canon` is what
        # the ENGINE sees. Absent the flag no canon_reply is emitted, `canon` is
        # never set, and this is byte-identical to the previous behaviour.
        turns = ["%s\x1f%s" % (t["role"], clean(t.get("canon") or t["text"]))
                 for t in st.session_state.chat_log]
        proc.stdin.write("\x1e".join(turns).encode() + b"\n")
        proc.stdin.flush()
        st.session_state.gen_active = True  # if Stop is hit, this stays True -> clean restart next turn
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
                    if thinking and show_thinking:
                        with think_ph.container():
                            with st.expander("Thinking", icon=":material/psychology:",
                                             expanded=(not final)):
                                st.text(thinking)
                    elif thinking and not final:
                        # thinking hidden and no visible answer yet: without this the
                        # user stares at a blank pane for the whole reasoning phase,
                        # which at High is the longest part of the wait
                        with think_ph.container():
                            st.caption(":material/psychology: Thinking…")
                    answer_ph.markdown((final + ("▌" if end < 0 else ""))
                                       if (final or not thinking) else "")
                elif raw and b"<<<END>>>" in buf:
                    # generation is finished but the engine is still canonicalising
                    # the KV (5-10 s). Without this the spinner reads as "stuck".
                    status.update(label="Finishing turn — preparing a fast next turn…",
                                  state="running")
                if b"<<<READY>>>" in buf:
                    done = True
            engine_died = proc.poll() is not None
            if engine_died:
                done = True
            if not chunk and not echunk:
                time.sleep(0.05)

        # engine crashed mid-request (e.g. D12 union>slots): surface the real
        # stderr reason instead of a misleading "no tokens", and clear the
        # dead handle so the next message starts a fresh engine
        if engine_died and not raw.strip():
            slot = _server_slot()
            slot.update(proc=None, key=None)
            status.update(label="Engine stopped mid-request", state="error")
            reason = [l for l in stderr_tail.splitlines()
                      if any(w in l.lower() for w in ("union", "slot", "exit", "error", "fail", "abort"))]
            st.error("The engine stopped before answering. Most likely cause:\n\n"
                     + ("\n".join(reason[-4:]) if reason else stderr_tail[-500:] or "no stderr captured")
                     + "\n\nTry CPU backend, or lower Memory / raise it so the cache fits the prompt.")
            st.session_state.chat_log.append(
                {"role": "assistant", "text": "*(engine stopped before answering — see the error above)*",
                 "thinking": "", "timing": ""})
            st.stop()

        total = time.time() - t0
        tail = buf.decode(errors="replace")
        met = parse_metrics(tail)
        capture_turn(raw, stderr_tail,
                     f"turn={len(st.session_state.chat_log)} total={total:.1f}s "
                     f"metrics={met} canon_env={os.environ.get('LLMSTREAM_KV_CANON')}")
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
        a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
        analysis = strip_harmony(a.group(1)) if a else ""
        if not final_clean:
            # No clean <|channel|>final|> wrapper. High margin (Fast) can derail
            # the harmony channel structure, so the model's real text ends up in
            # an unterminated analysis channel. Show that text — never claim
            # "no tokens" when the engine actually generated some.
            fallback = strip_harmony(raw)
            if fallback:
                # The model never emitted a final channel — it was still reasoning when
                # it stopped (token cap, or Fast mode derailing the channel structure).
                # Showing that text AS the answer misrepresents thinking as a reply,
                # which is what the live UI did. Say what happened and keep the text in
                # the expander where it belongs.
                final_clean, analysis = "", fallback
                st.warning(
                    ":material/warning: This reply has no final answer — the model was "
                    "still reasoning when it stopped. Its thinking is below. "
                    "Try again, raise **Max new tokens**, or switch quality to "
                    "**Balanced** if you are on Fast.")
            elif met.get("generated") == ("0",):
                final_clean = ("*(the model chose to stop immediately — try Balanced "
                               "quality, or rephrase your message)*")
            else:
                final_clean = "*(empty response)*"
        # E41b: the engine's canonical assistant text, if it canonicalized the KV.
        canon = parse_canon_reply(tail)
        if canon is not None and canon.strip() != final_clean.strip():
            # NEVER silent. A mismatch means the next turn will re-prefill the whole
            # history and the only symptom the user would see is "it got slow" —
            # which is exactly the class of bug that cost us a day on E38.
            st.warning(
                ":material/warning: KV-canon mismatch — the reply we stored differs "
                "from the engine's canonical form, so the next turn will re-prefill "
                "the whole conversation instead of reusing it. Speed only; the answer "
                "is unaffected.\n\n"
                f"- stored (ours): `{final_clean.strip()[:120]}…` ({len(final_clean.strip())} chars)\n"
                f"- engine canon: `{canon.strip()[:120]}…` ({len(canon.strip())} chars)")
        st.session_state.gen_active = False  # completed normally - engine is clean
        turn = {"role": "assistant", "text": final_clean,
                "thinking": analysis, "timing": timing}
        if canon is not None:
            turn["canon"] = canon
        st.session_state.chat_log.append(turn)
        st.rerun()

# ---------------- live stats, rendered LAST ----------------
# A generating run exits via st.rerun() above and never reaches this line, so the
# run_every fragment is never scheduled while the script is blocked. See the comment
# at its definition.
with stats_ph.container(border=True):
    live_stats()
