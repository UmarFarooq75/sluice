# streamlit_probe -- llmstream test bench

A one-file Streamlit UI for driving the llmstream engine binaries
(`csrc/stream_run` for CPU, `csrc/stream_run_metal` for Metal). Pick a
`.gguf`, set slots / margin / tokens / prompt, press Run, and watch the live
engine log plus parsed metrics grouped into the three project pillars:
SPEED (decode + prefill tok/s), QUALITY (router agreement, margin fidelity),
COMPUTE (expert-cache size, hit rate, peak RSS). A per-session history table
makes config A/B comparisons easy.

## Run

```sh
cd /Users/umarfarooq/Desktop/research && .venv/bin/pip install streamlit && .venv/bin/streamlit run examples/streamlit_probe/app.py
```

## Notes

- Models are discovered under `models/*.gguf` and `hf_home/hub/**/*.gguf`
  (use "Rescan model dirs" after downloading something new).
- Only one engine process is allowed at a time -- starting a new run or
  pressing "Kill run" terminates the previous process first (16 GB machine;
  two model processes would swap).
- Margin 0 is bit-exact; higher margins run faster at a measured quality cost
  (router agreement is reported when margin > 0).
- Optional: `.venv/bin/pip install psutil` enables the Peak RSS tile.
