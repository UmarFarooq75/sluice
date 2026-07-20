# sluice — second-machine validation

Thanks for testing. This takes ~30–45 min (most of it a one-time build + a model
download). You'll produce two results we cannot get on the original 16 GB machine,
and they settle open questions in our lab log (E31).

## Why your machine matters

We developed sluice on a **16 GB M2 Air**. Two things are impossible there but
easy on a bigger Mac, and they're exactly what we need you to measure:

1. **Bit-exact proof on a 35B model.** sluice streams MoE experts from disk to run
   models bigger than RAM. We prove it's *faithful* by comparing streamed output
   to the same model held fully resident — bit for bit. On 16 GB we can't hold a
   35B model resident, so that proof has a hole. **If you have ≥ 32 GB RAM, the
   model fits resident on your machine and you can close it.**
2. **The residency speed prediction (E31).** We predict Qwen3.6-35B-A3B runs at
   **10–13 tok/s fully resident** on a 32 GB machine (measured 8.3 tok/s *streamed*
   on our 16 GB). Your number grades that prediction.

## Requirements

- **Apple Silicon Mac (M1/M2/M3/M4), macOS.** The engine uses macOS-only APIs
  (`F_NOCACHE` direct reads, Metal, `NSProcessInfo`); it will **not** build/run on
  Linux/Windows/Intel.
- **≥ 32 GB RAM strongly preferred** (16 GB works but only reproduces our numbers;
  32/64 GB is where the new data is).
- **~40 GB free disk** (26.5 GB model + build).
- Xcode command-line tools, `cmake`, `git`, and `huggingface-cli`
  (`pip install huggingface_hub`). Install CLI tools with `xcode-select --install`.

---

## Step 1 — Clone and build the streaming llama.cpp (one time, ~10 min)

```bash
git clone https://github.com/UmarFarooq75/sluice.git
cd sluice

# build the vendored llama.cpp fork with Metal
cmake -B vendor/llama.cpp/build -S vendor/llama.cpp \
      -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build --config Release -j
```

## Step 2 — Build the sluice driver

```bash
bash scripts/build_driver.sh
# expect: "csrc/stream_run built"
```

## Step 3 — Download the test model (26.5 GB)

This is the high-total / low-active MoE that our whole thesis is about (35B total,
only 3B active per token):

```bash
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "*UD-Q5_K_M*" --local-dir models/qwen36
# then point a stable path at it:
ln -sf "$(ls models/qwen36/*UD-Q5_K_M*.gguf | head -1)" models/qwen36.gguf
```

If that exact repo/file name has changed, any `Q5_K_M` GGUF of **Qwen3.6-35B-A3B**
(or Qwen3-30B-A3B) is fine — just update the path below.

## Step 4 — Record your machine

```bash
echo "chip: $(sysctl -n machdep.cpu.brand_string)"
echo "RAM:  $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
system_profiler SPHardwareDataType | grep -Ei "chip|memory"
```

Note your chip's memory bandwidth (M2 ~100, M2 Pro ~200, M2/M3 Max ~400 GB/s) —
it sets the speed ceiling.

---

## Step 5 — The bit-exact gate (the proof only you can run)

This runs the model **resident** (all experts in RAM) and **streamed** (small
expert cache, rest from disk) and checks the output logits are **bit-identical**.
Needs the model to fit resident, i.e. ≥ 32 GB RAM.

```bash
GGUF=models/qwen36.gguf N_GEN=24 bash scripts/m1_gate.sh
```

**Report:** the final line — `GATE PASS` (with the shared hash) or `GATE FAIL`.
A PASS on your machine is the first bit-exact streamed-vs-resident proof of sluice
on a 35B model. That's the headline result.

## Step 6 — The speed benchmark (grades the E31 prediction)

Sweep the expert-cache size and record decode tok/s + hit rate. `slots` is experts
cached per layer; higher = more RAM, fewer disk reads.

```bash
MODEL=models/qwen36.gguf
PROMPT="Explain quicksort, then write a Python web scraper, then describe the French Revolution in detail."
for S in 16 32 64 128; do
  echo "== slots=$S =="
  LLMSTREAM_SLOTS=$S LLMSTREAM_MARGIN=0 \
    ./csrc/stream_run "$MODEL" 128 "$PROMPT" 1 2>&1 \
    | grep -E "prefill:|decode:|hit|maximum resident|logits_hash"
done
```

- `LLMSTREAM_MARGIN=0` = exact routing (zero quality change — the honest number).
- On a 32 GB+ machine, a high `slots` value effectively makes the model resident;
  that decode tok/s is the number to compare against our predicted **10–13**.

Optional — measure peak RAM per config by prefixing `/usr/bin/time -l` and reading
`maximum resident set size` (bytes).

---

## What to send back

Copy this filled in:

```
Machine:  <chip>, <RAM> GB, <bandwidth if known> GB/s
Build:    llama.cpp OK? [y/n]   driver OK? [y/n]
Model:    <exact GGUF file used>

GATE (step 5):   PASS / FAIL   hash: __________

SPEED (step 6), decode tok/s | hit | peak RAM (GB):
  slots=16:   __ tok/s | __ | __ GB
  slots=32:   __ tok/s | __ | __ GB
  slots=64:   __ tok/s | __ | __ GB
  slots=128:  __ tok/s | __ | __ GB

Anything weird (crashes, garbage output, machine froze): ______
```

Our pre-registered prediction (E31): resident/high-slots decode lands **10–13
tok/s** on a 32 GB machine — it *falsifies* our model if it comes in **< 8 or > 16**.
Either way the number is useful; send the real one.

## If something breaks

- **cmake can't find Metal / build fails:** make sure full Xcode (not just CLI
  tools) isn't required — `xcode-select --install` is usually enough; if Metal
  errors persist, add `-DGGML_METAL_EMBED_LIBRARY=ON` to the Step 1 cmake.
- **`stream_run` dylib error at runtime:** the driver rpaths
  `vendor/llama.cpp/build/bin`; run from the repo root so the relative path holds.
- **Garbage / repeated tokens:** add `LLMSTREAM_TEMP=0.8 LLMSTREAM_REP_PEN=1.1`
  before the command (sampling; exact-gate runs stay greedy on purpose).
- **Model won't load / unknown arch:** tell us the arch string it prints — the
  Qwen3-A3B family adapter is in the fork, but note anything unexpected.

Questions → ping the repo owner. Thank you!
