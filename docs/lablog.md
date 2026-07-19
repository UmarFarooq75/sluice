# Lab log — chronological record of every change and what happened

Rule: every experiment gets an entry the same day — the change, the expectation,
the measured result (with artifact path), and the conclusion or open question.
Negative results and self-inflicted failures are logged with the same weight as
wins; they steer the roadmap just as hard. Thematic writeups live in
`docs/findings-phase0.md` / `docs/research-findings.md`; this file is the raw
"we changed X → Y happened" timeline.

Format per entry:
- **Change/Experiment** — what we did
- **Expected** — prediction beforehand (forces honesty about surprises)
- **Result** — measured numbers + artifact
- **Conclusion / next question**

---

## Phase 0 (2026-07-17 → 07-18, backfill pointer)

OLMoE tracing → cache simulation → physical streamer PoC (experts as SSD
extents, page cache bypassed): resident 76.7 tok/s vs all-SSD floor 3.5 tok/s
vs half-cache+margin 16.2 tok/s, NLL 3.002 bit-consistent in all three
(`results/streamer_poc.json`). Qwen3.6-35B-A3B streamed at 8.3 tok/s where
Ollama v0.32.1 on the same machine swap-collapsed to ~0.03 tok/s (its own logs
archived). Full detail in the findings docs; this log starts fine-grained at
the gpt-oss-120b bring-up.

---

## 2026-07-18/19 — gpt-oss-120b (117B params) bring-up on the 16GB M2 Air

### E1. gpt-oss family support in fork + driver
- **Change**: openai-moe adapter with slot-backed experts; two firsts for the
  engine: per-expert *bias* tensors (2D `{n, n_expert}` extents striding on
  `nb[1]`) and raw-logit gating (SOFTMAX_WEIGHT), which forced the margin mask
  to write `-INFINITY` instead of `0` (0 is not a floor in logit space).
- **Expected**: no behavior change for existing families.
- **Result**: OLMoE gate PASS, hash `b6869f5b6ef36376` bit-identical across
  resident/slots32/slots12. Commits: fork `7d62a14`, repo `28636d7`.
- **Conclusion**: the slot mechanism generalizes across gating families and
  tensor layouts with a ~1-file adapter. Universality evidence, not proof —
  next family (DeepSeek-style shared experts) is M3.

### E2. Model download
- 63,387,346,208 bytes, byte-exact vs HuggingFace `x-linked-size`. MXFP4
  experts are native to *every* published gpt-oss-120b GGUF (~63GB; no smaller
  quant exists — experts are already ~4.25 bit).

### E3. First streamed run — SLOTS=12, margin 0 (exact routing)
- **Expected**: 3–6 tok/s (estimate from OLMoE/Qwen scaling).
- **Result**: **0.75 tok/s decode**, 0.73 prefill; hit 52.6%; 64.0GB read at
  713 MB/s; stall 50.5s of 76s; text fully coherent and correct
  (`results/gptoss_m0_slots12.txt`).
- **Conclusion**: a 117B model runs, quality-anchored, on a 16GB laptop — but
  4–8× below estimate. Why: 68 misses/token × 13.25MB ≈ 900MB/token; entirely
  I/O-bound; estimate had assumed OLMoE-like hit rates, gpt-oss's 128-expert
  top-4 routing has far less temporal locality at 12 slots. Estimation error
  logged: never scale hit rates across architectures.

### E4. SELF-INFLICTED: ladder loop bug → machine hang + OOM kill
- **Change**: benchmark ladder launched via a `for … set -- $cfg` loop.
- **What happened**: the harness shell is zsh; zsh does not word-split `$cfg`,
  so `LLMSTREAM_SLOTS` came through **empty** → runs fell back toward
  resident-loading a 63GB model → swap hit 6.6GB, the laptop hung repeatedly,
  macOS SIGKILLed rung 1 (exit 137). Two further stray `stream_run` processes
  (one duplicate launch, one orphan from a crashed session) compounded RAM
  pressure earlier in the day.
- **Fixes**: ladder moved into a `#!/bin/bash` script; `run()` now aborts on
  empty slots/margin; one-process-at-a-time rule; bad artifacts deleted.
- **Deeper conclusion (product defect, not script defect)**: the engine
  trusted its configuration blindly. A product that streams huge models MUST
  protect the host machine itself → E10 (engine-level pressure guard).

### E5. First ladder — margin sweep at slots12 + RAM curve (CONTAMINATED)
- **Result**: m0.02 → 0.55 tok/s (hit .533); m0.5 → 0.48 (hit .704);
  m1.0 → 0.73 (hit .877); m2.0 → **10.47 tok/s** (hit .998);
  slots16 m1 → 0.19 tok/s (48 MB/s effective bandwidth); slots8 m1 → 2.99.
- **Audit verdict**: the slots12 margin rungs and slots16 ran minutes after
  the E4 swap blowup — their effective bandwidth (216–521 MB/s vs 1128–1161
  clean) shows page-fault contamination. The m1.0-slots12 run has ~58s of
  decode time that is neither stall nor compute — swapped-out resident pages.
  **"Margin doesn't help at slots12" is UNPROVEN; re-run required.** The
  slots16 collapse mechanism (cache + weights + KV exceeding headroom) is
  directionally real; magnitude unquotable.
- **Two numbers that survive**: 10.47 tok/s = the CPU compute ceiling (99.8%
  hit ⇒ I/O hidden; ran fast despite elevated swap, so ceiling is ≥ this);
  and slots8's 2.99 with full SSD bandwidth.

### E6. m=2.0 is fool's gold — quality collapse
- **Result**: text degenerates to "the the" + newline flood
  (`results/gptoss_m2_slots12.txt`). At 12 slots, margin 2.0 pins routing to
  the cached set ~always (hit .998) — effectively a fixed 12-expert model.
- **Conclusion**: speed numbers are meaningless without a quality metric
  attached. Never report a margin rung without fidelity → E9.

### E7. Cliff hunt at the slots8/10 sweet spot (clean system)
- **Result** (all coherent unless noted):
  | config | tok/s | hit | text |
  |---|---|---|---|
  | slots8 m1.0 | 2.99 | .792 | clean |
  | slots8 m1.25 | 5.11 | .885 | clean |
  | slots8 m1.5 | 8.25 | .958 | answer correct, then newline flood — termination breaks first |
  | slots10 m1.0 | 3.78 | .837 | clean |
  | slots10 m1.25 | **5.72** | .922 | clean |
- **Conclusions**: (a) quality-intact best = 5.72 tok/s; (b) the governing
  formula holds quantitatively: time/token ≈ misses×13.25MB÷1.15GB/s + 0.095s
  — predicts all clean rungs within ~15%; (c) degradation onsets at the
  *end-of-answer* boundary (termination/control routing leaves the topic-local
  cached set) — margin cliff for this model sits between 1.25 and 1.5 at 8
  slots; (d) smaller cache beat bigger cache twice (slots8 > slots12 > slots16)
  because memory headroom governs effective SSD bandwidth — on a 16GB machine
  the cache must be sized to *pressure*, not to "more is better".

### E8. Methodology error found in our own quality reads
- gpt-oss is harmony-chat-format trained; every run used a raw text prompt.
  Post-answer fragments ("Sure!", "I'm sorry…") are template mismatch, not
  necessarily margin damage. All cross-rung *eyeball* quality comparisons to
  date are confounded; speed/hit numbers unaffected. Fix: `LLMSTREAM_CHAT=1`
  wraps prompts in the model's own template (E9 build).

### E9. Instrumentation build (turn quality & I/O into numbers)
- **Change**: (1) router-agreement counter — per token, true pre-mask top-k vs
  post-mask selection, exact overlap %; (2) `read_work`/`preads` — summed
  worker pread time vs eval-thread stall separates device latency from
  queueing; (3) `LLMSTREAM_NLL=1` — teacher-forced mean −log p over a fixed
  passage, deterministic quality number per config; (4) `LLMSTREAM_CHAT=1`.
- **Result**: OLMoE gate PASS unchanged (`b6869f5b6ef36376`) — counters cost
  nothing at m=0. Commit `d45da99`.
- **Result (agreement, generation runs)**: router agreement at slots8 is
  **0.57 @ m=1.0, 0.44 @ m=1.25, 0.35 @ m=1.5** (slots10 m1.25: 0.45) — the
  majority of expert picks differ from true routing at every useful margin,
  yet generated text reads clean through m=1.25. Coherent text survives
  massive substitution.
- **Result (NLL, teacher-forced on fixed Dickens passage, slots8)**:
  m=0 anchor **avg_nll 2.845 (ppl 17.2)**; m=1.0 **3.534 (ppl 34.3)** — a 2×
  perplexity hit that generated text completely hid. Remaining margins
  running.
- **Interpretation (hypothesis, testable)**: margin routing biases the model
  toward what its cached experts can do; free-running generation then *steers
  into its own comfort zone*, so transcripts look clean while the underlying
  distribution is measurably off. Teacher forcing removes the steering and
  exposes it. Consequence: NLL (not transcript reading) is the quality gate
  from now on.
- **Sweep bug found**: first sweep aborted (SIGABRT) after the m=0 NLL rung —
  the NLL early-return skipped I/O-pool thread shutdown (std::thread dtor on
  joinable thread). Fixed with shared cleanup; gate re-PASS.
- **Protocol gap noted**: NLL scores include the cold-start window (cache
  empty → masking against an irrelevant resident set for the first tokens).
  Refinement queued: score only the passage's second half, or prepend warmup
  text, to separate cold-start damage from steady-state damage.
- **Full NLL curve (slots8, Dickens passage, cold-start included)**:
  | margin | avg_nll | ppl | agreement |
  |---|---|---|---|
  | 0 | 2.845 | 17.2 | 1.000 |
  | 1.0 | 3.534 | 34.3 | 0.574 |
  | 1.25 | 3.663 | 39.0 | 0.488 |
  | 1.5 | 3.781 | 43.9 | 0.409 |
  | 2.0 | 4.287 | 72.7 | 0.302 |
  Conclusions: (a) degradation is SMOOTH — the generation "cliff" at m≥1.5 was
  a termination artifact, not a knowledge cliff; (b) m=2.0 spikes 4× — the
  metric catches the known-broken config, validating itself; (c) NLL tracks
  router agreement ≈linearly → agreement is a free live quality proxy the
  engine can watch at runtime; (d) margin is a real quality dial, so the
  no-compromise speed path is Metal + prefetch + I/O engineering at low
  margin, not bigger margins. 80–100 tok/s physics note: gpt-oss-120b moves
  ~2.7GB of active weights per token → ~37 tok/s is the M2 Air's 100GB/s bus
  ceiling even fully resident; 80–100 on this machine is a ~1B-active-model
  target, and a Max-class bus (300–400GB/s) target for the 120B. The engine's
  job on any machine: reach that machine's ceiling.

### E10. Engine-level machine protection
- **Change**: `LLMSTREAM_SLOTS=auto` — size the cache from *this machine's
  available memory* read from the OS at startup, not from the caller's guess;
  plus a runtime pressure monitor that sheds cache slots (MADV_FREE's their
  pages) when macOS signals memory pressure, and grows back when calm.
- **Why**: E4/E5 proved the host must never be collateral damage. Ollama-class
  engines avoid this by refusing to load; we run the model AND stay polite.
- **Result — build**: OLMoE gate PASS unchanged (`b6869f5b6ef36376`); commit
  `2184673` (also fixes the E9 NLL-path thread-shutdown abort).
- **Result — stress test 1** (`results/gptoss_guard_{baseline,stress}.txt`):
  auto picked slots=7 (9.6GB avail, 2.3GB resident) / slots=6 under load.
  5GB hog attacking mid-generation: run completed (96 tok, 4.18 tok/s, hit
  .837), **swap growth 0.1MB**, machine responsive throughout. Per-token speed
  unchanged vs no-hog baseline. Prevention (headroom by construction) PASSED.
- **Expected (written before result)**: reactive path — an 8GB hog should
  push memorystatus to warning/critical; guard should log cap drops within
  ~2s, shed slots via MADV_FREE, run completes slower but alive, swap bounded.
- **Result — stress test 2 (8GB hog)**: run survived (96 tok, 3.04 tok/s at
  auto slots=5, swap flat) but the guard never fired — and the reason is an
  experiment bug, not a guard bug: the hog allocated zero-filled pages, and
  the macOS memory compressor absorbs those ~100:1. A compressible attack is
  a fake attack. Lesson for the protocol: memory-pressure experiments must use
  incompressible (random) pages, or they measure the compressor, not the
  pressure path (`results/gptoss_guard_stress8.txt`).
- **Result — stress test 3 (8GB incompressible hog)**: run survived real
  pressure (96 tok, 2.31 tok/s at auto slots=4; swap **grew +1.5GB** — attack
  landed), machine stayed usable. Guard STILL silent. Two root causes found
  by questioning: (1) **the monitor thread started after model load**, and the
  hog's whole life fit inside the ~2-min load window — the guard was born
  after the war; (2) memorystatus stayed at "normal" while swap grew 1.5GB —
  the jetsam signal is sluggish; one opaque OS signal is not enough
  (`results/gptoss_guard_stress8r.txt`).
- **Fix (guard v2)**: monitor starts BEFORE model load; dual trigger — shed
  on memorystatus ≥ warning OR available memory < floor (1.8GB warn / 1.2GB
  severe, `LLMSTREAM_GUARD_FLOOR` tunable). Gate re-PASS `b6869f5b6ef36376`.
- **Result — stress test 4 (hog timed at t+160s, guard v2)**: survived (256
  tok), but decode finished before the timer — because of a NEW positive
  finding: **long generations warm the cache** (hit .927 over 256 tok vs .885
  over 64; 6.96 tok/s at only 6 slots). Short benchmarks undersell the
  engine; sustained sessions run faster. Timing-based attack design failed
  twice → switched to event-triggered.
- **Result — stress test 5 (event-triggered hog, guard v2)**: **GUARD FIRED**:
  `pressure lvl=2 avail=3.5GB -> slot cap 4`, pressure_drops=1, run completed
  512 tok at 6.37 tok/s, swap bounded, cap recovered to 5 after calm.
  Detection → response → recovery proven under real memorystatus pressure.
  Residual gap: cap_evictions=0 — the artifact "dots" that triggered the hog
  are model-LOAD progress dots (llama.cpp progress callback), not decode
  tokens; pressure hit while the cache was still empty, so there was nothing
  to shed, and the cap recovered before decode. Warm-cache shedding still
  unexercised (`results/gptoss_guard_stress5.txt`).
- **Result — forced-floor test (`LLMSTREAM_GUARD_FLOOR=20`)**: guard fired
  pre-load ⇒ cache grew up UNDER the cap (assignment-blocking enforced it) ⇒
  evictions correctly 0 — proved capping-at-birth, still not warm shedding.
  Added decode-relative fault injection (`LLMSTREAM_GUARD_TEST_AT=N` seconds
  after first decode token; process-relative timers lost the race twice —
  load time varies 60–120s with OS page-cache warmth).
- **Result — injection test (decode+8s, warm 8-slot cache): E10 CLOSED.**
  `cap_evictions=144` — exactly 36 layers × 4 shed slots, predicted in
  advance; cap 8→4, hit adapted .92→.781, decode continued at 3.93 tok/s,
  text coherent, final_cap=5 (recovery), gate PASS. Full chain proven:
  prevention (auto-sizing; 3 real attacks survived, swap bounded) →
  detection (real memorystatus lvl=2 in test 5; avail-floor as backstop) →
  shedding (144 evictions + MADV_FREE) → recovery (+1 slot per 30s calm) →
  service continuity (coherent output throughout)
  (`results/gptoss_guard_inject.txt`). Product claim now measured: **this
  engine cannot hang the host; under pressure it sheds its own memory, keeps
  generating, and heals afterward.**
- **Open question**: auto chose 7 slots where manual best was 8 — the safety
  factor costs ~10-20% speed. Tune the 0.80/2GB constants only with more
  cross-model data, never to zero headroom (that's how engines hang laptops).

### E11. Metal bring-up: resident PROVEN, naive hybrid REFUTED (2026-07-19)
- **Change**: built the fork with GGML_METAL in a separate `build-metal` tree
  (CPU build untouched as gate reference); `LLMSTREAM_NGL` env for GPU layer
  offload (default 0 = CPU, gate unaffected — re-PASS `b6869f5b6ef36376`);
  driver variant `stream_run_metal`.
- **Expected**: resident OLMoE faster than CPU's 73.9 tok/s; hybrid (GPU
  graph + CPU slot tensors) either works or fails loudly.
- **Result — resident**: **84.12 tok/s** on Metal, text identical opening to
  CPU reference — first GPU numbers, backend validated
  (`results/olmoe_metal_resident.txt`).
- **Result — hybrid**: ran "cleanly" (exit 0, hit .974, sane counters,
  12.08 tok/s) but **output is garbage** (`** ** ** …`) — silent numerical
  corruption, not slowness. Root-cause hypothesis: the cb_eval id-rewrite
  mutates the ids tensor via host pointer; with a Metal graph the scheduler's
  copy/ownership of that buffer races or ignores the host write. Naive hybrid
  is therefore both slower (~16×2 GPU↔CPU crossings/token) AND wrong
  (`results/olmoe_metal_slots32.txt`).
- **Conclusions**: (a) counters cannot certify correctness — only output
  checks (text/NLL/hash) can; the protocol already demands this and it just
  paid off; (b) the road to GPU streaming is a **Metal slot backend**: slot
  tensors in shared MTLBuffers (unified memory), id-rewrite before command
  submission, explicit sync — an engineering project, now scoped by a
  measured failure; (c) until then the production config is pure-CPU
  streaming (5.7 clean / ~7 warm tok/s on gpt-oss-120b).
- **Open experiment**: gpt-oss hybrid unmeasured; pointless until the
  correctness seam is fixed — deferred, not forgotten.

### E13. Warm-split NLL — hypothesis REVERSED, at-anchor point found
- **Expected (written in E9)**: cold-start masking overstates margin damage;
  warm halves should look better.
- **Result**: the opposite. Warm-half ppl deltas vs the anchor's own warm
  half (31.7): **m0.25 +5%, m0.5 +34%, m1.0 +158%, m1.25 +217%** — larger
  than the averaged deltas. Mechanism: while the cache is too cold to
  restrict, the mask is SKIPPED (`res.size() < top_k` guard), so early tokens
  are exact — cold-start was DILUTING the damage, not causing it. Margin
  damage concentrates exactly where sessions live: steady state.
- **Conclusion**: the no-compromise operating point is **margin 0.25 at
  slots8** — avg NLL 2.859 vs anchor 2.845 (+0.5%), warm +5%, agreement
  91.6%. Above it, steady-state quality pays more than average-NLL suggested.
  Speed at that point comes from prefetch/Metal/prefill engineering, not from
  routing substitution. Artifacts: `results/gptoss_nll_*_slots8*.txt`.
- **Protocol note**: batch-2 gen rungs died on a bash quirk (`${5:+VAR=1}`
  expands to a word, not an assignment → exit 127) — rerun as batch2b with
  `env`; second shell-portability incident this session (see E4).

### E14. GPU streaming lands: OLMoE bit-exact, then gpt-oss-120b on Metal
- **OLMoE proof**: `SLOT_DEV` first silently no-opped — device is named `MTL0`
  not "Metal", and the fallback WARN was invisible at the driver's ERROR-only
  log filter (both fixed: `gpu` alias picks the first GPU-type device from
  the registry — portable to CUDA later; log filter now WARN+, INFO under
  LLMSTREAM_VERBOSE). With it engaged: **GPU-streamed OLMoE logits_hash
  `7ef6c26bdadc0cbd` = bit-identical to GPU-resident**, 39.09 tok/s at
  half-cache (32/64 slots). E12's snapshot-staleness diagnosis confirmed by
  the strongest possible evidence (`results/olmoe_gpuslots.txt`).
- **gpt-oss OOM wall**: first GPU attempt aborted — verbose buffers showed
  `MTL0_Mapped model buffer size = 60438 MiB` vs 12.7GB working set: with
  mmap, the GPU path maps the ENTIRE file as one device buffer; skipped
  experts cost nothing as tensors but everything as mapping. Fix:
  `use_mmap=false` whenever SLOT_DEV is set → only loaded tensors allocate
  (~2.9GB weights + 3.8GB slot cache). OLMoE never hit this (whole file <
  working set).
- **Milestone**: **gpt-oss-120b (117B) ran on the M2 GPU with streamed
  experts, exact routing, correct text** (matches CPU anchor opening
  word-for-word): decode 1.30 tok/s at hit .443, slots8, m=0 — 1.7× the CPU
  exact-mode figure before any tuning (`results/gptoss_gpu_m0.txt`).
- **Known defect**: post-output SIGABRT at teardown (Metal buffer free
  ordering in stream_state dtor path) — does not taint measurements; fix
  queued. GPU ladder (m0.25/m1.25/slots16) running.

### E16. Clean GPU lifecycle + where GPU streaming actually pays
- **Teardown abort FIXED**: root cause was our never-freed device slot buffer
  vs Metal's registry destructor at exit; added `llmstream_free()` (fork API)
  called before exit — verified exit 0 on all subsequent GPU runs.
- **Device-budget auto-sizing VERIFIED**: `SLOTS=auto` + `SLOT_DEV=gpu` now
  caps by `ggml_backend_dev_memory` — picked 7 slots, ran with ZERO guard
  interventions, clean exit. Detect→size→run→exit works end-to-end.
- **Sync-gating (look nodes only hooked while prefetch live)**: no effect at
  m0.25 (1.37 tok/s unchanged) — measured reason: 75 misses/token ≈ 990MB ≈
  0.73s I/O dwarfs ~15ms of syncs. Sync work only pays in high-hit regimes.
- **Prefetch is family-dependent**: OLMoE GPU-streamed 39.09 tok/s with
  prefetch vs 11.06 without (hit .974 vs .828) — a 3.5× win on the 64-expert
  model, the same mechanism that LOSES on 128-expert gpt-oss (E-batch2b:
  saturates SSD with waste). Follows phase-0 lookahead recall (.839 OLMoE).
  Engine should autotune prefetch per family from measured recall.
- **16GB strategic conclusion**: gpt-oss GPU path pays a ~2.9GB no-mmap
  weight tax that the CPU path's mmap avoids → the guard trims slots →
  CPU streaming stays the production config for the 120B on this machine
  (5.4–7 tok/s). GPU streaming = correct everywhere, wins on bigger-memory
  machines, and already frees all CPU cores (responsiveness win) for
  small-model serving at 39 tok/s.

### E15. Controlled CPU-vs-GPU table; guard's first real-world saves
- **Puzzle**: GPU ladder showed hit .573 at m125_s8 (CPU: .885) and slots16
  hitting LESS than slots8 (.320 vs .482). Controls run with identical
  prompt/binary/configs on CPU.
- **Resolution (from the GPU runs' stderr)**: the pressure guard fired in
  BOTH odd runs — six drops walking 16→4 slots (avail 1.3–1.6GB, floor
  trigger; 278 live evictions incl. MADV_FREE on Metal-shared pages) and one
  critical-level drop to 4 at m125. First NON-injected activations: real
  Metal memory pressure (no-mmap full weights + device cache), machine never
  hung, runs completed. Dual trigger vindicated — memorystatus stayed at
  lvl=1 while the avail-floor did the work.
- **Controlled results**: m025_s8 unpressured on both backends — CPU/GPU
  hit .484/.482, agreement .9535/.9550: **margin path is backend-identical**.
  At equal conditions GPU streaming is ~20% SLOWER than CPU today
  (1.37 vs 1.71 tok/s) — per-hooked-node sync cost; OLMoE resident (84 GPU vs
  74 CPU) shows the ceiling once syncs are batched. "Steering-created
  locality beats bigger caches" is REFUTED as stated — the slots16 deficit
  was the guard, not physics.
- **Artifacts**: `results/gptoss_{cpu,gpu}_m025_s8.txt`, `_m125_s8`,
  `_m025_s16` + .err files.
- **Open**: (a) sync batching for GPU streaming; (b) guard/Metal interplay:
  auto-sizing should subtract the device cache from the budget up front so
  16-slot GPU configs aren't attempted on 16GB machines; (c) teardown abort.

### E12. Metal slot buffers — the corruption root cause, found in scheduler source
- **Investigation**: read `ggml-backend.cpp` sched execution. Two facts:
  (1) cross-backend split inputs are **snapshot-copied before the split runs**
  (`ggml_backend_tensor_copy(input, input_cpy)`); (2) the sched synchronizes
  the backend before every `ask=false` callback. Our architecture fills slot
  tensors **mid-graph** (demand fetch inside the callback) — legal when
  everything is one backend (no copies), broken the moment a snapshot exists:
  the GPU computes from pre-fetch stale slot data. Explains E11's corrupt
  hybrid output with perfect counters, and the per-hooked-node sync explains
  its slowness.
- **Fix implemented (fork)**: `LLMSTREAM_SLOT_DEV=<device>` — allocate slot
  tensors in that backend's buffer type (Metal = host-visible shared memory
  on Apple Silicon: pread fills the same pages the GPU reads; no snapshot, no
  staleness). Device-agnostic via the backend registry (CUDA later needs a
  staging path — host-visible only for now).
- **Expected**: OLMoE NGL=99 + SLOT_DEV=Metal + slots32 produces text matching
  the Metal-resident run; then gpt-oss m=0 exact anchor on GPU, then speed.
- **Result**: pending (build blocked until batch-2 rungs finish — never swap
  dylibs under a running measurement).

---

## Defect ledger — every known defect, questioned to root cause

| # | Defect / symptom | Questioning → root cause | Status / experiment |
|---|---|---|---|
| D1 | Laptop hung during runs | Why did RAM run out? Runs launched unbounded (shell bug) + duplicates + slots16 over headroom. Why could they take the machine down? **Engine has no self-protection.** | Script guards done; engine guard = E10 |
| D2 | Prefill 0.7–2.9 tok/s → minutes of TTFT on real prompts | Why so slow? Prefill streams experts per token serially, same as decode. Why? Expert-major prefill (measured 16–599× on OLMoE) not yet ported to gpt-oss. | Next after E10 |
| D3 | Margin units are per-family (0.02 meant something on OLMoE probs; no-op on gpt-oss logits) | Why? Mask threshold lives in the gating score's native scale. A universal product cannot ask users to know gating families. | Design: normalized margin (percentile of per-token score gaps) or entropy-adaptive margin. Experiment queued |
| D4 | Quality was judged by eyeball | Why unacceptable? m1.5 rung *looks* fine until the last tokens; template mismatch (E8) fools the eye both ways. | Fixed by E9 (agreement + NLL); sweep running |
| D5 | At high hit rates, stall shrinks but decode doesn't reach ceiling | Where does the time go? Not device reads (read_work now measured), suspicion: fetch-path serialization / page faults / repack. | Decompose with E9 counters on clean re-runs |
| D6 | All results = one English prompt, 64 tokens | Why risky? Routing locality is topic-dependent; margin cliff may move on code/multilingual/long-context. | Prompt battery + 1k-context runs queued |
| D7 | prefetch_issued=0 in every gpt-oss run | Why off? We disabled it for first runs (phase-0 rule: prefetch pays only below hit-EMA 0.80). At slots8 m1.25 hit=.885 → gate would keep it off. Is the 0.80 gate right for THIS regime? | Re-test: force prefetch on at slots8 m1.25 |
| D8 | Raw prompts to a chat-trained model | See E8. | `LLMSTREAM_CHAT=1` built; use in all future quality runs |
| D9 | Universality boundary: dense models | A dense 72B touches all weights every token — nothing to stream selectively; that is physics, not engineering (docs/dense-strategy.md). Scope stated honestly: MoE-first. | M3: DeepSeek family next (shared+routed experts) |
| D10 | Thermal never logged (fanless chassis) | Are late rungs slower because hot? Counter-evidence: fastest rungs ran last. Still unproven either way. | Add powermetrics logging to protocol |

## Standing protocol (enforced from 2026-07-19)
1. One model process at a time; check for strays before launch.
2. No rung starts under memory pressure (script guard + engine guard).
3. Artifacts labeled clean/contaminated; contaminated numbers never quoted.
4. Bit-exact OLMoE gate before every commit touching fork or driver.
5. Every experiment → lablog entry same day, prediction written before result.
6. Quality claims only with agreement/NLL attached (eyeball is a smell test).
