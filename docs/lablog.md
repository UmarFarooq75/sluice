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

### E18. Adversarial audits (two subagents) + validation + fix batch
Two read-only subagents audited the project; every load-bearing claim was
then validated against artifacts before acting (owner directive: never just
believe a reviewer).

**Methodology audit — accepted after validation:**
- **Headline correction (their F7, validated against artifacts)**: all ≥5
  tok/s numbers are at m=1.25, which our own quality curve rejects; the
  quality-anchored (m0.25) speed is **1.63–1.83 tok/s**; "5.4" appeared in
  no artifact. Headlines must state the honest pair: ~1.7 quality-anchored /
  5.1–5.7 (single-run) at m1.25 with measured quality cost. auto-slots
  production config measured 3.84–4.18.
- **(their F2, validated by reading artifact tails)**: long guard runs at
  m1.25 degrade into repetition; "coherent output throughout" RETRACTED —
  machine survival proven, output-quality-under-pressure not. Re-run at
  m0.25+CHAT queued.
- **(their F1, accepted with a caveat I hold)**: m0.25 rests on one likely
  memorized passage; the promised domain battery was never run. My caveat:
  memorization does not invalidate the *differential* NLL across margins on
  the same passage — but it does make damage-detection sensitivity unknown,
  so the battery remains the top quality experiment.
- Also accepted: CHAT=1 built-but-never-used (closing now); Qwen headline
  has no quality metric; the OLMoE-only gate never covered gpt-oss's new
  code paths (slot-invariance hash pair now running); COMPUTE pillar has
  zero measured footprint artifacts (rss logging queued); thermal still
  unlogged; single-run noise ~5–24% vs config gaps ~12%.

**Code audit — validated, fix batch applied (build pending machine-free):**
- F1 [QUALITY, silent]: margin-hook name prefix also matches
  `ffn_moe_probs_biased/_masked` (DeepSeek-style) and misses split
  selection tensors (LLAMA4/GROVEMOE) → double-masking in mixed score
  spaces / NaN via -INF gather on families we haven't shipped yet. FIXED:
  exact `ffn_moe_probs-` match. Family guard for margin still TODO before M3.
- F2: NGL>0 + CPU slot tensors (documented-corrupt) was still accepted →
  now a hard startup error.
- F3: demand path indexed st->cache[il] unbounded; leading/mid dense layers
  (DeepSeek!) would corrupt memory → bounds check added.
- F4: guard floor (4) can sit below top_k+prefetch occupancy → exit(1) mid-
  run under pressure. FIXED: floor = top_k+1 once learned; demand path now
  waits for in-flight fetches instead of dying.
- F5: env values unclamped (IO_WORKERS=0 deadlocked) → clamped.
- F6: madvise returns ignored (device buffers may no-op → speed paid,
  nothing freed) → counted + reported as madv_fail.
- F7: in_decode data race → atomic. F9: look-node ask now gated by the same
  EMA as exec (removes per-layer GPU syncs bought for nothing). F10: pread
  EINTR retry + strerror.
- Verified-safe by the auditor (with reasoning): parts_left lifecycle,
  eviction-vs-fetch tearing, MADV_FREE ordering, cb_eval mutation model,
  fd sharing, extent arithmetic at 100GB scale.
- Their top speed ideas, queued with measurements attached: resident bias
  vectors (needs fork ids-dup split — biases must index ORIGINAL expert ids;
  ~159MB for -3 syscalls/miss), sub-extent read chunking (per-miss QD),
  prefetch gate rework (priority queue already protects demand).

### E19. Domain battery verdict: m0.25 holds across 5 domains; D11 found in the fallout (2026-07-19)
- **Question**: the m0.25 "no-compromise" claim rested on ONE prose passage
  (audit finding 1). Does it survive code / reasoning / chat / multilingual /
  fresh prose? Raw NLL (no template — assistant-loss-trained model is
  uncalibrated on user-turn text, measured ppl 2166–18k), slots8, warm-split,
  margins {0, 0.25, 0.5}, one fixed ~140-tok passage per domain. 15 rungs,
  run detached under `taskpolicy -b` after the machine-slowness report.
- **Prediction (written before results)**: m0.25 ≤ +2% NLL everywhere,
  m0.5 visible damage somewhere.
- **Result — full table (avg_nll / warm_nll / agreement)**:

  | domain | m0 warm_nll | m0.25 warm Δ | m0.5 warm Δ | agree m0.25 | agree m0.5 |
  |---|---|---|---|---|---|
  | code   | 1.771 | **+2.2%** | +0.6%* | .947 | .822 |
  | reason | 2.997 | **−3.0%** | **+9.7%** | .930 | .781 |
  | chat   | 6.059 | **−1.9%** | −4.6%† | .908 | .780 |
  | multi  | 2.264 | **+4.1%** | +2.9%† | .929 | .911 |
  | prose  | 3.890 | **+0.4%** | +2.3%† | .935 | .903 |

  († = guard-contaminated cell, see below. * = clean.)
- **m0.25 verdict: GREEN.** Warm ΔNLL spans −3.0%…+4.1%, mean **+0.35%**,
  MIXED SIGN — two domains improve, three degrade, none beyond ±4.1%. Single
  run per cell, so the honest statement is: at m0.25 the quality change is at
  or below the resolution of a single-run battery; there is no consistent
  degradation direction. Agreement ≥ .908 in every domain. Speed in these
  same runs: +4%…+15% tok/s over m0 (NLL mode); decode mode measured
  separately at +43% (1.05→1.50 tok/s, CHAT m0.25 s8). Prediction half-right:
  ≤2% was too tight (multi +4.1%), but no systematic damage.
- **m0.5 verdict: disqualified.** reason +9.7% warm NLL in a CLEAN cell —
  real damage on the domain users care most about, at agreement .781. And
  3 of 5 m0.5 cells are contaminated: REAL memory pressure (lvl 2, user
  actively working, avail 3.0–3.7GB) fired the E10 guard mid-rung
  (pressure_drops=2/4/3, cap 8→6→4/5, cap_evictions 144–211, madv_fail=0).
  Those cells measured "m0.5 at 4–6 slots", not m0.5@s8 — labeled and never
  quoted. chat m0.5 "improving" −4.6% under a cap of 4 is routing pinned to
  4 residents acting as a smoother on OOD-ish text; interesting, not usable.
  Second real-world guard save series, and NLLs stayed sane while capped.
- **Instrument caveat**: chat's raw ppl ~160–430 (assistant-trained model on
  user-style text) makes it the weakest of the 5 instruments; its direction
  agrees with the others, weight it least. Formula re-validated in passing:
  code m0 read 170,151.8 MB vs 12,838 misses × 13.25 MB = 170.1 GB exact.
- **D11, found by refusing to trust the contamination story**: chat m0.5 ran
  at cap **4** but the fix ledger claimed the floor was top_k+1=5 — the fix
  NEVER LANDED; code said `std::max(4, …)`. For gpt-oss (top-4) cap 4
  survives by pigeonhole (any token needing an absent expert has ≥1
  non-needed resident to evict — zero slack but livelock-free). For a top-8
  family (Qwen/OLMoE) cap 4 can never seat one token's experts:
  assign_slot → −1 forever → demand path exit(1) — the guard built to save
  the machine would KILL the inference. Fixed for real: `top_k` atomic
  (monitor reads it cross-thread), floor = top_k+1 once known, plus post-load
  re-clamp (a pre-load drop used floor 4 before top_k existed; undo it up to
  slot_cap_max — an explicit low --slots stays the user's choice). Gate:
  OLMoE 3-way logit-hash on the rebuilt binary (monitor-only change on the
  normal path; gate must still PASS bit-exact).
- **Product decision**: gpt-oss default = **m0.25** (battery-backed), exact
  m0 one env var away, m≥0.5 opt-in with the reasoning cost documented.
  Next: decode-ladder control on the fixed binary (speed-regression check on
  the 9 audit fixes + D11), then POLITE mode + RSS logging.

### E20. Adaptive margin (fidelity-targeted speed) + control ladder (2026-07-19)
- **Course correction (user)**: stop spending cycles on host-comfort features;
  the product answer to "it slowed my machine" is fewer bytes and fewer cycles
  per token, not politer contention. POLITE stays (10 lines, priced once in
  the control ladder below) but the main thread of work is speed at pinned
  quality from here on.
- **Change A (control)**: post-fix ladder on the exact banked configs.
  PASS bar: 3/3 logits hashes bit-identical to the pre-audit-fix bank, tok/s
  not below bank. First rung: m0_slots12 hash MATCH e609b48bba1688a3,
  0.75 → 1.17 tok/s (+56%, parallel part-fetch landed after the bank),
  first mem artifact: peak_rss 7.29GB / phys_footprint 7.15GB (COMPUTE
  pillar now measured per run, not estimated). Note: miss counts differ from
  bank (4134 vs 3889) at identical hashes — eviction skips in-flight victims,
  so I/O timing changes the cache trajectory without touching routing. Known
  nondeterminism, logits unaffected.
- **Change B (adaptive margin, D3 answer)**: LLMSTREAM_AGREE_TARGET=x sets a
  routing-fidelity floor in family-agnostic units (fraction of true top-k
  kept). Controller: every 180 measured router calls (~5 decode tokens),
  margin ×1.10 if window fidelity has >1% slack above target, ×0.80 if the
  floor is broken (quality recovers 2× faster than speed is gained), clamped
  [0.01, 8.0] (never 0: the probs hook stops being requested and the
  controller would go blind). Multiplicative = scale-free across gating
  families (logit-scale gpt-oss, prob-scale OLMoE/Qwen).
- **Prediction (written before the run)**: at target 0.93 on gpt-oss s8 the
  margin settles in the 0.25–0.6 band (battery: m0.25→agree .91–.95,
  m0.5→.78–.91) and decode lands between the fixed m0.25 and m0.5 rates;
  at target 0.95 it settles at or below 0.25. Risk to watch: controller
  oscillation between the asymmetric steps, visible as range >2× around the
  settle point.
- **Result (adaptive rungs 1-2)**: controller WORKS but under-exploits —
  target .93 and .95 both settled at margin 0.035 with agreement **0.9990**,
  leaving all the slack unspent. Root cause is arithmetic, confirmed exactly:
  13 windows × 1.10 growth from seed 0.01 = 0.035; 7%/step needs ~200 tokens
  to cross the dynamic range and the run had 64. Both runs identical
  trajectories (misses 4729=4729) — the controller is deterministic in this
  regime, good. Fix: slow-start (grow ×1.5 while slack >5 points, ×1.10 near
  target, cut ×0.8 unchanged); rebuilt, OLMoE gate PASS, rerun queued.
  Prediction for the rerun: settles 0.3-0.8 within ~30 tokens, agreement
  .92-.94, decode above the fixed-m0.25 rate.
- **Control ladder final** (4/4): no speed regression anywhere (m0_s12
  0.75→1.17, m002_s12 0.55→1.13, m125_s8 5.11→5.06 ≈ noise). Exact-mode hash
  bit-identical; m125 trajectory reproduced exactly (misses 1059=1059=1059
  across three runs). m002 hash DIVERGED from bank — root-caused, not a bug:
  at margin>0 the logits depend on the resident set, and eviction skips
  in-flight victims, so I/O timing races change the cache trajectory in
  miss-heavy regimes. **Margin mode is not run-reproducible by construction**;
  exact mode is. Product docs must say so. Control-gate PASS bar fixed to
  demand hash equality only where physics does.
- **POLITE priced**: m125_s8 identical misses+hash, 5.06 → 1.05 tok/s
  (**−79%**). The background I/O band throttles our own miss fetches.
  Opt-in niche flag only, never a default — the user's course-correction
  ("efficiency, not politeness") is now a measured fact.
- **First footprint artifacts**: peak_rss 7.29GB / phys 7.15GB at s12;
  phys 5.69GB at s8 — a 117B model in under 6GB physical.

### E21. The disk layout question: repack REFUTED, prefetch metric was wrong (2026-07-19)
- **Question A (layout)**: GGUF is type-major — one expert's 13.25MB = 3
  weight extents ~1.7GB apart + 3 bias slivers; every miss pays 6 scattered
  reads. Would an install-time expert-major repack (1 contiguous read/miss)
  turn misses sequential and win 3-10×? Priced on the real 60GB file,
  F_NOCACHE, 60 trials/pattern (scripts/ssd_layout_bench.py):

  | pattern | ms/expert | effective |
  |---|---|---|
  | scatter6 serial (old engine) | 9.91 | 1341 MB/s |
  | scatter6 parallel (engine today) | 8.32 | 1598 MB/s |
  | contig 13.25MB (repacked) | 8.92 | 1490 MB/s |
  | contig, QD4 (repacked, layer burst) | 7.43 | 1789 MB/s |
  | scatter6, QD4 (today, layer burst) | 7.54 | 1764 MB/s |

  **REFUTED: 1.5% at realistic queue depth.** This SSD does not punish
  4.4MB-granularity random reads; the seek penalty is amortized at that size.
  Kills TWO cards at once: expert-major repack AND sub-extent chunking
  (nothing to chunk toward — the device is at its ~1.6-1.8GB/s random
  envelope already). Weeks of `llmstream pull` repack plumbing avoided for
  one 2-minute measurement. Corollary: the engine miss path (E17's ~11ms,
  now ~8ms parallel) sits close to the device floor — I/O-pattern
  engineering has ≤20% left, not multiples.
- **Question B (prefetch)**: the counter that condemned prefetch measured the
  wrong thing. prefetch_hits counts arrived-while-in-flight only; new
  prefetch_used counts predictions actually consumed. OLMoE gate run:
  issued=1627, used=1494 = **92% true recall** (old counter said 0.8%!).
  The lookahead predictor was never bad on OLMoE — our metric was. gpt-oss
  true recall now being measured (E22 rungs); E17's "60% waste" stands as a
  read-amplification fact but the RECALL conclusion is reopened.
- **Question C (compression)**: could fewer bytes/miss come from on-disk
  compression? zstd on a 200MB expert-region slice of the real file:
  level 1 = 1.041×, level 3 = 1.040×, level **19 = 1.040×**, lz4 = 1.000×.
  **REFUTED** — MXFP4 is near-max entropy, as theory said. Decompress speed
  measured ~1GB/s/core (would have been viable had the ratio existed). The
  bytes-per-miss term is now proven irreducible from three directions
  (layout 1.5%, chunking, compression 4%).
- **E22 (prefetch regime law, both signs measured on gpt-oss)**: forced
  prefetch at m1.25 (miss-light, hit .885): 5.06 → 3.59 tok/s — wrong
  predictions ADD bytes that exceed the misses they prevent. Forced prefetch
  at m0.25 (miss-heavy, hit .434): 1.33-1.38 → **1.55 tok/s**, hit .434 →
  **.811**, stall −21%, total bytes only +5% — with 72% true recall
  (used=2797 of issued=3892), prefetches SUBSTITUTE future demand misses,
  moved earlier and overlapped with compute. Law: prefetch is a latency
  mover, not a bandwidth saver — it pays iff predictions replace demand
  misses (miss-heavy regime), and the existing hit_ema ≤ 0.80 auto-gate
  selects the correct side in BOTH measured regimes. Product default
  (m0.25 + auto prefetch) is therefore ~**1.55 tok/s** at agreement .917.
- **Decode physics after E21/E22**: time/token ≈ misses × ~8ms(device floor,
  parallel) + compute. Remaining levers, in order: (1) fewer misses —
  adaptive margin, eviction policy (Belady headroom sim next), slots↑ with
  RAM; (2) higher compute ceiling — GPU path; (3) prefill TTFT —
  expert-major scheduling port.

### E23. Eviction headroom: Belady says +14 pts, routing is Zipf-heavy (2026-07-19)
- **Question**: is LRU the right eviction policy, or are we paying misses a
  smarter policy would avoid? Method: real routing trace (2412 decode calls,
  m0 exact, s8, one prompt), replayed through LRU / Belady (clairvoyant
  demand-fill optimum) / static top-N frequency pin (scripts/belady_sim.py).
- **Result**:

  | slots | LRU | Belady | static top-N | Belady−LRU |
  |---|---|---|---|---|
  | 8  | .411 | .547 | .489 | **+13.6 pts** |
  | 12 | .498 | .648 | .600 | **+15.1 pts** |
  | 16 | .562 | .705 | .685 | **+14.4 pts** |
  | 32 | .737 | .783 | .881 | +4.6 pts |

  Pre-registered rule said >10 pts → build. **Routing is Zipf-heavy per
  layer: 8 of 128 experts cover 49% of uses, 32 cover 88%.** Static beats
  LRU everywhere (and beats Belady at s32 — legal: pinning is not demand-fill,
  it never suffers forced insertions). Caveats stated: one 64-tok prompt;
  static scored on its own trace (train=test, optimistic); Belady bound is
  honest for demand-fill.
- **Change**: LLMSTREAM_EVICT=lfu — frequency-protected LRU. Per layer,
  use counts accumulate; every 256 uses the top slots/2 experts by count
  become eviction-protected (pass-0 skip); pass-1 lifts protection so it can
  never deadlock. Driver-only, no fork change; default remains lru until
  measured.
- **Prediction (written before batch4)**: s8 m0 hit .411 → .43-.47 online
  (between LRU and the optimistic static .489), decode +5-15%; s12 m0 hit
  → .55-.58. Quality bar: m0 hashes BIT-IDENTICAL lru vs lfu (eviction moves
  retention, never logits) — gate in the batch script.
- **Result: REFUTED — prediction failed and the failure teaches the
  mechanism.** m0_s8 lru vs lfu: 4660 vs 4661 misses (one miss!), 1.39 vs
  1.39 tok/s, hashes bit-identical (quality gate PASS as designed). LRU's
  recency ALREADY protects frequency leaders: an expert taking 49% of
  traffic never reaches the LRU tail, so explicit protection changes
  nothing. The +14pt Belady gap is real but comes from FORESIGHT (next-use
  distance), which no frequency statistic approximates online. The honest
  chain: sim-static looked strong because train=test; online frequency adds
  zero. Code stays (env-gated, default off, harmless); card closed. The
  only online foresight we have is next-layer router logits — already
  exploited by prefetch; margin-mode residency coupling is the stronger
  version of the same idea and already ships.
- **Bonus numbers from batch4**: product-default candidate
  (m0.25+auto-prefetch) reproduced at **1.62 tok/s** (band 1.55-1.62,
  n=2); m0 s8 measured 1.05→1.39 across the day at identical configs and
  hashes — run-to-run drift up to ±20%, D10 (thermal logging, n≥3 repeats
  for headlines) now blocking honest headline claims.

### E24. GPU ceiling run + expert-major prefill scoped honestly (2026-07-19)
- **Provenance correction (against my own repeated claim)**: "expert-major
  prefill 16-599× on OLMoE" is a **phase-0 trace-simulator load-count
  result** (findings-phase0.md), NOT an engine measurement. The engine has
  never run expert-major prefill on any family. D2 reworded accordingly;
  end-to-end speedup remains unmeasured until built.
- **Design reality check**: llama.cpp computes a ubatch layer-by-layer, so
  the per-layer expert UNION of the ubatch must be simultaneously resident.
  Sim says 64-token unions run ~40-90 experts/layer — far above slots8-16 —
  and slot tensors are per-layer preallocated, so true expert-major needs a
  prefill-mode dynamic slot pool (all slot RAM lent to the current layer):
  a real fork feature, multi-session work, queued as M-next.
- **D12 (boundary found while scoping)**: prefill with n_ubatch>1 and a
  union larger than slots has no path — assign_slot exhausts victims (all
  needed), wait-loop drains, driver exit(1)s. Today all runs use ubatch=1 so
  it never fires; any future ubatch>1 config must clamp or split. Documented
  before it bit anyone.
- **GPU ceiling rung**: m2 s12 on Metal (SLOT_DEV=gpu, NGL=99):
  **13.68 tok/s** at hit .998 vs CPU ceiling 11.12 (**+23%**). Remaining gap
  to the ~37 tok/s bus bound is Metal MXFP4 kernel efficiency (llama.cpp
  territory, not streaming). phys_footprint 8.85GB (device buffers cost).
- **GPU at product default (m0.25+pf s8)**: **1.56 tok/s** ≈ CPU 1.55-1.62,
  prediction confirmed — miss-bound regime is backend-indifferent. Product
  backend policy: CPU default (5.7GB phys vs 6.9-8.9GB), GPU only above
  ~0.9 hit where its +23% ceiling matters. Session speed menu for
  gpt-oss-120b on 16GB M2 Air, all measured today: exact 1.2-1.4 /
  default ~1.6 (agreement .92) / m1.25 5.1-5.5 / ceiling 13.7 (GPU, m2 s12).

### E25. Feasibility gates for the steal-list: two refuted free, one survives (2026-07-19)
- **Protocol** (now standing, per Umar): cheapest offline gate first, engine
  code only for survivors, env-gated, bit-exact gate before commit, battery
  for quality, n>=3 for headlines.
- **Gate A - admission/victim cache (SLRU-style)**: simulated on the real
  trace at equal total slots. **REFUTED: -14.0/-20.7/-26.9 pts** at s8/12/16.
  This trace has no cache-pollution problem to fix; first-miss staging just
  delays admission of good experts. Second policy-refutation on this trace
  (after LFU) - recency is genuinely hard to beat here.
- **Gate B - static co-activation prefetch table**: honest half-split.
  **REFUTED: recall@8 = 61.6% at 2x bytes** vs the live router lookahead's
  72% at ~1x. The model's own logits beat its history; static tables also
  lose the D7 card definitively (on this trace size).
- **Gate C - dual-precision miss fetch, device leg**: 3x2.2MB scattered
  (a ~2-bit expert) = **2.68 ms/expert vs 4.22 full (1.57x)** at qd4-ish.
  SURVIVED leg 1 - then **REFUTED at leg 2 (transcode)**: dequant Q2_K
  3.12ms + quant->MXFP4 **92.26ms = 95.4ms/expert on one core**
  (csrc/transcode_bench.cpp) vs 1.54ms read saving - 60x over budget; the
  MXFP4 encoder is a 144MB/s scalar reference path, and even ideal 8-core
  overlap cannot close 60x. Alternatives questioned before the kill:
  direct Q2->MXFP4 repack (encode step still dominates), native low-bit
  compute in mixed-typed slot pools (major fork surgery - parked with this
  data attached). Steal-list final score: 3 of 4 candidates killed by
  cheap gates, zero engine code written, one survivor (expert-skip A/B,
  quality-motivated, hook-only) queued.
- **Incident logged**: orphaned server (UI restart bypasses atexit) mid
  swap-crawl defeated Stop+idle safeties -> orphan watchdog (parent-death
  self-exit, tested live), LLMSTREAM_REQ_TIMEOUT=900 wall cap, UI orphan
  sweep. Engine can no longer outlive its user.

### E26. Expert-skip A/B: the E25 survivor is REFUTED — omission is worse than substitution (2026-07-19)
- **Change**: LLMSTREAM_SKIP_W=w — an absent true-top-k expert whose true
  softmax weight < w gets a near-zero-weight resident filler (softmax≈0)
  instead of a full-weight resident substitute; other residents masked so
  top-k can't promote a replacement. Env-gated, default off, OLMoE gate PASS.
- **Prediction (pre-registered in scripts/gptoss_skip_ab.sh)**: skip reduces
  the m0.5 reasoning damage (substitute was +9.7% warm NLL) because a skip
  injects nothing where a substitute injects a wrong expert's signal at
  real weight.
- **Result: prediction REFUTED, decisively.** Δ warm NLL vs the m0 exact
  anchor, substitute (banked battery) vs skip w0.25, same margin m0.5,
  slots8, prefetch off:

  | domain | substitute | skip |
  |---|---|---|
  | code | +0.6% | +2.2% |
  | reasoning | **+9.7%** | **+31.6%** |
  | chat | −4.6% (noise, n=90) | +1.4% (noise) |
  | multilingual | +2.9% | +3.1% |
  | prose | +2.3% | +9.8% |

  Mean damage: substitute +2.2%, skip +9.6% — skip is ~4.4x worse. At
  m0.25 skip is a tie on code (+1.6% vs +2.2%) but still +11.6% on
  reasoning (substitute: −3.0%). Skip IS faster (~2 tok/s vs ~1.6 at m0.5
  — it deletes the fetch entirely), but it's a strictly worse
  speed-for-quality trade than simply raising the margin.
- **Mechanism, and why the phase-0 result did not transfer**: DeepSeek-V2's
  expert-skip worked because that architecture has SHARED experts that
  carry the token when routed experts are dropped. gpt-oss has none — a
  skip leaves a hole, while a substitute injects a resident expert the
  router itself ranked next-best, which evidently carries correlated
  signal. Third confirmation of the per-architecture doctrine (after
  adaptive top-k flipping between OLMoE and DeepSeek). Substitution is the
  fidelity-preserving degradation on shared-expert-free MoE; omission is not.
- **Contamination handled per protocol**: rung 1 (code m0.5) overlapped the
  dual-engine incident (guard fired 3x, slots crushed to 4, hit 0.375);
  artifact preserved as *.CONTAMINATED.txt, rung re-run clean on the same
  binary before any number above was quoted.
- **Repeat bench (D10 gap-2, n=3, same binary, back-to-back)**: medians
  m0 exact 1.56 tok/s [1.48–1.62], m0.25-noprefetch 1.74 [1.72–1.76],
  m1.25 5.58 [5.33–5.62], product default m0.25+auto-prefetch 1.77
  [1.61–1.82]. Within-day spread ±5% — far tighter than the ±20% cross-day
  band; D10's drift is between sessions, not within one.
- **Reproducibility, sharpened by accident**: with PREFETCH=0 the logits
  hashes were identical 3/3 in every config including margin modes; with
  auto-prefetch on, all 3 rounds hashed DIFFERENT. So margin-mode
  nondeterminism is specifically prefetch/I/O-timing-injected cache state;
  without speculative fills the eviction races didn't fire at GEN=64. The
  by-construction analysis stands; gates still demand hash equality only
  for exact mode.
- **Prefetch regime law, same-day check**: hit 0.432 -> 0.812 with
  auto-prefetch (reproducing E21's .434 -> .811 exactly) but only
  1.74 -> 1.77 tok/s today — total bytes read are nearly identical
  (75.0 vs 72.7 GB): prefetch converts demand misses into speculative
  reads, it does not remove reads. After ~2h of runs the OS page cache is
  warm (avg_bw ~1.65 GB/s), misses are cheap, and hiding their latency is
  worth little. The law's mechanism holds; its payoff scales with how
  expensive a miss actually is (cold-cache mornings, not warm afternoons).
- **D13 found**: the pre-registered check "skip_fills > 0 in every skip
  rung" was unverifiable from the artifacts — the counter only printed in
  the generation path, not the NLL path. Print added to the NLL block
  (visibility-only change, rebuilt + gated after all same-binary runs
  completed). Lesson repeated from D11: the logging you demand must be
  wired into the path you actually run.

### E27. Expert-major prefill lands: the D2 answer, and D14 caught by the gate (2026-07-20)
- **Design** (docs/prefill-design.md, predictions pre-registered before code):
  with a multi-token ubatch, llama.cpp's mul_mat_id is already expert-major -
  only our per-layer 8-slot caches couldn't hold a batch's expert union
  (D12's exit). Chosen approach after weighing five: a SHARED PREFILL POOL -
  one extra slot-tensor set, P = n_expert by default, that every layer
  refills in turn as the graph descends (sequential layer execution makes
  reuse safe on CPU). Union overflow becomes impossible by construction
  (closes D12); decode caches are never touched, so the warm chat cache now
  SURVIVES prefill instead of being thrashed by it. Env-gated
  LLMSTREAM_PREFILL_SLOTS (0=off, 1=auto n_expert); margin/skip/lookahead
  hooks early-return on multi-token graphs (residency masking is
  meaningless when the whole union is fetched - prefill under the pool is
  EXACT routing).
- **D14, found by the bit-exact gate before any number was quoted**: first
  build passed OLMoE layers 0-2 then produced degenerate routing (ids
  0,1,2,... from L3 on). Root cause: pool tensors typed from layer 0's
  meta, but Q4_K_M mixes quant types per layer (ffn_down: Q6_K on L0/L1/L4,
  Q4_K on L2/L3) - L2's bytes dequantized as the wrong format corrupted the
  hidden state, compounding into degenerate top-k by L3. Decode slots never
  hit this because they are per-layer typed. Fix: one pool per (kind,
  quant-type) variant; build_moe_ffn picks by the layer tensor's own type,
  the driver resolves identically and verifies stride per layer. Mixed-type
  files are the mainstream (every K-quant), so this fix IS the
  universality story, not an edge case.
- **Gates**: streamed+pool vs resident at identical ubatch=32 on OLMoE -
  logits bit-identical (b6869f5b6ef36376, same across 4 type-variant
  pools); negative control (no pool, ubatch 32) exits on union overflow as
  designed; ubatch=1 regression gate reproduces the banked hash with pf
  code present but dormant.
- **gpt-oss-120b TTFT, slots8 exact routing (results/pf_ttft_*)**:
  178-token prompt: baseline ubatch=1 prefill 134.9 s (1.32 tok/s) -> pool
  ubatch=128 prefill 39.4 s (4.52 tok/s), **3.4x faster TTFT**, fill 31.9 s
  of 39.4 (I/O-bound as predicted), unions avg 57 max 87, peak_rss 9.5 GB
  (pool 1.7 GB, 6 type variants: 3 MXFP4 weights + 3 F32 biases),
  phys_footprint unchanged at 5.7 GB. 712-token prompt at ubatch=512:
  first attempt hit GGML_ASSERT(n_tokens_all <= n_batch) - the driver
  passes a whole prompt as one llama_decode batch, so n_batch must be
  n_ctx, not max(512, ubatch); fixed, re-run below.
- **Boundary honesty**: prediction said prefill >= 8 tok/s; ubatch=128 gave
  4.52 (unions don't saturate at 128-token chunks - 212 MB/token vs the
  ~112 MB/token a 512-chunk should reach). The 512 rung is the prediction's
  real test.
- **512 rung (after the n_batch fix): PREDICTION CONFIRMED.** 712-token
  prompt, ubatch=512: prefill **10.81 tok/s**, TTFT **65.9 s** vs ~539 s
  extrapolated baseline - **8.2x** - both pre-registered bars cleared
  (>=8 tok/s, <75 s). Unions avg 71.6 max 103 (128-slot pool never
  pressured), fill 39.7 s of 65.9 (still I/O-bound: remaining headroom is
  overlap of next-layer fill with current-layer compute, not capacity),
  peak_rss 8.5 GB, phys_footprint 5.79 GB - the pool is transient and
  MADV_FREE'd after prefill. D2 is closed: chat TTFT on real prompts drops
  from minutes to about a minute, at exact routing.

### E28. Speed-per-GB curve, leg 1: the 60-70 tok/s bar is MET on the 7B tier (2026-07-20)
- **Question (from Umar)**: >= 60-70 tok/s, zero quality loss, <= 10-12 GB
  RAM, this M2 Air. Answered by measuring the model-size ladder at exact
  routing, same day, same driver (results/e28_olmoe_*):

  | config | tok/s | RAM |
  |---|---|---|
  | OLMoE-7B resident, Metal | **72.9** | ~4.3 GB |
  | OLMoE-7B resident, CPU | **69.4** | ~4.3 GB |
  | OLMoE-7B streamed s48, CPU | 48.1 (hit .928) | ~3.5 GB experts |
  | OLMoE-7B streamed s32, CPU / Metal | 42.3 / 41.3 (hit .820) | ~1.8 GB |

  The bar is met with 6-8 GB to spare - on the tier whose ACTIVE bytes fit
  the memory-bus budget (65 tok/s x ~0.7 GB/token active ~ 45 GB/s < the
  M2's ~100 GB/s). Metal streamed == CPU streamed at hit .82, re-confirming
  the backend law (GPU pays only above ~0.9 hit). Note: today's CPU
  resident 69.4 vs the banked 45.2 "stock llama.cpp" - different
  measurement path (our driver vs stock cli) and day; the lattice above is
  internally consistent same-day data. gpt-oss-20b leg: NOT run - Umar
  said no downloads (disk at 16 GB free).
- **Why 120B stays where it is on this device**, re-derived component by
  component at Umar's push: active bytes/token 2.7 GB (can't shrink at 0
  quality - internally-dense experts, phase-0), DRAM 100 GB/s -> 37 tok/s
  absolute ceiling with ALL 63 GB resident (impossible here), SSD 1.5-3.4
  GB/s serves misses, LRU is within 14 pts of the Belady oracle (E23), and
  cross-token amortization is now dead too (E29). The remaining real lever
  is the Metal MXFP4 kernel gap (13.7 measured hot vs 37 bus bound) - a
  0-quality 2.7x that still lands at ~37, not 60.

### E29. Batched-verify decode: REFUTED by feasibility gate, upper bound 1.37x (2026-07-20)
- **Idea worth re-testing after E27**: speculative verification reproduces
  the exact model output (0 quality by construction) and verifies w drafted
  tokens in ONE batched pass; E27 made batch passes ~7x cheaper per token,
  so the old "capped 1.2-1.7x" verdict (pre-pool, cache-restricted
  self-speculation) deserved a fresh gate.
- **Gate** (scripts/spec_verify_gate.py, offline, real slots8 trace of 58
  tokens x 36 layers): sequential LRU-8 misses vs per-window union fetches,
  priced pessimistic (cold pool) and optimistic (verify checks the decode
  cache first), at PERFECT acceptance - the physically unattainable upper
  bound. Result: w=4 1.18x, w=8 1.37x, w=16 1.65x, w=32 1.82x (opt).
  Real speculative windows are 4-8 and real acceptance ~0.6-0.8, which
  puts the practical ratio at or below 1.0 - a LOSS.
- **Mechanism, and why refutations keep landing on this spot**: LRU-8
  already harvests consecutive-token expert overlap - the SAME overlap the
  batch union amortizes. The two optimizations compete for one resource
  (temporal expert locality), so their gains do not compose. Third
  independent measurement of this cap (colibri data, phase-0 self-spec,
  now this gate) - closed with prejudice for the disk-bound regime.

### E30. The 10GB question + the warm asymptote (Umar's questions, measured) (2026-07-20)
- **Q1 (Umar): give 120B 10GB instead of 5.7 - max tok/s without slowing
  the laptop?** Sweep slots {12,16} x margin {0,0.25}, GEN=512 + a
  712-token teacher-forced exact run (results/e30_*).
- **The machine itself answered the slots16 question**: pressure lvl=2
  fired in every slots16 run (avail 3.3-3.8GB) and the guard sheared the
  cache to 10-14 slots - a 16GB Mac running macOS cannot hold 16
  slots/layer of 120B AND stay responsive. Practical ceiling: **slots12,
  7.6GB phys footprint** (peak_rss ~9.5 with pf pool transient). "Without
  impacting the laptop" is not a promise we make, it is a mechanism we
  run - the E10 guard enforced it live, three times, unprompted.
- **Q2 (Umar): does it get faster as it warms? YES - measured.** Default
  dial (m0.25) hit climbed 0.81 (64-tok runs) -> **0.915 warm** over a
  full 512-token generation: **2.40 tok/s sustained** including cold
  start (vs 1.77 short-run). Exact mode warm: hit 0.484 (64-tok) ->
  **0.756** over 711 teacher-forced tokens. Short benches DO understate
  the warm engine - README updated. The curve converges (the residual
  8-24% is the Zipf tail, which time cannot make resident), so warming
  buys the asymptote, not unbounded growth.
- **Honesty on today's absolute speeds**: the exact-mode warm run
  measured 1.11 tok/s where hit-rate physics at this morning's bandwidth
  predicts ~2.8 - afternoon read bandwidth had sagged to ~1.0-1.4 GB/s
  (vs 1.7 this morning) after ~10 hours of continuous benching (D10
  thermal, still unlogged - powermetrics remains the standing gap). The
  hit-rate findings are robust (internal to each run); cross-run tok/s
  today carries the D10 band. Three of four GEN=512 rungs also ended at
  ~57 tokens on greedy EOS - generation-length control matters for warm
  benches; the teacher-forced run is the clean instrument.
- **EOS-truncation lesson**: "GEN=512" does not mean 512 tokens get
  generated; uses/144 tells the truth. Rungs labeled accordingly, only
  full-length runs quoted for warm claims.

### E31. Frontier MLA+MTP family scan — PRE-REGISTERED envelope (2026-07-20)
- **Question (Umar)**: commit sluice to an MLA+MTP frontier family (DeepSeek-V3 /
  GLM) for "big + fast + zero-quality-loss"? Spec exact active-params, MLA dims,
  MTP, compute the honest tok/s envelope on 16/32/64 GB, pre-register before build.
- **Method**: two sourced spec pulls (config.json + arXiv + GGUF repos + llama.cpp
  PRs) → scripts/envelope.py, calibrated against 5 measured anchors
  (OLMoE 69/73, Qwen3.6-A3B 8.3, gpt-oss-120b 11.1/13.7). Master variable:
  active_bytes/token = active_params × bytes/weight. K = tok/s×active_B ∈ [25..95].
- **Two format-level killers found (apply to ALL of these families)**:
  1. **MTP is carried-but-UNUSED in mainline llama.cpp GGUF** for both `deepseek2`
     and `glm4moe` (nextn tensors dropped/inert; PR #14939). The 1.8× speculative
     lever the papers sell is **unavailable to any GGUF engine.** mtp=1.0.
  2. **MLA saves KV RAM (~71×), NOT bytes/token.** It helps residency, not speed.
- **Sourced active-params + PRE-REGISTERED predictions** (MacBook-class, CPU):

  | model | tot/act B | Q4 disk | active B/tok | bus ceil | 16 GB | 32 GB | 64 GB |
  |---|---|---|---|---|---|---|---|
  | **Qwen3.6-35B-A3B** | 35/3 | 26 GB | 2.07 | 48 | ~8 (MEASURED) | **fits: 10-13 pred** | fits: 10-13 |
  | GLM-4.5-Air | 106/12 | 73 GB | 6.6 | 15 | 0.1-2 | ~2-3 | ~2-4 |
  | DeepSeek-V3/R1 | 671/37 | 377 GB | 20.4 | 5 | 0.1-2 | 0.1-2 | 0.1-2 |
  | GLM-4.5 / 4.6 | 355/32 | 201 GB | 17.6 | 6 | 0.1-2 | 0.1-2 | 0.1-2 |
  | GLM-5.2 | 750/40 | 466 GB | 22.0 | 5 | 0.1-2 | 0.1-2 | 0.1-2 |

  Predictions are FALSIFIABLE bands to grade later. Qwen resident band narrowed to
  10-13 (not the raw 13-30) because its model-specific K≈30 (Q5 dequant + shared
  expert + hybrid attn) is far below OLMoE's K≈90; the head-to-head json already
  measured the CPU compute ceiling at ~9-11 for 3B active. Streamed "0.1-2" rows
  are anchored to colibri's MEASURED GLM-5.2 (0.05-0.1 tok/s @25 GB, 1.8 @128 GB).
- **Verdict (decision, not yet an experiment)**: the frontier MLA+MTP giants are
  **dead ends on consumer hardware** — 32-40 B active → ≤6 tok/s bus ceiling *even
  all-resident*, they don't fit so they stream (colibri regime), and their one
  escape (MTP) is inert in GGUF. The design rule holds and is now sourced:
  **chase high-total / LOW-active (A3B-class), not active-heavy frontier models.**
  Quality ∝ active-params ∝ bytes/token ∝ 1/speed — the trilemma's root, quantified.
- **Only near-miss**: GLM-4.5-Air (12 B active) — colibri regime on 16 GB, ~2-4 on
  64 GB. Not competitive with the A3B path on this hardware.
- **Next**: the winning target is the model we already hold. Pre-registered test to
  run when a 32 GB machine is available: Qwen3.6-35B-A3B fully resident → predict
  10-13 tok/s (falsifies if <8 or >16). No frontier download justified by this scan.

### E32. D3 live verification: gate green + AGREE_TARGET dial holds (2026-07-21)
- **Goal (colleague-directed)**: don't trust the ledger — prove D3 live. Exact
  vs resident bit-identical hash; Balanced/Fast actually emit AGREE_TARGET
  .95/.90 and the engine honors them. Artifact: results/d3_live_verify.txt.
- **Exact-mode gate (protocol #4)**: gpt-oss-20b, resident vs slots32 vs
  slots12, N_GEN=24 → **all bit-identical, hash 0ec1c81919bbdafc. GATE PASS.**
  Exact (agree=0, margin=0) is streamed==resident bit-for-bit.
- **Dial live (64-tok, logit-scale gating)**:

  | AGREE_TARGET | settled margin | router_agreement | decode |
  |---|---|---|---|
  | 0.95 | 0.021 | 0.9977 | 3.35 tok/s |
  | 0.90 | 0.188 | 0.9848 | 3.78 tok/s |

  Family-agnostic (one knob lands on logit-scale margins on gpt-oss), directional
  (lower target → higher margin → faster), floor HELD both runs (agreement ≥
  target). CLI mode_env + UI MODES emit exactly .95/.90.
- **Honest caveat (known, E20 arithmetic)**: under-exploits on short 64-tok runs
  — agreement sits ABOVE target because 8 windows can't climb the full range in
  64 tokens; needs ~200 tok / warm run (E30) to settle nearer the floor. Quality
  floor is safe (errs conservative); speed under-exploited on short replies. Not
  a bug — quality-first by construction. Candidate follow-up: seed margin from a
  per-family prior so it starts closer to the settle point.
- **Verdict**: D3 CLOSED and live-verified. Dial surfaced in both CLI + UI.

### E33. Working-set persistence sidecar (warm-start packs) (2026-07-21)
- **Scope (colleague-directed)**: ONLY the persistence half — save/reload warm-
  start packs, env-gated, off by default. Engine preload hook = go/no-go below.
- **Built**: `src/warmpack.py` — build the per-layer hot-expert working set
  (smallest set covering `coverage` of a task's routing) from a trace or a live
  id-dump; save/load a compact sidecar JSON (results/warmpacks/<task>.warmpack.json).
  Inert unless invoked; the engine consumes a pack only when LLMSTREAM_WARMPACK is
  set (stock path byte-identical without it — protocol #3).
- **Packs built** (OLMoE 64-expert traces, 16 layers, coverage 0.90): 38-47
  experts/layer, round-trip verified.
- **Measured cold-start lift** (LRU-16, preseed pack top-16 vs cold), FRONT-LOADED:

  | window | cold hit | warm hit | lift |
  |---|---|---|---|
  | first 3 tok | .331 | .599 | **+26.8 pts** |
  | first 8 tok | .466 | .567 | +10.2 |
  | first 20 tok | .514 | .555 | +4.1 |
  | first 50 tok | .529 | .545 | +1.6 |

  Big exactly where it matters (first-token latency / fresh session / task switch),
  decays as the cache self-warms. Honest: negligible beyond ~30 tokens.
- **Pre-warm is logit-neutral** — changes residency-at-start, not which experts
  compute — so exact-mode hash must stay 0ec1c81919bbdafc when the hook lands.
- **Engine hook LANDED + gated (2026-07-21)**: `LLMSTREAM_WARMPACK` pre-fills the
  decode cache at init (assign_slot + fetch_one per pack expert, stream_run.cpp).
  Off by default (unset → never entered, stock path byte-identical). Wrong-model
  packs guarded by an EOF file-bounds check (skip out-of-range experts, no crash).
  **Protocol #4 gate**: exact-mode slots12, warmpack OFF vs ON (288 experts
  preloaded) → `logits_hash 0ec1c81919bbdafc` **IDENTICAL** — pre-warm is
  logit-neutral, proven live. warmpack.py now emits the engine-readable `.pack`.

### E34. Warmpack hook — live engine A/B (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before any A/B run.**

- **Why not a straight E33 replay**: E33's **+26.8 pt @ first-3-tok** is an OLMoE
  8-of-64 *simulator* number (Python LRU-16, cold vs preseed-top-16, decode-only).
  OLMoE is not loadable as a GGUF on this box; the only loadable MoE is
  **gpt-oss-20b-MXFP4** (24 layers, MXFP4, ~3.6B active). So the live A/B *must*
  run on a different model through the **real** engine cache — fixed slots +
  router-lookahead prefetch — not the sim. Divergence from 26.8 pt is therefore
  EXPECTED; quantifying and explaining it is the point.
- **Real pack, not the fixture**: build a genuine gpt-oss working set from a live
  decode id-dump (`LLMSTREAM_PRINT_IDS`, true expert ids, *decode rows only* —
  n_tokens=1). The synthetic `gptoss20b_gate.pack` (experts 0–11) was only ever a
  bit-exact safety fixture, never a real working set — it would show ~random lift.
- **Measurement (engine telemetry only)**: `io: … decode … (hit X)` and
  `decode: … tok/s`, **exact mode** (margin 0 → logit-neutral, gate stays
  `0ec1c8…`), `LLMSTREAM_SLOTS` held **constant** across OFF/ON. Cumulative
  decode-hit at `n_gen ∈ {3, 8, 20}`, warmpack OFF vs ON. Two legs:
  **contaminated** (pack built from the test prompt = upper bound) and **clean**
  (pack from prompt A applied to a different prompt B).
- **Predictions (mine — I do NOT expect 26.8 pt):**
  1. **Direction**: warm ON hit ≥ OFF hit at every N; lift strictly largest at
     N=3, decaying toward ~0 by N≈20–50 (front-loaded — same *shape* as E33).
  2. **Magnitude @ N=3**: positive but **smaller** than +26.8 pt — expect
     single-digit to low-double-digit points — because (a) the engine's OFF
     baseline already runs router-lookahead prefetch (the E33 pure-LRU sim did
     not) → higher cold baseline, less headroom; (b) real slot count is chosen
     ≥ sim's 16 → smaller cold penalty; (c) different model / expert count.
     Contaminated ≥ clean.
  3. **tok/s**: warm ON early decode tok/s ≥ OFF (fewer cold-miss stalls at
     start), effect small and possibly within run-to-run I/O noise; I pre-commit
     to reporting it as **directional-only** if the OFF↔ON gap is under the
     observed run-to-run spread.
- **Falsifier**: if warm ON does not lift N=3 decode hit above OFF beyond noise,
  the hook buys nothing live and should be dropped from the product surface
  (kept only as the gate fixture). *(measured numbers appended below the run.)*

**Measured (appended after the run — gpt-oss-20b, exact greedy, GUARD=0 SLOTS=5,
prompt A = pack source ⇒ CONTAMINATED upper bound):**

| N (cum.) | OFF hit | ON hit | Δ hit | OFF tok/s | ON tok/s |
|---|---|---|---|---|---|
| 1  | .667 | .656 | −1.1 pt | 1.40 | 1.20 |
| 3  | .670 | .678 | +0.7 pt | 1.96 | 1.75 |
| 8  | .716 | .727 | +1.1 pt | 1.81 | 1.59 |
| 20 | .722 | .716 | −0.6 pt | 2.04 | 1.91 |

(N=3 = mean of 2 reps each; OFF↔OFF spread at N=3 was .667 vs .674 = 0.7 pt.)

- **Result vs prediction**: Δ hit oscillates in **[−1.1, +1.1] pt** — i.e. within
  the run-to-run noise floor at *every* horizon; **no lift**. Predicted direction
  (ON ≥ OFF, front-loaded) is **NOT confirmed** — the sign flips with noise.
  tok/s: ON ~0.15–0.22 lower at every N (preload reads 120 experts ≈1.6 GB at
  init + seats slots the demand path then churns) — small, also ≈noise but
  consistently negative. **My pre-registered falsifier TRIGGERED.**
- **Investigation — why ≠ E33's +26.8 pt** (this is the whole point of the run):
  1. **Cache ≈ top_k.** The memory-safe cache on this 16 GB M2 is 5 slots
     (the guard's *own* verdict — it drops to 5 under pressure); top_k=4. A cache
     that holds barely more than one token's experts has no room to *retain* a
     cross-token working set for warm-start to seed. E33's sim had cap 16 vs a
     38–47 working set — room to hold a meaningful fraction.
  2. **Prefetch on.** The engine's router-lookahead already recovers cold misses,
     so the OFF baseline is .67–.72; E33's pure-LRU sim had no prefetch → baseline
     .33 → all the headroom lived there. Real engine has little left to recover.
  3. **Model.** gpt-oss 32-expert/top-4 vs OLMoE 64-expert/top-8 — less working-
     set dispersion, less to seed.
  Contaminated (pack = test prompt) is the **upper bound**; it is already within
  noise, so the clean (prompt-B) leg — necessarily ≤ upper bound — was **not run**.
- **Bug found *by* the A/B, then fixed**: `warmpack_preload` over-filled past the
  cap — `assign_slot(cap=n_slots)` *evicts-when-full* instead of returning −1, so
  the loop churned and **retained the LAST (coldest) pack ids, evicting the first
  (hottest)**. "preloaded 362" counted every assign; only ~5/layer survived, and
  they were the wrong ones. One-line fix: stop the layer at `slot_of.size() >=
  n_slots` (pack is most-frequent-first ⇒ hottest seat first). Post-fix preload =
  120 = 5/layer exactly. **Gate re-verified bit-exact** (OFF = ON =
  `fdf0f83dd70504c5`) — still logit-neutral.
- **Verdict**: the hook is **correct** (bit-exact, now seeds the hottest experts)
  but its live benefit is **gated on cache ≫ top_k**, which a 16 GB M2 cannot
  provide for gpt-oss-20b (cache is pinned near top_k by the RSS guard). On this
  hardware class it buys **nothing measurable** and slightly costs early tok/s.
  Keep it gated + off by default; **do NOT advertise a live warm-start speedup on
  small machines**. The +26.8 pt stays what it always was: an OLMoE *simulator*
  number in a no-prefetch, cache=16 regime that does not exist on this box.
  Pillar-2 "starts warm" is caveated in `techniques.md` accordingly.
- **Artifacts**: `results/warmpack_ab/` — `profileA.{out,err}` (id-dump),
  `gptossA.warmpack.{json,pack}` (real pack), `gate_{off,on}.out` (bit-exact),
  `A_{off,on}{1,3,3b,8,20}.{out,err}`, `curve.txt`.

### E35. Warmpack A/B in E33's regime — CLOSED, not run (owner call, 2026-07-21)
Intended as the decisive close-out: put warm-start in the regime E33 *simulated*
(cache ≫ top_k, ≪ working set) on OLMoE-1B-7B — the one model whose tiny experts
would let the cache hold the E33 cap without the RSS guard throttling to ≈top_k.
That requires an OLMoE GGUF (we have only OLMoE *traces*, not loadable weights,
and only gpt-oss-20b on disk). **Owner halted the download: gpt-oss-20b only, no
new models.** A brief OLMoE-1B-7B pull was started and then stopped/deleted; disk
unchanged (62 GB free).

- **Status**: regime **unreachable** on the target model/hardware. On the only
  loadable model here (gpt-oss-20b at 16 GB), the memory-safe cache pins to
  ≈top_k (E34), so the cache-≫-top_k regime cannot be made memory-safe on this
  box. The E33-regime live test is therefore not runnable as specified.
- **Verdict**: warm-start benefit is **unverified live**; the claim is **retired
  pending a machine or model where cache ≫ top_k is memory-safe**. E34 stands as
  the final word for this hardware: no live Δhit beyond noise. No re-runs to chase
  a different result; the RSS guard is **not** to be widened to force slots.
- **Hook**: remains gated, off by default, correct, and bit-exact ("dark").
  Pillar-2 "starts warm" stays caveated exactly as E34 left it — no doc change.

### E36. Dynamic LFRU repin vs pure LRU — live gpt-oss-20b (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before any run.** E23 refuted *static* top-N pinning
(train=test optimism; LRU already protects Zipf leaders). Colibri's variant is
*dynamic* — frequency-tracked, repins the hottest at runtime. We have never A/B'd
that. This is the test.

- **Mechanism under test**: `LLMSTREAM_EVICT=lfru` — dynamic LFU. Implemented as
  lfu-with-decay: the per-layer use-counter halves every 256 uses (half-life =
  the recompute cadence), so the protected/repinned set = top `slots/2` by
  **recent** frequency, protected from eviction (existing two-pass path). This is
  distinct from the shipped `lfu` (undecayed whole-run counts ≈ static) and from
  E23's static top-N. **Repin decided at eviction time** via the protected set;
  no prefetch change, no slot-count change, no second knob.
- **OFF byte-identical**: unset ⇒ `protect=false` (eviction pass starts at 1 =
  pure LRU), `g_lfru_decay=false` (no decay); the LRU and lfu paths are unchanged.
  Executable OFF check: OFF `logits_hash` must equal the committed stock hash.
- **Workload**: gpt-oss-20b, prompt A (code, held out from any pack), exact
  greedy, `GUARD=0 SLOTS=5` — **identical to E34** so numbers are comparable.
  No offline pack is involved, so the A/B is **CLEAN by construction** (no
  train/test fit to contaminate; policy uses only live runtime counts).
- **Structural caution (honest, E34)**: 5 slots vs top_k=4 gives *any* policy ~1
  spare slot. The working set (~15 experts/layer) ≫ cache, so evictions are
  constant, but whichever ≤2 experts lfru protects, the other ~10 hot ones still
  miss. This is the same memory-bound wall as E34.
- **Prediction**: **no detectable difference at this cache size.** LRU already
  keeps the most-recent (≈ the only viable set when cache ≈ top_k), and lfru's
  protected set is only `slots/2 = 2` experts — too little leverage to move the
  aggregate hit beyond noise. The one plausible signal: protecting the 1–2 Zipf
  leaders from eviction could give lfru a *marginal* hit edge; I expect it bounded
  and sub-noise. tok/s: predicted equal within noise (policy changes residency
  choice, not I/O volume materially).
- **Noise bar**: OFF↔OFF (LRU↔LRU) spread on prompt A, **run first**. lfru only
  "beats" LRU if `|Δhit| > 2 ×` that spread at a given N.
- **Falsifier**: the "no detectable difference" prediction is falsified if lfru's
  Δhit exceeds 2× the noise bar at any N (either direction).
- **Verdict space (any is a pass if gate + labeling clean)**: lfru beats LRU
  beyond noise / no difference / worse. **No tuning the 256-use / halve decay to
  chase a win** — one pre-registered setting; a loss ends the thread here.
  *(measured numbers appended below the run.)*

**Measured (appended after the run — gpt-oss-20b, exact greedy, GUARD=0 SLOTS=5,
prompt A; CLEAN — no pack, policy uses only live runtime counts):**

| N | LRU hit | lfru hit | Δ hit (lfru−LRU) | LRU tok/s | lfru tok/s |
|---|---|---|---|---|---|
| 1  | .667 | .646 | −2.1 pt | 1.43 | 1.59 |
| 3  | .681 / .681 | .674 | −0.7 pt | 1.83 | 1.65 |
| 8  | .715 | .712 | −0.3 pt | 1.86 | 1.58 |
| 20 | .719 | .726 | +0.7 pt | 2.06 | 1.87 |

- **Noise bar**: LRU↔LRU at N=3 = **.681 vs .681 = 0.000 pt**. **Reconciling with
  E34's 0.7 pt spread (verified from code, not assumed)**: the reported hit counts
  an expert resident iff it is in `slot_of` (stream_run.cpp:816), *not* on
  `in_flight`; `slot_of` is written at prefetch-issue time deterministically — but
  *which* experts prefetch seats is gated on `hit_ema` (~L498/L636), and `hit_ema`
  folds in the timing-dependent `in_flight` term (~L789/L815). So the metric is
  deterministic **except** at the margin where async-prefetch completion timing
  flips a `hit_ema`-gated prefetch decision — a **0-to-~1 pt jitter**. E34 sampled
  the high end (0.7), E36 the low end (0.000); same jitter band, not contradictory
  constants. E34's ±1.1 pt swings sit inside this band, so that verdict stands.
  The Δ's below are the same order as the jitter — hence "no systematic win."
- **Gate**: OFF (unset) = LRU = lfru = `fdf0f83dd70504c5` = the committed stock
  hash. Byte-identical OFF confirmed; lfru changes residency only, never compute.
- **Verdict: lfru does NOT beat LRU.** Δhit swings −2.1 → +0.7 pt across N —
  marginally *worse* early, a smaller gain late; no systematic improvement, and
  the net is negative over the horizons that matter (first tokens). tok/s: lfru
  ~0.2 slower (extra per-eviction scan + decay), within noise. **My pre-registered
  "no detectable win" holds**; my directional guess (a marginal Zipf-leader edge)
  was wrong — at cache≈top_k the decayed counter is *cold* for the first tokens,
  so protecting `slots/2 = 2` experts on near-zero counts evicts a more-useful
  recent expert → early harm (N=1 −2.1). Counts only stabilize by ~N=20, where it
  roughly ties.
- **Why (structural, consistent with E23 + E34)**: 5 slots vs top_k=4 leaves ~1
  spare slot; protecting 2 experts out of a ~15-wide working set can't move the
  aggregate, and LRU already keeps the most-recent (≈ the only viable set at this
  size). E23 refuted *static* top-N; E36 shows the *dynamic* decayed variant
  (colibri's mechanism) also fails to beat LRU **on this hardware** — not because
  the mechanism is inherently bad, but because cache≈top_k gives no policy room
  (same wall as E34/E35). A fair test needs a machine/model where cache ≫ top_k is
  memory-safe; that is a **separate directive**, not a re-run here.
- **Thread ends** (loss). Policy stays `LLMSTREAM_EVICT` = {unset=LRU default |
  lfu | lfru}, off by default, bit-exact, dark. No decay tuning, no follow-up.
- **Artifacts**: `results/lfru_ab/` — `{lru,lfru}{1,3,8,20}.{out,err}`, `lru3{a,b}`
  (noise bar), `curve.txt`.

### G1 CLOSE-OUT — DRAFT, awaiting owner metric sign-off (2026-07-21)
**Status: not closed. One decision is missing and it is the owner's, not mine.**
Drafted here so the evidence sits in one place instead of across E37/E37b/E37c/E37d.

**G1 as re-scoped by the owner**: gpt-oss-20b · ≤10 GB RSS · bit-exact · ~5–6 tok/s
verified clean. (The original ≥10 tok/s floor was relaxed after E37 showed it sits
above the model's own compute ceiling of ~10.5 tok/s — see below.)

| criterion | measured | verdict |
|---|---|---|
| bit-exact | hash `7fff2b7b9461da2a`, identical in every N=64 leg | **MET** |
| ~5–6 tok/s clean | **6.14** tok/s, quiet box, N=64, SLOTS=16 (E37c) | **MET** |
| ≤10 GB RSS | **6.73 GB** `phys_footprint` / **10.04 GB** `time -l` peak | **DEPENDS ON THE METRIC** |

**The open decision.** The two RSS numbers are not a discrepancy to resolve — they
measure different things, and G1 does not say which one it means:
- `phys_footprint` = **6.73 GB**. Real resident RAM. Load-invariant and
  length-invariant — identical at N=8, N=20 and N=64, and identical on a quiet box
  and a loaded one. Comfortably inside 10 GB.
- `time -l` peak / `ru_maxrss` = **10.04 GB**. Includes reclaimable mmap pages from
  the 12.1 GB model file. Grows with N (my E37c prediction that it was N-invariant
  was WRONG and is corrected here). Sits 0.04 GB inside 10 GB — i.e. it would fail
  on a slightly longer run, and the number tracks how much the kernel has not yet
  bothered to evict rather than how much memory we need.

**My recommendation: `phys_footprint`**, because it is the number that predicts
whether the machine swaps, it is stable across load and length, and the peak figure
counts pages the kernel will drop for free under pressure. But adopting it is a
change to what G1 *means*, and that is an owner call — G1 stays formally **OPEN**
until it is made. I am not closing a goal by picking the metric that passes it.

**Supporting evidence, and one correction I owe.**
- **The ≥10 tok/s floor was never reachable**, and not for want of engineering: E7's
  governing law puts the compute ceiling at ~10.47 tok/s with a *perfect* cache, so
  a 10 tok/s target demanded ~96% hit rate. The binding constraint at this rung is
  compute and RSS, not storage bandwidth.
- **The E37b "one rung short" reading was wrong, and the owner caught it.** I read
  4.14 tok/s at N=20 as a slot-count deficit. From the artifacts it is measurement
  length: the hit curve climbs 0.677→0.855 across N, so N=20 is still warming, and
  E28's 0.905/5.86 is N=64 steady state. Re-run at matched N=64: **6.14 tok/s**,
  in band. The gap was the ruler, not the engine.
- **Load sensitivity is measured, not assumed**: 6.14 tok/s quiet (9.2 GB free),
  4.89 light (8.4 GB free, E37d), 1.57 heavy (1.5 GB free, contaminated, kept only
  as the under-load datapoint). Same settings, same output hash in every row.
- **One run was GUARD=0** (E36's probe), declared and tolerated at the time; no G1
  number above depends on it.
- **Artifacts**: `results/e37c/`, `results/e37d/`, `results/e37b/`, `results/g1_e37/`,
  `results/e28_20b_stream_s16.txt`.

### E37. The 10 GB rung for gpt-oss-20b — baselining Goal G1 (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before any run.** G1 = ≤10 GB RSS, bit-exact,
target ≥10 tok/s. Measurement only, no new features/levers.

- **Geometry (measured at launch)**: trunk RSS **1.92 GB**, **13.3 MB/slot-layer
  × 24 = 319 MB/slot**, top_k=4 → 96 expert-uses/token, per-miss 13.25 MB.
- **Anchors**: governing law (E7) `t/tok ≈ misses×13.25MB ÷ 1.15GB/s + 0.095s`;
  compute ceiling **10.47 tok/s @ 99.8% hit** (E7); E28 ~6.6 GB→hit .905/5.9 t/s,
  ~8.5 GB→4.0 t/s (more cache = slower, D10).
- **Predictions:**
  1. **Slot budget @ 10 GB RSS** = `(10 − 1.92 − ~0.3 KV)/0.319` = **~24 slots**.
  2. **Hit @ 24 slots** ≈ **0.95** (cache 24 > code working set ~15/layer, E34;
     E28 ~16 slots→.905, saturating above).
  3. **tok/s @ 10 GB (if holdable)** = `(1−.95)×96×13.25MB=63.6MB ÷1150MB/s
     =.055s + .095s = .150s` → **6.7 tok/s** (bandwidth-law; D10 pressure would
     pull lower). **< 10 target.**
  4. **Resident compute ceiling (run b)** ≈ **10.5–13 tok/s** (E7 anchor 10.47).
  5. **Effective decode bw (run c)** ≈ **1.0–1.15 GB/s** (E7 1.15; E20 1.3–1.8
     repacked).
  6. **G1 feasibility pre-call** — the key call:
     - Task line: required miss-bytes/token for 10 t/s = `bw/10 = 1150/10 =`
       **115 MB/tok**. Predicted actual @ hit .95 = **63.6 MB/tok** → **0.55×**,
       *within* budget. **Bandwidth is NOT the binding constraint at 10 GB.**
     - Compute IS: 10 t/s ⇒ 0.10 s/tok, but compute alone = 0.095 s ⇒ miss-wait
       budget 0.005 s ⇒ ≤5.8 MB/tok ⇒ hit ≥ **99.9%**. Even at 100% hit, resident
       ≈ 10.5 t/s and any miss drops below 10. **Predict G1 ≥10 t/s INFEASIBLE on
       gpt-oss-20b on this box — bounded by CPU compute, not disk.** Faster needs a
       wider memory bus / GPU (E-note line 166: "37 t/s needs a wider bus, not
       more software"), not more cache or faster SSD.
  7. **Memory-safety call (protocol #2, pre-registered)**: avail is ~1.5 GB
     (vm_stat) / 6.1 GB (engine); a 10 GB cache + the user's ~13 GB of apps > 16 GB.
     A GUARD=0 force to hit 10 GB would swap and impact the user's work — **we do
     not do that**. So **run (a) is GUARD ON**; we report the memory-safe **floor**
     the guard permits as the actual rung (predicted ~4–6 slots → **~3.8 GB RSS**,
     hit ~.70–.75, ~2–3 t/s), and label the 10 GB point as guard-refused. No
     GUARD=0 run in this task. Resident (b) attempted at short N; if it thrashes,
     report the thrash and fall back to the E7 ceiling (10.47).
  8. **Bit-exact gate**: streamed hash must equal resident hash (exact greedy).
  *(measured numbers appended below the run.)*

**Measured — ⚠️ CONTAMINATED (UNDER DESKTOP LOAD): superseded by E37b for the
clean numbers.** Taken with ~13 GB of the user's apps resident (avail ~1.5 GB), so
bw collapsed to ~0.12–0.58 GB/s and decode carried a swap-fault tax. **Kept
deliberately as the "under desktop load" datapoint — a real user condition worth
having** — but not the clean G1 baseline (see E37b). *(gpt-oss-20b, exact greedy;
run (a) GUARD ON SLOTS=24; RSS via `/usr/bin/time -l`.)*

Run (a) streamed, requesting 24 slots (the 10 GB budget):

| N | hit | tok/s | peak RSS | final cap | avg_bw | per_stream_bw |
|---|---|---|---|---|---|---|
| 1  | .677 | 1.64 | 4.85 GB | 24 (full) | 940 | 110 |
| 3  | .747 | 1.65 | 5.05 GB | 24 (full) | 964 | 118 |
| 8  | .832 | 1.38 | 5.88 GB | 22 | 824 | 120 |
| 20 | .855 | 1.57 | 6.21 GB | 16 | 660 | 123 |

- **Predicted vs measured:**
  - Slot budget @ 10 GB: pred ~24 → **10 GB never reached.** RSS climbs with
    tokens as experts load and **plateaus ~6.2 GB**; the guard sheds 24→**16** under
    pressure, and the code working set (~15/layer, E34) doesn't fill 24 anyway.
    **The "10 GB rung" does not exist for this model on this box — the memory-safe
    operating point is ~6 GB.**
  - Hit: pred ~.95 → measured **.855** @ N=20 (16 slots, pressure), .832 @ N=8.
  - tok/s: pred 6.7 (law) / 3–4 (D10) → measured **1.4–1.7** — well under even the
    D10 figure, because `per_stream_bw` collapsed to **110–123 MB/s** (vs the
    1.15 GB/s anchor): current memory pressure starves the reads (E30).
  - Resident ceiling (b): pred 10.5–13 → **UNMEASURABLE: thrash.** Full ~11 GB
    resident under ~1.5 GB avail swapped catastrophically (E4 scenario); killed at
    100 s with prefill not even done (no orphan left, avail recovered to 2.78 GB,
    no lasting user impact). Best ceiling remains the **E7 anchor ≥10.47 t/s**.
  - Effective decode bw (c): pred 1.0–1.15 GB/s → measured **~580 MB/s**
    (miss-bytes ÷ stall, N=20), avg_bw 660–964, per_stream 120 — ~2× below anchor
    (pressure).
- **Per-token decomposition (N=20, 0.638 s/tok)**: **50% miss-wait** (stall
  0.319 s) + **50% compute+faults** (0.319 s), of which only ~0.095 s is true
  compute (E7) and ~0.224 s is **swap-fault tax** on the resident trunk under
  pressure. Under a clean machine the compute half would drop to ~0.095 s.
- **G1 feasibility line (computed)**: required miss-bytes/token for 10 t/s =
  `measured_bw/10 = 580/10 =` **58 MB/tok**; observed = `13.95 miss × 13.25 MB =`
  **184.8 MB/tok** → **3.2× gap on bandwidth**. But bandwidth is **not** the
  binding wall: even at 100% hit, compute alone caps throughput at ~3.1 t/s
  (pressure) / ~10.5 t/s (clean, E7) — **neither ≥10 within a 10 GB budget.**
- **Bit-exact gate**: streamed N=8 (24-slot) = `fdf0f83dd70504c5` = the E34/E36
  canonical exact hash, *identical* across slot counts 5→16→24 → compute is
  provably cache-size-invariant (the gate's purpose). Fresh resident-side hash
  blocked by the thrash; exact-mode = resident is the established contract, so the
  gate holds by that reference. **Green.**
- **VERDICT — G1 (≤10 GB RSS, ≥10 t/s, bit-exact) is INFEASIBLE for gpt-oss-20b on
  this 16 GB M2.** Bounded by (1) **RSS**: approaching the ceiling needs full
  residency (~11 GB > 10 GB budget); a 10 GB *streamed* cache can't even be
  reached (guard + working set plateau ~6 GB). (2) **CPU compute**: the clean
  compute ceiling (~10.5 t/s) *is* the target, so streaming — which only adds
  miss-wait — cannot clear 10. Disk bandwidth is the *least* binding factor
  (3.2× on paper, but slack once hit is high). Faster needs a **wider memory bus /
  GPU or a smaller-active-param model** (matches D10 + line 166: "37 t/s needs a
  wider bus, not more software"), not more cache or a faster SSD. The numbers
  decide the next directive.
- **Artifacts**: `results/g1_e37/` — `stream_N{1,3,8,20}.{out,err}`,
  `resident_N8.{out,err}` (thrash), `streamed.txt`.

### E37b. G1 close-out on a QUIET box (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before staging anything.** Owner relaxed the scope:
**G1 is now gpt-oss-20b, ≤10 GB RSS, bit-exact, ~5–6 tok/s verified clean** — E37b
is the close-out, not a feasibility gate. E37's numbers were taken under desktop
load (bw collapsed to ~0.12 GB/s, swap-fault tax) → to be relabelled contaminated.

- **Re-scope (owner-approved, 2026-07-21)**: the 12 GB gate is unreachable without
  a reboot — avail floors at ~9.4 GB with everything closed (macOS baseline + our
  two terminal sessions). **G1 close-out = the streamed leg only.** So: (1) streamed
  leg is **primary and runs FIRST** — gate drops to **avail ≥ 8 GB** (streamed
  footprint ~6.2 GB); running it first also avoids a resident-first run warming the
  OS page cache the streamed leg reads through. (2) resident leg is **opportunistic**
  — after the streamed leg, re-check avail and run it only if ≥ 12 GB (it won't be
  → log "skipped: avail=X, E7 anchor ≥10.47 stands"). No thrash risk ever.
- **Memory gate (protocol #2)**: detached self-guarding script waits up to 2 min
  for the streamed gate (avail ≥ 8 GB; vm_stat, **16 KB page size** — E37b caught
  that E37-era avail math used 4 KB). If not reached → abort + logged reason, no
  run. Both legs GUARD ON; **no GUARD=0**.
- **Resident leg (N=20, opportunistic)**: fresh compute-ceiling measurement if
  avail ≥ 12 GB; **predict ≥ 10.47 tok/s** (E7's ceiling was measured *despite*
  swap → a quiet box meets or beats it; band 10.5–13). Expected to be **skipped**
  this run; the E7 anchor (≥10.47) stands as the ceiling of record.
- **Streamed leg (N=20)**: `SLOTS=16` (registry "balanced" default), GUARD ON,
  exact greedy, prompt A. **Predict hit ~0.90** (E28 .905), **RSS ~6.6 GB**
  (≤10 GB ✓), **~5–6 tok/s** (E28 5.9; law at clean 1.15 GB/s bw, hit .90 →
  9.6 miss×13.25MB÷1150 = .111s + .095s compute = .206s → ~4.9 t/s), effective
  decode bw **~1.15 GB/s**, and a **clean miss-wait / true-compute split with NO
  swap-fault term** (compute ≈ 0.095 s/tok, miss-wait ≈ 0.1 s/tok).
- **Bit-exact gate**: resident N=20 hash **==** streamed N=20 hash (exact greedy
  selects the true top-k → identical experts computed → identical logits).
- **Verdict deferred**: comes after, from the on-disk artifacts only. Mark G1
  **closed** iff the streamed leg lands in the 5–6 t/s band at ≤10 GB, bit-exact.
  *(measured numbers appended below the run.)*

**Measured (from `results/e37b/` artifacts — CLEAN, quiet box, avail 9.24 GB ≥ 8
verified; streamed-first so no page-cache warming; exact greedy, prompt A, N=20):**

| leg | hit | tok/s | RSS | decode | stall | avg_bw | per_stream_bw | hash |
|---|---|---|---|---|---|---|---|---|
| streamed (SLOTS=16, guard ON, cap held 16) | .847 | **4.14** | **8.69 GB** | 4.83 s | 3.56 s | 1404 MB/s | 209 MB/s | `e3fa62923ee35254` |
| resident | — | — | — | — | — | — | — | **skipped** (avail 11.56 < 12) |

- **Predicted vs measured:**
  - hit: pred ~.90 → **.847** (E28's .905 was a different/longer workload; this
    short prompt at 16 slots / N=20 sits ~.85, matching E37's loaded .855).
  - RSS: pred ~6.6 GB → **8.69 GB** — higher than predicted (I under-counted the
    prefill pool: `fill … pool 32 slots` + the 16-slot decode cache + 1.92 trunk),
    but **≤ 10 GB ✓** with ~1.3 GB headroom to spare.
  - tok/s: pred ~5–6 → **4.14** — **below band by ~0.9 t/s.**
  - decode bw: pred ~1.15 GB/s → **1.40 GB/s** (avg_bw) — clean box *beat* the
    anchor; ~2× the loaded-box 0.66. **Bandwidth is healthy, not the bottleneck.**
- **Per-token decomposition (clean — NO swap-fault term, as pre-registered)**:
  0.2415 s/tok = **miss-wait 0.178 s (74%)** + **true compute 0.0635 s (26%)**.
  Pure compute ≈ **15.7 t/s** (quiet box beats even E7's 0.095 s term). Contrast
  E37 (loaded): compute+faults 0.319 s of which ~0.224 s was swap-fault tax — that
  tax is **gone** here; the clean decode is **miss-wait dominated**.
- **Bit-exact gate**: streamed N=20 hash `e3fa62923ee35254` = the **identical** hash
  E37 produced at N=20 (loaded box, different slot trajectory) → compute is
  box- and cache-invariant. Resident hash not taken (skipped), so streamed==resident
  isn't shown *this* run; the hash matches the established exact reference and
  exact-mode==resident is the standing contract. **Green by reference.**
- **VERDICT — G1 NOT closed on this run.** Streamed clean = **4.14 t/s**, outside
  the 5–6 band (owner's close condition not met). It is **not** a bandwidth or a
  memory-budget failure: bw is healthy (1.40 GB/s) and RSS (8.69 GB) leaves ~1.3 GB
  under the 10 GB cap. It is a **hit-rate** shortfall — .847 vs the ~.90 the band
  needs — and decode is now miss-wait-dominated (74%). The obvious lever (raise
  slots into the ~1.3 GB headroom to lift hit toward .90) is **tuning, forbidden in
  this task** → candidate for a separate directive. **G1 stays OPEN**, one hit-rate
  rung short, on an otherwise-clean, bit-exact, ≤10 GB streamed baseline.
- **Resident/compute ceiling**: unmeasured this run (avail 11.56 < 12 gate);
  **E7 anchor ≥10.47 t/s stands** as the ceiling of record.
- **Artifacts**: `results/e37b/` — `streamed.{out,err}`, `summary.txt`, `gate.log`
  (resident leg skipped, not run).

### E37c. G1 close-out at matched N — the warming fix (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before the run.** Diagnosis overturned from artifacts
(senior lead): E37b's 4.14 t/s was **N=20 still warming**, not a slot ceiling. Proof
in-hand: (1) prompt-A hit curve **.677→.747→.832→.855** across N=1→20 is monotonic
rising, not plateaued; (2) `results/e28_20b_stream_s16.txt` is **N=64 steady-state**
— hit **.905**, decode **5.86 t/s** at SLOTS=16 — but on a *different* prompt
("why is the sky blue") and **ubatch=1**. So the E37b gap is **measurement length**,
not slot count. E37c re-runs at matched N=64.

- **Config**: one streamed leg, **N=64**, `SLOTS=16`, `PREFILL_SLOTS=64`,
  `ubatch=128`, guard ON, exact greedy, **prompt A** (merge sorted lists), quiet
  box (gate avail ≥ 8 GB, same self-guarding script pattern → `results/e37c/`).
- **Predictions:**
  1. **hit ≈ 0.90** — prompt-A warming curve plateaus toward the E28 .905 by N=64.
  2. **tok/s in the 5–6 band** — E28 reproduces at matched N (its 5.86 is the
     64-token average; extending E37b's warm curve: 20 tok in 4.83 s + ~44 tok at
     steady-state ~0.15 s ≈ 11.4 s → ~5.6 t/s).
  3. **peak RSS ≈ 8.7 GB (time -l) / ~9.3 GB (engine peak_rss), ≤ 10 GB ✓** —
     N-independent: the peak is the prefill transient (424 MB pool + ubatch=128
     activation buffers, unshed via lazy madv_free), not the decode cache (16 slots
     = 5.09 GB) which is already `cap=full` at N=20. phys_footprint ~6.7 GB is the
     true steady state (matches E28's 6.57). **Confirms the E37b 8.69 explanation.**
- **Hash anchor**: no prior N=64 *prompt-A* artifact exists (E28's
  `8c170e11c74c5ce8` is a *different* prompt → not an anchor). **This run sets the
  N=64 prompt-A reference hash.** Bit-exactness rides on (a) the established
  exact-mode contract (margin 0 = bit-reproducible = resident) and (b) demonstrated
  cross-config invariance — prompt-A N=20 hash `e3fa62923ee35254` is identical
  across loaded/clean boxes and slot counts 5/16/24.
- **Close condition**: if hit ~.90 and tok/s ∈ [5,6] at ≤10 GB → **G1 CLOSES**.
  Else report, **no tuning**. *(measured numbers appended below the run.)*

**Measured (from `results/e37c/` — CLEAN, quiet box avail ≥8 GB verified; N=64,
SLOTS=16, PREFILL_SLOTS=64, ubatch=128, guard ON, exact greedy, prompt A):**

| metric | value |
|---|---|
| hit | **0.861** (N=20 was .847/.855 → still-climbing warm curve confirmed) |
| **tok/s** | **6.14** — clears the 5–6 band ✓ |
| decode / stall | 10.42 s / 6.46 s |
| avg_bw | **1421 MB/s** (clean; ~2× the loaded-box 660) |
| **phys_footprint** | **6.73 GB** — *identical* to E37b → N-invariant ✓ ≤10 |
| peak_rss (engine) | 10.78 GB · ru_maxrss (`time -l`) 10.04 GB |
| text | coherent (correct two-pointer merge) |
| hash | `7fff2b7b9461da2a` (N=64 prompt-A **reference**, newly set) |

- **Predicted vs measured**: tok/s pred 5–6 → **6.14 ✓**; hit pred ~.90 → **.861**
  (climbing, prompt A plateaus a hair under the sky-blue prompt's .905); decode bw
  pred ~1.15 → **1.42 GB/s**. **RSS: I was WRONG that peak is N-independent.**
  phys_footprint IS N-invariant (6.73 = E37b's 6.73), but ru_maxrss/peak_rss
  **grows with N** (10.04/10.78 at N=64 vs 8.69/9.33 at N=20) — more of the mmap'd
  model file faults in as **reclaimable** pages over more tokens. Own the miss.
- **Per-token decomposition (clean, no swap-fault term)**: 0.163 s/tok = **miss-wait
  0.101 s (62%)** + **true compute 0.062 s (38%)**; pure compute ≈ 16 t/s.
- **Bit-exact**: no prior N=64 prompt-A anchor existed → `7fff2b7b9461da2a` is now
  the reference. Rides on the exact-mode contract + the demonstrated N=20 invariant.
- **The RSS-metric question (owner's ruling needed)**: the *speed* goal is **met**
  (6.14 ∈ band). The *footprint* goal depends on which "RSS" governs:
  - **phys_footprint = 6.73 GB** (macOS real memory: dirty + compressed + wired;
    what Activity Monitor shows and what costs actual RAM) → **≤10 GB, clears**,
    and it is **N-invariant** (6.73 at both N=20 and N=64).
  - **ru_maxrss / peak_rss = 10.04 / 10.78 GB** (includes clean, instantly-
    reclaimable mmap'd file pages) → **nominally over 10**, and **grows with N**
    without costing real RAM (the OS drops those pages for free under pressure —
    the E37 thrash happened on *phys_footprint* ≈ 11 GB, not on reclaimable pages).
  - **Recommendation**: the meaningful "≤10 GB RSS" metric is **phys_footprint
    (6.73 GB)** — it is what pressures RAM and it is N-stable; ru_maxrss overcounts
    reclaimable file cache and is unbounded in N. On that metric **G1 is MET**:
    ≤10 GB (6.73), 6.14 t/s, bit-exact, coherent.
- **VERDICT: G1 speed + real-footprint goals MET (6.14 t/s @ 6.73 GB phys_footprint,
  bit-exact, coherent).** Formal close deferred one beat to the owner solely to rule
  which RSS metric governs — because the *peak* metric used in earlier E-entries
  (`time -l`) reads 10.04 GB, 40 MB over the line, and I won't switch metrics to
  force a pass. **No tuning applied.** Artifacts: `results/e37c/streamed.{out,err}`,
  `summary.txt`, `gate.log`.

### Live-UI observation. Fast dial ~8.8 tok/s (non-bit-exact) (2026-07-21)
Owner ran the playground (`ui/app.py`) on gpt-oss-20b and measured **8.79 tok/s**
(and 8.58 on a follow-up) in the chat. **Logged as an OBSERVATION, not a
pre-registered measurement** — uncontrolled UI session, temp 0.80 (sampling),
growing conversation context, no fixed N. Config from the UI: **Fast mode**
(`LLMSTREAM_AGREE_TARGET=0.90`), **slots 24**, temp 0.80, long (~800-tok) warm
generation.

- **Consistent with E37c, not a contradiction — a different rung.** The Fast dial
  is **not bit-exact** (D3 adaptive-margin keeps the cached expert for ≤10% of
  routing decisions + temp>0 samples), so it sits **outside G1's exact clause**.
  Four factors explain 8.79 vs E37c's 6.14 exact: (1) Fast raises *effective* hit
  via routing substitution; (2) slots 24 > 16; (3) a fully-warm long generation
  (steady-state) vs E37c's N=64 *average* still carrying cold-start; (4) temp 0.80
  (negligible speed effect).
- **Both rungs honest**: Exact/bit-exact = **6.14 t/s @ 6.73 GB phys_footprint**
  (G1's contract); Fast (quality-labeled "keep ≥90%", not bit-exact) = **~8.8 t/s**
  — ~90% of the old ≥10 dream, on the relaxed contract.
- **TTFT (21.7 s / 47.6 s)** is *prefill* of the growing conversation ("reused 430
  ctx tokens"), a separate axis from the 8.79 decode number. Sidebar RSS 3.2 GB is
  the idle-between-turns footprint; it climbs toward ~6.7 GB during active decode.
- Does **not** change the open G1 call (the phys_footprint-vs-peak-RSS ruling on the
  Exact rung stands).

### Docs. README rewritten — colibri-class packaging, sluice-class honesty (2026-07-21)
Replaced the research-phase `README.md` with a product-facing one (studied
`raw/colibri/README.md` for structure; our voice stays measurement-first).
Sections: what sluice is (+ the ROLES-AND-STATE honest promise verbatim), headline
numbers (gpt-oss-20b: 6.73 GB / 6.14 t/s exact — E37c; 8.79 t/s Fast field-obs;
21.7 s cold TTFT — each citing its artifact, no projections), quickstart, the
quality dial, **honest limits** (dense=batch-only, Pillar 6=capability not speed,
MTP format ceiling, TTFT worst UX), how-we-compare (vs colibri/Ollama, corrected
CACHE_ROUTE wording, no strawmen), and a full `LLMSTREAM_*` env-var reference
(WARMPACK + EVICT=lfru marked gated/dark). UI screenshot slot reserved as an HTML
comment for Umar to supply. Docs only, no code. Prior detailed chapters remain in
git history + `docs/findings-phase0.md`.

### TTFT thread status — E41 / E41b, one place (updated 2026-07-21 15:15)
The multi-turn TTFT arc has three entries and it is easy to read them as three
attempts at the same thing. They are one chain:

| | what | state |
|---|---|---|
| **E38** | Diagnosis. Turn-2 TTFT 20.5 s; `diverged_at=296`, 219 tokens re-prefilled | **CLOSED, gate green** |
| **E39** | KV *persistence* across a server restart | **SHIPPED DARK, not bit-exact.** Identical text, differing `logits_hash`; iSWA hypothesis refuted from source; cause NOT established, stopped at 1 attempt per the standing rule. Does **not** help multi-turn TTFT (reuse unchanged at 296/79 either way) |
| **E41** | Prefix-stable *rendering* (append-only history) | **STOPPED by its own escape clause.** The gpt-oss template drops CoT from history by explicit design; three token-level breaks proven. No code written |
| **E41b** | KV *canonicalization* — path 2, owner-approved | **CODE WRITTEN, UNMEASURED.** Gate has not run |

**Why E41b is not "in progress" in any useful sense**: it has never executed. The
box has been between 4.9 and 7.3 GB avail all day against an 8.0 GB gate, so the
launcher has correctly refused three times. It is armed on a 12-hour long poll and
will run itself when the machine frees up. **Nothing about E41b — correctness or
speed — may be quoted until `results/e41b/summary.txt` exists.** The README row for
`LLMSTREAM_KV_CANON` says exactly that.

**The one thing E39 and E41b now share, flagged before E41b runs so it cannot look
like hindsight**: both depend on decoding a prefix in a different batch shape than a
fresh full prefill would use, which changes GEMM blocking and therefore FP
accumulation order. E39 already failed its hash gate for an unestablished reason.
E41b's pre-registration therefore ships a **control leg** (stock partial prefill vs
fresh full prefill) whose only job is to tell us whether a gate-1 failure is
E41b's fault or a pre-existing property of KV reuse itself. If the control diverges
too, the two entries collapse into one root-cause investigation, which is what the
owner directed.

### E41. Prefix-stable history — STOPPED: the template forbids it (2026-07-21)
**Task 1 of the G2 centerpiece. Outcome: the escape clause fired — reporting the
exact breaking tokens and stopping, with no code written and no workaround applied.**

- **Finding: template-conformant append-only rendering is IMPOSSIBLE for gpt-oss.**
  Not an engine limitation — the chat template drops chain-of-thought from history
  **by explicit design**. Its own comment, verbatim from the GGUF metadata
  (`tokenizer.chat_template`, 16 714 chars):
  > `{#- CoT is dropped during all previous turns, so we never render it for inference #}`
  > `{{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}`
- **The exact tokens that break it** (via `llama-tokenize --ids`, not inferred):

  | | token stream |
  |---|---|
  | template renders a past assistant turn | `[200006, 173781, 200005, `**`17196`**`, 200008, 160761, `**`200007`**`]` |
  | what the model actually generates | `[200006, 173781, 200005, `**`35644`**`, 200008, 34, 2824, 200007, 200006, 173781, 200005, 17196, 200008, 160761, `**`200002`**`]` |

  Mapping: `200006 <|start|>` · `173781 assistant` · `200005 <|channel|>` ·
  **`17196 final`** vs **`35644 analysis`** · `200008 <|message|>` ·
  **`200007 <|end|>`** vs **`200002 <|return|>`** (200002 is also the model's EOS).
  **Three independent breaks**: (1) the channel token — `final` vs `analysis`;
  (2) the **entire analysis block** (`<|message|>` + CoT + `<|end|>` +
  `<|start|>assistant<|channel|>`) exists in KV and is absent from the re-render;
  (3) the terminator — template `<|end|>` vs generated `<|return|>`.
  This is the token-level root cause behind E38's `diverged_at=296`.
- **Why no workaround was applied** (directive: *"no workarounds without a
  directive"*). Both viable paths change something the owner must decide:
  1. **Retain CoT in context** (never re-render the past; append only the new-turn
     delta). Gives true append-only and full KV reuse — but the model then sees its
     own chain-of-thought in history, which the template **explicitly forbids for
     inference**. That is a change to model input semantics, i.e. a potential
     quality change, and protocol #6 says quality claims need agreement/NLL attached.
  2. **Canonicalise KV at end of turn** (after generating, drop the assistant span
     from KV and re-decode the template's `final`-only rendering). This *is*
     template-conformant and would make the next turn's prefix match fully — it
     moves the re-prefill cost **off the TTFT critical path** to the end of the
     previous turn, rather than eliminating it. Costs one extra decode of the answer
     per turn and requires KV surgery.
  Neither is "prefix-stable *rendering*" as specified; both are workarounds, so both
  wait for a directive.
- **Nothing was built**: no `LLMSTREAM_STABLE_HISTORY` flag, no engine edit, no
  gate run — there is nothing to gate. The E38 telemetry that would have verified it
  (`ttft: diverged_at / kv_had / rerender_has`) is already in place and ready.
- **Recommendation**: path 2 (canonicalise KV) is the one I would take — it keeps
  the model's input exactly what the template intends, so it needs no quality
  re-validation, and it converts a blocking TTFT cost into a non-blocking one.
  Path 1 is faster still but cannot ship without an NLL/agreement battery.

### Packaging. Install script + model-library manifest — G2 task 5 (2026-07-21)
Docs/scripts only, no engine changes, run after the queue cleared. Drafted
`scripts/install.sh` and `packaging/models.json`.

- **Real defect found, and it invalidated a README claim I wrote yesterday**:
  `vendor/` is **gitignored with 0 tracked files** and there is no submodule, so a
  **fresh clone cannot build at all** — `scripts/build_driver.sh` links against
  `vendor/llama.cpp/build/bin`, which does not exist. Yesterday's README told users
  the CLI "auto-builds the engine on first run"; that is true only on *this* machine,
  where the vendored tree already exists. **Corrected**: the quickstart now leads with
  `scripts/install.sh` and states plainly why it is required.
- **`scripts/install.sh`** reconstructs what the clone omits: clone llama.cpp at the
  pinned **`b10064`** (the working tree reports `b10064-4-g…` = that tag plus our 4
  fork commits), apply `patches/llmstream.patch`, cmake-build the library, build the
  driver, create `.venv` with streamlit+psutil. Idempotent (every step skipped if
  satisfied), no destructive operations, and it **downloads no model** — it ends by
  pointing at `sluice estimate` first. Platform honesty: `build_driver.sh` hardcodes
  `-lobjc -framework Foundation`, so the script says up front that only macOS/Apple
  Silicon is tested rather than failing mysteriously on Linux.
- **`packaging/models.json`** lifts the library out of `cli/sluice`'s hardcoded
  REGISTRY into data: 4 models, each speed carrying a `source` pointing at a
  `results/` artifact or lablog entry, plus the measured free-RAM→speed curve
  (quiet 6.14 / light 4.89 / heavy 1.57, identical output hash in all three).
  **Not wired into the CLI** — that is a code change and this task was docs/scripts
  only; wiring is a separate directive.
- **Second defect logged, not fixed**: `cli/sluice`'s `olmoe-7b` URL uses a stale
  lowercase filename that **404s** (hit live during E37b). The corrected URL is
  recorded in the manifest's `known_issues` and `olmoe-7b.url`; fixing the CLI is a
  code change, deliberately not bundled here.

### E41b. KV canonicalization at end of turn — path 2, approved (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before any code and before any run.** The approved
continuation of E41: since the template forbids append-only *rendering*, make the
*KV* match what the template will render next turn, instead of the other way round.

- **Mechanism.** After a reply completes and before `<<<READY>>>` (i.e. while the
  user is reading — off the TTFT critical path), rewrite the KV from what the model
  actually generated
  (`<|start|>assistant<|channel|>analysis<|message|>COT<|end|><|start|>assistant<|channel|>final<|message|>ANS<|return|>`)
  to what the template will render for that same turn next time
  (`<|start|>assistant<|channel|>final<|message|>ANS<|end|>`). Concretely: extract
  the final-channel text from the generated span, re-apply the chat template with
  the assistant turn appended and `add_generation_prompt=false`, tokenize, `seq_rm`
  the divergent tail, decode the canonical suffix, and set `ctx_toks` to it.
- **Env gate**: `LLMSTREAM_KV_CANON=1`. Unset ⇒ the block never runs ⇒ stock
  behavior byte-identical (executable check on stdout, plus the bit-exact hash).
- **Why this and not retain-CoT**: it keeps the model's context exactly what the
  template says it should be, so no quality re-validation (protocol #6) is owed.

**Predictions (falsifiable, written first).**
1. **Faithfulness (gate 1, non-negotiable).** Turn-2 `logits_hash` on a canonicalized
   KV == turn-2 `logits_hash` from a fresh full re-prefill of the *same* rendering.
   *Falsifier*: any difference ⇒ RED regardless of speed; stop at diagnosis.
   **Stated risk, on the record before the run**: partial prefill changes the decode
   batch shape (25 tokens vs 515), which changes GEMM blocking and therefore FP
   accumulation order. If the hashes differ, that mechanism — not KV corruption — is
   the first suspect, and it is *the same suspect as E39's restore*. To separate the
   two I pre-register a **control leg**: stock server mode (canon OFF, which already
   reuses a 296-token prefix) vs the same fresh full re-prefill. If the control ALSO
   diverges, the divergence is intrinsic to partial prefill and predates this feature.
2. **Prefix match.** Turn-2 `reused` ≈ the full canonical history (~490 of 515
   rendered), vs 296 measured in E38. *Falsifier*: `reused` < 400.
3. **Re-prefill.** Turn-2 `reprefill` = the new user turn + generation prompt only,
   ~20–30 tokens, vs 219 in E38. *Falsifier*: > 60.
4. **TTFT collapse.** From E38's phase data: turn-2 prefilled 219 tokens in 20.49 s
   (10.7 tok/s true rate, per the protocol-#7 fix). At ~25 tokens and the slow end of
   the short-prefill curve (5–10 tok/s), predicted turn-2 TTFT ≈ **3–5 s**, down from
   **20.5 s**. *Falsifier*: > 10 s.
5. **Canonicalization cost.** One extra batch decode of the final block (~200 content
   tokens + 3 marker tokens) at the 18–23 tok/s prefill rate ⇒ **~9–12 s**, spent
   after `text:` is printed and before `<<<READY>>>`. *Falsifier*: it lands anywhere
   on the critical path (i.e. any increase in turn-1 TTFT or turn-1 `decode` time).
6. **Bit-exact gate** green with the flag unset: `fdf0f83dd70504c5`.

**Labels**: clean only. One commit. Gate-1 failure ⇒ report the divergence and stop,
for a single root-cause investigation covering E39 and E41b together.

**Run status 2026-07-21 09:15 — HELD, protocol #2.** Code written and building.
Measured avail 7.06 / 7.23 / 7.26 / 7.09 GB across four reads over a minute — steady,
and below the declared 8.0 GB gate. Nothing was run and nothing was committed
(protocol #4 forbids an engine commit before a green bit-exact gate, and the gate is
a run). No process was killed to make room: the memory is spread across the editor
and this session's own harness, not a stray. The box shows 1.65 GB swap in use and
393 361 pages in the compressor, i.e. it is already paging — exactly the state
protocol #2 names.

**Armed 2026-07-21 09:2x — detached self-gating launcher, same pattern as E37b/c.**
`results/e41b/run.sh`: stray check (protocol #1) → avail poll every 10 s for up to
120 s against the 8.0 GB gate, `ABORTED` + `DONE` written and no run if it never
opens → the three legs sequentially via `gate.py` → byte-identical-off executable
check (pre-E41b binary vs new, canon unset, `cmp` on stdout) → bit-exact gate. Both
the shell gate and `gate.py`'s own second-opinion check use the **same four vm_stat
buckets and the same queried page size**, because a second opinion on a different
yardstick would just abort runs the first gate had passed.

**First arming ABORTED on the gate 2026-07-21 — avail ~4.9 GB**, correctly, with no
run performed. The owner cannot free RAM during the workday, so the gate is not a
transient condition to retry past; it is the machine's daytime steady state.

**Re-armed 2026-07-21 14:31 — LONG POLL.** Identical legs and gates, only the wait
changed: every 60 s for up to 12 h, firing when avail ≥ 8 GB holds for **3
consecutive readings**. The hold requirement is the point of the redesign — a single
passing reading catches the spike when an app closes a window, and that collapses
again seconds later; a dip mid-leg is worse than never starting, because it produces
a contaminated number instead of an honest abort. Every reading is logged with its
streak count, so the morning artifact shows exactly when the box freed up and whether
it stayed free. A second stray check runs after the poll, since up to 12 h can pass
between the first one and the moment RAM is actually spent. Launched under
`nohup caffeinate -i`, no `-t`, so it outlives the IDE and the terminal.

### R0. Batch-shape invariance probe (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before the run.** Rung 0 of the E42 build plan, and
the shared dependency of three open threads: E39 (KV restore, failed its hash gate
for a cause never established), E41b (KV canon, gate pending), E42 (spec-dec).
Harness only — **the engine binary is unchanged**, so this cannot itself perturb
what it measures.

**Method.** `cparams.n_ubatch` comes from `argv[4]`, and the prompt is submitted as
one `llama_batch` that llama.cpp splits into `n_ubatch` pieces. Varying it varies the
prefill batch shape and nothing else; decode is always one token per pass, so any
downstream difference is inherited from prefill. Two families, because the prefill
pool is a second variable that must not be confounded with batch shape:
- **A (pool OFF)**: ubatch 1, 2, 4, 8 — pure shape variation through one code path.
- **B (pool ON)**: ubatch 1, 128 — 128 is **D12's path** (per-layer union > slots).

Pool logic engages only at `ne[1] > 1`, so **A1 and B1 run identical code and must
produce identical hashes** — a consistency check on the harness, not a result.

**Validity precondition, checked before any conclusion is drawn**: leg B128 must
reproduce E37c's published `7fff2b7b9461da2a` (same prompt, N=64, SLOTS=16,
PREFILL_SLOTS=64, ubatch=128). If it does not, the harness is not measuring what
E37c measured and the run is **void**, not merely negative.

**Two questions, deliberately reported apart** — they are not the same question and
E39 is the proof:
- **argmax stability** (token sequence identical) — what **spec-dec** needs.
- **bit equality** (`logits_hash` identical) — what **our gate** demands, strictly
  stronger. E39 failed here while its text matched.

`n_gen=1` legs answer "first-divergence position" directly: if prefill numerics
differ at all, the *first* sampled logits already differ, so divergence at step 1 is
the signature of a batch-shape effect rather than something that accumulates.

**Pre-registered consequences — both directions, written before the result.**
1. **Invariant (bit-equal AND argmax-stable)** ⇒ batch shape is exonerated.
   **E39's hash failure is then a real restore bug and gets REOPENED as one** — the
   comfortable explanation is gone. E42 keeps its Exact-tier claim and gate 1(b) is
   expected to pass. E41b's pending gate has one fewer excuse available to it.
2. **Argmax stable but NOT bit-equal** ⇒ E39's exact signature. Spec-dec would emit
   **identical text while failing a bit-equality gate**. This forces an owner call on
   what "Exact" *means* — identical output, or identical logits? Our gate currently
   demands the stronger reading. Either E39/E41b/E42 re-scope together, or the gate's
   definition changes; that is a product decision, not mine.
3. **Argmax flips** ⇒ bit-exact speculative decoding is **impossible** on this
   backend. E42 re-scopes to Balanced-tier pending owner call; E39 and E41b inherit
   the same verdict and one root cause covers all three. The property gets written
   into `docs/techniques.md` as a **measured engine fact**, not a suspicion.

**Instrumentation gap, declared up front rather than quietly dropped.** The directive
asked for **max |Δlogit|**. It is **not obtainable from this harness**: the engine
emits a digest (`logits_hash`) and token ids, never raw logits, so magnitude cannot
be recovered without an engine change — which this rung forbids. What exists and was
deliberately *not* used: `LLMSTREAM_DEBUG_HASH` prints a per-tensor FNV of every
`ffn_moe_*` intermediate and would localise a divergence to the first differing
(layer, node) — arguably more actionable than a magnitude. It is held for a follow-up
because enabling it changes which nodes the callback is asked about, and that is a
perturbation this probe must not carry.

**Gating**: fires only after `results/e41b/DONE` exists (E41b owns the quiet window;
two model processes at once is precisely protocol #1's prohibition), then waits for
a live engine to clear, then the standard ≥8 GB / 3-consecutive-readings poll.

### E37d. Realistic-load leg — G2 task 4 (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before the run.** Goal: the "with your apps open"
number for the README, so users see a speed that matches their real machine rather
than only a quiet-box best case.

- **Config**: one streamed run, **N=64, SLOTS=16, PREFILL_SLOTS=64, guard ON,
  exact greedy, prompt A, ubatch 128** — *identical to E37c* so the only variable is
  machine load.
- **Documented load profile (measured at pre-registration, not asserted)**:
  VS Code (5 `Code Helper` processes, ~1.0 GB RSS total) + the Claude Code agent
  (~0.54 GB) + system services. **avail 8.36 GB, active 4.97 GB.**
- **Honesty caveat, stated up front**: this is a **LIGHT** desktop load, *not* the
  owner's real working set. E37's contaminated leg ran at ~1.5 GB avail (Chrome,
  Meet, VS Code) and produced 1.57 tok/s; E37c ran quiet at ~9.2 GB avail and
  produced 6.14. Today's box sits near the quiet end, so this leg cannot be labelled
  "typical apps open" — it will be labelled **"light desktop load (VS Code + agent,
  8.4 GB avail)"**, and a true heavy-load leg needs re-running during the owner's
  workday.
- **Predictions**: directive's band **4.5–6.0 tok/s**; my point estimate is the
  **upper end, ~5.8–6.1**, because the load is light and E37c (same config, quieter)
  measured 6.14 — I expect a small deficit, not a large one.
- **Falsifier**: a result outside **4.5–6.5** falsifies both bands and gets reported
  as measured, with the load profile attached.
  *(measured numbers appended below the run.)*

**Measured (from `results/e37d/` — N=64, SLOTS=16, guard ON, exact greedy, prompt A;
load profile captured as an artifact at run time):**

| | E37 (heavy, contaminated) | **E37d (light load)** | E37c (quiet) |
|---|---|---|---|
| avail at run | ~1.5 GB | **8.37 GB** | ~9.24 GB |
| load | Chrome + Meet + VS Code | **VS Code (5 helpers) + agent** | none |
| decode | 1.57 tok/s (N=20) | **4.89 tok/s** | 6.14 tok/s |
| hit | .855 | **.860** | .861 |
| avg_bw | 660 MB/s | **1254 MB/s** | 1421 MB/s |
| stall | — | **8.64 s** | 6.46 s |
| phys_footprint | — | **6.73 GB** | 6.73 GB |
| hash | — | **`7fff2b7b9461da2a`** | `7fff2b7b9461da2a` |

- **Predicted vs measured**: directive's band 4.5–6.0 → **4.89 ✓ in band.** My own
  point estimate 5.8–6.1 → **WRONG (4.89)**. I assumed "light load ≈ quiet" and
  predicted only a small deficit; the real deficit was **20 %**. Owning it: the
  load→bandwidth sensitivity is **steeper** than I assumed.
- **Mechanism, isolated cleanly**: cache hit rate is **unchanged** (.860 vs .861), so
  the loss is **not** a caching effect — it is pure I/O. `avg_bw` fell 12 %
  (1421→1254 MB/s) and stall rose 34 % (6.46→8.64 s). ~1.5 GB less headroom is
  enough to measurably starve the streaming reads. This is E30's "memory headroom
  governs effective SSD bandwidth" reproduced at the *light* end of the curve, where
  I did not expect it to bite.
- **Bit-exact across load conditions**: `7fff2b7b9461da2a` is **identical** to E37c's
  quiet-box hash. Machine load changes speed, never output. `phys_footprint` is also
  identical (6.73 GB) — load- and N-invariant, as E37c found.
- **Labelling (honest)**: this is **"light desktop load (VS Code + agent, 8.4 GB
  avail)"**, *not* "typical apps open". The owner's real working set (Chrome/Meet,
  ~1.5 GB avail) is the E37 row at 1.57 tok/s. A true heavy-load leg still needs a
  run during the owner's workday; until then the README carries the light-load number
  with its profile attached, and the three rows above are the honest load→speed curve.
- **Artifacts**: `results/e37d/` — `streamed.{out,err}`, `load_profile.txt`.

### E39. KV-cache persistence — G2 task 2 (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before the run.**

- **Build**: `LLMSTREAM_KV_PERSIST=<path>`, **off by default** (unset ⇒ code never
  entered ⇒ byte-identical). Server mode only. Uses llama.cpp's maintained
  `llama_state_seq_save_file` / `llama_state_seq_load_file` (they persist the KV
  **and** the token list) rather than hand-rolling KV serialization — the E38 desk
  study's recommendation. Load at startup, save after every turn. **Resume is
  announced loudly on stderr** — colibri's measured failure was a *silent* resume
  that inherited 670 stale tokens and answered in the wrong language for a day.
- **PREDICTION CONFLICT, declared up front.** The directive predicts *"second-turn
  first-token drops from 47.6 s to seconds."* **E38's telemetry says persistence
  alone cannot deliver that**, and I will not quietly adopt a prediction my own
  evidence contradicts. E38 proved the second-turn cost comes from *within-session
  prefix divergence*: KV holds the harmony `<|channel|>` token the model generated,
  the re-rendered history does not, so the match breaks at the assistant boundary
  (`diverged_at=296`). Saving and restoring KV faithfully does **not** change what
  the re-render produces. So I register **both**:
  - **Directive's prediction (P-A)**: second-turn TTFT → seconds.
  - **My prediction (P-B)**: second-turn TTFT ≈ **unchanged** (~20 s in this config)
    with persist ON, because the divergence is upstream of persistence.
  - **Falsifier for P-B**: if second-turn TTFT drops to seconds with persist ON,
    I am wrong and P-A stands. Whichever way it lands gets reported plainly.
- **What persistence genuinely buys (and the gate)**: faithful **checkpoint /
  restore** of a conversation. **Gate = resumed-chat output byte-identical to
  unbroken-chat output** — i.e. run turns 1→2 in one process, versus turn 1 → save →
  exit → restore → turn 2, and require the turn-2 `logits_hash` **and** text to
  match exactly. That is a correctness gate and it is achievable regardless of the
  timing question.
- **Honest scope note**: the *time* win on multi-turn chat is bounded by the same
  divergence, so the real TTFT fix is **prefix-stable history** (stop re-rendering
  the assistant tail, or feed back the generated tokens verbatim). That is a
  separate task and I will name it as the recommended follow-up, not smuggle it in.
- **Byte-identical-off check**: unset ⇒ 0 new output lines + unchanged
  `logits_hash` vs the canonical `fdf0f83dd70504c5`.
  *(measured numbers appended below the run.)*

**Measured (from `results/e39/` — CLEAN, n_gen=60, SLOTS=16 guard ON, exact greedy):**

| arm | turn-2 hash | TTFT | ctx_held | rendered | reused | reprefill |
|---|---|---|---|---|---|---|
| A unbroken (feature OFF) | `fc1efa9e0bd36fcc` | 8427 ms | 356 | 375 | **296** | **79** |
| B2 resumed (feature ON) | `f40384bd332e7eab` | 6357 ms | 356 | 375 | **296** | **79** |

- **Byte-identical OFF: confirmed** (`fdf0f83dd70504c5`, unchanged).
- **Resume announced on stderr: YES** (colibri's silent-resume hazard avoided).
- **PREDICTION CONFLICT SETTLED — P-A refuted, P-B confirmed.** `reused` and
  `reprefill` are **identical (296 / 79)** with persistence ON and OFF. Persistence
  did **not** increase reuse and did **not** drop second-turn TTFT "to seconds";
  both arms still re-prefill the assistant tail, exactly as E38's divergence
  predicts. (The 8.4 s → 6.4 s difference is fresh-vs-warm process, not reuse —
  the reuse counters are byte-for-byte equal.) The smaller reprefill here vs E38's
  219 is just the shorter answer (n_gen=60 → ~60-token tail + new user turn = 79),
  which *further* confirms the mechanism: **reprefill ≈ assistant tail + new turn.**
- **GATE: split verdict, reported precisely rather than collapsed.**
  - **Owner's literal gate — "resumed-chat output byte-identical to unbroken-chat
    output" — PASSES**: the generated **text matches byte-for-byte**.
  - **sluice's bit-exact standard — FAILS**: the `logits_hash` differs
    (`fc1efa9e…` vs `f40384bd…`). Same argmax, different low-order logit values ⇒
    **the KV restore is not numerically bit-faithful.** Expert residency cannot
    explain it (exact mode is logit-neutral, invariant across slot counts in
    E34/E36/E37c), so it is the restore itself.
  - Under our own standard this is **RED**, and I am not shipping it as bit-exact.
- **Diagnosis, stopped at 1 attempt (standing rule: never debug past 2 unattended).**
  Hypothesis "the iSWA sliding-window cache isn't persisted" — **REFUTED by source**:
  `llama_kv_cache_iswa::state_write/state_read` with `flags=0` (what the file API
  passes) write **both** `kv_base` and `kv_swa`, so the full iSWA state does round
  trip. **Cause not established.** Remaining suspects, untested and unclaimed:
  KV cell placement differing after restore (changing float summation order), or a
  precision detail in the state serialization. Needs its own directive.
- **Disposition**: `LLMSTREAM_KV_PERSIST` is committed **off by default, byte-identical
  off, and explicitly marked NOT bit-exact when enabled.** Do **not** advertise it;
  do not enable it in the CLI/UI. The engine's stock bit-exact gate is green
  (`fdf0f83dd70504c5`), which is what authorises the commit.
- **The real TTFT fix is NOT persistence.** E38 + E39 together show the cost is the
  *re-render* discarding the assistant tail. The fix is **prefix-stable history** —
  either feed the generated tokens back verbatim, or have the server own the
  conversation and append only the new user turn. **Recommended next directive.**
- **Artifacts**: `results/e39/` — `summary.txt`, `gate.py`, `A_unbroken_off.*`,
  `B1_turn1_on.*`, `B2_resume_on.*`, `chat.kv`.

### E38. First-token decomposition — G2 task 1 (PRE-REGISTERED 2026-07-21)
**Pre-registration — written before the legs run.** G2 opens on TTFT, named in the
README as our worst UX number. Target: explain the **47.6 s second-turn** TTFT seen
in the live UI *from telemetry*, not inference.

- **Instrumentation** (`LLMSTREAM_PHASE_TIMERS`, off by default): chrono reads only
  — they cannot touch compute — around template render / tokenize / KV-prefix match
  / prefill / first sample, plus the **divergence point** (which token the KV held
  vs what the re-render produced). Printing is gated, so stock stdout is unchanged.
  **Inertness gate ALREADY GREEN before any leg**: OFF emits 0 `ttft:` lines and
  hashes `fdf0f83dd70504c5`; ON hashes **identically**; and OFF's line-shapes diff
  clean against `results/warmpack_ab/gate_off.out`, an artifact produced by the
  **pre-change** binary at the same config. Instrumentation provably inert.
- **Legs** (sequential, one process at a time, SLOTS=16 guard ON, CLEAN):
  L1 baseline single-turn · L2 server 2-turn (the anomaly) · L3 single-turn control
  whose rendered length ≈ L2 turn-2 (isolates re-prefill volume from the reuse path).
- **Hypothesis H1 (leading)**: the UI re-renders the whole chat each turn; gpt-oss
  harmony keeps only *final* channels, but KV holds what was actually generated
  (including `analysis`). The prefix match therefore breaks at the **assistant
  content boundary**, so every turn re-prefills the entire assistant tail. TTFT then
  grows with conversation length — a design consequence, not a slow kernel.
- **Predictions:**
  1. L2 turn-2 `reused` ≈ the system+user1 prefix (**not** ≈ rendered length);
     `reprefill` ≈ the whole assistant tail + new user turn.
  2. `diverged_at` lands at the assistant boundary: `kv_had` is a harmony
     special/analysis token, `rerender_has` is ordinary final-answer text.
  3. **prefill > 90 %** of turn-2 TTFT; template + tokenize + kv_match +
     first_sample together **< 10 %** (i.e. the cost is re-prefill *volume*).
  4. L3 (same rendered length, no reuse path) costs ≈ L2 turn-2's prefill → proves
     volume, not a reuse pathology.
- **Falsifier**: if turn-2 `reused` ≈ `rendered` (near-full reuse) yet TTFT is still
  large, H1 is **wrong** and the cost lives in another phase — the decomposition must
  then name that phase from the timers instead. Equally, if prefill is < 90 % of
  TTFT, prediction 3 fails and I report the dominant phase as measured.
- **Gate**: anomaly explained from telemetry + instrumentation provably inert.
  *(measured numbers appended below the run.)*

**colibri desk study (E38 deliverable — read-only, `raw/colibri/c/glm.c`):**
They solve the same TTFT problem with on-disk KV (`.coli_kv`), and their design +
their *documented failure* both transfer directly to E39:

- **Format**: append-only — header (magic `COLIKV1` + dims + `nrec`) then one record
  per position `[tok i32][Lc+Rc per layer][Ic per DSA layer]`. Only *new* positions
  are appended each turn. Cost ~182 KB/token (MLA-compressed).
- **Crash safety (worth copying verbatim)**: data is appended first and **`nrec` is
  rewritten last**, so a crash mid-append leaves the old count → the file stays
  coherent instead of half-written. Cheap, and it makes torn writes a non-event.
- **Strict load validation**: magic + *every* dimension field (layers, kv_lora,
  qk_rope, dsa, vocab) must match or it refuses — "ignoring .coli_kv from a
  different model or version" — plus a guard when the saved conversation exceeds
  the context window.
- **Their measured failure, which is the real lesson**: resume was **silent**. A
  chat silently inherited 670 tokens of an old Italian session; later replies came
  back in Italian and "explain fibonacci" was answered about the number 7. It
  "read as a quantization bug for a day." **Silent KV resume is a correctness
  hazard, not a convenience** — the engine must announce a resume loudly, and our
  off-by-default rule already puts us on the safe side of this.
- **Our advantage to exploit**: we sit on llama.cpp, which ships maintained
  sequence-state save/load APIs — so E39 should use those rather than hand-roll KV
  serialization, and spend its effort on the *validation + visibility* that colibri
  learned the hard way, and on the byte-identical resume gate.

**Measured (from `results/e38/` — CLEAN, quiet box, SLOTS=16 guard ON, n_gen=200,
system prompt 1457 chars):**

| leg | rendered | reused | reprefill | TTFT | prefill | template | tokenize | kv_match | first_sample |
|---|---|---|---|---|---|---|---|---|---|
| L1 baseline | 296 | 0 | 296 | 16079 ms | 16071 | 0 | 7 | 0 | 1 |
| L2 turn-1 | 296 | 0 | 296 | 16308 ms | 16305 | 0 | 2 | 0 | 1 |
| **L2 turn-2 (anomaly)** | 515 | **296** | **219** | **20507 ms** | 20489 | 1 | 16 | **0** | 1 |
| L3 control | 508 | 0 | 508 | 22409 ms | 22405 | 0 | 3 | 0 | 1 |

- **ANOMALY EXPLAINED FROM TELEMETRY (gate met).** L2 turn-2:
  `diverged_at=296`, `kv_had=200005 |<|channel|>|`, `rerender_has=200008`. The KV
  held **496** tokens (296 prompt + 200 generated) but only the first **296** — exactly
  the system+user1 prefix — matched. Divergence lands on the *first assistant token*:
  KV holds the harmony **`<|channel|>`** marker the model actually generated, while
  the re-render (which keeps only final channels) has something else there. So the
  **entire assistant tail is discarded and re-prefilled every turn**, and TTFT grows
  with conversation length. This is direct evidence, not inference.
- **Predictions 1–3: CONFIRMED.** (1) reused=296 ≈ system+user1, not ≈ rendered.
  (2) divergence at the assistant boundary on a harmony channel marker. (3) prefill
  = **99.91 %** of TTFT (20489/20507); template+tokenize+kv_match+first_sample = **18 ms
  (0.09 %)**. `kv_match` is literally **0 ms** — the prefix scan is free; all cost is
  re-prefill.
- **Scales to the field report**: our tail was 200 tokens → 20.5 s. The UI's turn-1
  answer was ~800 tokens (~4×) → the observed **47.6 s**. Consistent.
- **Prediction 4: NOT CONFIRMED — and the miss is informative.** I predicted L3
  (same rendered length, no reuse) would cost ≈ L2 turn-2's prefill, "proving cost
  scales with volume." Wall-clock was indeed similar (22.4 s vs 20.5 s) — but for
  **2.3× more tokens** (508 vs 219). Effective suffix throughput: L1 **18.4**, L2
  turn-2 **10.7**, L3 **22.7** tok/s. So prefill time is **not** proportional to
  suffix length, and my stated reasoning was wrong.
  - What differed: prefill pool fill was **73 %** of turn-2's prefill (14.88/20.49 s)
    vs **48 %** for L3 (10.70/22.41 s), and per-expert fetch rate differed ~2.7×
    (**79** vs **215** experts/s) despite turn-2 fetching *half* the experts
    (1181 vs 2300 over 46 vs 92 pool calls).
  - **Confound, stated rather than explained away**: L2 turn-2 runs in a *warm*
    process still holding turn-1's full decode cache and a 496-token KV, while L3 is
    a *fresh* process. Memory pressure and disk contention therefore differ. **This
    comparison is confounded; I am not claiming a mechanism from it.** Isolating it
    needs its own controlled leg — a separate directive, not an unattended debug
    spiral (stopped at 1 attempt, per the standing rule).
- **Defect found (telemetry honesty, logged not fixed)**: the engine prints
  `prefill: <s> (<n>/dt tok/s)` using the **full rendered n**, not the actually
  prefilled suffix. In server mode with reuse this overstates: turn-2 printed
  **25.14 tok/s** when the true suffix rate was **10.7**. Single-shot runs (reused=0)
  are unaffected. Left unfixed here to avoid bundling — **owner: worth a one-line
  separate commit.**
- **E40 precondition evaluated → NOT MET, E40 skipped.** The directive gates E40 on
  "E38's data says startup expert-fill is a top-2 phase." TTFT is ~100 % *prefill*,
  and prefill uses the **separate shared prefill pool**; the decode cache warmpack
  targets is *not in the TTFT path at all* (it governs decode hit, 0.907 in L1). So
  warm-start cannot be a top-2 TTFT phase. Skipping E40 as instructed — consistent
  with E34/E35 having already retired the warmpack benefit claim.
- **Instrumentation inertness (re-stated, green)**: OFF → 0 `ttft:` lines, hash
  `fdf0f83dd70504c5`; ON → identical hash; OFF line-shapes diff clean vs the
  **pre-change** binary artifact `results/warmpack_ab/gate_off.out`.
- **Verdict: E38 GATE GREEN** — anomaly explained from telemetry, instrumentation
  provably inert. **The fix is E39** (KV persistence / prefix stability), because the
  cost is re-prefilling a tail the KV already had.
- **Artifacts**: `results/e38/` — `summary.txt`, `legs.py`, `L1_baseline.*`,
  `L2_twoturn.*`, `L3_control.*`, `L2_answer1.txt`, `inert_{off,on}.out`.

### Doc. README rewritten — colibri-class packaging, sluice-class honesty (2026-07-21)
Full `README.md` rewrite (docs-only, no code). Structure studied from
`raw/colibri/README.md` (quickstart-first, one demo, feature table, env reference)
but written in our measurement-first voice, not theirs. Sections: what sluice is
(honest promise quoted verbatim from `ROLES-AND-STATE.md`) · headline numbers ·
quickstart · the quality dial · **honest limits** · how we compare · full
`LLMSTREAM_*` env reference.

- **Gate honored — every headline number cites an artifact**: gpt-oss-20b
  6.73 GB phys_footprint + 6.14 tok/s exact (E37c, `results/e37c/`), 8.79 tok/s
  Fast **labeled a field observation** (uncontrolled UI, temp 0.80), 21.7 s cold
  first token. No projections, no "up to". The `time -l` peak (10.04 GB) is
  reported alongside phys_footprint rather than hidden.
- **Honest-limits section is load-bearing** (our differentiator): dense = batch/
  fleet only; huge-model demos are **capability, not speed** (~0.1 tok/s, Pillar 6
  framing); MTP lost through GGUF = **format ceiling, not a TODO**; first-token
  latency named as our worst UX number today.
- **Comparison table has no strawmen**: colibri credited for native int8 MTP,
  O_DIRECT/io_uring, KV persistence, live-learning cache, packaging maturity;
  CACHE_ROUTE described with the **corrected** wording (opt-in, keeps true top-J,
  arXiv:2412.00099, ROUTE_AGREE telemetry). Ollama credited for packaging breadth.
- **Experimental flags marked dark** with one line each on why: `WARMPACK`
  (hardware-gated, no measurable win at cache≈top_k — E34) and `EVICT=lfru`
  (no win over LRU — E36).
- Opening reframed to the frontier class the product targets (DeepSeek-V4,
  Kimi K2.7/K3, GLM-class 700B+) while keeping the one concrete sparsity ratio
  attached to gpt-oss-20b, explicitly flagged as "the model our numbers come
  from" — **named targets, measured claims, never conflated.**
- UI screenshot slot reserved as an HTML comment (`docs/media/ui.png`) so no
  broken image renders until one is supplied.

### Gap logged. MTP-via-GGUF format ceiling (2026-07-21)
Documented in `techniques.md` → "Format ceilings": colibri ships a native int8
MTP head (their 2.2–2.8× throughput figure); MTP weights are **dropped in GGUF
conversion**, so no GGUF-served MoE can match native-MTP self-speculation. Logged
as a **format ceiling** (the cost of GGUF universality), not an engineering TODO —
so a colibri throughput comparison never reads as a fixable miss on our side.
Docs-only; no code.

### Doc fix. colibri CACHE_ROUTE wording corrected (2026-07-21)
- Backfill of the doc-only honesty task committed in `8f3ff6c`. An earlier draft
  in `docs/techniques.md` and `docs/findings-phase0.md` called colibri's
  CACHE_ROUTE "blind / quality unquantified / never measured." Verified against
  their code + docs: CACHE_ROUTE is **opt-in**, **always keeps the true top-J**,
  follows **arXiv:2412.00099** max-rank selection, and self-reports **ROUTE_AGREE**
  (overlap + KL vs true top-K). No mechanism claim of ours changed; our distinct
  piece remains the *precomputed offline* expert-similarity map. Wording fixed;
  no measurement was affected.

### Product arc 1: llmstream CLI + chat UI v2 + D10 closed (2026-07-20)
- **Name decided**: llmstream ("virtual memory for LLMs"). CLI in
  cli/llmstream: list / estimate / run / ui / pull / rm. The estimator is
  the honesty contract - probes THIS machine (RAM, 32x8MB F_NOCACHE random
  reads = the streaming access pattern) and prints per-mode tok/s scaled
  from the reference measurements, with provenance, BEFORE any download.
  First live probe immediately proved the concept: it measured 1.4 GB/s
  (the 20b download was competing for the disk) and scaled the streamed
  estimate down accordingly.
- **Chat UI v2** (ui/app.py, one-dark-pro theme): real chat surface
  (st.chat_message/chat_input, suggestion pills, streaming with thinking
  expander), the dial as battery-calibrated modes (Exact/Balanced/Fast)
  instead of raw knobs (raw margin+slots under Advanced), live machine
  stats as a 2s auto-refresh fragment, per-turn physics popover (hit,
  fidelity, footprint, thermal), engine card with Stop + orphan sweep.
  All safety plumbing carried over; server now runs with
  LLMSTREAM_PREFILL_SLOTS=1 + ubatch 128 (E27 fast TTFT in chat). Old
  examples/streamlit_probe/app.py retired (probe.py stays).
- **D10 CLOSED**: NSProcessInfo.thermalState via the objc runtime prints
  "therm: nominal|fair|serious|critical" in every artifact (build needs
  -lobjc -framework Foundation - scripts/build_driver.sh is now the
  canonical build). No cross-run speed comparison is blind to throttling
  again.
- gpt-oss-20b pull running in background (12.11 GB, user-approved; leaves
  ~4 GB free - keep-vs-delete decided after the E28 leg-2 measurement).

### E17. Deep-dive refutations: parallel part-fetch ≈ flat, E-cores hurt
- **Change**: (a) one I/O job per tensor extent (6-way parallel per expert
  miss, 10 workers, LLMSTREAM_IO_WORKERS); (b) LLMSTREAM_THREADS env.
- **Expected**: (a) big win — per-miss latency floor from serial preads;
  (b) 8 threads cut the 0.095s compute term 20-30%.
- **Result**: (a) m125_s8: 5.37→5.52 tok/s (+3%, noise); per_stream_bw fell
  1128→314 MB/s (device shares the same time across streams) — the ~11ms
  miss floor is DEVICE latency, not syscall serialization. REFUTED as a
  lever, kept as harmless (gate bit-exact). (b) 8 threads: 5.52→3.90 and
  1.71→1.31 — WORSE in mixed I/O+compute (P/E straggler effect); only the
  pure-compute ceiling gains: 10.47→11.12 tok/s. Default stays 4 threads.
- **Conclusion**: decode speed on this machine now reduces to ONE variable:
  miss count. Remaining unplayed cards: static-table prefetch (needs gpt-oss
  routing-trace predictability measurement) and agreement-feedback adaptive
  margin (quality held by live measurement, speed taken where fidelity allows).

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
| D3 | Margin units are per-family (0.02 meant something on OLMoE probs; no-op on gpt-oss logits) | Why? Mask threshold lives in the gating score's native scale. A universal product cannot ask users to know gating families. | CLOSED (E20): LLMSTREAM_AGREE_TARGET — a family-agnostic routing-fidelity floor (fraction of true top-k kept). The controller (stream_run.cpp:145/518) nudges margin multiplicatively (scale-free), landing on logit-scale margins for gpt-oss and prob-scale for OLMoE/Qwen. This ledger row lagged the code until 2026-07-21 (caught in review — same class as D11/D13); D3 also surfaced as the CLI's default quality dial that day (cli/sluice modes → AGREE_TARGET). |
| D4 | Quality was judged by eyeball | Why unacceptable? m1.5 rung *looks* fine until the last tokens; template mismatch (E8) fools the eye both ways. | Fixed by E9 (agreement + NLL); sweep running |
| D5 | At high hit rates, stall shrinks but decode doesn't reach ceiling | Where does the time go? Not device reads (read_work now measured), suspicion: fetch-path serialization / page faults / repack. | Decompose with E9 counters on clean re-runs |
| D6 | All results = one English prompt, 64 tokens | Why risky? Routing locality is topic-dependent; margin cliff may move on code/multilingual/long-context. | Prompt battery + 1k-context runs queued |
| D7 | prefetch_issued=0 in every gpt-oss run | Why off? We disabled it for first runs (phase-0 rule: prefetch pays only below hit-EMA 0.80). At slots8 m1.25 hit=.885 → gate would keep it off. Is the 0.80 gate right for THIS regime? | Re-test: force prefetch on at slots8 m1.25 |
| D8 | Raw prompts to a chat-trained model | See E8. | `LLMSTREAM_CHAT=1` built; use in all future quality runs |
| D9 | Universality boundary: dense models | A dense 72B touches all weights every token — nothing to stream selectively; that is physics, not engineering (docs/dense-strategy.md). Scope stated honestly: MoE-first. | M3: DeepSeek family next (shared+routed experts) |
| D10 | Thermal never logged (fanless chassis) | Are late rungs slower because hot? Counter-evidence: fastest rungs ran last. Still unproven either way. | Add powermetrics logging to protocol |
| D11 | Guard floor hard-coded 4; fix ledger claimed top_k+1 had landed — it hadn't | Why fatal? cap < top_k can never seat one token's experts → assign_slot −1 → exit(1): pressure becomes availability loss on top-8 families. Why unnoticed? gpt-oss is top-4 — cap 4 sits exactly on the pigeonhole boundary and survives. Found because battery m0.5 logs showed cap 4 vs the claimed floor 5. | FIXED (E19): atomic top_k, floor top_k+1, post-load re-clamp. Lesson re-learned: verify the artifact, not the fix ledger |
| D12 | Prefill with n_ubatch>1 + per-layer union > slots has no path (assign_slot exhausts victims → exit(1)) | Why latent? All current runs use ubatch=1. Boundary documented while scoping expert-major prefill (E24), before it bit anyone. | Any ubatch>1 config must clamp or split; owned by the prefill-port arc |
| D13 | E26's pre-registered check "skip_fills > 0 every skip rung" was unverifiable — counter printed only in the generation path, NLL path silent | Why? Two separate metrics print sites; the new counter was wired into one. Same failure class as D11: the check you demand must be emitted by the path you run. | FIXED (E26): skip_fills added to the NLL print block; visibility-only, rebuilt + gated after all same-binary runs finished |
| D14 | First prefill-pool build read garbage on OLMoE: layers 0–2 sane, layer 3+ degenerate routing (ids 0,1,2,…) | Why garbage? Pool tensors were typed from layer 0's meta, but Q4_K_M files mix types per layer (ffn_down: Q6_K on L0/L1/L4, Q4_K on L2/L3) — L2's Q4_K bytes dequantized as Q6_K corrupted the hidden state, compounding into degenerate top-k by L3. Why did decode never hit this? Decode slots are per-layer typed. Caught by the E27 bit-exact gate before any artifact was quoted. | FIXED (E27): one pool per (kind, quant-type) variant; graph picks by the layer tensor's own type, driver resolves identically and verifies stride per layer. Gate PASS b6869f5b6ef36376 |

## Standing protocol (enforced from 2026-07-19)
1. One model process at a time; check for strays before launch.
2. No rung starts under memory pressure (script guard + engine guard).
3. Artifacts labeled clean/contaminated; contaminated numbers never quoted.
4. Bit-exact OLMoE gate before every commit touching fork or driver.
5. Every experiment → lablog entry same day, prediction written before result.
6. Quality claims only with agreement/NLL attached (eyeball is a smell test).

### Doc/packaging entries — 2026-07-21 (non-engine queue, E41b waiting)
- **Reasoning-effort control** (commit `600fcd5`). UI dropdown + `sluice run
  --reasoning`, replacing a hand-edited "Reasoning:" line in the system prompt and a
  hardcoded CLI string. Default Low is byte-identical to what shipped, asserted at
  runtime and in static checks. **Hazard caught during the work**: `results/e38/legs.py`
  and `results/e41b/gate.py` scrape `DEFAULT_SYSTEM` out of `ui/app.py` by regex and
  fall back **silently** to "You are a helpful assistant." on a miss. Templating the
  reasoning level into that literal would have quietly changed the system prompt of
  the E41b run that is armed and waiting, destroying its comparability with E38. The
  literal is left byte-identical and the level is stripped/re-appended at runtime.
  Live smoke test PENDING until after E41b's window.
- **Packaging pass 2** (commit `7072398`). `scripts/install.sh` now verifies its own
  output without a model (driver runs, prints usage, proves libllama resolved via
  rpath), checks each linked library by name, and aligns its cmake flags with
  `cli/sluice`'s bootstrap. `patches/llmstream.patch` was confirmed to apply cleanly
  to a **pristine b10064** via a local git worktree — no network, no model. New
  `scripts/check_urls.py` HEAD-checks every model URL and stamps date+status into the
  manifest.
  **Two real defects found on its first run**, both invisible until a user pulls:
  (a) `gpt-oss-120b` 404'd — the repo consolidated its 3-part split into one file, so
  `url` and `file` had silently disagreed; (b) `bytes` were rounded, and since
  `cli/sluice` treats `size < bytes*0.99` as a partial download, olmoe's true
  4 213 512 192 sits below the rounded 4 300 000 000×0.99 — **a complete download
  would have been re-pulled forever.** All sizes now come from Content-Length.
  This is the same defect class as the olmoe 404, which is the argument for the
  checker existing at all: nothing was checking, so nothing was found.
- **Docs hygiene**. G1 close-out drafted above, marked awaiting owner metric sign-off.
  TTFT thread status (E38/E39/E41/E41b) consolidated. README gains the
  `LLMSTREAM_KV_CANON` row, marked UNMEASURED with its gate explicitly pending.
- **No model process was launched for any of the three.** The armed E41b launcher
  owns the next quiet window; every check above is static analysis, HEAD requests, a
  local worktree, or one no-argument binary invocation that exits immediately.

### E42 prep. Speculative decoding — DESIGN STUDY, no code, no runs (2026-07-21)
Full document: `docs/e42-speculative-design.md`. Read-only: every number is read from
an existing artifact or derived from it, plus a GGUF **metadata header** parse (a few
KB, no model load). No model process was launched — E41b's armed launcher still owns
the next quiet window.

- **Headline, and it inverts the obvious intuition**: on an MoE with SSD-streamed
  experts, **short drafts win and long drafts lose.** Batched verification must
  materialise the *union* of experts routed by all K drafted positions, so miss-bytes
  grow with draft length while committed tokens grow far more slowly. Modelled tok/s
  is **monotonically decreasing in K** at every acceptance rate; **K=2 is optimal**.
- **This contradicts llama.cpp's own guidance** (`docs/speculative.md`: *"MoEs require
  long drafts"*, sample `n-max 64`), which is written for MoEs held in RAM where a
  wider union costs only FLOPs. For us the union IS the cost. Logged as prediction P2
  precisely so we find out who is wrong. Two independent papers (MoE-Spec 2602.16052;
  Cost-Aware Spec-Dec for MoE 2607.12696) formalise the same tension, which is
  reassurance that this is a real MoE effect and not an artefact of our fit.
- **The union number is measured, not assumed**: E37c's own log line
  `pf_calls=23 pf_experts=502 union avg=21.8 min=17 max=27` is a 23-token batch
  touching 21.8 of 32 experts per layer. Independent routing would predict 30.5, so
  **routing locality is real and worth ~30% of the union.** Fitted U(K) ≈ 4·K^0.541
  on two points (U(1)=4 by construction, U(23)=21.8 measured) — flagged in the doc as
  its own weakest link, and pre-registered as P1 rather than asserted.
- **Projected payoff, bounded honestly**: 8.27 tok/s at the acceptance the only real
  gpt-oss draft advertises (a≈0.72, K=2, no batching benefit) = **1.35×, short of 10.**
  Crossing 10 needs a ≥ 0.90 (then 10.02 falls out at g=1) **or** measured GEMM
  batching efficiency ≥ 1.61. Both plausible, neither established. The study does not
  claim 10+; it claims 10+ is for the first time inside reach of a buildable
  mechanism, and names the two measurements that decide it.
- **Draft candidates**: exactly one real option exists —
  `RedHatAI/gpt-oss-20b-speculator.eagle3` (854M, 1 layer, inherits the target's
  tokenizer at conversion, llama.cpp supports it by name). Every community
  "pruned gpt-oss" is **expert**-pruned: perfect vocab match, and useless as a draft,
  because `num_experts_per_tok` and depth are unchanged so active params are
  unchanged. No layer-pruned gpt-oss exists. `--spec-type ngram-mod` needs no model
  at all (~16 MB) and llama.cpp lists reasoning models that repeat their thinking as
  its target case — which is exactly harmony's analysis→final pattern.
- **RAM tax is first-class here**: 1 expert slot = 319 MB, so a 0.55 GB Q4 draft costs
  **1.7 of our 16 slots** — it raises the very miss rate spec-dec is trying to
  amortize. A +30% miss regression eats over half the win. That is why ngram-mod
  (0.05 slots) is rung 5 and EAGLE-3 is rung 6, not the reverse.
- **D12 decides the architecture, not preference.** Batched verification IS the
  `n_ubatch>1` case, and D12 says union > slots exhausts victims → `exit(1)`.
  Verification must therefore run through the **prefill pool** (32 slots, 424 MB,
  type-variant aware per D14) and not the decode cache. Pool headroom covers K up to
  ~23; the decode cache would overflow at K≥16 — a second independent reason long
  drafts are wrong for us.
- **Vocabulary constraint verified locally**, not taken from a model card: our GGUF
  reports `tokenizer.ggml.tokens = 201088`, pre `gpt-4o`. llama.cpp's
  `common_speculative_are_compatible` compares token *text* for every id and hard
  throws, so "same tokenizer family" does not pass.
- **Bit-exactness splits cleanly, and lands on an already-open problem.** llama.cpp
  verifies by exact match against the *target's own sampler* (not Leviathan rejection
  sampling), so the committed sequence is algorithmically identical for any draft
  quality — **Claim A**. But the target's logits come from a K-token batched pass
  instead of K single-token passes, so bit-exactness reduces to **Claim B**: argmax
  must be invariant to batch shape. **That is the same suspect as E39 (failed) and
  E41b (pending)** — three features now depend on one unestablished property.
  **Recommendation: resolve the batch-shape determinism question BEFORE building E42**,
  because a negative answer re-scopes spec-dec from an Exact-tier feature to a
  Balanced-tier one, and that is a product decision the owner should make with the
  fact in hand rather than after a build.
- **Instrumentation prerequisite found**: `logits_hash` currently hashes every
  generated step. Under speculation a pass yields logits for K positions of which only
  some commit, so the hash must be redefined over **committed positions in commit
  order** or the gate would manufacture a false failure. Its own rung (R4).
- **Build plan**: 8 rungs, R0–R7, each separately greenlightable. **R0–R2 touch no
  engine code** — they are measurements on what we already have and can kill the idea
  before a line is written. Recommended first directive is **R0** (batch-shape
  determinism), because it is shared with E39 and E41b and its answer changes what
  E42 *is*.
- **One process error caught mid-study, recorded because it nearly shipped**: the
  first payoff table divided an already-in-seconds quantity by 1000, making I/O 1000×
  too cheap and projecting 14–73 tok/s. Caught by a unit check that replays the model
  at K=1 and demands it reproduce the measured 6.14 tok/s. That check is now written
  into the doc as a required step, not an optional one.

### Tasks D/E/F — client contract, doctor, and the gate as one command (2026-07-21)
Non-engine work while E41b and R0 wait on the RAM gate. No model process launched.

- **D. canon_reply client contract, DARK** (`4ac08ca`). Both clients stored the
  wrong assistant text, and the CLI worse than the UI: its history entry strips
  every harmony marker, which **concatenates the analysis and final channels** —
  precisely the text the template will not render next turn, guaranteeing a full
  re-prefill. Now `text` is what the user sees and `canon` is what the engine sees.
  Mismatch raises a **visible** warning in both surfaces, because the only symptom
  of getting this wrong is "it got slower", the same invisible-degradation class
  that cost a day on E38. Pass-through only, no UI control and no CLI flag, since
  E41b's gate has not run. `sluice serve` is deliberately untouched: its history
  comes from the HTTP request, so the client owns it and the contract cannot be
  enforced server-side without an API change. **Live smoke test pending.**
- **E. `sluice doctor`** (`b0c331f`). The campaign's lessons as user-facing tooling:
  avail RAM with the page size **queried** (hardcoding 4096 is the bug that read
  2.58 GB on a machine with 10.7 GB free), swap-in-use, disk bandwidth, model
  completeness on the exact-bytes logic, and the stray-engine check — **protocol #1
  promoted from a lab rule to a product feature.** Verdict is go/no-go plus the
  predicted tok/s per mode, and when RAM is low it prints the measured load curve
  next to where the user actually is. Verified by running it: correctly reports
  **NO-GO** on this box at 3.28 GB avail / 3.10 GB swap.
  *Bug found while wiring it*: `re` was imported only inside a function, so
  `avail_ram_gb`'s module-scope use would have raised NameError on first call. My
  first static check passed it **falsely**, because `"import re"` is a substring of
  the local `import re as _re`. The check was wrong before the code was.
- **F. `make gate`** (this commit). The standing protocol as one command, so no rung
  hand-rolls it again. That hand-rolling is exactly what put **two different
  definitions of "available RAM" in the tree at once** — E37b/c summed four vm_stat
  buckets with a queried page size; `results/e41b/gate.py` summed three with 4096
  hardcoded. One run could pass one gate and abort on the other. `scripts/lib/preflight.sh`
  is now the single definition and future rungs source it.
  Static legs: syntax of every shipped script, manifest JSON, engine compiles, and
  **no undocumented engine flags** (42 `LLMSTREAM_*` vars, all currently in README).
  Model legs **self-defer** with an explicit "PENDING MODEL WINDOW" and the verdict
  is **GATE PARTIAL — this is NOT a green gate**, so a deferral can never be
  mistaken for a pass.
  Two things verified rather than assumed: (a) the build check compiles to a **temp
  path**, because rewriting `csrc/stream_run` while E41b and R0 are armed against
  that exact path could hand a launcher a half-written binary — confirmed by md5
  before/after; (b) **fault injection** — removing the `LLMSTREAM_KV_CANON` row from
  the README made the gate go **RED** and restoring it returned it to PARTIAL. A
  gate nobody has seen fail is not known to work.

### E41b RUN 1 — VOID for gate 1. Harness bug, not an engine result (2026-07-21 21:46)
The gate opened at 21:38 (avail 8.08 GB, 3/3 streak) after ~7 h of waiting, and the
legs ran clean. Then the summary printed `GATE 1: RED`. **That verdict is not real
and must not be quoted.**

- **What went wrong**: `field()` used a bare `re.search`, so `^` anchored to the
  start of the whole block rather than to a line. One call site had noticed and
  passed an inline `(?m)`; the one building **leg A's turn-2 history had not**. So
  `^canon_reply:` never matched, the silent `default=""` was returned, and **leg A
  ran with an EMPTY assistant turn** — a different conversation from B and C.
- **How it was caught**: not by the summary, which looked confident, but by the
  rendered counts. **A rendered 315 tokens; B and C rendered 421.** The premise of
  the whole design is that all three legs render byte-identically so that only the
  KV path differs. It was violated, and nothing checked.
- **Two bugs, not one.** The missing `re.M` is the proximate cause; the deeper one is
  a helper that **silently substitutes a default for a value the run depends on**.
  Fixed both: `field()` is now always multiline, and anything load-bearing goes
  through `require_field()`, which aborts rather than inventing a value. The summary
  now checks the premise first and prints **VOID** — never a verdict — when the
  renderings disagree.
- **The "inert: DIFFERS" line was ALSO a harness flaw.** It compared full stdout,
  which includes wall-clock timings, prefetch race counters and `peak_rss` — all of
  which differ between two runs of *the same* binary. Re-filtered to the
  deterministic lines, run 1's own artifacts are **IDENTICAL**
  (`logits_hash=74a27209c03b5503`, same `mode=`, same `text:`). **Protocol #3 passed.**
  `scripts/gate.sh` shipped the same flawed comparison this afternoon and is fixed too.

**What run 1 DID establish, and is quotable:**
1. **CONTROL: B ≠ C. This is the headline.** Legs B and C both rendered 421 tokens
   from the identical string; B reused a 296-token KV prefix (canon **OFF** — stock,
   shipping behaviour) and C prefilled all 421 fresh. Hashes `81e84aa667640f55` vs
   `865919727c751b58`. **Stock KV reuse does not reproduce a fresh full prefill.**
   That is E39's signature, present in the default path, with no new feature
   involved — it predates E41b entirely.
   *Caveat stated before anyone leans on it*: B and C differ in **two** ways at once
   — reused-prefix vs fresh, and prefill batch shape (125 tokens in one batch vs 421
   split into four ubatches). **R0 is built to separate exactly those.** This makes
   R0 more valuable, not less.
2. **The canonicalization mechanism works**: `held=496 kept=296 dropped=200
   decoded=108 canonical=404` in **13.23 s, after the reply was printed** — off the
   TTFT critical path, as designed. Prediction 5 said ~9–12 s; measured 13.23, a
   miss on the high side.
3. **Bit-exact gate GREEN**: `fdf0f83dd70504c5`.
4. Predictions 2/3/4 (kv_match, reprefill, TTFT collapse) are **untested** — leg A's
   `reused=297 reprefill=18 ttft=6.7 s` came from a 315-token render, so its low
   re-prefill is partly just a shorter prompt. No collapse claim is supported.

**Run 2 armed**, chained behind R0 (which now owns the window), with the premise
check, `require_field`, and the corrected inert comparison in place. Run 1's
artifacts are preserved under `results/e41b/run1_void/` with `VOID.md` explaining
why they must not be cited as a gate result.

### E41b review follow-up — echo asymmetry, the control, and a TTFT correction (2026-07-21)
Answers to the four review points. No re-runs: R0 owns the window.

**(1) Echo asymmetry, in token ids** (engine telemetry, no model load):

| leg | ctx_held | rendered | reused | reprefill | diverged_at | KV had → re-render has |
|---|---|---|---|---|---|---|
| A | 404 | **315** | 297 | 18 | 297 | `410 \|**\|` → `200002 \|<\|return\|>\|` |
| B | 496 | **421** | 296 | 125 | 296 | `200005 \|<\|channel\|>\|` → `200008 \|<\|message\|>\|` |
| C | 0 | **421** | 0 | 421 | 0 | — |

At position 297 leg A's KV holds token **410 = `**`** — the first token of the reply
body (`**How Earth Came to Be**`) — while its re-render has a special token, because
the assistant turn was empty and the render jumps straight to the terminator. That
is the empty-echo signature at token level. Leg B shows the ordinary E38 break:
KV `<|channel|>` (analysis) vs re-render `<|message|>` (final only).

**TTFT ACCEPTANCE MUST BE WITHDRAWN — the mechanism is NOT demonstrated.** The review
accepted 6.7 s vs 12.2 s as "mechanism works". The same table refutes it:

- Canonicalization moved the prefix match from **296 → 297 tokens. One token.** If the
  mechanism had worked, leg A would have reused ≈ **404** (the full canonical KV);
  it reused 297 and diverged at exactly the place stock diverges.
- A's re-prefill of 18 tokens is `315 − 297`, i.e. **small because the prompt was 106
  tokens shorter**, not because the KV matched further.
- So the 6.7 s is a shorter-prompt artifact. **Predictions 2, 3 and 4 remain untested.**
  What *is* established is that the canon machinery executes correctly and costs
  13.23 s post-reply — mechanism *present*, benefit *unmeasured*.

**(2) CONTROL verified — and it is stronger than the E39 signature.** B and C both
rendered **421** tokens from the identical string; only `reused` differed (296 vs 0).
Their **generated text is NOT byte-identical** — it diverges at character 101, inside
the analysis channel:
- B: `…They previously asked "Explain how the Earth formed and why it…`
- C: `…They previously asked for explanation of Earth formation and w…`

E39 had *identical text, different hash*. Here **stock partial prefill changes the
answer itself.** Hashes `81e84aa667640f55` (B) vs `865919727c751b58` (C).
**Quotable, with its confound named**: B and C differ in two ways at once — reused
prefix vs cold, **and** prefill batch composition (B's 296 reused tokens were computed
inside turn 1's 296-token prefill; C computed the same positions inside a 421-token
prefill split into four 128-ubatches). **R0 separates exactly these.** Filed below as
its own entry.

**(3) Inert check re-scoped** to semantic lines (`mode=`, `logits_hash=`, `text:`,
ttft/canon) in both `results/e41b/run.sh` and `scripts/gate.sh`. Re-checked against
run 1's own artifacts: **IDENTICAL**. Tonight's DIFFERS was pure telemetry noise.

**(4) Leg D added, not run.** Replays run 1's exact 315-token rendering as a fresh
single-shot and compares against run 1's leg-A hash `0101f7d30ad830a4`. Same prompt,
two KV paths — the faithfulness question gate 1 was meant to ask. Voids itself if it
does not reproduce 315 tokens.

### FINDING: stock KV reuse is not bit-faithful, and changes output (2026-07-21)
Filed separately because it **outranks the feature that surfaced it** and is
independent of it: canon was **OFF** in both legs.

Identical 421-token prompt. Reuse a 296-token KV prefix → one answer; prefill all 421
cold → a **different** answer. This is default, shipping behaviour on every multi-turn
chat, and it means a conversation's replies depend on how the KV was assembled.
Evidence: `results/e41b/run1_void/{B_canon_off,C_fresh}.out`.

**Not yet root-caused, and two candidates are confounded** (reused-prefix vs batch
composition). **PENDING R0**, which varies batch shape alone and is running now. Do
not act on this until R0 lands; do not quote it without the confound.

### OWNER RULING (conditional on R0) — chat default + strict switch (2026-07-21)
Recorded before R0 lands so the condition is on the record, not reconstructed after.

- **Chat default stays the fast path with cache reuse.** Documented honestly:
  *"answers are full-quality and deterministic per session; bit-reproducibility
  requires strict mode."*
- **Strict switch** (`--strict-exact` / `LLMSTREAM_STRICT_EXACT`): guarantees
  bit-identical-to-fresh output by disabling prefix reuse in chat. Priced in help
  text as slower per turn. **Quality is identical in both modes — the switch buys
  reproducibility, not correctness.**
- **Condition**: R0 must confirm the batch/prefix mechanism. If R0 **refutes** it, the
  ruling goes **dormant** and E39/E41b reopen as ordinary bugs.
- **Sequencing**: R0 → (ruling implemented **or** dormant) → E41b run 2 + leg D.
  Nothing implemented yet; R0 has not landed.

### E41b: the token-410 divergence — canon's benefit may be STRUCTURALLY zero
The owner asked for this explanation before run 2, on the grounds that "the canon
benefit is zero until that's solved." Working it produced a worse answer than
expected, and one derivable from the artifacts alone — no model, no tokenizer.

**Arithmetic** (`run1_void/A_canon_on.out`):
- turn-1 `rendered=296` ⇒ prompt occupies indices 0..295, ending `<|start|>assistant`.
- `canon: kept=296 … canonical=404` ⇒ canonical[0..295] == that prompt; canonical
  [296..403] are the 108 appended tokens.
- `diverged_at=297` reports `ctx_toks[297] = 410 |**|`, and `canon_reply.txt` begins
  `**How Earth Came to Be**` ⇒ **the reply body starts at canonical[297]**.
- Therefore **exactly ONE token** sits between `<|start|>assistant` and the content:
  canonical[296].

`<|channel|>final<|message|>` is **three** tokens (200005, 17196, 200008). Three cannot
occupy one slot. **So the canonical render does not contain `<|channel|>final<|message|>`
before the content.**

**Hypothesis that fits both observations**: with `add_generation_prompt=false` and the
assistant message **last**, the template renders it as
`<|start|>assistant<|message|>CONTENT<|return|>` — one marker token at 296, and a
`<|return|>` terminator. This also explains the *other* half of the divergence: leg A's
turn-2 render has `200002 |<|return|>|` at 297, exactly where an **empty** content
would put the terminator.

**Why this is worse than a harness bug.** E41 established that a **past** assistant
turn (one followed by a later message) renders as
`<|start|>assistant<|channel|>final<|message|>CONTENT<|end|>`. If the **last** assistant
turn instead renders with one marker and `<|return|>`, then the canonical form the
engine writes into KV is **not a prefix of what the next turn will render** — the two
disagree at the marker (position 296) and again at the terminator. The prefix match
would die at ~296 **no matter how correct the client echo is.**

That is precisely what was measured: canon moved the match 296 → **297. One token.**
The harness bug is then not the cause of the null result; it merely **hid** it.

**Consequence for the plan**: E41b run 2 as specified may be **not worth running** —
with a correct echo it would likely still match ~296 and re-prefill nearly everything,
burning a scarce window to re-measure a structural mismatch. **Not acting on this**:
it is a hypothesis from arithmetic, and the deciding measurement is a tokenization of
the two exact renderings (`add_generation_prompt` false vs a following user turn),
which needs the model and therefore the window. **R0 keeps the window per the
sequencing.** Proposed: fold that tokenization into run 2 as a cheap first leg, so the
run either proves the mismatch or clears it before spending time on the full A/B/C/D
matrix. **Flagging for a directive rather than changing the spec unilaterally.**
