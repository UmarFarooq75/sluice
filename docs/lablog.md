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
| D3 | Margin units are per-family (0.02 meant something on OLMoE probs; no-op on gpt-oss logits) | Why? Mask threshold lives in the gating score's native scale. A universal product cannot ask users to know gating families. | Design: normalized margin (percentile of per-token score gaps) or entropy-adaptive margin. Experiment queued |
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
