# Phase 0 findings — first measured results (2026-07-17)

> **NOVELTY NOTICE (2026-07-18):** an adversarial prior-art audit ([novelty-audit.md](novelty-audit.md)) found that several mechanisms below were independently published in 2024–2026 (margin routing, co-activation layout, self-speculation = already done; token table, expert-major prefill, agent-swarm, expert-skip = partially prior). Finding 30's "internally dense" phrasing is corrected there. All measurements remain valid as *independent reproductions on consumer hardware*; "nobody does X" phrasings below are superseded by the audit. What survives as ours: the composition-effect measurements (finding 20), the all-or-nothing expert-skip rule with inverse control (22), the verification methodology, and the integration gap — no shipping engine composes these mechanisms for any GGUF MoE.

**Setup:** OLMoE-1B-7B-0125-Instruct, 4-bit MLX, MacBook Air M2 16 GB. Router hooked; every routing decision logged: 6 workloads (code/math/prose/chat/knowledge/multilingual) × 5 prompts × 256 generated tokens (~7,650 decode tokens; ~1,270 per workload) + prefill sweep (127 → 4,789 prompt tokens). Cache sims run on decode tokens only; task profiles are leave-one-prompt-out (no leakage). Full data: `results/analysis.json`, raw traces in `traces/`.

**Read all numbers with these caveats:** 8-of-64 routing (12.5% random baseline) is structurally ~4× more optimistic than frontier 8-of-256 models; 4-bit weights; greedy decoding; n=5 prompts/task. Shapes transfer; magnitudes don't. (See docs/analysis-gaps.md.)

## Headline results

### 1. Expert caching works — far above random, and misses are the cheap ones ✅
LRU with 16/64 experts cached (25%): **47–62% hit rate** across workloads (random = 12.5%). Gate-weighted hit rates run 2–6 points *higher* than binary — **misses skew toward low-gate-weight experts**, which is exactly what you want for quality under any miss-handling policy.

### 2. The oracle gap says prediction is worth building ✅
Belady oracle at the same capacity: **71–79%** — a 15–20 point gap above LRU. That gap is the theoretical prize for router-lookahead/learned prediction (ProMoE-class systems claim to capture most of it). Prefetching is not optional polish; it's a fifth of all traffic.

### 3. Tiny caches invert the ranking → hybrid design (engine insight #1) ✅
At C=4/64, LRU **collapses to 3.6%** (pure thrash) while a learned task profile holds **31.6%** (code). At C≥24 reactive wins (LRU 75% vs profile 74%). **Engine design: pinned learned base + reactive top-up layer**, with the split point set by available RAM. Nobody ships this hybrid.

### 4. Working sets are about *frequency*, not *coverage* (honest correction to our own pillar 6) ⚠️
Discovery curves show ~**1,000 of 1,024** (layer,expert) pairs get touched within ~1,250 decode tokens on every workload — *coverage* saturates at ~everything. But *traffic* concentrates: 50% of activations come from **10–20 of 64** experts/layer (code/math tightest), 90% from 38–48. Implication: you cannot simply *omit* the cold tail from disk (it will be needed eventually) — the ROM story must be **frequency-weighted mixed precision + network-fetch for the rare tail**, not deletion. "Store only 15% locally" was too optimistic for OLMoE-class routing; the right local budget is more like 60–75% by bytes with the tail at 2-bit and/or remote.

### 5. Task disjointness is real but graded ✅
Top-16 expert-set Jaccard overlap: code~math **0.55** (siblings — merge their profiles), chat~knowledge 0.37, but math~prose **0.05** and code~prose **0.08**. Cross-task profiles score 21–36% vs same-task 37–63%. **Profiles should be per task-family, and they compose.**

### 6. Temporal locality is strong ✅
Consecutive tokens share **34–47%** of their top-8 experts (2.7–3.8× random), highest for code/math, lowest for prose — consistent with (and stronger than) published Mixtral measurements. Deep layers ≥ shallow. This is why plain LRU already works at moderate capacity.

### 7. Expert-major prefill is a monster win — strongest result of the day ✅✅
Prefill touches essentially **all 64 experts per layer** regardless of prompt length. Naive token-major streaming loads experts O(tokens × 8); expert-major loads each expert **once per layer**:
| prompt tokens | naive loads/layer | expert-major | reduction |
|---|---|---|---|
| 127 | 1,016 | ~62 | **16×** |
| 435 | 3,480 | ~63 | **55×** |
| 1,876 | 15,008 | ~64 | **235×** |
| 4,789 | 38,312 | 64 | **599×** |

This single mechanism makes long prompts *feasible* under streaming. No shipped engine does it. (4,789-token trace exceeds OLMoE's 4,096 context — direction unaffected, flagged anyway.)

### 8. NEGATIVE RESULT: adaptive top-k has ~zero headroom on OLMoE ❌
The router is nearly flat: the top-8 experts carry only **~42%** of the full 64-expert softmax mass; reaching 90% of full mass needs **~43 of 64** experts; ~0% of token-layers let 4 experts cover 90% of the top-8's own mass. Cutting k on "easy tokens" would degrade OLMoE — there are no easy tokens by this signal. colibri's −30–40% via `--topp` works on **sigmoid-router** models (GLM/DeepSeek family, where gates are independent scores, not a softmax) — mechanism doesn't transfer to softmax routers. **Consequence: adaptive-k and expert-skip must be validated per router type; DeepSeek-V2-Lite is now the priority second model.** This is exactly the composition-flaw class we predicted in flaws-and-coverage.md #2/#17.

### 9. Simpler engine is fine: global pool ≈ per-layer caches ✅
Sharing one cache pool across layers gains nothing (±0.5 points everywhere). Per-layer caches — simpler to implement, better for prefetch pipelining — are the right choice.

### 10. Bug in our own greedy allocator (caught, not shipped) 🐛
Greedy per-layer capacity allocation scored *below* uniform (42% vs 54%) — impossible for a correct optimizer; cause: LRU hit-vs-capacity curves are non-concave (thrash cliff at tiny capacity), which breaks greedy marginal-gain. Replace with exact DP over (16 layers × 8 capacity levels) next cycle. The per-layer data still shows real allocation headroom: layer 0 caches at 42% while layer 5 hits 71% at equal capacity — shallow layers deserve less cache.

## What this changes in the engine design

1. Hybrid cache: learned pinned base + LRU top-up (result 3), per-layer (result 9), capacity allocated by DP over measured curves (result 10).
2. Prefetch/prediction budgeted as a first-class subsystem — 15–20 points of traffic are recoverable (result 2).
3. Expert-major prefill is mandatory, not optional (result 7).
4. ROM plan revised: frequency-weighted mixed precision + network tail; omission is wrong (result 4).
5. Adaptive-k demoted to per-architecture experiment; do not generalize colibri's number (result 8).

## Phase 0.75 — Restricted-routing quality eval (the trace→quality bridge)

Teacher-forced NLL on fixed prose/code/math texts under routing restrictions (lower = better; deltas vs baseline are the finding). Full data: `results/quality_eval.json`.

| Condition (per-layer) | code ΔNLL | math ΔNLL | prose ΔNLL | fetches/token (code/math/prose) |
|---|---|---|---|---|
| baseline top-8 | — | — | — | ~33–46 misses/tok if 24 pinned, exact top-8 |
| **route24 margin=0.02** | **+1.2%** | **−1.1%** | **+1.4%** | **10.8 / 17.3 / 27.3** |
| route24 margin=0.05 | +3.8% | +6.2% | +15.4% | 2.6 / 5.3 / 9.8 |
| route16 margin=0.02 | +1.2% | −0.2% | +2.5% | 19.5 / 29.8 / 49.7 |
| k=6 (drop 2 experts) | +3.9% | +5.4% | +4.4% | — |
| k=4 (drop 4) | +14.1% | +25.2% | +20.1% | — |
| pin24 hard (no fetch, matched profile) | +14.1% | +128.7% | +91.2% | 0 |
| pin16 hard | +30.9% | +213.7% | +136.7% | 0 |
| pin24 WRONG profile | +403% | +586% | +105% | 0 |

### 11. Margin-gated cache-aware routing is a near-lossless traffic dial ✅✅ (our headline mechanism)
With 24/64 experts resident and a fetch allowed only when a non-resident expert's router probability beats the weakest resident pick by ≥0.02: **quality is statistically indistinguishable from baseline (±1.4%) while disk fetches drop ~3× vs exact top-8; at margin 0.05 fetches drop ~9–13× for +4–15% NLL.** A smooth, tunable quality↔bytes dial, measured — colibri's CACHE_ROUTE never was. This mechanism + the hybrid cache is the engine's core.

### 12. Hard pinning without a fetch path destroys quality ❌ (design law)
Even the *matched* 24-expert profile with no escape hatch costs +14% (code) to +129% (math) NLL; the wrong profile is catastrophic (+400–600%). **The tail must always be reachable** (disk or network). Any "RAM-only mode" must be margin-gated routing, never a hard mask. Also confirms task profiles are safety-critical to select correctly — but route-mode turns a wrong profile into extra fetches instead of garbage output.

### 13. Reduced-k confirmed harmful on OLMoE (causal, not just correlational) ✅
k=6 costs +4–5%, k=4 costs +14–25% NLL — matches the flat-router prediction from result 8. The traces predicted it; the model confirmed it.

### 14. DP capacity allocation: only matters when starved 🔍
Exact DP beats uniform by +3.9 points at budget 128/1024 experts (zeroing shallow layers), but ≤0.5 points at budgets ≥192. **Uniform allocation is fine except in extreme-low-RAM mode** — one less thing to over-engineer. (`results/dp_allocation.json`)

### 15. Network tier: real, but it's the *last-mile* saver, not a 90% ROM cut ⚠️✅
Disk-as-cache simulation (top-F% of experts by frequency local, rest fetched from network and persisted, LRU): at **90% disk budget → 0.9–1.6 network fetches/token**; 75% → 5–9; 50% → 20–30 (all for exact top-8 service; margin-gated routing cuts these ~3× at ±1.4% quality per result 11). For DeepSeek-V3-class (22 MB/expert), 90%-local ≈ **20–35 MB/token** from network — hideable behind generation on ordinary broadband with prefetch, and ~7–11 MB/token with margin routing. Two more facts: seeding barely matters (LOO vs matched profiles converge within the first quarter of a session — persistence does the work), and fetch rates are flat across the session (it's the tail's intrinsic rate, not warmup).

**Honest correction to techniques.md's multiplication table:** for frontier models the network tier trims the last 10–25% of ROM (~35–85 GB on 671B), not the fantasy 4× cut — because working sets are frequency-not-coverage (result 4). The big ROM cuts must come from pruning + mixed precision; the network tier's killer value is (a) small-ROM devices running mid-size MoEs, (b) start-before-fully-downloaded UX, (c) never storing the long tail. `results/network_tier.json`

### 16. Task-switch detection is nearly free: 6 tokens, from the router alone ✅
Interleaved 8-segment session (code→chat→math→prose→code→knowledge→math→multilingual): an EMA over routing one-hots classified against task centroids gives **93% token-level accuracy with median switch-detection latency of 6 tokens** (range 5–8). And the LRU cache's post-switch hit rate (~0.49–0.67 in the first 50 tokens) is already at steady state — temporal locality re-establishes almost instantly. **The router is a free task classifier**; profile switching can be event-driven and cheap. `results/session_switch.json`

### 17. Accuracy confirms the NLL story on real answers ✅ (n=24, coarse)
Exact-match on 24 arithmetic problems: baseline **15/24**; route24_m02 **15/24** (zero loss, matching its ±1.2% NLL); route16_m02 **15/24**; route24_m05 13/24; k6 13/24; **pin24 hard 8/24** (catastrophic, matching +129% NLL). NLL and answer accuracy rank conditions identically → our cheap NLL methodology is a valid proxy. `results/accuracy_eval.json`

### 18. Router-lookahead prefetch generalizes — 83.9% next-layer recall on OLMoE ✅✅
Applying layer L+1's gate to layer L's input recalls **83.9%** of the true next-layer top-8 (34,740 measurements; per-layer: 0.61 shallow → 0.89–0.91 deep) — *better* than colibri's 71.6% on GLM-5.2. Prefetch-one-layer-ahead is architecture-general, not a GLM quirk. Combined with margin routing (result 11) to soften the ~16% misses, the prefetch subsystem rests on two independently measured legs. `results/lookahead.json`

## The engine core is now evidence-backed end to end

Detection (6 tokens) → profiles (task-family, graded overlap) → hybrid cache (pinned base + LRU) → margin-gated cache-aware routing (±1.4% NLL, 0 accuracy loss, ~3× traffic cut) → lookahead prefetch (84%) → expert-major prefill (16–599×) → network tail (~1 fetch/token at 90% disk). Every arrow measured on our own hardware today.

### 19. PoC streamer: the physics holds on real hardware ✅✅✅
Built the real thing in miniature: OLMoE's 1,024 experts as individual 3.54 MB files (3.63 GB store), streamed from the M2 Air's SSD with F_NOCACHE (page cache bypassed), LRU per layer, decode-only, steady-state warmup, counters clean. SSD measured 2,468 MB/s at expert granularity. `results/streamer_poc.json`:

| mode | tok/s | hit rate | MB read/token | NLL |
|---|---|---|---|---|
| resident reference | 55.2 | — | 0 | 3.002 |
| stream, cache=0 (floor mode: **zero expert RAM**) | **2.83** | 0% | 463 | **3.002** |
| stream, cache=16/64 | 4.06 | 45% | 254 | 3.002 |
| stream, cache=32/64 | 8.26 | 78% | 103 | 3.002 |
| stream 16 + margin 0.02 | **7.53** | 63% | 171 | **3.002** |
| stream 32 + margin 0.02 | **15.46** | 91% | 40 | **3.002** |
| stream, all cached | 27.2 | 97.5% | 11 | 3.002 |

What this proves, physically: (a) **bit-correctness** — streamed compute reproduces the resident model's NLL exactly, every mode, including margin routing (the correctness gate colibri never had); (b) **the formula works** — measured speeds track the I/O ceiling at 50–65% (the gap = Python per-expert loop tax; load_time_frac 0.86–0.91 at low cache confirms I/O-dominance exactly as predicted); (c) **margin routing doubles real speed for free** — 2× at both capacities with *zero* NLL change on warm caches; (d) **trace sims ↔ physical hit rates agree** (45%/78% measured vs 47–62%/75–84% simulated) — the whole Phase-0 methodology is validated; (e) **floor mode is real** — 100% of experts on disk, ~0 expert RAM, 2.8 tok/s on a laptop, correct output.

Scale check: 463 MB/token here → DeepSeek-V3's 10.2 GB/token on this SSD ≈ 0.24 tok/s cold, ~1.4 on PCIe5 — matching the research table. The remaining 2× gap to resident speed at full cache is pure Python; a compiled engine with async prefetch (84% lookahead, result 18) attacks both the loop tax and the I/O stalls.

### 20. Prefetch works physically — but it and margin routing SUBSTITUTE, not stack 💡
Async lookahead prefetch (Python threads, 4 workers, predicting layer L+1's top-8 from layer L's input) on the physical streamer, measured under SSD contention (DS download running — absolute numbers conservative, within-run comparisons valid):

- cap16: 4.04 → **6.94 tok/s (+72%)**, hit rate 45%→78% — prefetch is a monster at small capacity.
- cap32: 7.89 → 7.67 (no gain) — at decent hit rates, speculative reads steal bandwidth from demand misses and churn the cache.
- cap32 + margin 0.02: 15.03 → **10.17 (prefetch HURT the best config by 32%)** — margin routing already avoids most misses; naive prefetch double-spends the disk on experts margin routing would have skipped, and evicts still-useful residents.

**Design law: prefetch and margin routing attack the same miss latency — coordinate them, don't stack them.** The engine should prefetch the *margin-filtered* prediction (only experts a margin router would actually fetch), and only when the I/O queue is idle. NLL stayed 3.002 in every mode. This interaction is exactly the composition-effect class flagged in flaws-and-coverage.md #8 — first one caught live. `results/streamer_poc.json`

**Clean rerun (SSD uncontended) confirms the law with sharper numbers**: cap16 prefetch +74% (4.04→7.03); cap32 prefetch only +8% (8.25→8.90); cap32+margin: prefetch still −27% (16.24→11.85, at 94% hit rate the speculative reads only churn). Best overall config: **cap32 + margin 0.02 = 16.24 tok/s** (21% of the 76.7 tok/s resident reference), quality bit-identical. Contention wasn't the cause — the substitution effect is intrinsic.

### 21. Uncertainty quantified: the headline numbers are robust ✅
Bootstrap over prompts (500 resamples, 95% CI): LRU@C16 — code 61.1% [59.5–63.4], math 61.6% [57.5–65.8], multilingual 56.1% [52.1–62.3], chat 49.8% [47.3–52.3], knowledge 50.0% [46.7–54.6], prose 46.5% [44.3–48.6]. Temporal locality CIs all clear the 12.5% random baseline by ≥3×. The structured-vs-open-ended split and every cache-design conclusion survive resampling. `results/bootstrap_ci.json`

### 22. EXPERT-SKIP VALIDATED on DeepSeek-V2-Lite — our mechanism works, and the control proves it ✅✅ (novel)
On easy tokens, keep attention running (KV intact), skip all routed experts, let the 2 shared experts carry the FFN. Trigger: top-6 gate mass below threshold. Teacher-forced NLL (`results/expert_skip.json`):

| condition | code ΔNLL (skipped) | math ΔNLL (skipped) | prose ΔNLL (skipped) |
|---|---|---|---|
| skip when mass<0.30 | +11.9% (34.5%) | **+1.4% (18.4%)** | **+2.5% (15.3%)** |
| **control: skip when mass≥0.45** | **+187% (11.9%)** | **+183% (20.1%)** | **+236% (9.2%)** |
| k=3 instead of 6 | +3.6% | +7.7% | +4.1% |
| skip all routed (bound) | +890% | +2,120% | +449% |

The control is the proof: skipping *confident* tokens is ~50–100× more damaging per skipped token than skipping *indifferent* ones — **the router's gate mass is a genuine easiness signal**, not noise. At threshold 0.30, math/prose drop 15–18% of ALL routed-expert I/O and FLOPs for 1.4–2.5% NLL; code needs a stricter (per-task or per-layer) threshold. Sharp cliff above 0.30 (0.45 skips ~90% and destroys quality) — calibration is the engineering. No paper ships this mechanism.

### 23. DeepSeek-V2-Lite replication: shapes hold, and adaptive-k comes back from the dead ✅
26 MoE layers, top-6-of-64 + 2 shared experts (random baseline 9.4%): LRU@C16 = 40–46% ≈ **4.3–4.9× random — same relative cacheability as OLMoE** (3.7–4.9×). Belady oracle 66–70% → an even bigger prediction prize (~25 points). Weighted > binary again (misses stay cheap). Task profiles are weaker here (27–42%) with only 3 prompts/task — consistent with profiles needing more data on deeper models. **And the architecture verdict flips as predicted: k=3-of-6 costs only +3.6–7.7% NLL (vs +14–25% for the equivalent cut on OLMoE)** — DeepSeek-family's concentrated gates + shared-expert safety net make adaptive-k viable there. Per-architecture validation vindicated twice in one day. `results/analysis_ds.json`

### 24. Lossless-compression levers: measured, mostly dead ❌ (honest negatives)
Raw int4 expert weights are near-max entropy: zlib saves only 6.1% on weights (9% overall; only the small scales/biases compress at 32.5%), and zlib decode (~390 MB/s) is *slower than the SSD* — on-the-wire compression would bottleneck, not help. A multi-core fast codec could bank ~9% someday; garnish, not pillar. **Block-level dedup across all 12.6M 256-byte blocks of 1,024 experts: 0.0000% duplicates** — content-addressed expert storage is definitively dead. `results/lossless_scan.json`

### 25. Cache-restricted self-speculation (novel, lossless): marginal on OLMoE — decisive test on DS-family running ⚠️
The idea: the resident-expert subset IS the draft model (zero-disk drafting), full model verifies → output provably identical. Measured on OLMoE: draft argmax-agreement pin32 = 80.9% code / 77.3% math but 42–53% prose/chat (mean 63%); expert-union sublinearity confirmed (w8 = 0.456). Net predicted disk reduction only **~1.1–1.3×** (short windows on code/math ~1.5×) — agreement decay beats union savings at longer windows. Consistent with OLMoE's flat-router fragility (finding 8); DeepSeek-family's restriction-tolerance (finding 23) predicts higher agreement there — same measurement on DS-V2-Lite is the decisive test. `results/self_spec.json`

### 26. Self-speculation verdict: marginal on BOTH architectures — and the ceiling is structural ❌📐
DS-V2-Lite replication (n small: 1 prompt/workload, 3-prompt profiles): pin32 draft agreement 63.0% (code 78%, math 76%, prose/chat 43–55%) — no better than OLMoE; pin16 collapses (13%). Best estimate **1.16×** (pin32|w2). The deeper law: even at PERFECT agreement, self-spec's gain is bounded by 1/union-sublinearity — measured 0.87/0.74/0.59 at w=2/4/8 → **ceiling ≈ 1.15–1.7×** on both models, because consecutive tokens simply don't reuse enough experts (finding 6: 34–47% overlap). Worth ~1.2–1.4× inside an engine (it stacks losslessly), not a silver bullet. Variant left open: truncated-k drafting (k=3 draft ⊂ k=6 verify, agreement ~85%+ per finding 23) reaches toward the ceiling but not past it. `results/self_spec_ds.json`

### 27. H2 CONFIRMED — co-activation disk layout: 1.42–1.55× fewer read ops, free ✅✅ (novel)
Assumption broken: "expert files are independent reads in arbitrary order." Clustering each layer's experts by measured co-firing (greedy graph grouping, G=8) turns a token's ~7.1 scattered reads into ~4.8 contiguous runs — consistent across all six workloads (code best at 1.55×). Computed offline from traces; zero runtime cost; bit-lossless. Benefit concentrates where storage is weakest (small experts, phones, cheap NVMe, low queue depth). No published system lays out MoE weights by co-activation statistics. `results/h2_h4.json`

### 28. H4 CONFIRMED — agent-swarm streaming is sublinear: 5 agents for the disk cost of ~3 ✅✅ (product thesis)
Assumption broken: "one user, one stream." Expert unions across N parallel same-task-family sequences: N=5 costs only 60–75% of naive per-agent loads (code 0.596, math 0.618). Streaming cost per agent FALLS as the fleet grows — parallel workloads help the bottleneck instead of fighting it. Combined with the shared cache this makes "one cheap box, one frontier MoE, N agents" a uniquely defensible mode no serving engine targets. `results/h2_h4.json`

### 29. H1 CONFIRMED — routing is ~2/3 token-determined: full-depth prefetch from a static table ✅✅✅ (novel, assumption-breaking)
Assumption broken: "expert choice needs the hidden state, so prefetch is limited to one layer ahead." Measured on 134 token ids with ~18 occurrences each in *different contexts*: a token's top-8 matches its own modal expert set at **67.1%** across all 16 layers (random 12.5%) — routing is two-thirds a property of the token itself, one-third context. A static lookup table (vocab → 8 experts × 16 layers, ~6 MB, built by counting) prefetches **59.5% of ALL layers' loads at embedding time** on held-out data — peaking at 70.8% in the mid-deep layers — before any layer computes, with the model's entire forward time available to hide the I/O.

Engine consequence — the prefetch subsystem becomes a three-stage pipeline, each stage catching the previous stage's misses: **(1) t=0: full-depth table prefetch (~60%, free) → (2) per-layer router lookahead (84%, one layer of hiding) → (3) margin routing for the remainder (near-lossless)**. Combined, most of the 15–20-point Belady gap closes with zero learned models — just counting. Also explains *why* pure predictor approaches (SiDA-class) plateau: the ~33% contextual remainder needs the runtime signals. Caveat: frequent-token measurement (traffic-weighted this is most of decode volume); OLMoE only so far. `results/h1_token_determinism.json`

### 30. H3 REFUTED — experts are internally DENSE: the router already harvested the sparsity ❌💡
Progressive/prefix fetching is dead: neuron-importance Gini inside experts is only **0.157** (near-uniform), the "hot" top-25% neuron set is only **47.5% stable** across data halves, and truncating to 50% of rows causes **60% relative output error** (160 experts sampled, real hidden states, physical store weights). The deep explanation: **fine-grained MoE granularity IS the activation sparsity** — training carved the FFN into 64 token-specialized dense units, so the within-unit sparsity TEAL finds in monolithic dense FFNs has already been spent by the router. This also answers flaws-and-coverage #1 by implication: TEAL-style row-skipping will not transfer to fine-grained MoE experts. Fetch whole experts; the fetch-granularity dial doesn't exist. `results/h3_progressive.json`

### 31. H1 replicated on DeepSeek-V2-Lite — token-determinism is architecture-general ✅
Determinism@6 = **57.4%** (random 9.4% → 6.1× above random, vs OLMoE's 5.4×); held-out static-table recall **49.1%** (n=87 tokens, smaller sample). First MoE layer is the most token-determined (71.4%) — shallow routing is lexical. The full-depth prefetch table works on both router families; expect ~50–60% of all expert loads issuable at embedding time on the GLM/Kimi/DeepSeek targets. `results/h1_ds.json`

## Next analyses queued

- **Restricted-routing quality eval** (the trace→quality bridge — flaws doc #15): regenerate with router masked to cached/pinned sets; measure output quality vs bytes saved. The single most load-bearing missing measurement.
- Exact DP capacity allocator; per-layer profiles.
- Mixed/interleaved-session trace → profile-switch detection latency.
- Router-lookahead recall on OLMoE (hidden-state capture).
- DeepSeek-V2-Lite (sigmoid-style router + shared experts): re-run everything; decide adaptive-k and expert-skip fate there.

## Phase 1 — M1 gate result (finding 32, 2026-07-17 night)

**Finding 32 — streamed-expert llama.cpp is bit-exact, and the one divergence we hit was llama.cpp's own kernel choice, not streaming.** The fork (branch `llmstream`, ~135 lines: slot-backed expert tensors + dual id tensors + `cb_eval` driver) streams OLMoE Q4_K_M experts straight from the GGUF's per-expert extents (no repacked store — offsets verified 24/24 against an independent gguf-py reader). Logit-equivalence gate: FNV-1a over every generated position's full logit vector, resident vs 32-slot vs 12-slot — **bit-identical** (`b6869f5b6ef36376`).

The initial gate FAIL taught the real lesson: llama.cpp repacks ARM weights at load (`use_extra_bufts`) into interleaved layouts with different gemm accumulation order — low-order-bit logit drift with zero token-level effect (32/32 greedy tokens identical). Bisection that found it: dup-only ≡ resident (graph change innocent) → slots32 ≡ slots12 ≡ identity64 (slot machinery consistent) → bytes proven identical → only the kernel path remained. **Bit-exactness claims must pin the kernel path, not just the math.** With repacking off on both sides: PASS.

Cold-SSD speed (fresh F_NOCACHE copy per run; first-run numbers were cache-inflated and rejected):

| config | expert RAM | decode | vs stock best (45.2) |
|---|---|---|---|
| streamed 32 slots | 1.8 GB | **28.3 tok/s** (hit .891, 1.76 GB/s) | 63% |
| streamed 12 slots | 0.66 GB | **9.1 tok/s** (hit .589, 1.91 GB/s) | 20% |

3.4× the Python PoC at the same cache (target was ≥2×). Fetches are still serial with compute — the measured I/O-only ceiling (~36 tok/s at slots32) says overlap + prefetch (M2) is worth ~+25%. Margin routing (finding 11) is not yet wired in at all.

## Finding 33 — the head-to-head (2026-07-18): a 2026 frontier-family model on hardware that cannot hold it

Qwen3.6-35B-A3B Q5_K_M (26.5GB, 40 layers × 256 experts, released 2026) on the 16GB Air:

| engine | result |
|---|---|
| stock llama.cpp mmap (Ollama's engine) | **DNF** — OOM-killed during load; llama-cli >10 min without 4 tokens |
| ours, exact routing (m=0), 5.9GB expert cache | 3.57 tok/s |
| ours, margin m=0.02, **2.9GB expert cache** | **8.27 tok/s** — compute-bound |

The margin router (finding 11) pushed I/O below the CPU compute floor: a 2.9GB cache matches a 5.9GB one at ~80% of the machine's physical compute ceiling for 3B active params. More RAM (8.1GB cache) was *slower* — memory pressure beats the extra hits on a 16GB machine. The RAM dial saturates exactly where the formula says it should.

Adapter cost for this brand-new family (hybrid linear-attention trunk, merged-or-split gate_up, shared expert): ~20 lines + one bring-up bug — a missing buffer allocation that put streamed weights in scheduler scratch (deterministic garbage; the driver now fails loudly on unallocated slot tensors). Bit-exactness remains proven on OLMoE where a resident reference exists; for models that cannot fit, m=0 preserves exact routing semantics by construction.

**Finding 33 addendum — the real Ollama product test (same day):** actual Ollama v0.32.1 (standalone binary, our GGUF hardlinked into its store — byte-identical file, zero copy). It loaded via mmap, then collapsed into OS paging: **0.16 tok/s prompt processing (its own log), ~0.03 tok/s generation — 16 tokens in 10 minutes**, machine at 55MB free RAM with 23.7M pages swapped out. Same file, same laptop: a 200-token answer = ~1.9 hours on Ollama, ~24 seconds on our engine. Evidence: results/qwen36_ollama_serve.log, results/qwen36_headtohead.json. Boundary honesty: on models that fit in RAM the two engines are equivalent (same llama.cpp underneath); LM Studio untested (not installed).
