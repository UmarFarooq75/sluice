# The 3 GB question: how low can RAM go for a 671B model?

Target set by Umar (2026-07-17): 24 GB is too much. Goal: **671B-class model in 2–3 GB RAM, with low compute.** This doc is the physics-honest analysis: the RAM floor anatomy, the leakage inventory, the byte-reduction stack, and what speed survives.

## First principle: nothing *must* be resident

RAM residency is a **speed optimization, not a requirement**. Every weight can live on storage and stream through a small working buffer. Therefore the true RAM floor is set not by model size but by:

1. working buffers (the weights currently in flight),
2. KV cache (grows with context),
3. activations + runtime overhead,
4. whatever we *choose* to pin because it's hyper-hot.

## RAM ledger for DeepSeek-V3-class (671B) at int4, 4k context, batch 1

| Component | Size | Policy |
|---|---|---|
| Weight working buffers (1 layer attention ~95 MB + 8 experts × 22 MB, double-buffered for prefetch) | ~0.55 GB | stream |
| lm_head (output projection, fired every token, 0.46 GB) | 0.46 GB | **pin** |
| Routers, all 61 layers (needed early for prefetch) + norms | ~0.08 GB | **pin** |
| Shared expert, all layers (fired every token, 2.55B params) | 1.28 GB | **pin** (2-bit it → 0.64 GB if squeezed) |
| Embedding table (0.46 GB) — only 1 row/token needed | ~0 | stream per-row (3.5 KB reads) |
| KV cache, MLA-compressed, int8, 4k tokens (~70 KB/token fp16) | 0.14 GB | RAM (spill to disk beyond budget) |
| Activations (batch 1) + logits | ~0.05 GB | RAM |
| Runtime + OS overhead (lean C runtime) | ~0.3 GB | RAM |
| **Total** | **≈ 2.8 GB** | ✅ inside 3 GB |

**Verdict: a 671B model in ~3 GB RAM is architecturally feasible today.** No exotic hardware, no retraining. MLA models (DeepSeek/GLM/Kimi family) are the right target because their KV cache is ~70 KB/token — Mixtral-style GQA would blow the budget on KV alone at long context.

The catch is speed, and it's pure arithmetic:

- At 3 GB there is **no room for an expert cache** (even a 15% task working set of a 327 GB expert pool is ~49 GB). Expert hit rate ≈ 0. RAM-tier caching is dead at the floor; it only starts paying above ~8–16 GB.
- So bytes streamed per token ≈ full routed traffic (10.2 GB) + non-pinned trunk (~5.8 GB) ≈ **16 GB/token** → PCIe4: ~0.5 tok/s, PCIe5: ~0.9 tok/s. Runs, but a demo.

**Consequence: at the RAM floor, the ONLY lever left is reducing bytes-per-token.** That's where the real research is.

## The byte-reduction stack (each multiplier verified or clearly flagged)

| Lever | Mechanism | Bytes multiplier | Quality/status |
|---|---|---|---|
| 2-bit codebook quant (AQLM/QTIP-class) for *streamed* weights | vector/trellis codebooks beat uniform 2-bit cliff; decode via LUT | ×0.5 | proven on dense 70B; **MoE-at-scale = our experiment** |
| TEAL activation sparsity (training-free) | skip 40–50% of weight *rows* whose input activations are near-zero — cuts reads AND FLOPs | ×0.6 | verified on Llama/Mistral 7–70B, minimal loss ([arXiv 2408.14690](https://arxiv.org/abs/2408.14690)); needs row-bundled storage layout to keep reads sequential |
| Adaptive top-k (gate top-p ~0.7) | don't fetch experts with negligible gate weight | ×0.6 on expert traffic | colibri measured −30–40% disk, 1.6× e2e; quality cost small, tunable |
| Early exit / layer skip on easy tokens | confidence-based exit; KV for skipped layers approximated | ×0.75–0.8 | research-grade (LayerSkip et al.); riskiest of the stack |
| Mixed precision by usage (hot 4-bit / cold 2-bit) | ROM layout + cheaper cold fetches | overlaps with codebook row | validated direction (HOBBIT runtime version) |

Stacked: 16 GB/token → **~2.5–3.5 GB/token** with the full stack. Resulting ceilings at 3 GB RAM:

| Storage | tok/s (full stack) |
|---|---|
| PCIe5 NVMe (14.5 GB/s) | ~4–6 |
| PCIe4 NVMe (7.45 GB/s) | ~2–3 |
| UFS 4.0 phone (4.2 GB/s) | ~1.2–1.7 |

**A 671B model at ~2–5 tok/s in 3 GB RAM is on-paper reachable.** Nobody has stacked these. Every individual multiplier exists in the literature; the composition is the project.

## The compute side (Umar's second constraint)

FLOPs/token ≈ 2 × active params ≈ 74 GFLOP for DeepSeek-V3. On a laptop-class CPU (~50–100 int8-GFLOPS sustained) that's 0.7–1.5 s/token — at the reduced byte budget, **compute becomes the bottleneck, not memory**. The compute stack:

1. **Fewer FLOPs**: adaptive top-k (−30–40%) and TEAL (−40–50%) cut compute with the same knobs that cut bytes — they pay twice. Early exit adds −20–30%.
2. **Cheaper FLOPs**: [T-MAC](https://arxiv.org/abs/2407.00088)-style LUT matmuls — no dequantization, no multiplies, table lookups instead: up to 6.6× kernel / 2.8× end-to-end vs llama.cpp at 2-bit, 60–70% less energy, and cost scales *linearly down* with bits (2-bit is 2× cheaper than 4-bit — synergizes with the codebook quant lever). Runs on Raspberry Pi-class CPUs.
3. **Free silicon**: phone NPUs / Apple AMX for the dense trunk (PowerInfer-2's split: dense→NPU, sparse→CPU).

Net: ~74 → ~25–30 effective GFLOP/token, executed 2–3× cheaper per FLOP. A plain M2/Snapdragon CPU sustains 2–5 tok/s of that — matching the bandwidth ceiling. Balanced system, no GPU.

## Leakage inventory (where today's stacks waste, ranked by cost)

1. **Prefill re-streams the whole model per prompt** → expert-major prefill (our pillar 3). Biggest single win for real usage.
2. **Reading experts the router barely wants** (gate weight 2–3%) → adaptive top-k.
3. **Computing/reading rows for near-zero activations** → TEAL + row-bundled layout.
4. **Uniform bit-width everywhere** despite 5–15% of experts doing most work → usage-weighted mixed precision.
5. **Dequantize-then-multiply kernels** → LUT matmul (T-MAC).
6. **mmap page-granularity read amplification + double buffering** (llama.cpp path) → O_DIRECT + exact coalesced expert reads (colibri already proves this).
7. **Cold start every session** — re-learning what's hot, re-reading prompt KV → persistent working sets + persistent KV cache on disk.
8. **Full-vocab logits every token** (0.46 GB matmul) → pin lm_head; later: vocab shortlisting for the easy steps.
9. **Runtime bloat** — Python/framework overhead measured in GB → lean C/C++ runtime (~50 MB idle).
10. **KV recomputation and fp16 KV** → MLA-family targets + int8/int4 KV + disk spill.

## Reframe of the working-set idea at the floor

At 3 GB, the learned working set can't live in RAM — but it still runs the show **one tier down**: it decides which experts live on *disk* (vs network), which get 4-bit (vs 2-bit), and what the prefetcher speculates on. RAM tiers: floor mode (~3 GB) / balanced (8–16 GB, expert cache starts paying) / comfort (24 GB+). Same engine, one knob.

## What we validate on the OLMoE testbed (Phase 0.5, this machine)

1. **Floor-mode emulation**: cap the expert cache to zero, stream OLMoE's experts from SSD, measure tok/s vs the formula's prediction (validates the whole bandwidth model end-to-end).
2. **Adaptive top-k quality**: rerun evals at gate top-p 0.9/0.8/0.7 — quality vs expert-traffic curve, on real weights.
3. **TEAL on a MoE**: apply magnitude thresholding to OLMoE expert activations — does 40–50% sparsity hold for fine-grained MoE experts? (Novel data point — TEAL was validated on dense models.)
4. **Gate-mass concentration** from our traces: how many of the top-8 experts carry 90% of gate weight (already in the simulator as `gatemass`).

## Honest boundaries

- Long context eats the budget: at 3 GB, MLA int8 KV supports ~10–15k tokens; beyond that, KV spills to disk (adds read traffic) or context is windowed.
- The 2-bit codebook + TEAL + early-exit multipliers each carry quality risk; they must be measured together, not assumed independent. If the stack only reaches ×0.35 instead of ×0.2, the PCIe5 ceiling is ~2.5 tok/s, not 6.
- microSD/eMMC-class storage stays hopeless for 671B (0.1–0.4 GB/s → minutes/token). Floor mode needs NVMe/UFS-class storage. Small devices run small MoEs — that's what they're for.
