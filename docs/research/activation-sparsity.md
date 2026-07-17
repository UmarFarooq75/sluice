# Agent report 3/3: training-free activation sparsity for dense SiLU/GELU LLMs — through mid-2026

> Collected 2026-07-18 by a research subagent (web sweep). Feeds docs/dense-strategy.md.
> Key strategic reads for us:
> - **Training-free per-token sparsity on stock SiLU models cannot reach streaming grade.** Quality-safe ceiling is 40-65% (TEAL/WINA/R-Sparse/LaRoSA); streaming needs ~97%+ of touched bytes to come from cache or be skipped. Verdict: 0.2-0.5 tok/s SSD-bound for a 70B — dead for interactive dense streaming.
> - **Perplexity lies at high sparsity**: Sirius shows ~50% contextual sparsity "almost collapses" GSM8K reasoning while looking fine on summarization. Any dense dial we ship must be evaluated on reasoning, not PPL.
> - **The proven dense path needs model conversion** (ReLUfication/ProSparse ~2.7% of pretrain cost, dReLU ~150B tokens) -> 87-90% sparsity at parity -> 5-15 tok/s for a 30B on a laptop. No public parity-grade converted checkpoints exist for Llama-3/Qwen-class. We don't train; WATCH for community checkpoints.
> - **The gem — sequence-level selection (GRIFFIN/CoreInfer/GLASS)**: pick ~50% of FF neurons once per prompt, one bulk SEQUENTIAL fetch, decode from RAM with zero per-token I/O. Training-free, SSD-native I/O pattern, cost ~ TEAL@50 (with the reasoning caveat). The only training-free dense scheme whose I/O pattern storage likes. PROTOTYPE CANDIDATE.
> - **Dense locality stats mirror our MoE findings**: adjacent-token active-set overlap >90% (drops to 70% by distance 10); hot 20-40% of neurons cover ~80% of activations; cross-layer firing predictability >90%. Same physics, finer granularity.
> - **Prefetch lead time matters**: TEAL/WINA know active rows only at execution time; predictor-based (Deja Vu/SparseInfer/SVD-predictors 2603.14110) and sequence-level schemes give the I/O system time to fetch. Mirrors our lookahead-vs-demand design.
> - MoEfication trend: conversion cost collapsed (200B tokens -> 4M tokens -> training-free ExpertWeaver 2602.15521; DOT-MoE ~90% of dense at 50% active). Converts dense into OUR native fetch unit (contiguous expert blocks). WATCH, prototype later.

---

[Full agent report preserved]

## TEAL (baseline): per-token magnitude thresholding, all 7 projections; ICLR'25. 40%: -0.5 to -1.5 pts zero-shot; 50%: -2 to -4 pts (Llama-3 worse); cliff past 50-65%. 1.53-1.8x decode. ~167 stars, gpt-fast only, no llama.cpp integration (#4559 open).

## CATS: gate-only thresholding ~ 25% model-wide; subsumed. Deja Vu: contextual predictors, 85% MLP sparsity on OPT(ReLU); needs ReLU zeros; ancestor of PowerInfer/LLM-in-a-flash. Sirius (2409.03856): reasoning collapse caveat at ~50%.

## ReLUfication: ReLU Strikes Back (~30B tokens); ProSparse (2402.13516): 87.9-89.3% sparsity at parity, 34.6B-89B tokens, public Llama-2-era checkpoints; TurboSparse/dReLU (2406.05955): ~90%, 150B tokens, vendor-scale. Sparsing Law (2411.02335): SiLU models get LESS sparse with more training - frontier dense won't drift sparse on its own.

## Post-TEAL training-free frontier: WINA (2505.19427, +2-3 pts over TEAL at equal sparsity, ICLR'26); R-Sparse (2504.19449, 50% robustly); LaRoSA (2507.01299, rotations, exact per-token ratio - I/O-schedulable); ActTail (2603.12272, pushes to 80% but lossy); SPON (2512.12744, quality floor improvement); GLASS (2508.14302, long-form fix); Polar Sparsity (2505.14884): MLP sparsity dies under batching, attention-head sparsity survives - batch-1 laptop decode is where MLP sparsity is maximal.

## Locality (the prefetch case): PowerInfer: 26-43% neurons = 80% activations. Hermes (2502.16963): adjacent-token similarity >90% (95% Falcon), 70% at distance 10; ~52% of static hot set churns during inference; layer-to-layer firing >90% predictable. LLM-in-a-flash windowing: 2.4-3.1% of FFN newly fetched per token at 95-97% sparsity. Co-activation bundling fails beyond hot set (<20% pairwise) - Ripple solves via Hamiltonian-path neuron placement (1.8x I/O bandwidth). Input-similarity pattern reuse (2412.12178): 70%-similar inputs -> 100% matching FFN patterns in 9/12 samples. 2509.00454: power-law + cross-token mask stability confirmed across Gemma/Llama/Qwen/DeepSeek.

## MoEfication line: MoEfication (2110.01786, ReLU-only); LLaMA-MoE v1 (200B tokens) -> v2 (7B tokens); CMoE (2502.04416: 4M tokens + 1h LoRA, >76% recovery); ExpertWeaver (2602.15521: training-free); DOT-MoE (2606.01666, ICML'26: ~90% of dense at 50% active).

## Verdict
- Training-free on stock SiLU: NO for interactive dense streaming (0.2-0.5 tok/s; quality collapse at streaming-grade sparsity).
- ReLUfied/dReLU converted models: YES marginally (30B at 5-15 tok/s laptop, 70B borderline 2-8), but no modern public checkpoints; conversion is vendor/community-scale.
- Middle path to prototype: sequence-level selection (GRIFFIN-class) - one bulk sequential fetch per prompt, zero per-token I/O, training-free, quality ~ TEAL@50 with reasoning caveat.
