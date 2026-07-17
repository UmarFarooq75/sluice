# Agent report 1/3: dense-LLM streaming from flash/SSD — prior art through mid-2026

> Collected 2026-07-18 by a research subagent (web sweep). Feeds docs/dense-strategy.md.
> Key strategic reads for us, before the raw report:
> - **The gap is integration, again**: no maintained GGUF/llama.cpp-native dense streaming exists; PowerInfer's fork is stale+ReLU-only, PowerInfer-2 was never open-sourced, ActiveFlow/Neuralink/Lever are paper-only. No GGUF metadata standard for predictors/thresholds/neuron layout.
> - **Exact dense sparsity requires vendor-scale retraining** (ReLUfication/dReLU, ~150B tokens); training-free (TEAL/CATS/DIP) stalls at 40-70% with visible quality cost. An order of magnitude short of the 90-97% that made the ReLU-era papers fast.
> - **The 2026 frontier moved to speculative amortization** (Lever): draft in DRAM, flash-resident target verifies many tokens per sweep — lossless. Composes with everything.
> - **PowerInfer's own team pivoted from dense streaming to natively-sparse MoE** (SmallThinker) — independent confirmation of our core bet.
> - **Batch>1 destroys activation-sparsity savings** (Polar Sparsity) — the dense-sparsity path and the batch-throughput path CONFLICT; they serve different products.
> - Watch: FlashMoE (arXiv:2601.17063) — learned cache replacement for SSD-resident MoE experts, adjacent to our core; add to novelty tracking.

---

# Dense-LLM inference from flash/SSD on small-RAM machines — state of the art through mid-2026

## 1. Apple "LLM in a flash" line

**Original paper** — "LLM in a flash: Efficient Large Language Model Inference with Limited Memory," Alizadeh et al., Apple, arXiv:2312.11514 (v3 Jul 2024), ACL 2024. https://arxiv.org/abs/2312.11514 ; https://machinelearning.apple.com/research/efficient-large-language

- **Mechanism.** Keep attention weights + embeddings resident in DRAM (~1/3 of model); FFN weights live in flash and are loaded on demand. Three components: (a) a small low-rank per-layer **predictor** guesses which FFN neurons will fire after ReLU; (b) **windowing** — maintain a sliding window (k≈5 recent tokens) of loaded neurons in DRAM, so per token you only load newly-activated neurons and evict stale ones (incremental transfer ~2-4% of FFN per token); (c) **row-column bundling** — store the i-th column of the up-projection and i-th row of the down-projection contiguously so one neuron = one larger sequential read (flash throughput scales with chunk size; small random reads are the killer).
- **Sparsity type: exact** (true ReLU zeros). OPT-6.7B ~97% FFN sparsity; Falcon-7B ~95% *after ReLUfication fine-tuning*; also evaluated Persimmon-8B (ReLU-family). Non-ReLU models must be ReLUfied first — that is the paper's hard model-family constraint.
- **Numbers.** DRAM budget ~50% of model size; runs models up to ~2x DRAM. Per-token FFN I/O latency for OPT-6.7B cut from ~2196 ms (naive full-FFN load) to ~87 ms. End-to-end: **4-5x vs naive loading on CPU (Apple M1 Max), 20-25x on GPU (RTX 4090 + NVMe)**. Note the baseline is *naive flash reload*, not an in-DRAM model — absolute speeds remain single-digit tok/s territory on CPU.
- **Quality cost.** Predictor-based selective loading itself reported near-zero zero-shot deltas (false negatives tuned to be rare); the real quality cost is **ReLUfication** of SiLU/GELU models (Falcon/Persimmon), which needs fine-tuning and doesn't fully recover the original model. Precursor: "ReLU Strikes Back" (Apple, ICLR 2024) https://machinelearning.apple.com/research/relu.
- **Follow-ups 2025-2026 in this line:**
  - **DIP — "Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking"** (Qualcomm AI Research; arXiv:2412.01380, rev. Apr 2025). https://arxiv.org/abs/2412.01380 . Drops the predictor and the ReLU requirement: magnitude-based dynamic input pruning for **SwiGLU** models, with a masking policy that prefers neurons already in the DRAM cache (trades a slightly worse neuron choice for a cache hit), plus optional LoRA "recovery" adapters. Phi-3-Medium in a mobile-simulation harness: **46% memory reduction, +40% throughput, <0.1 perplexity increase**. Sparsity: **induced** (thresholding).
  - **VLM in a flash — Neuron Chunking** (Yang et al., SNU et al., arXiv:2511.18692, Nov 2025). https://arxiv.org/abs/2511.18692 . Extends flash-offloaded sparse inference to non-ReLU VLMs (which only tolerate 40-60% sparsity); selects **contiguous chunks of neurons** by importance-per-latency utility, aligning sparsity choice with storage access cost. I/O efficiency up **4.65x (Jetson Orin Nano) / 5.76x (Jetson AGX Orin)**. Sparsity: **induced**.
  - Apple itself has published no direct public successor system; Apple Intelligence ships a ~3B fully-in-DRAM model; no Apple flash-streaming runtime in MLX or Core ML.

## 2. PowerInfer and PowerInfer-2

**PowerInfer** (SJTU-IPADS, SOSP 2024; repo now under Tiiny-AI). https://github.com/Tiiny-AI/PowerInfer ; https://powerinfer.ai

- **Mechanism.** GPU-CPU hybrid for consumer PCs (cold tier = host DRAM/CPU, not SSD): power-law "hot" neurons preloaded to GPU; "cold" neurons on CPU. Online per-layer MLP **activation predictors**; offline ILP solver assigns neurons; adaptive sparse operators. Ships as a **llama.cpp fork** with custom "PowerInfer GGUF" bundling weights + predictor weights.
- **Numbers.** RTX 4090 (24GB): avg **13.20 tok/s, peak 29.08 tok/s** on Falcon(ReLU)-40B-FP16, **up to 11.69x vs llama.cpp**; ~3x on Llama-2-70B class; RTX 2080Ti INT4 Falcon-40B up to ~8x.
- **Model constraint.** ReLU/ReGLU/squared-ReLU families only: ReluLLaMA-2 (7B/13B/70B), ReLU-Falcon-40B, ProSparse-LLaMA-2 (7B/13B, ~88-89% sparsity, ~parity accuracy per https://arxiv.org/abs/2402.13516), Bamboo-7B. **No stock Mistral/Llama/Qwen (SiLU)**. Quality caveat in README: ReluLLaMA-70B recovered with only ~5B tokens → measurable capability loss; predictor misprediction adds a small further hit (<~1% task delta on the ReLU models).
- **Sparsity type: exact** (true ReLU zeros), predicted ahead of compute.
- **Maintenance.** Repo alive but redirected: last significant update **Jan 5, 2026** ("Tiiny AI Pocket Lab"); macOS Metal "coming soon"; core dense-from-SSD engine effectively frozen. Team's research pivoted to natively-sparse **MoE** trained for offload (SmallThinker, arXiv:2507.20984).

**PowerInfer-2** (arXiv:2406.06282). https://arxiv.org/abs/2406.06282 ; https://powerinfer.ai/v2/

- **Mechanism.** Phone version: matmuls decomposed into **fine-grained neuron clusters**. Prefill = NPU-centric dense; decode = CPU-centric with predictors (~10% neurons active), neuron-cluster I/O/compute pipelining, segmented neuron cache, UFS-aware I/O (single UFS command queue; multi-core reads degrade up to 40%; 4KB-granular reads; bundled gate/up/down as 2x4KB).
- **Hardware/numbers.** OnePlus 12 (SD 8 Gen 3, 24GB, UFS 4.0; seq ~4 GB/s, 4KB random ~450 MB/s) and OnePlus Ace 2. **TurboSparse-Mixtral-47B: 11.68 tok/s in-memory (21-22x vs llama.cpp), 9.96 tok/s at 50% offload (25.4x)**; TurboSparse-Mistral-7B (dense) ~11-12 tok/s with 50% FFN offloaded (~20x); vs their LLM-in-a-flash reimpl: **avg 3.94x, up to 4.38x**. ~40% DRAM reduction.
- **Model constraint.** dReLU "TurboSparse" conversions only. **Quality:** system paper reports *no accuracy numbers*; TurboSparse paper (arXiv:2406.05955): dReLU + **150B tokens continued pretraining** → FFN sparsity 90%/97%, benchmarks match-or-exceed originals (recovery paid with vendor-scale compute).
- **Status.** **Engine never open-sourced** — only TurboSparse weights (https://huggingface.co/PowerInfer); no releases since 2024.

## 3. 2025-2026 successors: neuron/row-granular dense streaming

- **Neuralink (formerly Ripple)** — ASPLOS 2025, arXiv:2410.19274. Offline reorganizes **neuron placement in flash by co-activation** so activated sets become contiguous reads (IOPS→bandwidth shift); online access-strategy engine. OnePlus phones; 7 models incl. **SiLU/GELU (MobiLlama, Phi-2)**. **1.49x avg over LLM-in-a-flash reimpl, 2.37x over llama.cpp; I/O bandwidth +1.80x/3.28x**; up to ~4.6x. Layout-only → no quality cost of its own. [Same insight as our finding: co-activation layout, at neuron granularity.]
- **ActiveFlow** — MSR + Tsinghua/CSU; arXiv:2504.08378. Most credible dense-SiLU answer: **top-K magnitude active weights** (no predictor, no ReLU), **cross-layer preloading**, **sparsity-aware self-distillation** (~10 GPU-hours per model), DRAM-flash pipeline. Phones (UFS 2.2-4.0): Q4 Llama-2/3 + Mixtral-8x7B at 2.9-4.3GB DRAM; ~0.4-5.9 tok/s; ~40% memory cut at parity speed; at 70% sparsity PPL 9.51 vs TEAL's 13.68; GSM8K up to +11 pts vs raw top-K. Limits: CPU-only, ~5% demand-loaded, 54% collapse on slowest flash.
- **Lever** — arXiv:2605.16786 (May 2026). Draft model in DRAM, flash-resident target; I/O- and compute-aware token trees, early-exit branch pruning, CPU-NPU mapping. **2.93x avg over flash-offloaded baseline, 1.50x over vanilla speculative.** Lossless (speculative verification). The 2026 frontier: amortize each flash sweep over multiple drafted tokens.
- **FlexInfer** — EuroMLSys 2025, arXiv:2503.03777. Sparsity-free offload engineering: async prefetch, balanced memory locking, tensor-preservation under RAM budget; up to 12.5x vs mmap-thrash baselines. No sparsity → bounded by full-weight bandwidth/token.
- **NimbleEdge sparse_transformers** — https://github.com/NimbleEdge/sparse_transformers (Apache-2, 2025). Deja-Vu-style LoRA predictors + fused sparse kernels for **Llama-3.2/Mistral (SiLU)**: 1.6-1.8x TTFT/TPS, ~5x MLP speedup on CPU. DRAM-resident today; flash streaming on roadmap. Closest maintained OSS successor; not GGUF/llama.cpp.
- **llama.cpp mainline: nothing neuron-granular for dense.** Only mmap paging, KV/layer offload, MoE expert offload (`--cpu-moe`/`--override-tensor`). PowerInfer fork never upstreamed; **GGUF has no predictor/sparsity-metadata standard.**
- Adjacent: ENDOR (2406.11674) sparse-format offload; InstInfer (2409.04992) in-storage attention; KVNAND (2512.03608) DRAM-free in-flash compute; **FlashMoE (2601.17063) learned cache replacement for SSD-resident experts (MoE — adjacent to our core)**; Polar Sparsity (2505.14884): **activation sparsity vanishes at batch>1**.

## 4. Quality/provenance table

| System | Sparsity | Quality cost reported |
|---|---|---|
| LLM in a flash | Exact (ReLU, predicted) | ~0 for native-ReLU; ReLUfication imperfect for others |
| PowerInfer | Exact (ReLU, predicted) | <~1% predictor delta; ReLUfied checkpoints degraded |
| PowerInfer-2/TurboSparse | Exact (dReLU) | parity after 150B-token retrain (vendor-scale) |
| TEAL (2408.14690) | Induced | ~0 @25%, minimal @40%, visible @50% (worse on Llama-3); 1.53-1.8x |
| CATS (2404.08763) | Induced | within 1-2% @50% FFN, no finetune; ~15% wall-clock |
| DIP (Qualcomm) | Induced + LoRA | <0.1 PPL @46% memory cut |
| ActiveFlow | Induced + distill | PPL 9.51 vs TEAL 13.68 @70% |
| Neuralink/Ripple | Layout-only | none |
| Lever | None (speculative) | lossless |

Pattern: **exact sparsity needs vendor-scale retraining and is stuck on 2023-24 models; induced sparsity works on modern SiLU models but saturates ~40-60% before quality slides.**

## 5. Three biggest unshipped gaps

1. **No maintained GGUF/llama.cpp-native neuron-granular dense streaming stack.** No GGUF metadata standard for predictors/thresholds/neuron layout. Nobody can `llama-cli -m model.gguf --stream-ffn-from-ssd` a dense model today.
2. **High (>70-90%) quality-neutral sparsity on current SwiGLU dense models without vendor-scale retraining does not exist.** Training-free methods stall at 40-70% with visible degradation.
3. **Everything is single-sequence decode.** Prefill is dense (sparsity unions to ~everything), batch>1 destroys sparsity savings (Polar Sparsity), long-context/agentic workloads unmeasured, and system papers ship speed numbers with no accuracy evals — no standardized speed-vs-quality benchmark for flash-offloaded inference.
