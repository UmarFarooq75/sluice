# Agent report 2/3: batch/throughput inference of bigger-than-RAM models — prior art through mid-2026

> Collected 2026-07-18 by a research subagent (web sweep). Feeds docs/dense-strategy.md.
> Key strategic reads for us, before the raw report:
> - **The niche is unowned and measured to be so**: llama.cpp's bigger-than-RAM batch path shows ~1.0x amortization (8 slots -> 2.08 tok/s aggregate vs 2.13 single, DeepSeek-R1 212GB on NVMe); FlexGen archived Dec 2024; all MoE batch systems (MoE-Gen 91 tok/s etc.) need CUDA + 256-512GB DRAM; colibri and AirLLM are single-stream/GPU respectively.
> - **The bound is reachable without a GPU**: dense 40GB @ 2.4GB/s, B=64 -> 3.8 tok/s aggregate (~110k tokens/night); laptop CPU int8 keeps pace with the streamed pass. Nobody has claimed it.
> - **Three proven multipliers never combined in one tool**: layer/expert-major sequential streaming (FlexGen/MoE-Gen), large effective batch, speculative verification of many tokens per streamed pass (SpecExec: ~20 accepted/pass).
> - **Existential risk named**: llama.cpp could absorb this niche with a single zig-zag disk scheduler; colibri's obvious pivot is batch mode. Speed matters.
> - **Batch amortization is EXACT-quality** — unlike activation sparsity (which also vanishes at batch>1, see report 1/3). Batch throughput and induced sparsity are competing designs for different products; batch wins on quality.
> - colibri today: 15.6k stars, GLM-5.2 only, 0.37 tok/s on a Framework 13 CPU-only laptop, ~11GB reads/token cold. Our A/B target confirmed.

---

[Full agent report follows verbatim]

## 1. FlexGen: mechanism, numbers, status, CPU-only viability

**Mechanism.** FlexGen (ICML 2023, arXiv:2303.06865): offloaded generation as graph traversal — "zig-zag block schedule" walks the (layer x micro-batch) grid column-wise so each layer's weights, fetched once from RAM or disk, are reused across a large effective batch. LP policy search picks weight/KV/activation placement; optional 4-bit group quant for weights and KV.

**Measured (single 16GB T4, 208GB CPU RAM, 1.5TB SSD ~2GB/s, prompt 512/gen 32):** OPT-175B no compression (weights partly on disk): 0.69 tok/s at effective batch 144. With 4-bit (fits CPU RAM): 1.12 tok/s. OPT-30B: 7.32-8.70 tok/s. Up to 100x over DeepSpeed ZeRO-Inference / HF Accelerate.

**GPU-dependence.** Hard CUDA requirement. No CPU-only mode ever shipped.

**Status.** Renamed FlexLLMGen, archived read-only Dec 1 2024 (9.4k stars, OPT-family only). Fork: FlexGen-Extension (UC Merced) added LLaMA. No active dense successor — the lineage moved to MoE systems. FlexGen is now a baseline newer papers beat.

## 2. llama.cpp / GGUF batch mode for bigger-than-RAM (2026)

**Nothing ships purpose-built offline-throughput for bigger-than-RAM — any hardware.** What exists:
- llama.cpp: mmap demand paging; llama-server continuous batching; batched-bench; --override-tensor/--cpu-moe; spec decoding. Batch scaling proven only in-memory (RTX 3090 ~15x single-stream aggregate; CPU-only Ultra 9 285K: 170 tok/s aggregate at batch 64 on an in-RAM 2.3GB model). Discussion #18030.
- Canonical bigger-than-RAM measurement (unsloth DeepSeek-R1 Q2_K_XL, 212GB on 96GB DDR5 + RTX 3090 + Gen5 NVMe 12.4GB/s): single stream 1.3-2.28 tok/s; **8 parallel slots -> 2.08 tok/s aggregate = zero batch amortization**; SSD reads only 2-5GB/s (mmap random faults). CPU-only 2.20 vs GPU 2.28 — disk-bound.
- Why: no layer-order/expert-order streaming scheduler, no double-buffered sequential O_DIRECT reads, no admission control aligning slots' expert choices. Zig-zag never implemented for GGUF.
- **Colibri**: GLM-5.2 (744B) on 25GB RAM — 9.9GB dense in RAM, ~370GB int4 experts NVMe-streamed (custom container, NOT GGUF, model-specific ~1,300 LOC), per-layer LRU + learned pinning + MTP self-spec, token-exact vs transformers. 0.05-0.1 tok/s (25GB box), **0.37 tok/s Framework 13 CPU-only**, 1.23 (9950X3D+5090), 2.06 (M5 Max Metal). ~11GB reads/token cold. **Single-generation, latency-oriented, no batching.** 15.6k stars.
- **AirLLM** v3.0 (Jun 2026): layer-by-layer disk streaming of HF checkpoints (DeepSeek-V3 671B on ~12GB VRAM); 35-100+ s/token single-prompt, **~5.3 s/token with a 50-prompt batch (~9 tok/s aggregate) — the only OSS data point showing layer-streaming batch amortization working**; GPU-oriented (community CPU fork exists); research-grade.
- Micro-Expert-Router (Rust, io_uring): theoretical only, author has no benchmark hardware.
- KTransformers: 512GB DRAM + 4090 workstation tier.

**Answer: no batch/offline-throughput bigger-than-RAM tool for GPU-less consumer laptops exists as of mid-2026.**

## 3. MoE-specific batch systems

- **MoE-Gen** (2503.09716): module-based batching — small attention batches, tokens accumulated in host memory, launched as huge expert-major batches. A5000/A6000 + 256-512GB DRAM, **no SSD tier**. Mixtral-8x22B decode 54-91 tok/s (DeepSpeed: 4); 8-31x over FlexGen/MoE-Lightning/DeepSpeed. Code public, research-grade.
- **MoE-Lightning** (ASPLOS'25, 2411.11217): CGOPipe CPU-GPU-I/O pipeline, paged weights, Hierarchical Roofline; up to 10.3x over SOTA (Mixtral 8x7B, single T4); 2-3x less CPU RAM. DRAM tier only.
- **Klotski** (ASPLOS'25, 2502.06888): expert-aware multi-batch pipeline; **the one academic system with a disk tier**; Mixtral-8x22B >1.3 tok/s on one RTX 3090. No public code.
- **MoE-Lens** (2504.09345): hardware-limit model (94% accuracy), CPU attention / GPU experts; 4.6x avg over SOTA — DRAM-tier MoE offload now near theoretical bound.
- 2026 frontier (all GPU+DRAM): DALI (2602.03495: 3.97x over llama.cpp), CoX-MoE (DAC'26, AMX), **WiSP (2606.21868: PCIe bandwidth, not expert-prediction accuracy, is the bottleneck)**, PreScope, HybriMoE, SpecMoEOff.

**Verdict: no shipped tool combines expert-major scheduling with SSD streaming.** Expert-major schedulers stop at DRAM + CUDA; shipped SSD streamers are single-stream. Klotski touches disk but is unreleased.

## 4. Speculative decoding x layer-streaming/offloading

- **SpecExec** (NeurIPS'24, 2406.02532): built for offloaded 70B; giant draft trees -> **up to ~20 accepted tokens per target pass**; Llama-2-70B 4-6 tok/s (4-bit) on 4090+RAM offload; 10.6-18.7x vs offloaded autoregressive.
- **Sequoia** (2402.12374): hardware-aware speculation trees; Llama2-70B on one 4090 with offload at 0.57 s/token (8x best offload baseline).
- **SpecOffload** (2505.10259): draft model in idle GPU capacity inside offload pipeline; Mixtral 8x22B 2.53x over FlexGen.
- **SpecMoEOff** (2508.21706): spec-dec to enlarge per-expert workload under MoE offload; CPU chunked-attention verification.
- OSS reality: llama.cpp spec-dec is in-memory only; colibri MTP is self-spec over SSD. **No published system pairs spec-dec with SSD-tier weight streaming; none CPU-only.** The amortization is demonstrated in principle, unshipped as product.

## 5. Economics: distance from the B x BW / bytes bound

Dense 40GB @ 2.4GB/s: pass = 16.7s -> aggregate <= 0.06 x B: **B=16 -> 0.96; B=64 -> 3.8; B=256 -> 15.4 tok/s** (~110k tokens per 8-hour night at B=64). Reads ~69TB/night at full bandwidth — NAND read-wear immaterial, laptop SSD thermals not.

- **FlexGen = record for SSD-tier efficiency: ~84% of its bound** (OPT-175B, 0.69 vs 0.82 theoretical) — but T4 GPU, 2023, unmaintained.
- DeepSpeed ZeRO-Inference: OPT-30B full-NVMe 30 tok/s on V100; Llama2-70B CPU-offload 3.65 tok/s @ batch 200 (A6000).
- DRAM-tier MoE systems within ~6% of their bound (MoE-Lens); frontier collapsed to interconnect bandwidth (WiSP).
- **Consumer CPU-only: nobody near the SSD bound.** llama.cpp ~1.0x amortization; colibri 0.37 tok/s single. CPU verify at B~64 keeps pace with a 16.7s pass -> bound reachable GPU-less, unclaimed.

## Verdict

The niche — CPU-only, SSD-streamed, batch-amortized overnight inference of huge models on laptops — is genuinely unowned as of mid-2026. Nobody combines the three proven multipliers (layer/expert-major streaming + large effective batch + speculative verification per pass) in one tool on any hardware tier.

Competitors: (1) llama.cpp — existential threat: GGUF gravity + all primitives; one contributor adding a zig-zag disk scheduler absorbs the niche; today it demonstrably doesn't amortize. (2) Colibri — proof of demand; best CPU-first SSD codebase; single-model, single-stream, non-GGUF; batch pivot is its obvious next move. (3) AirLLM — maintained, model-general, real amortization data, GPU-oriented, engineering-weak. (4) Academic MoE systems — wrong hardware tier, research code, but they published the playbook. Strategic notes: open-model frontier is MoE (expert-major + SSD is the harder, higher-value variant); Apple unified-memory laptops partially defuse the top of the consumer market.

[Sources preserved in agent transcript; key: arXiv 2303.06865, 2503.09716, 2411.11217, 2502.06888, 2504.09345, 2602.03495, 2606.21868, 2406.02532, 2402.12374, 2505.10259, 2508.21706; github FMInference/FlexLLMGen, JustVugg/colibri, lyogavin/airllm, EfficientMoE/MoE-Gen; llama.cpp discussion #18030; HF unsloth/DeepSeek-R1-GGUF discussion 13]
