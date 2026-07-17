# Research findings (verified July 2026)

Distilled from a 20-agent research sweep (5 research dimensions, ~15 load-bearing claims adversarially verified against primary sources). This is the evidence base for the project — cite these numbers, don't re-derive them.

## The governing formula

```
tok/s ≤ storage_bandwidth / ((1 − hit_rate) × routed_bytes_per_token)

routed_bytes_per_token = n_moe_layers × top_k × 3 × d_model × d_expert_ff × bytes_per_param
```

Only routed experts stream; attention/dense/shared-expert weights are always active and stay resident in RAM. Every percentage point of cache hit rate matters linearly.

## Routed traffic per token at int4 (from HF config.json, verified)

| Model | Total / active | GB streamed per token (0% hits) | Expert size |
|---|---|---|---|
| Mixtral 8x22B | 141B / 39B | **16.9 GB** (worst surveyed) | 151 MB |
| GLM-5 | 744B / 40B | 11.3 GB | 18.9 MB |
| Kimi K2 | 1.04T / 32B | 10.6 GB | 22 MB |
| DeepSeek-V3/R1 | 671B / 37B | 10.2 GB | 22 MB |
| Qwen3-235B-A22B | 235B / 22B | 7.1 GB | 9.4 MB |
| GLM-4.5-Air | 106B / 12B | 3.1 GB | 8.7 MB |
| gpt-oss-120b | 117B / 5.1B | 1.9 GB (MXFP4) | 13.2 MB |
| Qwen3-30B-A3B | 30B / 3.3B | 0.91 GB | 2.4 MB |
| Qwen3-Next-80B | 80B / 3B | **0.76 GB** (best per capability) | 1.6 MB |
| OLMoE-1B-7B | 6.9B / 1.3B | 0.40 GB | ~3 MB |

Key structural insight: **expert granularity beats total size.** Mixtral 8x22B (141B, monolithic experts) streams more per token than 1T Kimi K2 (fine-grained). Fine-grained MoE (many small experts, high top-k) is the streaming-friendly direction, and it's where the field is going.

## Storage tiers (sequential read; verified specs/measurements)

PCIe5 NVMe 14.5 GB/s · PCIe4 7.45 · UFS 4.0 (phones) 4.2 · PCIe3 3.5 · UFS 3.1 2.1 · Pi5 NVMe Gen3x1 ~0.84 · SATA 0.56 · eMMC 0.4 · microSD 0.104. DRAM is 6–56× above the best SSD (DDR5 dual 89.6 GB/s, M4 Max 546). Expert reads are multi-MB so sequential bandwidth is the right bound — but QD1 4K random is ~86× slower, so reads must be coalesced/prefetched.

## Ceilings that matter (PCIe4 NVMe, at 0% / 90% / 98% cache hits)

- DeepSeek-V3: 0.73 / 7.3 / 36.5 tok/s
- gpt-oss-120b: 3.9 / 39 / 196 tok/s
- Qwen3-30B-A3B: 8.2 / 82 / 411 tok/s

**No frontier MoE exceeds ~2 tok/s uncached on any consumer storage.** Cold streaming is physics-dead; the cache/predictor is everything.

## colibri (github.com/JustVugg/colibri) — what it proved

- 744B GLM-5.2 genuinely runs in ~25 GB RAM (dense trunk ~9.9 GB int4 resident; ~370 GB experts streamed via per-layer LRU + learned pin file + io_uring; router-lookahead prefetch 71.6% next-layer recall).
- But community benchmarks show **speed scales with RAM, not streaming cleverness**: 0.07–0.11 tok/s @ 24 GB → 1–2 tok/s @ 128 GB → 3.6 @ 1TB → 6.0 @ 6×RTX 5090.
- "Gets faster the more you use it" is real: 0.16 → 0.40 tok/s over five sessions via learned pinning (per-task working sets).
- **VERIFIED: MTP/speculative decoding is a net LOSS when disk-bound** (0.85→0.61 tok/s measured) — draft tokens amplify expert loads faster than acceptance pays back. Only enable speculation when cache is warm.
- **VERIFIED: quality is unresolved** — only eval is a confounded 62.5% acc_norm (n=40) vs 85–95% reference. No A/B vs llama.cpp mmap exists. Both are gaps we must not repeat.

## Prior art map

- **Apple LLM-in-a-flash** (arXiv 2312.11514): flash streaming + FFN sparsity prediction, 2× DRAM headroom, windowing + row-column bundling.
- **PowerInfer-2**: 47B MoE at 11.68 tok/s *on a phone* from UFS flash — but requires a retrained sparse model (TurboSparse).
- **SmallThinker** (same team): MoE *trained for streaming* — router fires pre-attention so SSD fetch overlaps attention compute; 21B-A3B > 20 tok/s in 8 GB RAM; up to 85× Qwen3-30B at same memory cap. **Co-design beats post-hoc by an order of magnitude.**
- **KTransformers**: DeepSeek-R1 671B at 11–13 tok/s decode — but needs 24 GB GPU + ~400 GB DDR5.
- **llama.cpp**: `--override-tensor`/`--cpu-moe` = expert offload to RAM; disk story is plain mmap paging (no expert semantics). DeepSeek-R1 from NVMe: ~2–3.5 tok/s on 96 GB rigs. This is our A/B baseline.
- **ProMoE / SiDA-MoE / SpecMD / Pre-gated MoE / ExpertFlow**: learned expert predictors, 84–92% accuracy, 1.3–3.9× speedups — all research code, none installable. The gap we fill.
- **HOBBIT**: mixed-precision expert fetching (cache-missed low-importance experts fetched at int4 instead of fp16) — up to 9.93× on edge. Validates our mixed-precision storage idea at runtime.
- **AirLLM**: dense layer-by-layer streaming, 0.5–2 tok/s — shows why sparsity is mandatory.

## Expert locality & caching (the science)

- **No global hot experts.** Aux-loss-free balancing (DeepSeek-style) drives global usage to near-uniform (MaxVio 0.04, verified). Static pinning is structurally weak.
- **Strong per-task concentration.** Within a task, 5–15% of experts carry most gate weight; different tasks use nearly disjoint sets (ESFT, DeepSeek-V2-Lite). → Learn the user's working set, per task.
- **Temporal locality is modest**: consecutive tokens repeat first-choice expert 24–28% at deep layers (random = 12.5%); code/math show the most locality. Plain LRU: only ~60–70% hits caching 2/8 experts; LRU precision measured at just 29% — expert access is deterministic layer-sequential, not recency-driven.
- **Prediction works**: one-layer-ahead gate lookahead ~84.6% precision; learned predictors (ProMoE 84.7%, SiDA >90%, SpecMD 88–92% at 5% cache). Accuracy decays fast with lookahead distance (30–40% at 2 layers).
- **Expert pruning (REAP, Cerebras)**: one-shot removal of 25% of experts ≈ −2.8% coding quality (Qwen3-Coder-480B: −0.2%); 50% ≈ −8%. Pruning ≫ merging. Fine-grained MoEs prune gracefully.
- **Speculative decoding amortizes loads only above the disk tier**: distinct experts across draft tokens grow sub-linearly (SpecMoEOff 2.5×, SP-MoE 88.9% prediction) — when experts live in DRAM/PCIe. On disk-bound setups it inverts to a loss (see colibri).

## Small-footprint reality (phones, Pi, small laptops)

- Disk cost ~0.60 GB per B total params at int4; ~0.35–0.40 at 2-bit. 64 GB storage fits a 30–35B MoE at int4; 256 GB fits 235B-class.
- **Quality cliff below ~3 bits** for uniform MoE quant (KLD roughly triples 3-bit → 2-bit, Unsloth + ByteShape independently). No trained ternary MoE exists yet (BitNet is 2B dense).
- **If the whole MoE fits in RAM it wins; if not, naive MoE loses to dense** (verified: OLMoE 31% slower than Llama-3.2-1B on Jetson) — because bandwidth-bound cost tracks *total* params. Intelligent streaming is what flips this.
- Best small MoEs mid-2026: Qwen3.6-35B-A3B (73.4% SWE-bench, ~21 GB Q4), gpt-oss-20b (12.8 GB), LFM2.5-8B-A1B (~5 GB, ~30 tok/s on phones), SmallThinker-21B (23 tok/s on Snapdragon 8 Elite), Granite-4.0-h-tiny (4.2 GB), OLMoE-1B-7B (4.2 GB).
- Measured: Qwen3-30B-A3B at 8 tok/s on a Raspberry Pi 5 (16 GB, 2.7bpw). The small-MoE class is genuinely good now.

## Key primary sources

colibri: https://github.com/JustVugg/colibri · LLM-in-a-flash: arXiv 2312.11514 · PowerInfer-2: arXiv 2406.06282 · SmallThinker: arXiv 2507.20984 · KTransformers: github.com/kvcache-ai/ktransformers · llama.cpp expert offload: PR #11397 · ProMoE: arXiv 2410.22134 · SiDA-MoE: MLSys 2024 · REAP: arXiv 2510.13999 · Loss-free balancing: arXiv 2408.15664 · SpecMoEOff: arXiv 2508.21706 · ESFT: arXiv 2407.01906 · MoE-on-edge study: arXiv 2506.12119 (+ 2606.21428) · HOBBIT: arXiv 2411.01433 · FlashMoE: arXiv 2601.17063
