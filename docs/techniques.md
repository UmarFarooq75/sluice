# The technique stack

Our approach = six pillars. 1–2 are proven elsewhere (we make them universal); 3–6 are the genuinely missing pieces nobody has shipped. They **compound** — see the multiplication table at the bottom.

## 1. Expert-granular streaming (foundation)

Dense trunk (attention, embeddings, shared experts, router) stays resident in RAM at int4 — it's small (e.g. 9.9 GB for 744B GLM; ~2 GB for 30B-class). Routed experts live on storage and stream through an expert-aware cache on demand. One coalesced multi-MB read per expert (sequential-bandwidth friendly). Cache auto-sized from free RAM. Proven by colibri/KTransformers; our contribution is doing it for **any GGUF MoE** via architecture-family adapters, not one hardcoded model.

## 2. Predictive prefetch + learned working sets (the brain)

- **Router lookahead**: apply layer L+1's router to layer L's output → ~71–85% recall on next-layer experts; fetch overlaps compute. Decays fast beyond 1 layer, so predict exactly one ahead, pipelined.
- **Learned per-task working set**: record which experts each of the user's tasks actually uses (5–15% of the model). Persist across sessions (tiny sidecar file). Sessions start warm — "gets faster the more you use it."
- **Shareable community profiles**: working-set files are kilobytes and portable. A "coding profile" for DeepSeek, published next to the model, warm-starts everyone. Open-source-native data flywheel; impossible for Ollama (no expert semantics).
- Eviction: least-stale / layer-aware, NOT plain LRU (measured LRU precision is only ~29% — expert access is layer-sequential, not recency-driven).

## 3. Expert-major prefill (the unsolved half)

Prefill under streaming naively re-reads the entire model: a 2,000-token prompt unions to nearly all experts per layer. Invert the loop:

```
for each MoE layer L:                      # instead of: for each token
    route all pending tokens through L's router
    for each expert e needed at L:
        load e once → process ALL tokens routed to e → evict
```

Disk cost becomes O(experts_per_layer), independent of prompt length. FlexGen-style batching applied to the expert dimension; no single-user streaming engine has this. Without it long prompts are unusable; with it, prefill cost for a 128-expert layer is the same for 100 tokens as for 10,000.

## 4. Usage-weighted mixed-precision storage

Uniform quantization hits a quality cliff below ~3 bits. Non-uniform doesn't have to: keep the learned hot ~15% of experts at 4-bit, store the cold tail at 2-bit. Rarely-fired experts contribute least (gate-weighted saliency — the same signal REAP prunes by). HOBBIT validated mixed-precision *fetching* at runtime; we make it the *storage layout*. Effective ROM ≈ 0.30 GB per B params instead of 0.60, with degradation concentrated where it's least felt. Re-tier periodically as the working set evolves.

## 5. Prune-at-install

"Install this 480B model at the size that fits your drive." REAP one-shot expert pruning: −25% size ≈ −2.8% coding quality (near-lossless on the biggest fine-grained models); −50% ≈ −8%. Offered as an install-time slider with honest quality labels per level. Also cuts streaming bandwidth proportionally. **Architecture-gated (do not promise a universal cut)**: near-lossless on shared-expert / fine-grained MoEs (OLMoE, DeepSeek — findings 22/23), but expert removal is ~4.4× more damaging on models with **no shared expert** to carry the hole (gpt-oss: substitute +2.2% vs skip +9.6% NLL, E26). The slider must label quality per family.

## 6. Network as the coldest tier (ROM = cache, not copy)

The biggest miss in the field. Store only the working set + margin locally (e.g. 35–40 GB of a 336 GB model); on a router miss for an expert not on disk, substitute-or-stall *once* while fetching it from HF/CDN in the background, then persist it and evict something cold. Internet at 100–500 MB/s is microSD-class bandwidth — a legitimate tier, and misses are rare by definition once warm. Game-industry asset streaming applied to weights. Enables the killer demo: **a 671B model from a 40 GB disk footprint** — but label it honestly as a *capability* demo (it runs **at all**), not a speed one. At ~37B active/token a frontier model streamed this way runs at **~0.1 tok/s** (colibri's measured GLM-5.2 on 25 GB RAM; E31), i.e. batch/agentic/overnight use, never interactive chat. The value is "runs where nothing else can," not "runs fast."

Cold-miss policy (tunable): (a) wait for network fetch; (b) similarity-substitute with the nearest *resident* expert (precomputed offline expert-similarity map; substitute only when the gate-weight gap is small) — a measurable quality/speed dial backed by our bit-exact gate. (Correction: an earlier draft called colibri's CACHE_ROUTE "blind, quality unquantified." That is inaccurate — CACHE_ROUTE is **opt-in**, **keeps the true top-J always**, follows **arXiv:2412.00099** max-rank selection, and self-reports **ROUTE_AGREE** overlap+KL telemetry vs true top-K. Our distinct piece is the precomputed offline similarity map, not that they lack instrumentation.)

## Anti-features (verified traps we refuse)

- **No speculative decoding while disk-bound** — measured net loss (colibri: 0.85→0.61 tok/s). Auto-enable only at high cache warmth.
- **No static global hot-expert pinning as a headline** — modern MoEs are balance-trained to uniformity; only per-task sets are cacheable.
- **No silent quality changes** — pruning, mixed precision, and substitution are explicit, labeled, user-chosen.
- **No speed lies** — measure the user's disk at install, print the expected tok/s range from the governing formula *before* loading.

## The multiplication table (DeepSeek-V3-class, 671B, int4 = 336 GB)

| Stack so far | ROM | RAM | Streamed/token |
|---|---|---|---|
| baseline int4 | 336 GB | 336 GB | — |
| + streaming (pillar 1) | 336 GB | ~24 GB | 10.2 GB cold |
| + prune 25% (pillar 5) | 252 GB | ~24 GB | 7.6 GB |
| + mixed precision (pillar 4) | ~150 GB | ~24 GB | ~5 GB |
| + working set @ 90%+ hits (pillar 2) | 150 GB | ~24 GB | ~0.5 GB → usable tok/s |
| + network tier (pillar 6) | **~40 GB** | ~24 GB | same |

Every row is individually validated in the literature; the composition is the invention.
