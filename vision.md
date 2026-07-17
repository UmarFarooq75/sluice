# Vision: Virtual Memory for LLMs

**Run any large MoE model on hardware that "can't fit it" — small RAM, small storage, ordinary CPUs — with quality and speed that degrade gracefully instead of failing to load.**

## The one-line pitch

Ollama solved "run a model that fits your machine." We solve "run a model that doesn't."

## Why this is possible at all

A Mixture-of-Experts model only *uses* a tiny slice of itself per token (DeepSeek-V3: 37B of 671B; GLM-5: 40B of 744B; Qwen3-30B: 3.3B of 30B). Today's tools still demand the whole model sit in fast memory. We treat the model as a **cache hierarchy** instead:

```
VRAM  →  RAM  →  SSD  →  network (HuggingFace/CDN)
```

Every tier is a cache over the tier below. The router tells us which experts fire — and research shows it's 85–92% predictable ahead of time. So we keep the dense trunk resident, stream experts, prefetch what the router will want, and **learn each user's working set** (real tasks touch only 5–15% of experts).

## What makes ours different from everything that exists

| Existing project | Its ceiling |
|---|---|
| colibri | one hardcoded model (GLM-5.2); reactive cache; speed scales with RAM anyway |
| llama.cpp | dumb OS page-cache streaming; zero expert awareness |
| KTransformers | needs ~400 GB DDR5 + GPU |
| PowerInfer-2 / SmallThinker | fastest, but require retrained custom models |
| ProMoE / MoE-Infinity / SpecMD | smart predictors, research code nobody can install |

**Us: any GGUF MoE × any device × learned working sets × storage as a cache (not a copy).**

The six pillars (see [docs/techniques.md](docs/techniques.md) for detail):

1. **Expert-granular streaming** — dense weights resident, experts stream from disk.
2. **Predictive prefetch** — router-lookahead + learned per-task working sets; sessions start warm ("gets faster the more you use it").
3. **Expert-major prefill** — load each expert once per layer regardless of prompt length; makes long prompts feasible under streaming (nobody has shipped this).
4. **Usage-weighted mixed-precision storage** — your hot experts at 4-bit, the cold tail at 2-bit; ROM budget follows your usage.
5. **Prune-at-install** — REAP-style expert pruning to your disk budget (25% off for ~2.8% quality loss).
6. **Network as the coldest tier** — store only the working set locally; cold experts fetched and persisted on demand. **ROM becomes a cache, not a copy. Killer demo: a 671B model from a ~35–40 GB disk footprint.**

## Honesty as a feature

- The physics is one formula: `tok/s ≤ storage_bandwidth / ((1−hit_rate) × routed_bytes_per_token)`. The engine measures your disk and **tells you the expected speed before loading** — no bait-and-switch.
- Speculative decoding is a measured net *loss* while disk-bound (colibri confirmed) — we enable it only when the cache is warm.
- There are no globally-hot experts (modern MoEs are balance-trained to uniformity) — only *per-task* working sets are cacheable. That's why learning the user is the core mechanic, not an add-on.

## Current status

**Phase 0 (started 2026-07-17):** trace real router decisions, simulate cache policies offline, prove the numbers before building the engine. Lab rat: OLMoE-1B-7B 4-bit via MLX on the M2 Air — our dev machine *is* the target hardware. See [docs/plan.md](docs/plan.md).
