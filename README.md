# Working title: an engine for models that don't fit

> **Status: research phase (Phase 0 complete, working draft).** Everything below is measured, reproducible from this repo, and honest about its limits. Started 2026-07-17.
> **Novelty disclosure:** an adversarial prior-art audit ([docs/novelty-audit.md](docs/novelty-audit.md)) found much of the mechanism space was independently published in 2024–2026 (we cite the closest works there, including mbolt, Cache-Conditional Experts, SS-MoE, OpenMoE's context-independent specialization, Klotski, Sem-MoE, MoDES). Our measurements stand as independent reproductions on a $1,000 laptop; our genuine contributions are the composition-effect measurements, the controls/verification culture, and the *integration* — no shipping engine composes these mechanisms for arbitrary GGUF MoE models. That integration is the project.

**The idea: run large Mixture-of-Experts LLMs on machines that "can't fit them" — small RAM, small storage, no GPU — by treating the model as a cache hierarchy (VRAM → RAM → SSD → network) with a learned page table, instead of a blob that must fit in memory.**

Ollama solved *running a model that fits your machine*. We're solving *the model that doesn't*.

## Why this is physically possible

A MoE model touches only a sliver of itself per token (DeepSeek-V3: 37B of 671B params). Streaming exactly that sliver from storage is bounded by one formula:

```
tok/s ≤ storage_bandwidth / ((1 − cache_hit_rate) × routed_bytes_per_token)
```

Everything in this project is an attack on one of those three variables.

## What we've proven so far (all on a $1,000 MacBook Air M2, 16 GB)

We hooked the router of a real MoE (OLMoE-1B-7B, 64 experts/layer), logged every routing decision across six workload types, simulated cache policies offline, then **built a physical proof-of-concept**: the model's 1,024 experts stored as individual files on SSD, streamed per token with the page cache bypassed.

**The physical result table** (`results/streamer_poc.json`):

| mode | tok/s | expert RAM | quality (NLL) |
|---|---|---|---|
| everything resident (reference) | 76.7 | 3.6 GB | 3.002 |
| **floor mode: all experts on SSD** | **3.5** | **~0** | **3.002 — bit-consistent** |
| half cached + margin routing | **16.2** | 1.8 GB | **3.002 — bit-consistent** |
| half cached + lookahead prefetch | 8.9 | 1.8 GB | 3.002 |
| all cached, warm | 25.9 | 3.6 GB | 3.002 |

Key measured findings (details in [docs/findings-phase0.md](docs/findings-phase0.md)):

1. **Streamed compute is bit-correct** — identical NLL to the resident model in every mode. (The correctness gate prior systems skipped.)
2. **The bandwidth formula holds on real hardware** — measured speed tracks the I/O ceiling; I/O dominates exactly when predicted.
3. **Margin-gated cache-aware routing** — fetch a non-resident expert only when the router wants it ≥m more than the weakest resident pick — **doubles real speed at zero measured quality cost** (NLL *and* exact-match answer accuracy). A smooth quality↔traffic dial, quantified for the first time.
4. **Expert-major prefill**: prompts touch ~all experts, so loading each expert once per layer instead of per-token cuts prefill I/O **16×–599×** (grows with prompt length). No shipped engine does this.
5. **Router-lookahead prefetch generalizes and works physically**: applying layer L+1's router to layer L's state recalls **83.9%** of the true next-layer experts (beats the 71.6% reported on GLM-5.2); on the physical streamer it delivers **+72% throughput at small cache sizes**. Measured composition effect: prefetch and margin routing *substitute* rather than stack (they attack the same miss latency) — the engine coordinates them instead of enabling both blindly.
6. **The router is a free task classifier**: task switches detected in ~6 tokens from routing statistics alone; per-task expert profiles are real (code≈math, but code∩prose ≈ 8%) — so the engine *learns you* and starts warm.
6b. **Expert-skip (our mechanism, validated on DeepSeek-V2-Lite)**: on router-indifferent tokens, attention runs normally (KV stays intact) but routed experts are skipped — shared experts carry the token. Skips 15–18% of all routed I/O + FLOPs for 1.4–2.5% NLL on math/prose, and the inverse control (skipping *confident* tokens: 50–100× more damage per token) proves the trigger signal is real. Replication also revived adaptive top-k for DeepSeek-family models (k=3-of-6 costs just +3.6–7.7% there vs +14–25% on OLMoE) — per-architecture validation matters, twice over.
7. **Hybrid caching is the right design**: learned pinned base beats LRU 9× at tiny capacities; LRU wins when RAM is comfortable; oracle (Belady) shows a further 15–20-point prize for prediction.
8. **Honest negatives we caught early**: hard-pinning without a fetch path destroys quality (+129% NLL — the tail must always be reachable); adaptive top-k is dead on flat softmax routers like OLMoE's (per-architecture, not universal); the network tier trims the *last-mile* of ROM (~10–25%), it is not a 4× cut; speculative decoding is a net loss while disk-bound (from colibri's own data).

## The head-to-head (2026-07-18): a model this laptop "cannot run"

**Qwen3.6-35B-A3B** (2026 release, 26.5GB Q5_K_M) on the same 16GB MacBook Air:

| engine | result |
|---|---|
| stock llama.cpp mmap — the engine under Ollama/LM Studio | **did not finish**: OOM-killed during load; retry produced no output in 10 minutes |
| **ours, margin m=0.02, 2.9GB expert cache** | **8.3 tok/s, coherent output** — ~80% of this CPU's physical ceiling for the model's active params |
| ours, exact routing (m=0), 5.9GB cache | 3.6 tok/s |

The margin router pushes I/O below the compute floor: a 2.9GB cache ties a 5.9GB one. Total RAM used ≈ 6GB for a 26.5GB model. Family adapter for this brand-new architecture (hybrid attention, merged gate_up, shared expert): ~20 lines. Details: `results/qwen36_headtohead.json`, finding 33.

## Phase 1 milestone M1 — it now runs inside llama.cpp, bit-exact (2026-07-17)

A ~135-line private fork of llama.cpp (`patches/llmstream.patch`) plus an out-of-tree driver (`csrc/stream_run.cpp`) streams experts **straight from the GGUF file's per-expert extents** (no conversion, no repacked store) into per-layer slot caches, rewriting router ids to cache slots mid-graph via the public `cb_eval` hook. The correctness gate hashes every generated position's full logit vector:

| config (OLMoE Q4_K_M, M2 Air, CPU, cold SSD) | expert RAM | decode | quality |
|---|---|---|---|
| stock llama.cpp, fully resident (best case) | 3.6 GB | 45.2 tok/s | reference |
| **streamed, 32 slots/layer** | **1.8 GB** | **28.3 tok/s** | **bit-identical** |
| **streamed, 12 slots/layer** | **0.66 GB** | **9.1 tok/s** | **bit-identical** |

3.4× our Python PoC at the same cache. Fetches don't yet overlap compute, prefetch and margin routing aren't wired in — those are M2 and each has measured headroom. One lesson worth stealing: our first gate run *failed* because llama.cpp silently repacks ARM weights into interleaved layouts whose gemm kernels round differently — **bit-exactness claims must pin the kernel path, not just the math** (finding 32).

**Assumption-breaking round (day 2):** questioning our own premises produced three confirmed novel results — **routing is ~2/3 token-determined, on both router families** (a ~6 MB static table built by counting prefetches ~50–60% of *all* layers' experts at embedding time, before any compute — full-depth prefetch nobody ships); **expert placement is a graph problem** (laying out experts by co-activation cuts read ops 1.42–1.55×, offline, lossless); and **agent swarms stream sublinearly** (5 parallel agents cost the disk of ~3 — fleets get *cheaper* per agent). Three honest negatives with explanations: expert files barely compress and contain zero duplicate blocks (int4 ≈ max entropy); cache-restricted self-speculation is structurally capped at ~1.2–1.7× by consecutive-token expert overlap; and progressive/partial-expert fetching is dead because **fine-grained experts are internally dense — the router already harvested the activation sparsity at training time** (which also predicts TEAL-class row-skipping won't transfer to fine-grained MoE).

## The engine these findings design

Compiled core (the PoC's only remaining gap to resident speed is interpreter overhead), GGUF-compatible, no GPU required:

- **Per-layer expert store** on disk, direct I/O, one coalesced read per expert
- **Hybrid cache**: learned pinned base + reactive top-up, uniform per-layer allocation (measured: smart allocation only pays when starved)
- **Margin router** (the quality↔bytes dial, default m≈0.02)
- **Lookahead prefetcher** (one layer ahead, async, 84% recall)
- **Expert-major prefill** (mandatory, not optional)
- **Working-set persistence**: per-task profiles saved across sessions, shareable ("warm-start packs" for popular models)
- **Honest speed estimator**: measures your disk at install, prints expected tok/s *before* downloading anything
- **Correctness gate in CI**: streamed output must match resident output on every code path
- Architecture adapters: OLMoE (test), Qwen3-MoE, DeepSeek-family (covers GLM/Kimi), gpt-oss

Target machines, in order: ordinary no-GPU laptops (8–16 GB) · cheap NVMe mini-PCs (the 671B-class showcase tier) · flagship phones (UFS 4.0) · Pi 5/Jetson for small MoEs. Non-targets: GPU rigs, datacenters, SD-card devices.

## Roadmap

- **Phase 0 — evidence** ✅ (this repo: traces, simulators, quality/accuracy evals, physical PoC)
- **Phase 0.9 — replication**: DeepSeek-V2-Lite (sigmoid router + shared experts) re-run; decides adaptive-k and expert-skip per-architecture
- **Phase 1 — engine core**: compiled runtime, GGUF in, A/B vs llama.cpp mmap on identical hardware
- **Phase 2 — the brain**: prefetcher + persistent working sets + shareable profiles
- **Phase 3 — the ROM stack**: usage-weighted mixed precision (hot 4-bit / cold 2-bit), REAP-style prune-at-install, network-as-coldest-tier
- **Phase 4 — edges**: Metal/NPU tiers, phone build

## Repo layout

```
vision.md                 north star
docs/                     research base, techniques, red-teams, findings
src/tracer.py             router hook → routing traces
src/simulate.py           cache-policy simulator + analyses (v2)
src/quality_eval.py       restricted-routing NLL eval
src/accuracy_eval.py      exact-match answer eval
src/session_switch.py     task-switch detection
src/lookahead.py          prefetch recall measurement
src/network_tier.py       disk-budget / network-tail simulation
src/dp_alloc.py           exact cache-allocation DP
src/build_store.py        per-expert store builder
src/streamer_poc.py       the physical SSD-streaming proof of concept
traces/ results/          data (generated)
```

Every number in this README has a JSON artifact in `results/` and a script that regenerates it.
