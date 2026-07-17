# Flaws in the borrowed research — and how WE cover them

Principle (Umar, 2026-07-17): using others' research is fine, but every borrowed result has flaws/hidden assumptions we must find and cover ourselves. Each item: the flaw → our coverage. The coverage list IS our experiment roadmap; closing these gaps is what makes the project defensible rather than a glue-job.

## 1. TEAL activation sparsity — validated on DENSE models only

**Flaws:** (a) MoE experts are already a sparsity mechanism — expert activations may be denser and less prunable than dense-model FFNs; the 40–50% may not transfer. (b) Assumes zero-mean unimodal activation distributions — unverified for expert MLPs. (c) Row-skipping turns sequential disk reads into scattered gathers — from SSD this can be *slower* than reading the full expert unless the storage layout bundles co-surviving rows.
**Coverage:** measure TEAL-style thresholding directly on OLMoE expert activations (Phase 0.5); if sparsity holds, design the row-bundled expert layout before claiming the byte win; fallback = apply TEAL only to RAM-resident trunk (pure compute win, zero I/O risk).

## 2. Adaptive top-k (expert top-p) — quality never rigorously measured

**Flaws:** colibri's −30–40% traffic number comes with zero quality evals. Renormalizing fewer experts changes the function (norm_topk_prob interaction). Some tokens may genuinely need tail experts (rare-knowledge/reasoning tokens) — mean quality can hide catastrophic per-token failures.
**Coverage:** we already log full router distributions in our tracer — gate-mass analysis tells us the safe cutoff per layer. Then: perplexity + task evals across cutoffs (0.9/0.8/0.7), per-layer thresholds (deep layers may need more experts), and an entropy-triggered "full-k" escape hatch for uncertain tokens.

## 3. 2-bit codebook quantization (AQLM/QTIP) — proven dense, unknown for MoE at scale

**Flaws:** (a) per-expert calibration is thin — rarely-fired experts see almost no calibration data, so their codebooks may be garbage (though they also matter least — needs measuring, not assuming). (b) Codebook decode is compute-heavy — can erase the T-MAC kernel gains on CPU; QuIP#/QTIP need fast transforms. (c) Quantizing routers/norms poorly poisons everything downstream.
**Coverage:** 2-bit ONLY the cold tail (mixed precision keeps the hot path int4/LUT-friendly — decode cost paid only on rare fetches); routers/norms stay high-precision always; benchmark codebook-decode cost on M2 before committing; publish per-expert quality sensitivity.

## 4. Early exit / layer skip — the KV-cache hole

**Flaws:** skipping a layer leaves no KV for that layer, corrupting attention for all future tokens; published workarounds (copy adjacent-layer KV, lazy recompute) degrade on long generations. Confidence estimators cost compute themselves. MoE interaction unstudied.
**Coverage (novel idea — ours):** don't skip the *layer*, skip the *routed experts*: on easy tokens run attention normally (KV stays intact) and use the shared expert only. DeepSeek-family always has a shared expert, so the fallback path is free, KV-safe, and saves exactly the expensive part (10.2 GB/token routed traffic). "Expert-skip" > layer-skip for streaming MoE. Measure trigger policies (router entropy, gate mass) on OLMoE first. Classic early-exit stays out of scope until this is exhausted.

## 5. Router-lookahead prefetch (71.6% recall) — measured on ONE model

**Flaws:** number comes from colibri on GLM-5.2 only; 28% of fetches still stall the pipeline; recall decays hard beyond 1 layer of lookahead; unknown for fine-grained-expert models with different router temperature.
**Coverage:** measure lookahead recall per architecture in our tracer (add hidden-state hook in Phase 0.5); design for misses: miss-tolerant pipeline (compute proceeds on arrived experts, latecomers merge) + similarity-substitution with a measured quality bound, not a blind swap.

## 6. The 5–15% working set (ESFT) — measured on curated task data

**Flaws:** ESFT used clean single-task datasets. Real sessions mix tasks mid-conversation; topic drift silently invalidates a profile; new tasks cold-start. Aggregated "user working set" may be much bigger than any task working set.
**Coverage:** this is exactly what Phase 0's taskprofile/crossprofile simulations measure on realistic mixed workloads — our own number, not ESFT's. Online decay (recent usage outweighs old), cheap task-switch detection from router-distribution shift (we have the data to build this), and profile stacking (chat+code both warm).

## 7. colibri's lessons — feasibility yes, but two fatal gaps to not repeat

**Flaws:** (a) quality unverified (one confounded 62.5% eval) — a streaming engine that subtly corrupts outputs is worse than useless; (b) no A/B against llama.cpp mmap — the baseline comparison that would justify the added complexity was never run.
**Coverage:** correctness gate from day 1: bit-level logit-equivalence tests vs the reference implementation on every code path (streamed == resident, always), then task evals; and the llama.cpp A/B is a Phase 1 deliverable, same hardware, published.

## 8. THE BIG ONE — composition: multipliers measured in isolation may not multiply

**Flaws:** TEAL's thresholds were calibrated on fp16 activations — 2-bit weight noise shifts activation distributions; adaptive top-k changes the expert mix TEAL/quant calibration saw; expert-skip changes deep-layer statistics. Every paper measured its technique alone, on different models, different evals. The stack could interfere destructively — ×0.2 on paper becoming ×0.5 in practice, or quality collapsing at the intersection.
**Coverage:** this is our core research contribution: the full interaction matrix (each technique alone, pairs, full stack) on OLMoE, then DeepSeek-V2-Lite — quality AND bytes AND FLOPs per cell. Nobody has published this. If we do one rigorous thing, it's this.

## 9. Hardware reality — lab numbers vs consumer devices

**Flaws:** consumer NVMe throttles under sustained max-bandwidth reads (30–60 s bursts, then thermal cliff — our workload is a *sustained* reader); macOS has no O_DIRECT/io_uring (needs F_NOCACHE/posix fadvise paths); phone UFS shares its bus with the OS; page cache fights explicit caching for the same RAM.
**Coverage:** sustained-read (not burst) benchmarks in the install-time speed estimator; per-OS I/O backends abstracted from day 1; thermal-aware pacing (drop prefetch aggressiveness when throttled); measure, on battery, energy per token — T-MAC's 60–70% energy claim needs verifying in our pipeline.

## 10. Network tier — the flaws nobody will have hit yet because nobody built it

**Flaws:** (a) expert-fetch patterns leak what the user is doing (a coding session fetches "coding experts") — a privacy hole unique to our design; (b) CDN latency spikes stall generation unpredictably; (c) integrity/versioning of per-expert shards; (d) offline = missing experts.
**Coverage:** padded/batched fetches so the server sees shard-level not expert-level access; strict hash-pinned shard manifest; degrade-gracefully policy offline (similarity substitution + honest quality banner); prefetch-ahead-of-need so network latency hides behind generation.

## 11. Evaluation culture — the field's own flaw

**Flaws:** every system paper reports tok/s; almost none report quality-at-that-speed. colibri is the norm, not the exception.
**Coverage:** every number we publish is a point on a quality-vs-speed-vs-RAM Pareto curve, with the eval harness in the repo. This alone differentiates the project.

---

**Priority order for coverage experiments (feeds docs/plan.md):**
1. Composition interaction matrix (flaw 8) — start with pairs on OLMoE.
2. Working-set reality on mixed sessions (flaw 6) — Phase 0, already built.
3. Adaptive top-k quality curve (flaw 2) — cheapest big win, data already logged.
4. Expert-skip vs layer-skip (flaw 4) — our novel mechanism, high upside.
5. TEAL-on-MoE transfer (flaw 1).
6. Lookahead recall per architecture (flaw 5).
