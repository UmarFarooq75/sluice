# Red-teaming our own Phase 0 analysis

Deep pass over our experiment design (2026-07-17, while first traces run): what could make our numbers misleading, what we're not measuring, and what to fix before drawing conclusions. Items marked **[FIX NOW]** were corrected in `src/simulate.py` before the first analysis ran.

## A. Bugs / design flaws caught in our own simulator

1. **[FIX NOW] Prefill tokens pollute the cache simulations.** Prefill routes a whole batch at once (and will use expert-major loading in the real engine anyway); mixing it into sequential decode-cache sims inflates apparent reuse. Fix: cache sims run on decode-phase tokens only; prefill rows feed only the union/prefill analyses.
2. **[FIX NOW] Half-split leaks: profile fit on the "first half of the token stream" shares a prompt with the eval half.** A profile learned partly from prompt 3's first tokens, evaluated on prompt 3's remaining tokens, overstates transfer. Fix: leave-one-prompt-out (profile from prompts 1..n−1, evaluate on held-out prompt n, rotate).
3. **[FIX NOW] Binary hit rates hide quality-relevant misses.** Missing a gate-0.40 expert ≠ missing a gate-0.05 expert. Fix: report gate-mass-weighted hit rate alongside binary.
4. **[FIX NOW] Per-layer averaging hides the design signal.** Literature says locality concentrates in deep layers; layer 0 is near-random. If true, cache capacity should be allocated *unevenly* per layer. Fix: per-layer hit-rate breakdown + a greedy optimal-allocation analysis (given a global budget of cached experts, how to split across layers) — directly actionable for the engine, and nobody publishes this.
5. **[FIX NOW] Per-layer-only caches never tested against a shared pool.** A global cache pool (any layer's experts compete for slots) might beat fixed per-layer splits. Fix: simulate both.
6. **[FIX NOW] No saturation check on working sets.** With only ~1–2k decode tokens per task, unique-expert counts may still be climbing — we'd underestimate working sets. Fix: expert-discovery curves (unique experts vs tokens seen); only trust working-set sizes if the curve plateaus. If it doesn't plateau, collect more prompts before concluding anything.
7. **No confidence intervals.** Fix (cheap): bootstrap over prompts, report spread, state n everywhere.

## B. Validity threats in the traces themselves

8. **Scale gap — the biggest caveat on everything Phase 0 produces.** OLMoE routes 8-of-64 (12.5% of experts/layer); DeepSeek-V3 routes 8-of-256 (3.1%). Raw hit rates on OLMoE are structurally ~4× more optimistic than frontier models. Mitigation: report everything **relative to the random baseline** (k/N), not raw; treat OLMoE numbers as upper-bound shape, not absolute truth; repeat on DeepSeek-V2-Lite (fine-grained + shared experts) before believing any number applies to the 671B target.
9. **We trace the model's own generations, not human text.** Decode-token routing may differ from routing over human-written text (distribution shift). Mitigation: add a forced-decode workload (teacher-forcing over real code files / articles, no generation) in tracer v2.
10. **Quantization-specific routing.** Traces come from the 4-bit model; borderline top-k picks flip vs bf16. Working sets may be partially quantization artifacts. Mitigation: flag on all findings; spot-check a few prompts on bf16 if RAM allows (needs ~14 GB, machine otherwise idle).
11. **Clean task separation is unrealistic.** Real sessions interleave tasks; profile-switch cost and drift detection are untested by separated workloads. Mitigation: add a mixed/interleaved session trace; measure detection latency of router-distribution shift (we log full distributions — a KL-divergence change detector is buildable from existing data).
12. **Greedy decoding only.** Real use samples at temperature; token distribution differs, so routing does too. Mitigation: rerun one workload sampled; compare working sets (cheap sanity check, not a rerun of everything).
13. **Context-length hazard in the prefill sweep.** OLMoE's context is 4,096 tokens; the 3,072-word document tokenizes near/over that. Check the prefill_3072w trace completed sanely (no truncation surprises) before using it.
14. **Multi-turn reality.** Chat re-prefills conversation history every turn unless KV persists — token traffic in real chat is prefill-dominated. Our engine design already answers this (persistent KV + expert-major prefill), but the analysis should quantify it: tokens-per-response vs tokens-re-prefilled in a realistic 10-turn session.

## C. Measurements we're missing entirely (add to roadmap)

15. **The trace→quality bridge (most important gap).** Hit rates say nothing about output quality under restriction. Add the **restricted-routing eval**: rerun generation with the router masked to (a) top-p expert subsets, (b) a fixed working-set/pinned mask, (c) reduced k on "easy" tokens — and measure output quality (reference perplexity + small task set like GSM8K-subset) against unrestricted. This converts every cache/policy simulation into a *quality-at-bytes* claim, which is the currency of the whole project. (Phase 0.75.)
16. **Router-lookahead recall on OLMoE.** Needs hidden-state capture (≈65 KB/token — feasible for one workload). Tests whether colibri's 71.6% next-layer recall generalizes; feeds the prefetch design.
17. **Expert-skip can't be tested on OLMoE** — it has *no shared expert* (no fallback path). The reduced-k-on-easy-tokens variant CAN be tested now (via gate-mass triggers from existing traces + restricted-routing eval). True expert-skip waits for DeepSeek-V2-Lite. Flagged so we don't accidentally "conclude" expert-skip works from a model that can't express it.
18. **Easy-token statistics.** From existing traces: what fraction of tokens have top-1 gate mass > 0.5 / router entropy below threshold? That's the addressable market for reduced-k/expert-skip — computable today, zero new runs.
19. **Working-set overlap ACROSS quantization/scale later** — when DS-V2-Lite lands, check whether task-profiles transfer across models at all (they won't, but confirming kills a tempting-but-wrong "universal profile" idea cheaply).

## D. Honest-reporting rules adopted

- Every hit-rate figure names its baseline (random = k/N) and its phase (decode-only).
- Every working-set size ships with its discovery-curve status (plateaued or still climbing).
- OLMoE findings are labeled "shape, not magnitude" until replicated on DS-V2-Lite.
- Negative results get written up with the same care as positive ones — they prune the engine design space, which is the point of Phase 0.
