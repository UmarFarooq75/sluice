# Dense-model strategy — synthesis of the 2026-07-18 three-agent research sweep

Sources: [research/dense-flash-streaming.md](research/dense-flash-streaming.md) ·
[research/activation-sparsity.md](research/activation-sparsity.md) ·
[research/batch-throughput-offload.md](research/batch-throughput-offload.md)

The question Umar asked: *can we serve dense models too — reduce resources without
hurting quality or speed — and convert extra resources into advantage over
Ollama/LM Studio/colibri?* Verdicts, ranked by evidence:

## GO — Fleet mode: batch-amortized exact inference (dense AND MoE)

**The one dense offering with ZERO quality cost.** Stream weights in layer order
(dense) or expert-major (MoE), amortize each pass over a batch of sequences.
FlexGen proved 84%-of-bound on GPUs and is archived; llama.cpp measurably does
NOT amortize (8 slots -> 2.08 tok/s aggregate vs 2.13 single on a 212GB model);
nobody serves this niche CPU-only. The bound on a laptop: dense 40GB @ 2.4GB/s,
batch 64 -> ~3.8 tok/s aggregate = ~110k tokens per night. Exact quality.
- Fits the agent-fleet vision directly (our sublinearity finding: 5 agents ≈ disk
  cost of 3 makes MoE fleet mode even better than the dense bound suggests).
- Composes with speculative verification (SpecExec: ~20 accepted tokens per
  streamed pass, GPU-proven, unshipped on SSD/CPU). NOTE: this REOPENS
  speculation for us — phase 0 killed *cache-restricted self-spec* for
  interactive decode; *draft-model spec amortizing streamed passes in batch
  mode* is different math and is proven.
- Risk named by the research: llama.cpp could absorb the niche with one zig-zag
  scheduler. Speed-to-ship matters.
- Roadmap: Phase 3.5 "fleet mode" — after M2/M3. Design: zig-zag (layer x
  micro-batch) scheduler + double-buffered sequential reads in our existing
  I/O pool; MoE expert-major variant first (it is our home turf), dense second.

## PROTOTYPE — Session working sets for dense (GRIFFIN-class, training-free)

Pick ~50% of FF neurons once per prompt (sequence-level "flocking"), do ONE bulk
sequential fetch, decode entirely from RAM with zero per-token I/O. The only
training-free dense scheme whose I/O pattern storage likes. Quality ≈ TEAL@50%
— real but bounded; MUST be evaluated on reasoning (Sirius: PPL hides GSM8K
collapse), not perplexity. Ships as an explicit dial ("session mode"), never a
default. Cheap prototype on a Qwen dense model; measure exact-match deltas.
This is the dense mirror of our per-task expert working sets (finding 5/16).

## NO-GO — Per-token neuron streaming for stock SiLU dense models

Training-free per-token sparsity saturates at 40-65% quality-safe; streaming
needs ~97% of touched bytes cached-or-skipped. Result: 0.2-0.5 tok/s SSD-bound
on a 70B, with reasoning damage at the sparsity levels that would fix it. The
proven alternative (ReLUfication/dReLU at 87-90% + parity) requires model
conversion training we don't do, and no parity-grade converted checkpoints
exist for Llama-3/Qwen-class. WATCH: if the community ships ProSparse-style
checkpoints for modern models, LLM-in-a-flash mechanics (windowing, bundling,
predictors) slot into our engine at neuron granularity — the locality physics
(adjacent-token overlap >90%, hot 20-40% covers 80%, cross-layer >90%
predictable) mirrors our MoE findings exactly.

## WATCH — MoEfication (dense -> MoE conversion at install)

Conversion cost collapsed in 2025-26 (200B tokens -> 4M -> training-free
ExpertWeaver; DOT-MoE ~90% of dense at 50% active). Converts a dense model into
OUR native fetch unit (contiguous expert blocks instead of scattered neuron
rows). Not parity yet — hold until a conversion shows <2% reasoning delta, then
it becomes "install-time dense support" in our existing engine unchanged.

## Strategic confirmations from the sweep

1. **The core MoE bet is re-confirmed**: PowerInfer's own team pivoted from
   dense streaming to natively-sparse MoE; the frontier ships MoE; batch
   amortization and expert streaming both work best there.
2. **Integration remains the moat**: no maintained GGUF/llama.cpp-native dense
   streaming exists; no GGUF metadata standard for sparsity/predictors; every
   competitor is paper-only, stale, closed, or single-model.
3. **Resource elasticity is a differentiator worth naming in the vision**:
   speed must be a smooth monotone function of every resource (RAM cache dial —
   measured; SSD bandwidth — in the formula; GPU -> VRAM tier-0 — Phase 4;
   network — coldest tier). Ollama/LM Studio are binary fits-or-cliff; colibri
   is fixed-target.
4. **Honesty gap to exploit**: system papers in this space ship speed numbers
   with no accuracy evals (PowerInfer-2 reports none). Our bit-exact gate +
   reasoning-eval-per-dial is a publishable differentiator.
5. New adjacent prior art to track in novelty-audit: FlashMoE (2601.17063,
   learned cache replacement for SSD experts), Lever (2605.16786, spec-decode
   over flash), WiSP (2606.21868, bandwidth-not-prediction bottleneck).
