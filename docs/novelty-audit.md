# Novelty audit — adversarial prior-art review of our claims (2026-07-18)

Eight parallel researchers, each instructed to *disprove* the novelty of one claimed finding. Full evidence with links in the workflow output; verdict table and consequences here. This document supersedes any "nobody has done this" phrasing elsewhere in the repo.

## Verdict table

| Our claim | Verdict | Closest prior art | What actually remains ours |
|---|---|---|---|
| Margin-gated cache-aware routing (f.11) | **ALREADY_DONE** | arXiv 2412.00099 (Cache-Prior boost ≈ mathematically our margin gate, same headline curve); SMoE 2508.18983 (explicit fetch-vs-substitute score threshold + sweep); BuddyMoE; HOBBIT | Only parameterization details and our NLL+exact-match measurement on our models |
| Co-activation disk layout (f.27) | **ALREADY_DONE** | **mbolt (2026-07-13 — four days before us): identical trace-driven GGUF re-layout, better results (1.83×; interleave 2.23×)**; LLM-in-a-flash bundling; ExFlow affinity | Nothing on mechanism. Adopt mbolt's interleave idea; possibly contribute upstream |
| Cache-restricted self-speculation (f.25/26) | **ALREADY_DONE** | SS-MoE (WWW 2026) states it verbatim; ELMoE-3D; SpecMoE | Nothing — and our own measurement showed it marginal anyway; the ceiling analysis (1/sublinearity bound) may be a small contribution |
| Expert-major prefill (f.7) | PARTIAL | FlexGen zig-zag (dense/disk), MoE-Gen module batching (RAM), Klotski expert-aware pipeline | The single-user SSD-prefill framing + prompt-length-independence measurement (O(experts) demonstrated 16–599×); the inversion itself is established |
| Token→expert static table (f.29/31) | PARTIAL | **OpenMoE (ICML 2024) named the phenomenon "Context-Independent Specialization"**; Sem-MoE builds literally a per-token table on DS-V2-Lite; ExpertFlow does full-depth learned prediction | Our held-out modal-set metric, the cross-architecture comparison, and using a *counting* table (training-free) specifically as a t=0 prefetcher in a streaming engine |
| Agent-swarm sublinearity (f.28) | PARTIAL | XShare & ELDR measure sublinear batch-unions; MoE-Gen/MoE-Lightning build throughput on batch amortization | The agent-fleet-on-cheap-hardware framing and cross-*sequence* (not batch) measurement in a streaming context |
| Expert-skip via shared experts (f.22) | PARTIAL | Not-All-Experts-Are-Equal, MoDES, AdapMoE, HOBBIT (all skip a *subset*, keep ≥ top-1) | The all-or-nothing rule (skip ALL routed experts, shared-only FFN) keyed on total gate mass, with the inverse control. Narrow but real |
| Negatives: internally-dense experts (f.30) | **PARTIAL — and (a) CONTRADICTED** | arXiv 2605.08575 & 2509.00454 & MoNE report fine-grained experts ARE internally sparse (up to 90% *dynamic* sparsity) on the same model families | See correction below |
| Negatives: compression/dedup dead (f.24) | PARTIAL | Consistent with general quantized-weight entropy knowledge; not previously published as a streaming-specific negative | The specific measurements |

## Correction to finding 30 (issued now)

Our claim "experts are internally dense" is **wrong as stated**. The literature shows high *per-token dynamic* sparsity inside fine-grained experts (different few neurons fire per token). Our measurement showed the *aggregate/static* importance is near-uniform and temporally unstable. **Both are consistent**: per-token sparsity exists (exploitable for *compute* skipping when weights are already resident — their result), but the sparse set changes token-to-token, so there is no stable subset to *store or fetch* preferentially (our result: no fetch-granularity dial for *streaming*). Finding 30 is hereby narrowed to: *"no temporally-stable within-expert structure exploitable for storage/streaming; dynamic compute-level sparsity per literature remains open for the engine's compute path."*

## Strategic consequences (the honest reading)

1. **The mechanism space is commoditized.** 2024–2026 produced papers for nearly every trick we derived independently. Two days of our first-principles work reproduced ~6 published results — which *validates our methodology* and *devalues mechanism novelty as a moat*.
2. **What does not exist anywhere: the integration.** mbolt is a layout tool. colibri is one model. SS-MoE, SMoE, Klotski, Sem-MoE, MoDES are papers with prototypes on different stacks, never composed, with no composition-effect measurements (our finding 20 — prefetch↔margin substitution — is exactly the kind of interaction nobody's isolated systems can see). **No installable engine ships table-prefetch + margin routing + expert-major prefill + co-activation layout + learned working sets + correctness gates for any GGUF MoE.** That gap is intact — it is the original universality+rigor positioning, now confirmed by exhaustive search.
3. **Our verification culture is the differentiator**: bit-consistency gates, inverse controls, held-out splits, published negatives, and now this audit itself. The literature we surveyed largely lacks it.
4. **Actions**: cite all prior art prominently (README + findings amended); consider contributing to/benchmarking against mbolt rather than competing on layout; drop "novel" from findings 11/25/27/29 language; keep 20/22-control/audit-methodology as our genuine contributions until someone shows otherwise.
