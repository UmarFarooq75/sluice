# E42 — Speculative decoding for sluice: design study

**Status: DESIGN ONLY. No code, no runs, no downloads. Build only on owner greenlight.**
Date: 2026-07-21. Every number below is either measured in this repo (cited to an
artifact) or derived from those measurements by arithmetic shown in full.

---

## 0. The verdict up front

Speculative decoding is **the only lever we have left that can plausibly reach 10+
tok/s exact on this box**, and the reason is structural: it is the only technique
that attacks the *per-token* miss-byte cost rather than the bandwidth or the cache.
But the study produced a result that contradicts both the obvious intuition and
llama.cpp's own guidance, so it needs stating plainly:

> **Short drafts win. Long drafts lose.** On an MoE with SSD-streamed experts, the
> verification pass must load the *union* of experts routed by all K drafted
> positions. Our measured union grows as ≈ K^0.54, so miss-bytes grow with draft
> length while committed tokens grow much more slowly. Past K≈3 the extra I/O eats
> the win.

llama.cpp's `docs/speculative.md` says the opposite — *"MoEs require long drafts"*,
with a sample config at `n-max 64`. That advice is written for MoEs held in RAM,
where a wider expert union costs nothing but FLOPs. **For us the union IS the cost.**
This disagreement is the single most important thing to test first, and it is cheap
to test, so it becomes rung 1.

**Expected outcome, honestly bounded**: 8.3 tok/s at the acceptance rate the only
available gpt-oss draft actually advertises (a≈0.72, K=2, no batching benefit
assumed) — a **1.35× win** that lands short of 10. Crossing 10 needs *either*
acceptance ≥0.90 *or* a measured GEMM batching benefit of ≥1.6×. Both are plausible;
neither is established. **This study does not claim 10+ tok/s. It claims 10+ is for
the first time inside the range of a mechanism we can build, and it names the two
measurements that decide it.**

---

## 1. Measured baseline — everything downstream rests on these

From `results/e37c/streamed.out` (CLEAN, quiet box, N=64, SLOTS=16, prompt A,
bit-exact hash `7fff2b7b9461da2a`):

| quantity | value | derivation |
|---|---|---|
| decode | 6.14 tok/s | 64 tokens / 10.42 s |
| expert-uses / token | 96 | 6144/64 = 24 layers × top_k 4 ✓ |
| misses / token | 13.375 | 856/64 |
| hit rate | 0.861 | 1 − 856/6144 |
| **miss-bytes / token** | **177 MB** | 13.375 × 13.25 MB |
| stall / token | 100.9 ms | 6.46 s / 64 |
| compute / token | 61.9 ms | (10.42 − 6.46) / 64 |
| stall-path bandwidth | 1756 MB/s | 856 × 13.25 MB / 6.46 s |

**Unit check** (run before trusting any projection): the model
`t = miss_bytes/bw + t_compute` reproduces **6.14 tok/s exactly** at K=1. Any
projection below that fails this check is a bug, not a finding — this check already
caught one order-of-magnitude error in this study's first draft.

Geometry confirmed by parsing our own GGUF header (metadata read only, no model
load): `block_count 24`, `expert_count 32`, `expert_used_count 4`,
`embedding_length 2880`, `tokenizer.ggml.tokens = 201088`, pre-tokenizer `gpt-4o`.

---

## 2. Draft-model candidates, against our RAM reality

**The vocabulary constraint is absolute.** llama.cpp's
`common_speculative_are_compatible` compares *token text for every id* from id 5 up
and hard-throws on mismatch. "Same tokenizer family" does not pass. Our target vocab
is **201088** (verified locally, above).

| candidate | vocab | size | RAM cost in expert slots¹ | verdict |
|---|---|---|---|---|
| **`RedHatAI/gpt-oss-20b-speculator.eagle3`** | 201088 ✓ (inherits target's tokenizer at conversion) | 1.71 GB BF16, ~0.55 GB Q4 | **1.7 slots** (Q4) | **VIABLE — the only real draft model that exists for gpt-oss-20b.** 854M params, 1 layer, EAGLE-3. llama.cpp supports it by name (`--spec-type draft-eagle3`, PR #18039). Card reports 2.75 accepted at k=5 ⇒ a≈0.72 |
| **`--spec-type ngram-mod`** (no model) | n/a — no vocab constraint | ~16 MB | **0.05 slots** | **VIABLE, and free.** llama.cpp's `--spec-default`. Its own docs list "reasoning models when they repeat their thinking in the final answer" as a target application — which is precisely gpt-oss's analysis→final channel behaviour |
| gpt-oss-20b drafting for gpt-oss-120b | 201088 ✓ | — | — | **NOT VIABLE.** 3.6B vs 5.1B active params is a 1.4× ratio (drafts want ≥10×), and 120b does not fit in 16 GB regardless |
| expert-pruned derivatives (`AmanPriyanshu/*`, `leeminwaan/*`, `TOk-Atsuru/*`) | 201088 ✓ | 7.7–10.4 GB | — | **NOT VIABLE.** Expert pruning cuts *footprint*, not *active* params: `num_experts_per_tok` stays 4 and depth stays 24. Same per-token compute as the target ⇒ zero latency advantage. A perfect vocab match that is still useless |
| layer-pruned / depth-reduced gpt-oss | — | — | — | **DOES NOT EXIST.** Every community prune found is expert-pruning. Depth is the thing a draft needs to cut, and nobody has cut it |

¹ 1 expert slot = 13.25 MB × 24 layers = **319 MB**. We run 16.

**The RAM tax is a first-class cost here, not a footnote.** On a box where the expert
cache is the binding constraint, a 0.55 GB draft model is 1.7 fewer expert slots,
which raises the miss rate, which is exactly the quantity spec-dec is trying to
amortize. Sensitivity at K=2, a=0.72:

| miss rate | tok/s |
|---|---|
| 0.139 (measured, 16 slots) | 8.27 |
| +15% | 7.65 |
| +30% | 7.11 |

A 30% miss-rate regression would eat over half the win. **`ngram-mod` costs 0.05
slots and is therefore strictly the right thing to try first** — not because it is
better, but because it is the only option whose cost is provably negligible.

---

## 3. The expert-union math — the heart of this study

### 3.1 What we actually measured

Ordinary decode: one token per pass, 4 experts per layer, 13.375 misses ⇒ 177 MB.

Batched verification of K drafted tokens is one forward pass over K positions, so
per layer it must materialise the **union** of experts routed by all K positions.

We own exactly one measured multi-token union, and it is in the E37c log:

```
io: pf_calls=23 pf_experts=502 union avg=21.8 min=17 max=27 (pool 32 slots)
```

That is a 23-token prefill batch: **21.8 of 32 experts touched per layer, min 17,
max 27.**

### 3.2 Routing is correlated — measurably

If the K tokens routed independently, E[union] = 32(1 − 0.875^K), giving **30.5** at
K=23. We measured **21.8**. Routing locality is real and is worth ~30% of the union.

Anchoring a power law on the two points we own (U(1)=4 by construction, U(23)=21.8
measured):

> **U(K) ≈ 4 · K^0.541**

| K | U(K) fitted | independent model | vs decode cache (16) | vs prefill pool (32) |
|---|---|---|---|---|
| 2 | 5.8 | 7.5 | OK | OK |
| 3 | 7.2 | 10.6 | OK | OK |
| 4 | 8.5 | 13.2 | OK | OK |
| 5 | 9.6 | 15.6 | OK | OK |
| 8 | 12.3 | 21.0 | OK | OK |
| 16 | 17.9 | 28.2 | **OVERFLOW** | OK |
| 23 | 21.8 (measured) | 30.5 | **OVERFLOW** | OK |

**This fit is the weakest link in the study and must be treated as such.** It is two
points, one prompt, and the measured point comes from the *prefill* regime rather
than mid-generation decode. Its exponent is a pre-registered prediction to falsify
(§6, P1), not an established constant.

> **MEASURED CAVEAT — added 2026-07-22 from R0, and it goes the wrong way.**
> R0's leg A8 (ubatch=8, 16 slots, no pool) aborted with the engine's own message:
> `ubatch expert union 18 > n_slots 16 at layer 1`.
> **A layer reached a union of 18 at K=8**, against this fit's predicted **mean of
> 12.3** and its claim that overflow starts near K=16.
>
> *Directional, not a clean refutation of P1*: 18 is a per-layer value at the first
> violating layer, which is an upper-tail observation, while the fit predicts a mean.
> The engine aborts on the first violation, so it never reports the distribution.
> But the direction is unambiguous and unfavourable: **the union grows faster than
> fitted.** Every payoff figure in §3.4 is therefore optimistic, the break-even
> acceptance rates in §3.5 are too generous, and the optimal draft length is pushed
> **shorter than K=2**, not longer. It also means D12's boundary is reachable at
> K=8 rather than K=16 — verification must use the prefill pool from the very
> smallest useful draft, not only for long ones.

### 3.3 Cost per committed token

With draft length K and per-token acceptance probability a, the target commits
`E[committed] = Σ(i=1..K) a^i + 1` tokens per pass (the accepted run, plus the
target's own token at the divergence point).

```
miss-bytes per pass       = 24 layers × U(K) × miss_rate × 13.25 MB
time per pass             = miss-bytes/1756 MB/s  +  K × 61.9 ms / g
tok/s                     = E[committed] / time per pass
```

`g` = **GEMM batching efficiency**: how much cheaper a K-token pass is than K
single-token passes. `g=1` means batching buys nothing; `g=K` is perfect
amortization. **We have never measured g. It is the second decisive unknown.**

### 3.4 The payoff table

Baseline 6.14. G1 band 5–6. E7 compute ceiling 10.47. Owner target 10+.

**a = 0.72** (what the EAGLE-3 card's "2.75 accepted at k=5" implies):

| K | g=1 | g=1.5 | g=2 | g=3 | committed |
|---|---|---|---|---|---|
| **2** | **8.27** | 9.76 | 10.72 | 11.90 | 2.24 |
| 3 | 7.09 | 8.52 | 9.47 | 10.67 | 2.61 |
| 4 | 6.25 | 7.61 | 8.54 | 9.73 | 2.88 |
| 5 | 5.58 | 6.87 | 7.77 | 8.93 | 3.07 |
| 8 | 4.20 | 5.28 | 6.06 | 7.12 | 3.39 |

**a = 0.90:**

| K | g=1 | g=1.5 | g=2 | g=3 | committed |
|---|---|---|---|---|---|
| **2** | **10.02** | 11.82 | 12.98 | 14.41 | 2.71 |
| 3 | 9.33 | 11.22 | 12.48 | 14.05 | 3.44 |
| 4 | 8.88 | 10.82 | 12.14 | 13.83 | 4.10 |
| 8 | 7.60 | 9.56 | 10.97 | 12.88 | 6.13 |

Read the columns, not the rows: **tok/s falls monotonically with K at every
acceptance rate.** K=2 is optimal everywhere in the plausible region.

### 3.5 Break-even, and what 10 tok/s costs

Minimum acceptance for spec-dec to merely **not lose** (g=1):

| K | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| need a ≥ | **0.455** | 0.626 | 0.711 | 0.763 | 0.797 | 0.840 |

K=2 tolerates a coin-flip draft. K=8 needs a draft that is right 84% of the time
just to break even — which is why long drafts are a trap here.

GEMM efficiency `g` required to reach exactly 10.0 tok/s:

| K | a=0.72 | a=0.8 | a=0.9 | a=0.95 |
|---|---|---|---|---|
| 2 | 1.61 | 1.27 | **1.00** | 0.89 |
| 3 | 2.37 | 1.65 | 1.15 | 0.99 |
| 4 | 3.33 | 2.02 | 1.26 | 1.04 |
| 8 | 17.80 | 4.05 | 1.64 | 1.15 |

**At a≥0.90 and K=2, 10 tok/s falls out with no batching benefit at all** (g=1.00).
At the advertised a=0.72 it needs g≥1.61, i.e. a 2-token batched pass must cost less
than 1.24× a single-token pass. Plausible on CPU (GEMV→GEMM), unmeasured.

---

## 4. Where verification lands: prefill pool vs decode cache

This is not a design preference. **It is forced by a defect already in our ledger.**

> **D12**: *"Prefill with n_ubatch>1 + per-layer union > slots has no path
> (assign_slot exhausts victims → exit(1))."*

Batched verification **is** the n_ubatch>1 case. Every position in the pass must have
its experts simultaneously resident for that layer's compute; `assign_slot`
evicts-when-full (the same behaviour that caused the E34 warmpack churn bug), so if
`U(K) > slots` an expert seated for position 1 is evicted before position 5 reads it.

Two consequences:

1. **Verification must route through the prefill pool, not the decode cache.** The
   pool already exists and is already correct for this exact case: 32 slots, 424 MB,
   shared across layers, and **type-variant aware** — D14 taught us the hard way that
   a single pool typed from layer 0's metadata corrupts Q4_K/Q6_K mixed files, and
   the fix (one pool per (kind, quant-type) variant) is what makes a multi-token
   batch safe. `results/e37c/streamed.out` confirms it is live: *"prefill pool 32
   slots x 6 type-variants (424 MB)"*.
2. **32 pool slots cover U(K) up to K≈23**, far beyond the K=2–4 this study
   recommends. The pool is not the constraint. **The decode cache is** — it would
   overflow at K≥16, which is one more independent reason the llama.cpp "long drafts
   for MoE" advice is wrong for us.

**Open design question for a rung, not for this doc**: whether the accepted prefix
should be *migrated* from the pool into the decode cache after commit, so the next
pass starts warm. Migration costs a copy; not migrating costs re-fetches. Unmeasured
either way, and it should not be guessed at.

---

## 5. Bit-exactness — precise enough to become gate 1

### 5.1 The algorithmic half: exact by construction

llama.cpp does **not** use Leviathan-style rejection sampling. It does exact-match
verification against the target's own sampler (`common/sampling.cpp`):

```cpp
for (; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) { break; }   // disagreement ends the run
}
```

Every emitted token is sampled by the **target**. The draft only gates *how many*
target-sampled tokens commit per pass. So:

> **Claim A (algorithmic).** For any sampler configuration, the committed token
> sequence under speculative decoding is identical to non-speculative decoding, for
> any draft quality. A bad draft costs speed, never correctness.

Under our greedy default (`LLMSTREAM_TEMP=0`, argmax) this reduces to: token p is
`argmax(logits_target(p | prefix))` in both paths.

### 5.2 The numerical half: this is where it will break

Claim A holds **iff the target's logits are identical between the two paths.** They
are computed differently:

- non-speculative: K separate 1-token forward passes
- speculative: one K-token batched forward pass

Different batch shape ⇒ different GEMM blocking ⇒ different FP accumulation order ⇒
logits can differ in the last bits ⇒ **argmax can flip on a near-tie.**

> **Claim B (numerical).** Speculative decoding is bit-exact iff
> `argmax` is invariant to batch shape at every committed position. This is a claim
> about GEMM determinism, **not** about the speculative algorithm.

**This is the third feature in a row to depend on that same property**, and that is
now the most important structural fact in this document:

| feature | depends on | status |
|---|---|---|
| E39 KV persist | restored prefix decoded in a different shape | **FAILED its hash gate**, cause never established |
| E41b KV canon | canonical suffix decoded in a different shape | armed, **gate has never run** |
| **E42 spec-dec** | K-token verification vs 1-token decode | this document |

E39 already failed exactly this way. **Recommendation: the batch-shape determinism
root-cause investigation should be resolved BEFORE E42 is built, not after.** If
argmax is not batch-shape-invariant on this backend, E42 cannot be bit-exact by
construction and the whole feature must be re-scoped as a Balanced/Fast-tier
optimisation — which is a different product decision, and one the owner should make
with that fact in hand rather than after a build.

### 5.3 Gate 1, stated for pre-registration

> **E42 GATE 1 (faithfulness).** With `LLMSTREAM_TEMP=0` and speculation ON, a run
> must produce **(a)** a byte-identical output token sequence, **and (b)** an
> identical `logits_hash`, versus the same prompt with speculation OFF.
> (a) failing means the *implementation* is wrong. (b) failing with (a) passing means
> the *backend* is not batch-shape-deterministic — the E39 disease — and is reported
> as such, not rationalised.

**Instrumentation prerequisite** (a real code change, and a rung of its own): our
`logits_hash` currently accumulates the logits of every generated step in the decode
loop. Under speculation, a pass produces logits for K positions of which only some
commit. The hash must be redefined over **committed positions only, in commit order**,
or the two paths cannot be compared even when they agree. Getting this wrong would
manufacture a false gate failure.

---

## 6. Pre-registered predictions

Written before any build, falsifiable, with the falsifier stated.

**P1 — union growth.** Mid-generation, per-layer expert union for a K-token batch
follows U(K) ≈ 4·K^0.54 within ±20% for K∈{2,4,8}.
*Falsifier*: any K outside ±20%. **If U grows faster (toward the independent model),
the whole approach weakens and K must shrink further; if slower, everything improves.**
This is the cheapest high-information measurement in the plan.

**P2 — draft length is inverted vs upstream advice.** tok/s is **monotonically
decreasing** in K over K∈{2,3,4,6,8} at fixed acceptance.
*Falsifier*: any interior maximum. **This directly contradicts llama.cpp's
`docs/speculative.md` ("MoEs require long drafts", sample `n-max 64`) and if we are
wrong, the union model in §3 is wrong.**

**P3 — ngram acceptance on harmony reasoning.** `ngram-mod` achieves a ≥ 0.455 at
K=2 (the break-even from §3.5) on a prompt whose final channel restates the analysis
channel.
*Falsifier*: a < 0.455 ⇒ ngram-mod cannot pay for itself even at zero RAM cost.

**P4 — EAGLE-3 acceptance transfers.** The published 2.75-at-k=5 (a≈0.72) holds
within ±0.1 on our prompts at our quantization.
*Falsifier*: a < 0.62 ⇒ below K=3 break-even; the 0.55 GB RAM tax is then unpayable.

**P5 — GEMM batching efficiency.** g ≥ 1.3 for K=2 on this CPU backend.
*Falsifier*: g < 1.1 ⇒ batching buys nothing, and only a ≥ 0.90 can reach 10 tok/s.

**P6 — the RAM tax is real and bounded.** Loading a 0.55 GB draft alongside 16 slots
raises the decode miss rate by ≤ 15% (i.e. ≤ 0.160).
*Falsifier*: > 30% ⇒ the draft must be quantized further or abandoned for ngram-mod.

**P7 — gate 1(b) will be the hard one.** (a) passes and (b) fails.
*Stated as a prediction on purpose*: if (b) passes, the E39 failure gets a strong new
constraint — batch-shape variance would then be established as *not* the universal
culprit, and E39's real cause is elsewhere.

---

## 7. Build plan — single-directive rungs

Each rung is independently greenlightable, independently abortable, and produces an
artifact. **Rungs 0–2 involve no engine changes at all** — they are measurements on
the code we already have, and they can kill the whole idea cheaply before we write a
line of it. That ordering is deliberate.

| rung | what | changes engine? | kills the idea if |
|---|---|---|---|
| **R0** | **Batch-shape determinism probe.** Same prompt, same prefix, decode the next token as part of a K-token batch vs a 1-token batch; compare logits bitwise. Answers §5.2 for E39, E41b **and** E42 at once | no (uses existing NLL/prefill paths) | argmax flips ⇒ **bit-exact spec-dec is impossible**; re-scope to Balanced tier |
| **R1** | **Union measurement (P1, P2) — now a NAMED, MANDATORY leg.** Instrument the existing prefill-pool union counter to report per-K unions mid-generation. We already print `union avg/min/max` — this extends where it is emitted. **Log the union whenever it is measurable, not only when it overflows**: R0 learned U(8)≥18 only because the engine crashed, which is a terrible way to acquire a number and gives an upper-tail sample instead of a distribution. Every future spec-dec rung carries this leg | telemetry only, gated | U(K) grows near-independently ⇒ payoff collapses |
| **R2** | **GEMM efficiency probe (P5).** Time a K-token batched forward pass vs K single-token passes at fixed prefix. Pure timing | no | g < 1.1 and P4 says a<0.9 ⇒ cannot reach 10 |
| **R3** | **Rebase decision.** Our fork is pinned at `b10064`; the `--spec-type` overhaul and EAGLE-3 support are newer (b10075+). Assess whether `patches/llmstream.patch` applies to a base that has speculation, or whether we implement verification inside our own decode loop | no (assessment) | patch conflicts are unbounded ⇒ implement in-house instead |
| **R4** | **Hash redefinition** (§5.3): `logits_hash` over committed positions in commit order. Required before any gate can run | yes, small | — |
| **R5** | **ngram-mod leg (P3).** Cheapest real trial: no download, 16 MB, no vocab risk | integration only | a < 0.455 |
| **R6** | **EAGLE-3 leg (P4, P6).** Convert + quantize the RedHat draft, measure acceptance and the RAM tax | integration | a < 0.62 or miss rate +30% |
| **R7** | **Gate 1 + the K sweep.** Full faithfulness gate, K∈{2,3,4,6,8}, against the E37c baseline | — | gate 1(a) fails |

**Recommended first directive: R0.** It is the smallest, it is shared with two
already-open threads (E39's unexplained hash failure and E41b's pending gate), and a
negative result there changes what E42 *is* before any of it is built. Doing R0 last
would be the expensive mistake.

**Explicitly out of scope** until greenlit: any model download (R6 needs one — 1.71 GB
— and must be authorised separately), any fork rebase (R3 is an assessment, not a
rebase), and any change to the default path. Speculation would ship gated and off,
like every other experimental feature here.

---

## 8. Prior art

- **SpecMoEOff** (arXiv 2508.21706) — "the first MoE offloading system that leverages
  speculative decoding to increase hardware utilization", up to 2.5× decode
  throughput over SOTA MoE offloading. Closest published work to this architecture.
- **MoE-SpeQ** (arXiv 2511.14102) — uses the *draft* to predict which experts the
  target will need and prefetch them. **Directly relevant to us**: we already have a
  router-lookahead prefetcher (`LLMSTREAM_PREFETCH`); a draft model is a strictly
  better lookahead signal than the one we use now. Not in this plan; a natural
  follow-on if E42 lands.
- **MoE-Spec** (arXiv 2602.16052) and *Cost-Aware Speculative Decoding for MoE*
  (arXiv 2607.12696) — both formalise the exact tension measured in §3: batched
  verification touches more experts than single-token decode, eroding the speedup.
  **Independent confirmation that §3.4's inverted-K result is a real MoE effect and
  not an artefact of our fit.**
- **Self-Speculative Decoding for On-device MoE** (ACM WWW 2026) — 3.72×, "nearly
  lossless". Note "nearly": their acceptance is quality-lossy in a way our bit-exact
  contract does not permit.
- **Snowflake Arctic Inference** — trained a dedicated 1.76B LSTM speculator for
  gpt-oss rather than use 20b as a draft; 1.6× on 20b, 44–50% acceptance. Datacenter
  GPU, vLLM — **does not transfer to a bandwidth-bound M2 Air** and is cited only as
  evidence that a purpose-built small draft beats a scaled-down sibling.

---

## 9. What this study did not establish

Stated so nothing here gets quoted as more than it is:

- **U(K) is a two-point fit** from one prompt, and its measured anchor is a prefill
  batch, not mid-generation decode. P1 exists to test it. **It is already known to be
  optimistic**: R0 observed a per-layer union of 18 at K=8 where the fit predicts a
  mean of 12.3 (see §3.2). Treat every payoff number here as an upper bound.
- **`g` has never been measured** on this backend. Every table with a `g` column is a
  projection across a parameter we do not know.
- **Acceptance rates are from model cards and papers, not from our hardware, our
  quantization, or our prompts.** P3/P4 exist to test them.
- **The miss rate is held constant at the measured 0.139 across all K.** In reality a
  larger union changes cache dynamics, and §3.4 does not model that. The direction is
  unfavourable, so the tables are optimistic.
- **No measurement in this document was taken during this study.** Every figure is
  read from an existing artifact or derived from it. No model process was launched.
