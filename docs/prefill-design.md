# Expert-major prefill (E27 design) — written before implementation, 2026-07-20

## Problem (D2)

Prefill streams experts per token, exactly like decode: a 512-token prompt
costs 512 × (misses × 13.25MB) of I/O because every token's top-4 is fetched
against an 8-slot cache that thrashes. Measured prefill: 0.7–2.9 tok/s →
minutes of TTFT on real chat prompts. Phase-0 simulator said expert-major
order is worth 16–599×; that number was never ported to the real engine.

## Physics

With a multi-token ubatch, llama.cpp's `mul_mat_id` already processes the
whole batch per layer — the graph IS expert-major; only our cache is not.
Per layer, the batch needs the UNION of its tokens' experts. For gpt-oss
(128 experts, top-4), a 512-token chunk's union saturates toward all 128:
cost per chunk ≈ n_layers × union × 13.25MB ≈ 36 × 128 × 13.25MB ≈ 61GB
≈ one full read of the expert data per 512 tokens, amortized over 512
tokens instead of paid per token. At ~1.5GB/s that is ~40s + batched
compute ≈ 10+ tok/s prefill vs ~1 today.

The blocker is capacity, not order: unions of 40–128 > n_slots 8, which is
exactly D12's exit.

## Approaches considered

- **A. Dedicated shared prefill pool (CHOSEN)**: one extra slot-tensor set
  per expert-tensor kind, `P` slots, shared by ALL layers and refilled per
  layer as the graph descends (graph executes layers sequentially, so reuse
  is safe on the CPU backend). `P = n_expert` by default → a union can
  never overflow: closes D12 by construction. Decode caches are untouched
  → the warm decode cache SURVIVES prefill (today prefill thrashes it).
  Cost: P × 13.25MB extra RAM while prefilling (1.7GB for gpt-oss at 128),
  MADV_FREE'd back after prefill.
- **B. Chunk to fit existing 8 slots**: guaranteed-fit ubatch is
  slots/top_k = 2 tokens. ~2× best case. Not the prize. Killed.
- **C. Lend decode slot RAM to the active layer**: per-layer slot tensors
  are separate ggml allocations; `mul_mat_id` needs ONE contiguous slot
  tensor, so "lending" means aliasing 36 scattered buffers — impossible
  without reallocating, and it destroys the decode cache. Killed.
- **D. Raise LLMSTREAM_SLOTS for prefill**: slots are baked into every
  layer's tensor at load; 64 slots/layer × 36 = 30.5GB. Killed.
- **E. Sub-batch waves inside one layer**: `mul_mat_id` executes once per
  layer per ubatch; all ids must resolve simultaneously. Killed.

## Design (A)

Fork (~90 lines, all inert unless `LLMSTREAM_PREFILL_SLOTS` set):
1. `llmstream_prefill_slots(n_expert)`: env `LLMSTREAM_PREFILL_SLOTS`;
   0/unset = off; 1 = auto (n_expert); values clamp to ≤ n_expert.
2. `llmstream_pf_tensor[_2d]()`: like slot tensors but registered under
   synthetic names `llmstream_pf.<kind>` (meta type from the layer-0 real
   tensor); created once in the model file, allocated in the same buffer.
3. `build_moe_ffn`: entry block — if streaming on AND pf pool exists AND
   `n_tokens > 1`, swap every non-null expert-tensor argument for its pf
   tensor. All-or-nothing per call (never mix pools in one layer).
   Families without pf creation (qwen v1) are unchanged.
4. Models wired in v1: openai-moe (6 kinds), olmoe (3 kinds).

Driver:
5. Discover pf tensors alongside per-layer extents (same kind order,
   stride checks). File extents are the existing per-layer `lc.ext`.
6. `cb_eval` ids hook, `n_tokens > 1` + pf enabled → prefill branch:
   union → pf slots assigned 0..U−1, all fetched via the worker pool
   (`job.pf` flag targets pf tensors; `pf_outstanding` counter, no decode
   map collisions), wait, rewrite ids. Decode LRU/guard/margin untouched.
7. Margin/skip/agreement hooks early-return on `n_tokens > 1`: residency
   masking is meaningless when the whole union is fetched (and the current
   mask code reads only token 0's row in a batch — it was never
   batch-correct). Prefill under pf is therefore EXACT routing.
8. Union > P (only possible when env forces P < n_expert): explicit exit
   with remedy message (D12's silent boundary becomes a labeled dial).
9. After the last pf call, first single-token call MADV_FREEs the pool.
10. Metal: pf pool is CPU-only in v1; SLOT_DEV=gpu + pf → refuse at start
    (same corrupt-config guard pattern as NGL).
11. Logging BEFORE experiment: `io: pf_calls= pf_experts= pf_fill_s=
    union avg/min/max`, plus existing prefill tok/s line.

## Pre-registered predictions

1. OLMoE gate, resident vs streamed+pf at identical ubatch: bit-exact
   (pure data movement; same kernels, same accumulation order).
2. OLMoE gate at ubatch=1 with pf code present but dormant: banked hash
   unchanged (isolation).
3. gpt-oss 512-token prompt, slots8 m0: prefill ≥ 8 tok/s (vs 0.7–2.9),
   TTFT < 75s (vs ~5–8 min). Speedup 5–15× (unions saturate near 128, so
   the bound is one model-read per 512-token chunk).
4. phys_footprint during prefill rises by ≈ P × 13.25MB (1.7GB at P=128)
   and returns after MADV_FREE.
5. First decode tokens after prefill hit the WARM pre-prefill decode
   cache (today: thrashed cold). Visible as decode hit-rate ≥ battery
   steady-state within the first 64 tokens.

## Risks

- Graph reuse across ubatch shapes must rebuild (llama.cpp builds per
  ubatch; if a cached graph ever served both shapes the swap would be
  stale) — verified by the ubatch>1 gate hashing correctly.
- ggml_add_id on 2D pf biases: same stride mechanism as decode slots.
- Prefill exactness changes quality vs today's margin-masked prefill:
  strictly closer to the resident model; NLL battery (ubatch=1) unaffected.
- OLMoE top-8: ubatch 32 → union ≤ 64 = P. Gate covers the saturated case.
