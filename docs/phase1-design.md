# Phase 1 — Engine core design

Goal: an installable engine that runs any GGUF MoE larger than RAM at the best speed the storage physics allows, with bit-verified correctness. Positioning per the novelty audit: **we are the integrator** — compose the field's mechanisms (cited) with our measured composition laws; novelty lives in the integration, the coordination policies, and the verification culture.

## Architecture (informed by findings 1–31 + audit)

```
┌────────────────────────────────────────────────────────┐
│ Frontend: GGUF loader + architecture adapters          │
│   (OLMoE test-adapter → Qwen3-MoE → DeepSeek family)   │
├────────────────────────────────────────────────────────┤
│ Compute: ggml-based forward (dense trunk resident)     │
│   correctness gate: streamed logits ≡ resident logits  │
├────────────────────────────────────────────────────────┤
│ Expert Streaming Core (the product)                    │
│  • Expert store: per-expert extents, co-activation +   │
│    interleave layout (credit: mbolt; benchmark both)   │
│  • Hybrid cache: pinned learned base + LRU top-up      │
│    (finding 3), per-layer (9), uniform alloc (14)      │
│  • Margin router (11; variants per SMoE/CachePrior     │
│    (audit), calibrated per architecture per finding 8) │
│  • 3-stage prefetch: token-table (29/31) → gate        │
│    lookahead (18) → margin fallback; coordinated per   │
│    the substitution law (20): prefetch only margin-    │
│    filtered predictions, only when I/O queue idle      │
│  • Expert-major prefill (7; credit Klotski/MoE-Gen)    │
│  • Working sets: persistent per-task profiles (5,16),  │
│    task-switch detector (router KL, 6-token latency)   │
├────────────────────────────────────────────────────────┤
│ I/O backend: pread+F_NOCACHE (macOS) / io_uring        │
│   (Linux), 4-8 worker threads, QD-aware               │
├────────────────────────────────────────────────────────┤
│ Honesty layer: install-time disk benchmark → predicted │
│   tok/s before download; quality dials explicit        │
└────────────────────────────────────────────────────────┘
```

## Milestones

- **M0 (now): compiled streaming core benchmark** — C++ expert fetcher replaying real routing traces vs the Python PoC's I/O path. Success: ≥90% of measured SSD bandwidth at the same trace (Python achieved 50–65%).
- **M1: OLMoE end-to-end in C++/ggml** — dense trunk resident, experts streamed, logit-equivalence vs MLX reference. Success: ≥2× the Python PoC at equal cache; bit-gate green.
- **M2: the brain** — 3-stage prefetch + margin router + hybrid cache integrated; A/B vs llama.cpp mmap same hardware (the comparison colibri never published).
- **M3: DeepSeek-family adapter** (covers GLM/Kimi targets) + Qwen3-MoE.
- **M4: ship** — installer with disk benchmark + honest speed estimate; docs; the Phase 0 findings as launch evidence.

## M1 integration plan (llama.cpp spike result, day 2)

Spike read: `vendor/llama.cpp` (shallow clone, current master — has DeepSeek-V4 arch,
merged gate_up path, per-expert scales). Upstream policy note: llama.cpp does not accept
AI-generated PRs; **private forks are explicitly exempt** (AGENTS.md). Our integration is
a fork/patch-set in our repo. We never PR upstream, never open issues/comments there.

### The three facts the plan stands on

1. **Public mid-graph hook exists.** `llama_context_params.cb_eval`
   (include/llama.h:366) installs a `ggml_backend_sched_eval_callback`
   (ggml/include/ggml-backend.h:314): ask-phase selects nodes to observe by name;
   observe-phase runs with the node computed and data readable/writable (CPU backend).
   The scheduler splits the graph at observed nodes — same mechanism imatrix uses.
2. **Experts are contiguous extents in the GGUF file.** The loader records each tensor's
   absolute file offset (`llama_tensor_weight::offs`, src/llama-model-loader.h:35).
   Expert tensors are created `{n_embd, n_ff, n_expert}` (expert = outermost dim), so
   expert `e` of `ffn_up_exps` layer `l` lives at `offs + e * nb[2]`, one contiguous
   read. **M1 streams directly from the GGUF — no repacked store needed.** (The
   co-activation layout store returns in M2 as an optimization, not a requirement.)
3. **The graph consumes expert ids twice.** In `build_moe_ffn` (src/llama-graph.cpp:1799):
   `ggml_argsort_top_k` → ids feed `ggml_get_rows(probs, ids)` for gate weights, then the
   same ids index `ggml_mul_mat_id(up/gate/down_exps, …)`. Streaming must therefore split
   the ids: **original expert ids** for the weight gather, **slot-remapped ids** for the
   expert matmuls.

### Patch set (small, localized)

- **P1 — slot-backed expert tensors** (`llama-model.cpp` create_tensor path, flag-guarded):
  in streaming mode create `ffn_{up,gate,down}_exps` as `{n_embd, n_ff, n_slots}` in a
  malloc'd CPU buffer (not mmap), skip their GGUF data load, keep their file `offs` in a
  side table. Everything else (trunk, router, norms, shared experts) loads/mmaps as stock.
- **P2 — dual id tensor** (`build_moe_ffn`, flag-guarded): add
  `ids_slots = ggml_cpy(ids)` named `ffn_moe_topk_slots`; `get_rows` keeps original
  `ids`; the three `mul_mat_id` calls take `ids_slots`.
- **P3 — the streaming brain** (new file, our code): `cb_eval` observes each layer's
  `ffn_moe_topk_slots` post-compute → per-layer LRU over `n_slots` (M0's structure) →
  missing experts fetched from the GGUF extents by the M0 thread pool
  (pread+F_NOCACHE, 4–6 workers) into their assigned slots → ids rewritten
  expert→slot in place → return true, graph proceeds.
- **P4 — correctness gate** (script): same prompt, `--streaming off` (stock mmap) vs
  `--streaming on`, CPU backend both sides, compare logits. Gate: bit-identical
  (CPU backend is deterministic; any mismatch is a bug, not noise).

### Prefill discipline (M1 scope)

`mul_mat_id` needs every expert selected by the ubatch resident *simultaneously*, so the
per-ubatch expert union must fit `n_slots`. M1: clamp `n_ubatch` (start 1 = pure
correctness, raise to 8–16 = union stays well under 32/layer per finding 7's
union-growth curve). Expert-major prefill replaces this crutch in M2.

### M1 numbers (OLMoE, Q4_K_M GGUF ≈ 4.3 GB)

- experts ≈ 3.6 GB of the file; `n_slots=32` → ~1.8 GB expert RAM, trunk mmap'd
- M0 measured the I/O side of this exact configuration at **20.9 tok/s** I/O-limited
  (cache=32, real code-workload trace); Python PoC did 16.2 tok/s *total* — so the
  compiled path has real headroom
- backend: CPU-only for the gate (deterministic, honest); Metal slot-upload is M2+

### M1 risks, named

- scheduler split overhead at 16 observed nodes/token — measure; if it hurts, batch the
  observation (one hook per graph, walk all layers)
- `ggml_cpy` of ids must not be fused/elided — verify node survives graph optimization
- warm-path regression: when all experts hit cache, rewrite is a no-op memcpy of 8 ints;
  overhead must be ≤ a few µs/layer

## Non-goals for Phase 1
Mixed-precision store, prune-at-install, network tier (Phase 3); phone builds (Phase 4); anything that changes model outputs silently (never).
