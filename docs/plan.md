# Plan & how we work together

## Constraints (real, and by design)

- **Machine**: MacBook Air M2, 16 GB RAM, ~41 GB free disk. Umar uses it for his day job — the project must never get in the way.
- Everything lives in `~/Desktop/research/` — local `.venv/`, models in `models/`, no global installs. Delete the folder and the project is gone from the machine.
- Total footprint budget: ~6–8 GB disk. RAM is only used while a trace/experiment is actually running (~5 GB peak, minutes at a time); nothing stays resident.

This is a feature: the dev machine IS the target hardware class.

## Working model (day-job friendly)

- Umar's time: decisions, direction, review — minutes, not hours. Claude does the heavy lifting (research, code, experiments, analysis) in sessions.
- Heavy compute (trace collection, simulations) runs in short batches, backgrounded, or overnight — never long foreground hogs during work hours.
- Cross-session continuity via Claude's memory + these docs. Any session can pick up where the last left off; docs in `docs/` are the source of truth, updated as findings land.
- Cloud/borrowed hardware only when Phase 0 results justify it (e.g. a Linux NVMe box for the big-model demo).

## Lab-rat model choice

**OLMoE-1B-7B-0924-Instruct, 4-bit MLX** (`models/`, ~4 GB):
- True fine-grained MoE: 16 layers × 64 experts, top-8 → rich routing signal (1,024 experts total).
- Runs comfortably in ~5 GB RAM on the M2 Air.
- Fully open (weights + data + training code), and it's the exact model used in the papers we benchmark against (SpecMD, edge-MoE study) → direct comparability.
- **Why not gemma4:12b**: it's dense — no experts, no router, nothing to trace or stream. Dense models can't benefit from our technique at all.
- Later additions: DeepSeek-V2-Lite (10.4 GB Q4) as the frontier-architecture representative (needs ~11 GB RAM — only when nothing else is running); Qwen3-30B-A3B on borrowed hardware.

## Phase 0 — Trace & simulate (now; no engine)

Prove the numbers offline before writing any engine code. All experiments are trace-driven — model runs once per workload, simulations replay traces cheaply.

1. **Tracer**: hook OLMoE's router in MLX (its model code is plain Python), log `(token, layer, top-8 expert ids, gate weights)` for every step.
2. **Workloads**: diverse prompt suite — code, math, prose, chat, multilingual, long-prefill — few hundred tokens each.
3. **Simulator**, over the traces:
   - Cache policies: LRU vs LFU vs least-stale vs learned working set — hit-rate curves vs cache size.
   - Working-set concentration: what % of experts covers 90% of activations, per task vs globally (validates pillar 2 + 6).
   - Temporal locality: consecutive-token expert overlap (calibrates against published Mixtral numbers).
   - Prefill expert-union: unique experts touched vs prompt length (validates expert-major prefill, pillar 3).
   - Usage-frequency distribution: bits-budget allocation for mixed precision (pillar 4).
   - Network-tier sim: disk-cache miss rate over a multi-task session at various local-storage budgets (pillar 6).
4. **Deliverable**: findings report with charts → becomes the launch blog post / README evidence. Success = we know, with our own data, which policies hit ≥90% and how small the working set really is.

## Phase 1 — Streaming core (MVP engine)

GGUF in → dense-resident + expert streaming + the winning cache policy from Phase 0. **Expert-major prefill from day one.** One architecture family first (Qwen3-MoE), then DeepSeek-family (covers GLM + Kimi too). Ship with an honest same-hardware A/B vs llama.cpp mmap — the comparison colibri never published.

## Phase 2 — The brain

Router-lookahead prefetch, persistent learned working sets, shareable task profiles.

## Phase 3 — The ROM stack

Mixed-precision storage layout · prune-at-install (REAP) converter · network-as-coldest-tier. This phase is where the project becomes something that doesn't exist anywhere.

## Phase 4 — Edges

Metal/CUDA expert tiers · phone build (Android/Termux) · more architecture families.

## Phase 0.5 — Cover the flaws (interleaved with Phase 0)

Target revised by Umar: **≤3 GB RAM for 671B-class, low compute** (24 GB is too much). See [minimal-ram.md](minimal-ram.md) for the RAM-floor ledger and byte-reduction stack, and [flaws-and-coverage.md](flaws-and-coverage.md) for the full red-team. Experiments, in priority order:

1. **Composition interaction matrix** — techniques alone vs pairs vs full stack (quality + bytes + FLOPs) on OLMoE. Our core contribution; nobody has published this.
2. **Adaptive top-k quality curve** — gate top-p 0.9/0.8/0.7 vs quality; per-layer cutoffs (traces already log full router distributions).
3. **Expert-skip on easy tokens** (our novel mechanism — attention runs normally so KV stays intact; easy tokens use shared-expert only, skipping all routed-expert I/O). Trigger = router entropy / gate mass.
4. **TEAL-on-MoE transfer** — does 40–50% training-free activation sparsity hold for expert MLPs?
5. **Floor-mode emulation** — zero expert cache, stream OLMoE from SSD, measure tok/s vs formula prediction.
6. **Router-lookahead recall** on OLMoE (hidden-state hook) — is colibri's 71.6% general?

## Status log

- **2026-07-17 (night, Phase 1 / M1)**: M0 passed (csrc/stream_bench.cpp: 20.9 tok/s I/O-limited at cache=32 on real trace, ~90% of SSD ceiling vs Python's 50-65%). llama.cpp spike complete — three load-bearing facts: public `cb_eval` hook fires mid-graph with data writable (imatrix's mechanism); expert tensors are contiguous per-expert extents in the GGUF (offs + e*nb[2]) so **no repacked store needed**; ids feed both the weight-gather and mul_mat_id, so streaming needs dual id tensors. Full M1 integration plan in phase1-design.md. **Fork built** (branch `llmstream` in vendor/llama.cpp; private fork — upstream forbids AI PRs, we never submit): `src/llmstream.{h,cpp}` (slot tensors in own CPU buffer, env LLMSTREAM_SLOTS), olmoe.cpp slot branch with TENSOR_SKIP accounting, build_moe_ffn dual-ids (`ffn_moe_topk_slots` dup node feeding all 8 expert-indexed ops). Out-of-tree driver `csrc/stream_run.cpp` (GGUF extents + per-layer LRU + parallel F_NOCACHE fetch + in-place id rewrite + FNV logit hashing) and `csrc/probe_cb_eval.cpp` (M1-A mechanism probe) + `scripts/m1_gate.sh` (3-way bit-exact gate) all compiled. Blocked only on the OLMoE Q4_K_M GGUF download (allenai official; link slow ~0.5MB/s, resumed with xet off). Gate criterion: resident ≡ slots32 ≡ slots12 logit hashes at n_ubatch=1.

- **2026-07-17/18 (overnight autonomous run)**: DS-V2-Lite downloaded (12GB models total on disk); full replication pipeline ran unattended. **Finding 22: expert-skip VALIDATED** (skip 15–18% of routed I/O for 1.4–2.5% NLL on math/prose; inverse control proves gate-mass is a real easiness signal — 50–100× more damage skipping confident tokens). **Finding 23: replication confirms cache shapes** (LRU@C16 ≈ 4.3–4.9× random on both architectures) and **revives adaptive-k for DeepSeek-family** (k=3-of-6 only +3.6–7.7% vs +14–25% on OLMoE). Finding 21: bootstrap CIs — all headline numbers robust. Finding 20 (clean rerun): prefetch↔margin substitution law is intrinsic, not contention; best PoC config = cap32+margin = 16.2 tok/s (21% of resident), bit-identical quality. Docs + README consolidated. **Next: Phase 1 — compiled engine core** (design fully specified in README from findings 1–23).

- **2026-07-17**: Deep research + brainstorm done (see research-findings.md, techniques.md). venv + mlx-lm installed locally. OLMoE-1B-7B-4bit downloading to `models/` (correct repo: mlx-community/OLMoE-1B-7B-0125-Instruct-4bit). Tracer (src/tracer.py) + simulator (src/simulate.py) written. RAM target revised to ≤3 GB (minimal-ram.md); flaw red-team done (flaws-and-coverage.md).
- **2026-07-17 (evening)**: Quality+accuracy bridge done — margin-gated cache-aware routing = near-lossless traffic dial (findings 11-13, 17); DP allocator verdict: uniform fine except starved regime (14); network tier = last-mile saver, ~1 fetch/tok at 90% disk (15); task-switch detection 6 tokens (16); lookahead recall 83.9% on OLMoE, beats colibri's GLM number (18). Engine core evidence-complete. PoC SSD streamer built (expert_store/ 3.6GB, 1024 files; src/streamer_poc.py): SSD 1.75GB/s at 3.5MB granularity, resident ref 66.8 tok/s — streamed sweep running. DS-V2-Lite downloading for sigmoid-router replication.
- **2026-07-17 (later)**: Analysis red-team (analysis-gaps.md, 19 gaps, 6 simulator fixes). Traces collected (~7.6k decode tokens, 6 workloads + prefill sweep). Simulator v2 run. **First findings: docs/findings-phase0.md** — caching 4–5× above random, belady oracle shows 15–20pt prediction prize, tiny-cache inversion → hybrid pinned+LRU design, expert-major prefill 16–599× fewer loads, working sets are frequency-not-coverage (ROM plan revised), NEGATIVE: adaptive top-k dead on OLMoE's flat softmax router (retest on sigmoid-router models). Autonomous /loop running: next = restricted-routing quality eval, DP allocator, mixed-session trace, DeepSeek-V2-Lite.
