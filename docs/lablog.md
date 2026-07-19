# Lab log — chronological record of every change and what happened

Rule: every experiment gets an entry the same day — the change, the expectation,
the measured result (with artifact path), and the conclusion or open question.
Negative results and self-inflicted failures are logged with the same weight as
wins; they steer the roadmap just as hard. Thematic writeups live in
`docs/findings-phase0.md` / `docs/research-findings.md`; this file is the raw
"we changed X → Y happened" timeline.

Format per entry:
- **Change/Experiment** — what we did
- **Expected** — prediction beforehand (forces honesty about surprises)
- **Result** — measured numbers + artifact
- **Conclusion / next question**

---

## Phase 0 (2026-07-17 → 07-18, backfill pointer)

OLMoE tracing → cache simulation → physical streamer PoC (experts as SSD
extents, page cache bypassed): resident 76.7 tok/s vs all-SSD floor 3.5 tok/s
vs half-cache+margin 16.2 tok/s, NLL 3.002 bit-consistent in all three
(`results/streamer_poc.json`). Qwen3.6-35B-A3B streamed at 8.3 tok/s where
Ollama v0.32.1 on the same machine swap-collapsed to ~0.03 tok/s (its own logs
archived). Full detail in the findings docs; this log starts fine-grained at
the gpt-oss-120b bring-up.

---

## 2026-07-18/19 — gpt-oss-120b (117B params) bring-up on the 16GB M2 Air

### E1. gpt-oss family support in fork + driver
- **Change**: openai-moe adapter with slot-backed experts; two firsts for the
  engine: per-expert *bias* tensors (2D `{n, n_expert}` extents striding on
  `nb[1]`) and raw-logit gating (SOFTMAX_WEIGHT), which forced the margin mask
  to write `-INFINITY` instead of `0` (0 is not a floor in logit space).
- **Expected**: no behavior change for existing families.
- **Result**: OLMoE gate PASS, hash `b6869f5b6ef36376` bit-identical across
  resident/slots32/slots12. Commits: fork `7d62a14`, repo `28636d7`.
- **Conclusion**: the slot mechanism generalizes across gating families and
  tensor layouts with a ~1-file adapter. Universality evidence, not proof —
  next family (DeepSeek-style shared experts) is M3.

### E2. Model download
- 63,387,346,208 bytes, byte-exact vs HuggingFace `x-linked-size`. MXFP4
  experts are native to *every* published gpt-oss-120b GGUF (~63GB; no smaller
  quant exists — experts are already ~4.25 bit).

### E3. First streamed run — SLOTS=12, margin 0 (exact routing)
- **Expected**: 3–6 tok/s (estimate from OLMoE/Qwen scaling).
- **Result**: **0.75 tok/s decode**, 0.73 prefill; hit 52.6%; 64.0GB read at
  713 MB/s; stall 50.5s of 76s; text fully coherent and correct
  (`results/gptoss_m0_slots12.txt`).
- **Conclusion**: a 117B model runs, quality-anchored, on a 16GB laptop — but
  4–8× below estimate. Why: 68 misses/token × 13.25MB ≈ 900MB/token; entirely
  I/O-bound; estimate had assumed OLMoE-like hit rates, gpt-oss's 128-expert
  top-4 routing has far less temporal locality at 12 slots. Estimation error
  logged: never scale hit rates across architectures.

### E4. SELF-INFLICTED: ladder loop bug → machine hang + OOM kill
- **Change**: benchmark ladder launched via a `for … set -- $cfg` loop.
- **What happened**: the harness shell is zsh; zsh does not word-split `$cfg`,
  so `LLMSTREAM_SLOTS` came through **empty** → runs fell back toward
  resident-loading a 63GB model → swap hit 6.6GB, the laptop hung repeatedly,
  macOS SIGKILLed rung 1 (exit 137). Two further stray `stream_run` processes
  (one duplicate launch, one orphan from a crashed session) compounded RAM
  pressure earlier in the day.
- **Fixes**: ladder moved into a `#!/bin/bash` script; `run()` now aborts on
  empty slots/margin; one-process-at-a-time rule; bad artifacts deleted.
- **Deeper conclusion (product defect, not script defect)**: the engine
  trusted its configuration blindly. A product that streams huge models MUST
  protect the host machine itself → E10 (engine-level pressure guard).

### E5. First ladder — margin sweep at slots12 + RAM curve (CONTAMINATED)
- **Result**: m0.02 → 0.55 tok/s (hit .533); m0.5 → 0.48 (hit .704);
  m1.0 → 0.73 (hit .877); m2.0 → **10.47 tok/s** (hit .998);
  slots16 m1 → 0.19 tok/s (48 MB/s effective bandwidth); slots8 m1 → 2.99.
- **Audit verdict**: the slots12 margin rungs and slots16 ran minutes after
  the E4 swap blowup — their effective bandwidth (216–521 MB/s vs 1128–1161
  clean) shows page-fault contamination. The m1.0-slots12 run has ~58s of
  decode time that is neither stall nor compute — swapped-out resident pages.
  **"Margin doesn't help at slots12" is UNPROVEN; re-run required.** The
  slots16 collapse mechanism (cache + weights + KV exceeding headroom) is
  directionally real; magnitude unquotable.
- **Two numbers that survive**: 10.47 tok/s = the CPU compute ceiling (99.8%
  hit ⇒ I/O hidden; ran fast despite elevated swap, so ceiling is ≥ this);
  and slots8's 2.99 with full SSD bandwidth.

### E6. m=2.0 is fool's gold — quality collapse
- **Result**: text degenerates to "the the" + newline flood
  (`results/gptoss_m2_slots12.txt`). At 12 slots, margin 2.0 pins routing to
  the cached set ~always (hit .998) — effectively a fixed 12-expert model.
- **Conclusion**: speed numbers are meaningless without a quality metric
  attached. Never report a margin rung without fidelity → E9.

### E7. Cliff hunt at the slots8/10 sweet spot (clean system)
- **Result** (all coherent unless noted):
  | config | tok/s | hit | text |
  |---|---|---|---|
  | slots8 m1.0 | 2.99 | .792 | clean |
  | slots8 m1.25 | 5.11 | .885 | clean |
  | slots8 m1.5 | 8.25 | .958 | answer correct, then newline flood — termination breaks first |
  | slots10 m1.0 | 3.78 | .837 | clean |
  | slots10 m1.25 | **5.72** | .922 | clean |
- **Conclusions**: (a) quality-intact best = 5.72 tok/s; (b) the governing
  formula holds quantitatively: time/token ≈ misses×13.25MB÷1.15GB/s + 0.095s
  — predicts all clean rungs within ~15%; (c) degradation onsets at the
  *end-of-answer* boundary (termination/control routing leaves the topic-local
  cached set) — margin cliff for this model sits between 1.25 and 1.5 at 8
  slots; (d) smaller cache beat bigger cache twice (slots8 > slots12 > slots16)
  because memory headroom governs effective SSD bandwidth — on a 16GB machine
  the cache must be sized to *pressure*, not to "more is better".

### E8. Methodology error found in our own quality reads
- gpt-oss is harmony-chat-format trained; every run used a raw text prompt.
  Post-answer fragments ("Sure!", "I'm sorry…") are template mismatch, not
  necessarily margin damage. All cross-rung *eyeball* quality comparisons to
  date are confounded; speed/hit numbers unaffected. Fix: `LLMSTREAM_CHAT=1`
  wraps prompts in the model's own template (E9 build).

### E9. Instrumentation build (turn quality & I/O into numbers)
- **Change**: (1) router-agreement counter — per token, true pre-mask top-k vs
  post-mask selection, exact overlap %; (2) `read_work`/`preads` — summed
  worker pread time vs eval-thread stall separates device latency from
  queueing; (3) `LLMSTREAM_NLL=1` — teacher-forced mean −log p over a fixed
  passage, deterministic quality number per config; (4) `LLMSTREAM_CHAT=1`.
- **Result**: OLMoE gate PASS unchanged (`b6869f5b6ef36376`) — counters cost
  nothing at m=0. Commit `d45da99`.
- **Running now**: 9-rung agreement+NLL sweep at slots8/10 under a
  memory-guarded protocol (refuse rung start under pressure)
  (`scripts/gptoss_quality_sweep.sh`, artifacts `results/gptoss_agree_*.txt`,
  `results/gptoss_nll_*.txt`).

### E10. Engine-level machine protection (in progress, this entry updates)
- **Change**: `LLMSTREAM_SLOTS=auto` — size the cache from *this machine's
  available memory* read from the OS at startup, not from the caller's guess;
  plus a runtime pressure monitor that sheds cache slots (MADV_FREE's their
  pages) when macOS signals memory pressure, and grows back when calm.
- **Why**: E4/E5 proved the host must never be collateral damage. Ollama-class
  engines avoid this by refusing to load; we run the model AND stay polite.
- **Result**: pending build + gate after the quality sweep completes.

---

## Defect ledger — every known defect, questioned to root cause

| # | Defect / symptom | Questioning → root cause | Status / experiment |
|---|---|---|---|
| D1 | Laptop hung during runs | Why did RAM run out? Runs launched unbounded (shell bug) + duplicates + slots16 over headroom. Why could they take the machine down? **Engine has no self-protection.** | Script guards done; engine guard = E10 |
| D2 | Prefill 0.7–2.9 tok/s → minutes of TTFT on real prompts | Why so slow? Prefill streams experts per token serially, same as decode. Why? Expert-major prefill (measured 16–599× on OLMoE) not yet ported to gpt-oss. | Next after E10 |
| D3 | Margin units are per-family (0.02 meant something on OLMoE probs; no-op on gpt-oss logits) | Why? Mask threshold lives in the gating score's native scale. A universal product cannot ask users to know gating families. | Design: normalized margin (percentile of per-token score gaps) or entropy-adaptive margin. Experiment queued |
| D4 | Quality was judged by eyeball | Why unacceptable? m1.5 rung *looks* fine until the last tokens; template mismatch (E8) fools the eye both ways. | Fixed by E9 (agreement + NLL); sweep running |
| D5 | At high hit rates, stall shrinks but decode doesn't reach ceiling | Where does the time go? Not device reads (read_work now measured), suspicion: fetch-path serialization / page faults / repack. | Decompose with E9 counters on clean re-runs |
| D6 | All results = one English prompt, 64 tokens | Why risky? Routing locality is topic-dependent; margin cliff may move on code/multilingual/long-context. | Prompt battery + 1k-context runs queued |
| D7 | prefetch_issued=0 in every gpt-oss run | Why off? We disabled it for first runs (phase-0 rule: prefetch pays only below hit-EMA 0.80). At slots8 m1.25 hit=.885 → gate would keep it off. Is the 0.80 gate right for THIS regime? | Re-test: force prefetch on at slots8 m1.25 |
| D8 | Raw prompts to a chat-trained model | See E8. | `LLMSTREAM_CHAT=1` built; use in all future quality runs |
| D9 | Universality boundary: dense models | A dense 72B touches all weights every token — nothing to stream selectively; that is physics, not engineering (docs/dense-strategy.md). Scope stated honestly: MoE-first. | M3: DeepSeek family next (shared+routed experts) |
| D10 | Thermal never logged (fanless chassis) | Are late rungs slower because hot? Counter-evidence: fastest rungs ran last. Still unproven either way. | Add powermetrics logging to protocol |

## Standing protocol (enforced from 2026-07-19)
1. One model process at a time; check for strays before launch.
2. No rung starts under memory pressure (script guard + engine guard).
3. Artifacts labeled clean/contaminated; contaminated numbers never quoted.
4. Bit-exact OLMoE gate before every commit touching fork or driver.
5. Every experiment → lablog entry same day, prediction written before result.
6. Quality claims only with agreement/NLL attached (eyeball is a smell test).
