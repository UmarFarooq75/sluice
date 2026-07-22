# E43 — comparative benchmark: sluice vs Ollama vs vanilla llama.cpp

> **PROVENANCE, stated because it matters: I authored this spec. It was not
> recalled.** The directive referred to "the E43 spec already in your queue notes";
> there is no E43 reference in `docs/`, `results/`, or `ROLES-AND-STATE.md`. Rather
> than present invention as recall, this document is written from first principles,
> grounded in the one prior head-to-head artifact the repo does own
> (`results/qwen36_headtohead.json`, 2026-07). **Every design decision I had to make
> myself is marked §D-n and needs review.**

---

## 1. What this benchmark must not do

The dishonest version of this benchmark is easy and we should name it before
building the honest one.

**sluice is SLOWER than vanilla llama.cpp when the model fits in RAM.** That is not
a defect; it is the trade. We stream experts from SSD, which costs I/O that a
resident runtime does not pay. A benchmark that reports one tok/s number per runtime
will either flatter us (by choosing a model that doesn't fit the competition) or
damn us (by choosing one that does), and both are marketing.

**§D-1 — the primary result is a frontier, not a number.** Each runtime is plotted
as (peak RAM footprint, tok/s, load outcome) at a fixed model and machine. The claim
under test is not "sluice is faster" but:

> *At a RAM budget where the competition cannot run the model at all, sluice runs it
> at a usable speed; and where the competition can run it, sluice costs you speed and
> we say how much.*

Both halves get measured. A run where vanilla wins is a result, not a failure.

---

## 2. The three arms

| arm | what it is | how it is obtained |
|---|---|---|
| **sluice** | `csrc/stream_run`, streamed (`LLMSTREAM_SLOTS=16`) and resident (`SLOTS` unset) | in-tree |
| **vanilla llama.cpp** | `llama-cli` built from **pristine b10064**, no fork patch | git worktree at `b10064`, separate build dir (**§D-2**) |
| **Ollama** | the shipped product, same GGUF hardlinked into its blob store | **not installed on this machine** (**§D-3**) |

**§D-2 — why a pristine build rather than the patched tree's `llama-cli`.**
`vendor/llama.cpp/build/bin/llama-cli` exists, but it is built from the *patched*
tree. With our env vars unset the patch should be inert — but "should be" is not a
measurement, and we have never verified it. So E43 builds pristine `b10064` and
**additionally runs a fourth, cheap arm**: patched-`llama-cli`-with-all-flags-unset
vs pristine `llama-cli`, same prompt, greedy.

> **This closes a real hole in our own protocol.** Protocol #3 ("byte-identical when
> off") has only ever been checked on *our driver's stdout*. It has never been
> checked at the *llama.cpp inference* level — i.e. whether our fork changes upstream
> behaviour when nobody asks it to. E43 establishes that or finds it false. Either
> outcome is worth more than the headline comparison.

**§D-3 — Ollama is not installed.** Installing it is a download, and downloads need
authorization under the standing rules. The script therefore **detects and defers**:
if `ollama` is absent, that arm reports `NOT INSTALLED` and the run continues. It
never installs anything. The exact step for the owner to authorize is printed.
The prior artifact's method is reused verbatim once it exists: **hardlink our GGUF
into Ollama's blob store (zero-copy) and serve it via a Modelfile**, so all arms read
byte-identical weights and no second 12 GB copy lands on disk.

---

## 3. Fairness controls

Every one of these exists because a violation would silently produce a flattering
number.

| control | why |
|---|---|
| identical GGUF file (hardlinked, verified by inode) | different quants are different models; this is the single easiest way to fake a win |
| greedy decoding, `temp=0`, everywhere | sampling variance would swamp the differences |
| identical prompt and `n_gen` | prompt length drives TTFT and the expert union |
| identical thread count, pinned explicitly | llama.cpp and Ollama pick different defaults |
| one process at a time (protocol #1) | two engines already took this machine into swap once |
| avail ≥8 GB and **swap delta recorded per leg** | the prior artifact shows swap thrash *is* the differentiator, so it must be data, not a footnote |
| cold vs warm page cache stated per leg | a warm run can look 3× faster than the same run cold |
| output text captured for every arm | a fast runtime producing garbage is not a faster runtime (E6) |

**§D-4 — model choice.** Two models, because one cannot show both halves of the
frontier on a 16 GB box:
- **gpt-oss-20b (12.1 GB)** — *fits, barely.* All three arms should run. Expect
  **sluice to lose on speed** and win on footprint. This is the honest-cost half.
- **a >16 GB model** — *does not fit.* Expected: vanilla DNF/OOM, Ollama swap
  collapse, sluice runs. This is the capability half. **The only such model already on
  disk is none** — `qwen3.6-35b-a3b` was measured but has no pinned URL in the
  manifest, and gpt-oss-120b is 63 GB and not downloaded. **So this half CANNOT run
  today without a download**, and the script marks it `PENDING MODEL` rather than
  quietly reporting only the half that happens to be runnable. Flagged for a
  directive, not resolved here.

---

## 4. Safety — this benchmark can take the machine down

It has before. `results/qwen36_headtohead.json`: Ollama "loaded via mmap … then
collapsed into OS paging … swapouts 23.7M pages — machine unusable." That is the
result we want to *record*, not the state we want to be in unattended.

**§D-5 — a watchdog is mandatory, not optional.** Every leg runs under a monitor
that samples `avail` and swap every 5 s and **kills the leg** if either:
- avail drops below **1.0 GB**, or
- swap grows by more than **4 GB** from the leg's own start, or
- the leg exceeds its wall-clock cap (default **600 s**).

A killed leg is recorded as `THRASH-KILLED` **with the telemetry that triggered it** —
which is the honest form of the qwen result, and strictly more informative than a
tok/s number nobody could reproduce without hanging their laptop.

---

## 5. Metrics per leg

`load_outcome` (loaded / OOM-killed / thrash-killed / DNF-timeout) ·
`ttft_s` · `prefill_tok_s` · `decode_tok_s` · `peak_rss_gb` (`time -l`) ·
`phys_footprint_gb` (sluice only; the others don't report it) ·
`swap_delta_gb` · `avail_min_gb` · `output_text` · `logits_hash` (sluice only).

**§D-6 — the runtimes do not report the same things.** Ollama reports its own
tok/s in its server log; llama.cpp reports timings on stderr; we report both plus
`phys_footprint`. The script parses each in its own format and **never synthesizes a
missing metric** — absent is recorded as `null`, never as 0 and never inferred. (This
is the harness VOID rule applied to a benchmark: a missing value is not a data point.)

---

## 6. Pre-registered expectations

Written before any run, so the result can embarrass them.

1. **On gpt-oss-20b (fits), vanilla llama.cpp resident beats sluice streamed on
   decode tok/s** — I expect roughly 8–10 vs our 6.14. *Falsifier*: sluice wins, which
   would mean I have misunderstood our own cost model.
2. **sluice's peak footprint is the lowest of the three** at its streamed setting.
   *Falsifier*: anything lower than our 6.73 GB `phys_footprint`.
3. **Ollama, once installed, runs gpt-oss-20b successfully** on this box (12.1 GB
   does fit) at a speed close to vanilla llama.cpp, since it wraps the same engine.
   *Falsifier*: Ollama DNFs on a model that fits — that would be a packaging finding,
   not an engine one.
4. **Patched-flags-unset == pristine** on greedy output text for the same prompt.
   *Falsifier*: they differ ⇒ our fork changes upstream inference when nobody asked it
   to, which is a **protocol #3 violation at the engine level** and outranks the entire
   benchmark.

---

## 7. What is deliberately out of scope

Quality benchmarking (MMLU-style). We have no harness for it, competitors publish
theirs, and a speed comparison that silently implies quality parity would be exactly
the dishonesty §1 warns about. Our quality claim is bit-exactness against ourselves,
which is a different axis and already gated.
