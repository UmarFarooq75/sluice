# Canon ship spec — `LLMSTREAM_KV_CANON` from dark to default

**Status: SPEC FOR REVIEW. Not implemented.** Per the owner ruling (fast default +
honest docs + strict switch later) and the 10:07 directive.

---

## 1. What ships, and why it is now shippable

`LLMSTREAM_KV_CANON` rewrites the KV at end of turn into the form the chat template
will render for that turn next time, so the next turn's prefix match covers the whole
history instead of dying at the assistant boundary.

**Measured (E41b run 2, matched 421-token rendering):**

| | reuse | re-prefill | TTFT |
|---|---|---|---|
| canon ON | **404 / 421** | **17** | **3.04 s** |
| stock | 296 | 125 | 8.22 s |
| cold | 0 | 421 | 16.22 s |

**Why it was blocked and no longer is:** gate 1 (canon KV vs cold prefill) was RED.
**RC4 established that as inherited upstream `n_ubatch` sensitivity** — pristine
b10064 splits the same way with no patch, no streaming, no pool, argmax stable. The
comparison was cross-shape by construction. It was never a defect in this feature.

---

## 2. Enable path

Default **ON** for chat surfaces; **OFF** for anything asserting bit-identity.

| surface | canon | why |
|---|---|---|
| `cli/sluice run` | **ON** | interactive chat, TTFT is the felt cost |
| UI (`ui/app.py`) | **ON** | same |
| `cli/sluice serve` | **OFF** initially | the HTTP client owns history; the echo contract cannot be enforced server-side without an API change (§5) |
| bit-exact gate, all `results/*` harnesses | **OFF** | gates pin configuration; canon changes KV shape |

Implementation is a default flip plus explicit `LLMSTREAM_KV_CANON=0` in the gate
paths — **not** removal of the env var. Keeping the switch is what makes the later
strict mode a one-line change rather than a revert.

---

## 3. The client contract (already built, `4ac08ca`)

When canon is on the engine prints `canon_reply:`; the client **must** echo that exact
string back as the assistant turn or KV reuse silently evaporates — no error, just a
slow turn. UI and CLI already do this, with a visible mismatch warning. Shipping canon
without that wiring would be the failure mode; it is already in place.

---

## 4. Pre-registered gates (all must pass before default flips)

1. **Faithfulness at matched shape.** Turn-2 `argmax` sequence and decoded text vs a
   fresh prefill, **3 distinct prompts**. *Pass*: text identical on all three.
   *Note*: `logits_hash` equality is **not** required and **not** expected — RC4
   showed that is cross-shape upstream variance. This gate tests what users see.
2. **TTFT reproduces.** Turn-2 TTFT ≈ **3 s** (accept ≤ 4.5 s), matched 421-class
   rendering, quiet box. *Falsifier*: > 6 s, i.e. no meaningful win over stock's 8.2 s.
3. **Byte-identical with canon unset.** Semantic-line comparison vs the pre-canon
   reference binary. Already green in E41b run 2.
4. **Bit-exact gate green** (`fdf0f83dd70504c5`), canon off. Already green.
5. **UI live smoke test.** Real 3-turn conversation through the playground: reuse
   climbs on turns 2–3, no mismatch warning fires, replies coherent. **This is the one
   thing never yet run live** — every canon measurement so far is harness-driven.

Gate 5 is the reason this is a spec and not a diff.

---

## 5. Known limits shipped with it

- **`serve` excluded.** HTTP clients own their history; enforcing the echo needs an API
  change. Excluded rather than silently degraded.
- **Cost is 7.3 s post-reply** (measured), off the TTFT path but real: it occupies the
  engine briefly after each turn. A user who sends the next message instantly will
  queue behind it. **Not measured under rapid-fire turns** — worth a leg.
- **Long conversations untested.** All measurements are turn 2. Canonicalization cost
  should scale with reply length, and reuse benefit with history length; neither is
  measured beyond one turn.
- **`argmax` stability is observed, not proven** (RC3, RC4, pristine). Gate 1 tests it
  on 3 prompts; that is evidence, not a theorem.

---

## 6. What I recommend against

**Do not** remove `LLMSTREAM_KV_CANON` or hard-wire canon on. The strict switch in the
owner ruling needs exactly this flag to turn it back off, and a feature that cannot be
disabled cannot be bisected when something later goes wrong.

---

## 7. Rungs

| # | | model window? |
|---|---|---|
| S1 | Gates 1–2 on 3 prompts, harness-driven | yes |
| S2 | Gate 5, UI live smoke test | yes (interactive) |
| S3 | Default flip + docs, gates 3–4 re-run | yes (gate only) |
| S4 | *(separate directive)* strict switch `LLMSTREAM_STRICT_EXACT` | — |

**Awaiting review. No code written.**
