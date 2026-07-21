# Run 1 (2026-07-21 21:38–21:47) — VOID for gate 1. Kept as the record.

**Harness bug, not an engine result.** `field()` used `re.search` without
`re.MULTILINE`. At line 130 the call site passed an inline `(?m)`; at line 127 —
leg A's turn-2 history builder — it did not. So `^canon_reply:` never matched
there, `default=""` was returned, and **leg A's turn-2 history carried an EMPTY
assistant turn**.

Proof from the artifacts: leg A rendered **315** tokens, legs B and C rendered
**421**. The three legs were supposed to render byte-identical prompts so that only
the KV path differed. A was a different conversation, so:

- **GATE 1 (A vs C) is meaningless** — two different prompts will not share a hash.
  The `RED` in run 1's summary is not a faithfulness result.
- **A's TTFT (6.74 s) proves nothing** about the intended collapse. It is low partly
  because A's prompt was 106 tokens shorter.

**What survives from run 1, and is quoted in the lablog:**
- The **control, B vs C, is valid** — both rendered 421 from the same string, and
  their hashes DIFFER. Stock KV reuse (canon OFF) already fails to reproduce a
  fresh full prefill.
- The **canonicalization mechanism works**: `held=496 kept=296 dropped=200
  decoded=108 canonical=404` in 13.23 s, after the reply, off the TTFT path.
- The **bit-exact gate was GREEN** (`fdf0f83dd70504c5`).
- The **inert check's "DIFFERS" was also a harness flaw** — it compared full stdout
  including wall-clock timings, prefetch counters and peak_rss, which vary between
  any two runs of the same binary. The deterministic lines (`mode=`,
  `logits_hash=74a27209c03b5503`, `text:`) were IDENTICAL.
