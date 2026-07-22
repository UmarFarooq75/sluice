# Contributing

The most valuable thing you can send us is **a row for the measured table with its
receipts attached.**

## Add a benchmark row

Every number in this repo traces to an artifact. A community row is held to the same
standard — not because we distrust you, but because a number without its conditions
cannot be compared to anything.

1. `bash scripts/install.sh` then `./cli/sluice doctor gpt-oss-20b`
2. Run it, and copy the caption the UI prints under the reply, or the CLI's
   `[Ns · N tok/s · reused N ctx]` line.
3. Open a **Benchmark report** issue and paste: machine, full `doctor` output, the exact
   preset/quality/reasoning settings, and the caption.

**What makes a row usable:** the configuration, the machine, and whether the box was
quiet. A tok/s figure without those three is not a datapoint, it is a rumour.

## Standards this repo holds itself to

These are not aspirations; they are the rules the existing history was built under, and
a PR is measured against them.

- **A prediction is written before the run**, in `docs/lablog.md`, with a falsifier.
- **A missing value is not a data point.** A crashed or empty leg is VOID and can never
  become one side of a comparison — enforced by `make gate`.
- **Bit-exactness is config-pinned**: same settings → same bits, gate-enforced. Across
  batch shapes the upstream backend itself varies in deep decimals. We say so.
- **`make gate` must be green** before any engine change is committed.
- **A screenshot is a claim** and is captioned with the configuration visible in it.

## Running the gates

```bash
make gate-ref     # snapshot the current engine BEFORE you change it
make gate         # static checks + bit-exact + byte-identical-off
make static       # static only, no model needed
```

Model-dependent legs self-defer when the machine cannot host them. **A deferred gate is
not a green gate** and the output says so.
