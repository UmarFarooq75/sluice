# RC2 run 1 — VOID. Two harness defects, no engine defect.

**Nothing here is quotable.** No leg produced a comparison.

## 1. Shell layer reported success over a dead harness
`bisect.py` died on `BrokenPipeError`; `run.sh` logged `rc=0` and wrote `DONE: done`.

```sh
echo "[$(date '+%m-%d %H:%M:%S')] legs finished rc=$? (avail=$(sl_avail_gb) GB)"
```
bash expands left to right, so `$(date ...)` runs **first** and resets `$?` to *date's*
status. `rc` was always 0. Fixed by capturing `rc=$?` on its own line and writing the
marker via `sl_finish`, which emits `void` on non-zero. Enforced by
`scripts/check_shell_rc.py` in `make gate`.

## 2. The engine never crashed — the harness never asked it to be a server
`env_for()` did not set `LLMSTREAM_SERVER=1`. Without it the engine runs
**single-shot**: it treats the sentinel argv as the prompt, generates, and exits
normally. `<<<READY>>>` is printed **only** in server mode
(`csrc/stream_run.cpp:1242`, inside `if (server_mode)`), so `until_ready()` read to
EOF and the first `stdin.write` hit a closed pipe.

### The SWA line is a red herring — stated because the opposite reading was offered
`S1a_under_t1.err` ends at `llama_kv_cache_iswa: using full-size SWA cache`, which
*looks* like a crash in the very subsystem RC2 is probing. It is not. That is simply
the last line llama.cpp writes to **stderr** during load; everything after it goes to
**stdout**, which the harness had already drained and discarded. The process exited
0 after a normal single-shot generation.

**There is no evidence here about SWA, ironic or otherwise.** Reading it as evidence
would have been the flattering interpretation, and it would have been wrong.
