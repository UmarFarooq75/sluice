# Security

sluice runs models **locally**. It does not phone home, and it has no network path
except the model downloads you ask for explicitly (`sluice pull`), which go to
HuggingFace over HTTPS.

## Reporting a vulnerability

Open a **private** security advisory via GitHub's *Security → Report a vulnerability*
on this repository. Please do not open a public issue for a vulnerability.

Include what you did, what happened, and what you expected. A reproduction matters more
than a severity rating.

## What is in scope

- The engine (`csrc/`) and the fork patch (`patches/`) — memory safety, malformed GGUF
  handling, anything that turns a model file into code execution.
- `cli/sluice` and `ui/app.py` — command injection, path traversal, unsafe deserialization.
- `scripts/install.sh` — it fetches and builds third-party code at a pinned revision;
  supply-chain concerns about that pin are in scope.

## What is not

- **Model output.** What the model says is a model property, not a vulnerability in
  sluice. Quality trades are documented in the README's quality dial.
- **Resource exhaustion you asked for.** Running a 63 GB model on a 16 GB machine will
  make it swap. The engine has a memory guard and `sluice doctor` will warn you first,
  but choosing to push past that is a supported, documented use.

## Known operational risk, stated plainly

Streaming a model larger than RAM puts sustained pressure on the page cache. On a
machine with little free memory this can degrade the whole system, not just sluice. The
guard sheds cache slots and `sluice doctor` reports a go/no-go before you start.
