#!/bin/bash
# sluice installer — reconstructs everything a fresh clone does NOT ship.
#
# `vendor/` is gitignored, so a clone has no llama.cpp. This script fetches the
# pinned upstream revision, applies our fork patch, builds the library and the
# streaming driver, and creates the Python venv for the CLI/UI.
#
# It does NOT download any model. Use `./cli/sluice pull <name>` for that, after
# `./cli/sluice estimate <name>` has told you what your machine will actually do.
#
# Safe to re-run: every step is skipped if already satisfied.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

# Pinned upstream revision our fork patch applies against (git describe on the
# working tree reports b10064-4-g… = this tag plus our 4 fork commits).
LLAMA_REV="${LLAMA_REV:-b10064}"
LLAMA_URL="${LLAMA_URL:-https://github.com/ggml-org/llama.cpp}"

say()  { printf "\033[1m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[33m!!\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[31mxx\033[0m %s\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- platform ----
OS="$(uname -s)"; ARCH="$(uname -m)"
say "platform: $OS/$ARCH"
if [ "$OS" != "Darwin" ]; then
  warn "Only macOS (Apple Silicon) is tested. scripts/build_driver.sh links"
  warn "-lobjc -framework Foundation for the thermal probe, which will NOT build"
  warn "on $OS without edits. Continuing, but expect the driver build to fail."
fi
[ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ] && \
  warn "Intel Mac detected; measurements in this repo are all Apple Silicon."

# ------------------------------------------------------------ prerequisites ---
for tool in git python3; do
  command -v "$tool" >/dev/null 2>&1 || die "missing prerequisite: $tool"
done

# cmake resolution. Bare `cmake` is NOT reliably on PATH: on this project's own
# reference machine it lives inside the Python venv (pip-installed), which is how
# vendor/ was originally built — so the quickstart's very first command failed on a
# clean shell. Resolve it, and if it is genuinely absent, say how to get it rather
# than dying on "command not found".
CMAKE=""
for cand in cmake \
            "$PWD/.venv/lib/python3."*/site-packages/cmake/data/bin/cmake \
            /opt/homebrew/bin/cmake /usr/local/bin/cmake \
            /Applications/CMake.app/Contents/bin/cmake; do
  if command -v "$cand" >/dev/null 2>&1; then CMAKE=$(command -v "$cand"); break
  elif [ -x "$cand" ]; then CMAKE="$cand"; break; fi
done
if [ -z "$CMAKE" ]; then
  # last resort: install it into the venv we are about to create anyway
  say "cmake not found — installing it into .venv (pip)"
  python3 -m venv .venv 2>/dev/null || true
  ./.venv/bin/python3 -m pip install --quiet cmake 2>/dev/null || true
  CMAKE=$(ls "$PWD"/.venv/lib/python3.*/site-packages/cmake/data/bin/cmake 2>/dev/null | head -1)
fi
[ -n "$CMAKE" ] && [ -x "$CMAKE" ] || die "cmake not found and could not be installed.
     Install it with:  brew install cmake     (or)  pip install cmake"
say "cmake: $CMAKE"
command -v clang++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || die "missing a C++ compiler"
say "prerequisites ok (git, cmake, python3, c++)"

# --------------------------------------------------------------- llama.cpp ----
if [ ! -d vendor/llama.cpp/.git ]; then
  say "fetching llama.cpp @ $LLAMA_REV (vendor/ is gitignored, so this is not in the clone)"
  mkdir -p vendor
  git clone --filter=blob:none "$LLAMA_URL" vendor/llama.cpp
  git -C vendor/llama.cpp checkout --quiet "$LLAMA_REV"
  say "applying patches/llmstream.patch (the expert-streaming fork)"
  git -C vendor/llama.cpp apply "$ROOT/patches/llmstream.patch" \
    || die "patch did not apply against $LLAMA_REV — set LLAMA_REV to the matching upstream tag"
else
  say "vendor/llama.cpp already present — leaving it alone"
fi

if [ ! -e vendor/llama.cpp/build/bin/libllama.dylib ] && [ ! -e vendor/llama.cpp/build/bin/libllama.so ]; then
  say "building llama.cpp (this is the slow step)"
  # GGML_METAL is passed explicitly rather than left to the platform default so
  # this and cli/sluice's own bootstrap produce the SAME library. Two build paths
  # that disagree on a flag is how you get a "works for me" that isn't.
  CM_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
  [ "$OS" = "Darwin" ] && CM_FLAGS+=(-DGGML_METAL=ON)
  "$CMAKE" -S vendor/llama.cpp -B vendor/llama.cpp/build "${CM_FLAGS[@]}" >/dev/null
  "$CMAKE" --build vendor/llama.cpp/build --config Release -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
else
  say "llama.cpp already built — skipping"
fi

# the driver links -lllama -lggml -lggml-base -lggml-cpu by name; if the build
# half-finished, failing here with the missing library named beats a link error
for lib in llama ggml ggml-base ggml-cpu; do
  ls vendor/llama.cpp/build/bin/lib${lib}.dylib >/dev/null 2>&1 || \
    ls vendor/llama.cpp/build/bin/lib${lib}.so  >/dev/null 2>&1 || \
    die "llama.cpp build incomplete: lib${lib} missing from vendor/llama.cpp/build/bin"
done

# ------------------------------------------------------------------ driver ----
say "building the streaming driver"
bash scripts/build_driver.sh

# ------------------------------------------------------------------ verify ----
# End-to-end check of everything this script built, WITHOUT a model: the driver
# prints its usage line and exits when given too few arguments, which only
# happens if the binary linked and the rpath resolved libllama at load time.
# Verifying an actual inference needs a ~12 GB download, so that last mile is
# documented below rather than run here.
say "verifying the build (no model needed)"
if out="$(./csrc/stream_run 2>&1)"; case "$out" in *"usage:"*) true;; *) false;; esac; then
  say "  driver runs and resolves libllama: ok"
else
  die "driver built but will not start — check the rpath in scripts/build_driver.sh
     output was: ${out:-<empty>}"
fi

# --------------------------------------------------------------- python env ---
if [ ! -x .venv/bin/python3 ]; then
  say "creating .venv"
  python3 -m venv .venv
fi
say "installing Python deps (streamlit, psutil)"
./.venv/bin/python3 -m pip install --quiet --upgrade pip
./.venv/bin/python3 -m pip install --quiet streamlit psutil

# ------------------------------------------------------------------- done -----
cat <<'EOF'

sluice is installed and the engine has been verified to start. No model has been
downloaded yet — on purpose, and that means the one thing this script could NOT
verify is an actual inference: that needs a ~12 GB file. The first `run`/`ui`
below is therefore the real end-to-end test.

  ./cli/sluice estimate gpt-oss-20b   # what THIS machine will actually do, before you commit
  ./cli/sluice pull     gpt-oss-20b   # ~12 GB
  ./cli/sluice ui                     # playground at http://localhost:8501
  ./cli/sluice run      gpt-oss-20b   # terminal chat

Speed depends on your free RAM (measured, see README): ~6.1 tok/s on a quiet
16 GB M2 Air, ~4.9 with an editor open, ~1.6 with a browser and calls running.
EOF
