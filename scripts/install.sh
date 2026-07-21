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
for tool in git cmake python3; do
  command -v "$tool" >/dev/null 2>&1 || die "missing prerequisite: $tool"
done
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
  cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
  cmake --build vendor/llama.cpp/build --config Release -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
else
  say "llama.cpp already built — skipping"
fi

# ------------------------------------------------------------------ driver ----
say "building the streaming driver"
bash scripts/build_driver.sh

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

sluice is installed. No model has been downloaded yet — on purpose.

  ./cli/sluice estimate gpt-oss-20b   # what THIS machine will actually do, before you commit
  ./cli/sluice pull     gpt-oss-20b   # ~12 GB
  ./cli/sluice ui                     # playground at http://localhost:8501
  ./cli/sluice run      gpt-oss-20b   # terminal chat

Speed depends on your free RAM (measured, see README): ~6.1 tok/s on a quiet
16 GB M2 Air, ~4.9 with an editor open, ~1.6 with a browser and calls running.
EOF
