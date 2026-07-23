#!/bin/bash
# Canonical driver build. -lobjc -framework Foundation: therm_state() (D10)
# reads NSProcessInfo.thermalState via the objc runtime.
set -e
cd "$(dirname "$0")/.."
# OUT lets a caller compile somewhere else. scripts/gate.sh uses it to verify the
# engine still builds WITHOUT overwriting csrc/stream_run — a detached experiment
# may be armed against that exact path and would otherwise exec a binary being
# rewritten underneath it.
OUT="${OUT:-csrc/stream_run}"
CXX="${CXX:-clang++}"
command -v "$CXX" >/dev/null 2>&1 || CXX=g++
PLATFORM_FLAGS=()
if [ "$(uname -s)" = "Darwin" ]; then
    # therm_state() (D10) reads NSProcessInfo.thermalState via the objc runtime
    PLATFORM_FLAGS=(-lobjc -framework Foundation)
else
    PLATFORM_FLAGS=(-pthread)
fi
"$CXX" -O3 -std=c++17 \
    -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include -Ivendor/llama.cpp/src \
    csrc/stream_run.cpp \
    -Lvendor/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu \
    "${PLATFORM_FLAGS[@]}" \
    -Wl,-rpath,"$PWD/vendor/llama.cpp/build/bin" \
    -o "$OUT"
echo "$OUT built"
