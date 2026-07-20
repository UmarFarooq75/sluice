#!/bin/bash
# Canonical driver build. -lobjc -framework Foundation: therm_state() (D10)
# reads NSProcessInfo.thermalState via the objc runtime.
set -e
cd "$(dirname "$0")/.."
clang++ -O3 -std=c++17 \
    -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include -Ivendor/llama.cpp/src \
    csrc/stream_run.cpp \
    -Lvendor/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu \
    -lobjc -framework Foundation \
    -Wl,-rpath,"$PWD/vendor/llama.cpp/build/bin" \
    -o csrc/stream_run
echo "csrc/stream_run built"
