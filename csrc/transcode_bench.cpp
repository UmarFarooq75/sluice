// E25 Gate C leg 2: dual-precision transcode cost.
// A cold expert fetched as ~2-bit (Q2_K sidecar) must land in the slot
// tensor's native MXFP4 layout: dequant Q2_K -> fp32 -> quant MXFP4.
// This prices that pipeline per expert (24.9M weights = 13.25MB MXFP4)
// on one core. Survival bar: transcode <= ~1.5ms/expert-per-core budget
// (the read-time saving measured in leg 1), assuming 2+ idle E-cores
// overlap it with I/O.
//
// Build: clang++ -O3 -std=c++17 -Ivendor/llama.cpp/ggml/include \
//   csrc/transcode_bench.cpp -Lvendor/llama.cpp/build/bin -lggml-base \
//   -lggml-cpu -lggml -Wl,-rpath,$PWD/vendor/llama.cpp/build/bin \
//   -o csrc/transcode_bench
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    // throughput bench: row length only needs the type's block multiple
    // (Q2_K: 256). gpt-oss's real row dim (2880) needs padding to 3072 in
    // the sidecar - a 6.7% byte tax, accounted in the verdict.
    const int64_t n_per_row = 2560, n_rows = 9728;      // ~24.9M weights
    const int64_t n = n_per_row * n_rows;
    std::vector<float> src(n), mid(n);
    srand(42);
    for (int64_t i = 0; i < n; i++) src[i] = (float) rand() / RAND_MAX - 0.5f;

    std::vector<uint8_t> q2(ggml_row_size(GGML_TYPE_Q2_K, n_per_row) * n_rows);
    std::vector<uint8_t> mx(ggml_row_size(GGML_TYPE_MXFP4, n_per_row) * n_rows);

    ggml_quantize_chunk(GGML_TYPE_Q2_K, src.data(), q2.data(), 0, n_rows, n_per_row, nullptr);

    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_K);
    auto t0 = std::chrono::steady_clock::now();
    tr->to_float(q2.data(), mid.data(), n);
    auto t1 = std::chrono::steady_clock::now();
    ggml_quantize_chunk(GGML_TYPE_MXFP4, mid.data(), mx.data(), 0, n_rows, n_per_row, nullptr);
    auto t2 = std::chrono::steady_clock::now();

    const double dq = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double qz = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("weights: %.1fM  (one expert equivalent)\n", n / 1e6);
    printf("dequant Q2_K -> fp32 : %7.2f ms  (%.0f MB/s of MXFP4-equiv)\n", dq, 13.25 / dq * 1e3);
    printf("quant fp32 -> MXFP4  : %7.2f ms  (%.0f MB/s of MXFP4-equiv)\n", qz, 13.25 / qz * 1e3);
    printf("TOTAL transcode      : %7.2f ms/expert on ONE core\n", dq + qz);
    printf("read saving (leg 1)  :    1.54 ms/expert -> verdict: %s\n",
           (dq + qz) / 2.0 <= 1.54 ? "SURVIVES with 2-core overlap" : "transcode-bound: needs >2 cores or dies");
    return 0;
}
