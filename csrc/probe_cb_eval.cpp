// M1-A: validate the cb_eval interception mechanism on STOCK llama.cpp.
// Proves, before any patch:
//   1. ffn_moe_topk-<il> nodes are observable by name via cb_eval (all layers, every token)
//   2. the ids tensor is readable at observe time ([k, n_tokens] i32)
//   3. in-place mutation of ids propagates to downstream nodes: swapping two ids
//      within a token's top-k is mathematically a no-op, so logits must be identical
//      to the untouched run. If they differ, the callback fires too late (or a copy
//      is consumed) and the M1-B design is wrong.
//
// Build (after llama.cpp is built):
//   clang++ -O3 -std=c++17 -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \
//     csrc/probe_cb_eval.cpp -Lvendor/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu \
//     -Wl,-rpath,@executable_path/../vendor/llama.cpp/build/bin -o csrc/probe_cb_eval
// Run:
//   ./csrc/probe_cb_eval <model.gguf> observe|swap "prompt text"

#include "llama.h"
#include "ggml.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct probe_state {
    bool do_swap = false;
    long n_observed = 0;
    long n_swapped = 0;
    bool printed_first = false;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    probe_state * st = (probe_state *) user_data;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk", 12) == 0 &&
                         strstr(t->name, "slots") == nullptr;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }
    st->n_observed++;
    const int64_t k = t->ne[0], n_tokens = t->ne[1];
    int32_t * ids = (int32_t *) t->data;
    if (!st->printed_first) {
        st->printed_first = true;
        printf("first observed node: %s type=%s ne=[%" PRId64 ",%" PRId64 "]\n",
               t->name, ggml_type_name(t->type), k, n_tokens);
        printf("  token0 experts:");
        for (int64_t j = 0; j < k; j++) printf(" %d", ids[j]);
        printf("\n");
    }
    if (st->do_swap && k >= 2) {
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            int32_t tmp = ids[tok * k + 0];
            ids[tok * k + 0] = ids[tok * k + 1];
            ids[tok * k + 1] = tmp;
        }
        st->n_swapped++;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> observe|swap <prompt>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    probe_state st;
    st.do_swap = strcmp(argv[2], "swap") == 0;
    const std::string prompt = argv[3];

    llama_log_set([](ggml_log_level lvl, const char * msg, void *) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(msg, stderr);
    }, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 512;
    cparams.n_batch  = 64;
    cparams.n_ubatch = 64;
    cparams.cb_eval  = cb_eval;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                           toks.data(), (int32_t) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    printf("prompt tokens: %d  mode: %s\n", n, st.do_swap ? "swap" : "observe");

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    double checksum = 0.0;
    float lmax = -1e30f; int amax = -1;
    for (int i = 0; i < n_vocab; i++) {
        checksum += fabs((double) logits[i]);
        if (logits[i] > lmax) { lmax = logits[i]; amax = i; }
    }
    printf("observed_topk_nodes=%ld swapped=%ld\n", st.n_observed, st.n_swapped);
    printf("logits: checksum=%.6f argmax=%d max=%.6f\n", checksum, amax, lmax);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
