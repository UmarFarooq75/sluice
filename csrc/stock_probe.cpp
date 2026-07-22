// RC4-STOCK — the attribution leg. Does chunk-alignment sensitivity exist in
// PRISTINE llama.cpp, or did our fork introduce it?
//
// Stock llama.cpp has no logits_hash, so this is the smallest instrument that can
// ask the question at all: load a model, tokenize a prompt, prefill it as ONE batch
// with a given n_ubatch, and print an FNV-1a hash of the final logits row. Same
// digest function as csrc/stream_run.cpp, so the two are directly comparable.
//
// NOTHING from llmstream is linked or referenced. Build it against a PRISTINE
// b10064 tree and the answer is about upstream. Build it against our vendor tree
// and the difference between the two builds is our patch — which is exactly the
// attribution the directive asks for:
//
//   pristine splits the same way  -> inherited upstream numerics. E39 / E41b gate 1
//                                    reclassify as a BACKEND PROPERTY, and our gates
//                                    remain sound as config-pinned guarantees.
//   pristine does NOT split       -> ours. Localize within patches/llmstream.patch.
//
//   stock_probe <model.gguf> <prompt-file> <n_ubatch> [n_ctx]
//
// Prints: n_tokens, n_ubatch, and logits_hash. Greedy/none — it does not generate.
#include "llama.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * data, size_t n, uint64_t h) {
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt-file> <n_ubatch> [n_ctx]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int n_ubatch = atoi(argv[3]);
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 4096;

    std::string prompt;
    {
        FILE * f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) prompt.append(buf, n);
        fclose(f);
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;                  // plain CPU path, as directed
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) n_ctx;
    cparams.n_batch   = (uint32_t) n_ctx;      // whole prompt submitted as one batch;
    cparams.n_ubatch  = (uint32_t) n_ubatch;   // llama.cpp splits it into n_ubatch chunks
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    std::vector<llama_token> toks(prompt.size() + 64);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                           toks.data(), (int32_t) toks.size(), true, true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    const uint64_t h = fnv1a(logits, sizeof(float) * n_vocab, 0xcbf29ce484222325ULL);

    // also report argmax: bit-equality and argmax stability are different questions
    int best = 0; float bv = -1e30f;
    for (int i = 0; i < n_vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }

    printf("n_tokens=%d n_ubatch=%d n_ctx=%d\n", n, n_ubatch, n_ctx);
    printf("argmax=%d\n", best);
    printf("logits_hash=%016" PRIx64 "\n", h);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
