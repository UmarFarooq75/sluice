// M1 streaming driver for the llmstream fork of llama.cpp.
//
// With LLMSTREAM_SLOTS=N the patched libllama creates each layer's expert
// tensors as N-slot caches (no data loaded). This driver:
//   1. reads each expert's file extents straight from the GGUF (expert = one
//      contiguous slice per tensor: off + e*nb[2], no repacked store needed)
//   2. installs cb_eval; on every "ffn_moe_topk_slots-<il>" node it takes the
//      selected expert ids, fetches missing experts from SSD (pread+F_NOCACHE,
//      parallel) into LRU-assigned slots, and rewrites ids -> slot ids in place
//   3. runs prefill + greedy decode, hashing the logits of every generated
//      position (FNV-1a over raw bytes) so runs can be compared bit-exactly
//
// With LLMSTREAM_SLOTS unset the same binary is the resident reference.
//
// Build:
//   clang++ -O3 -std=c++17 -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \
//     -Ivendor/llama.cpp/src csrc/stream_run.cpp -Lvendor/llama.cpp/build/bin \
//     -lllama -lggml -lggml-base -lggml-cpu \
//     -Wl,-rpath,/Users/umarfarooq/Desktop/research/vendor/llama.cpp/build/bin \
//     -o csrc/stream_run
// Run:
//   ./csrc/stream_run <model.gguf> <n_gen> "prompt" [n_ubatch]           # reference
//   LLMSTREAM_SLOTS=32 ./csrc/stream_run <model.gguf> <n_gen> "prompt" [n_ubatch]  # streamed
// The logit-equivalence gate compares runs at the SAME n_ubatch (float
// accumulation order depends on it); default 1 in both modes.

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "llmstream.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <list>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct expert_extents {
    ggml_tensor * slot_t[3] = {nullptr, nullptr, nullptr}; // up, gate, down slot tensors
    size_t file_off[3] = {0, 0, 0};                        // file offset of expert 0
    size_t stride[3]   = {0, 0, 0};                        // bytes per expert (= nb[2])
};

struct layer_cache {
    std::list<int> lru; // front = most recent
    std::unordered_map<int, std::list<int>::iterator> lru_pos;
    std::unordered_map<int, int> slot_of; // expert -> slot
    std::vector<int> expert_of;           // slot -> expert (-1 empty)
    int n_slots = 0;
};

struct stream_state {
    int fd = -1;
    int n_layers = 0;
    int n_threads_io = 4;
    std::vector<expert_extents> ext;  // per layer
    std::vector<layer_cache> cache;   // per layer
    // counters
    std::atomic<uint64_t> bytes{0};
    uint64_t uses = 0, misses = 0, cb_calls = 0;
    bool counting = false; // false during prefill warm phase if desired
    uint64_t uses_prefill = 0, misses_prefill = 0;
    bool in_decode = false;
};

static void fetch_experts(stream_state & st, int il, const std::vector<std::pair<int,int>> & pending) {
    // pending: (expert, slot); 3 reads each (up/gate/down)
    struct job { int t, e, s; };
    std::vector<job> jobs;
    jobs.reserve(pending.size() * 3);
    for (auto & p : pending)
        for (int t = 0; t < 3; t++)
            jobs.push_back({t, p.first, p.second});

    std::atomic<size_t> next{0};
    int nt = (int) jobs.size() < st.n_threads_io ? (int) jobs.size() : st.n_threads_io;
    std::vector<std::thread> pool;
    for (int w = 0; w < nt; w++) {
        pool.emplace_back([&]() {
            size_t i;
            while ((i = next.fetch_add(1)) < jobs.size()) {
                const job & j = jobs[i];
                const expert_extents & ex = st.ext[il];
                size_t sz  = ex.stride[j.t];
                size_t off = ex.file_off[j.t] + (size_t) j.e * sz;
                char * dst = (char *) ex.slot_t[j.t]->data + (size_t) j.s * ex.slot_t[j.t]->nb[2];
                size_t done = 0;
                while (done < sz) {
                    ssize_t r = pread(st.fd, dst + done, sz - done, off + done);
                    if (r <= 0) { fprintf(stderr, "pread failed L%d E%d t%d\n", il, j.e, j.t); exit(1); }
                    done += r;
                }
                st.bytes += sz;
                static const bool hash_fetch = getenv("LLMSTREAM_HASH_FETCH") != nullptr;
                if (hash_fetch) {
                    uint64_t h = 0xcbf29ce484222325ULL;
                    for (size_t b = 0; b < sz; b++) { h ^= (uint8_t) dst[b]; h *= 0x100000001b3ULL; }
                    printf("fetch L%d t%d E%d %016llx\n", il, j.t, j.e, (unsigned long long) h);
                }
            }
        });
    }
    for (auto & th : pool) th.join();
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    stream_state * st = (stream_state *) user_data;
    const bool is_slots = strncmp(t->name, "ffn_moe_topk_slots", 18) == 0;
    // LLMSTREAM_OBSERVE: also observe plain topk nodes without touching them,
    // to isolate the effect of observation-induced graph splits on rounding
    static const bool observe_topk = getenv("LLMSTREAM_OBSERVE") != nullptr;
    // LLMSTREAM_DEBUG_HASH: hash every ffn_moe_* intermediate to localize divergence
    static const bool dbg = getenv("LLMSTREAM_DEBUG_HASH") != nullptr;
    if (ask) return is_slots
        || (observe_topk && strncmp(t->name, "ffn_moe_topk", 12) == 0)
        || (dbg && strncmp(t->name, "ffn_moe_", 8) == 0);
    if (!is_slots) {
        if (dbg) {
            uint64_t h = 0xcbf29ce484222325ULL;
            const uint8_t * p = (const uint8_t *) t->data;
            for (size_t i = 0; i < ggml_nbytes(t); i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
            printf("dbg %-28s %016llx\n", t->name, (unsigned long long) h);
        }
        return true;
    }

    st->cb_calls++;
    if (st->ext.empty()) return true; // dup-only debug mode: observe, don't rewrite
    const char * dash = strrchr(t->name, '-');
    const int il = dash ? atoi(dash + 1) : -1;
    if (il < 0 || il >= st->n_layers) { fprintf(stderr, "bad node name %s\n", t->name); exit(1); }

    const int64_t k = t->ne[0], n_tokens = t->ne[1];
    int32_t * ids = (int32_t *) t->data;
    layer_cache & lc = st->cache[il];

    // unique experts needed by this ubatch
    std::unordered_set<int> need;
    for (int64_t i = 0; i < k * n_tokens; i++) need.insert(ids[i]);
    if ((int) need.size() > lc.n_slots) {
        fprintf(stderr, "ubatch expert union %zu > n_slots %d at layer %d - lower n_ubatch\n",
                need.size(), lc.n_slots, il);
        exit(1);
    }

    // classify hits/misses, assign slots to misses (LRU victim not needed by this ubatch)
    std::vector<std::pair<int,int>> pending;
    for (int e : need) {
        if (st->in_decode) st->uses++; else st->uses_prefill++;
        if (lc.slot_of.count(e)) continue;
        if (st->in_decode) st->misses++; else st->misses_prefill++;
        static const bool identity = getenv("LLMSTREAM_IDENTITY") != nullptr;
        int slot;
        if (identity) {
            slot = e; // requires n_slots == n_expert; isolates buffer effects from remapping
        } else if ((int) lc.slot_of.size() < lc.n_slots) {
            slot = (int) lc.slot_of.size();
        } else {
            // evict least-recent expert not needed now
            auto it = lc.lru.end();
            do { --it; } while (need.count(*it));
            int victim = *it;
            slot = lc.slot_of[victim];
            lc.slot_of.erase(victim);
            lc.lru.erase(lc.lru_pos[victim]);
            lc.lru_pos.erase(victim);
        }
        lc.slot_of[e] = slot;
        lc.expert_of[slot] = e;
        lc.lru.push_front(e);
        lc.lru_pos[e] = lc.lru.begin();
        pending.push_back({e, slot});
    }

    if (!pending.empty()) fetch_experts(*st, il, pending);

    // rewrite ids -> slots, refresh recency
    for (int64_t i = 0; i < k * n_tokens; i++) {
        int e = ids[i];
        auto pit = lc.lru_pos.find(e);
        if (pit != lc.lru_pos.end() && pit->second != lc.lru.begin()) {
            lc.lru.erase(pit->second);
            lc.lru.push_front(e);
            lc.lru_pos[e] = lc.lru.begin();
        }
        ids[i] = lc.slot_of[e];
    }
    return true;
}

static uint64_t fnv1a(const void * data, size_t n, uint64_t h) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <n_gen> <prompt>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int n_gen = atoi(argv[2]);
    const std::string prompt = argv[3];
    const int64_t n_slots = llmstream_slots();

    llama_log_set([](ggml_log_level lvl, const char * msg, void *) {
        if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(msg, stderr);
    }, nullptr);

    stream_state st;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // repacked (interleaved) weight layouts use different gemm kernels than the
    // plain vec_dot path our slot tensors take; disable for bit-exact comparisons
    if (getenv("LLMSTREAM_NO_REPACK")) mparams.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    if (n_slots > 0) {
        // expert extents from GGUF metadata
        ggml_context * meta_ctx = nullptr;
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
        gguf_context * g = gguf_init_from_file(model_path, gp);
        if (!g) { fprintf(stderr, "gguf meta parse failed\n"); return 1; }
        const size_t data_off = gguf_get_data_offset(g);
        const char * pat[3] = {"blk.%d.ffn_up_exps.weight", "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_down_exps.weight"};
        for (int il = 0; ; il++) {
            char name[128];
            snprintf(name, sizeof(name), pat[0], il);
            if (gguf_find_tensor(g, name) < 0) { st.n_layers = il; break; }
            expert_extents ex;
            for (int t = 0; t < 3; t++) {
                snprintf(name, sizeof(name), pat[t], il);
                int64_t ti = gguf_find_tensor(g, name);
                if (ti < 0) { fprintf(stderr, "missing %s\n", name); return 1; }
                ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
                ex.file_off[t] = data_off + gguf_get_tensor_offset(g, ti);
                ex.stride[t]   = meta->nb[2];
                ex.slot_t[t]   = llmstream_get_tensor(name);
                if (!ex.slot_t[t]) { fprintf(stderr, "no slot tensor %s\n", name); return 1; }
                if (ex.slot_t[t]->nb[2] != ex.stride[t]) {
                    fprintf(stderr, "stride mismatch %s: slot %zu file %zu\n", name, ex.slot_t[t]->nb[2], ex.stride[t]);
                    return 1;
                }
            }
            st.ext.push_back(ex);
        }
        gguf_free(g);
        ggml_free(meta_ctx);

        st.cache.resize(st.n_layers);
        for (auto & lc : st.cache) { lc.n_slots = (int) n_slots; lc.expert_of.assign(n_slots, -1); }

        st.fd = open(model_path, O_RDONLY);
        if (st.fd < 0) { perror("open gguf"); return 1; }
#ifdef F_NOCACHE
        fcntl(st.fd, F_NOCACHE, 1);
#endif
        printf("llmstream: %d layers, %" PRId64 " slots/layer, expert stride %.2f MB x3\n",
               st.n_layers, n_slots, st.ext[0].stride[0] / 1e6);
    }

    llama_context_params cparams = llama_context_default_params();
    const int n_ubatch = argc > 4 ? atoi(argv[4]) : 1;
    cparams.n_ctx    = 1024;
    cparams.n_batch  = 512;
    cparams.n_ubatch = n_ubatch; // M1: per-ubatch expert union must fit slots when streaming
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
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // prefill
    auto t0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();

    // greedy decode, hash logits every step
    st.in_decode = true;
    uint64_t hash = 0xcbf29ce484222325ULL;
    std::string out;
    llama_token cur = 0;
    {
        const float * logits = llama_get_logits_ith(ctx, -1);
        hash = fnv1a(logits, sizeof(float) * n_vocab, hash);
        float best = -1e30f;
        for (int i = 0; i < n_vocab; i++) if (logits[i] > best) { best = logits[i]; cur = i; }
    }
    int generated = 0;
    for (int s = 0; s < n_gen; s++) {
        if (llama_vocab_is_eog(vocab, cur)) break;
        char piece[128];
        int pn = llama_token_to_piece(vocab, cur, piece, sizeof(piece), 0, false);
        if (pn > 0) out.append(piece, pn);
        llama_batch b = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        generated++;
        const float * logits = llama_get_logits_ith(ctx, -1);
        hash = fnv1a(logits, sizeof(float) * n_vocab, hash);
        float best = -1e30f;
        for (int i = 0; i < n_vocab; i++) if (logits[i] > best) { best = logits[i]; cur = i; }
    }
    auto t2 = std::chrono::steady_clock::now();

    const double dt_prefill = std::chrono::duration<double>(t1 - t0).count();
    const double dt_decode  = std::chrono::duration<double>(t2 - t1).count();

    printf("mode=%s prompt_toks=%d generated=%d\n", n_slots > 0 ? "streamed" : "resident", n, generated);
    printf("prefill: %.2f s (%.2f tok/s)\n", dt_prefill, n / dt_prefill);
    printf("decode:  %.2f s (%.2f tok/s)\n", dt_decode, generated / dt_decode);
    if (n_slots > 0) {
        const double mb = st.bytes / 1e6;
        printf("io: cb_calls=%" PRIu64 " decode_uses=%" PRIu64 " decode_misses=%" PRIu64 " (hit %.3f) prefill_uses=%" PRIu64 " prefill_misses=%" PRIu64 "\n",
               st.cb_calls, st.uses, st.misses, st.uses ? 1.0 - (double) st.misses / st.uses : 0.0,
               st.uses_prefill, st.misses_prefill);
        printf("io: total_read=%.1f MB avg_bw=%.0f MB/s (whole run)\n", mb, mb / (dt_prefill + dt_decode));
    }
    printf("logits_hash=%016" PRIx64 "\n", hash);
    printf("text: %s\n", out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    if (st.fd >= 0) close(st.fd);
    return 0;
}
