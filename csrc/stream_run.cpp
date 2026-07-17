// M1/M2 streaming driver for the llmstream fork of llama.cpp.
//
// M1: slot-backed expert tensors filled on demand from the GGUF's per-expert
// extents (pread+F_NOCACHE), router ids rewritten expert->slot in place via
// cb_eval on the "ffn_moe_topk_slots" nodes. Logit-equivalence gate proved
// bit-identical output vs the resident model.
//
// M2a: persistent I/O worker pool + lookahead prefetch. The fork emits
// "llmstream_look-<il>" = next layer's router logits computed on this layer's
// pre-FFN state (83.9% top-k recall measured in phase 0). Predicted-missing
// experts are fetched asynchronously while the current layer computes; the
// demand path waits on in-flight fetches instead of re-reading. Prefetch obeys
// the phase-0 substitution law: only into an idle queue, top-k only.
//
// Tensor sets are discovered per layer from GGUF metadata: merged
// ffn_gate_up_exps (Qwen3.5/3.6 family) or separate gate+up (OLMoE), plus down.
//
// Modes/env:
//   LLMSTREAM_SLOTS=N     slots per layer (fork reads it too); unset = resident
//   LLMSTREAM_PREFETCH=0  disable lookahead prefetch (default on when slots>0)
//   LLMSTREAM_NO_REPACK   disable weight repacking (required for bit-exact gate)
//   LLMSTREAM_OBSERVE     observe plain topk nodes (debug)
//   LLMSTREAM_DEBUG_HASH  hash all ffn_moe_* intermediates (debug)
//   LLMSTREAM_HASH_FETCH  hash every fetched extent (debug)
//   LLMSTREAM_IDENTITY    slot = expert id (debug; needs slots == n_expert)
//
// Build:
//   clang++ -O3 -std=c++17 -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \
//     -Ivendor/llama.cpp/src csrc/stream_run.cpp -Lvendor/llama.cpp/build/bin \
//     -lllama -lggml -lggml-base -lggml-cpu \
//     -Wl,-rpath,/Users/umarfarooq/Desktop/research/vendor/llama.cpp/build/bin \
//     -o csrc/stream_run
// Run:
//   ./csrc/stream_run <model.gguf> <n_gen> <prompt> [n_ubatch]

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "llmstream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct tensor_extent {
    ggml_tensor * slot_t = nullptr;
    size_t file_off = 0; // expert 0
    size_t stride   = 0; // bytes per expert
};

struct layer_cache {
    std::vector<tensor_extent> ext;       // 2 (merged) or 3 (separate) tensors
    std::list<int> lru;                   // front = most recent
    std::unordered_map<int, std::list<int>::iterator> lru_pos;
    std::unordered_map<int, int> slot_of; // expert -> slot
    std::unordered_set<int> in_flight;    // experts with pending fetches
    int n_slots = 0;
    int next_free = 0;
};

struct stream_state;

struct io_pool {
    struct job { int il, e, slot; bool prefetch; };
    std::deque<job> q;
    std::mutex m;
    std::condition_variable cv_work;   // workers wait for jobs
    std::condition_variable cv_done;   // eval thread waits for completions
    std::vector<std::thread> workers;
    bool stop = false;
    stream_state * st = nullptr;
    size_t queued() { std::lock_guard<std::mutex> l(m); return q.size(); }
};

struct stream_state {
    int fd = -1;
    int n_layers = 0;
    bool prefetch_on = true;
    std::vector<layer_cache> cache;
    io_pool pool;
    // counters (eval thread except bytes)
    std::atomic<uint64_t> bytes{0};
    uint64_t uses = 0, misses = 0, prefetch_hits = 0, prefetch_issued = 0, cb_calls = 0;
    uint64_t uses_prefill = 0, misses_prefill = 0;
    double stall_s = 0.0;
    bool in_decode = false;
    int top_k = 0;         // learned from first topk node
    double hit_ema = 0.0;  // rolling demand hit rate; prefetch active only while cold
    float margin = 0.0f;   // LLMSTREAM_MARGIN: mask non-resident experts within
                           // margin of the weakest resident pick (finding 11);
                           // 0 = off = bit-exact
    uint64_t margin_masked = 0;
};

static void fetch_one(stream_state & st, const io_pool::job & j) {
    layer_cache & lc = st.cache[j.il];
    for (size_t t = 0; t < lc.ext.size(); t++) {
        const tensor_extent & ex = lc.ext[t];
        size_t sz  = ex.stride;
        size_t off = ex.file_off + (size_t) j.e * sz;
        char * dst = (char *) ex.slot_t->data + (size_t) j.slot * ex.slot_t->nb[2];
        size_t done = 0;
        while (done < sz) {
            ssize_t r = pread(st.fd, dst + done, sz - done, off + done);
            if (r <= 0) { fprintf(stderr, "pread failed L%d E%d t%zu\n", j.il, j.e, t); exit(1); }
            done += r;
        }
        st.bytes += sz;
        static const bool hash_fetch = getenv("LLMSTREAM_HASH_FETCH") != nullptr;
        if (hash_fetch) {
            uint64_t h = 0xcbf29ce484222325ULL;
            for (size_t b = 0; b < sz; b++) { h ^= (uint8_t) dst[b]; h *= 0x100000001b3ULL; }
            printf("fetch L%d t%zu E%d %016llx\n", j.il, t, j.e, (unsigned long long) h);
        }
    }
}

static void pool_worker(io_pool * p) {
    for (;;) {
        io_pool::job j;
        {
            std::unique_lock<std::mutex> l(p->m);
            p->cv_work.wait(l, [p] { return p->stop || !p->q.empty(); });
            if (p->stop && p->q.empty()) return;
            j = p->q.front();
            p->q.pop_front();
        }
        fetch_one(*p->st, j);
        {
            std::lock_guard<std::mutex> l(p->m);
            p->st->cache[j.il].in_flight.erase(j.e);
        }
        p->cv_done.notify_all();
    }
}

// assign a slot for expert e in layer il; caller is the eval thread.
// returns -1 if no evictable slot (all resident slots needed or in flight).
static int assign_slot(layer_cache & lc, int e, const std::unordered_set<int> * needed) {
    static const bool identity = getenv("LLMSTREAM_IDENTITY") != nullptr;
    if (identity) return e;
    if (lc.next_free < lc.n_slots) return lc.next_free++;
    // evict least-recent expert that is neither needed now nor in flight
    for (auto it = lc.lru.rbegin(); it != lc.lru.rend(); ++it) {
        int victim = *it;
        if (needed && needed->count(victim)) continue;
        if (lc.in_flight.count(victim)) continue;
        int slot = lc.slot_of[victim];
        lc.slot_of.erase(victim);
        lc.lru.erase(lc.lru_pos[victim]);
        lc.lru_pos.erase(victim);
        return slot;
    }
    return -1;
}

static void lru_touch(layer_cache & lc, int e) {
    auto pit = lc.lru_pos.find(e);
    if (pit != lc.lru_pos.end()) {
        if (pit->second != lc.lru.begin()) {
            lc.lru.erase(pit->second);
            lc.lru.push_front(e);
            lc.lru_pos[e] = lc.lru.begin();
        }
    } else {
        lc.lru.push_front(e);
        lc.lru_pos[e] = lc.lru.begin();
    }
}

static uint64_t fnv1a(const void * data, size_t n, uint64_t h) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    stream_state * st = (stream_state *) user_data;
    const bool is_slots = strncmp(t->name, "ffn_moe_topk_slots", 18) == 0;
    const bool is_look  = strncmp(t->name, "llmstream_look", 14) == 0;
    const bool is_probs = strncmp(t->name, "ffn_moe_probs", 13) == 0 && strchr(t->name, ' ') == nullptr;
    static const bool observe_topk = getenv("LLMSTREAM_OBSERVE") != nullptr;
    static const bool dbg = getenv("LLMSTREAM_DEBUG_HASH") != nullptr;
    if (ask) return is_slots || is_look
        || (st->margin > 0.0f && is_probs)
        || (observe_topk && strncmp(t->name, "ffn_moe_topk", 12) == 0)
        || (dbg && strncmp(t->name, "ffn_moe_", 8) == 0);

    if (st->margin > 0.0f && is_probs && !st->cache.empty()) {
        // margin-gated cache-aware routing (finding 11): before top-k, zero out
        // any non-resident expert that does not beat the k-th best RESIDENT
        // prob by at least margin. selection, gate weights and normalization
        // all read these probs downstream, so the substitution stays coherent.
        const char * dash = strrchr(t->name, '-');
        const int il = dash ? atoi(dash + 1) : -1;
        if (il >= 0 && il < st->n_layers && st->top_k > 0) {
            layer_cache & lc = st->cache[il];
            const int64_t n_expert = t->ne[0], n_tokens = t->ne[1];
            float * probs = (float *) t->data;
            std::vector<float> res;
            for (int64_t tok = 0; tok < n_tokens; tok++) {
                float * p = probs + tok * n_expert;
                res.clear();
                for (auto & [e, s] : lc.slot_of) res.push_back(p[e]);
                if ((int) res.size() < st->top_k) continue; // cache too cold to restrict
                std::nth_element(res.begin(), res.begin() + st->top_k - 1, res.end(), std::greater<float>());
                const float kth_res = res[st->top_k - 1];
                for (int64_t e = 0; e < n_expert; e++) {
                    if (p[e] > 0.0f && !lc.slot_of.count((int) e) && p[e] < kth_res + st->margin) {
                        p[e] = 0.0f;
                        st->margin_masked++;
                    }
                }
            }
        }
        return true;
    }

    if (!is_slots && !is_look) {
        if (dbg) {
            uint64_t h = fnv1a(t->data, ggml_nbytes(t), 0xcbf29ce484222325ULL);
            printf("dbg %-28s %016llx\n", t->name, (unsigned long long) h);
        }
        return true;
    }
    if (st->cache.empty()) return true; // dup-only debug mode

    const char * dash = strrchr(t->name, '-');
    const int il = dash ? atoi(dash + 1) : -1;

    if (is_look) {
        // prefetch for layer il+1 from its router logits on layer il's state.
        // finding 20 coordination rules: idle I/O only, confident predictions
        // only, bounded in-flight so demand fetches never queue behind us
        const int nl = il + 1;
        if (!st->prefetch_on || st->top_k == 0 || nl >= st->n_layers) return true;
        // phase-0 measurement: prefetch pays at cold/small caches and is pure
        // contention once LRU is warm - it substitutes for, not stacks with, it
        if (st->hit_ema > 0.80) return true;
        if (st->pool.queued() > 0) return true;
        layer_cache & lc = st->cache[nl];
        const int64_t n_expert = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        const float * logits = (const float *) t->data;
        const int k = st->top_k;
        std::vector<std::pair<float, int>> pred; // (confidence, expert)
        std::vector<int> idx(n_expert);
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            const float * l = logits + tok * n_expert;
            for (int64_t i = 0; i < n_expert; i++) idx[i] = (int) i;
            std::partial_sort(idx.begin(), idx.begin() + k + 1, idx.end(),
                              [l](int a, int b) { return l[a] > l[b]; });
            // confidence = margin over the first excluded expert; keep only
            // predictions clearly inside the top-k (the uncertain boundary is
            // exactly what LRU already fails on - fetching it wastes bandwidth)
            const float span = l[idx[0]] - l[idx[k]] + 1e-9f;
            for (int j = 0; j < k; j++) {
                const float conf = (l[idx[j]] - l[idx[k]]) / span;
                if (conf > 0.15f) pred.push_back({conf, idx[j]});
            }
        }
        std::sort(pred.begin(), pred.end(), [](auto & a, auto & b) { return a.first > b.first; });
        std::lock_guard<std::mutex> l(st->pool.m);
        int budget = 3; // max prefetches issued per look node
        for (auto & [conf, e] : pred) {
            if (budget == 0) break;
            if (lc.slot_of.count(e) || lc.in_flight.count(e)) continue;
            int slot = assign_slot(lc, e, nullptr);
            if (slot < 0) break;
            lc.slot_of[e] = slot;
            lru_touch(lc, e);
            lc.in_flight.insert(e);
            st->pool.q.push_back({nl, e, slot, true});
            st->prefetch_issued++;
            budget--;
        }
        st->pool.cv_work.notify_all();
        return true;
    }

    // demand path: ffn_moe_topk_slots-<il>
    st->cb_calls++;
    const int64_t k = t->ne[0], n_tokens = t->ne[1];
    if (st->top_k == 0) st->top_k = (int) k;
    int32_t * ids = (int32_t *) t->data;
    layer_cache & lc = st->cache[il];

    std::unordered_set<int> need;
    for (int64_t i = 0; i < k * n_tokens; i++) need.insert(ids[i]);
    if ((int) need.size() > lc.n_slots) {
        fprintf(stderr, "ubatch expert union %zu > n_slots %d at layer %d - lower n_ubatch\n",
                need.size(), lc.n_slots, il);
        exit(1);
    }

    bool waited_any = false;
    {
        std::lock_guard<std::mutex> l(st->pool.m);
        for (int e : need) {
            if (st->in_decode) st->uses++; else st->uses_prefill++;
            const bool hit = lc.slot_of.count(e) && !lc.in_flight.count(e);
            st->hit_ema = 0.995 * st->hit_ema + (hit ? 0.005 : 0.0);
            if (lc.slot_of.count(e)) {
                if (lc.in_flight.count(e)) { st->prefetch_hits++; waited_any = true; }
                continue;
            }
            if (st->in_decode) st->misses++; else st->misses_prefill++;
            int slot = assign_slot(lc, e, &need);
            if (slot < 0) { fprintf(stderr, "no evictable slot L%d\n", il); exit(1); }
            lc.slot_of[e] = slot;
            lc.in_flight.insert(e);
            st->pool.q.push_front({il, e, slot, false}); // demand outranks prefetch
            waited_any = true;
        }
        st->pool.cv_work.notify_all();
    }

    if (waited_any) {
        auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> l(st->pool.m);
        st->pool.cv_done.wait(l, [&] {
            for (int e : need) if (lc.in_flight.count(e)) return false;
            return true;
        });
        st->stall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    for (int64_t i = 0; i < k * n_tokens; i++) {
        int e = ids[i];
        lru_touch(lc, e);
        ids[i] = lc.slot_of[e];
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <n_gen> <prompt> [n_ubatch]\n", argv[0]);
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
    {
        const char * pf = getenv("LLMSTREAM_PREFETCH");
        st.prefetch_on = !(pf && atoi(pf) == 0);
        const char * mg = getenv("LLMSTREAM_MARGIN");
        st.margin = mg ? (float) atof(mg) : 0.0f;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // repacked (interleaved) weight layouts use different gemm kernels than the
    // plain vec_dot path slot tensors take; disable for bit-exact comparisons
    if (getenv("LLMSTREAM_NO_REPACK")) mparams.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    if (n_slots > 0) {
        ggml_context * meta_ctx = nullptr;
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
        gguf_context * g = gguf_init_from_file(model_path, gp);
        if (!g) { fprintf(stderr, "gguf meta parse failed\n"); return 1; }
        const size_t data_off = gguf_get_data_offset(g);

        auto extent_for = [&](const char * name, tensor_extent & ex) -> bool {
            int64_t ti = gguf_find_tensor(g, name);
            if (ti < 0) return false;
            ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
            ex.file_off = data_off + gguf_get_tensor_offset(g, ti);
            ex.stride   = meta->nb[2];
            ex.slot_t   = llmstream_get_tensor(name);
            if (!ex.slot_t) { fprintf(stderr, "no slot tensor %s\n", name); exit(1); }
            if (ex.slot_t->nb[2] != ex.stride) { fprintf(stderr, "stride mismatch %s\n", name); exit(1); }
            return true;
        };

        for (int il = 0; ; il++) {
            char name[128];
            layer_cache lc;
            tensor_extent ex;
            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", il);
            if (!extent_for(name, ex)) { st.n_layers = il; break; }
            lc.ext.push_back(ex);
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.weight", il);
            if (extent_for(name, ex)) {
                lc.ext.push_back(ex);
            } else {
                snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", il);
                if (!extent_for(name, ex)) { fprintf(stderr, "no gate tensors L%d\n", il); return 1; }
                lc.ext.push_back(ex);
                snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
                if (!extent_for(name, ex)) { fprintf(stderr, "no up tensor L%d\n", il); return 1; }
                lc.ext.push_back(ex);
            }
            lc.n_slots = (int) n_slots;
            st.cache.push_back(std::move(lc));
        }
        gguf_free(g);
        ggml_free(meta_ctx);

        st.fd = open(model_path, O_RDONLY);
        if (st.fd < 0) { perror("open gguf"); return 1; }
#ifdef F_NOCACHE
        fcntl(st.fd, F_NOCACHE, 1);
#endif
        double per_exp = 0;
        for (auto & ex : st.cache[0].ext) per_exp += ex.stride;
        printf("llmstream: %d layers, %" PRId64 " slots/layer, %zu tensors/expert, %.2f MB/expert, prefetch=%s\n",
               st.n_layers, n_slots, st.cache[0].ext.size(), per_exp / 1e6,
               st.prefetch_on ? "on" : "off");

        st.pool.st = &st;
        for (int w = 0; w < 6; w++) st.pool.workers.emplace_back(pool_worker, &st.pool);
    }

    const int n_ubatch = argc > 4 ? atoi(argv[4]) : 1;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 1024;
    cparams.n_batch  = 512;
    cparams.n_ubatch = n_ubatch;
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

    auto t0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();

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

    printf("mode=%s prompt_toks=%d generated=%d n_ubatch=%d\n",
           n_slots > 0 ? "streamed" : "resident", n, generated, n_ubatch);
    printf("prefill: %.2f s (%.2f tok/s)\n", dt_prefill, n / dt_prefill);
    printf("decode:  %.2f s (%.2f tok/s)\n", dt_decode, generated / dt_decode);
    if (n_slots > 0) {
        const double mb = st.bytes / 1e6;
        printf("io: cb_calls=%" PRIu64 " decode_uses=%" PRIu64 " decode_misses=%" PRIu64 " (hit %.3f) prefill_uses=%" PRIu64 " prefill_misses=%" PRIu64 "\n",
               st.cb_calls, st.uses, st.misses, st.uses ? 1.0 - (double) st.misses / st.uses : 0.0,
               st.uses_prefill, st.misses_prefill);
        printf("io: prefetch_issued=%" PRIu64 " prefetch_hits=%" PRIu64 " stall=%.2f s total_read=%.1f MB avg_bw=%.0f MB/s\n",
               st.prefetch_issued, st.prefetch_hits, st.stall_s, mb, mb / (dt_prefill + dt_decode));
        if (st.margin > 0.0f) {
            printf("io: margin=%.3f masked=%" PRIu64 "\n", st.margin, st.margin_masked);
        }
    }
    printf("logits_hash=%016" PRIx64 "\n", hash);
    printf("text: %s\n", out.c_str());

    if (n_slots > 0) {
        {
            std::lock_guard<std::mutex> l(st.pool.m);
            st.pool.stop = true;
        }
        st.pool.cv_work.notify_all();
        for (auto & w : st.pool.workers) w.join();
    }
    llama_free(ctx);
    llama_model_free(model);
    if (st.fd >= 0) close(st.fd);
    return 0;
}
