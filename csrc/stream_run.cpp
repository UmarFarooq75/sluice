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
// ffn_gate_up_exps (Qwen3.5/3.6 family) or separate gate+up (OLMoE), plus down,
// plus optional per-expert bias vectors (gpt-oss family) which are 2D
// {n, n_expert} extents striding on nb[1] instead of nb[2].
//
// Modes/env:
//   LLMSTREAM_SLOTS=N     slots per layer (fork reads it too); unset = resident
//   LLMSTREAM_SLOTS=auto  size the cache from this machine's available memory
//   LLMSTREAM_GUARD=0     disable the runtime memory-pressure guard
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
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <deque>
#include <fcntl.h>
#include <list>
#include <mach/mach.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct tensor_extent {
    ggml_tensor * slot_t = nullptr;
    size_t file_off    = 0; // expert 0
    size_t stride      = 0; // bytes per expert in the file
    size_t slot_stride = 0; // bytes per slot in the slot tensor
};

struct layer_cache {
    std::vector<tensor_extent> ext;       // 2 (merged) or 3 (separate) tensors
    std::list<int> lru;                   // front = most recent
    std::unordered_map<int, std::list<int>::iterator> lru_pos;
    std::unordered_map<int, int> slot_of; // expert -> slot
    std::unordered_set<int> in_flight;    // experts with pending fetches
    std::unordered_map<int, int> parts_left; // in-flight expert -> parts not yet read
    std::vector<int> free_slots;          // slots shed by the pressure guard
    int n_slots = 0;
    int next_free = 0;
};

struct stream_state;

struct io_pool {
    struct job { int il, e, slot, part; bool prefetch; };
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
    std::atomic<bool> in_decode{false};
    int top_k = 0;         // learned from first topk node
    double hit_ema = 0.0;  // rolling demand hit rate; prefetch active only while cold
    float margin = 0.0f;   // LLMSTREAM_MARGIN: mask non-resident experts within
                           // margin of the weakest resident pick (finding 11);
                           // 0 = off = bit-exact
    uint64_t margin_masked = 0;
    // routing fidelity: overlap between the true (pre-mask) top-k and the
    // top-k actually selected after masking. exact ground truth, free to read
    // because the hook holds the raw scores before it mutates them.
    uint64_t agree_hits = 0, agree_total = 0, swapped_tokens = 0;
    // I/O worker accounting: summed pread wall time across workers vs the eval
    // thread's stall tells queueing from raw device latency apart
    std::atomic<uint64_t> read_us{0}, read_calls{0};
    // E10 pressure guard: the host machine is never collateral damage. cap is
    // the live per-layer occupancy limit the monitor thread lowers under
    // memory pressure and raises back when calm.
    std::atomic<int> slot_cap{INT_MAX};
    int slot_cap_max = 0;
    std::atomic<uint64_t> cap_drops{0};
    uint64_t cap_evictions = 0;   // eval thread, under pool.m
    std::atomic<bool> mon_stop{false};
    std::thread mon;
};

// one job = one tensor extent of one expert: the 6 reads of an expert run
// in parallel across workers instead of serially on one (measured: effective
// bandwidth collapsed to 216-349 MB/s at low miss counts - a latency floor,
// ~11ms/miss, from serial preads)
static void fetch_one(stream_state & st, const io_pool::job & j) {
    layer_cache & lc = st.cache[j.il];
    {
        const size_t t = (size_t) j.part;
        const tensor_extent & ex = lc.ext[t];
        size_t sz  = ex.stride;
        size_t off = ex.file_off + (size_t) j.e * sz;
        char * dst = (char *) ex.slot_t->data + (size_t) j.slot * ex.slot_stride;
        size_t done = 0;
        auto r0 = std::chrono::steady_clock::now();
        while (done < sz) {
            ssize_t r = pread(st.fd, dst + done, sz - done, off + done);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) {
                fprintf(stderr, "pread failed L%d E%d part%zu: %s\n", j.il, j.e, t, r < 0 ? strerror(errno) : "eof");
                exit(1);
            }
            done += r;
        }
        st.read_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - r0).count();
        st.read_calls++;
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
            layer_cache & lc = p->st->cache[j.il];
            auto it = lc.parts_left.find(j.e);
            if (it != lc.parts_left.end() && --it->second <= 0) {
                lc.parts_left.erase(it);
                lc.in_flight.erase(j.e);
            }
        }
        p->cv_done.notify_all();
    }
}

// return a shed slot's pages to the OS. content correctness is unaffected:
// the expert mapping is gone, so any future use refetches over these bytes.
static uint64_t g_madv_fail = 0; // eval thread only (called under pool.m)
static void madv_free_slot(layer_cache & lc, int slot) {
    const uintptr_t ps = (uintptr_t) sysconf(_SC_PAGESIZE);
    for (auto & ex : lc.ext) {
        const uintptr_t a  = (uintptr_t) ex.slot_t->data + (uintptr_t) slot * ex.slot_stride;
        const uintptr_t pa = (a + ps - 1) & ~(ps - 1);
        const uintptr_t pb = (a + ex.slot_stride) & ~(ps - 1);
        // audit F6: on device (Metal) buffers madvise can fail or be a no-op;
        // count it so "shed" never silently means "paid speed, freed nothing"
        if (pb > pa && madvise((void *) pa, pb - pa, MADV_FREE) != 0) g_madv_fail++;
    }
}

static size_t avail_mem_bytes(void) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &cnt) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t) (vm.free_count + vm.inactive_count + vm.purgeable_count) *
           (size_t) sysconf(_SC_PAGESIZE);
}

// E10: dual trigger — the memorystatus signal jetsam kills on, OR available
// memory under a hard floor (measured: swap grew 1.5GB in stress test 3 while
// memorystatus never left "normal"; one opaque OS signal is not enough).
// warning sheds 2 slots/layer, critical/floor halves; 30s of calm earns one
// back. shed slots get MADV_FREE'd so the OS can actually reclaim the pages.
static void pressure_monitor(stream_state * st) {
    static const double floor_gb = getenv("LLMSTREAM_GUARD_FLOOR")
        ? atof(getenv("LLMSTREAM_GUARD_FLOOR")) : 1.2;
    // fault injection for the shed path: force one severe event N seconds
    // after DECODE begins (a warm cache must exist to prove eviction+MADV_FREE
    // actually runs; process-relative timers lose the race with load variance)
    static const int test_at = getenv("LLMSTREAM_GUARD_TEST_AT")
        ? atoi(getenv("LLMSTREAM_GUARD_TEST_AT")) : 0;
    std::chrono::steady_clock::time_point t_decode{};
    bool injected = false;
    int calm = 0;
    while (!st->mon_stop.load()) {
        for (int i = 0; i < 20 && !st->mon_stop.load(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        uint32_t lvl = 0;
        size_t sz = sizeof(lvl);
        if (sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &sz, nullptr, 0) != 0) lvl = 1;
        const double avail_gb = avail_mem_bytes() / 1e9;
        bool severe = lvl >= 4 || avail_gb < floor_gb;
        bool warn   = lvl >= 2 || avail_gb < floor_gb * 1.5;
        if (test_at > 0 && !injected && st->in_decode) {
            if (t_decode == std::chrono::steady_clock::time_point{}) {
                t_decode = std::chrono::steady_clock::now();
            } else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode).count() >= test_at) {
                injected = severe = warn = true;
                fprintf(stderr, "llmstream: guard TEST injection at decode+%ds\n", test_at);
            }
        }
        const int cap = st->slot_cap.load();
        if (severe || warn) {
            const int ncap = std::max(4, severe ? cap / 2 : cap - 2);
            if (ncap < cap) {
                st->slot_cap = ncap;
                st->cap_drops++;
                fprintf(stderr, "llmstream: pressure lvl=%u avail=%.1fGB -> slot cap %d\n",
                        lvl, avail_gb, ncap);
            }
            calm = 0;
        } else if (cap < st->slot_cap_max && ++calm >= 15) {
            st->slot_cap = cap + 1;
            calm = 0;
        }
    }
}

// assign a slot for expert e in layer il; caller is the eval thread.
// returns -1 if no evictable slot (all resident slots needed or in flight).
static int assign_slot(layer_cache & lc, int e, const std::unordered_set<int> * needed, int cap) {
    static const bool identity = getenv("LLMSTREAM_IDENTITY") != nullptr;
    if (identity) return e;
    if ((int) lc.slot_of.size() < cap) {
        if (!lc.free_slots.empty()) {
            const int s = lc.free_slots.back();
            lc.free_slots.pop_back();
            return s;
        }
        if (lc.next_free < lc.n_slots) return lc.next_free++;
    }
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
    const bool is_probs = strncmp(t->name, "ffn_moe_probs-", 14) == 0 && strchr(t->name, ' ') == nullptr;
    static const bool observe_topk = getenv("LLMSTREAM_OBSERVE") != nullptr;
    static const bool dbg = getenv("LLMSTREAM_DEBUG_HASH") != nullptr;
    // is_look only matters while prefetch is live; asking for it anyway ends
    // a scheduler compute range per look node - on GPU that is one extra
    // command-buffer sync per layer per token (2x the necessary syncs)
    if (ask) return is_slots || (is_look && st->prefetch_on && st->hit_ema <= 0.80)
        || (st->margin > 0.0f && is_probs)
        || (observe_topk && strncmp(t->name, "ffn_moe_topk", 12) == 0)
        || (dbg && strncmp(t->name, "ffn_moe_", 8) == 0);

    if (st->margin > 0.0f && is_probs && !st->cache.empty()) {
        // margin-gated cache-aware routing (finding 11): before top-k, mask out
        // any non-resident expert that does not beat the k-th best RESIDENT
        // score by at least margin. selection, gate weights and normalization
        // all read these probs downstream, so the substitution stays coherent.
        // masking is -inf, not 0: SOFTMAX_WEIGHT models (gpt-oss) select on raw
        // logits where 0 is not a floor; for softmax/sigmoid probs the masked
        // experts were unselectable either way, so the outcome is identical.
        const char * dash = strrchr(t->name, '-');
        const int il = dash ? atoi(dash + 1) : -1;
        if (il >= 0 && il < st->n_layers && st->top_k > 0) {
            layer_cache & lc = st->cache[il];
            const int64_t n_expert = t->ne[0], n_tokens = t->ne[1];
            float * probs = (float *) t->data;
            std::vector<float> res;
            std::vector<int> idx((size_t) n_expert);
            const int k = st->top_k;
            for (int64_t tok = 0; tok < n_tokens; tok++) {
                float * p = probs + tok * n_expert;
                res.clear();
                for (auto & [e, s] : lc.slot_of) res.push_back(p[e]);
                if ((int) res.size() < k) continue; // cache too cold to restrict
                // true top-k before masking = the routing ground truth
                for (int64_t i = 0; i < n_expert; i++) idx[i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                  [p](int a, int b) { return p[a] > p[b]; });
                std::unordered_set<int> true_topk(idx.begin(), idx.begin() + k);
                std::nth_element(res.begin(), res.begin() + k - 1, res.end(), std::greater<float>());
                const float kth_res = res[k - 1];
                for (int64_t e = 0; e < n_expert; e++) {
                    if (p[e] != -INFINITY && !lc.slot_of.count((int) e) && p[e] < kth_res + st->margin) {
                        p[e] = -INFINITY;
                        st->margin_masked++;
                    }
                }
                // top-k after masking; overlap with truth = agreement
                for (int64_t i = 0; i < n_expert; i++) idx[i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                  [p](int a, int b) { return p[a] > p[b]; });
                int overlap = 0;
                for (int i = 0; i < k; i++) overlap += true_topk.count(idx[i]) ? 1 : 0;
                st->agree_hits  += overlap;
                st->agree_total += k;
                if (overlap < k) st->swapped_tokens++;
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
        // contention once LRU is warm - it substitutes for, not stacks with, it.
        // that gate was tuned on OLMoE; PREFETCH_FORCE bypasses it so the
        // big-model regime can be measured rather than assumed.
        static const bool pf_force = getenv("LLMSTREAM_PREFETCH_FORCE") != nullptr;
        if (!pf_force && st->hit_ema > 0.80) return true;
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
            int slot = assign_slot(lc, e, nullptr, st->slot_cap.load(std::memory_order_relaxed));
            if (slot < 0) break;
            lc.slot_of[e] = slot;
            lru_touch(lc, e);
            lc.in_flight.insert(e);
            lc.parts_left[e] = (int) lc.ext.size();
            for (int part = 0; part < (int) lc.ext.size(); part++) {
                st->pool.q.push_back({nl, e, slot, part, true});
            }
            st->prefetch_issued++;
            budget--;
        }
        st->pool.cv_work.notify_all();
        return true;
    }

    // demand path: ffn_moe_topk_slots-<il>
    st->cb_calls++;
    if (il < 0 || il >= st->n_layers) return true; // dense/malformed layer: never index cache OOB
    const int64_t k = t->ne[0], n_tokens = t->ne[1];
    static int print_ids = getenv("LLMSTREAM_PRINT_IDS") ? atoi(getenv("LLMSTREAM_PRINT_IDS")) : 0;
    if (print_ids > 0) {
        print_ids--;
        printf("ids %-24s ne=[%lld,%lld]:", t->name, (long long) k, (long long) n_tokens);
        for (int64_t i = 0; i < k && i < 8; i++) printf(" %d", ((int32_t *) t->data)[i]);
        printf("\n");
    }
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
        std::unique_lock<std::mutex> l(st->pool.m);
        // E10: honor a lowered cap first - shed coldest experts and hand their
        // pages back so the OS sees relief before we add any new load
        const int cap = st->slot_cap.load(std::memory_order_relaxed);
        while ((int) lc.slot_of.size() > cap) {
            bool evicted = false;
            for (auto it = lc.lru.rbegin(); it != lc.lru.rend(); ++it) {
                const int victim = *it;
                if (need.count(victim) || lc.in_flight.count(victim)) continue;
                const int s = lc.slot_of[victim];
                lc.slot_of.erase(victim);
                lc.lru.erase(lc.lru_pos[victim]);
                lc.lru_pos.erase(victim);
                lc.free_slots.push_back(s);
                madv_free_slot(lc, s);
                st->cap_evictions++;
                evicted = true;
                break;
            }
            if (!evicted) break;
        }
        for (int e : need) {
            if (st->in_decode) st->uses++; else st->uses_prefill++;
            const bool hit = lc.slot_of.count(e) && !lc.in_flight.count(e);
            st->hit_ema = 0.995 * st->hit_ema + (hit ? 0.005 : 0.0);
            if (lc.slot_of.count(e)) {
                if (lc.in_flight.count(e)) { st->prefetch_hits++; waited_any = true; }
                continue;
            }
            if (st->in_decode) st->misses++; else st->misses_prefill++;
            int slot = assign_slot(lc, e, &need, cap);
            while (slot < 0 && !lc.in_flight.empty()) {
                // every victim is needed or in flight: wait for a fetch to
                // land, then retry - dying here would turn memory pressure
                // into an availability loss (audit F4)
                st->pool.cv_done.wait(l);
                slot = assign_slot(lc, e, &need, cap);
            }
            if (slot < 0) { fprintf(stderr, "no evictable slot L%d (slots < top_k?)\n", il); exit(1); }
            lc.slot_of[e] = slot;
            lc.in_flight.insert(e);
            lc.parts_left[e] = (int) lc.ext.size();
            for (int part = (int) lc.ext.size() - 1; part >= 0; part--) {
                st->pool.q.push_front({il, e, slot, part, false}); // demand outranks prefetch
            }
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

    // E10: LLMSTREAM_SLOTS=auto sizes the cache to THIS machine's spare
    // memory. Measured motivation: slots16 on the 16GB Air collapsed to
    // 48 MB/s effective SSD bandwidth purely from memory pressure - headroom,
    // not cache size, governs throughput. Resolve before any llmstream_slots()
    // reader (the fork reads the env at model load too).
    {
        const char * senv = getenv("LLMSTREAM_SLOTS");
        if (senv && strcmp(senv, "auto") == 0) {
            int64_t slots = 8; // conservative fallback if the meta scan fails
            ggml_context * mctx = nullptr;
            gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &mctx };
            gguf_context * g = gguf_init_from_file(model_path, gp);
            if (g) {
                size_t total_b = 0, exp_b = 0;
                int64_t n_expert = 0;
                int max_blk = -1;
                for (ggml_tensor * t = ggml_get_first_tensor(mctx); t; t = ggml_get_next_tensor(mctx, t)) {
                    total_b += ggml_nbytes(t);
                    if (strstr(t->name, "_exps.")) {
                        exp_b += ggml_nbytes(t);
                        if (t->ne[2] > 1 && t->ne[2] > n_expert) n_expert = t->ne[2];
                        int b = -1;
                        if (sscanf(t->name, "blk.%d.", &b) == 1 && b > max_blk) max_blk = b;
                    }
                }
                const int nl = max_blk + 1;
                if (nl > 0 && n_expert > 0 && exp_b > 0) {
                    const double per_slot_layer = (double) exp_b / nl / n_expert;
                    const double resident = (double) (total_b - exp_b);
                    const double reserve  = 2.0e9; // KV + compute buffers + OS breathing room
                    double budget = (double) avail_mem_bytes() * 0.80 - resident - reserve;
                    // slots on a device: the device working set is its own,
                    // usually tighter, budget (E15: 16-slot Metal configs on a
                    // 16GB machine only survived because the guard shed live)
                    const char * sd = getenv("LLMSTREAM_SLOT_DEV");
                    if (sd && *sd) {
                        ggml_backend_dev_t dev = ggml_backend_dev_by_name(sd);
                        if (!dev && strcmp(sd, "gpu") == 0) {
                            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { dev = d; break; }
                            }
                        }
                        if (dev) {
                            size_t dfree = 0, dtotal = 0;
                            ggml_backend_dev_memory(dev, &dfree, &dtotal);
                            const double dev_budget = (double) dfree * 0.80 - resident - 0.5e9;
                            if (dev_budget < budget) budget = dev_budget;
                        }
                    }
                    slots = (int64_t) (budget / (per_slot_layer * nl));
                    if (slots < 4) slots = 4;
                    if (slots > n_expert) slots = n_expert;
                    fprintf(stderr, "llmstream: auto slots=%lld (avail=%.1f GB resident=%.2f GB %.1f MB/slot-layer %d layers)\n",
                            (long long) slots, avail_mem_bytes() / 1e9, resident / 1e9, per_slot_layer / 1e6, nl);
                }
                gguf_free(g);
                ggml_free(mctx);
            }
            char sb[32];
            snprintf(sb, sizeof(sb), "%lld", (long long) slots);
            setenv("LLMSTREAM_SLOTS", sb, 1);
        }
    }
    const int64_t n_slots = llmstream_slots();

    llama_log_set([](ggml_log_level lvl, const char * msg, void *) {
        // WARN and up: a swallowed warning cost us a corrupt-run debugging
        // cycle (SLOT_DEV fallback was invisible at ERROR-only)
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(msg, stderr);
        else if (getenv("LLMSTREAM_VERBOSE") && lvl >= GGML_LOG_LEVEL_INFO) fputs(msg, stderr);
    }, nullptr);

    stream_state st;
    {
        const char * pf = getenv("LLMSTREAM_PREFETCH");
        st.prefetch_on = !(pf && atoi(pf) == 0);
        const char * mg = getenv("LLMSTREAM_MARGIN");
        st.margin = mg ? (float) atof(mg) : 0.0f;
    }

    // E10: the guard must exist BEFORE model load - stress test 3 measured
    // +1.5GB swap growth entirely inside the load window, when a late-started
    // monitor could not see it
    if (n_slots > 0) {
        st.slot_cap     = (int) n_slots;
        st.slot_cap_max = (int) n_slots;
        if (!(getenv("LLMSTREAM_GUARD") && atoi(getenv("LLMSTREAM_GUARD")) == 0)) {
            st.mon = std::thread(pressure_monitor, &st);
        }
    }

    llama_model_params mparams = llama_model_default_params();
    // LLMSTREAM_NGL: layers to offload to GPU (Metal build only). default 0 =
    // CPU; the bit-exact gate is defined on the CPU backend.
    mparams.n_gpu_layers = getenv("LLMSTREAM_NGL") ? atoi(getenv("LLMSTREAM_NGL")) : 0;
    // With device-resident slot tensors, mmap is fatal for big models: the
    // GPU path maps the ENTIRE file as one device buffer (measured: 60.4GB
    // MTL0_Mapped for gpt-oss vs a 12.7GB working set -> command-buffer OOM).
    // Without mmap only actually-loaded (non-skipped) tensors allocate.
    if (getenv("LLMSTREAM_SLOT_DEV")) mparams.use_mmap = false;
    if (n_slots > 0 && mparams.n_gpu_layers > 0 && !getenv("LLMSTREAM_SLOT_DEV")) {
        fprintf(stderr, "llmstream: NGL>0 with CPU slot tensors is a corrupt configuration "
                        "(scheduler snapshots cross-backend inputs before mid-graph fills; "
                        "measured garbage output). Set LLMSTREAM_SLOT_DEV=gpu or NGL=0.\n");
        return 1;
    }
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
            // 3D weights stride per expert on nb[2]; 2D biases on nb[1]
            const bool is_2d = meta->ne[2] == 1;
            ex.file_off = data_off + gguf_get_tensor_offset(g, ti);
            ex.stride   = is_2d ? meta->nb[1] : meta->nb[2];
            ex.slot_t   = llmstream_get_tensor(name);
            if (!ex.slot_t) { fprintf(stderr, "no slot tensor %s\n", name); exit(1); }
            ex.slot_stride = is_2d ? ex.slot_t->nb[1] : ex.slot_t->nb[2];
            if (ex.slot_stride != ex.stride) { fprintf(stderr, "stride mismatch %s\n", name); exit(1); }
            if (!ex.slot_t->data || !ex.slot_t->buffer) {
                fprintf(stderr, "slot tensor %s has no backing buffer - adapter is missing llmstream_alloc()\n", name);
                exit(1);
            }
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
            // per-expert bias vectors (gpt-oss family), streamed with the weights
            static const char * bias_fmt[] = {
                "blk.%d.ffn_gate_exps.bias", "blk.%d.ffn_up_exps.bias", "blk.%d.ffn_down_exps.bias",
            };
            for (const char * fmt : bias_fmt) {
                snprintf(name, sizeof(name), fmt, il);
                if (extent_for(name, ex)) lc.ext.push_back(ex);
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
        int n_workers = getenv("LLMSTREAM_IO_WORKERS") ? atoi(getenv("LLMSTREAM_IO_WORKERS")) : 10;
        if (n_workers < 1) n_workers = 1;
        if (n_workers > 32) n_workers = 32;
        for (int w = 0; w < n_workers; w++) st.pool.workers.emplace_back(pool_worker, &st.pool);
    }

    const int n_ubatch = argc > 4 ? atoi(argv[4]) : 1;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 1024;
    cparams.n_batch  = 512;
    cparams.n_ubatch = n_ubatch;
    cparams.cb_eval  = cb_eval;
    cparams.cb_eval_user_data = &st;
    if (getenv("LLMSTREAM_THREADS")) {
        int nt = atoi(getenv("LLMSTREAM_THREADS"));
        if (nt < 1) nt = 1;
        cparams.n_threads       = nt;
        cparams.n_threads_batch = nt;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // LLMSTREAM_CHAT=1: wrap the prompt in the model's own chat template
    // (gpt-oss is harmony-format trained; raw text prompts confound quality
    // reads with template mismatch). parse_special so template tokens survive.
    std::string ptext = prompt;
    bool chat = false;
    if (getenv("LLMSTREAM_CHAT")) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl) {
            llama_chat_message msg = { "user", prompt.c_str() };
            std::vector<char> buf(prompt.size() * 2 + 4096);
            int32_t r = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int32_t) buf.size());
            if (r > 0 && r <= (int32_t) buf.size()) { ptext.assign(buf.data(), r); chat = true; }
        }
        if (!chat) fprintf(stderr, "warn: LLMSTREAM_CHAT set but no usable template; raw prompt\n");
    }
    std::vector<llama_token> toks(ptext.size() + 64);
    int n = llama_tokenize(vocab, ptext.c_str(), (int32_t) ptext.size(),
                           toks.data(), (int32_t) toks.size(), true, chat);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // LLMSTREAM_NLL=1: teacher-forced scoring of the prompt instead of
    // generation. deterministic, sampling-free quality number: mean -log p of
    // each prompt token given its prefix, directly comparable across margins.
    if (getenv("LLMSTREAM_NLL")) {
        if (n < 8) { fprintf(stderr, "NLL mode needs a longer prompt\n"); return 1; }
        auto tt0 = std::chrono::steady_clock::now();
        double nll = 0.0, nll_tail = 0.0; int scored = 0, scored_tail = 0;
        llama_token first = toks[0];
        llama_batch b0 = llama_batch_get_one(&first, 1);
        if (llama_decode(ctx, b0) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 1; i < n; i++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            float mx = -1e30f;
            for (int v = 0; v < n_vocab; v++) if (lg[v] > mx) mx = lg[v];
            double se = 0.0;
            for (int v = 0; v < n_vocab; v++) se += exp((double) lg[v] - mx);
            const double t_nll = -((double) lg[toks[i]] - mx - log(se));
            nll += t_nll;
            scored++;
            // second half separately: by then the cache is warm, so this
            // isolates steady-state routing damage from cold-start damage
            if (i >= n / 2) { nll_tail += t_nll; scored_tail++; }
            llama_batch b = llama_batch_get_one(&toks[i], 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tt0).count();
        printf("mode=%s nll_tokens=%d avg_nll=%.5f ppl=%.4f warm_nll=%.5f warm_ppl=%.4f (%.2f tok/s)\n",
               n_slots > 0 ? "streamed" : "resident", scored, nll / scored, exp(nll / scored),
               scored_tail ? nll_tail / scored_tail : 0.0,
               scored_tail ? exp(nll_tail / scored_tail) : 0.0, (n - 1) / dt);
        if (n_slots > 0) {
            printf("io: uses=%" PRIu64 " misses=%" PRIu64 " (hit %.3f) stall=%.2f s read=%.1f MB read_work=%.2f s preads=%" PRIu64 "\n",
                   st.uses + st.uses_prefill, st.misses + st.misses_prefill,
                   1.0 - (double) (st.misses + st.misses_prefill) / (st.uses + st.uses_prefill),
                   st.stall_s, st.bytes / 1e6, st.read_us / 1e6, st.read_calls.load());
            if (st.margin > 0.0f && st.agree_total > 0) {
                printf("io: margin=%.3f router_agreement=%.4f swapped_calls=%" PRIu64 "\n",
                       st.margin, (double) st.agree_hits / st.agree_total, st.swapped_tokens);
            }
            if (st.cap_drops.load() > 0) {
                printf("io: guard pressure_drops=%" PRIu64 " cap_evictions=%" PRIu64 " final_cap=%d madv_fail=%" PRIu64 "\n",
                       st.cap_drops.load(), st.cap_evictions, st.slot_cap.load(), g_madv_fail);
            }
        }
        st.mon_stop = true;
        if (st.mon.joinable()) st.mon.join();
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
    llmstream_free();
        if (st.fd >= 0) close(st.fd);
        return 0;
    }

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
    static const bool print_toks = getenv("LLMSTREAM_PRINT_TOKS") != nullptr;
    int generated = 0;
    for (int s = 0; s < n_gen; s++) {
        if (llama_vocab_is_eog(vocab, cur)) break;
        char piece[128];
        int pn = llama_token_to_piece(vocab, cur, piece, sizeof(piece), 0, print_toks);
        if (print_toks) printf("tok %6d |%.*s|\n", cur, pn > 0 ? pn : 0, piece);
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
        // read_work is summed across the 6 pool workers (can exceed wall time);
        // per_stream_bw = bytes / that sum, i.e. what one queue depth delivers
        printf("io: read_work=%.2f s preads=%" PRIu64 " per_stream_bw=%.0f MB/s\n",
               st.read_us / 1e6, st.read_calls.load(),
               st.read_us ? mb / (st.read_us / 1e6) : 0.0);
        if (st.margin > 0.0f) {
            printf("io: margin=%.3f masked=%" PRIu64 "\n", st.margin, st.margin_masked);
            if (st.agree_total > 0) {
                printf("io: router_agreement=%.4f swapped_calls=%" PRIu64 " of=%" PRIu64 "\n",
                       (double) st.agree_hits / st.agree_total, st.swapped_tokens, st.agree_total / (uint64_t) st.top_k);
            }
        }
        if (st.cap_drops.load() > 0) {
            printf("io: guard pressure_drops=%" PRIu64 " cap_evictions=%" PRIu64 " final_cap=%d\n",
                   st.cap_drops.load(), st.cap_evictions, st.slot_cap.load());
        }
    }
    printf("logits_hash=%016" PRIx64 "\n", hash);
    printf("text: %s\n", out.c_str());

    st.mon_stop = true;
    if (st.mon.joinable()) st.mon.join();
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
    llmstream_free();
    if (st.fd >= 0) close(st.fd);
    return 0;
}
