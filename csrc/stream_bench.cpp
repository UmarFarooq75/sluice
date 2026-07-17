// M0: compiled expert-streaming core benchmark.
// Replays a REAL routing trace (traces/demand_code.bin) against the physical
// expert_store/ with F_NOCACHE reads, per-layer LRU cache, and N I/O worker
// threads fetching a token's missing experts in parallel (queue depth).
// Reports effective bandwidth and I/O-limited tok/s, to compare against the
// Python PoC (which reached only ~50-65% of SSD bandwidth) and the SSD
// microbench ceiling (~2.4-2.5 GB/s at this granularity).
//
// Build: clang++ -O3 -std=c++17 -pthread csrc/stream_bench.cpp -o csrc/stream_bench
// Run:   ./csrc/stream_bench <cache_capacity_per_layer> [n_io_threads]

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <list>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

static const int N_LAYERS = 16, N_EXPERTS = 64;

struct LRU {
  int capacity;
  std::list<int> order;                       // front = most recent
  std::unordered_map<int, std::list<int>::iterator> pos;
  bool touch(int e) {                          // returns true on hit
    auto it = pos.find(e);
    if (it != pos.end()) { order.erase(it->second); order.push_front(e); pos[e] = order.begin(); return true; }
    return false;
  }
  void insert(int e) {
    if (capacity <= 0) return;
    if ((int)pos.size() >= capacity) { pos.erase(order.back()); order.pop_back(); }
    order.push_front(e); pos[e] = order.begin();
  }
};

static size_t read_expert(const std::string& path, std::vector<char>& buf) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) { perror(path.c_str()); exit(1); }
#ifdef F_NOCACHE
  fcntl(fd, F_NOCACHE, 1);
#endif
  struct stat st; fstat(fd, &st);
  if ((size_t)st.st_size > buf.size()) buf.resize(st.st_size);
  size_t total = 0;
  while (total < (size_t)st.st_size) {
    ssize_t r = pread(fd, buf.data() + total, st.st_size - total, total);
    if (r <= 0) break;
    total += r;
  }
  close(fd);
  return total;
}

int main(int argc, char** argv) {
  int capacity = argc > 1 ? atoi(argv[1]) : 16;
  int n_threads = argc > 2 ? atoi(argv[2]) : 6;

  FILE* f = fopen("traces/demand_code.bin", "rb");
  if (!f) { fprintf(stderr, "run the exporter first\n"); return 1; }
  uint32_t hdr[3];
  if (fread(hdr, 4, 3, f) != 3) return 1;
  uint32_t n_tok = hdr[0], n_layers = hdr[1], k = hdr[2];
  std::vector<uint16_t> demand(n_tok * n_layers * k);
  if (fread(demand.data(), 2, demand.size(), f) != demand.size()) return 1;
  fclose(f);

  std::vector<LRU> cache(N_LAYERS);
  for (auto& c : cache) c.capacity = capacity;

  // warmup: fill caches from the first 12 tokens (uncounted, like the PoC)
  uint32_t warm = 12 < n_tok ? 12 : 0;
  char pathbuf[256];
  std::vector<char> tmp(4 << 20);
  for (uint32_t t = 0; t < warm; t++)
    for (uint32_t l = 0; l < n_layers; l++)
      for (uint32_t j = 0; j < k; j++) {
        int e = demand[(t * n_layers + l) * k + j];
        if (!cache[l].touch(e)) cache[l].insert(e);
      }

  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> misses{0};
  uint64_t uses = 0;
  auto t0 = std::chrono::steady_clock::now();

  std::vector<std::thread> pool;
  std::vector<int> pending;
  for (uint32_t t = warm; t < n_tok; t++) {
    for (uint32_t l = 0; l < n_layers; l++) {
      pending.clear();
      for (uint32_t j = 0; j < k; j++) {
        int e = demand[(t * n_layers + l) * k + j];
        uses++;
        if (cache[l].touch(e)) continue;
        pending.push_back(e);
        cache[l].insert(e);
      }
      if (pending.empty()) continue;
      misses += pending.size();
      // fetch this layer's missing experts with up to n_threads parallel reads
      std::atomic<size_t> next{0};
      int nt = (int)pending.size() < n_threads ? (int)pending.size() : n_threads;
      pool.clear();
      for (int w = 0; w < nt; w++)
        pool.emplace_back([&, l]() {
          std::vector<char> buf(4 << 20);
          size_t i;
          while ((i = next.fetch_add(1)) < pending.size()) {
            char p[256];
            snprintf(p, sizeof(p), "expert_store/L%02u_E%02d.npz", l, pending[i]);
            bytes += read_expert(p, buf);
          }
        });
      for (auto& th : pool) th.join();
    }
  }
  auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double mb = bytes / 1e6;
  uint32_t counted = n_tok - warm;
  printf("cache=%d threads=%d tokens=%u\n", capacity, n_threads, counted);
  printf("misses/token=%.1f hit_rate=%.3f\n", (double)misses / counted, 1.0 - (double)misses / uses);
  printf("MB_read/token=%.1f effective_bandwidth=%.0f MB/s\n", mb / counted, mb / dt);
  printf("io_limited_tok_s=%.2f\n", counted / dt);
  return 0;
}
