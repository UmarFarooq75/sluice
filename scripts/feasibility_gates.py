#!/usr/bin/env python3
# Zero-risk feasibility gates for the E25 candidates, run on the EXISTING
# m0 s8 routing trace (no engine changes, no model process).
# Gate A (admission filter): 2-slot victim cache with second-use promotion
#   vs plain LRU at the same total size. Build bar: >= +2 pts hit rate.
# Gate B (co-activation prefetch table): predict layer l+1's experts from
#   layer l's active set using first-half-of-trace counts, score recall on
#   the second half (honest split - the LFU lesson: train=test flatters).
#   Build bar: recall@8 >= 80% (beats the 72% online lookahead).
import re, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/trace_m0_s8.txt"
pat = re.compile(r"ids ffn_moe_topk_slots-(\d+)\s+ne=\[(\d+),1\]:((?:\s+\d+)+)")
per_layer = defaultdict(list)
for line in open(path, errors="replace"):
    m = pat.search(line)
    if m:
        per_layer[int(m.group(1))].append([int(x) for x in m.group(3).split()][: int(m.group(2))])

def sim_lru(seq, S):
    cache, order, hits, uses = set(), [], 0, 0
    for step in seq:
        for e in step:
            uses += 1
            if e in cache:
                hits += 1; order.remove(e); order.append(e)
            else:
                if len(cache) >= S:
                    for v in order:
                        if v not in step:
                            cache.discard(v); order.remove(v); break
                cache.add(e); order.append(e)
    return hits, uses

def sim_victim(seq, S, V=2):
    # main LRU (S-V) + FIFO victim/staging (V); promotion on second use
    main, order, stage, hits, uses = set(), [], [], 0, 0
    for step in seq:
        for e in step:
            uses += 1
            if e in main:
                hits += 1; order.remove(e); order.append(e)
            elif e in stage:
                hits += 1
                stage.remove(e)          # promote: proven reuse
                if len(main) >= S - V:
                    for v in order:
                        if v not in step:
                            main.discard(v); order.remove(v); break
                main.add(e); order.append(e)
            else:
                if len(stage) >= V:
                    stage.pop(0)
                stage.append(e)          # first miss lands in staging only
    return hits, uses

print("== Gate A: admission/victim cache vs LRU (same total slots) ==")
for S in (8, 12, 16):
    lh = lu = vh = vu = 0
    for seq in per_layer.values():
        h, u = sim_lru(seq, S); lh += h; lu += u
        h, u = sim_victim(seq, S); vh += h; vu += u
    print(f"slots={S}: lru={lh/lu:.3f} victim={vh/vu:.3f} delta={(vh/vu-lh/lu)*100:+.1f} pts")

print("== Gate B: co-activation table, honest half-split ==")
layers = sorted(per_layer)
n_tok = min(len(per_layer[l]) for l in layers)
half = n_tok // 2
for M in (4, 8, 12):
    covered = total = 0
    for i, l in enumerate(layers[:-1]):
        nl = layers[i + 1]
        co = defaultdict(lambda: defaultdict(int))
        for t in range(half):
            for a in per_layer[l][t]:
                for b in per_layer[nl][t]:
                    co[a][b] += 1
        for t in range(half, n_tok):
            score = defaultdict(int)
            for a in per_layer[l][t]:
                for b, c in co[a].items():
                    score[b] += c
            pred = set(sorted(score, key=score.get, reverse=True)[:M])
            true = set(per_layer[nl][t])
            covered += len(pred & true); total += len(true)
    print(f"recall@{M}: {covered/total:.3f}  (bytes fetched per predicted expert: {M/4:.1f}x actual need)")
