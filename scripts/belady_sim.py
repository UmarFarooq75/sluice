#!/usr/bin/env python3
# E23: how much hit-rate is LRU leaving on the table at small slot counts?
# Replays a real routing trace (LLMSTREAM_PRINT_IDS decode lines, exact m=0
# routing) through three eviction policies per layer:
#   lru    - what the engine does today
#   belady - clairvoyant optimum (evict the expert used farthest in the future);
#            upper bound no online policy can beat
#   static - pin the K most frequent experts of this trace (frequency prior);
#            if this approaches LRU, routing is Zipf-heavy and pinning is cheap
# Decision rule: belady-lru gap < 3 points at s8 -> kill the eviction card;
# > 10 points -> router-informed eviction is worth building.
import re, sys
from collections import defaultdict

path = sys.argv[1]
slots_list = [int(x) for x in (sys.argv[2:] or ["8", "12", "16"])]

# "ids ffn_moe_topk_slots-<il> ne=[4,1]: a b c d"  (decode calls only)
pat = re.compile(r"ids ffn_moe_topk_slots-(\d+)\s+ne=\[(\d+),1\]:((?:\s+\d+)+)")
per_layer = defaultdict(list)  # il -> [set(top_k), ...] in call order
for line in open(path, errors="replace"):
    m = pat.search(line)
    if m:
        il, k, ids = int(m.group(1)), int(m.group(2)), [int(x) for x in m.group(3).split()]
        per_layer[il].append(ids[:k])

def sim_lru(seq, S):
    cache, order, hits, uses = set(), [], 0, 0
    for step in seq:
        for e in step:
            uses += 1
            if e in cache:
                hits += 1
                order.remove(e); order.append(e)
            else:
                if len(cache) >= S:
                    # never evict an expert needed by this same step
                    for v in order:
                        if v not in step:
                            cache.discard(v); order.remove(v); break
                cache.add(e); order.append(e)
    return hits, uses

def sim_belady(seq, S):
    # flatten with future-use index per expert
    flat = [e for step in seq for e in step]
    nxt = [None] * len(flat)
    last = {}
    for i in range(len(flat) - 1, -1, -1):
        nxt[i] = last.get(flat[i], float("inf"))
        last[flat[i]] = i
    cache = {}  # expert -> next use index
    hits = uses = 0
    i = 0
    for step in seq:
        step_set = set(step)
        for e in step:
            uses += 1
            if e in cache:
                hits += 1
            else:
                if len(cache) >= S:
                    victim = max((v for v in cache if v not in step_set),
                                 key=lambda v: cache[v], default=None)
                    if victim is not None: del cache[victim]
                cache[e] = None
            cache[e] = nxt[i]
            i += 1
    return hits, uses

def sim_static(seq, S):
    freq = defaultdict(int)
    for step in seq:
        for e in step: freq[e] += 1
    pinned = set(sorted(freq, key=freq.get, reverse=True)[:S])
    hits = uses = 0
    for step in seq:
        for e in step:
            uses += 1
            if e in pinned: hits += 1
    return hits, uses

if not per_layer:
    sys.exit("no decode id lines found in trace")
print(f"layers={len(per_layer)} calls/layer~{len(next(iter(per_layer.values())))}")
for S in slots_list:
    tot = {"lru": [0, 0], "belady": [0, 0], "static": [0, 0]}
    for il, seq in per_layer.items():
        for name, fn in (("lru", sim_lru), ("belady", sim_belady), ("static", sim_static)):
            h, u = fn(seq, S)
            tot[name][0] += h; tot[name][1] += u
    line = f"slots={S:3d}  " + "  ".join(
        f"{n}={tot[n][0]/tot[n][1]:.3f}" for n in ("lru", "belady", "static"))
    gap = tot["belady"][0]/tot["belady"][1] - tot["lru"][0]/tot["lru"][1]
    print(line + f"  belady-lru gap={gap*100:+.1f} pts")
