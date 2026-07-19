#!/usr/bin/env python3
# E29 feasibility gate: can speculative BATCHED VERIFY (exact-output decoding)
# beat sequential decode on I/O, given the E27 pool makes batch passes cheap?
#
# Sequential baseline: per-layer LRU-8 over the real token stream (mirrors the
# measured slots8 exact hit rate). Batched verify, window w, PERFECT draft
# (acceptance=1, the upper bound): per window per layer the model needs the
# UNION of the w tokens' top-4 sets - priced two ways:
#   pess: full union fetched into the pool (pool cold each pass)
#   opt:  union minus what the decode LRU-8 already holds (verify checks cache
#         first), LRU updated as if tokens ran sequentially
# Verdict bar: if even the acceptance=1 optimistic ratio is < 2x, the lever is
# dead on this trace - the LRU already harvests the same consecutive-token
# overlap the union would. (Why refutations keep landing here: same physics.)
import re, sys
from collections import defaultdict

trace = "results/trace_m0_s8.txt"
per_tok = defaultdict(dict)  # token -> layer -> set(ids)
count = defaultdict(int)
for line in open(trace):
    m = re.match(r"ids ffn_moe_topk_slots-(\d+)\s+ne=\[4,1\]: (.+)", line)
    if not m:
        continue
    il = int(m.group(1))
    ids = frozenset(int(x) for x in m.group(2).split())
    per_tok[count[il]][il] = ids
    count[il] += 1

n_tok = min(count.values())
n_lay = len(count)
print(f"trace: {n_tok} tokens x {n_lay} layers")

SLOTS = 8
def lru_misses():
    total = 0
    lru = {il: [] for il in range(n_lay)}
    for t in range(n_tok):
        for il in range(n_lay):
            for e in per_tok[t][il]:
                if e in lru[il]:
                    lru[il].remove(e)
                else:
                    total += 1
                    if len(lru[il]) >= SLOTS:
                        lru[il].pop()
                lru[il].insert(0, e)
    return total

seq = lru_misses()
print(f"sequential LRU-{SLOTS}: {seq} misses ({seq/n_tok:.1f}/token, hit {1-seq/(n_tok*n_lay*4):.3f})")

for w in (4, 8, 16, 32):
    pess = opt = 0
    lru = {il: [] for il in range(n_lay)}
    for start in range(0, n_tok - n_tok % w, w):
        toks = range(start, start + w)
        for il in range(n_lay):
            union = set()
            for t in toks:
                union |= per_tok[t][il]
            pess += len(union)
            opt += len([e for e in union if e not in lru[il]])
            # LRU advances as if the w tokens ran sequentially (verify accepts all)
            for t in toks:
                for e in per_tok[t][il]:
                    if e in lru[il]:
                        lru[il].remove(e)
                    elif len(lru[il]) >= SLOTS:
                        lru[il].pop()
                    lru[il].insert(0, e)
    n_used = (n_tok // w) * w
    seq_scaled = seq * n_used / n_tok
    print(f"w={w:2d}: union-fetch pess {pess} opt {opt} vs seq {seq_scaled:.0f} "
          f"-> ratio pess {seq_scaled/pess:.2f}x opt {seq_scaled/opt:.2f}x (acceptance=1 UPPER bound)")
