"""Exact DP capacity allocator — replaces the buggy greedy in simulate.py v2.

Given a global budget of cached experts (RAM proxy), find the per-layer
cache capacities that maximize the mean LRU hit rate. Greedy fails here
because LRU hit-vs-capacity curves are non-concave (thrash cliff at tiny
capacities); DP over (layer, budget) is exact for the discretized levels.

Also reports uniform-allocation baseline and the gain, for budgets
128/192/256/384/512 experts (of 16*64=1024 total).

Output: results/dp_allocation.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from simulate import decode_only, load_traces, sim_reactive  # noqa: E402

LEVELS = [0, 4, 8, 12, 16, 24, 32, 48, 64]
BUDGETS = [128, 192, 256, 384, 512]
GRAN = 4  # all levels are multiples of 4


def main():
    dec = {k: decode_only(v) for k, v in load_traces().items()}
    idx = np.concatenate([tr["indices"] for tr in dec.values()], axis=0)
    sc = np.concatenate([tr["scores"].astype(np.float32) for tr in dec.values()], axis=0)
    n_layers = idx.shape[1]

    print("Building per-layer hit-rate table ...", flush=True)
    table = np.zeros((n_layers, len(LEVELS)))
    for l in range(n_layers):
        for li, C in enumerate(LEVELS):
            table[l, li] = (
                0.0 if C == 0 else sim_reactive(idx[:, l, :], sc[:, l, :], C, "lru")[0]
            )
        print(f"  layer {l}: {[round(x,3) for x in table[l]]}", flush=True)

    results = {"levels": LEVELS, "per_layer_hit_table": table.round(4).tolist()}
    for budget in BUDGETS:
        B = budget // GRAN
        NEG = -1e9
        dp = np.full((n_layers + 1, B + 1), NEG)
        choice = np.zeros((n_layers + 1, B + 1), dtype=int)
        dp[0, :] = 0.0
        for l in range(1, n_layers + 1):
            for b in range(B + 1):
                for li, C in enumerate(LEVELS):
                    cb = C // GRAN
                    if cb > b:
                        break
                    v = dp[l - 1, b - cb] + table[l - 1, li]
                    if v > dp[l, b]:
                        dp[l, b] = v
                        choice[l, b] = li
        # backtrack
        b = B
        alloc = []
        for l in range(n_layers, 0, -1):
            li = choice[l, b]
            alloc.append(LEVELS[li])
            b -= LEVELS[li] // GRAN
        alloc = alloc[::-1]
        dp_hit = dp[n_layers, B] / n_layers
        uniform_C = min(LEVELS, key=lambda c: abs(c - budget // n_layers))
        uniform_hit = float(np.mean([table[l, LEVELS.index(uniform_C)] for l in range(n_layers)]))
        results[f"budget_{budget}"] = {
            "dp_hit_rate": round(float(dp_hit), 4),
            "uniform_capacity": uniform_C,
            "uniform_hit_rate": round(uniform_hit, 4),
            "gain_points": round(float(dp_hit - uniform_hit) * 100, 2),
            "allocation": alloc,
        }
        print(f"budget {budget}: DP {dp_hit:.1%} vs uniform({uniform_C}/layer) {uniform_hit:.1%}  alloc={alloc}", flush=True)

    out = ROOT / "results" / "dp_allocation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
