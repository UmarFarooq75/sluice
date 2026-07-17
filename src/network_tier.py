"""Network-tier simulation: ROM as a cache — what disk budget do you need?

Models pillar 6 (docs/techniques.md) with our traces: only the top-F% of
experts (by learned frequency) live on local disk; the rest live on the
network. Every routed expert not on disk is a network fetch (which then
persists — LRU-evicting the coldest disk resident).

Questions answered per disk budget F:
 1. steady-state network fetches per token (cost of the tail)
 2. how fetch rate decays over the session (does the disk warm up?)
 3. bytes math for a real target: DeepSeek-V3-class (22 MB/expert int4)

Profile source: leave-one-workload-out (disk stocked from OTHER tasks'
frequencies — pessimistic cold-ish start), and matched-task (optimistic).

Pure numpy over existing traces; no model runs.
Output: results/network_tier.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from simulate import decode_only, load_traces  # noqa: E402

N_EXPERTS = 64
BUDGET_FRACS = [0.25, 0.4, 0.5, 0.6, 0.75, 0.9]


def freq_ranking(idx_list):
    """idx_list: list of [n,L,k] arrays -> per-layer expert ranking by freq."""
    n_layers = idx_list[0].shape[1]
    rank = []
    for l in range(n_layers):
        agg = defaultdict(int)
        for idx in idx_list:
            vals, counts = np.unique(idx[:, l, :].reshape(-1), return_counts=True)
            for v, c in zip(vals, counts):
                agg[int(v)] += int(c)
        order = sorted(range(N_EXPERTS), key=lambda e: -agg.get(e, 0))
        rank.append(order)
    return rank


def simulate_disk(stream, rank, budget_frac):
    """Disk holds budget experts per layer (seeded by rank); misses fetch
    from network and persist (evict coldest by recency). Returns fetch
    rate per token overall and per quarter of the session."""
    n_tok, n_layers, k = stream.shape
    cap = max(1, int(round(budget_frac * N_EXPERTS)))
    disk = [dict((e, 0) for e in rank[l][:cap]) for l in range(n_layers)]
    fetches = np.zeros(n_tok)
    t = 0
    for i in range(n_tok):
        t += 1
        for l in range(n_layers):
            for e in stream[i, l]:
                e = int(e)
                if e in disk[l]:
                    disk[l][e] = t
                else:
                    fetches[i] += 1
                    if len(disk[l]) >= cap:
                        del disk[l][min(disk[l], key=disk[l].get)]
                    disk[l][e] = t
    q = n_tok // 4
    return {
        "fetches_per_token_overall": round(float(fetches.mean()), 3),
        "per_quarter": [round(float(fetches[i * q : (i + 1) * q].mean()), 3) for i in range(4)],
    }


def main():
    dec = {k: decode_only(v) for k, v in load_traces().items()}
    names = sorted(dec)
    results = {"note": "fetches/token out of 128 expert-uses (16 layers x top-8); DeepSeek-V3-class cost = fetches x 22 MB"}
    for name in names:
        stream = dec[name]["indices"]
        others = [dec[n]["indices"] for n in names if n != name]
        rank_loo = freq_ranking(others)
        rank_matched = freq_ranking([stream])
        entry = {}
        for f in BUDGET_FRACS:
            entry[f"disk_{int(f*100)}pct"] = {
                "seeded_leave_one_out": simulate_disk(stream, rank_loo, f),
                "seeded_matched": simulate_disk(stream, rank_matched, f),
            }
        results[name] = entry
        print(name, json.dumps(entry[f"disk_50pct"]), flush=True)

    out = ROOT / "results" / "network_tier.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
