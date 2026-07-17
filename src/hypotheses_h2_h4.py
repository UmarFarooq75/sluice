"""H2 + H4: two assumption-breaking hypotheses, tested from existing traces.

H2 — Co-activation storage layout.
  Assumption questioned: expert files are independent reads in arbitrary
  order. We build the co-firing graph (experts x experts, same layer,
  same token), cluster it greedily into disk-adjacent groups of G
  experts, and count how many CONTIGUOUS READ RUNS a token's top-8 needs
  under (a) random layout vs (b) co-activation layout. Fewer runs =
  fewer IO ops and larger sequential reads.

H4 — Cross-sequence amortization (agent-swarm mode).
  Assumption questioned: one user, one stream. For N parallel sequences
  (prompts within a workload = independent streams), measure the union
  of experts needed per layer per decode step across N streams vs N x 8.
  Sublinearity here = per-agent streaming cost collapse for agent fleets.

Output: results/h2_h4.json
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
GROUP = 8  # experts per disk-adjacent cluster


def build_layout(cofire):
    """Greedy clustering: repeatedly seed with the highest-degree unplaced
    expert and grow the group by strongest co-activation."""
    placed = set()
    groups = []
    strength = cofire.sum(axis=1)
    while len(placed) < N_EXPERTS:
        seed = int(max((e for e in range(N_EXPERTS) if e not in placed), key=lambda e: strength[e]))
        group = [seed]
        placed.add(seed)
        while len(group) < GROUP and len(placed) < N_EXPERTS:
            aff = cofire[group].sum(axis=0)
            cand = int(max((e for e in range(N_EXPERTS) if e not in placed), key=lambda e: aff[e]))
            group.append(cand)
            placed.add(cand)
        groups.append(group)
    pos = {}
    p = 0
    for g in groups:
        for e in g:
            pos[e] = p
            p += 1
    return pos


def read_runs(experts, pos):
    """Number of contiguous runs when fetching this expert set under layout pos."""
    s = sorted(pos[int(e)] for e in experts)
    runs = 1
    for a, b in zip(s, s[1:]):
        if b != a + 1:
            runs += 1
    return runs


def h2(dec):
    out = {}
    for name, tr in dec.items():
        idx = tr["indices"]
        n_tok, n_layers, k = idx.shape
        runs_rand, runs_opt = [], []
        rng = np.random.RandomState(0)
        for l in range(n_layers):
            co = np.zeros((N_EXPERTS, N_EXPERTS))
            for row in idx[:, l, :]:
                r = row.astype(int)
                for i in range(k):
                    for j in range(i + 1, k):
                        co[r[i], r[j]] += 1
                        co[r[j], r[i]] += 1
            pos_opt = build_layout(co)
            perm = rng.permutation(N_EXPERTS)
            pos_rand = {int(e): int(p) for p, e in enumerate(perm)}
            for row in idx[:, l, :]:
                runs_rand.append(read_runs(row, pos_rand))
                runs_opt.append(read_runs(row, pos_opt))
        out[name] = {
            "mean_read_runs_random_layout": round(float(np.mean(runs_rand)), 2),
            "mean_read_runs_coactivation_layout": round(float(np.mean(runs_opt)), 2),
            "io_ops_reduction_x": round(float(np.mean(runs_rand)) / float(np.mean(runs_opt)), 2),
        }
    return out


def h4(dec):
    out = {}
    for name, tr in dec.items():
        idx, pids = tr["indices"], tr["prompt_id"]
        n_tok, n_layers, k = idx.shape
        streams = [idx[pids == p] for p in np.unique(pids)]
        L = min(s.shape[0] for s in streams)
        streams = [s[:L] for s in streams]
        for N in (2, 3, 5):
            if len(streams) < N:
                continue
            unions = []
            for t in range(L):
                for l in range(n_layers):
                    u = np.unique(np.concatenate([s[t, l, :] for s in streams[:N]]))
                    unions.append(len(u))
            out.setdefault(name, {})[f"agents_{N}"] = {
                "mean_union": round(float(np.mean(unions)), 2),
                "naive": N * k,
                "sublinearity": round(float(np.mean(unions)) / (N * k), 3),
                "per_agent_cost_x": round(float(np.mean(unions)) / (N * k), 3),
            }
    return out


def main():
    dec = {k: decode_only(v) for k, v in load_traces().items()}
    res = {"H2_coactivation_layout": h2(dec), "H4_cross_sequence": h4(dec)}
    (ROOT / "results" / "h2_h4.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:2200])


if __name__ == "__main__":
    main()
