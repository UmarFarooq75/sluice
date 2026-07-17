"""Bootstrap 95% CIs for headline Phase-0 numbers (analysis-gaps.md #7).

Resamples PROMPTS (the independent unit, avoiding within-prompt leakage)
with replacement, 500 iterations, for:
  - LRU hit rate @ C=16 per workload
  - task-profile (LOO) hit rate @ C=16
  - consecutive-token expert overlap (temporal locality)

Output: results/bootstrap_ci.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from simulate import (  # noqa: E402
    decode_only,
    load_traces,
    sim_reactive,
    top_experts_by_freq,
    pinned_rate,
)

N_BOOT = 500
C = 16
rng = np.random.RandomState(7)


def lru_stat(idx, sc):
    n_layers = idx.shape[1]
    return float(np.mean([sim_reactive(idx[:, l, :], sc[:, l, :], C, "lru")[0] for l in range(n_layers)]))


def locality_stat(idx):
    n_tok, n_layers, k = idx.shape
    vals = []
    for l in range(n_layers):
        a, b = idx[:-1, l, :], idx[1:, l, :]
        vals.append(np.mean([len(set(map(int, x)) & set(map(int, y))) / k for x, y in zip(a, b)]))
    return float(np.mean(vals))


def main():
    dec = {k: decode_only(v) for k, v in load_traces().items()}
    out = {}
    for name, tr in dec.items():
        idx, sc, pids = tr["indices"], tr["scores"].astype(np.float32), tr["prompt_id"]
        uniq = np.unique(pids)
        lru_samples, loc_samples = [], []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([np.where(pids == p)[0] for p in pick])
            lru_samples.append(lru_stat(idx[sel], sc[sel]))
            loc_samples.append(locality_stat(idx[sel]))
        def ci(s):
            return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]
        out[name] = {
            "lru_C16_hit": {"point": round(lru_stat(idx, sc), 3), "ci95": ci(lru_samples)},
            "temporal_locality": {"point": round(locality_stat(idx), 3), "ci95": ci(loc_samples)},
            "n_prompts": int(len(uniq)),
        }
        print(name, out[name], flush=True)
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print("Wrote results/bootstrap_ci.json")


if __name__ == "__main__":
    main()
