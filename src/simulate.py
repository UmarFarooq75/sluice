"""Cache-policy simulator + trace analyses over routing traces. v2

v2 incorporates the fixes from docs/analysis-gaps.md:
  - cache sims run on DECODE tokens only (prefill feeds union analysis only)
  - task profiles use leave-one-prompt-out, not a leaky half-split
  - gate-mass-weighted hit rates reported alongside binary
  - per-layer breakdown + greedy cache-allocation analysis
  - global shared pool simulated against per-layer caches
  - expert-discovery (saturation) curves gate every working-set claim
  - easy-token statistics (top-1 mass / top-k cumulative gate mass)

All hit rates must be read against the random baseline k/N = 8/64 = 12.5%.
OLMoE routes 8-of-64; frontier models route 8-of-256 — treat shapes as
transferable, magnitudes as optimistic (docs/analysis-gaps.md #8).

Usage: python src/simulate.py     # writes results/analysis.json + prints summary
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = Path(os.environ.get("TRACES_DIR", ROOT / "traces"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
OUT_NAME = os.environ.get("OUT_NAME", "analysis.json")

CACHE_SIZES = [4, 8, 12, 16, 24, 32, 48]
N_EXPERTS = int(os.environ.get("N_EXPERTS", 64))
TOP_K = int(os.environ.get("TOP_K", 8))


def load_traces():
    out = {}
    for f in sorted(TRACES.glob("*.npz")):
        if f.stem.startswith("prefill_"):
            continue
        out[f.stem] = dict(np.load(f))
    return out


def decode_only(tr):
    m = tr["phase"] == 1
    return {k: v[m] for k, v in tr.items()}


# ---------------------------------------------------------------- reactive sims


def sim_reactive(indices_layer, scores_layer, capacity, policy):
    """One layer. Returns (binary_hit_rate, weighted_hit_rate)."""
    cache = {}
    freq = defaultdict(float)
    hits = misses = 0
    whits = wtot = 0.0
    t = 0
    for row, srow in zip(indices_layer, scores_layer):
        t += 1
        for e, s in zip(row, srow):
            e, s = int(e), float(s)
            freq[e] = freq[e] * 0.99 + 1.0
            wtot += s
            if e in cache:
                hits += 1
                whits += s
            else:
                misses += 1
                if len(cache) >= capacity:
                    victim = (
                        min(cache, key=cache.get)
                        if policy == "lru"
                        else min(cache, key=lambda x: freq[x])
                    )
                    del cache[victim]
            cache[e] = t
    return hits / max(hits + misses, 1), whits / max(wtot, 1e-9)


def sim_belady(indices_layer, capacity):
    flat = indices_layer.reshape(-1)
    next_use = np.full(len(flat), np.inf)
    last_seen = {}
    for i in range(len(flat) - 1, -1, -1):
        e = int(flat[i])
        next_use[i] = last_seen.get(e, np.inf)
        last_seen[e] = i
    cache = {}
    hits = misses = 0
    for i, e in enumerate(flat):
        e = int(e)
        if e in cache:
            hits += 1
        else:
            misses += 1
            if len(cache) >= capacity:
                victim = max(cache, key=cache.get)
                del cache[victim]
        cache[e] = next_use[i]
    return hits / max(hits + misses, 1)


def sim_global_pool(indices, scores, capacity_total):
    """Single LRU pool shared across layers; keys are (layer, expert)."""
    n_tok, n_layers, k = indices.shape
    cache = {}
    hits = misses = 0
    whits = wtot = 0.0
    t = 0
    for i in range(n_tok):
        for l in range(n_layers):
            t += 1
            for e, s in zip(indices[i, l], scores[i, l]):
                key = (l, int(e))
                s = float(s)
                wtot += s
                if key in cache:
                    hits += 1
                    whits += s
                else:
                    misses += 1
                    if len(cache) >= capacity_total:
                        victim = min(cache, key=cache.get)
                        del cache[victim]
                cache[key] = t
    return hits / max(hits + misses, 1), whits / max(wtot, 1e-9)


# ---------------------------------------------------------------- pinned sims


def top_experts_by_freq(indices_layer, capacity):
    vals, counts = np.unique(indices_layer.reshape(-1), return_counts=True)
    order = vals[np.argsort(-counts)]
    return set(int(e) for e in order[:capacity])


def pinned_rate(indices_layer, scores_layer, pinned_set):
    flat_i = indices_layer.reshape(-1)
    flat_s = scores_layer.reshape(-1).astype(np.float64)
    hit_mask = np.array([int(e) in pinned_set for e in flat_i])
    binary = hit_mask.mean() if len(flat_i) else 0.0
    weighted = flat_s[hit_mask].sum() / max(flat_s.sum(), 1e-9)
    return float(binary), float(weighted)


def taskprofile_loo(idx, sc, pids, C, l):
    """Leave-one-prompt-out: profile from other prompts, eval on held-out."""
    rates_b, rates_w = [], []
    for held in np.unique(pids):
        train = idx[pids != held][:, l, :]
        test_i = idx[pids == held][:, l, :]
        test_s = sc[pids == held][:, l, :]
        if len(train) == 0 or len(test_i) == 0:
            continue
        b, w = pinned_rate(test_i, test_s, top_experts_by_freq(train, C))
        rates_b.append(b)
        rates_w.append(w)
    return float(np.mean(rates_b)), float(np.mean(rates_w))


# ---------------------------------------------------------------- main sims


def run_cache_sims(dec):
    rows = []
    names = list(dec)
    pooled = np.concatenate([dec[n]["indices"] for n in names], axis=0)
    for name, tr in dec.items():
        idx, sc, pids = tr["indices"], tr["scores"].astype(np.float32), tr["prompt_id"]
        n_tok, n_layers, k = idx.shape
        other = names[(names.index(name) + 1) % len(names)]
        idx_other = dec[other]["indices"]
        for C in CACHE_SIZES:
            acc = defaultdict(lambda: [[], []])  # policy -> [binary list, weighted list]
            belady_list = []
            for l in range(n_layers):
                li, ls = idx[:, l, :], sc[:, l, :]
                for pol in ("lru", "lfu"):
                    b, w = sim_reactive(li, ls, C, pol)
                    acc[pol][0].append(b)
                    acc[pol][1].append(w)
                belady_list.append(sim_belady(li, C))
                b, w = pinned_rate(li, ls, top_experts_by_freq(pooled[:, l, :], C))
                acc["static"][0].append(b)
                acc["static"][1].append(w)
                b, w = taskprofile_loo(idx, sc, pids, C, l)
                acc["taskprofile"][0].append(b)
                acc["taskprofile"][1].append(w)
                b, w = pinned_rate(li, ls, top_experts_by_freq(idx_other[:, l, :], C))
                acc["crossprofile"][0].append(b)
                acc["crossprofile"][1].append(w)
            for pol, (bs, ws) in acc.items():
                rows.append(
                    dict(workload=name, capacity=C, policy=pol,
                         hit_rate=round(float(np.mean(bs)), 4),
                         weighted_hit_rate=round(float(np.mean(ws)), 4))
                )
            rows.append(
                dict(workload=name, capacity=C, policy="belady_oracle",
                     hit_rate=round(float(np.mean(belady_list)), 4),
                     weighted_hit_rate=None)
            )
            gb, gw = sim_global_pool(idx, sc, C * n_layers)
            rows.append(
                dict(workload=name, capacity=C, policy="global_pool_lru",
                     hit_rate=round(gb, 4), weighted_hit_rate=round(gw, 4))
            )
    return rows


def per_layer_breakdown(dec, C=16):
    out = {}
    for name, tr in dec.items():
        idx, sc = tr["indices"], tr["scores"].astype(np.float32)
        n_layers = idx.shape[1]
        layers = []
        for l in range(n_layers):
            b, w = sim_reactive(idx[:, l, :], sc[:, l, :], C, "lru")
            layers.append(round(b, 4))
        out[name] = layers
    return out


def greedy_allocation(dec, total_budget=256):
    """Given a global budget of cached experts, allocate per layer greedily
    by marginal LRU hit-rate gain. Uses pooled decode traces."""
    idx = np.concatenate([tr["indices"] for tr in dec.values()], axis=0)
    sc = np.concatenate([tr["scores"].astype(np.float32) for tr in dec.values()], axis=0)
    n_layers = idx.shape[1]
    steps = [4, 8, 12, 16, 24, 32, 48, 64]
    # hit rate per layer per step
    table = np.zeros((n_layers, len(steps)))
    for l in range(n_layers):
        for si, C in enumerate(steps):
            table[l, si] = sim_reactive(idx[:, l, :], sc[:, l, :], C, "lru")[0]
    alloc = [0] * n_layers
    level = [0] * n_layers  # index into steps, 0 = nothing cached yet
    # greedy: repeatedly give the next step to the layer with best marginal gain per slot
    budget = total_budget
    cur_hit = [0.0] * n_layers
    while budget > 0:
        best, best_gain = None, -1
        for l in range(n_layers):
            if level[l] >= len(steps):
                continue
            cost = steps[level[l]] - (steps[level[l] - 1] if level[l] > 0 else 0)
            if cost > budget:
                continue
            gain = (table[l, level[l]] - cur_hit[l]) / cost
            if gain > best_gain:
                best, best_gain = l, gain
        if best is None:
            break
        cost = steps[level[best]] - (steps[level[best] - 1] if level[best] > 0 else 0)
        budget -= cost
        cur_hit[best] = table[best, level[best]]
        alloc[best] = steps[level[best]]
        level[best] += 1
    uniform_C = total_budget // n_layers
    uniform = float(np.mean([sim_reactive(idx[:, l, :], sc[:, l, :], uniform_C, "lru")[0] for l in range(n_layers)]))
    return {
        "total_budget_experts": total_budget,
        "uniform_hit_rate": round(uniform, 4),
        "greedy_hit_rate": round(float(np.mean(cur_hit)), 4),
        "allocation_per_layer": alloc,
    }


# ---------------------------------------------------------------- analyses


def discovery_curves(dec):
    """Unique (layer,expert) pairs vs decode tokens seen; plateau check."""
    out = {}
    for name, tr in dec.items():
        idx = tr["indices"]
        n_tok, n_layers, k = idx.shape
        seen = set()
        curve = []
        for i in range(n_tok):
            for l in range(n_layers):
                for e in idx[i, l]:
                    seen.add((l, int(e)))
            curve.append(len(seen))
        q = [curve[int(len(curve) * f) - 1] for f in (0.25, 0.5, 0.75, 1.0)]
        last_quarter_new = curve[-1] - curve[int(len(curve) * 0.75) - 1]
        per100 = last_quarter_new / max(n_tok * 0.25, 1) * 100
        out[name] = {
            "unique_at_25/50/75/100pct_tokens": q,
            "total_possible": n_layers * N_EXPERTS,
            "new_experts_per_100tok_last_quarter": round(per100, 2),
            "plateaued": bool(per100 < 2.0),
        }
    return out


def concentration(dec):
    out = {}
    agg_by_layer = defaultdict(lambda: defaultdict(int))
    for name, tr in dec.items():
        idx = tr["indices"]
        n_layers = idx.shape[1]
        covers = {50: [], 90: [], 99: []}
        for l in range(n_layers):
            vals, counts = np.unique(idx[:, l, :].reshape(-1), return_counts=True)
            for v, c in zip(vals, counts):
                agg_by_layer[l][int(v)] += int(c)
            counts = np.sort(counts)[::-1]
            cum = np.cumsum(counts) / counts.sum()
            for pct in covers:
                covers[pct].append(int(np.searchsorted(cum, pct / 100) + 1))
        out[name] = {f"experts_for_{p}pct": round(float(np.mean(v)), 1) for p, v in covers.items()}
    covers = {50: [], 90: [], 99: []}
    for l, agg in agg_by_layer.items():
        counts = np.sort(np.array(list(agg.values())))[::-1]
        cum = np.cumsum(counts) / counts.sum()
        for pct in covers:
            covers[pct].append(int(np.searchsorted(cum, pct / 100) + 1))
    out["_pooled_all_tasks"] = {f"experts_for_{p}pct": round(float(np.mean(v)), 1) for p, v in covers.items()}
    return out


def cross_task_overlap(dec, top_n=16):
    names = list(dec)
    n_layers = dec[names[0]]["indices"].shape[1]
    tops = {
        name: [top_experts_by_freq(tr["indices"][:, l, :], top_n) for l in range(n_layers)]
        for name, tr in dec.items()
    }
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            jac = [len(tops[a][l] & tops[b][l]) / len(tops[a][l] | tops[b][l]) for l in range(n_layers)]
            rows.append({"pair": f"{a}~{b}", "jaccard_top16": round(float(np.mean(jac)), 3)})
    return rows


def temporal_locality(dec):
    out = {}
    for name, tr in dec.items():
        idx = tr["indices"]
        n_tok, n_layers, k = idx.shape
        per_layer = []
        for l in range(n_layers):
            a, b = idx[:-1, l, :], idx[1:, l, :]
            inter = [len(set(map(int, x)) & set(map(int, y))) / k for x, y in zip(a, b)]
            per_layer.append(float(np.mean(inter)))
        out[name] = {
            "mean_consec_overlap_top8": round(float(np.mean(per_layer)), 3),
            "layer0": round(per_layer[0], 3),
            "deepest_layer": round(per_layer[-1], 3),
            "random_baseline": round(TOP_K / N_EXPERTS, 3),
        }
    return out


def prefill_union():
    rows = []
    for f in sorted(TRACES.glob("prefill_*.npz")):
        tr = np.load(f)
        idx = tr["indices"]
        n_tok, n_layers, k = idx.shape
        uniq = float(np.mean([len(np.unique(idx[:, l, :])) for l in range(n_layers)]))
        rows.append(
            {
                "workload": f.stem,
                "prompt_tokens": int(n_tok),
                "over_context_4096": bool(n_tok > 4096),
                "mean_unique_experts_per_layer": round(uniq, 1),
                "naive_expert_loads_per_layer": n_tok * k,
                "io_reduction_x": round(n_tok * k / uniq, 1),
            }
        )
    return rows


def easy_token_stats(dec):
    """Addressable market for reduced-k / expert-skip: gate-mass concentration."""
    out = {}
    for name, tr in dec.items():
        sc = np.sort(tr["scores"].astype(np.float32), axis=-1)[..., ::-1]
        cum = np.cumsum(sc, axis=-1) / np.maximum(sc.sum(axis=-1, keepdims=True), 1e-9)
        # per token-layer: how many experts needed for 90% of top-8 gate mass
        need90 = (cum < 0.9).sum(axis=-1) + 1
        out[name] = {
            "mean_experts_for_90pct_mass": round(float(need90.mean()), 2),
            "pct_tokenlayers_top4_covers_90pct": round(float((need90 <= 4).mean()), 3),
            "mean_top1_share_of_top8": round(float((sc[..., 0] / np.maximum(sc.sum(-1), 1e-9)).mean()), 3),
        }
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    raw = load_traces()
    if not raw:
        raise SystemExit("No traces found - run src/tracer.py first")
    dec = {k: decode_only(v) for k, v in raw.items()}
    for k in dec:
        print(f"{k}: {dec[k]['indices'].shape[0]} decode tokens (of {raw[k]['indices'].shape[0]} total)")

    results = {
        "note": "OLMoE 8-of-64 routing; random baseline hit rate = 12.5%; decode-only cache sims",
        "cache_sims": run_cache_sims(dec),
        "per_layer_lru_C16": per_layer_breakdown(dec),
        "greedy_allocation_budget256": greedy_allocation(dec, 256),
        "discovery_curves": discovery_curves(dec),
        "concentration": concentration(dec),
        "cross_task_overlap": cross_task_overlap(dec),
        "temporal_locality": temporal_locality(dec),
        "easy_token_stats": easy_token_stats(dec),
        "prefill_union": prefill_union(),
    }
    (RESULTS / OUT_NAME).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS / OUT_NAME}")

    print("\n=== Binary (weighted) hit rates at C=16/64 per layer — random baseline 12.5% ===")
    for r in results["cache_sims"]:
        if r["capacity"] == 16:
            w = f"({r['weighted_hit_rate']:.1%})" if r["weighted_hit_rate"] is not None else ""
            print(f"  {r['workload']:14s} {r['policy']:16s} {r['hit_rate']:.1%} {w}")


if __name__ == "__main__":
    main()
