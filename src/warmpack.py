"""Working-set persistence — save/reload "warm-start packs".

A warm-start pack is the per-layer hot-expert working set for a task: the
smallest set of experts that covers `coverage` of that task's routing
selections. Persisted to a small sidecar JSON, it lets a session START WARM —
the engine can pre-fill those experts into slots so early tokens hit instead
of stalling, instead of discovering the working set cold over the first ~50
tokens (session_switch.py measured that cold-start dip).

This module is the SAVE/RELOAD half. It is inert by itself: nothing here runs
unless invoked, and the engine only consumes a pack when LLMSTREAM_WARMPACK is
set (off by default — the stock path is byte-identical without it).

Two ways to build a pack:
  * from an offline routing trace (traces/*.npz via src/simulate.py), or
  * from a live run's expert-id dump (engine LLMSTREAM_PRINT_IDS output).

Pre-warming is LOGIT-NEUTRAL: it only changes which experts are resident when
generation starts, never which experts get computed — so exact-mode output
stays bit-identical (the gate that must stay green when the engine hook lands).

Usage:
  python src/warmpack.py build            # build packs for all traces
  python src/warmpack.py eval code        # measure the cold-start lift a pack buys
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "results" / "warmpacks"
PACK_VERSION = 1


def build_pack(indices, coverage=0.90, task="unknown", model="unknown"):
    """indices: [n_tok, n_layers, k] top-k expert ids. Returns a pack dict:
    per layer, the smallest set of experts covering `coverage` of selections."""
    n_tok, n_layers, k = indices.shape
    hot = []
    covered = []
    for l in range(n_layers):
        counts = Counter(int(e) for e in indices[:, l, :].reshape(-1))
        total = sum(counts.values())
        ranked = counts.most_common()
        acc, keep = 0, []
        for e, c in ranked:
            keep.append(e)
            acc += c
            if acc / total >= coverage:
                break
        hot.append(keep)
        covered.append(round(acc / total, 4))
    return {
        "warmpack_version": PACK_VERSION,
        "task": task, "model": model, "coverage_target": coverage,
        "n_layers": n_layers, "k": k, "n_tokens_profiled": int(n_tok),
        "experts_per_layer": [len(h) for h in hot],
        "coverage_achieved": covered,
        "hot": hot,  # [n_layers][variable] expert ids, most-frequent first
    }


def pack_text(pack):
    """Engine-readable format (LLMSTREAM_WARMPACK): header + one line of expert
    ids per layer, most-frequent first."""
    lines = [f"warmpack {PACK_VERSION} {pack['n_layers']}"]
    lines += [" ".join(str(e) for e in layer) for layer in pack["hot"]]
    return "\n".join(lines) + "\n"


def save_pack(pack, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, separators=(",", ":")))
    path.with_suffix(".pack").write_text(pack_text(pack))  # engine consumes this
    return path


def load_pack(path):
    pack = json.loads(Path(path).read_text())
    if pack.get("warmpack_version") != PACK_VERSION:
        raise ValueError(f"warmpack version {pack.get('warmpack_version')} != {PACK_VERSION}")
    return pack


def _lru_hit_curve(stream_layer, cap=16, preseed=None):
    """Replay one layer's per-token expert selections through LRU(cap).
    preseed: experts resident at t=0 (the warm start). Returns per-token hit frac."""
    cache = {}
    if preseed:
        for i, e in enumerate(preseed[:cap]):
            cache[int(e)] = -len(preseed) + i  # oldest, so real uses refresh them
    hits = []
    t = 0
    for row in stream_layer:
        t += 1
        h = 0
        for e in row:
            e = int(e)
            if e in cache:
                h += 1
            elif len(cache) >= cap:
                del cache[min(cache, key=cache.get)]
            cache[e] = t
        hits.append(h / len(row))
    return hits


def eval_pack(indices, pack, cap=16, window=50):
    """Measure the cold-start lift the pack buys: first-`window`-token hit-rate,
    cache preseeded with the pack's hot set vs cold. Averaged over layers."""
    import statistics
    n_tok, n_layers, k = indices.shape
    cold, warm = [], []
    for l in range(n_layers):
        sl = indices[:, l, :]
        cold.append(statistics.mean(_lru_hit_curve(sl, cap)[:window]))
        warm.append(statistics.mean(_lru_hit_curve(sl, cap, preseed=pack["hot"][l])[:window]))
    return {"cold_first%d" % window: round(statistics.mean(cold), 4),
            "warm_first%d" % window: round(statistics.mean(warm), 4),
            "lift_pts": round((statistics.mean(warm) - statistics.mean(cold)) * 100, 1)}


def _load_traces():
    sys.path.insert(0, str(ROOT / "src"))
    from simulate import decode_only, load_traces  # noqa
    return {k: decode_only(v)["indices"] for k, v in load_traces().items()}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    traces = _load_traces()
    if cmd == "build":
        for task, idx in traces.items():
            pack = build_pack(idx, task=task, model="olmoe-1b-7b(trace)")
            p = save_pack(pack, PACK_DIR / f"{task}.warmpack.json")
            print(f"{task:12s} {sum(pack['experts_per_layer'])//pack['n_layers']:>3d} experts/layer "
                  f"(cover {pack['coverage_achieved'][0]:.2f})  ->  {p.name}")
    elif cmd == "eval":
        task = sys.argv[2] if len(sys.argv) > 2 else "code"
        idx = traces[task]
        pack = build_pack(idx, task=task)
        print(f"warm-start lift for '{task}':", json.dumps(eval_pack(idx, pack)))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
