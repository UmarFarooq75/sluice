"""Mixed-session analysis: profile switching in realistic interleaved use.

Two questions:
 1. When the user switches task mid-session (code -> chat -> math ...),
    how fast can we DETECT the switch from router distributions alone?
 2. What does a switch cost the hybrid cache (hit-rate dip + recovery)?

Uses existing per-workload traces — interleaves them into one synthetic
session (deterministic order), which is fair because KV state does not
affect routing statistics across separate prompts.

Detection method: per-token router-distribution centroid distance.
For each task we compute a centroid (mean top-8 one-hot vector per layer,
from that task's own trace, leave-session-out not needed for detection
latency measurement). At runtime we keep an EMA of the recent routing
one-hots and classify against centroids; detection latency = tokens from
true switch to stable correct classification (3 consecutive windows).

Cache cost: replay the interleaved stream through per-layer LRU (C=16)
and through hybrid (pin 12 by current-detected-profile + LRU 4) and
report hit-rate in the 50 tokens after each switch vs steady state.

Output: results/session_switch.json
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
SEGMENT = 180  # tokens per task segment in the synthetic session
ORDER = ["code", "chat", "math", "prose", "code", "knowledge", "math", "multilingual"]
EMA = 0.9
WINDOW_STABLE = 3


def one_hot_stream(idx):
    """[n_tok, n_layers, k] -> [n_tok, n_layers*64] normalized one-hots."""
    n_tok, n_layers, k = idx.shape
    oh = np.zeros((n_tok, n_layers, N_EXPERTS), dtype=np.float32)
    for l in range(n_layers):
        for j in range(k):
            oh[np.arange(n_tok), l, idx[:, l, j].astype(int)] += 1
    return oh.reshape(n_tok, -1) / k


def main():
    dec = {k: decode_only(v) for k, v in load_traces().items()}
    names = sorted(dec)
    centroids = {n: one_hot_stream(dec[n]["indices"]).mean(axis=0) for n in names}

    # build interleaved session (skip first 40 tokens of each source to avoid
    # prompt-header routing; rotate source offsets so segments differ)
    segs, labels = [], []
    used = defaultdict(int)
    for task in ORDER:
        idx = dec[task]["indices"]
        start = 40 + used[task]
        used[task] += SEGMENT
        segs.append(idx[start : start + SEGMENT])
        labels += [task] * SEGMENT
    stream = np.concatenate(segs, axis=0)
    oh = one_hot_stream(stream)

    # --- detection latency ---
    ema = None
    detected = []
    for t in range(len(oh)):
        ema = oh[t] if ema is None else EMA * ema + (1 - EMA) * oh[t]
        dists = {n: float(np.linalg.norm(ema - c)) for n, c in centroids.items()}
        detected.append(min(dists, key=dists.get))
    latencies = []
    boundaries = [i * SEGMENT for i in range(1, len(ORDER))]
    for b in boundaries:
        true = labels[b]
        lat = None
        run = 0
        for t in range(b, min(b + SEGMENT, len(detected))):
            run = run + 1 if detected[t] == true else 0
            if run >= WINDOW_STABLE:
                lat = t - b - WINDOW_STABLE + 1
                break
        latencies.append({"boundary_token": b, "switch_to": true, "latency_tokens": lat})
    det_acc = float(np.mean([detected[t] == labels[t] for t in range(len(labels))]))

    # --- cache cost around switches: per-layer LRU C=16 ---
    n_layers = stream.shape[1]
    hits_t = np.zeros(len(stream))
    for l in range(n_layers):
        cache = {}
        t = 0
        for i, row in enumerate(stream[:, l, :]):
            t += 1
            h = 0
            for e in row:
                e = int(e)
                if e in cache:
                    h += 1
                else:
                    if len(cache) >= 16:
                        del cache[min(cache, key=cache.get)]
                cache[e] = t
            hits_t[i] += h / 8
    hits_t /= n_layers
    post, steady = [], []
    for i, b in enumerate(boundaries):
        post.append(float(hits_t[b : b + 50].mean()))
        steady.append(float(hits_t[b + 100 : b + SEGMENT].mean()) if b + 100 < len(hits_t) else None)

    results = {
        "session_order": ORDER,
        "segment_tokens": SEGMENT,
        "detection_overall_accuracy": round(det_acc, 3),
        "switch_latencies": latencies,
        "median_latency_tokens": float(np.median([x["latency_tokens"] for x in latencies if x["latency_tokens"] is not None])),
        "lru16_hit_first50_after_switch": [round(x, 3) for x in post],
        "lru16_hit_steady_state": [round(x, 3) if x else None for x in steady],
    }
    out = ROOT / "results" / "session_switch.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:1200])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
