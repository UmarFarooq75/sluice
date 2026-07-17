"""H1: Is MoE routing token-determined or context-determined?

Assumption questioned: expert choice needs the hidden state, so prefetch
can only look one layer ahead. If a token ID largely determines its
expert path regardless of context, a static per-token table prefetches
ALL layers' experts at embedding time — full-depth prefetch, closing
most of the Belady gap without any learned predictor.

Method: trace OLMoE decode steps recording (token_id, layer, top-8).
For token ids seen >= 5 times in different contexts, measure:
  determinism@8 = mean fraction of a token's top-8 that matches that
                  token's MODAL top-8 set (majority experts across
                  occurrences), per layer.
  table_recall  = if we prefetch the modal set (8 experts/layer) from a
                  static table, what fraction of actual loads it covers
                  (evaluated on held-out occurrences, 50/50 split).
Baselines: random = 12.5%; previous-token temporal locality = 34-47%;
one-layer lookahead = 84%.

Output: traces_ids/*.npz + results/h1_token_determinism.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "src"))

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models import olmoe  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
MAX_TOKENS = 256


class Col:
    rows = []  # per decode step: {"layer": l, "experts": np[8]}
    enabled = False


def install():
    def patched(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        k = self.top_k
        indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if Col.enabled and L == 1:
            Col.rows.append({"layer": self._layer_idx, "experts": np.array(indices)[0]})
        y = self.switch_mlp(x_flat, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(B, L, D)

    olmoe.OlmoeSparseMoeBlock.__call__ = patched


def main():
    model, tok = load(MODEL)
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    install()

    # token_events[token_id][layer] -> list of expert sets (across occurrences)
    token_events = defaultdict(lambda: defaultdict(list))
    prompts = [p for ws in WORKLOADS.values() for p in ws[:3]]
    for pi, p in enumerate(prompts):
        pids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        Col.enabled = True
        step_layers = []
        gen_ids = []
        for resp in stream_generate(model, tok, pids, max_tokens=MAX_TOKENS):
            gen_ids.append(resp.token)
        Col.enabled = False
        # rows arrive layer0..15 per step, repeated per generated token.
        # The experts at step t were computed while PROCESSING token t-1
        # (input token) -> attribute to the INPUT token id.
        steps = len(Col.rows) // n_layers
        inputs = ([pids[-1]] + gen_ids)[:steps]
        for t in range(steps):
            tid = int(inputs[t])
            for j in range(n_layers):
                r = Col.rows[t * n_layers + j]
                token_events[tid][r["layer"]].append(frozenset(int(e) for e in r["experts"]))
        Col.rows = []
        print(f"prompt {pi+1}/{len(prompts)} done", flush=True)

    # analysis
    per_layer_det, per_layer_recall, counts = defaultdict(list), defaultdict(list), []
    for tid, layers in token_events.items():
        occ = len(next(iter(layers.values())))
        if occ < 5:
            continue
        counts.append(occ)
        for l, sets in layers.items():
            half = len(sets) // 2
            train, test = sets[:half], sets[half:]
            if not train or not test:
                continue
            freq = defaultdict(int)
            for s in train:
                for e in s:
                    freq[e] += 1
            modal = set(sorted(freq, key=freq.get, reverse=True)[:8])
            for s in test:
                per_layer_recall[l].append(len(s & modal) / 8)
            allsets = sets
            f2 = defaultdict(int)
            for s in allsets:
                for e in s:
                    f2[e] += 1
            modal_all = set(sorted(f2, key=f2.get, reverse=True)[:8])
            for s in allsets:
                per_layer_det[l].append(len(s & modal_all) / 8)

    det = {l: round(float(np.mean(v)), 3) for l, v in sorted(per_layer_det.items())}
    rec = {l: round(float(np.mean(v)), 3) for l, v in sorted(per_layer_recall.items())}
    res = {
        "tokens_analyzed": len(counts),
        "mean_occurrences": round(float(np.mean(counts)), 1),
        "determinism_at8_per_layer": det,
        "heldout_table_recall_per_layer": rec,
        "mean_determinism": round(float(np.mean(list(det.values()))), 3),
        "mean_heldout_table_recall": round(float(np.mean(list(rec.values()))), 3),
        "baselines": {"random": 0.125, "prev_token": "0.34-0.47", "one_layer_lookahead": 0.839},
    }
    (ROOT / "results" / "h1_token_determinism.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
