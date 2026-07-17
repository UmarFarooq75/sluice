"""Router-lookahead recall: can layer L's input predict layer L+1's experts?

colibri (PILOT) reports 71.6% next-layer top-8 recall on GLM-5.2 by applying
layer L+1's router to layer L's hidden state. If this generalizes, prefetch
can hide most disk latency. We measure it on OLMoE on the fly: inside the
patched MoE block at layer L we ALSO run layer L+1's gate on the same input
and later compare its predicted top-8 with layer L+1's actual top-8.

Note the prediction uses the input to layer L's MoE block (post-attention,
pre-MoE state) — one full block earlier than the true gate input, same as
a real prefetcher would have. Baseline for comparison: same-experts-as-
previous-token (temporal locality, ~34-47% from findings 6).

Output: results/lookahead.json
"""

import json
import os
import sys
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
TOP_K = 8

STATE = {"gates": None, "pred": {}, "recalls": [], "per_layer": None, "enabled": False}


def install():
    def patched(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        k = self.top_k
        indices = mx.stop_gradient(
            mx.argpartition(-routing_weights, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        if STATE["enabled"] and L == 1:  # decode steps only
            l = self._layer_idx
            actual = set(int(e) for e in np.array(indices)[0])
            if l in STATE["pred"]:
                pred = STATE["pred"].pop(l)
                r = len(actual & pred) / TOP_K
                STATE["recalls"].append(r)
                STATE["per_layer"][l].append(r)
            if l + 1 < len(STATE["gates"]):
                nxt_logits = STATE["gates"][l + 1](x_flat)
                nxt = mx.argpartition(-nxt_logits, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
                STATE["pred"][l + 1] = set(int(e) for e in np.array(nxt)[0])

        y = self.switch_mlp(x_flat, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(B, L, D)

    olmoe.OlmoeSparseMoeBlock.__call__ = patched


def main():
    model, tok = load(MODEL)
    layers = model.model.layers
    for i, layer in enumerate(layers):
        layer.mlp._layer_idx = i
    STATE["gates"] = [layer.mlp.gate for layer in layers]
    STATE["per_layer"] = {i: [] for i in range(len(layers))}
    install()

    prompts = [ws[i] for i in range(2) for ws in WORKLOADS.values()]  # 12 prompts
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        STATE["enabled"] = True
        for _ in stream_generate(model, tok, text, max_tokens=192):
            pass
        STATE["enabled"] = False
        STATE["pred"].clear()
        print(f"prompt done; running mean recall={np.mean(STATE['recalls']):.3f}", flush=True)

    per_layer = {
        l: round(float(np.mean(v)), 3) for l, v in STATE["per_layer"].items() if v
    }
    results = {
        "model": MODEL,
        "mean_next_layer_top8_recall": round(float(np.mean(STATE["recalls"])), 3),
        "n_measurements": len(STATE["recalls"]),
        "per_layer_recall": per_layer,
        "colibri_glm_reference": 0.716,
        "temporal_locality_baseline": "0.34-0.47 (findings #6)",
    }
    out = ROOT / "results" / "lookahead.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
