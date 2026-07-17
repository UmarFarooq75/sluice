"""H3: Progressive experts — is within-expert neuron usage skewed enough
to fetch file PREFIXES instead of whole experts?

Assumption questioned: the fetch unit must be the whole expert (~3.5 MB).
Each expert's FFN has 1024 intermediate neurons; if activation mass
concentrates in a stable subset, we can store rows sorted by importance
and stream only the first X% — a per-fetch bytes dial with sequential
reads intact (progressive-JPEG for experts).

Phase A (model run): teacher-forced pass over eval texts capturing each
  MoE layer's input hidden states + routed experts + gate scores.
Phase B (offline, expert_store): per (layer, expert) compute inner
  activations a = silu(x@gate.T) * (x@up.T) for its routed tokens:
    - neuron importance = mean |a| per neuron; stability = split-half
      rank correlation (is the hot set STABLE across tokens?)
    - output error: y_topX = down(a masked to top-X% neurons) vs y_full
      -> relative L2 error at X in {25, 50, 75}%.

Output: results/h3_progressive.json
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

from mlx_lm import load  # noqa: E402
from mlx_lm.models import olmoe  # noqa: E402
from streamer_poc import GROUP_SIZE, BITS, load_expert  # noqa: E402

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
KEEPS = (0.25, 0.5, 0.75)


class Cap:
    enabled = False
    x_by_layer = defaultdict(list)
    routes_by_layer = defaultdict(list)


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
        if Cap.enabled:
            Cap.x_by_layer[self._layer_idx].append(np.array(x_flat.astype(mx.float16)))
            Cap.routes_by_layer[self._layer_idx].append(np.array(indices))
        y = self.switch_mlp(x_flat, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(B, L, D)

    olmoe.OlmoeSparseMoeBlock.__call__ = patched


def qmm(x, w):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=GROUP_SIZE, bits=BITS)


def main():
    model, tok = load(MODEL)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    install()

    texts = "\n\n".join((ROOT / "eval" / f).read_text() for f in ["prose.txt", "math.txt"])
    ids = tok.encode(texts)[:1400]
    Cap.enabled = True
    model(mx.array(ids)[None, :])
    mx.eval(mx.zeros(1))
    Cap.enabled = False
    print(f"captured {len(ids)} tokens x 16 layers", flush=True)

    per_expert_tokens = defaultdict(list)  # (l, e) -> row indices
    X = {l: np.concatenate(v, axis=0) for l, v in Cap.x_by_layer.items()}
    R = {l: np.concatenate(v, axis=0) for l, v in Cap.routes_by_layer.items()}
    for l in R:
        for i, row in enumerate(R[l]):
            for e in row:
                per_expert_tokens[(l, int(e))].append(i)

    ginis, stability, errors = [], [], {k: [] for k in KEEPS}
    sampled = 0
    for (l, e), rows in sorted(per_expert_tokens.items()):
        if len(rows) < 24 or sampled >= 160:
            continue
        sampled += 1
        w, _ = load_expert(l, e)
        x = mx.array(X[l][rows])
        g = qmm(x, w["gate_proj"])
        a = (mx.sigmoid(g) * g * qmm(x, w["up_proj"])).astype(mx.float32)
        a_np = np.array(a)
        imp = np.abs(a_np).mean(axis=0)
        # gini of neuron importance
        s = np.sort(imp)
        n = len(s)
        gini = (2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())
        ginis.append(float(gini))
        # split-half stability of top-25% set
        h = len(rows) // 2
        i1 = np.abs(a_np[:h]).mean(axis=0)
        i2 = np.abs(a_np[h:]).mean(axis=0)
        t1 = set(np.argsort(-i1)[: n // 4])
        t2 = set(np.argsort(-i2)[: n // 4])
        stability.append(len(t1 & t2) / (n // 4))
        # output error keeping top-X neurons (train importance on first half, eval on second)
        y_full = np.array(qmm(a.astype(mx.float16), w["down_proj"]))
        for keep in KEEPS:
            top = np.argsort(-i1)[: int(n * keep)]
            mask = np.zeros(n, dtype=np.float16)
            mask[top] = 1
            y_trunc = np.array(qmm((a * mx.array(mask)).astype(mx.float16), w["down_proj"]))
            num = np.linalg.norm(y_full[h:] - y_trunc[h:])
            den = np.linalg.norm(y_full[h:]) + 1e-9
            errors[keep].append(float(num / den))

    res = {
        "experts_sampled": sampled,
        "mean_gini_neuron_importance": round(float(np.mean(ginis)), 3),
        "top25pct_set_split_half_stability": round(float(np.mean(stability)), 3),
        "rel_output_error_keeping": {
            f"{int(k*100)}pct_rows": round(float(np.mean(v)), 4) for k, v in errors.items()
        },
        "interpretation": "high gini + high stability + low error at 50pct => progressive fetch viable",
    }
    (ROOT / "results" / "h3_progressive.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
