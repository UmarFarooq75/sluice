"""Self-speculation measurement on DeepSeek-V2-Lite (the decisive test).

Same design as self_spec.py, on the architecture family of our actual
targets (GLM/Kimi/DeepSeek): restricted-to-resident draft agreement +
expert-union growth from traces_ds. Hypothesis from finding 23: the
shared experts + concentrated gates make the restricted draft far more
faithful than OLMoE's (where the verdict was ~1.1-1.3x, marginal).

Output: results/self_spec_ds.json
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

from mlx_lm import load, generate  # noqa: E402
from mlx_lm.models import deepseek_v2  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

MODEL = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
EXPERT_MB = 4.33  # 3 x 2048 x 1408 int4 + scales
TOP_K = 6
GEN_TOKENS = 96


class Policy:
    allowed = None  # per-MoE-layer bool mask [64] or None


def install():
    orig = deepseek_v2.MoEGate.__call__

    def patched(self, x):
        if Policy.allowed is None:
            return orig(self, x)
        gates = x @ self.weight.T
        scores = mx.softmax(gates, axis=-1, precise=True)
        mask = mx.array(Policy.allowed[self._moe_idx].astype(np.float32))
        scores = scores * mask.reshape((1,) * (scores.ndim - 1) + (-1,))
        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        sel = mx.take_along_axis(scores, inds, axis=-1) * self.routed_scaling_factor
        return inds, sel

    deepseek_v2.MoEGate.__call__ = patched


def pinned_sets(workload, capacity, n_layers):
    tr = np.load(ROOT / "traces_ds" / f"{workload}.npz")
    idx = tr["indices"][tr["phase"] == 1]
    masks = []
    for l in range(n_layers):
        vals, counts = np.unique(idx[:, l, :].reshape(-1), return_counts=True)
        top = vals[np.argsort(-counts)][:capacity]
        m = np.zeros(64, dtype=bool)
        m[top.astype(int)] = True
        masks.append(m)
    return np.stack(masks)


def agreement(model, ids):
    inputs = mx.array(ids)[None, :]
    logits = model(inputs[:, :-1])
    pred = np.array(mx.argmax(logits, axis=-1))[0]
    return float((pred == np.array(ids[1:])).mean())


def main():
    model, tok = load(MODEL)
    n_moe = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE):
            layer.mlp.gate._moe_idx = n_moe
            n_moe += 1
    install()

    workloads = ["code", "math", "prose", "chat"]
    refs = {}
    Policy.allowed = None
    for w in workloads:
        pids = tok.apply_chat_template(
            [{"role": "user", "content": WORKLOADS[w][0]}], add_generation_prompt=True
        )
        out = generate(model, tok, pids, max_tokens=GEN_TOKENS)
        oids = tok.encode(out)
        if oids and oids[0] == tok.bos_token_id:
            oids = oids[1:]
        refs[w] = list(pids) + oids
        print(f"ref[{w}] {len(refs[w])} tokens", flush=True)

    res = {"draft_agreement": {}}
    for cap in (16, 24, 32):
        per = {}
        for w in workloads:
            Policy.allowed = pinned_sets(w, cap, n_moe)
            per[w] = round(agreement(model, refs[w]), 3)
        Policy.allowed = None
        res["draft_agreement"][f"pin{cap}"] = {
            "per_workload": per,
            "mean": round(float(np.mean(list(per.values()))), 3),
        }
        print(f"pin{cap}: {per}", flush=True)

    unions = defaultdict(list)
    for f in sorted((ROOT / "traces_ds").glob("*.npz")):
        tr = np.load(f)
        idx = tr["indices"][tr["phase"] == 1]
        n_tok, n_layers, k = idx.shape
        for wsize in (2, 4, 8):
            for l in range(n_layers):
                for i in range(0, n_tok - wsize, wsize):
                    unions[wsize].append(len(np.unique(idx[i : i + wsize, l, :])))
    res["union_growth"] = {
        f"w{w}": {
            "mean_union": round(float(np.mean(v)), 2),
            "naive": w * TOP_K,
            "sublinearity": round(float(np.mean(v)) / (w * TOP_K), 3),
        }
        for w, v in unions.items()
    }

    est = {}
    for cap_key, a_ in res["draft_agreement"].items():
        a = a_["mean"]
        for wk, u in res["union_growth"].items():
            w = int(wk[1:])
            expected = sum(a ** i for i in range(1, w + 1)) + 1
            per_tok = u["mean_union"] * 0.5 * EXPERT_MB / expected
            baseline = TOP_K * 0.5 * EXPERT_MB
            est[f"{cap_key}|{wk}"] = {
                "expected_accepted": round(expected, 2),
                "disk_reduction_x": round(baseline / per_tok, 2),
            }
    res["estimate_miss50pct"] = est
    (ROOT / "results" / "self_spec_ds.json").write_text(json.dumps(res, indent=2))
    best = sorted(est.items(), key=lambda kv: -kv[1]["disk_reduction_x"])[:3]
    print("BEST:", best, flush=True)
    print("Wrote results/self_spec_ds.json")


if __name__ == "__main__":
    main()
