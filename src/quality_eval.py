"""Restricted-routing quality evaluation — the trace→quality bridge.

Question: when the engine restricts which experts the router may use
(because only some are resident in fast memory), how much does output
quality actually degrade, and what does each restriction save in bytes?

Method: teacher-forced NLL (negative log-likelihood per token) on fixed
eval texts under different routing policies. NLL is deterministic, fast,
and comparable across conditions; lower = better. Relative deltas vs
baseline are the finding, not absolute values.

Conditions:
  baseline      unrestricted top-8 (reference)
  k6 / k4       reduced top-k (tests the negative finding causally)
  pin16/24/32   router masked to per-layer pinned sets learned from OUR
                traces (matched task profile) — models "never touch disk"
  pin24_cross   pinned set from the WRONG task (profile-mismatch damage)
  route24_m     cache-aware routing: prefer the 24 resident experts, but
                fetch a non-resident expert when its router prob exceeds
                the weakest resident pick by >= margin. Counts fetches
                (disk-traffic proxy) — the quality/bytes dial.

Eval texts (eval/): prose.txt, code.py, math.txt — fixed, identical
across conditions.

Output: results/quality_eval.json
"""

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from mlx_lm import load  # noqa: E402
from mlx_lm.models import olmoe  # noqa: E402

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
MARGIN = 0.02


class Policy:
    """Global routing policy applied inside the patched MoE block."""

    def __init__(self):
        self.mode = "baseline"
        self.k = None  # override top_k
        self.allowed = None  # list per layer: np bool mask [64] or None
        self.margin = MARGIN
        self.fetches = 0  # route mode: non-resident experts fetched
        self.total_selects = 0

    def reset_counters(self):
        self.fetches = 0
        self.total_selects = 0


POLICY = Policy()


def install_patch():
    def patched_call(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        k = POLICY.k or self.top_k

        if POLICY.mode in ("pin", "route") and POLICY.allowed is not None:
            mask = POLICY.allowed[self._layer_idx]  # bool [64]
            mask_mx = mx.array(mask.astype(np.float32))[None, :]
            restricted = routing_weights * mask_mx
            if POLICY.mode == "pin":
                use_weights = restricted
            else:  # route: allow high-margin outsiders
                res_idx = mx.argpartition(-restricted, kth=k - 1, axis=-1)[..., :k]
                res_scores = mx.take_along_axis(restricted, res_idx, axis=-1)
                weakest_res = res_scores.min(axis=-1, keepdims=True)
                outsider = routing_weights * (1 - mask_mx)
                fetch_mask = outsider > (weakest_res + POLICY.margin)
                use_weights = restricted + routing_weights * fetch_mask
                POLICY.fetches += int(fetch_mask.sum().item())
        else:
            use_weights = routing_weights

        indices = mx.stop_gradient(
            mx.argpartition(-use_weights, kth=k - 1, axis=-1)[..., :k]
        )
        # scores must come from the TRUE router weights for selected experts
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        POLICY.total_selects += indices.size
        y = self.switch_mlp(x_flat, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(B, L, D)

    olmoe.OlmoeSparseMoeBlock.__call__ = patched_call


def teacher_forced_nll(model, token_ids):
    """Mean NLL per token over the sequence (predicting tokens[1:])."""
    inputs = mx.array(token_ids)[None, :]
    logits = model(inputs[:, :-1])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = inputs[:, 1:]
    nll = -mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
    return float(nll.mean().item())


def pinned_sets_from_traces(workload, capacity, n_layers=16):
    tr = np.load(ROOT / "traces" / f"{workload}.npz")
    idx = tr["indices"][tr["phase"] == 1]
    masks = []
    for l in range(n_layers):
        vals, counts = np.unique(idx[:, l, :].reshape(-1), return_counts=True)
        top = vals[np.argsort(-counts)][:capacity]
        m = np.zeros(64, dtype=bool)
        m[top.astype(int)] = True
        masks.append(m)
    return masks


PROFILE_FOR_TEXT = {"prose": "prose", "code": "code", "math": "math"}
CROSS_PROFILE = {"prose": "code", "code": "prose", "math": "prose"}


def main():
    model, tok = load(MODEL)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    install_patch()

    texts = {}
    for f in (ROOT / "eval").glob("*.*"):
        texts[f.stem] = f.read_text()

    results = {}
    for text_name, text in sorted(texts.items()):
        ids = tok.encode(text)[:1500]
        prof = PROFILE_FOR_TEXT.get(text_name, "prose")
        conditions = {}

        def run(cond_name, mode="baseline", k=None, allowed=None, margin=MARGIN):
            POLICY.mode, POLICY.k, POLICY.allowed, POLICY.margin = mode, k, allowed, margin
            POLICY.reset_counters()
            nll = teacher_forced_nll(model, ids)
            entry = {"nll": round(nll, 4)}
            if mode == "route":
                per_tok = POLICY.fetches / max(len(ids) - 1, 1)
                entry["fetches_per_token"] = round(per_tok, 3)
                entry["fetch_rate_of_selects"] = round(
                    POLICY.fetches / max(POLICY.total_selects, 1), 4
                )
            conditions[cond_name] = entry
            print(f"  {text_name:8s} {cond_name:14s} nll={nll:.4f} {entry.get('fetches_per_token','')}", flush=True)

        run("baseline")
        run("k6", k=6)
        run("k4", k=4)
        for C in (16, 24, 32):
            run(f"pin{C}", mode="pin", allowed=pinned_sets_from_traces(prof, C))
        run("pin24_cross", mode="pin", allowed=pinned_sets_from_traces(CROSS_PROFILE.get(text_name, "code"), 24))
        run("route24_m02", mode="route", allowed=pinned_sets_from_traces(prof, 24), margin=0.02)
        run("route24_m05", mode="route", allowed=pinned_sets_from_traces(prof, 24), margin=0.05)
        run("route16_m02", mode="route", allowed=pinned_sets_from_traces(prof, 16), margin=0.02)

        base = conditions["baseline"]["nll"]
        for c in conditions.values():
            c["delta_pct_vs_baseline"] = round((c["nll"] - base) / base * 100, 2)
        results[text_name] = conditions

    out = ROOT / "results" / "quality_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
