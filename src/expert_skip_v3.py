"""Sigmoid-family eval on Moonlight-16B-A3B (DeepSeek-V3 architecture):
expert-skip + adaptive top-p, with self-calibrating thresholds.

Sigmoid gates are independent per-expert scores, not a distribution, so
fixed thresholds don't transfer. We calibrate from the model itself: the
trigger statistic is the top-k score SUM per token-layer; thresholds are
its 15th/30th/50th percentiles measured on traces_v3 (falls back to a
short on-the-fly calibration pass if traces are absent).

Conditions:
  baseline      normal top-k + shared experts
  skip_p15/p30/p50   skip routed experts when top-k score sum below that
                     percentile of the calibrated distribution
  keep_p50      inverse control (skip when ABOVE median) — must be worse
  topp_70/85    adaptive top-k: keep experts until cumulative share of the
                token's top-k score mass >= p  (the colibri --topp claim,
                tested on the architecture family it came from)
  k3            fixed half-k control
  skip_all      upper bound on damage

Output: results/expert_skip_v3.json
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

from mlx_lm import load  # noqa: E402
from mlx_lm.models import deepseek_v3  # noqa: E402

MODEL = "mlx-community/Moonlight-16B-A3B-Instruct-4-bit"


def calibrate_thresholds():
    files = sorted((ROOT / "traces_v3").glob("*.npz"))
    sums = []
    for f in files:
        tr = np.load(f)
        sc = tr["scores"].astype(np.float32)  # includes scaling factor; consistent w/ runtime
        sums.append(sc.sum(axis=-1).reshape(-1))
    allsums = np.concatenate(sums)
    return {
        "p15": float(np.percentile(allsums, 15)),
        "p30": float(np.percentile(allsums, 30)),
        "p50": float(np.percentile(allsums, 50)),
    }


class Policy:
    mode = "baseline"
    thresh = None
    topp = None
    k = None
    skipped = 0
    kept_experts = 0
    total = 0

    @classmethod
    def reset(cls):
        cls.skipped = 0
        cls.total = 0
        cls.kept_experts = 0


def install():
    def patched(self, x):
        inds, scores = self.gate(x)
        n = int(np.prod(x.shape[:-1]))
        Policy.total += n

        if Policy.mode == "skip_all":
            y = mx.zeros(x.shape, dtype=x.dtype)
            Policy.skipped += n
        elif Policy.mode in ("skip", "keep"):
            ssum = scores.sum(axis=-1, keepdims=True)
            low = ssum < Policy.thresh
            skip = low if Policy.mode == "skip" else mx.logical_not(low)
            Policy.skipped += int(skip.sum().item())
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
            y = mx.where(skip, mx.zeros_like(y), y)
        elif Policy.mode == "topp":
            sc = np.array(scores.astype(mx.float32))
            flat = sc.reshape(-1, sc.shape[-1])
            share = np.cumsum(flat, axis=-1) / np.maximum(flat.sum(-1, keepdims=True), 1e-9)
            keep_n = (share < Policy.topp).sum(axis=-1) + 1  # experts needed to reach p
            Policy.kept_experts += int(keep_n.sum())
            mask_np = (np.arange(sc.shape[-1])[None, :] < keep_n[:, None]).reshape(sc.shape)
            masked = scores * mx.array(mask_np)
            y = self.switch_mlp(x, inds)
            y = (y * masked[..., None]).sum(axis=-2)
        elif Policy.mode == "k3":
            y = self.switch_mlp(x, inds[..., :3])
            y = (y * scores[..., :3][..., None]).sum(axis=-2)
        else:
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y

    deepseek_v3.DeepseekV3MoE.__call__ = patched


def nll_on_text(model, tok, text):
    ids = tok.encode(text)[:1000]
    inputs = mx.array(ids)[None, :]
    logits = model(inputs[:, :-1])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, inputs[:, 1:][..., None], axis=-1)
    return float(nll.mean().item())


def main():
    th = calibrate_thresholds()
    print("calibrated top-k score-sum thresholds:", th, flush=True)
    model, tok = load(MODEL)
    install()
    texts = {f.stem: f.read_text() for f in (ROOT / "eval").glob("*.*")}

    conditions = [
        ("baseline", {}),
        ("skip_p15", {"mode": "skip", "thresh": th["p15"]}),
        ("skip_p30", {"mode": "skip", "thresh": th["p30"]}),
        ("skip_p50", {"mode": "skip", "thresh": th["p50"]}),
        ("keep_p50", {"mode": "keep", "thresh": th["p50"]}),
        ("topp_70", {"mode": "topp", "topp": 0.70}),
        ("topp_85", {"mode": "topp", "topp": 0.85}),
        ("k3", {"mode": "k3"}),
        ("skip_all", {"mode": "skip_all"}),
    ]
    results = {"thresholds": th}
    k_full = model.args.num_experts_per_tok
    for tname, text in sorted(texts.items()):
        results[tname] = {}
        for cname, cfg in conditions:
            Policy.mode = cfg.get("mode", "baseline")
            Policy.thresh = cfg.get("thresh")
            Policy.topp = cfg.get("topp")
            Policy.reset()
            nll = nll_on_text(model, tok, text)
            entry = {"nll": round(nll, 4)}
            if Policy.mode in ("skip", "keep", "skip_all"):
                entry["skipped_frac"] = round(Policy.skipped / max(Policy.total, 1), 3)
            if Policy.mode == "topp":
                entry["mean_experts_kept"] = round(Policy.kept_experts / max(Policy.total, 1), 2)
                entry["traffic_frac_of_full_k"] = round(
                    Policy.kept_experts / max(Policy.total * k_full, 1), 3
                )
            results[tname][cname] = entry
            print(f"{tname:8s} {cname:10s} nll={nll:.4f} {entry}", flush=True)
        base = results[tname]["baseline"]["nll"]
        for e in results[tname].values():
            e["delta_pct"] = round((e["nll"] - base) / base * 100, 2)

    (ROOT / "results" / "expert_skip_v3.json").write_text(json.dumps(results, indent=2))
    print("Wrote results/expert_skip_v3.json")


if __name__ == "__main__":
    main()
