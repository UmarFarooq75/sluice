"""Expert-skip eval on DeepSeek-V2-Lite — the KV-safe compute/I-O skip.

Mechanism (ours, docs/flaws-and-coverage.md #4): on "easy" tokens keep
attention running normally (KV intact) but skip the ROUTED experts and
let the 2 always-resident SHARED experts carry the FFN. Every skipped
token saves top-6 routed-expert loads (~6 x 2.3 MB) and their FLOPs.

Conditions (teacher-forced NLL on eval texts, deltas vs baseline):
  baseline        normal top-6 + shared
  skip_gm_T       skip routed when top-6 softmax mass < T (low mass =
                  router indifferent = shared experts probably suffice)
  keep_gm_T       inverse control: skip when mass >= T (should be WORSE
                  if the trigger signal is meaningful)
  k3              always use top-3 instead of 6 (uniform reduction control)
  skip_all        never use routed experts (upper bound on damage)

Reports: NLL delta + fraction of token-layers skipped (= I/O saved).
Output: results/expert_skip.json
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
from mlx_lm.models import deepseek_v2  # noqa: E402

MODEL = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"


class Policy:
    mode = "baseline"
    thresh = 0.5
    k = None
    skipped = 0
    total = 0

    @classmethod
    def reset(cls):
        cls.skipped = 0
        cls.total = 0


def install():
    def patched(self, x):
        inds, scores = self.gate(x)
        shape = x.shape[:-1]
        n = int(np.prod(shape))
        Policy.total += n

        if Policy.mode == "skip_all":
            y = mx.zeros(x.shape, dtype=x.dtype)
            Policy.skipped += n
        elif Policy.mode in ("skip_gm", "keep_gm"):
            mass = scores.sum(axis=-1, keepdims=True) / self.gate.routed_scaling_factor
            low = mass < Policy.thresh
            skip = low if Policy.mode == "skip_gm" else mx.logical_not(low)
            Policy.skipped += int(skip.sum().item())
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
            y = mx.where(skip, mx.zeros_like(y), y)
        elif Policy.mode == "k3":
            inds3 = inds[..., :3]
            scores3 = scores[..., :3]
            y = self.switch_mlp(x, inds3)
            y = (y * scores3[..., None]).sum(axis=-2)
        else:
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y

    deepseek_v2.DeepseekV2MoE.__call__ = patched


def nll_on_text(model, tok, text):
    ids = tok.encode(text)[:1200]
    inputs = mx.array(ids)[None, :]
    logits = model(inputs[:, :-1])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, inputs[:, 1:][..., None], axis=-1)
    return float(nll.mean().item())


def main():
    model, tok = load(MODEL)
    install()
    texts = {f.stem: f.read_text() for f in (ROOT / "eval").glob("*.*")}

    conditions = [
        ("baseline", {}),
        ("skip_gm_030", {"mode": "skip_gm", "thresh": 0.30}),
        ("skip_gm_045", {"mode": "skip_gm", "thresh": 0.45}),
        ("skip_gm_060", {"mode": "skip_gm", "thresh": 0.60}),
        ("keep_gm_045", {"mode": "keep_gm", "thresh": 0.45}),
        ("k3", {"mode": "k3"}),
        ("skip_all", {"mode": "skip_all"}),
    ]
    results = {}
    for tname, text in sorted(texts.items()):
        results[tname] = {}
        for cname, cfg in conditions:
            Policy.mode = cfg.get("mode", "baseline")
            Policy.thresh = cfg.get("thresh", 0.5)
            Policy.reset()
            nll = nll_on_text(model, tok, text)
            entry = {
                "nll": round(nll, 4),
                "skipped_frac": round(Policy.skipped / max(Policy.total, 1), 3),
            }
            results[tname][cname] = entry
            print(f"{tname:8s} {cname:14s} nll={nll:.4f} skipped={entry['skipped_frac']:.1%}", flush=True)
        base = results[tname]["baseline"]["nll"]
        for e in results[tname].values():
            e["delta_pct"] = round((e["nll"] - base) / base * 100, 2)

    out = ROOT / "results" / "expert_skip.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
