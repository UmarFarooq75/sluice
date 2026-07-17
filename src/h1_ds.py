"""H1 replication on DeepSeek-V2-Lite: is routing token-determined there too?

Same protocol as h1_token_determinism.py (modal expert set per token id,
held-out table recall) on the fine-grained + shared-expert architecture.
Output: results/h1_ds.json
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
from mlx_lm.models import deepseek_v2  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

MODEL = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
MAX_TOKENS = 224


class Col:
    rows = []
    enabled = False
    n_moe = 0


def install():
    orig = deepseek_v2.MoEGate.__call__

    def patched(self, x):
        inds, scores = orig(self, x)
        if Col.enabled and int(np.prod(inds.shape[:-1])) == 1:
            Col.rows.append({"layer": self._moe_idx, "experts": np.array(inds).reshape(-1)})
        return inds, scores

    deepseek_v2.MoEGate.__call__ = patched


def main():
    model, tok = load(MODEL)
    n_moe = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE):
            layer.mlp.gate._moe_idx = n_moe
            n_moe += 1
    Col.n_moe = n_moe
    install()

    token_events = defaultdict(lambda: defaultdict(list))
    prompts = [p for ws in WORKLOADS.values() for p in ws[:2]]
    for pi, p in enumerate(prompts):
        pids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        Col.enabled = True
        gen_ids = []
        for resp in stream_generate(model, tok, pids, max_tokens=MAX_TOKENS):
            gen_ids.append(resp.token)
        Col.enabled = False
        steps = len(Col.rows) // n_moe
        inputs = ([pids[-1]] + gen_ids)[:steps]
        for t in range(steps):
            tid = int(inputs[t])
            for j in range(n_moe):
                r = Col.rows[t * n_moe + j]
                token_events[tid][r["layer"]].append(frozenset(int(e) for e in r["experts"]))
        Col.rows = []
        print(f"prompt {pi+1}/{len(prompts)}", flush=True)

    k = 6
    per_layer_det, per_layer_rec, counts = defaultdict(list), defaultdict(list), []
    for tid, layers in token_events.items():
        occ = len(next(iter(layers.values())))
        if occ < 5:
            continue
        counts.append(occ)
        for l, sets in layers.items():
            half = len(sets) // 2
            if half == 0 or half == len(sets):
                continue
            freq = defaultdict(int)
            for s in sets[:half]:
                for e in s:
                    freq[e] += 1
            modal = set(sorted(freq, key=freq.get, reverse=True)[:k])
            for s in sets[half:]:
                per_layer_rec[l].append(len(s & modal) / k)
            f2 = defaultdict(int)
            for s in sets:
                for e in s:
                    f2[e] += 1
            modal_all = set(sorted(f2, key=f2.get, reverse=True)[:k])
            for s in sets:
                per_layer_det[l].append(len(s & modal_all) / k)

    det = [float(np.mean(v)) for _, v in sorted(per_layer_det.items())]
    rec = [float(np.mean(v)) for _, v in sorted(per_layer_rec.items())]
    res = {
        "model": MODEL,
        "tokens_analyzed": len(counts),
        "mean_occurrences": round(float(np.mean(counts)), 1) if counts else 0,
        "mean_determinism_at6": round(float(np.mean(det)), 3) if det else None,
        "mean_heldout_table_recall": round(float(np.mean(rec)), 3) if rec else None,
        "det_first_mid_last": [round(det[0], 3), round(det[len(det) // 2], 3), round(det[-1], 3)] if det else None,
        "baselines": {"random": round(6 / 64, 3), "olmoe_determinism": 0.671, "olmoe_recall": 0.595},
    }
    (ROOT / "results" / "h1_ds.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
