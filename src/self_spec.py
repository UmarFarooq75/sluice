"""Cache-restricted self-speculation — measurement of the novel lossless idea.

Idea: the resident-expert-restricted model IS the draft model. Draft k
tokens with zero disk I/O (resident experts only), verify all k in one
full-model forward whose expert-loads amortize across the window.
Output is provably identical to the full model (standard speculative
verification). Question: is the draft's agreement rate high enough, and
does the expert-union grow slowly enough, for a real win?

Part A (model runs, OLMoE): teacher-forced agreement — generate the
  reference continuation with the FULL model (greedy), then run the
  RESTRICTED model over the same prefix and count argmax agreement.
  Restrictions tested: pin16/24/32 (task-profile pinned sets = what a
  warm cache holds; zero-disk drafts).

Part B (traces): expert-union growth over sliding windows of w
  consecutive decode tokens (w = 1,2,4,8) per layer -> bytes per
  verification forward.

Combined estimate: expected accepted tokens per window E = sum_{i<=w} a^i
  (a = agreement rate); disk bytes per ACCEPTED token under self-spec =
  union_bytes(w+1) x miss_rate / E, vs baseline = top_k x miss_rate x
  expert_MB. Report the multiplier.

Output: results/self_spec.json
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

from mlx_lm import load, generate  # noqa: E402
from quality_eval import MODEL, POLICY, install_patch, pinned_sets_from_traces  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

EXPERT_MB = 3.54
TOP_K = 8
GEN_TOKENS = 96


def teacher_forced_argmax_agreement(model, tok, ids):
    """Fraction of positions where restricted model's argmax == actual next id."""
    inputs = mx.array(ids)[None, :]
    logits = model(inputs[:, :-1])
    pred = np.array(mx.argmax(logits, axis=-1))[0]
    tgt = np.array(ids[1:])
    return float((pred == tgt).mean())


def part_a(model, tok):
    prompts = [WORKLOADS[w][0] for w in ["code", "math", "prose", "chat"]]
    profiles = ["code", "math", "prose", "chat"]
    results = {}
    refs = []
    POLICY.mode = "baseline"
    POLICY.allowed = None
    for p in prompts:
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        out = generate(model, tok, prompt_ids, max_tokens=GEN_TOKENS)
        out_ids = tok.encode(out)
        if out_ids and out_ids[0] == tok.bos_token_id:
            out_ids = out_ids[1:]
        refs.append(list(prompt_ids) + out_ids)
    for cap in (16, 24, 32):
        agrees = []
        for ids, prof in zip(refs, profiles):
            POLICY.mode = "pin"
            POLICY.k = None
            POLICY.allowed = pinned_sets_from_traces(prof, cap)
            agrees.append(teacher_forced_argmax_agreement(model, tok, ids))
        POLICY.mode = "baseline"
        POLICY.allowed = None
        results[f"pin{cap}_draft_agreement"] = {
            "per_workload": [round(a, 3) for a in agrees],
            "mean": round(float(np.mean(agrees)), 3),
        }
        print(f"pin{cap}: agreement {results[f'pin{cap}_draft_agreement']}", flush=True)
    return results


def part_b():
    out = {}
    for w in (2, 4, 8):
        unions = []
        for f in sorted((ROOT / "traces").glob("*.npz")):
            if f.stem.startswith("prefill"):
                continue
            tr = np.load(f)
            idx = tr["indices"][tr["phase"] == 1]
            n_tok, n_layers, k = idx.shape
            for l in range(n_layers):
                for i in range(0, n_tok - w, w):
                    unions.append(len(np.unique(idx[i : i + w, l, :])))
        out[f"w{w}"] = {
            "mean_union_experts_per_layer": round(float(np.mean(unions)), 2),
            "naive_would_be": w * TOP_K,
            "sublinearity": round(float(np.mean(unions)) / (w * TOP_K), 3),
        }
    return out


def combined_estimate(a_results, b_results, miss_rate=0.5):
    est = {}
    for cap_key, a in [(k, v["mean"]) for k, v in a_results.items()]:
        for w_key, u in b_results.items():
            w = int(w_key[1:])
            expected_accepted = sum(a ** i for i in range(1, w + 1)) + 1  # +1 bonus token
            union = u["mean_union_experts_per_layer"]
            bytes_per_accepted = union * miss_rate * EXPERT_MB / expected_accepted
            baseline = TOP_K * miss_rate * EXPERT_MB
            est[f"{cap_key}|{w_key}"] = {
                "expected_accepted_per_window": round(expected_accepted, 2),
                "disk_MB_per_accepted_token": round(bytes_per_accepted, 2),
                "baseline_MB_per_token": round(baseline, 2),
                "disk_reduction_x": round(baseline / bytes_per_accepted, 2),
            }
    return est


def main():
    model, tok = load(MODEL)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    install_patch()
    a = part_a(model, tok)
    b = part_b()
    res = {"draft_agreement": a, "union_growth": b, "estimate_miss50pct": combined_estimate(a, b)}
    (ROOT / "results" / "self_spec.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["estimate_miss50pct"], indent=2)[:1500])
    print("Wrote results/self_spec.json")


if __name__ == "__main__":
    main()
