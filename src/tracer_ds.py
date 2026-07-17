"""Routing tracer for DeepSeek-V2-Lite (fine-grained MoE + 2 shared experts).

Same trace format as tracer.py, into traces_ds/. Differences vs OLMoE:
- 27 layers, layer 0 dense -> 26 MoE layers (we index MoE layers 0..25)
- 64 routed experts, top-6, softmax gate with routed_scaling_factor
- 2 always-on shared experts (the expert-skip fallback path)

RAM note: ~9 GB resident — run alone, reduced workload set.
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models import deepseek_v2  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from workloads import WORKLOADS  # noqa: E402

MODEL = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
MAX_TOKENS = 192
PROMPTS_PER_WORKLOAD = 3


class Collector:
    def __init__(self):
        self.rows = []
        self.enabled = False


COL = Collector()


def install():
    orig = deepseek_v2.MoEGate.__call__

    def patched(self, x):
        gates = x @ self.weight.T
        probs = mx.softmax(gates, axis=-1, precise=True)
        inds, scores = orig(self, x)
        if COL.enabled:
            COL.rows.append(
                {
                    "layer": self._moe_idx,
                    "indices": np.array(inds.reshape(-1, inds.shape[-1]), copy=True),
                    "scores": np.array(scores.reshape(-1, scores.shape[-1]).astype(mx.float16), copy=True),
                    "probs": np.array(probs.reshape(-1, probs.shape[-1]).astype(mx.float16), copy=True),
                    "n_tok": int(np.prod(inds.shape[:-1])),
                }
            )
        return inds, scores

    deepseek_v2.MoEGate.__call__ = patched


def drain(prompt_id):
    by_layer = {}
    for r in COL.rows:
        by_layer.setdefault(r["layer"], []).append(r)
    COL.rows = []
    layers = sorted(by_layer)
    idxs, scs, prs, phases = [], [], [], None
    for l in layers:
        calls = by_layer[l]
        idxs.append(np.concatenate([c["indices"] for c in calls], axis=0))
        scs.append(np.concatenate([c["scores"] for c in calls], axis=0))
        prs.append(np.concatenate([c["probs"] for c in calls], axis=0))
        if phases is None:
            phases = np.concatenate(
                [np.zeros(c["n_tok"], np.int8) if c["n_tok"] > 1 else np.ones(1, np.int8) for c in calls]
            )
    n = min(a.shape[0] for a in idxs)
    return {
        "indices": np.stack([a[:n] for a in idxs], axis=1).astype(np.int8),
        "scores": np.stack([a[:n] for a in scs], axis=1),
        "probs": np.stack([a[:n] for a in prs], axis=1),
        "phase": phases[:n],
        "prompt_id": np.full(n, prompt_id, np.int16),
    }


def main():
    print(f"Loading {MODEL} ...", flush=True)
    model, tok = load(MODEL)
    moe_idx = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE):
            layer.mlp.gate._moe_idx = moe_idx
            moe_idx += 1
    print(f"MoE layers: {moe_idx}", flush=True)
    install()

    outdir = ROOT / "traces_ds"
    outdir.mkdir(exist_ok=True)
    for name in ["code", "math", "prose", "chat"]:
        prompts = WORKLOADS[name][:PROMPTS_PER_WORKLOAD]
        parts = []
        t0 = time.time()
        for pid, prompt in enumerate(prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
            COL.enabled = True
            n = sum(1 for _ in stream_generate(model, tok, text, max_tokens=MAX_TOKENS))
            COL.enabled = False
            parts.append(drain(pid))
            print(f"  [{name}] {pid+1}/{len(prompts)}: {n} tokens", flush=True)
        out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        np.savez_compressed(outdir / f"{name}.npz", **out)
        meta = {
            "model": MODEL, "workload": name, "prompts": prompts,
            "n_tokens": int(out["indices"].shape[0]),
            "n_layers": int(out["indices"].shape[1]),
            "top_k": int(out["indices"].shape[2]),
            "n_experts": 64,
            "seconds": round(time.time() - t0, 1),
        }
        (outdir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [{name}] saved {out['indices'].shape}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
