"""Build the per-expert store: one .npz per (layer, expert).

Turns OLMoE's monolithic safetensors into 1,024 individually loadable
expert files (~3.3 MB each) — the on-disk layout a streaming engine uses.
Each file holds 9 arrays: {gate,up,down}_proj x {weight,scales,biases}.

Output: expert_store/L{layer:02d}_E{expert:02d}.npz  (~3.4 GB total)
        expert_store/meta.json
"""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
SNAP = next((ROOT / "models/hub/models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots").iterdir())
OUT = ROOT / "expert_store"
N_LAYERS, N_EXPERTS = 16, 64
PARTS = ["gate_proj", "up_proj", "down_proj"]
COMPS = ["weight", "scales", "biases"]


def main():
    OUT.mkdir(exist_ok=True)
    f = safe_open(SNAP / "model.safetensors", framework="numpy")
    sizes = []
    for l in range(N_LAYERS):
        for e in range(N_EXPERTS):
            arrs = {}
            for p in PARTS:
                for c in COMPS:
                    arrs[f"{p}.{c}"] = f.get_tensor(f"model.layers.{l}.mlp.experts.{e}.{p}.{c}")
            path = OUT / f"L{l:02d}_E{e:02d}.npz"
            np.savez(path, **arrs)  # uncompressed: fast loads, honest read sizes
            sizes.append(path.stat().st_size)
        print(f"layer {l} done", flush=True)
    cfg = json.load(open(SNAP / "config.json"))
    meta = {
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "mean_file_mb": round(float(np.mean(sizes)) / 1e6, 3),
        "total_gb": round(float(np.sum(sizes)) / 1e9, 3),
        "quantization": cfg.get("quantization"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
