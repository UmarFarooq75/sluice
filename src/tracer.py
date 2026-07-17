"""Collect MoE routing traces from OLMoE-1B-7B via mlx-lm.

Patches OlmoeSparseMoeBlock.__call__ to record, for every token at every
MoE layer: the top-k expert indices, their gate weights, and the full
router probability distribution (64 experts — cheap and enables top-p /
substitution analysis later).

Output per workload: traces/<workload>.npz containing
  indices  int8   [n_tokens, n_layers, top_k]
  scores   f16    [n_tokens, n_layers, top_k]
  probs    f16    [n_tokens, n_layers, n_experts]
  phase    int8   [n_tokens]            0 = prefill, 1 = decode
  prompt_id int16 [n_tokens]            which prompt in the workload
plus traces/<workload>.meta.json with prompts, model id, timing.

Usage:
  python src/tracer.py                # all workloads
  python src/tracer.py code math     # subset
  python src/tracer.py --prefill     # long-prefill length sweep
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
MAX_TOKENS = 256

import os

os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models import olmoe  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from workloads import LONG_PREFILL_BASE, WORKLOADS  # noqa: E402


class TraceCollector:
    def __init__(self):
        self.rows = []  # one dict per forward call per layer
        self.enabled = False

    def install(self, n_layers):
        collector = self

        def traced_call(self, x):
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
            if collector.enabled:
                collector.rows.append(
                    {
                        "layer": self._layer_idx,
                        "indices": np.array(indices, copy=True),
                        "scores": np.array(scores.astype(mx.float16), copy=True),
                        "probs": np.array(
                            routing_weights.astype(mx.float16), copy=True
                        ),
                        "n_tok": L,
                    }
                )
            y = self.switch_mlp(x_flat, indices)
            y = (y * scores[..., None]).sum(axis=-2)
            return y.reshape(B, L, D)

        olmoe.OlmoeSparseMoeBlock.__call__ = traced_call
        self.n_layers = n_layers

    def drain(self, prompt_id):
        """Convert accumulated per-layer rows into per-token arrays."""
        by_layer = {}
        for r in self.rows:
            by_layer.setdefault(r["layer"], []).append(r)
        self.rows = []
        # Every layer sees the same token sequence in the same order:
        # first call covers the prefill batch (n_tok > 1), then decode steps.
        layers = sorted(by_layer)
        per_layer_idx, per_layer_sc, per_layer_pr, phases = [], [], [], None
        for l in layers:
            calls = by_layer[l]
            idx = np.concatenate([c["indices"] for c in calls], axis=0)
            sc = np.concatenate([c["scores"] for c in calls], axis=0)
            pr = np.concatenate([c["probs"] for c in calls], axis=0)
            per_layer_idx.append(idx)
            per_layer_sc.append(sc)
            per_layer_pr.append(pr)
            if phases is None:
                phases = np.concatenate(
                    [np.zeros(c["n_tok"], np.int8) if c["n_tok"] > 1 else np.ones(1, np.int8) for c in calls]
                )
        n_tok = min(a.shape[0] for a in per_layer_idx)
        return {
            "indices": np.stack([a[:n_tok] for a in per_layer_idx], axis=1).astype(np.int8),
            "scores": np.stack([a[:n_tok] for a in per_layer_sc], axis=1),
            "probs": np.stack([a[:n_tok] for a in per_layer_pr], axis=1),
            "phase": phases[:n_tok],
            "prompt_id": np.full(n_tok, prompt_id, np.int16),
        }


def run_workload(model, tokenizer, collector, name, prompts, max_tokens=MAX_TOKENS):
    parts = []
    t0 = time.time()
    for pid, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        collector.enabled = True
        n_out = 0
        for resp in stream_generate(model, tokenizer, text, max_tokens=max_tokens):
            n_out += 1
        collector.enabled = False
        parts.append(collector.drain(pid))
        print(f"  [{name}] prompt {pid + 1}/{len(prompts)}: {n_out} tokens generated", flush=True)
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    outdir = ROOT / "traces"
    outdir.mkdir(exist_ok=True)
    np.savez_compressed(outdir / f"{name}.npz", **out)
    meta = {
        "model": MODEL,
        "workload": name,
        "prompts": prompts,
        "n_tokens": int(out["indices"].shape[0]),
        "n_layers": int(out["indices"].shape[1]),
        "top_k": int(out["indices"].shape[2]),
        "seconds": round(time.time() - t0, 1),
    }
    (outdir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{name}] saved {out['indices'].shape} -> traces/{name}.npz ({meta['seconds']}s)", flush=True)


def run_prefill_sweep(model, tokenizer, collector):
    """Feed progressively longer documents; only 8 output tokens each.

    Measures how the unique-expert union per layer grows with prompt length.
    Uses this repo's own docs as realistic long text.
    """
    doc = "\n\n".join(
        p.read_text() for p in [ROOT / "vision.md", ROOT / "docs" / "research-findings.md", ROOT / "docs" / "techniques.md"]
    )
    words = doc.split()
    for length in [64, 256, 1024, 3072]:
        chunk = " ".join(words[: length])
        prompts = [LONG_PREFILL_BASE + chunk]
        run_workload(model, tokenizer, collector, f"prefill_{length}w", prompts, max_tokens=8)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    do_prefill = "--prefill" in sys.argv or not args
    names = args or list(WORKLOADS)

    print(f"Loading {MODEL} ...", flush=True)
    model, tokenizer = load(MODEL)
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    collector = TraceCollector()
    collector.install(n_layers)
    cfg = model.args
    print(f"Loaded: {n_layers} layers, {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}", flush=True)

    for name in names:
        if name in WORKLOADS:
            run_workload(model, tokenizer, collector, name, WORKLOADS[name])
    if do_prefill:
        run_prefill_sweep(model, tokenizer, collector)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
