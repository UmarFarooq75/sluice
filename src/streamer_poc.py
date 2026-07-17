"""PoC expert streamer: OLMoE with routed experts streamed from SSD.

The physical end-to-end test of the whole architecture. The dense trunk
stays resident (loaded normally); routed-expert weights are IGNORED in
RAM and instead loaded per-token from expert_store/ through an LRU cache,
with F_NOCACHE reads so macOS's page cache can't fake the disk.

Measures, per cache capacity (experts/layer):
  - decode tok/s (wall clock)
  - MB read from disk per token, misses per token
  - predicted ceiling from the formula BW / bytes_missed (using the
    microbenchmarked per-file read bandwidth)
  - correctness: NLL on a fixed text vs the resident reference

Modes: exact top-8 service, and margin-routing m=0.02 over the cache
(residents = cached experts) to measure the dial's real speed effect.

Usage:
  python src/streamer_poc.py bench     # SSD microbenchmark only
  python src/streamer_poc.py run      # full sweep
Output: results/streamer_poc.json
"""

import fcntl
import io
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "src"))

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models import olmoe  # noqa: E402

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
STORE = ROOT / "expert_store"
GROUP_SIZE, BITS = 64, 4
PARTS = ["gate_proj", "up_proj", "down_proj"]


def read_nocache(path):
    """Read file bytes bypassing the macOS page cache (F_NOCACHE)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        size = os.fstat(fd).st_size
        buf = os.read(fd, size)
    finally:
        os.close(fd)
    return buf


def load_expert(l, e):
    raw = read_nocache(STORE / f"L{l:02d}_E{e:02d}.npz")
    z = np.load(io.BytesIO(raw))
    out = {}
    for p in PARTS:
        out[p] = tuple(
            mx.array(z[f"{p}.{c}"]) for c in ("weight", "scales", "biases")
        )
    return out, len(raw)


class StreamerState:
    def __init__(self, capacity, margin=None):
        self.capacity = capacity
        self.margin = margin  # None = exact top-8; else margin routing
        self.cache = [dict() for _ in range(16)]  # layer -> expert -> (weights, tick)
        self.tick = 0
        self.misses = 0
        self.uses = 0
        self.bytes_read = 0
        self.load_time = 0.0
        self.enabled = False
        self.counting = True
        self.prefetch = False
        self.prefetched_bytes = 0

    def snapshot(self):
        return (self.misses, self.uses, self.bytes_read, self.load_time)

    def delta(self, snap):
        return {
            "misses": self.misses - snap[0],
            "uses": self.uses - snap[1],
            "bytes": self.bytes_read - snap[2],
            "load_s": self.load_time - snap[3],
        }

    def get(self, l, e):
        self.tick += 1
        if self.counting:
            self.uses += 1
        c = self.cache[l]
        if e in c:
            w, _ = c[e]
            c[e] = (w, self.tick)
            return w
        if self.counting:
            self.misses += 1
        t0 = time.perf_counter()
        w, nbytes = load_expert(l, e)
        if self.counting:
            self.load_time += time.perf_counter() - t0
            self.bytes_read += nbytes
        if self.capacity > 0:
            if len(c) >= self.capacity:
                victim = min(c, key=lambda k: c[k][1])
                del c[victim]
            c[e] = (w, self.tick)
        return w

    def resident_mask(self, l):
        m = np.zeros(64, dtype=bool)
        m[list(self.cache[l].keys())] = True
        return m

    def prefetch_load(self, l, e):
        """Thread-safe-enough insert for the PoC: skip if already resident."""
        c = self.cache[l]
        if e in c or self.capacity <= 0:
            return
        try:
            w, nbytes = load_expert(l, e)
        except Exception:
            return
        self.tick += 1
        if len(c) >= self.capacity:
            victim = min(c, key=lambda k2: c[k2][1])
            del c[victim]
        c[e] = (w, self.tick)
        if self.counting:
            self.prefetched_bytes += nbytes


ST = None
GATES = None  # per-layer gate modules for lookahead prefetch
PREFETCH_POOL = None


def expert_mlp(x, w):
    g = mx.quantized_matmul(x, *w["gate_proj"], transpose=True, group_size=GROUP_SIZE, bits=BITS)
    u = mx.quantized_matmul(x, *w["up_proj"], transpose=True, group_size=GROUP_SIZE, bits=BITS)
    return mx.quantized_matmul(mx.sigmoid(g) * g * u, *w["down_proj"], transpose=True, group_size=GROUP_SIZE, bits=BITS)


def install():
    def patched(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        k = self.top_k
        # stream DECODE steps only (L==1); prefill uses the resident path —
        # the engine answers prefill with expert-major loading (finding 7)
        if not (ST and ST.enabled and L == 1):
            indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=k - 1, axis=-1)[..., :k])
            scores = mx.take_along_axis(routing_weights, indices, axis=-1)
            y = self.switch_mlp(x_flat, indices)
            return (y * scores[..., None]).sum(axis=-2).reshape(B, L, D)

        l = self._layer_idx
        probs = np.array(routing_weights)  # [n_tok, 64]
        n_tok = probs.shape[0]

        # async lookahead prefetch: predict layer l+1's experts from THIS
        # layer's input (83.9% recall, finding 18) and load them on worker
        # threads while this layer computes
        if ST.prefetch and l + 1 < 16 and n_tok == 1:
            nxt = np.array(GATES[l + 1](x_flat))[0]
            pred = np.argsort(-nxt)[:k]
            for e in pred:
                PREFETCH_POOL.submit(ST.prefetch_load, l + 1, int(e))
        y = mx.zeros((n_tok, D), dtype=x_flat.dtype)
        for i in range(n_tok):
            p = probs[i]
            if ST.margin is not None and ST.capacity > 0:
                mask = ST.resident_mask(l)
                if mask.any():
                    rp = np.where(mask, p, -1.0)
                    res_top = np.argsort(-rp)[:k]
                    weakest = rp[res_top[-1]]
                    chosen = set(res_top.tolist())
                    for e in np.argsort(-p)[:k]:
                        if not mask[e] and p[e] > weakest + ST.margin:
                            chosen.add(int(e))
                    chosen = sorted(chosen, key=lambda e: -p[e])[:k]
                else:
                    chosen = np.argsort(-p)[:k].tolist()
            else:
                chosen = np.argsort(-p)[:k].tolist()
            acc = None
            xi = x_flat[i : i + 1]
            for e in chosen:
                w = ST.get(l, int(e))
                out = expert_mlp(xi, w) * float(p[e])
                acc = out if acc is None else acc + out
            y[i : i + 1] = acc
        return y.reshape(B, L, D)

    olmoe.OlmoeSparseMoeBlock.__call__ = patched


def ssd_microbench(n=80):
    files = sorted(STORE.glob("L*.npz"))
    idx = np.random.RandomState(0).choice(len(files), n, replace=False)
    t0 = time.perf_counter()
    total = 0
    for i in idx:
        total += len(read_nocache(files[i]))
    dt = time.perf_counter() - t0
    return {"files": n, "mb": round(total / 1e6, 1), "seconds": round(dt, 2), "mb_per_s": round(total / 1e6 / dt, 1)}


def nll_on_text(model, tok, text):
    ids = tok.encode(text)[:400]
    inputs = mx.array(ids)[None, :]
    logits = model(inputs[:, :-1])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, inputs[:, 1:][..., None], axis=-1)
    return round(float(nll.mean().item()), 4)


def main():
    global ST, GATES, PREFETCH_POOL
    from concurrent.futures import ThreadPoolExecutor

    PREFETCH_POOL = ThreadPoolExecutor(max_workers=4)
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    bench = ssd_microbench()
    print("SSD microbench:", bench, flush=True)
    if mode == "bench":
        return

    model, tok = load(MODEL)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    GATES = [layer.mlp.gate for layer in model.model.layers]
    install()
    eval_text = (ROOT / "eval" / "prose.txt").read_text()

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain in a few sentences why rivers meander."}],
        add_generation_prompt=True,
    )
    results = {"ssd_microbench": bench, "runs": []}

    ST = None  # resident reference
    t0 = time.perf_counter()
    ntoks = sum(1 for _ in stream_generate(model, tok, prompt, max_tokens=48))
    ref = {"mode": "resident_reference", "tok_s": round(ntoks / (time.perf_counter() - t0), 2),
           "nll": nll_on_text(model, tok, eval_text)}
    results["runs"].append(ref)
    print(ref, flush=True)

    sweep = [
        (0, None, False), (8, None, False), (16, None, False), (32, None, False),
        (16, 0.02, False), (32, 0.02, False), (64, None, False),
        (16, None, True), (32, None, True), (16, 0.02, True), (32, 0.02, True),
    ]
    for cap, margin, prefetch in sweep:
        ST = StreamerState(cap, margin)
        ST.prefetch = prefetch
        ST.enabled = True
        # warmup: fill the cache to steady state, untimed and uncounted
        ST.counting = False
        for _ in stream_generate(model, tok, prompt, max_tokens=12):
            pass
        ST.counting = True
        snap = ST.snapshot()
        t0 = time.perf_counter()
        ntoks = sum(1 for _ in stream_generate(model, tok, prompt, max_tokens=48))
        wall = time.perf_counter() - t0
        d = ST.delta(snap)
        ST.counting = False  # quality pass: policy on, counters frozen
        nll = nll_on_text(model, tok, eval_text)
        ST.enabled = False
        mb_tok = d["bytes"] / 1e6 / max(ntoks, 1)
        run = {
            "mode": f"stream_cap{cap}" + (f"_m{margin}" if margin else "") + ("_prefetch" if prefetch else ""),
            "prefetched_mb_per_token": round(ST.prefetched_bytes / 1e6 / max(ntoks, 1), 1) if prefetch else None,
            "tok_s": round(ntoks / wall, 2),
            "nll": nll,
            "misses_per_token": round(d["misses"] / max(ntoks, 1), 1),
            "hit_rate": round(1 - d["misses"] / max(d["uses"], 1), 3),
            "mb_read_per_token": round(mb_tok, 1),
            "load_time_frac": round(d["load_s"] / wall, 3),
            "io_ceiling_tok_s": round(bench["mb_per_s"] / mb_tok, 2) if mb_tok > 0.1 else None,
        }
        results["runs"].append(run)
        print(run, flush=True)

    out = ROOT / "results" / "streamer_poc.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
