"""E1+E3: lossless levers on the expert store — compression + dedup scan.

E1: Can entropy coding shrink streamed bytes at GB/s decode speeds?
    Compress a sample of expert files with zlib(1,6)/lzma and time
    decompression. Effective bandwidth gain = ratio x (1 if decode >> disk).
E2: Do quantized weight blocks repeat across experts (content dedup)?
    Hash 256-byte blocks of the packed int4 weight arrays across all
    experts; report duplicate fraction (within layer / across layers).

Zero model runs; pure CPU over expert_store/. Output: results/lossless_scan.json
"""

import hashlib
import io
import json
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "expert_store"
SAMPLE_N = 48
BLOCK = 256  # bytes per dedup block


def compression_test():
    files = sorted(STORE.glob("L*.npz"))
    rng = np.random.RandomState(3)
    picks = [files[i] for i in rng.choice(len(files), SAMPLE_N, replace=False)]
    raw = [p.read_bytes() for p in picks]
    total = sum(len(b) for b in raw)
    out = {}
    for name, level in [("zlib1", 1), ("zlib6", 6)]:
        t0 = time.perf_counter()
        comp = [zlib.compress(b, level) for b in raw]
        ct = time.perf_counter() - t0
        t0 = time.perf_counter()
        for c in comp:
            zlib.decompress(c)
        dt = time.perf_counter() - t0
        csize = sum(len(c) for c in comp)
        out[name] = {
            "ratio": round(total / csize, 4),
            "saved_pct": round(100 * (1 - csize / total), 2),
            "compress_MBps": round(total / 1e6 / ct, 1),
            "decompress_MBps": round(total / 1e6 / dt, 1),
        }
    # weights-only entropy check (exclude f16 scales/biases which may differ)
    z = np.load(io.BytesIO(raw[0]))
    wbytes = b"".join(z[k].tobytes() for k in z.files if k.endswith(".weight"))
    sbytes = b"".join(z[k].tobytes() for k in z.files if not k.endswith(".weight"))
    out["weights_only_zlib6_saved_pct"] = round(
        100 * (1 - len(zlib.compress(wbytes, 6)) / len(wbytes)), 2
    )
    out["scales_biases_zlib6_saved_pct"] = round(
        100 * (1 - len(zlib.compress(sbytes, 6)) / len(sbytes)), 2
    )
    out["sampled_files"] = SAMPLE_N
    out["sampled_mb"] = round(total / 1e6, 1)
    return out


def dedup_scan():
    files = sorted(STORE.glob("L*.npz"))
    seen = {}
    total_blocks = 0
    dup_blocks = 0
    dup_cross_layer = 0
    for p in files:
        layer = int(p.name[1:3])
        z = np.load(p)
        for k in z.files:
            if not k.endswith(".weight"):
                continue
            b = z[k].tobytes()
            for off in range(0, len(b) - BLOCK + 1, BLOCK):
                h = hashlib.blake2b(b[off : off + BLOCK], digest_size=12).digest()
                total_blocks += 1
                if h in seen:
                    dup_blocks += 1
                    if seen[h] != layer:
                        dup_cross_layer += 1
                else:
                    seen[h] = layer
    return {
        "block_bytes": BLOCK,
        "total_blocks": total_blocks,
        "duplicate_fraction": round(dup_blocks / max(total_blocks, 1), 5),
        "cross_layer_dup_fraction": round(dup_cross_layer / max(total_blocks, 1), 5),
    }


def main():
    res = {"compression": compression_test()}
    print(json.dumps(res, indent=2), flush=True)
    res["dedup"] = dedup_scan()
    (ROOT / "results" / "lossless_scan.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["dedup"], indent=2))
    print("Wrote results/lossless_scan.json")


if __name__ == "__main__":
    main()
