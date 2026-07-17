"""Independent check of per-expert GGUF extents: hash expert slices of layer-0
expert tensors via gguf-py's own offset logic, to compare against the driver's
fetched-slot hashes (LLMSTREAM_HASH_FETCH=1)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "llama.cpp", "gguf-py"))
from gguf.gguf_reader import GGUFReader

FNV_OFF, FNV_PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1

def fnv1a(data: bytes) -> int:
    h = FNV_OFF
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK
    return h

path = sys.argv[1]
experts = [int(x) for x in sys.argv[2].split(",")]
reader = GGUFReader(path)
names = {t.name: t for t in reader.tensors}
for tname, tag in [("blk.0.ffn_up_exps.weight", "t0"), ("blk.0.ffn_gate_exps.weight", "t1"), ("blk.0.ffn_down_exps.weight", "t2")]:
    t = names[tname]
    raw = t.data  # raw (possibly quantized) bytes, numpy array
    flat = raw.reshape(-1).view("uint8")
    stride = flat.nbytes // 64
    for e in experts:
        h = fnv1a(flat[e * stride:(e + 1) * stride].tobytes())
        print(f"gguf-py L0 {tag} E{e} {h:016x}  stride={stride}")
