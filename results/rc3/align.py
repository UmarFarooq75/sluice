"""RC3 — is it chunk ALIGNMENT, with zero reuse? (senior lead's hypothesis)

THE GAP THIS FILLS. Every RC2 DIFFER leg compared a fresh arm against a reused arm
that prefilled a DIFFERENT NUMBER OF TOKENS, so the two never shared a chunk
boundary:

    S2: fresh prefills 245 -> chunks [128, 117]
        reused prefills 217 -> chunks [128,  89]

Reuse and chunk alignment are therefore perfectly confounded in every RC2 result.
And R0 could not have caught it: its pool legs used a 23-token prompt, so ubatch=128
was ONE chunk. pool x multi-chunk was never covered by anything.

THE TEST: fresh vs fresh. Same tokens, same everything, pool ON, varying ONLY the
chunk boundaries by changing n_ubatch. Zero reuse anywhere. If the hashes split, the
bug is pool chunk-boundary numerics and reuse is fully exonerated.

A_ref  ub=512 -> [245]                one chunk, the reference
A_128  ub=128 -> [128, 117]           two chunks
A_123  ub=123 -> [123, 122]           two chunks, DIFFERENT boundary  <- isolates
                                       boundary position from chunk count
A_64   ub=64  -> [64,64,64,53]        four chunks
A_61   ub=61  -> [61,61,61,61,1]      five chunks, ragged tail

B_ub4_nopool  ub=4, pool OFF -> 62 chunks. Chunking WITHOUT the pool. R0 showed
    ubatch 2/4 keep the per-layer union under the 16-slot cap, so this runs; ubatch 8
    hits D12 at union 18. If this matches A_ref, chunking alone is clean and the
    pool is required.

T4_replicate  the RC2 S3 pair (reused vs fresh, ub=64, pool on) must still DIFFER,
    or RC2 was not reproducible and nothing here means anything.

LOCALIZE ONLY. No fixes.
"""
import importlib.util
import re
import sys
import time
from pathlib import Path

ROOT = Path("/Users/umarfarooq/Desktop/research")
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from harness import compare, void_banner, verdict_line  # noqa: E402

# reuse RC2's engine helpers rather than forking them; argv drives its OUT dir
sys.argv = [sys.argv[0], "rc3"]
_spec = importlib.util.spec_from_file_location("rc2bisect", ROOT / "results" / "rc2" / "bisect.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

OUT = ROOT / "results" / "rc3"
OUT.mkdir(parents=True, exist_ok=True)
NGEN = 200


def chunks(n, ub):
    return [min(ub, n - i * ub) for i in range((n + ub - 1) // ub)]


def main():
    a = B.avail_gb()
    if a < 8.0:
        sys.exit(f"ABORT (protocol #2): {a:.2f} GB avail < 8.0 GB")
    if __import__("subprocess").run(["pgrep", "-f", "csrc/stream_run"],
                                    capture_output=True).returncode == 0:
        sys.exit("ABORT (protocol #1): a stream_run is already running")

    # the exact turn-2 rendering RC2's failing legs used, rebuilt the same way
    t1, b1 = B.server("RC3_t1", [B.hist([("user", B.P1)])], NGEN, B.SHORT_SYS, 1, False)
    rep = B.reply_of(b1[0])
    turn2 = B.hist([("user", B.P1), ("assistant", rep), ("user", B.P2)])

    align = [("A_ref", 512, True), ("A_128", 128, True), ("A_123", 123, True),
             ("A_64", 64, True), ("A_61", 61, True), ("B_ub4_nopool", 4, False)]
    legs = {}
    for name, ub, pool in align:
        legs[name] = B.single(name, turn2, NGEN, B.SHORT_SYS, ub, pool)
        n = legs[name].extra["rendered"] or 0
        print(f"  {name:14} ub={ub:<4} pool={int(pool)} rendered={n} "
              f"chunks={chunks(n, ub)} hash={legs[name].hash}", flush=True)

    # reproducibility anchor: RC2's failing pair must still fail
    t4v, t4d, t4legs, t4f = B.pair("T4_replicate", B.SHORT_SYS, NGEN, 64, True)
    print(f"  T4_replicate   reused-vs-fresh ub=64 pool=1 -> {t4v}", flush=True)

    ref = legs["A_ref"]
    n = ref.extra["rendered"] or 0
    L = [f"RC3 — chunk alignment with ZERO reuse ({time.strftime('%Y-%m-%d %H:%M')})",
         f"avail at start {a:.2f} GB | SLOTS=16 greedy | rendered={n} | CLEAN",
         "All A_*/B_* legs are FRESH single-shots of the SAME tokens. Only the chunk",
         "boundaries differ. If these split, reuse is not involved at all.", "",
         void_banner(list(legs.values()) + t4legs), "",
         f"  {'leg':14} {'ub':>4} {'pool':>5}  {'chunks':<28} verdict vs A_ref"]
    split = []
    for name, ub, pool in align:
        lg = legs[name]
        c = str(chunks(lg.extra["rendered"] or 0, ub))
        if name == "A_ref":
            L.append(f"  {name:14} {ub:>4} {int(pool):>5}  {c:<28} (reference) {lg.hash}")
            continue
        v, d = compare(ref, lg)
        if v == "DIFFER":
            split.append(name)
        L.append(f"  {name:14} {ub:>4} {int(pool):>5}  {c:<28} {v}")
    L += ["", f"  T4_replicate (RC2 S3 pair, must DIFFER): {t4v}  {t4d}"]

    L += ["", "=" * 72]
    if t4v != "DIFFER":
        L.append("STOP — T4 did not reproduce RC2's failing pair. Nothing above is")
        L.append("interpretable; RC2's result is not reproducible and must be re-examined.")
    elif [s for s in split if s != "B_ub4_nopool"]:
        L.append("ALIGNMENT ALONE SPLITS THE HASHES, WITH ZERO REUSE.")
        L.append(f"  differing: {', '.join(split)}")
        L.append("=> REUSE IS FULLY EXONERATED. The bug is prefill-pool chunk-boundary")
        L.append("   numerics: the same tokens produce different logits depending only on")
        L.append("   where the ubatch boundaries fall. Every reuse-vs-fresh divergence")
        L.append("   measured so far (E41b B!=C, RC2 S2/S3) is explained by the two arms")
        L.append("   prefilling different token counts and therefore different chunks.")
        if "B_ub4_nopool" in split:
            L.append("   NOTE: the pool-OFF leg ALSO split -> chunking alone does it, pool")
            L.append("   not required. That widens the blast radius.")
        else:
            L.append("   The pool-OFF leg matched, so the pool is required: chunking alone")
            L.append("   is clean.")
    elif split == ["B_ub4_nopool"]:
        L.append("ONLY the pool-OFF leg split — unexpected; report, do not rationalise.")
    else:
        L.append("ALIGNMENT ALONE DOES NOT SPLIT. All fresh chunkings of the same tokens")
        L.append("agree, so chunk boundaries are not sufficient on their own and REUSE")
        L.append("REMAINS IMPLICATED. The RC2 confound is real but is not the whole story;")
        L.append("the next step is reuse-vs-fresh at MATCHED chunk boundaries.")
    L.append("=" * 72)
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
