#!/usr/bin/env python3
# E21 measurement: does the on-disk tensor layout gate our per-miss cost?
#
# GGUF lays tensors type-major: one expert's 13.25MB lives as 3 weight extents
# (~4.4MB, ~1.7GB apart) + 3 bias slivers. Every cache miss pays 6 scattered
# reads. Hypothesis: an install-time expert-major repack (one contiguous
# 13.25MB extent per expert) turns the miss term from random-read-bound into
# sequential-read-bound. This bench prices exactly that gap on THIS machine's
# SSD with the page cache bypassed (F_NOCACHE), against the real model file.
#
# Patterns (per trial, offsets seeded+aligned, uniform over the file):
#   scatter6-serial : 3x4.4MB + 3x12KB at 6 scattered offsets, one thread
#   scatter6-par    : same 6 reads issued from 6 threads (what the engine does)
#   contig1         : one 13.25MB contiguous read (the repacked layout)
#   contig1-qd4     : 4 concurrent contiguous reads (a layer's 4 misses)
#   scatter6-qd4    : 4 concurrent scatter groups, 6 threads each (today's decode)
import os, sys, time, random, threading, statistics

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/gpt-oss-120b-MXFP4.gguf"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
F_NOCACHE = 48  # darwin fcntl
WEIGHT = 4_419_584            # ~4.4MB per weight extent (13.25MB/3)
BIAS = 12_288
EXPERT = 3 * WEIGHT + 3 * BIAS

import fcntl
fd = os.open(MODEL, os.O_RDONLY)
fcntl.fcntl(fd, F_NOCACHE, 1)
fsize = os.fstat(fd).st_size
rng = random.Random(42)

def off(sz):  # aligned random offset with room for sz
    return (rng.randrange(0, fsize - sz) // 16384) * 16384

def read_at(o, sz):
    got = 0
    while got < sz:
        b = os.pread(fd, min(sz - got, 1 << 22), o + got)
        if not b: break
        got += len(b)
    return got

def timed(fn, n=TRIALS):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(ts)
    return med, statistics.quantiles(ts, n=20)[18]  # median, p95

def scatter6_serial():
    for sz in (WEIGHT, WEIGHT, WEIGHT, BIAS, BIAS, BIAS):
        read_at(off(sz), sz)

def par(jobs):
    th = [threading.Thread(target=read_at, args=a) for a in jobs]
    for t in th: t.start()
    for t in th: t.join()

def scatter6_par():
    par([(off(sz), sz) for sz in (WEIGHT, WEIGHT, WEIGHT, BIAS, BIAS, BIAS)])

def contig1():
    read_at(off(EXPERT), EXPERT)

def contig1_qd4():
    par([(off(EXPERT), EXPERT) for _ in range(4)])

def scatter6_qd4():
    jobs = []
    for _ in range(4):
        jobs += [(off(sz), sz) for sz in (WEIGHT, WEIGHT, WEIGHT, BIAS, BIAS, BIAS)]
    par(jobs)

MB = EXPERT / 1e6
for name, fn, experts in (("scatter6-serial", scatter6_serial, 1),
                          ("scatter6-par", scatter6_par, 1),
                          ("contig1", contig1, 1),
                          ("contig1-qd4", contig1_qd4, 4),
                          ("scatter6-qd4", scatter6_qd4, 4)):
    med, p95 = timed(fn)
    per = med / experts
    print(f"{name:16s} med={med:7.2f} ms p95={p95:7.2f} ms  -> {per:6.2f} ms/expert  {MB/ per * 1e3:7.0f} MB/s effective")
os.close(fd)
