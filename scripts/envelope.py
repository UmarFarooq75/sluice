#!/usr/bin/env python3
"""
tok/s envelope model for streamed MoE on Apple Silicon, CALIBRATED against our
three measured anchors (E28 OLMoE, qwen36_headtohead, E24 gpt-oss-120b).

The honest claim is NOT a precise predictor — it is a bounded envelope with
stated assumptions that BRACKETS every real number we have. Two regimes:

  * resident   : working set fits the RAM budget -> compute/kernel bound
  * streamed   : working set does not fit -> hit-rate penalty on the miss tail

Core quantity: active_bytes/token = active_params * bytes_per_param(quant).
Empirical law from anchors:  realistic tok/s ~= K / active_params_B,
K in [40 (CPU, Q5, heavy attn) .. 90 (GPU/Metal, Q4, light)].  See calibration.
"""

BUS_GBS = 100.0          # M2 unified memory bandwidth
SSD_GBS = (1.5, 3.4)     # measured F_NOCACHE random read band on this SSD

# bytes per stored weight by GGUF quant (incl. K-quant block overhead)
BPP = {"Q2_K": 0.32, "Q4_K_M": 0.55, "Q5_K_M": 0.69, "Q6_K": 0.82, "Q8_0": 1.06, "MXFP4": 0.53}

# ---- measured anchors: (name, active_params_B, quant, backend, measured tok/s, resident?) ----
ANCHORS = [
    ("OLMoE-7B",        1.3, "Q4_K_M", "CPU-resident",  69.4, True),
    ("OLMoE-7B",        1.3, "Q4_K_M", "Metal-resident",72.9, True),
    ("Qwen3.6-35B-A3B", 3.0, "Q5_K_M", "CPU-stream .91",  8.3, False),
    ("gpt-oss-120b",    5.1, "MXFP4",  "Metal .998",     13.7, True),
    ("gpt-oss-120b",    5.1, "MXFP4",  "CPU .998",       11.1, True),
]

def k_product(active_B, tok_s):
    return active_B * tok_s          # tok/s * active_B  (should sit in ~30..90)

print("== CALIBRATION: does tok/s * active_B stay in a tight band? ==")
for name, a, q, back, tok, res in ANCHORS:
    ab = a * BPP[q]
    print(f"  {name:16s} active {a}B {q:7s} {back:15s} {tok:5.1f} tok/s"
          f" | active_bytes {ab:.2f} GB/tok | K={k_product(a,tok):4.0f} | bus-ceil {BUS_GBS/ab:4.0f}")

def envelope(name, active_B, quant, mtp_speedup=1.0, mla=False):
    ab = active_B * BPP[quant]
    bus_ceil = BUS_GBS / ab
    lo = 40.0 / active_B            # CPU / heavy end
    hi = 90.0 / active_B            # GPU / light end
    if mla: hi *= 1.15             # cheaper attention lifts the ceiling a touch
    lo *= mtp_speedup; hi *= mtp_speedup
    hi = min(hi, bus_ceil)         # never beat the bus
    print(f"\n  {name}  active={active_B}B {quant}  active_bytes={ab:.2f} GB/tok")
    print(f"    bus ceiling (all-resident, perfect kernel): {bus_ceil:5.1f} tok/s")
    print(f"    realistic RESIDENT band:                    {lo:5.1f} - {hi:5.1f} tok/s"
          + (f"  (x{mtp_speedup} MTP)" if mtp_speedup>1 else ""))
    return ab, lo, hi

def fits(dense_core_gb, active_B, quant, n_experts, topk, kv_gb, budget_gb):
    # crude working-set check: dense core + a hot cache big enough for ~0.92 hit + KV
    exp_gb_each = (active_B/topk) * BPP[quant]     # rough per-expert active mass
    # to hold ~all hot experts you need most of the routed mass; report full-resident need
    return dense_core_gb, kv_gb

# MTP is carried-but-UNUSED in mainline llama.cpp GGUF for BOTH deepseek2 and
# glm4moe -> no speculative speedup for any GGUF engine. So mtp=1.0 everywhere.
# on-disk Q4_K_M size (GB) from sourced GGUF repos; usable RAM = total - 4 (OS).
MODELS = [
    # name, total_B, active_B, quant, disk_gb_Q4, mla
    ("Qwen3.6-35B-A3B", 35,  3.0, "Q5_K_M", 26.5, False),   # MEASURED anchor 8.3
    ("GLM-4.5-Air",    106, 12.0, "Q4_K_M", 73.0, False),
    ("DeepSeek-V3",    671, 37.0, "Q4_K_M", 377.0, True),
    ("GLM-4.5 / 4.6",  355, 32.0, "Q4_K_M", 201.0, False),
    ("GLM-5.2",        750, 40.0, "Q4_K_M", 466.0, True),
]
BUDGETS = [16, 32, 64]

def verdict(active_B, quant, disk_gb, budget_gb):
    usable = budget_gb - 4
    ab = active_B * BPP[quant]
    if disk_gb <= usable:                      # fully resident -> compute band
        lo, hi = 40.0/active_B, 90.0/active_B
        return f"FITS resident -> {lo:.0f}-{min(hi,BUS_GBS/ab):.0f} tok/s"
    frac = usable / disk_gb                     # fraction of model that can cache
    if frac < 0.25:                             # cache << model -> colibri regime
        return f"streams ({frac*100:.0f}% cached) -> ~0.1-2 tok/s (colibri)"
    return f"partial ({frac*100:.0f}% cached) -> ~2-{40.0/active_B:.0f} tok/s"

if __name__ == "__main__":
    print("\n== E31 DECISION TABLE (MTP unusable in GGUF for all; MLA aids RAM not speed) ==")
    print(f"{'model':16s} {'tot/act':>9s} {'Q4 disk':>8s} {'act B/tok':>10s} {'busceil':>8s} | 16GB / 32GB / 64GB")
    for name, tot, act, q, disk, mla in MODELS:
        ab = act * BPP[q]
        row = " | ".join(verdict(act, q, disk, b) for b in BUDGETS)
        print(f"{name:16s} {f'{tot}/{act}':>9s} {disk:6.0f}GB {ab:8.2f}GB {BUS_GBS/ab:6.0f} |")
        for b in BUDGETS:
            print(f"    {b}GB: {verdict(act, q, disk, b)}")
