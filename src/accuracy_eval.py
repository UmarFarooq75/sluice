"""Accuracy micro-suite: do ANSWERS stay correct under routing restrictions?

NLL (quality_eval.py) measures distribution shift; this measures the thing
users feel: exact-match correctness on 24 deterministic arithmetic/word
problems, generated greedily (max 160 tokens) under each routing condition.

Scoring: last number in the generation must equal the expected answer.
Small n (24) means coarse resolution (~±8%); treat differences <2 problems
as noise. Conditions mirror quality_eval's interesting subset.

Output: results/accuracy_eval.json
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "src"))

from mlx_lm import load, generate  # noqa: E402
from quality_eval import MODEL, POLICY, install_patch, pinned_sets_from_traces  # noqa: E402

PROBLEMS = [
    ("What is 17 + 26? Answer with just the number.", 43),
    ("What is 9 * 13? Answer with just the number.", 117),
    ("What is 144 / 12? Answer with just the number.", 12),
    ("What is 203 - 78? Answer with just the number.", 125),
    ("A shop sells pens at 4 for $3. How much do 20 pens cost in dollars? Answer with just the number.", 15),
    ("Solve for x: 3x + 5 = 26. Answer with just the number.", 7),
    ("A train travels 180 km in 3 hours. What is its speed in km per hour? Answer with just the number.", 60),
    ("What is 15% of 240? Answer with just the number.", 36),
    ("A rectangle is 9 cm long and 4 cm wide. What is its area in square cm? Answer with just the number.", 36),
    ("What is the sum of the first 10 positive integers? Answer with just the number.", 55),
    ("If 5 workers build a wall in 12 days, how many days do 10 workers need at the same rate? Answer with just the number.", 6),
    ("What is 2 to the power of 8? Answer with just the number.", 256),
    ("A bag has 3 red and 7 blue marbles. What is the probability of drawing a red marble, as a fraction like a/b?", "3/10"),
    ("What is the next number in the sequence 2, 6, 18, 54? Answer with just the number.", 162),
    ("Sara has 48 apples and gives away a quarter of them. How many does she keep? Answer with just the number.", 36),
    ("What is 1000 - 637? Answer with just the number.", 363),
    ("A car uses 8 liters per 100 km. How many liters for 350 km? Answer with just the number.", 28),
    ("What is the perimeter of a square with side 13? Answer with just the number.", 52),
    ("What is 7 factorial divided by 6 factorial? Answer with just the number.", 7),
    ("Tom is 3 times as old as Ana. Together they are 48. How old is Tom? Answer with just the number.", 36),
    ("What is the average of 12, 18, and 33? Answer with just the number.", 21),
    ("How many minutes are in 4.5 hours? Answer with just the number.", 270),
    ("What is 25 * 25? Answer with just the number.", 625),
    ("A book costs $18 after a 25% discount. What was the original price in dollars? Answer with just the number.", 24),
]


def extract_answer(text):
    frac = re.findall(r"\b(\d+/\d+)\b", text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return (frac[-1] if frac else None), (nums[-1] if nums else None)


def check(expected, text):
    frac, num = extract_answer(text)
    if isinstance(expected, str):
        return frac == expected
    if num is None:
        return False
    try:
        return abs(float(num) - float(expected)) < 1e-6
    except ValueError:
        return False


def main():
    model, tok = load(MODEL)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._layer_idx = i
    install_patch()

    conditions = {
        "baseline": dict(mode="baseline"),
        "route24_m02": dict(mode="route", allowed=pinned_sets_from_traces("math", 24), margin=0.02),
        "route24_m05": dict(mode="route", allowed=pinned_sets_from_traces("math", 24), margin=0.05),
        "route16_m02": dict(mode="route", allowed=pinned_sets_from_traces("math", 16), margin=0.02),
        "pin24": dict(mode="pin", allowed=pinned_sets_from_traces("math", 24)),
        "k6": dict(mode="baseline", k=6),
    }

    results = {}
    for cname, cfg in conditions.items():
        POLICY.mode = cfg.get("mode", "baseline")
        POLICY.k = cfg.get("k")
        POLICY.allowed = cfg.get("allowed")
        POLICY.margin = cfg.get("margin", 0.02)
        POLICY.reset_counters()
        correct, details = 0, []
        n_gen_tokens = 0
        for q, expected in PROBLEMS:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True
            )
            out = generate(model, tok, prompt, max_tokens=160)
            ok = check(expected, out)
            correct += ok
            n_gen_tokens += len(tok.encode(out))
            details.append({"q": q[:50], "expected": str(expected), "ok": bool(ok)})
        entry = {
            "accuracy": round(correct / len(PROBLEMS), 3),
            "correct": correct,
            "n": len(PROBLEMS),
            "details": details,
        }
        if POLICY.mode == "route":
            entry["fetches_per_gen_token"] = round(POLICY.fetches / max(n_gen_tokens, 1), 2)
        results[cname] = entry
        print(f"{cname:14s} accuracy={correct}/{len(PROBLEMS)}", flush=True)

    out = ROOT / "results" / "accuracy_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
