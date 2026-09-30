"""Tuning sweep: the viability experiment loop in one command.

Re-runs compute + evaluate across window / trim variants and prints a comparison
table (also written to data/sweep.json). config.json is restored afterwards.
"""
import copy, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config.json")
DATA = os.path.join(ROOT, "data")

VARIANTS = [
    ("full season, 10/10", None, .10, .10),
    ("window 60, 10/10",     60, .10, .10),
    ("window 40, 10/10",     40, .10, .10),
    ("full season, 5/5",   None, .05, .05),
    ("full season, ~1 val", None, .02, .02),
    ("window 40, 5/5",       40, .05, .05),
]


def main():
    base = json.load(open(CFG))
    rows = []
    try:
        for label, win, lo, hi in VARIANTS:
            c = copy.deepcopy(base)
            c["window"] = win
            c["prefilter"]["trim_low_pct"] = lo
            c["prefilter"]["trim_high_pct"] = hi
            json.dump(c, open(CFG, "w"), indent=2)
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compute.py")],
                           capture_output=True, check=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py")],
                           capture_output=True, check=True)
            e = json.load(open(os.path.join(DATA, "evaluation.json")))
            t = [x for x in json.load(open(os.path.join(DATA, "teams.json"))) if x["ok"]]
            wf, hold = e["walk_forward"], e.get("walk_forward_recalibrated_holdout")
            mix = {}
            for x in t:
                mix[x["method"]] = mix.get(x["method"], 0) + 1
            rows.append({"variant": label, "window": win, "trim": [lo, hi],
                         "n": wf["n"], "accuracy": wf["accuracy"], "brier": wf["brier"],
                         "brier_recalibrated_holdout": hold["brier"] if hold else None,
                         "mean_abs_skew_after_trim": round(sum(abs(x["skew_after_trim"]) for x in t) / len(t), 3),
                         "adjustment_mix": mix})
    finally:
        json.dump(base, open(CFG, "w"), indent=2)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compute.py")], capture_output=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py")], capture_output=True)

    json.dump(rows, open(os.path.join(DATA, "sweep.json"), "w"), indent=1)
    print(f"{'variant':<22}{'n':>6}{'acc':>8}{'brier':>9}{'brier cal':>11}{'mean|skew|':>12}  mix")
    for r in rows:
        print(f"{r['variant']:<22}{r['n']:>6}{r['accuracy']:>8.4f}{r['brier']:>9.4f}"
              f"{(r['brier_recalibrated_holdout'] or 0):>11.4f}{r['mean_abs_skew_after_trim']:>12}  {r['adjustment_mix']}")
    best = min(rows, key=lambda r: r["brier"])
    print(f"\nlowest raw Brier: {best['variant']}")


if __name__ == "__main__":
    main()
