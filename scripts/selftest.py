"""Invariant checks on the algorithm implementation. Run after any edit."""
import json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skewdist as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))
fails = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"   <- {detail}"))
    if not cond:
        fails.append(name)

# eq 1-3 against hand-computed values
xs = [2, 4, 4, 4, 5, 5, 7, 9]
check("eq1 mean", abs(S.mean(xs) - 5.0) < 1e-12, S.mean(xs))
check("eq2 population sigma", abs(S.pstdev(xs) - 2.0) < 1e-12, S.pstdev(xs))
check("eq3 skew symmetric set = 0", abs(S.skew([1, 2, 3, 4, 5])) < 1e-12, S.skew([1, 2, 3, 4, 5]))
check("eq3 skew sign on right-tailed set > 0", S.skew([1, 1, 1, 2, 9]) > 0, S.skew([1, 1, 1, 2, 9]))

# NORMINV
check("norm_inv median", abs(S.norm_inv(.5, 10, 2) - 10) < 1e-9)
check("norm_inv 95th pct", abs(S.norm_inv(.95, 0, 1) - 1.6448536) < 1e-5, S.norm_inv(.95, 0, 1))

# pre-filter
kept, lo, hi = S.trim(list(range(100)), .10, .10)
check("trim drops 10/10", (lo, hi, len(kept)) == (10, 10, 80), (lo, hi, len(kept)))

# eq 9 must never pick a worse skew than the base set
rnd = random.Random(3)
worse = 0
for _ in range(200):
    v = [rnd.lognormvariate(4.6, rnd.uniform(.08, .35)) * 100 for _ in range(rnd.randint(20, 160))]
    m, g, adj, _ = S.select_adjustment(v)
    if abs(S.skew(adj)) > abs(g) + 1e-9:
        worse += 1
check("eq9 never selects a worse skew than unadjusted", worse == 0, f"{worse}/200")

# candidate sets stay strictly positive (min cutoff at 0 stays meaningful)
bad = 0
for _ in range(200):
    v = [rnd.lognormvariate(0, .4) * 100 for _ in range(60)]
    for k, arr in S._candidates(v, S.skew(v)).items():
        if any(t <= 0 or not math.isfinite(t) for t in arr):
            bad += 1
check("all candidate adjustments strictly positive", bad == 0, bad)

# curve construction invariants
v = [rnd.lognormvariate(4.6, .18) for _ in range(140)]
v = [x * 100 / (sum(v) / len(v)) for x in v]
d = S.build(v, cfg)
damp = [min(a, b) for a, b in zip(d["y_normal"], d["y_skew"])]
peak = d["grid"][max(range(len(damp)), key=lambda i: damp[i])]
check("damped peak == reported mode", abs(peak - d["mode"]) <= 2 * d["step"], (peak, d["mode"]))
check("band_lo < mode < band_hi", d["band_lo"] < d["mode"] < d["band_hi"],
      (d["band_lo"], d["mode"], d["band_hi"]))
check("scaled curves peak at 1", abs(max(d["y_normal"]) - 1) < 1e-9 and abs(max(d["y_skew"]) - 1) < 1e-9)
check("density integrates to 1", abs(sum(d["density"]) * d["step"] - 1) < 1e-6,
      sum(d["density"]) * d["step"])
check("grid starts at the zero cutoff", d["grid"][0] == 0)
check("works with only 4 data points", S.build([90, 100, 104, 112], cfg) is not None)

# pairwise probability
v2 = [x * 1.06 for x in v]
d2 = S.build(v2, cfg)
p, q = S.win_probability(d2, d), S.win_probability(d, d2)
check("win probability is antisymmetric", abs(p + q - 1) < 5e-3, (p, q))
check("stronger team is favoured", p > 0.5, p)
check("self matchup is a coin flip", abs(S.win_probability(d, d) - .5) < 5e-3, S.win_probability(d, d))
check("home-field logit shift raises P(home)", S.apply_home_field(.5, .16) > .5)

print(("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
