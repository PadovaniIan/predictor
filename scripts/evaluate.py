"""Viability scoring.

(a) Walk-forward backtest over the season: for every completed game, rebuild both
    teams' distributions from games strictly BEFORE that game and predict it.
(b) Grade the prospective prediction log.
(c) Compare against baselines, because raw accuracy is meaningless without them.
    A single MLB game is close to a coin flip; published models land ~55-60%.
"""
import json, math, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skewdist as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def fit_platt(rows, iters=60):
    """p_cal = sigmoid(a*logit(p) + b), fitted by IRLS. Corrects the systematic
    over-confidence of the raw distribution comparison."""
    pts = [(math.log(max(min(p, 1 - 1e-6), 1e-6) / (1 - max(min(p, 1 - 1e-6), 1e-6))), 1.0 if y else 0.0)
           for p, y in rows]
    if len(pts) < 50:
        return None
    a, b = 1.0, 0.0
    for _ in range(iters):
        g = [0.0, 0.0]; H = [[1e-6, 0.0], [0.0, 1e-6]]
        for z, y in pts:
            q = 1.0 / (1.0 + math.exp(-(a * z + b)))
            w = max(q * (1 - q), 1e-9); e = q - y
            g[0] += e * z; g[1] += e
            H[0][0] += w * z * z; H[0][1] += w * z; H[1][0] += w * z; H[1][1] += w
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if abs(det) < 1e-12:
            break
        da = (g[0] * H[1][1] - g[1] * H[0][1]) / det
        db = (g[1] * H[0][0] - g[0] * H[1][0]) / det
        a -= da; b -= db
        if abs(da) + abs(db) < 1e-10:
            break
    return {"a": round(a, 6), "b": round(b, 6), "n": len(pts)}


def apply_platt(p, cal):
    if not cal:
        return p
    p = max(min(p, 1 - 1e-6), 1e-6)
    z = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-(cal["a"] * z + cal["b"])))


def metrics(rows):
    """rows = list of (p_pick_home, home_won)"""
    if not rows:
        return None
    n = len(rows)
    acc = sum(1 for p, y in rows if (p >= .5) == bool(y)) / n
    brier = sum((p - (1 if y else 0)) ** 2 for p, y in rows) / n
    ll = -sum(math.log(max(p if y else 1 - p, 1e-9)) for p, y in rows) / n
    bins = []
    for lo in [i / 10 for i in range(10)]:
        sel = [(p, y) for p, y in rows if lo <= p < lo + .1]
        if sel:
            bins.append({"bin": round(lo + .05, 2), "n": len(sel),
                         "pred": round(sum(p for p, _ in sel) / len(sel), 4),
                         "actual": round(sum(1 for _, y in sel if y) / len(sel), 4)})
    return {"n": n, "accuracy": round(acc, 4), "brier": round(brier, 4),
            "log_loss": round(ll, 4), "calibration": bins}


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    ecfg = json.loads(json.dumps(cfg))
    ecfg["distribution"]["grid_points"] = 500          # backtest speed
    raw = json.load(open(os.path.join(DATA, "raw", f"gamelogs_{cfg['season']}.json")))
    series = json.load(open(os.path.join(DATA, "strength_series.json")))["series"]

    strengths, results = {}, {}
    for tid, rows in series.items():
        for r in rows:
            strengths.setdefault(str(tid), {})[str(r["gamePk"])] = r["strength"]
    sched = [g for g in raw["schedule"] if g.get("state") == "Final"
             and g.get("home_score") is not None and g.get("away_score") is not None]
    sched.sort(key=lambda g: (g["gameDate"] or "", g["gamePk"]))

    hist = {}           # team -> chronological strength list
    wl = {}             # team -> [wins, games]
    algo, home_base, log5_base = [], [], []
    minn = cfg["prediction"]["min_games"]
    hfa = float(cfg["prediction"]["home_field_logit"])
    win = cfg.get("window")

    for g in sched:
        h, a = str(g["home_id"]), str(g["away_id"])
        home_won = g["home_score"] > g["away_score"]
        hv, av = hist.get(h, []), hist.get(a, [])
        if len(hv) >= minn and len(av) >= minn:
            A = S.build(hv[-win:] if win else hv, ecfg)
            B = S.build(av[-win:] if win else av, ecfg)
            if A and B:
                algo.append((S.apply_home_field(S.win_probability(A, B), hfa), home_won))
                home_base.append((0.5, home_won))
                hw, hg = wl.get(h, [0, 0]); aw, ag = wl.get(a, [0, 0])
                if hg and ag:
                    ph, pa = hw / hg, aw / ag
                    den = ph * (1 - pa) + pa * (1 - ph)
                    log5_base.append(((ph * (1 - pa) / den) if den else .5, home_won))
        for tid, pk in ((h, g["gamePk"]), (a, g["gamePk"])):
            v = strengths.get(tid, {}).get(str(pk))
            if v is not None:
                hist.setdefault(tid, []).append(v)
        wl.setdefault(h, [0, 0]); wl.setdefault(a, [0, 0])
        wl[h][0] += 1 if home_won else 0; wl[h][1] += 1
        wl[a][0] += 0 if home_won else 1; wl[a][1] += 1

    # ---- logistic recalibration: fit on the first 60%, grade on the last 40%
    cal_out, cal_metrics, cal_raw_holdout = None, None, None
    if len(algo) >= 120:
        cut = int(len(algo) * 0.6)
        fit = fit_platt(algo[:cut])
        hold = algo[cut:]
        if fit:
            cal_metrics = metrics([(apply_platt(p, fit), y) for p, y in hold])
            cal_raw_holdout = metrics(hold)
            cal_out = fit_platt(algo)          # ship a fit that uses everything
            cal_out.update({"fitted_utc": datetime.datetime.now(datetime.timezone.utc)
                            .isoformat(timespec="seconds"),
                            "holdout_brier_raw": cal_raw_holdout["brier"],
                            "holdout_brier_calibrated": cal_metrics["brier"]})
            json.dump(cal_out, open(os.path.join(DATA, "calibration.json"), "w"), indent=1)

    # prospective log
    pro = []
    log_path = os.path.join(DATA, "prediction_log.json")
    if os.path.exists(log_path):
        log = json.load(open(log_path))
        res = {str(g["gamePk"]): g["home_score"] > g["away_score"] for g in sched}
        pro = [(v["p_home"], res[k]) for k, v in log.items() if k in res]

    out = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "synthetic": bool(raw.get("synthetic")),
           "walk_forward": metrics(algo),
           "walk_forward_recalibrated_holdout": cal_metrics,
           "walk_forward_raw_same_holdout": cal_raw_holdout,
           "calibration_fit": cal_out,
           "prospective_log": metrics(pro),
           "baselines": {"always_home_50_50": metrics(home_base),
                         "log5_from_records": metrics(log5_base)},
           "notes": ["Judge viability on Brier score and calibration, not accuracy alone: "
                     "the paper's first principle is that an absolute result is impossible.",
                     "Reference points for MLB single-game win prediction: coin flip 50%, "
                     "home-team-always ~54%, strong public models and betting markets ~57-60%.",
                     "FRC-level accuracy (90-96%) is not attainable in MLB and is not the bar."]}
    json.dump(out, open(os.path.join(DATA, "evaluation.json"), "w"), indent=1)
    wf = out["walk_forward"]
    if wf:
        print(f"[eval] walk-forward n={wf['n']} acc={wf['accuracy']} brier={wf['brier']} ll={wf['log_loss']}")
        if cal_metrics:
            print(f"[eval] holdout raw brier={cal_raw_holdout['brier']} -> "
                  f"recalibrated brier={cal_metrics['brier']} (a={cal_out['a']}, b={cal_out['b']})")
        for k, v in out["baselines"].items():
            if v:
                print(f"       baseline {k:<22} acc={v['accuracy']} brier={v['brier']}")
    else:
        print("[eval] not enough games yet")


if __name__ == "__main__":
    main()
