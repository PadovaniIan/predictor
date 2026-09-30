"""Predict upcoming games and append them to an immutable prediction log.

The log is what makes the viability question answerable: predictions are written
BEFORE the game is played, so grading them later is genuinely out of sample.
"""
import json, math, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skewdist as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def rebuild(cfg, series, tid):
    vals = [r["strength"] for r in series.get(str(tid), [])]
    if cfg.get("window"):
        vals = vals[-int(cfg["window"]):]
    if len(vals) < cfg["prediction"]["min_games"]:
        return None
    return S.build(vals, cfg)


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    raw = json.load(open(os.path.join(DATA, "raw", f"gamelogs_{cfg['season']}.json")))
    series = json.load(open(os.path.join(DATA, "strength_series.json")))["series"]
    teams = {str(t["id"]): t for t in raw["teams"]}
    hfa = float(cfg["prediction"]["home_field_logit"])
    cal = None
    cal_path = os.path.join(DATA, "calibration.json")
    if cfg["prediction"].get("calibration", "auto") != "off" and os.path.exists(cal_path):
        cal = json.load(open(cal_path))

    def calibrate(p):
        if not cal:
            return p
        p = max(min(p, 1 - 1e-6), 1e-6)
        z = math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-(cal["a"] * z + cal["b"])))

    cache = {}
    def dist(tid):
        if tid not in cache:
            cache[tid] = rebuild(cfg, series, tid)
        return cache[tid]

    upcoming = [g for g in raw.get("schedule", [])
                if g.get("state") in ("Preview", "Live") or
                (g.get("state") != "Final" and g.get("detail") == "Scheduled")]
    upcoming.sort(key=lambda g: (g["gameDate"] or ""))

    out = []
    for g in upcoming[:120]:
        A, B = dist(g["home_id"]), dist(g["away_id"])
        if not A or not B:
            continue
        p_raw = S.apply_home_field(S.win_probability(A, B), hfa)
        p_home = calibrate(p_raw)
        out.append({
            "gamePk": g["gamePk"], "date": g["date"], "gameDate": g["gameDate"],
            "home_id": g["home_id"], "away_id": g["away_id"],
            "home": teams.get(str(g["home_id"]), {}).get("abbrev"),
            "away": teams.get(str(g["away_id"]), {}).get("abbrev"),
            "p_home": round(p_home, 4), "p_away": round(1 - p_home, 4),
            "p_home_raw": round(p_raw, 4), "calibrated": bool(cal),
            "home_mode": round(A["mode"], 2), "away_mode": round(B["mode"], 2),
            "home_band": [round(A["band_lo"], 2), round(A["band_hi"], 2)],
            "away_band": [round(B["band_lo"], 2), round(B["band_hi"], 2)],
        })

    json.dump({"generated_utc": datetime.datetime.now(datetime.timezone.utc)
               .isoformat(timespec="seconds"), "predictions": out},
              open(os.path.join(DATA, "predictions.json"), "w"), separators=(",", ":"))

    log_path = os.path.join(DATA, "prediction_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    new = 0
    for p in out:
        k = str(p["gamePk"])
        if k not in log:                     # never overwrite: keeps it honest
            log[k] = {"p_home": p["p_home"], "home_id": p["home_id"],
                      "away_id": p["away_id"], "date": p["date"], "made_utc": stamp}
            new += 1
    json.dump(log, open(log_path, "w"), separators=(",", ":"))
    print(f"[predict] calibration={'on' if cal else 'off'}")
    print(f"[predict] {len(out)} upcoming games priced, {new} newly logged")
    for p in out[:8]:
        print(f"  {p['date']} {p['away']}@{p['home']}  p(home)={p['p_home']:.3f}")


if __name__ == "__main__":
    main()
