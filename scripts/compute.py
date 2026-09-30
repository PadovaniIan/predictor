"""raw game logs -> per-game strength series -> per-team skewed distributions."""
import json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skewdist as S
from metrics import derive_fip_constant, league_rates, game_strength

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load(cfg):
    p = os.path.join(DATA, "raw", f"gamelogs_{cfg['season']}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} - run scripts/fetch_data.py first")
    return json.load(open(p))


def build_series(raw, cfg):
    """Per-game wOBA / FIP / strength for every team-game, plus league context."""
    all_games = [g for gs in raw["games"].values() for g in gs]
    lg = league_rates(all_games, cfg["woba_weights"])
    cfip = (cfg.get("fip_constant") if cfg.get("fip_constant") is not None
            else derive_fip_constant(raw.get("league_pitching_totals") or {}))
    if not raw.get("league_pitching_totals"):
        # fall back to summing the per-game pitching lines we already have
        tot = {}
        for g in all_games:
            for k, v in g["pitching"].items():
                if k == "inningsPitched":
                    from mlb_api import ip_to_float
                    tot[k] = tot.get(k, 0.0) + ip_to_float(v)
                elif isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + v
        cfip = cfg.get("fip_constant") or derive_fip_constant(tot)

    series = {}
    for tid, games in raw["games"].items():
        rows = []
        for g in games:
            m = game_strength(g, cfg, cfip, lg)
            if not m:
                continue
            rows.append({"gamePk": g["gamePk"], "date": g["date"],
                         "isHome": g.get("isHome"), "isWin": g.get("isWin"),
                         "opp": g.get("opponent_id"),
                         "woba": round(m["woba"], 5), "fip": round(m["fip"], 4),
                         "strength_raw": m["strength"]})
        series[tid] = rows

    flat = [r["strength_raw"] for rows in series.values() for r in rows]
    lg_mean = (sum(flat) / len(flat)) if flat else 1.0
    scale = (100.0 / lg_mean) if (cfg.get("relative_scale", True) and lg_mean > 0) else 1.0
    for rows in series.values():
        for r in rows:
            r["strength"] = round(r["strength_raw"] * scale, 4)
            del r["strength_raw"]
    return series, {"fip_constant": round(cfip, 4), "league_woba": round(lg["woba"], 5),
                    "league_fip_numer_per_ip": round(lg["fip_numer_per_ip"], 5),
                    "league_mean_strength_raw": round(lg_mean, 6),
                    "relative_scale_factor": round(scale, 4),
                    "team_games": len(flat)}


def curve_points(dist, k=140):
    """Downsample the three curves for the browser."""
    n = len(dist["grid"])
    idx = [int(i * (n - 1) / (k - 1)) for i in range(k)]
    return {
        "x": [round(dist["grid"][i], 3) for i in idx],
        "normal": [round(dist["y_normal"][i], 5) for i in idx],
        "skew": [round(dist["y_skew"][i], 5) for i in idx],
        "damped": [round(min(dist["y_normal"][i], dist["y_skew"][i]), 5) for i in idx],
        "density": [round(dist["density"][i], 8) for i in idx],
        "step": round(dist["grid"][idx[1]] - dist["grid"][idx[0]], 6),
    }


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    raw = load(cfg)
    series, ctx = build_series(raw, cfg)
    win = cfg.get("window")

    teams_meta = {str(t["id"]): t for t in raw["teams"]}
    teams_out, dists_out = [], {}
    for tid, rows in sorted(series.items()):
        vals = [r["strength"] for r in rows]
        if win:
            vals = vals[-int(win):]
        d = S.build(vals, cfg)
        t = teams_meta.get(tid, {"id": int(tid), "name": tid, "abbrev": tid})
        rec = {"id": int(tid), "name": t.get("name"), "abbrev": t.get("abbrev"),
               "league": t.get("league"), "division": t.get("division"),
               "games": len(rows), "ok": bool(d)}
        if d:
            rec.update({
                "n_used": d["n_used"], "trimmed": [d["trimmed_low"], d["trimmed_high"]],
                "mu": round(d["mu"], 3), "sigma": round(d["sigma"], 3),
                "skew_before_trim": round(d["skew_before_trim"], 4),
                "skew_after_trim": round(d["skew_after_trim"], 4),
                "method": d["method"], "skew_of_adjusted": round(d["skew_of_adjusted"], 4),
                "sigma_adj": round(d["sigma_adj"], 5), "sigma_adj_source": d["sigma_adj_source"],
                "mode": round(d["mode"], 3), "band_lo": round(d["band_lo"], 3),
                "band_hi": round(d["band_hi"], 3), "band_source": d["band_source"],
                "cutoff": round(d["x_max_cutoff"], 3),
                "candidate_skews": {k: round(v, 4) for k, v in d["candidate_skews"].items()},
            })
            dists_out[tid] = curve_points(d)
        teams_out.append(rec)

    teams_out.sort(key=lambda r: -(r.get("mode") or 0))
    for i, r in enumerate(teams_out, 1):
        r["rank"] = i

    played = [g for g in raw.get("schedule", []) if g.get("state") == "Final"]
    meta = {
        "season": cfg["season"], "synthetic": bool(raw.get("synthetic")),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "data_fetched_utc": raw.get("fetched_utc"),
        "games_final": len(played), "games_scheduled": len(raw.get("schedule", [])),
        "context": ctx, "config": cfg,
    }
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    json.dump(teams_out, open(os.path.join(DATA, "teams.json"), "w"), separators=(",", ":"))
    json.dump(dists_out, open(os.path.join(DATA, "distributions.json"), "w"), separators=(",", ":"))
    json.dump({"season": cfg["season"], "series": series},
              open(os.path.join(DATA, "strength_series.json"), "w"), separators=(",", ":"))
    print(f"[compute] cFIP={ctx['fip_constant']} lgwOBA={ctx['league_woba']} "
          f"scale={ctx['relative_scale_factor']} teams_ok={sum(1 for r in teams_out if r['ok'])}")
    for r in teams_out[:5]:
        print(f"  {r['rank']:>2}. {r['abbrev']:<4} mode={r.get('mode')} "
              f"band=[{r.get('band_lo')},{r.get('band_hi')}] method={r.get('method')} "
              f"skew {r.get('skew_before_trim')}->{r.get('skew_after_trim')}")


if __name__ == "__main__":
    main()
