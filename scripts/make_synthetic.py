"""Offline demo / regression data.

Writes data/raw/gamelogs_<season>.json in EXACTLY the format fetch_data.py
produces, with synthetic:true, so compute/predict/evaluate can be exercised
without network access and the site renders before the first real refresh.
Latent team talent + a 5-man rotation effect are baked in so the walk-forward
backtest has genuine signal to find.
"""
import json, os, random, datetime, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = [("ARI","Arizona Diamondbacks"),("ATL","Atlanta Braves"),("BAL","Baltimore Orioles"),
 ("BOS","Boston Red Sox"),("CHC","Chicago Cubs"),("CWS","Chicago White Sox"),
 ("CIN","Cincinnati Reds"),("CLE","Cleveland Guardians"),("COL","Colorado Rockies"),
 ("DET","Detroit Tigers"),("HOU","Houston Astros"),("KC","Kansas City Royals"),
 ("LAA","Los Angeles Angels"),("LAD","Los Angeles Dodgers"),("MIA","Miami Marlins"),
 ("MIL","Milwaukee Brewers"),("MIN","Minnesota Twins"),("NYM","New York Mets"),
 ("NYY","New York Yankees"),("OAK","Athletics"),("PHI","Philadelphia Phillies"),
 ("PIT","Pittsburgh Pirates"),("SD","San Diego Padres"),("SF","San Francisco Giants"),
 ("SEA","Seattle Mariners"),("STL","St. Louis Cardinals"),("TB","Tampa Bay Rays"),
 ("TEX","Texas Rangers"),("TOR","Toronto Blue Jays"),("WSH","Washington Nationals")]


def ip_str(outs):
    return f"{outs // 3}.{outs % 3}"


def main(season=2025, seed=7):
    rnd = random.Random(seed)
    teams = [{"id": 100 + i, "name": n, "abbrev": a,
              "league": "American League" if i % 2 else "National League",
              "division": f"Division {i % 6 + 1}"} for i, (a, n) in enumerate(NAMES)]
    off = {t["id"]: rnd.gauss(0.318, 0.016) for t in teams}          # wOBA talent
    rot = {t["id"]: [rnd.gauss(0.0, 0.55) for _ in range(5)] for t in teams}
    stf = {t["id"]: rnd.gauss(0.0, 0.42) for t in teams}             # staff talent

    pairs = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    plan = []
    for i, j in pairs:
        reps = 6 if (i % 6) == (j % 6) else 3
        for k in range(reps):
            h, a = (i, j) if k % 2 == 0 else (j, i)
            plan.append((h, a))
    rnd.shuffle(plan)
    start = datetime.date(season, 3, 27)

    games, sched, pk = {str(t["id"]): [] for t in teams}, [], 700000
    per_day = max(1, len(plan) // 183)
    for idx, (hi, ai) in enumerate(plan):
        pk += 1
        day = (start + datetime.timedelta(days=idx // per_day)).isoformat()
        H, A = teams[hi]["id"], teams[ai]["id"]
        lines, runs = {}, {}
        for me, opp in ((H, A), (A, H)):
            sp = rot[opp][rnd.randrange(5)] + stf[opp]               # who started for them
            pa = rnd.randint(33, 43)
            q = off[me] - 0.010 * sp
            p1 = max(.02, rnd.gauss(0.155 + (q - .318) * 1.6, .035))
            p2 = max(.005, rnd.gauss(0.048 + (q - .318) * 0.7, .017))
            p3 = max(.0, rnd.gauss(0.005, .004))
            ph = max(.0, rnd.gauss(0.030 + (q - .318) * 1.1, .015))
            pb = max(.01, rnd.gauss(0.083, .022))
            pk_ = max(.05, rnd.gauss(0.225 + 0.011 * sp, .045))
            s1 = sum(1 for _ in range(pa) if rnd.random() < p1)
            d2 = sum(1 for _ in range(pa) if rnd.random() < p2)
            d3 = sum(1 for _ in range(pa) if rnd.random() < p3)
            hr = sum(1 for _ in range(pa) if rnd.random() < ph)
            bb = sum(1 for _ in range(pa) if rnd.random() < pb)
            so = sum(1 for _ in range(pa) if rnd.random() < pk_)
            hbp = 1 if rnd.random() < .09 else 0
            ibb = 1 if rnd.random() < .06 else 0
            sf = 1 if rnd.random() < .22 else 0
            hits = s1 + d2 + d3 + hr
            lines[me] = {"batting": {"atBats": max(hits, pa - bb - hbp - sf),
                "hits": hits, "doubles": d2, "triples": d3, "homeRuns": hr,
                "baseOnBalls": bb, "intentionalWalks": min(ibb, bb), "hitByPitch": hbp,
                "sacFlies": sf, "strikeOuts": so, "plateAppearances": pa}}
            lam = max(0.4, 0.47 * s1 + 0.78 * d2 + 1.05 * d3 + 1.42 * hr + 0.32 * bb - 0.9)
            r = 0
            for _ in range(14):
                if rnd.random() < min(.95, lam / 14):
                    r += 1
            runs[me] = r
        if runs[H] == runs[A]:
            runs[H] += 1                                             # no ties
        for me, opp in ((H, A), (A, H)):
            outs = rnd.randint(24, 27) if runs[me] > runs[opp] else 27
            ob = lines[opp]["batting"]
            lines[me]["pitching"] = {"inningsPitched": ip_str(outs),
                "homeRuns": ob["homeRuns"], "baseOnBalls": ob["baseOnBalls"],
                "hitByPitch": ob["hitByPitch"], "strikeOuts": ob["strikeOuts"],
                "battersFaced": ob["plateAppearances"], "earnedRuns": max(0, runs[opp] - (1 if rnd.random() < .3 else 0))}
            games[str(me)].append({"gamePk": pk, "date": day, "isHome": me == H,
                "isWin": runs[me] > runs[opp], "opponent_id": opp,
                "batting": lines[me]["batting"], "pitching": lines[me]["pitching"]})
        sched.append({"gamePk": pk, "date": day, "gameDate": day + "T18:05:00Z",
                      "state": "Final", "detail": "Final", "home_id": H, "away_id": A,
                      "home_score": runs[H], "away_score": runs[A]})

    # leave the last 45 games unplayed so the "upcoming" view has content
    for g in sched[-45:]:
        g.update({"state": "Preview", "detail": "Scheduled", "home_score": None, "away_score": None})
    live = {g["gamePk"] for g in sched if g["state"] == "Final"}
    for tid in games:
        games[tid] = [g for g in games[tid] if g["gamePk"] in live]

    tot = {}
    for gs in games.values():
        for g in gs:
            for k, v in g["pitching"].items():
                if k == "inningsPitched":
                    w, f = str(v).split("."); tot[k] = tot.get(k, 0.0) + int(w) + int(f) / 3
                else:
                    tot[k] = tot.get(k, 0) + v

    store = {"season": season, "teams": teams, "games": games, "schedule": sched,
             "league_pitching_totals": tot, "synthetic": True,
             "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.join(ROOT, "data", "raw"), exist_ok=True)
    p = os.path.join(ROOT, "data", "raw", f"gamelogs_{season}.json")
    json.dump(store, open(p, "w"), separators=(",", ":"))
    counts = sorted(len(v) for v in games.values())
    print(f"[synthetic] {p}: {len(sched)} games, games/team {counts[0]}-{counts[-1]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
