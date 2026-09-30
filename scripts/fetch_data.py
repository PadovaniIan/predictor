"""Pull everything needed from the MLB Stats API into one cached raw file.

  data/raw/gamelogs_<season>.json

~62 HTTP requests per run (30 teams x 2 stat groups + schedule + league totals),
so it is cheap enough to run on a daily cron for the whole season. Existing
rows are merged, so a run only ever adds the games that have been played since
the last refresh.
"""
import json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlb_api as api
from metrics import BAT_KEYS, PIT_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slim(stat, keys):
    return {k: stat.get(k, 0) for k in keys}


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    season, gt = cfg["season"], cfg["game_type"]
    raw_dir = os.path.join(ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"gamelogs_{season}.json")

    store = {"season": season, "teams": [], "games": {}, "schedule": [],
             "league_pitching_totals": {}, "synthetic": False}
    if os.path.exists(path):
        try:
            store.update(json.load(open(path)))
        except json.JSONDecodeError:
            pass

    print(f"[fetch] season {season} gameType {gt}")
    store["teams"] = api.teams(season)
    store["schedule"] = api.schedule(season, gt)
    store["league_pitching_totals"] = api.league_pitching_totals(season, gt)
    print(f"[fetch] {len(store['teams'])} teams, {len(store['schedule'])} scheduled games")

    added = 0
    for t in store["teams"]:
        tid = str(t["id"])
        prev = {str(g["gamePk"]): g for g in store["games"].get(tid, [])}
        bat = {str(r["gamePk"]): r for r in api.team_game_log(t["id"], season, "hitting", gt)}
        pit = {str(r["gamePk"]): r for r in api.team_game_log(t["id"], season, "pitching", gt)}
        merged = dict(prev)
        for pk in sorted(set(bat) & set(pit)):
            b, p = bat[pk], pit[pk]
            if pk not in merged:
                added += 1
            merged[pk] = {
                "gamePk": int(pk), "date": b.get("date"),
                "isHome": b.get("isHome"), "isWin": b.get("isWin"),
                "opponent_id": b.get("opponent_id"),
                "batting": slim(b["stat"], BAT_KEYS),
                "pitching": slim(p["stat"], PIT_KEYS),
            }
        store["games"][tid] = sorted(merged.values(), key=lambda g: (g["date"] or "", g["gamePk"]))
        print(f"  {t['abbrev']:<4} {len(store['games'][tid]):>3} games")

    store["fetched_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    store["synthetic"] = False
    with open(path, "w") as f:
        json.dump(store, f, separators=(",", ":"))
    print(f"[fetch] wrote {path} (+{added} new team-games)")


if __name__ == "__main__":
    main()
