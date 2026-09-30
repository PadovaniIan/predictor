"""Thin, dependency-free client for the public MLB Stats API (statsapi.mlb.com).

No API key, no rate limit published; we keep the request count tiny (~60/day).
Only the standard library is used so GitHub Actions needs no pip install step.
"""
import json, time, urllib.request, urllib.error

BASE = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "mlb-skew-forecast/1.0 (github pages static site)"}


def get(path, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    url = f"{BASE}/{path}" + (f"?{qs}" if qs else "")
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after retries: {url} ({last})")


def teams(season):
    d = get("teams", sportId=1, season=season)
    out = []
    for t in d.get("teams", []):
        out.append({
            "id": t["id"],
            "name": t.get("name"),
            "abbrev": t.get("abbreviation"),
            "league": (t.get("league") or {}).get("name"),
            "division": (t.get("division") or {}).get("name"),
        })
    return sorted(out, key=lambda x: x["name"] or "")


def schedule(season, game_type="R"):
    """Every regular-season game with status + linescore result."""
    d = get("schedule", sportId=1, season=season, gameType=game_type,
            fields=",".join([
                "dates", "date", "games", "gamePk", "gameDate", "status",
                "detailedState", "abstractGameState", "teams", "home", "away",
                "team", "id", "score", "isWinner", "seriesDescription"]))
    games = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            h, a = g["teams"]["home"], g["teams"]["away"]
            games.append({
                "gamePk": g["gamePk"],
                "date": day.get("date"),
                "gameDate": g.get("gameDate"),
                "state": (g.get("status") or {}).get("abstractGameState"),
                "detail": (g.get("status") or {}).get("detailedState"),
                "home_id": h["team"]["id"], "away_id": a["team"]["id"],
                "home_score": h.get("score"), "away_score": a.get("score"),
            })
    games.sort(key=lambda x: (x["gameDate"] or "", x["gamePk"]))
    return games


def team_game_log(team_id, season, group, game_type="R"):
    """Per-game team batting or pitching splits."""
    d = get(f"teams/{team_id}/stats", stats="gameLog", group=group,
            season=season, gameType=game_type, sportId=1)
    rows = []
    for blk in d.get("stats", []):
        for sp in blk.get("splits", []):
            st = sp.get("stat", {}) or {}
            gm = sp.get("game", {}) or {}
            rows.append({
                "gamePk": gm.get("gamePk"),
                "date": sp.get("date"),
                "isHome": sp.get("isHome"),
                "isWin": sp.get("isWin"),
                "opponent_id": (sp.get("opponent") or {}).get("id"),
                "stat": st,
            })
    return rows


def league_pitching_totals(season, game_type="R"):
    """League-wide season pitching totals, for deriving cFIP."""
    d = get("teams/stats", season=season, group="pitching", stats="season",
            sportId=1, gameType=game_type)
    tot = {}
    for blk in d.get("stats", []):
        for sp in blk.get("splits", []):
            for k, v in (sp.get("stat") or {}).items():
                if k == "inningsPitched":
                    tot[k] = tot.get(k, 0.0) + ip_to_float(v)
                elif isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + v
    return tot


def ip_to_float(v):
    """'8.2' means 8 and 2/3 innings."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        whole = int(v); frac = round((float(v) - whole) * 10)
        return whole + (frac / 3.0 if frac in (1, 2) else 0.0)
    s = str(v).strip()
    if not s:
        return 0.0
    if "." not in s:
        return float(s)
    w, f = s.split(".", 1)
    try:
        return float(w) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(f[:1], 0.0)
    except ValueError:
        return 0.0
