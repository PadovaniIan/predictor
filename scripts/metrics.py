"""Per-game wOBA, per-game FIP, and the team-strength metric wOBA / FIP."""
import math
from mlb_api import ip_to_float

BAT_KEYS = ["atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
            "intentionalWalks", "hitByPitch", "sacFlies", "strikeOuts", "plateAppearances"]
PIT_KEYS = ["inningsPitched", "homeRuns", "baseOnBalls", "hitByPitch",
            "strikeOuts", "battersFaced", "earnedRuns"]


def num(d, k):
    v = d.get(k, 0)
    if v is None:
        return 0.0
    if k == "inningsPitched":
        return ip_to_float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def woba_numer_denom(bat, w):
    ab = num(bat, "atBats"); h = num(bat, "hits")
    d2 = num(bat, "doubles"); d3 = num(bat, "triples"); hr = num(bat, "homeRuns")
    bb = num(bat, "baseOnBalls"); ibb = num(bat, "intentionalWalks")
    hbp = num(bat, "hitByPitch"); sf = num(bat, "sacFlies")
    s1 = max(0.0, h - d2 - d3 - hr)
    ubb = max(0.0, bb - ibb)
    numer = (w["wBB"] * ubb + w["wHBP"] * hbp + w["w1B"] * s1 +
             w["w2B"] * d2 + w["w3B"] * d3 + w["wHR"] * hr)
    denom = ab + bb - ibb + sf + hbp
    return numer, denom


def fip_numer_ip(pit):
    hr = num(pit, "homeRuns"); bb = num(pit, "baseOnBalls")
    hbp = num(pit, "hitByPitch"); k = num(pit, "strikeOuts")
    return 13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k, num(pit, "inningsPitched")


def derive_fip_constant(league_totals):
    """cFIP = lgERA - (13HR + 3(BB+HBP) - 2K)/lgIP"""
    ip = num(league_totals, "inningsPitched")
    er = num(league_totals, "earnedRuns")
    if ip <= 0:
        return 3.15
    lg_era = 9.0 * er / ip
    raw, _ = fip_numer_ip(league_totals)
    return lg_era - raw / ip


def league_rates(team_games, w):
    """League-average wOBA-per-PA and FIP numerator-per-inning, used as the
    shrinkage prior so a single game cannot blow up the ratio."""
    wn = wd = fn = fi = 0.0
    for g in team_games:
        a, b = woba_numer_denom(g["batting"], w)
        wn += a; wd += b
        c, ip = fip_numer_ip(g["pitching"])
        fn += c; fi += ip
    return {
        "woba": (wn / wd) if wd else 0.315,
        "fip_numer_per_ip": (fn / fi) if fi else 0.0,
        "pa": wd, "ip": fi,
    }


def game_strength(g, cfg, cfip, lg):
    """One team, one game -> (wOBA, FIP, strength). Returns None if unusable."""
    w = cfg["woba_weights"]
    k_pa = float(cfg["shrinkage"]["prior_pa"])
    k_ip = float(cfg["shrinkage"]["prior_innings"])

    numer, denom = woba_numer_denom(g["batting"], w)
    fnum, ip = fip_numer_ip(g["pitching"])
    if denom + k_pa <= 0 or ip + k_ip <= 0:
        return None

    woba = (numer + k_pa * lg["woba"]) / (denom + k_pa)
    fip = (fnum + k_ip * lg["fip_numer_per_ip"]) / (ip + k_ip) + cfip
    fip = max(fip, float(cfg["fip_floor"]))
    if woba <= 0 or not math.isfinite(woba) or not math.isfinite(fip):
        return None
    return {"woba": woba, "fip": fip, "strength": woba / fip}
