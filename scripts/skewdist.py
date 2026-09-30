"""Skewed Distribution Algorithm for Practiced Task Performance Forecasting
   (Padovani, v2.1.2025) - MLB port.

Faithful implementation of the paper's pipeline:
  eq(1) mean, eq(2) population stdev, eq(3) skew
  eq(6)(7)(8) three candidate skew adjustments + the unadjusted base set
  eq(9)  pick the candidate whose skew is closest to zero
  eq(10) approximate the adjusted standard deviation
  scaled normal distribution + scaled skew-adjusted distribution on one x-axis
  three curve intersections -> damped mode and variance bounds
  eq(5)  probability as a ratio of integrals

Pure standard library so it runs in CI with no dependencies.
"""
import math

# ---------------------------------------------------------------- statistics

def mean(xs):                                        # eq (1)
    return sum(xs) / len(xs)


def pstdev(xs):                                      # eq (2)  population sigma
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def skew(xs):                                        # eq (3)
    n = len(xs)
    if n < 3:
        return 0.0
    m, s = mean(xs), pstdev(xs)
    if s == 0:
        return 0.0
    return n / ((n - 1) * (n - 2)) * sum(((x - m) / s) ** 3 for x in xs)


def normal_pdf(x, mu, sigma):                        # eq (4)
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def norm_inv(p, mu, sigma):
    """NORMINV. Acklam's rational approximation, |err| < 1.15e-9."""
    if p <= 0: return float("-inf")
    if p >= 1: return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= 1 - pl:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return mu + sigma * x

# ------------------------------------------------------------- pre-filtering

def trim(xs, low_pct, high_pct):
    """Drop the lowest low_pct and highest high_pct of values (the paper's
    experimentally-optimized pre-filter; here 10%/10% per the MLB spec)."""
    s = sorted(xs)
    n = len(s)
    lo = int(math.floor(n * low_pct))
    hi = int(math.floor(n * high_pct))
    kept = s[lo:n - hi] if n - hi > lo else s
    return kept, lo, hi

# ------------------------------------------------------- skew adjustment set

def _candidates(xs, gamma):
    """eq (6)(7)(8). Branch on the sign of the skew, as the paper requires."""
    out = {"base": list(xs)}
    pos = gamma >= 0
    try:
        out["sq"] = [math.sqrt(x) for x in xs] if pos else [x ** 2 for x in xs]
    except (ValueError, OverflowError):
        pass
    if pos:
        if all(x > 0 for x in xs):
            out["lg"] = [math.log10(x) for x in xs]
    else:
        try:
            out["lg"] = [x ** 3 for x in xs]
        except OverflowError:
            pass
    if pos:
        if all(x != 0 for x in xs):
            out["ed"] = [1.0 / x for x in xs]
    else:
        try:
            out["ed"] = [x ** 4 for x in xs]
        except OverflowError:
            pass
    # Guard: a transform that produces non-positive values breaks the
    # "no negative performance" cutoff and every downstream inverse.
    return {k: v for k, v in out.items()
            if v and all(math.isfinite(t) and t > 0 for t in v)}


def _forward(method, gamma, x):
    """Apply the chosen adjustment to an arbitrary x (used to evaluate the
    adjusted curve on the ORIGINAL x-axis, per the paper's scaling step)."""
    pos = gamma >= 0
    if method == "base":
        return x
    if x <= 0:
        return None
    if method == "sq":
        return math.sqrt(x) if pos else x ** 2
    if method == "lg":
        return math.log10(x) if pos else x ** 3
    if method == "ed":
        return 1.0 / x if pos else x ** 4
    return x


def select_adjustment(xs):
    """eq (9): the candidate whose skew is closest to zero wins."""
    g = skew(xs)
    cands = _candidates(xs, g)
    scored = sorted(((abs(0 - skew(v)), k) for k, v in cands.items()))
    best = scored[0][1]
    return best, g, cands[best], {k: skew(v) for k, v in cands.items()}


def adjusted_sigma(mu_base, adj_vals, method, mode="auto"):
    """eq (10) sigma_adj ~ sqrt(|mu_base^2 - mu_adj^2|), with a sanity net.

    eq (10) was calibrated on FRC point totals. On other scales it can return a
    value orders of magnitude away from the actual spread of the adjusted array,
    which flattens the skew curve into uselessness. 'auto' keeps eq (10) when it
    is within 5x of the direct stdev and otherwise falls back.
    """
    direct = pstdev(adj_vals)
    mu_adj = mean(adj_vals)
    paper = math.sqrt(abs(mu_base ** 2 - mu_adj ** 2))
    if method == "base":
        return direct, "direct(base-method)", {"paper": paper, "direct": direct}
    if mode == "paper":
        return (paper or direct), "paper-eq10", {"paper": paper, "direct": direct}
    if mode == "direct":
        return direct, "direct", {"paper": paper, "direct": direct}
    ok = direct > 0 and 0.2 * direct <= paper <= 5.0 * direct
    return (paper, "paper-eq10", {"paper": paper, "direct": direct}) if ok else \
           (direct, "direct(eq10-out-of-range)", {"paper": paper, "direct": direct})

# ----------------------------------------------------------------- the curves

def build(xs, cfg):
    """Full per-team build: pre-filter -> stats -> adjustment -> scaled curves
    -> intersections -> damped distribution."""
    dcfg = cfg["distribution"]
    kept, n_lo, n_hi = trim(xs, cfg["prefilter"]["trim_low_pct"],
                                cfg["prefilter"]["trim_high_pct"])
    if len(kept) < 4:                     # paper: works with as few as 4 points
        return None

    mu, sigma, g_raw = mean(kept), pstdev(kept), skew(kept)
    if sigma <= 0:
        return None

    method, g_used, adj, all_skews = select_adjustment(kept)
    mu_adj = mean(adj)
    sigma_adj, sigma_src, sigma_dbg = adjusted_sigma(
        mu, adj, method, dcfg.get("sigma_adj_mode", "auto"))
    if sigma_adj <= 0:
        sigma_adj = pstdev(adj) or 1e-9

    x_max = norm_inv(float(dcfg["max_cutoff_pct"]), mu, sigma)   # upper cutoff
    x_max = max(x_max, mu + 0.5 * sigma)
    n = int(dcfg["grid_points"])
    step = x_max / (n - 1)
    xs_grid = [i * step for i in range(n)]                      # min cutoff = 0

    y_norm = [normal_pdf(x, mu, sigma) for x in xs_grid]
    y_skew = []
    for x in xs_grid:
        t = _forward(method, g_used, x)
        y_skew.append(0.0 if t is None else normal_pdf(t, mu_adj, sigma_adj))

    mn, ms = max(y_norm) or 1.0, max(y_skew) or 1.0
    y_norm = [v / mn for v in y_norm]                 # scaled normal
    y_skew = [v / ms for v in y_skew]                 # scaled skew-adjusted

    i_pn = max(range(n), key=lambda i: y_norm[i])
    i_ps = max(range(n), key=lambda i: y_skew[i])
    lo_i, hi_i = min(i_pn, i_ps), max(i_pn, i_ps)

    cross = []
    for i in range(n - 1):
        d0 = y_norm[i] - y_skew[i]
        d1 = y_norm[i + 1] - y_skew[i + 1]
        if d0 == 0.0:
            cross.append((xs_grid[i], i))
        elif d0 * d1 < 0:
            f = abs(d0) / (abs(d0) + abs(d1))
            cross.append((xs_grid[i] + f * step, i))

    mid = [c for c in cross if lo_i <= c[1] <= hi_i]
    mode = mid[0][0] if mid else xs_grid[(i_pn + i_ps) // 2]
    left = [c[0] for c in cross if c[1] < lo_i]
    right = [c[0] for c in cross if c[1] > hi_i]
    band_lo = max(left) if left else max(0.0, mode - sigma)     # fallback: eq(2)
    band_hi = min(right) if right else mode + sigma
    band_src = ("intersection" if left else "normal-sigma") + "/" + \
               ("intersection" if right else "normal-sigma")

    if dcfg.get("damping", "min") == "mean":
        y_damp = [(a + b) / 2 for a, b in zip(y_norm, y_skew)]
    else:
        y_damp = [min(a, b) for a, b in zip(y_norm, y_skew)]

    area = sum(y_damp) * step
    if area <= 0:
        y_damp, area = y_norm, sum(y_norm) * step
    dens = [v / area for v in y_damp]                  # eq (5) normalisation

    return {
        "n_raw": len(xs), "n_used": len(kept), "trimmed_low": n_lo, "trimmed_high": n_hi,
        "mu": mu, "sigma": sigma,
        "skew_before_trim": skew(xs), "skew_after_trim": g_raw,
        "method": method, "skew_of_adjusted": skew(adj), "candidate_skews": all_skews,
        "mu_adj": mu_adj, "sigma_adj": sigma_adj, "sigma_adj_source": sigma_src,
        "sigma_adj_debug": sigma_dbg,
        "mode": mode, "band_lo": band_lo, "band_hi": band_hi, "band_source": band_src,
        "x_max_cutoff": x_max, "step": step,
        "grid": xs_grid, "y_normal": y_norm, "y_skew": y_skew, "density": dens,
    }

# -------------------------------------------------------------- head to head

def _resample(grid, ys, step, xs_new):
    out = []
    for x in xs_new:
        if x < grid[0] or x > grid[-1]:
            out.append(0.0); continue
        j = int(x / step)
        j = min(max(j, 0), len(grid) - 2)
        f = (x - grid[j]) / step
        out.append(ys[j] * (1 - f) + ys[j + 1] * f)
    return out


def win_probability(A, B, n=1500):
    """P(strength_A > strength_B) by integrating the two damped densities.
    This is eq (5) generalised to a pairwise comparison."""
    top = max(A["x_max_cutoff"], B["x_max_cutoff"])
    step = top / (n - 1)
    xs = [i * step for i in range(n)]
    a = _resample(A["grid"], A["density"], A["step"], xs)
    b = _resample(B["grid"], B["density"], B["step"], xs)
    sa, sb = sum(a) or 1.0, sum(b) or 1.0
    a = [v / sa for v in a]
    b = [v / sb for v in b]
    cum = 0.0
    p = 0.0
    for i in range(n):
        p += a[i] * (cum + 0.5 * b[i])
        cum += b[i]
    return min(max(p, 1e-6), 1 - 1e-6)


def apply_home_field(p_home, logit_shift):
    if not logit_shift:
        return p_home
    l = math.log(p_home / (1 - p_home)) + logit_shift
    return 1.0 / (1.0 + math.exp(-l))
