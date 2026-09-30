# Porting the algorithm from FRC to MLB

Clause-by-clause mapping of the white paper (v2.1.2025) onto baseball, followed by the
deviations I had to make, the open questions only you can settle, and the risks that will
decide whether this is viable.

---

## 1. Mapping

| Paper | FRC | MLB implementation |
|---|---|---|
| Player | One robot/team in one event | One franchise in one season |
| Performance aspect (objective metric correlated with the outcome) | Points scored per match | Team strength = per-game wOBA / per-game FIP |
| Data population per player | Matches at one event (~10) | Games played this season (up to 162) |
| Closed-system scope for relativism | One competition, one division | One league-season. Strength is indexed so the season mean = 100 |
| Pre-filter | Drop lowest (and sometimes highest) value | Drop bottom 10% and top 10% per team |
| eq 1-3 | mean, population sigma, skew | identical, `scripts/skewdist.py` |
| eq 6-9 | sqrt / log10 / reciprocal, branched on skew sign, pick whichever skew is closest to 0 | identical |
| eq 10 | sigma_adj approximation | implemented, with a fallback (see 3.1) |
| Scaled normal + scaled skew curves, three intersections | mode + variance bounds | identical; damped curve = pointwise min of the two scaled curves |
| Max cutoff at the 90-98th percentile | maximum reasonable performance | 95th percentile of the normal, configurable |
| Min cutoff at 0 | negative performance impossible | strength is strictly positive by construction |
| eq 5 probability as a ratio of integrals | score-range probability | pairwise `P(strength_A > strength_B)` for a win probability |
| Monte Carlo avoided | read the mode off the curve | identical, no simulation loop anywhere |

Two things about baseball make it a better fit than FRC in one respect and worse in another.

**Better:** the paper's most-quoted limitation is that it relies on individual performance and
degrades as multi-player strategy increases, which is why FRC accuracy falls to the mid-to-high
80s in high-strategy games. Baseball is a sequence of nearly independent one-on-one
pitcher-batter events with almost no emergent multi-player strategy. That whole failure mode is
largely absent.

**Worse:** an FRC match is a 2.5-minute execution test where the better robot almost always wins.
An MLB game is one sample from a very noisy run-scoring process where the better team wins about
55-60% of the time. The algorithm's job here is not to be right 90% of the time; it is to
produce a **calibrated** probability. That is exactly what the paper's first principle demands.

---

## 2. Choices you should know about

### 2.1 Single-game FIP is unstable, so it is stabilized
FIP = (13HR + 3(BB+HBP) - 2K)/IP + cFIP. Over nine innings that numerator is routinely
**negative** (0 HR, 0 BB, 11 K gives -22/9 = -2.44). Three things then break at once:

* strength = wOBA/FIP explodes as FIP approaches 0 and flips sign past it;
* the paper's sqrt, log10 and reciprocal adjustments all require strictly positive input;
* the "no negative performance" minimum cutoff at zero becomes meaningless.

Fix: every single-game line is credited with `prior_innings` of league-average pitching and
`prior_pa` of league-average hitting before the ratio is taken (defaults 3 IP / 10 PA), plus a
hard FIP floor of 1.00. This is a Bayesian shrinkage, it is monotone, and it preserves
between-team differences while killing the divide-by-almost-zero tail. **Treat `prior_innings`
as the tunable that matters most** - it is the real MLB analogue of the paper's
"experimentally optimized pre-filter."

### 2.2 Strength is indexed to 100
Raw wOBA/FIP is about 0.08. On that scale `log10` returns negative numbers and `1/x` returns ~12,
so eq 6-8 misbehave. Indexing to league mean = 100 puts every value in a domain where all four
candidate adjustments are well-defined, and it satisfies principle 2 (forecasted values must be
relativistic) with the league-season as the closed system.

### 2.3 cFIP and the shrinkage priors are derived, not hardcoded
`cFIP = lgERA - lgFIP_raw` is recomputed from league season totals on every refresh, so the
metric stays correct as run environment drifts. The wOBA linear weights **are** hardcoded in
`config.json` and must be refreshed each offseason from the FanGraphs "Guts!" table.

### 2.4 The damped curve is the pointwise minimum
The paper defines the damped result by three intersections of the scaled normal and scaled skew
curves - mode between the peaks, variance bounds outside them - but never states a full damped
density, which a probability requires. The pointwise minimum of the two scaled curves reproduces
all three landmarks exactly (its peak *is* the middle intersection) and needs no new parameters,
so that is what gets normalized into a density for eq 5. `distribution.damping: "mean"` is the
alternative if you prefer averaging. **Worth confirming against your spreadsheet.**

### 2.5 A recalibration layer sits on top
Raw `P(strength_A > strength_B)` is systematically over-confident: on the demo data, games priced
at 0.93 won about 0.80 of the time. That is expected - the strength distributions describe
performance spread, not the extra run-sequencing luck that decides a single game. `evaluate.py`
fits `p' = sigmoid(a*logit(p) + b)` on completed games (fit on the first 60%, graded on the last
40%) and writes `data/calibration.json`; the fitted slope came out near 0.63, i.e. shrink the log
odds by about a third. Set `prediction.calibration: "off"` to see the unshrunk algorithm.

---

## 3. Open questions for you

1. **eq (10) does not survive the change of scale.** `sigma_adj = sqrt(|mu_base^2 - mu_adj^2|)`
   returns roughly `mu_base` whenever the adjustment compresses the scale. On a strength index of
   ~100, a sqrt adjustment gives mu_adj ~10 and eq (10) returns ~99.5, while the actual spread of
   the adjusted array is ~0.7 - a 140x difference that flattens the skew curve into a straight
   line. Also note eq (10) returns exactly 0 when the unadjusted `base` set wins eq (9). The code
   defaults to `sigma_adj_mode: "auto"` (use eq 10 only when it is within 5x of the direct stdev,
   otherwise use the direct stdev) and records which branch fired per team in the UI. **Please
   run one known-good FRC team through `skewdist.build` and tell me which branch reproduces your
   spreadsheet** - if eq (10) is meant to be read on the adjusted scale, that is a one-line fix.
2. **Trimming 20% of the data removes much of the skew you are trying to measure.** On the demo
   data, team skew drops from 0.83 to 0.54 and from 1.34 to 0.44 after the 10%/10% trim. Since
   skew selection (eq 9) is the entire engine, a trim that flattens skew pushes the algorithm
   toward the `base`/normal case it was built to escape. Both values are displayed per team so you
   can see it. Recommend backtesting 10/10 against 5/5 and against the FRC single-value trim.
3. **162 games is past your own crossover point.** The paper puts the crossover where ordinary
   curve-fitting beats this approximation at about 100 data points. A full season trimmed to 130
   values is beyond it, which means a full-season run is arguably testing the algorithm outside
   its stated design envelope. `window: 40` keeps it inside the envelope *and* tracks current
   form. I would run the season three ways - full, 60, 40 - and compare Brier scores. That single
   experiment is also the cleanest published test of your crossover claim anyone has.
4. **Rotation slots make per-game FIP multimodal, not merely skewed.** A team's game-level FIP is
   dominated by which of five starters pitched. The distribution has up to five bumps, and
   selecting a unimodal skew adjustment blurs them. Two clean fixes: split strength into
   offense-side and pitching-side distributions, or condition the pitching side on the announced
   starter. Either one is a bigger change than I would make without your sign-off, but expect it
   to be the single largest accuracy lever.
5. **Strength as a ratio throws away the matchup structure.** wOBA/FIP compares a team to itself.
   The actual game pits *my* offense against *their* pitching. A v2 that keeps the two
   distributions separate and crosses them (my wOBA curve vs their FIP curve, and vice versa)
   respects the paper's rule that interconnected statistics must maintain their connection, and
   would let you predict a run range rather than only a winner. The current 1-D ratio is
   implemented exactly as you specified and works, but it is leaving information on the table.
6. **Park and opponent effects are unmodeled external randomness.** Coors Field inflates every
   hitting line; a soft schedule inflates strength. In FRC terms this is the same problem the
   paper solves by scoping to one division. Park factors are the obvious next adjustment.
