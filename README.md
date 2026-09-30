# MLB Skewed Distribution Forecast

Port of the **Skewed Distribution Algorithm for Practiced Task Performance Forecasting**
(Ian Padovani, FRC Team 4206, v2.1.2025) from FRC to Major League Baseball.

* **Performance aspect tracked:** team strength = *per-game wOBA / per-game FIP*, indexed so the
  league season mean = 100.
* **Population per team:** one strength value per game played, bottom 10% and top 10% trimmed.
* **Output:** a damped skewed distribution per team (mode + variance band) and a win probability
  for any matchup.
* **Hosting:** 100% static. GitHub Actions recomputes `/data/*.json` on a daily cron and commits it;
  GitHub Pages serves the repo as-is. No server, no database, no API key.

---

## Setup (about five minutes)

1. Push these files to the root of a repo, on the **default branch** (`main`). Scheduled workflows
   only run from the default branch.
2. Set `"season"` in `config.json` to the season you want, and commit it.
3. **Settings -> Actions -> General -> Workflow permissions -> "Read and write permissions"** -> Save.
   Without this the refresh job fetches and computes fine but fails on `git push` with a 403.
4. **Settings -> Pages -> Build and deployment -> Source: "Deploy from a branch"**, branch `main`,
   folder `/ (root)` -> Save.
5. **Actions** tab -> `refresh` in the left sidebar -> **Run workflow** -> **Run workflow**.
   Takes about a minute. After this it runs itself daily at 12:20 UTC.
6. Open `https://<user>.github.io/<repo>/` and confirm the yellow demo-data banner is **gone**.

The repo ships with synthetic demo data so the page renders before your first real refresh. The
site shows a loud banner whenever it is displaying synthetic numbers, and the workflow's
"Verify" step fails the run rather than publishing demo data by accident.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No `refresh` in the Actions sidebar | `.github/workflows/refresh.yml` is not on the default branch, or Actions is disabled (Settings -> Actions -> General -> "Allow all actions") |
| No **Run workflow** button | Same - `workflow_dispatch` is only offered for workflows present on the default branch |
| Commit step fails, `remote: Permission to ... denied` / 403 | Step 3 not done |
| Run is green but the site still shows old numbers | The Pages build has not run yet. Check for a `pages-build-deployment` run in the Actions tab after the data commit; hard-refresh the page |
| Yellow demo banner still showing | The run did not reach the commit step, or Pages has not rebuilt |
| Cron never fires | GitHub disables scheduled workflows after 60 days of repo inactivity - it emails you, and the Actions tab shows an "Enable workflow" button |
| Cron fires late | Normal; GitHub queues scheduled jobs and can delay them under load |

## Local use

```bash
python3 scripts/make_synthetic.py      # offline demo data (no network needed)
# or
python3 scripts/fetch_data.py          # real data from statsapi.mlb.com

python3 scripts/compute.py             # strength series + distributions
python3 scripts/evaluate.py            # walk-forward backtest + calibration fit
python3 scripts/predict.py             # price upcoming games
python3 -m http.server 8000            # open http://localhost:8000
```

No dependencies. Standard library only, Python 3.9+.

## Layout

```
index.html                 single-page UI (vanilla JS + canvas, no frameworks)
assets/app.js, style.css
config.json                every tunable knob, with comments
scripts/mlb_api.py         MLB Stats API client (~62 requests per refresh)
scripts/metrics.py         per-game wOBA, per-game FIP, strength
scripts/skewdist.py        the algorithm: eq 1-10, scaled curves, intersections, eq 5 probability
scripts/compute.py         raw -> strength series -> per-team distributions
scripts/predict.py         upcoming games + append-only prediction log
scripts/evaluate.py        walk-forward backtest, baselines, calibration fit
scripts/make_synthetic.py  offline demo/regression data
data/raw/gamelogs_*.json   cached game logs (the only thing that grows)
data/*.json                everything the browser reads
.github/workflows/refresh.yml
```

`scripts/fetch_data.py` and everything downstream are decoupled by
`data/raw/gamelogs_<season>.json`, so you can swap in a different data source (Statcast,
Retrosheet, a different sport entirely) by writing that one file.

## Knobs worth turning

| Knob | Why |
|---|---|
| `shrinkage.prior_innings` / `prior_pa` | How hard single-game FIP/wOBA are pulled toward league average. This is the MLB analogue of the paper's experimentally-optimized pre-filter. Start at 3 IP / 10 PA. |
| `prefilter.trim_*_pct` | Currently 10%/10%. Compare against 1-value trimming (the FRC setting) using `evaluate.py`. |
| `window` | `null` = all 162 games. Try 40-60: it keeps the population under the paper's ~100-point crossover and tracks current form. |
| `distribution.sigma_adj_mode` | `paper` = eq (10) verbatim, `direct` = stdev of the adjusted array, `auto` = eq (10) with a sanity net. See METHOD.md. |
| `prediction.home_field_logit` | 0 = pure algorithm. 0.16 approximates MLB home-field advantage. |
| `prediction.calibration` | `auto` applies the fitted logistic recalibration; `off` shows the raw algorithm. |

Change a knob, run `compute.py` then `evaluate.py`, and read the Brier score. That loop is the
whole viability experiment - `python3 scripts/sweep.py` runs six window/trim variants for you and
writes `data/sweep.json`.

`python3 scripts/selftest.py` checks the algorithm's invariants (eq 1-3 against hand-computed
values, eq 9 never selecting a worse skew, damped peak == reported mode, density integrating to 1,
antisymmetric win probabilities). Run it after any edit to `skewdist.py`.

## Reading the results

Accuracy alone is a trap here. One MLB game is close to a coin flip: the best public models and
the betting markets land around 57-60%. FRC-style 90-96% is not on the table, and chasing it would
violate the paper's own third principle. Judge the port on:

1. **Brier score vs the log5-from-records baseline** in section 5 of the page. Beating it means the
   wOBA/FIP distributions carry information that win-loss record does not.
2. **Calibration** - the dots should sit on the diagonal.
3. **The prospective log** - `data/prediction_log.json` is append-only and written before games are
   played, so it cannot be gamed.
