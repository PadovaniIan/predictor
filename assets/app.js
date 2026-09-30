/* MLB Skewed Distribution Forecast - static front end.
   Everything heavy is precomputed into /data by GitHub Actions; this file only
   draws it and does the (cheap) pairwise integral for arbitrary matchups. */
const J = p => fetch(p + '?v=' + Date.now()).then(r => r.ok ? r.json() : null).catch(() => null);
const S = {};

const fmt = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? '-' : Number(v).toFixed(d);
const pct = v => (100 * v).toFixed(1) + '%';

/* ---------------------------------------------------------------- charting */
function prep(cv) {
  const r = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.height;
  cv.width = w * r; cv.height = h * r;
  const c = cv.getContext('2d'); c.setTransform(r, 0, 0, r, 0, 0); c.clearRect(0, 0, w, h);
  return { c, w, h };
}
function axes(c, w, h, xr, yr, pad, xlab) {
  c.strokeStyle = '#283040'; c.fillStyle = '#93a1b5'; c.lineWidth = 1;
  c.font = '11px system-ui';
  c.beginPath(); c.moveTo(pad.l, pad.t); c.lineTo(pad.l, h - pad.b); c.lineTo(w - pad.r, h - pad.b); c.stroke();
  for (let i = 0; i <= 5; i++) {
    const x = pad.l + (w - pad.l - pad.r) * i / 5, v = xr[0] + (xr[1] - xr[0]) * i / 5;
    c.beginPath(); c.moveTo(x, h - pad.b); c.lineTo(x, h - pad.b + 4); c.stroke();
    c.textAlign = 'center'; c.fillText(fmt(v, 0), x, h - pad.b + 16);
  }
  for (let i = 0; i <= 4; i++) {
    const y = h - pad.b - (h - pad.t - pad.b) * i / 4, v = yr[0] + (yr[1] - yr[0]) * i / 4;
    c.strokeStyle = '#1f2632'; c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
    c.textAlign = 'right'; c.fillStyle = '#93a1b5'; c.fillText(fmt(v, 2), pad.l - 6, y + 3);
  }
  if (xlab) { c.textAlign = 'center'; c.fillText(xlab, pad.l + (w - pad.l - pad.r) / 2, h - 3); }
}
function line(c, w, h, pad, xs, ys, xr, yr, col, width = 2, fill = false) {
  const X = v => pad.l + (w - pad.l - pad.r) * (v - xr[0]) / (xr[1] - xr[0]);
  const Y = v => h - pad.b - (h - pad.t - pad.b) * (v - yr[0]) / (yr[1] - yr[0]);
  c.beginPath(); xs.forEach((x, i) => i ? c.lineTo(X(x), Y(ys[i])) : c.moveTo(X(x), Y(ys[i])));
  c.strokeStyle = col; c.lineWidth = width; c.stroke();
  if (fill) { c.lineTo(X(xs[xs.length - 1]), Y(yr[0])); c.lineTo(X(xs[0]), Y(yr[0])); c.closePath();
    c.globalAlpha = .13; c.fillStyle = col; c.fill(); c.globalAlpha = 1; }
}
function vline(c, w, h, pad, x, xr, col, label) {
  const X = pad.l + (w - pad.l - pad.r) * (x - xr[0]) / (xr[1] - xr[0]);
  c.strokeStyle = col; c.setLineDash([4, 3]); c.lineWidth = 1;
  c.beginPath(); c.moveTo(X, pad.t); c.lineTo(X, h - pad.b); c.stroke(); c.setLineDash([]);
  if (label) { c.fillStyle = col; c.font = '10px system-ui'; c.textAlign = 'center'; c.fillText(label, X, pad.t - 2); }
}

/* -------------------------------------------------------- matchup integral */
function resample(d, xs) {
  const g = d.x, step = d.step, out = [];
  for (const x of xs) {
    if (x < g[0] || x > g[g.length - 1]) { out.push(0); continue; }
    let j = Math.min(Math.max(Math.floor((x - g[0]) / step), 0), g.length - 2);
    const f = (x - g[j]) / step;
    out.push(d.density[j] * (1 - f) + d.density[j + 1] * f);
  }
  return out;
}
function winProb(A, B, n = 900) {
  const top = Math.max(A.x[A.x.length - 1], B.x[B.x.length - 1]), step = top / (n - 1);
  const xs = Array.from({ length: n }, (_, i) => i * step);
  let a = resample(A, xs), b = resample(B, xs);
  const sa = a.reduce((p, v) => p + v, 0) || 1, sb = b.reduce((p, v) => p + v, 0) || 1;
  a = a.map(v => v / sa); b = b.map(v => v / sb);
  let cum = 0, p = 0;
  for (let i = 0; i < n; i++) { p += a[i] * (cum + .5 * b[i]); cum += b[i]; }
  return Math.min(Math.max(p, 1e-6), 1 - 1e-6);
}
function calibrate(p) {
  const c = S.cal;
  if (!c || (S.meta && S.meta.config.prediction.calibration === 'off')) return p;
  p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  const z = Math.log(p / (1 - p));
  return 1 / (1 + Math.exp(-(c.a * z + c.b)));
}

/* ------------------------------------------------------------------- views */
function kpis() {
  const m = S.meta, e = S.evaluation || {}, wf = e.walk_forward, base = (e.baselines || {}).log5_from_records;
  const ok = S.teams.filter(t => t.ok);
  const card = (l, v, x) => `<div class="card"><div class="l">${l}</div><div class="v">${v}</div><div class="x">${x || ''}</div></div>`;
  document.getElementById('kpis').innerHTML =
    card('Season', m.season, `${m.games_final} of ${m.games_scheduled} games final`) +
    card('Teams modelled', ok.length + '/' + S.teams.length, `min ${Math.min(...ok.map(t => t.n_used))} values used`) +
    card('League cFIP', fmt(m.context.fip_constant, 3), 'derived from league totals') +
    card('League wOBA', fmt(m.context.league_woba, 3), 'shrinkage prior') +
    card('Walk-forward accuracy', wf ? pct(wf.accuracy) : '-', wf ? `n=${wf.n} games` : 'needs more games') +
    card('Brier score', wf ? fmt(wf.brier, 4) : '-', base ? `log5 baseline ${fmt(base.brier, 4)}` : '');
  document.getElementById('stamp').textContent =
    `data fetched ${m.data_fetched_utc || '?'} | computed ${m.generated_utc} | window ${m.config.window || 'full season'} | ` +
    `trim ${Math.round(m.config.prefilter.trim_low_pct * 100)}%/${Math.round(m.config.prefilter.trim_high_pct * 100)}%`;
  if (m.synthetic) document.getElementById('synthbanner').classList.remove('hidden');
}

function table() {
  const tb = document.querySelector('#ranktable tbody');
  let dir = 1, key = 'rank';
  const draw = () => {
    const rows = S.teams.filter(t => t.ok).slice().sort((a, b) =>
      (typeof a[key] === 'string') ? dir * String(a[key]).localeCompare(String(b[key])) : dir * ((a[key] || 0) - (b[key] || 0)));
    tb.innerHTML = rows.map(t => `<tr data-id="${t.id}" class="${t.id === S.sel ? 'sel' : ''}">
      <td>${t.rank}</td><td>${t.abbrev} <span style="color:#93a1b5">${t.name || ''}</span></td>
      <td><b>${fmt(t.mode)}</b></td><td>${fmt(t.band_lo)} - ${fmt(t.band_hi)}</td>
      <td>${fmt(t.mu)}</td><td>${fmt(t.sigma)}</td>
      <td>${fmt(t.skew_before_trim, 3)}</td><td>${fmt(t.skew_after_trim, 3)}</td>
      <td>${t.method}</td><td>${fmt(t.skew_of_adjusted, 3)}</td>
      <td>${t.n_used}</td><td>${t.games}</td></tr>`).join('');
    tb.querySelectorAll('tr').forEach(tr => tr.onclick = () => {
      document.getElementById('teamsel').value = tr.dataset.id; showTeam(+tr.dataset.id);
    });
  };
  document.querySelectorAll('#ranktable th').forEach(th => th.onclick = () => {
    const k = th.dataset.k; dir = (k === key) ? -dir : 1; key = k; draw();
  });
  draw();
}

function showTeam(id) {
  S.sel = id;
  const t = S.teams.find(x => x.id === id), d = S.dists[String(id)];
  document.querySelectorAll('#ranktable tbody tr').forEach(tr =>
    tr.classList.toggle('sel', +tr.dataset.id === id));
  if (!t || !d) return;
  const { c, w, h } = prep(document.getElementById('curve'));
  const pad = { l: 44, r: 14, t: 16, b: 30 }, xr = [d.x[0], d.x[d.x.length - 1]], yr = [0, 1.05];
  axes(c, w, h, xr, yr, pad, 'team strength index (100 = league average)');
  line(c, w, h, pad, d.x, d.damped, xr, yr, '#f2b53c', 2, true);
  line(c, w, h, pad, d.x, d.normal, xr, yr, '#4f8ef7', 2);
  line(c, w, h, pad, d.x, d.skew, xr, yr, '#e8534a', 2);
  vline(c, w, h, pad, t.mode, xr, '#f2b53c', 'mode ' + fmt(t.mode, 1));
  vline(c, w, h, pad, t.band_lo, xr, '#6b7a90', fmt(t.band_lo, 1));
  vline(c, w, h, pad, t.band_hi, xr, '#6b7a90', fmt(t.band_hi, 1));
  const kv = (k, v) => `<div>${k}<br><b>${v}</b></div>`;
  document.getElementById('curveinfo').innerHTML =
    kv('mode (damped)', fmt(t.mode)) + kv('variance band', `${fmt(t.band_lo)} - ${fmt(t.band_hi)}`) +
    kv('band source', t.band_source) + kv('adjustment chosen', t.method) +
    kv('skew raw &rarr; trimmed &rarr; adjusted', `${fmt(t.skew_before_trim, 3)} &rarr; ${fmt(t.skew_after_trim, 3)} &rarr; ${fmt(t.skew_of_adjusted, 3)}`) +
    kv('candidate skews', Object.entries(t.candidate_skews || {}).map(([k, v]) => `${k}:${fmt(v, 2)}`).join('  ')) +
    kv('&mu; / &sigma; (trimmed)', `${fmt(t.mu)} / ${fmt(t.sigma)}`) +
    kv('&sigma;<sub>adj</sub>', `${fmt(t.sigma_adj, 4)} <span class="pill">${t.sigma_adj_source}</span>`) +
    kv('values used', `${t.n_used} of ${t.games} (dropped ${t.trimmed[0]} low / ${t.trimmed[1]} high)`) +
    kv('upper cutoff', `${fmt(t.cutoff)} (${Math.round(S.meta.config.distribution.max_cutoff_pct * 100)}th pct)`);
  sparkline(id);
}

function sparkline(id) {
  const rows = (S.series || {})[String(id)] || [];
  const { c, w, h } = prep(document.getElementById('spark'));
  if (!rows.length) return;
  const pad = { l: 44, r: 14, t: 12, b: 26 };
  const ys = rows.map(r => r.strength), xs = rows.map((_, i) => i + 1);
  const xr = [1, xs.length], yr = [Math.min(...ys) * .96, Math.max(...ys) * 1.04];
  axes(c, w, h, xr, yr, pad, 'game number');
  const t = S.teams.find(x => x.id === id);
  if (t) { // trim thresholds
    const s = ys.slice().sort((a, b) => a - b);
    [s[t.trimmed[0]], s[s.length - 1 - t.trimmed[1]]].forEach(v => {
      const Y = h - pad.b - (h - pad.t - pad.b) * (v - yr[0]) / (yr[1] - yr[0]);
      c.strokeStyle = '#e8534a'; c.setLineDash([3, 3]); c.beginPath();
      c.moveTo(pad.l, Y); c.lineTo(w - pad.r, Y); c.stroke(); c.setLineDash([]);
    });
  }
  line(c, w, h, pad, xs, ys, xr, yr, '#39c07c', 1.4);
}

function matchup() {
  const a = +document.getElementById('awaysel').value, hm = +document.getElementById('homesel').value;
  const A = S.dists[String(a)], B = S.dists[String(hm)];
  const ta = S.teams.find(t => t.id === a), tb = S.teams.find(t => t.id === hm);
  if (!A || !B || a === hm) {
    document.getElementById('mresult').innerHTML = '<span class="note">Pick two different modelled teams.</span>';
    return;
  }
  const hfa = S.meta.config.prediction.home_field_logit || 0;
  let raw = winProb(B, A);                                   // P(home strength > away strength)
  if (hfa) { const z = Math.log(raw / (1 - raw)) + hfa; raw = 1 / (1 + Math.exp(-z)); }
  const p = calibrate(raw);
  document.getElementById('mresult').innerHTML =
    `<div class="bar"><span style="width:${(100 * (1 - p)).toFixed(1)}%;background:#e8534a">${ta.abbrev} ${pct(1 - p)}</span>
     <span style="width:${(100 * p).toFixed(1)}%;background:#4f8ef7">${tb.abbrev} ${pct(p)}</span></div>
     <div class="note">raw P(home) ${pct(raw)} ${S.cal ? '&rarr; recalibrated ' + pct(p) : '(no calibration fit yet)'} &middot;
     ${ta.abbrev} mode ${fmt(ta.mode, 1)} [${fmt(ta.band_lo, 1)}-${fmt(ta.band_hi, 1)}] vs
     ${tb.abbrev} mode ${fmt(tb.mode, 1)} [${fmt(tb.band_lo, 1)}-${fmt(tb.band_hi, 1)}]</div>`;
  const { c, w, h } = prep(document.getElementById('mcurve'));
  const pad = { l: 44, r: 14, t: 16, b: 30 };
  const xr = [0, Math.max(A.x[A.x.length - 1], B.x[B.x.length - 1])];
  const mx = Math.max(...A.damped, ...B.damped) * 1.06;
  axes(c, w, h, xr, yr_(mx), pad, 'team strength index');
  line(c, w, h, pad, A.x, A.damped, xr, yr_(mx), '#e8534a', 2, true);
  line(c, w, h, pad, B.x, B.damped, xr, yr_(mx), '#4f8ef7', 2, true);
  vline(c, w, h, pad, ta.mode, xr, '#e8534a', ta.abbrev);
  vline(c, w, h, pad, tb.mode, xr, '#4f8ef7', tb.abbrev);
  function yr_(m) { return [0, m]; }
}

function upcoming() {
  const el = document.getElementById('upcoming');
  const ps = (S.predictions || {}).predictions || [];
  if (!ps.length) { el.innerHTML = '<p class="note">No upcoming games priced. Either the season is over or fewer than ' +
      S.meta.config.prediction.min_games + ' games have been played by both teams.</p>'; return; }
  el.innerHTML = ps.slice(0, 36).map(p => {
    const fav = p.p_home >= .5;
    return `<div class="game"><div class="d">${p.date}</div>
      <div class="m">${p.away} @ ${p.home}</div>
      <div>${fav ? p.home : p.away} <b>${pct(fav ? p.p_home : p.p_away)}</b>
      <span class="pill">${p.calibrated ? 'calibrated' : 'raw'}</span></div>
      <div class="d">modes ${fmt(p.away_mode, 1)} / ${fmt(p.home_mode, 1)}</div></div>`;
  }).join('');
}

function evalview() {
  const e = S.evaluation, box = document.getElementById('evalbox');
  if (!e || !e.walk_forward) { box.innerHTML = '<p class="note">No evaluation yet - needs completed games.</p>'; return; }
  const rows = [['Skewed distribution (walk-forward)', e.walk_forward],
                ['Same, recalibrated (held-out 40%)', e.walk_forward_recalibrated_holdout],
                ['Raw on that same held-out 40%', e.walk_forward_raw_same_holdout],
                ['Baseline: coin flip / home always', e.baselines.always_home_50_50],
                ['Baseline: log5 from W-L records', e.baselines.log5_from_records],
                ['Prospective log (predicted before the game)', e.prospective_log]];
  box.innerHTML = '<div class="tablewrap"><table><thead><tr><th style="text-align:left">Model</th><th>n</th>' +
    '<th>Accuracy</th><th>Brier</th><th>Log loss</th></tr></thead><tbody>' +
    rows.filter(r => r[1]).map(([k, v]) => `<tr><td style="text-align:left">${k}</td><td>${v.n}</td>
      <td>${pct(v.accuracy)}</td><td>${fmt(v.brier, 4)}</td><td>${fmt(v.log_loss, 4)}</td></tr>`).join('') +
    '</tbody></table></div>' + (e.notes || []).map(n => `<p class="note">&bull; ${n}</p>`).join('');
  const cal = e.walk_forward.calibration || [];
  const { c, w, h } = prep(document.getElementById('calplot'));
  const pad = { l: 44, r: 14, t: 16, b: 30 }, xr = [0, 1], yr = [0, 1];
  axes(c, w, h, xr, yr, pad, 'predicted P(home win)  vs  observed');
  line(c, w, h, pad, [0, 1], [0, 1], xr, yr, '#6b7a90', 1);
  if (cal.length) line(c, w, h, pad, cal.map(b => b.pred), cal.map(b => b.actual), xr, yr, '#f2b53c', 2);
  cal.forEach(b => {
    const X = pad.l + (w - pad.l - pad.r) * b.pred, Y = h - pad.b - (h - pad.t - pad.b) * b.actual;
    c.fillStyle = '#f2b53c'; c.beginPath(); c.arc(X, Y, 3 + Math.min(6, b.n / 40), 0, 7); c.fill();
  });
}

/* -------------------------------------------------------------------- boot */
(async function () {
  const [meta, teams, dists, preds, ev, series, cal] = await Promise.all([
    J('data/meta.json'), J('data/teams.json'), J('data/distributions.json'),
    J('data/predictions.json'), J('data/evaluation.json'),
    J('data/strength_series.json'), J('data/calibration.json')]);
  if (!meta || !teams) {
    document.querySelector('main').innerHTML =
      '<p class="note" style="margin-top:30px">No data in <code>/data</code> yet. Run <code>python3 scripts/make_synthetic.py</code> ' +
      'for a demo, or the <code>refresh</code> workflow for real MLB data.</p>';
    return;
  }
  Object.assign(S, { meta, teams, dists: dists || {}, predictions: preds, evaluation: ev,
                     series: (series || {}).series || {}, cal });
  if (!teams.some(t => t.ok)) {
    document.querySelector('main').innerHTML = '<p class="note" style="margin-top:30px">Data loaded, but no team has enough games yet to build a distribution.</p>';
    return;
  }
  kpis(); table();
  const opts = teams.filter(t => t.ok).sort((a, b) => (a.abbrev || '').localeCompare(b.abbrev || ''))
    .map(t => `<option value="${t.id}">${t.abbrev} - ${t.name}</option>`).join('');
  ['teamsel', 'awaysel', 'homesel'].forEach(id => document.getElementById(id).innerHTML = opts);
  const ok = teams.filter(t => t.ok);
  document.getElementById('teamsel').value = ok[0].id;
  document.getElementById('awaysel').value = ok[0].id;
  document.getElementById('homesel').value = (ok[1] || ok[0]).id;
  document.getElementById('teamsel').onchange = e => showTeam(+e.target.value);
  document.getElementById('awaysel').onchange = matchup;
  document.getElementById('homesel').onchange = matchup;
  showTeam(ok[0].id); matchup(); upcoming(); evalview();
  window.addEventListener('resize', () => { showTeam(S.sel); matchup(); evalview(); });
})();
