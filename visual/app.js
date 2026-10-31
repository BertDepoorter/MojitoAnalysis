'use strict';
// Mojito EMRI explorer front end. Data products come from build.py (data/), live tracks and TD windows from serve.py.

const $ = s => document.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CFG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
const YR = 31558149.763545603, DAY = 86400;
const PLAB = ['M/M<sub>⊙</sub>', 'μ/M<sub>⊙</sub>', 'a', 'p<sub>0</sub>', 'e<sub>0</sub>', 'D<sub>L</sub>/Gpc',
  'θ<sub>S</sub>', 'φ<sub>S</sub>', 'θ<sub>K</sub>', 'φ<sub>K</sub>', 'Φ<sub>φ,0</sub>', 'Φ<sub>r,0</sub>'];
const TRACK_COLORS = ['#ffffff', '#ff8c69', '#f0a6ff', '#7fe3f0', '#ffe066', '#b4f08c'];
const HARM_OPTS = ['2:0:0', '2:0:1', '2:0:-1', '2:0:2', '1:0:0', '3:0:0', '3:0:1'];
const S = { k: null, P: null, loaded: {}, spec: null, tracks: [], chains: null, post: null };

// ------------------------------------------------------------------ helpers
const fmt = (v, d = 4) => v == null || !isFinite(v) ? '—'
  : (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(d - 1) : (+v.toPrecision(d)).toString();
const tex = s => window.katex ? katex.renderToString(s, { throwOnError: false }) : s;
const getJSON = async u => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); };
const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
const renderMath = el => window.renderMathInElement && renderMathInElement(el, {
  delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false });

function layout(extra = {}) {
  const ax = { gridcolor: css('--line'), zerolinecolor: css('--line'), linecolor: css('--line'), automargin: true };
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: { l: 60, r: 18, t: 30, b: 45 },
    font: { family: 'Inter, sans-serif', size: 12, color: css('--ink') }, legend: { bgcolor: 'rgba(0,0,0,0)' },
    xaxis: { ...ax }, yaxis: { ...ax }, hoverlabel: { font: { family: 'JetBrains Mono, monospace' } },
  };
  for (const [k, v] of Object.entries(extra))
    base[k] = (k.startsWith('xaxis') || k.startsWith('yaxis')) ? { ...ax, ...v } : v;
  return base;
}
const chColor = c => css({ X: '--chX', Y: '--chY', Z: '--chZ' }[c]);
const badge = (cls, text, title = '') => `<span class="badge ${cls}" title="${title}">${text}</span>`;
const grade = (v, ok, warn, lowerBetter = true) => v == null ? 'na'
  : lowerBetter ? (v < ok ? 'ok' : v < warn ? 'warn' : 'bad') : (v > ok ? 'ok' : v > warn ? 'warn' : 'bad');

// ------------------------------------------------------------------ init
async function init() {
  const man = await getJSON('data/manifest.json').catch(() => []);
  if (!man.length) { $('main').innerHTML = '<div class="placeholder">No data found. Run <code>python build.py</code> first.</div>'; return; }
  $('#chips').innerHTML = man.map(m =>
    `<button class="chip" data-k="${m.k}" aria-pressed="false"><b>Source ${m.k}</b><span>ρ ${m.SNR.toFixed(1)} · M ${(m.M / 1e5).toFixed(2)}e5</span></button>`).join('');
  $('#chips').onclick = e => { const b = e.target.closest('.chip'); if (b) select(+b.dataset.k); };
  $('#harms').innerHTML = HARM_OPTS.map((h, i) =>
    `<label><input type="checkbox" value="${h}" ${i < 2 ? 'checked' : ''}> ${h.replaceAll(':', '')}</label>`).join('');
  setupTheme();
  setupSpecControls();
  setupChainControls();
  const obs = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && loadSection(e.target.dataset.lazy)), { rootMargin: '200px' });
  document.querySelectorAll('[data-lazy]').forEach(el => obs.observe(el));
  renderMath(document.body);
  const k0 = +(location.hash.match(/source=(\d)/) || [0, man[0].k])[1];
  select(k0);
}

function setupTheme() {
  const modes = ['auto', 'light', 'dark'];
  const apply = m => {
    if (m === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.dataset.theme = m;
    $('#theme').textContent = m; try { localStorage.setItem('theme', m); } catch (e) { }
  };
  let m = 'auto'; try { m = localStorage.getItem('theme') || 'auto'; } catch (e) { }
  apply(m);
  $('#theme').onclick = () => { apply(modes[(modes.indexOf($('#theme').textContent) + 1) % 3]); rerender(); };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerender);
}
function rerender() { const k = S.k; S.k = null; select(k); }

async function select(k) {
  if (k === S.k) return;
  S.k = k; S.loaded = {}; S.spec = null; S.chains = null; S.post = null; S.tracks = [];
  history.replaceState(null, '', `#source=${k}`);
  document.querySelectorAll('.chip').forEach(b => b.setAttribute('aria-pressed', +b.dataset.k === k));
  const P = await getJSON(`data/source_${k}/params.json`);
  if (k !== S.k) return;
  S.P = P;
  renderParams();
  fillTruth();
  document.querySelectorAll('[data-lazy]').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.top < innerHeight + 200 && r.bottom > -200) loadSection(el.dataset.lazy);
  });
}

async function loadSection(name) {
  const k = S.k;
  if (k == null || !S.P || S.P.k !== k || S.loaded[name]) return;  // wait for params.json of the current source
  S.loaded[name] = true;
  try {
    await { td: loadTD, fd: loadFD, spec: loadSpec, chains: loadChains }[name](k);
  } catch (err) {
    console.error(err);
    S.loaded[name] = false;
  }
}

// ------------------------------------------------------------------ parameters + checks
const CAT_ROWS = [
  ['PrimaryMassSSBFrame', 'M', 'M_\\odot'], ['SecondaryMassSSBFrame', '\\mu', 'M_\\odot'], ['MassRatio', 'q=\\mu/M', ''],
  ['PrimarySpinParameter', 'a', ''], ['SemiLatusRectum', 'p_0', ''], ['Eccentricity', 'e_0', ''],
  ['InclinationAngle', '\\iota\\;(x_{I,0}=\\cos\\iota)', 'rad'], ['LuminosityDistance', 'D_L', '\\mathrm{Mpc}'], ['Redshift', 'z', ''],
  ['RightAscension', '\\alpha', 'rad'], ['Declination', '\\delta', 'rad'],
  ['PolarAnglePrimarySpin', '\\theta_K', 'rad'], ['AzimuthalAnglePrimarySpin', '\\phi_K', 'rad'],
  ['AzimuthalPhase', '\\Phi_{\\phi,0}', 'rad'], ['PolarPhase', '\\Phi_{\\theta,0}', 'rad'], ['RadialPhase', '\\Phi_{r,0}', 'rad'],
  ['PrimaryMassSourceFrame', 'M^{\\rm src}', 'M_\\odot'], ['SecondaryMassSourceFrame', '\\mu^{\\rm src}', 'M_\\odot'],
  ['ChirpMassSSBFrame', '\\mathcal{M}', 'M_\\odot'], ['TimeCoalescencePetersSSBFrame', 't_c^{\\rm Peters}', '\\mathrm{s}'],
];

function renderParams() {
  const P = S.P, c = P.catalog, d = P.derived, s = P.snr, m = P.meta;
  $('#brick').textContent = P.brick;
  const row = (sym, v, unit, flag = false) =>
    `<tr class="${flag ? 'flag' : ''}"><td>${tex(sym)}</td><td class="num">${typeof v === 'string' ? v : fmt(v)}</td><td>${unit ? tex(unit) : ''}</td></tr>`;
  const head = '<tr><th>Quantity</th><th style="text-align:right">Value</th><th>Unit</th></tr>';
  $('#tab-cat').innerHTML = head + CAT_ROWS.map(([key, sym, u]) =>
    row(sym, key === 'InclinationAngle' ? `${fmt(c[key])} (${Math.round(Math.cos(c[key])) > 0 ? '+1' : '−1'})` : c[key], u)).join('');
  const Tobs = m.T_obs / YR;
  $('#tab-der').innerHTML = head + [
    row('T_{\\rm plunge}\\ (\\text{from } t_{\\rm ref})', d.T_plunge / YR, '\\mathrm{yr}', !d.plunges_in_data),
    row('T_{\\rm plunge} - t_{0,\\rm L1}', d.T_plunge_data / YR, '\\mathrm{yr}'),
    row('T_{\\rm obs}', Tobs, '\\mathrm{yr}'),
    row('e_{\\rm final}', d.e_final, ''), row('p_{\\rm final}', d.p_final, ''),
    row('|f_\\phi|_{\\rm final}', d.f_phi_final, '\\mathrm{Hz}'),
    row('e\\ \\text{at end of data}', d.e_end_data, ''), row('p\\ \\text{at end of data}', d.p_end_data, ''),
    row('\\theta_S = \\pi/2-\\beta\\ (\\text{ecl.})', P.truth[6], 'rad'), row('\\phi_S = \\lambda\\ (\\text{ecl.})', P.truth[7], 'rad'),
    row('\\rho_{XYZ}', s.XYZ, ''), row('\\rho_{AET}', s.AET, ''), row('\\rho_{\\rm ref}\\ (\\texttt{results.md})', s.reference, ''),
    row('\\rho_X,\\ \\rho_Y,\\ \\rho_Z', [s.X, s.Y, s.Z].map(v => fmt(v)).join(', '), ''),
    row('\\rho_A,\\ \\rho_E,\\ \\rho_T', [s.A, s.E, s.T].map(v => fmt(v)).join(', '), ''),
  ].join('');
  renderBadges();
}

function renderBadges() {
  const ch = S.P.checks;
  const b = [
    badge(grade(ch.snr_rel_err, 0.01, 0.05), `SNR vs reference: ${ch.snr_rel_err == null ? 'n/a' : (100 * ch.snr_rel_err).toFixed(3) + ' %'}`, 'ρ_XYZ vs results.md, ok < 1 %'),
    badge(grade(ch.parseval_max_rel_err, 1e-3, 1e-2), `Parseval: ${fmt(ch.parseval_max_rel_err, 2)}`, 'Δt Σ(wd)² vs Δf Σ|d̃|², ok < 1e-3'),
    badge(grade(ch.ridge_offset_bins, 1, 2), `Ridge on f<sub>${ch.ridge_harmonic ?? '—'}</sub>: ${fmt(ch.ridge_offset_bins, 2)} bins`,
      `median |log offset| of the spectrogram ridge (X) to the nearest true harmonic |m f_phi + n f_r|, in log-f bins (offset to f_200: ${fmt(ch.ridge_offset_200_bins, 2)} bins)`),
    badge(ch.noise_finite_positive ? 'ok' : 'bad', 'Noise PSD finite & positive'),
    badge(ch.noise_covers_data ? 'ok' : 'warn', 'Noise span covers data'),
  ];
  $('#badges').innerHTML = b.join('');
}

// ------------------------------------------------------------------ time domain
async function loadTD(k) {
  const td = await getJSON(`data/source_${k}/td_overview.json`);
  if (k !== S.k) return;
  S.td = td;
  const el = $('#plot-td');
  drawTD(['X', 'Y', 'Z'].map(c => ({ c, t: td.t_days, min: td[c].min, max: td[c].max })), null);
  el.removeAllListeners?.('plotly_relayout');
  el.on('plotly_relayout', debounce(ev => onTDZoom(ev), 250));
}

function drawTD(series, range) {
  const traces = [], n = series.length;
  const ext = { xaxis: { title: 'time since first L1 sample [days]', range: range || undefined, autorange: !range } };
  series.forEach((s, i) => {
    const ax = i ? String(i + 1) : '', col = chColor(s.c);
    if (s.y) traces.push({ x: s.t, y: s.y, type: 'scattergl', mode: 'lines', line: { color: col, width: 1 }, name: s.c, yaxis: 'y' + ax, xaxis: 'x' });
    else {
      traces.push({ x: s.t, y: s.min, type: 'scatter', mode: 'lines', line: { color: col, width: 0.6 }, yaxis: 'y' + ax, showlegend: false, hoverinfo: 'skip' });
      traces.push({ x: s.t, y: s.max, type: 'scatter', mode: 'lines', line: { color: col, width: 0.6 }, fill: 'tonexty', fillcolor: col + '55', name: s.c, yaxis: 'y' + ax });
    }
    const h = 1 / n;
    ext['yaxis' + ax] = { domain: [1 - (i + 1) * h + 0.04, 1 - i * h], title: `d<sub>${s.c}</sub>`, exponentformat: 'power' };
  });
  S.tdSilent = true;
  Plotly.react('plot-td', traces, layout({ ...ext, showlegend: false, margin: { l: 70, r: 18, t: 10, b: 45 } }), CFG)
    .then(() => setTimeout(() => S.tdSilent = false, 50));
}

async function onTDZoom(ev) {
  if (S.tdSilent) return;
  if (ev['xaxis.autorange']) return drawTD(['X', 'Y', 'Z'].map(c => ({ c, t: S.td.t_days, min: S.td[c].min, max: S.td[c].max })), null);
  const r0 = ev['xaxis.range[0]'] ?? ev['xaxis.range']?.[0], r1 = ev['xaxis.range[1]'] ?? ev['xaxis.range']?.[1];
  if (r0 == null) return;
  const k = S.k;
  const res = await Promise.all(['X', 'Y', 'Z'].map(c =>
    getJSON(`api/td?k=${k}&ch=${c}&t0=${r0 * DAY}&t1=${r1 * DAY}`).then(d => ({ c, ...d }))));
  if (k !== S.k) return;
  drawTD(res.map(d => d.mode === 'raw' ? { c: d.c, t: d.t.map(t => t / DAY), y: d.y }
    : { c: d.c, t: d.t.map(t => t / DAY), min: d.min, max: d.max }), [r0, r1]);
}

// ------------------------------------------------------------------ frequency domain
async function loadFD(k) {
  const fd = await getJSON(`data/source_${k}/fd.json`);
  if (k !== S.k) return;
  const tr = [];
  for (const c of ['X', 'Y', 'Z']) {
    tr.push({ x: fd.f, y: fd[c], type: 'scattergl', mode: 'lines', name: `h<sub>${c}</sub>  (ρ<sub>${c}</sub>=${fmt(S.P.snr[c], 3)})`, legendgroup: c, line: { color: chColor(c), width: 1.4 } });
    tr.push({ x: fd.f, y: fd['n' + c], type: 'scatter', mode: 'lines', name: `n<sub>${c}</sub>`, legendgroup: c, line: { color: chColor(c), width: 1.2, dash: 'dash' }, opacity: 0.75 });
  }
  const finite = fd.X.filter(v => v > 0);
  const ymax = Math.max(...finite) * 5;
  Plotly.react('plot-fd', tr, layout({
    xaxis: { type: 'log', title: 'f [Hz]', exponentformat: 'power' },
    yaxis: { type: 'log', title: 'dimensionless', exponentformat: 'power', range: [Math.log10(ymax) - 6, Math.log10(ymax)] },
    legend: { orientation: 'h', y: 1.1 },
  }), CFG);
}

// ------------------------------------------------------------------ spectrogram + tracks
async function loadSpec(k) {
  const [meta, buf] = await Promise.all([getJSON(`data/source_${k}/spec.json`),
    fetch(`data/source_${k}/spec.bin`).then(r => r.arrayBuffer())]);
  if (k !== S.k) return;
  const [nc, nf, nt] = meta.shape, all = new Float32Array(buf);
  let mx = -Infinity, mn = Infinity;
  for (const v of all) { if (v > mx) mx = v; if (v < mn) mn = v; }
  S.spec = { meta, all, nf, nt, mx, mn };
  for (const id of ['#spec-zmin', '#spec-zmax']) { $(id).min = Math.floor(mn); $(id).max = Math.ceil(mx); }
  $('#spec-zmin').value = mx - 8; $('#spec-zmax').value = mx;
  const tt = await getJSON(`data/source_${k}/track_true.json`);
  if (k !== S.k) return;
  S.tracks = [{ label: 'injection', data: tt, color: TRACK_COLORS[0] }];
  drawSpec();
}

function specZ(ch) {
  const { all, nf, nt } = S.spec, off = ch * nf * nt, z = new Array(nf);
  for (let i = 0; i < nf; i++) z[i] = Array.from(all.subarray(off + i * nt, off + (i + 1) * nt));
  return z;
}

function drawSpec() {
  if (!S.spec) return;
  const ch = 'XYZ'.indexOf($('#spec-ch').value), m = S.spec.meta;
  const zmin = +$('#spec-zmin').value, zmax = +$('#spec-zmax').value;
  $('#spec-zlab').textContent = `[${zmin.toFixed(1)}, ${zmax.toFixed(1)}]`;
  const traces = [{
    type: 'heatmap', x: m.t_days, y: m.f, z: specZ(ch), zmin, zmax, colorscale: $('#spec-cmap').value,
    colorbar: { title: { text: 'log<sub>10</sub> P [Hz<sup>-1</sup>]', side: 'right' }, thickness: 12 },
    hovertemplate: 't=%{x:.1f} d<br>f=%{y:.3e} Hz<br>log₁₀P=%{z:.2f}<extra></extra>',
  }];
  S.tracks.forEach(T => {
    Object.entries(T.data.tracks).forEach(([h, f]) => {
      traces.push({
        type: 'scatter', mode: 'lines', x: T.data.t_data.map(t => t / DAY), y: f, name: `${T.label} · ${h}`,
        line: { color: T.color, width: 1.5, dash: h.endsWith('00') ? 'solid' : h.endsWith('01') ? 'dash' : 'dot' },
        hovertemplate: `${T.label} f<sub>${h}</sub>=%{y:.4e} Hz<extra></extra>`,
      });
    });
  });
  Plotly.react('plot-spec', traces, layout({
    xaxis: { title: 'time since first L1 sample [days]', range: [m.t_days[0], m.t_days[m.t_days.length - 1]] },
    yaxis: { type: 'log', title: 'f [Hz]', exponentformat: 'power', range: [Math.log10(m.f_edges[0]), Math.log10(m.f_edges[m.f_edges.length - 1])] },
    showlegend: false, margin: { l: 70, r: 10, t: 10, b: 45 },
  }), CFG);
  renderTrackList();
}

function setupSpecControls() {
  const redraw = debounce(drawSpec, 60);
  ['#spec-ch', '#spec-cmap', '#spec-zmin', '#spec-zmax'].forEach(id => $(id).addEventListener('input', redraw));
  $('#btn-track').onclick = () => computeTrack();
  $('#btn-truth').onclick = fillTruth;
  $('#btn-median').onclick = () => { if (S.post) fillForm(S.post.median); };
  $('#btn-draw').onclick = () => {
    if (!S.post) return;
    const s = S.post.samples, i = Math.floor(Math.random() * s.length);
    fillForm(s[i]); computeTrack(`draw #${i}`);
  };
}

function fillTruth() {
  const c = S.P.catalog;
  fillForm([c.PrimaryMassSSBFrame, c.SecondaryMassSSBFrame, c.PrimarySpinParameter, c.SemiLatusRectum, c.Eccentricity]);
  $('#f-x0').value = Math.round(Math.cos(c.InclinationAngle)) > 0 ? '1' : '-1';
  $('#f-T').value = (S.P.meta.T_obs / YR).toFixed(4);
}
function fillForm(v) { ['M', 'mu', 'a', 'p0', 'e0'].forEach((n, i) => $('#f-' + n).value = +(+v[i]).toPrecision(8)); }

async function computeTrack(label) {
  const q = new URLSearchParams();
  ['M', 'mu', 'a', 'p0', 'e0', 'x0', 'T'].forEach(n => q.set(n, $('#f-' + n).value));
  const harm = [...document.querySelectorAll('#harms input:checked')].map(i => i.value);
  if (!harm.length) return setMsg('select at least one harmonic', true);
  q.set('harm', harm.join(','));
  setMsg('computing trajectory…');
  const t0 = performance.now();
  const r = await fetch('api/trajectory?' + q);
  const d = await r.json();
  if (!r.ok) return setMsg(d.error || 'error', true);
  const used = new Set(S.tracks.map(t => t.color));
  const color = TRACK_COLORS.find(c => !used.has(c)) || TRACK_COLORS[S.tracks.length % TRACK_COLORS.length];
  S.tracks.push({ label: label || `track ${S.tracks.length}`, data: d, color });
  setMsg(`T_plunge = ${fmt(d.t_plunge / YR)} yr from t_ref · e_final = ${fmt(d.e_final)} · ${(performance.now() - t0).toFixed(0)} ms`
    + (d.plunged ? '' : ' (no plunge within T)'));
  drawSpec();
}
function setMsg(t, err = false) { const m = $('#traj-msg'); m.textContent = t; m.className = 'msg' + (err ? ' err' : ''); }
function renderTrackList() {
  $('#tracklist').innerHTML = S.tracks.map((t, i) =>
    `<li><span class="sw" style="background:${t.color}"></span>${t.label}<span class="mono muted">${Object.keys(t.data.tracks).join(' ')}</span>${i ? `<button data-i="${i}" title="remove">✕</button>` : ''}</li>`).join('');
  $('#tracklist').onclick = e => { const b = e.target.closest('button'); if (b) { S.tracks.splice(+b.dataset.i, 1); drawSpec(); } };
}

// ------------------------------------------------------------------ chains
async function loadChains(k) {
  if (!S.P.has_chains) {
    $('#chain-body').style.display = 'none'; $('#chain-info').innerHTML = '<span class="placeholder">No posterior chains for this source.</span>';
    return;
  }
  $('#chain-body').style.display = ''; $('#chain-info').textContent = 'loading chains…';
  const [meta, buf] = await Promise.all([getJSON(`data/source_${k}/chains.json`),
    fetch(`data/source_${k}/chains.bin`).then(r => r.arrayBuffer())]);
  if (k !== S.k) return;
  const { N_it, N_w, N_p } = meta, n = N_it * N_w * N_p;
  S.chains = { ...meta, x: new Float32Array(buf, 0, n), ll: new Float32Array(buf, 4 * n, N_it * N_w) };
  $('#chain-info').innerHTML = `${N_it} iterations × ${N_w} walkers × ${N_p} parameters (eryn, coldest temperature) · acceptance ${fmt(Math.min(...meta.acceptance_fraction), 2)}–${fmt(Math.max(...meta.acceptance_fraction), 2)}`;
  const opts = PLAB.map((l, i) => `<option value="${i}">${l.replace(/<\/?su[bp]>/g, '')}</option>`).join('');
  $('#c-par').innerHTML = opts; $('#c-px').innerHTML = opts; $('#c-py').innerHTML = opts;
  $('#c-px').value = 0; $('#c-py').value = 1;
  $('#c-nw').max = N_w;
  $('#c-burn').max = Math.max(0, N_it - 50); $('#c-burn').value = Math.floor(N_it / 2);
  ['#btn-median', '#btn-draw'].forEach(id => $(id).disabled = false);
  updateChains();
}

function setupChainControls() {
  const upd = debounce(updateChains, 120);
  $('#c-burn').addEventListener('input', () => { $('#c-burn-lab').textContent = $('#c-burn').value; upd(); });
  ['#c-par', '#c-nw', '#c-px', '#c-py', '#c-nothin'].forEach(id => $(id).addEventListener('input', upd));
}

const cx = (C, it, w, p) => C.x[(it * C.N_w + w) * C.N_p + p];

// radix-2 complex FFT, in place
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

// integrated autocorrelation time (walker-averaged ACF, Sokal window c=5; as emcee.autocorr)
function autocorrTime(C, p, b) {
  const n = C.N_it - b; let m = 1; while (m < 2 * n) m <<= 1;
  const acf = new Float64Array(n);
  for (let w = 0; w < C.N_w; w++) {
    const re = new Float64Array(m), im = new Float64Array(m);
    let mean = 0; for (let t = 0; t < n; t++) mean += cx(C, b + t, w, p); mean /= n;
    for (let t = 0; t < n; t++) re[t] = cx(C, b + t, w, p) - mean;
    fft(re, im);
    for (let i = 0; i < m; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
    fft(re, im); // power spectrum is real and even: forward FFT = m * inverse
    if (re[0] > 0) for (let t = 0; t < n; t++) acf[t] += re[t] / re[0];
  }
  for (let t = 0; t < n; t++) acf[t] /= C.N_w;
  let tau = 1;
  for (let l = 1; l < n; l++) { tau += 2 * acf[l]; if (l >= 5 * tau) return tau; }
  return tau;
}

function quantile(sorted, q) { const i = (sorted.length - 1) * q, lo = Math.floor(i); return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo); }

function updateChains() {
  const C = S.chains; if (!C) return;
  const b = Math.min(+$('#c-burn').value, C.N_it - 2), P = C.N_p, nW = Math.min(+$('#c-nw').value || 10, C.N_w);
  $('#c-burn-lab').textContent = b;
  const par = +$('#c-par').value, n = C.N_it - b;

  // autocorrelation and thinning
  const tau = Array.from({ length: P }, (_, p) => autocorrTime(C, p, b));
  const tauMax = Math.max(...tau), thin = $('#c-nothin').checked ? 1 : Math.max(1, Math.ceil(tauMax));
  const samples = [];
  for (let it = b; it < C.N_it; it += thin) for (let w = 0; w < C.N_w; w++) {
    const s = new Array(P); for (let p = 0; p < P; p++) s[p] = cx(C, it, w, p); samples.push(s);
  }
  const stats = Array.from({ length: P }, (_, p) => {
    const v = samples.map(s => s[p]).sort((a, c) => a - c), mu = v.reduce((a, c) => a + c, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, c) => a + (c - mu) ** 2, 0) / Math.max(1, v.length - 1));
    const med = quantile(v, 0.5);
    return { med, lo: quantile(v, 0.16), hi: quantile(v, 0.84), sd, z: (med - C.truth[p]) / sd, vals: v };
  });
  S.post = { samples, median: stats.map(s => s.med) };
  const neff = C.N_w * n / tauMax;

  // running variance (Welford over iterations, all walkers) and split-Rhat at checkpoints
  const runVar = Array.from({ length: P }, () => new Float64Array(n));
  for (let p = 0; p < P; p++) {
    let cnt = 0, mean = 0, M2 = 0;
    for (let t = 0; t < n; t++) {
      for (let w = 0; w < C.N_w; w++) { const x = cx(C, b + t, w, p); cnt++; const d = x - mean; mean += d / cnt; M2 += d * (x - mean); }
      runVar[p][t] = cnt > 1 ? M2 / (cnt - 1) : NaN;
    }
  }
  const its = Array.from({ length: n }, (_, t) => b + t);
  const ckpts = [], rhat = Array.from({ length: P }, () => []);
  for (let j = 1; j <= 50; j++) {
    const L = Math.floor(n * j / 50); if (L < 8) continue;
    ckpts.push(b + L);
    const h = Math.floor(L / 2);
    for (let p = 0; p < P; p++) {
      const means = [], vars = [];
      for (let w = 0; w < C.N_w; w++) for (let s = 0; s < 2; s++) {
        let m = 0; for (let t = 0; t < h; t++) m += cx(C, b + s * h + t, w, p); m /= h;
        let v = 0; for (let t = 0; t < h; t++) v += (cx(C, b + s * h + t, w, p) - m) ** 2; v /= (h - 1);
        means.push(m); vars.push(v);
      }
      const W = vars.reduce((a, c) => a + c, 0) / vars.length, mm = means.reduce((a, c) => a + c, 0) / means.length;
      const B = h * means.reduce((a, c) => a + (c - mm) ** 2, 0) / (means.length - 1);
      rhat[p].push(Math.sqrt(((h - 1) / h * W + B / h) / W));
    }
  }
  const rhatFinal = Math.max(...rhat.map(r => r[r.length - 1]));

  // badges + tau caption
  const acc = C.acceptance_fraction, accOk = acc.every(a => a >= 0.1 && a <= 0.6);
  const zmax = Math.max(...stats.map(s => Math.abs(s.z)));
  $('#chain-badges').innerHTML = [
    badge(accOk ? 'ok' : 'warn', `Acceptance ${fmt(Math.min(...acc), 2)}–${fmt(Math.max(...acc), 2)}`, 'ok within [0.1, 0.6]'),
    badge(grade(rhatFinal, 1.01, 1.1), `max split-R̂ = ${rhatFinal.toFixed(3)}`, 'ok < 1.01'),
    badge(grade(neff, 500, 100, false), `N_eff = ${neff.toFixed(0)}`, 'N_w (N_it − b)/τ_max, ok > 500'),
    badge(grade(zmax, 3, 5), `max |z| = ${zmax.toFixed(2)}`, '(median − truth)/σ, ok < 3'),
    badge(n >= 50 * tauMax ? 'ok' : 'warn', n >= 50 * tauMax ? 'τ estimate reliable' : `τ unreliable: N_it − b = ${n} < 50 τ_max = ${(50 * tauMax).toFixed(0)}`),
  ].join('');
  const te = C.tau_eryn;
  $('#chain-tau').innerHTML = `τ<sub>max</sub> = ${tauMax.toFixed(1)} iterations → thinning by ${thin} · ${samples.length} samples` +
    (te ? ` · eryn cross-check (discard N<sub>it</sub>/2): τ<sub>max</sub> = ${Math.max(...te).toFixed(1)}` : '');

  // trace + log-likelihood
  const tr = [], itsAll = Array.from({ length: C.N_it }, (_, i) => i);
  for (let w = 0; w < nW; w++) {
    tr.push({ x: itsAll, y: itsAll.map(i => cx(C, i, w, par)), type: 'scattergl', mode: 'lines', line: { width: 0.7, color: chColor('X') }, opacity: 0.45, showlegend: false, hoverinfo: 'skip' });
    tr.push({ x: itsAll, y: itsAll.map(i => C.ll[i * C.N_w + w]), type: 'scattergl', mode: 'lines', line: { width: 0.7, color: chColor('Y') }, opacity: 0.45, showlegend: false, hoverinfo: 'skip', yaxis: 'y2' });
  }
  const meanTrace = itsAll.map(i => { let s = 0; for (let w = 0; w < C.N_w; w++) s += cx(C, i, w, par); return s / C.N_w; });
  tr.push({ x: itsAll, y: meanTrace, type: 'scattergl', mode: 'lines', line: { width: 2, color: css('--title') }, name: 'walker mean' });
  const shade = yref => ({ type: 'rect', xref: 'x', yref, x0: 0, x1: b, y0: 0, y1: 1, fillcolor: css('--muted'), opacity: 0.12, line: { width: 0 } });
  const truthLine = { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: C.truth[par], y1: C.truth[par], line: { color: css('--bad'), width: 1, dash: 'dot' } };
  Plotly.react('plot-trace', tr, layout({
    xaxis: { title: 'iteration' }, yaxis: { domain: [0.42, 1], title: PLAB[par] }, yaxis2: { domain: [0, 0.34], title: 'log L' },
    shapes: [shade('y domain'), shade('y2 domain'), truthLine], showlegend: false, title: { text: 'Trace (unthinned)', font: { size: 13 } },
  }), CFG);

  // running variance, normalised to the final value
  Plotly.react('plot-var', runVar.map((v, p) => ({
    x: its, y: Array.from(v, x => x / v[n - 1]), type: 'scattergl', mode: 'lines', name: PLAB[p],
    line: { width: p === par ? 2.4 : 1 }, opacity: p === par ? 1 : 0.6,
  })), layout({
    xaxis: { title: 'iteration' }, yaxis: { type: 'log', title: 'Var<sub>t</sub>(θ) / Var<sub>final</sub>(θ)' },
    title: { text: `Running variance from burn-in b = ${b}`, font: { size: 13 } }, legend: { font: { size: 10 } },
  }), CFG);

  Plotly.react('plot-rhat', [
    { x: ckpts, y: ckpts.map((_, j) => Math.max(...rhat.map(r => r[j]))), type: 'scatter', mode: 'lines+markers', name: 'max over θ', line: { color: css('--title') }, marker: { size: 4 } },
    { x: ckpts, y: rhat[par], type: 'scatter', mode: 'lines', name: PLAB[par], line: { color: chColor('Z') } },
  ], layout({
    xaxis: { title: 'iteration' }, yaxis: { type: 'log', title: 'split-R̂' }, legend: { orientation: 'h', y: 1.12 },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1.01, y1: 1.01, line: { color: css('--ok'), dash: 'dash', width: 1 } }],
  }), CFG);

  const px = +$('#c-px').value, py = +$('#c-py').value;
  Plotly.react('plot-2d', [
    { type: 'histogram2dcontour', x: samples.map(s => s[px]), y: samples.map(s => s[py]), colorscale: 'Greens', reversescale: false, showscale: false, ncontours: 12, contours: { coloring: 'fill', showlines: false } },
    { type: 'scatter', mode: 'markers', x: [C.truth[px]], y: [C.truth[py]], marker: { symbol: 'x', size: 11, color: css('--bad') }, name: 'truth' },
  ], layout({ xaxis: { title: PLAB[px], exponentformat: 'e' }, yaxis: { title: PLAB[py], exponentformat: 'e' }, showlegend: false }), CFG);

  // 1D marginals
  const mt = [], shapes = [], ext = {};
  stats.forEach((s, p) => {
    const a = p ? String(p + 1) : '';
    mt.push({ type: 'histogram', x: s.vals, xaxis: 'x' + a, yaxis: 'y' + a, nbinsx: 30, marker: { color: chColor('X') }, opacity: 0.75, showlegend: false, hovertemplate: '%{x}<extra></extra>' });
    shapes.push({ type: 'line', xref: 'x' + a, yref: `y${a} domain`, x0: C.truth[p], x1: C.truth[p], y0: 0, y1: 1, line: { color: css('--bad'), width: 1.4 } });
    ext['xaxis' + a] = { title: { text: PLAB[p], font: { size: 11 } }, tickfont: { size: 9 }, nticks: 4, exponentformat: 'e' };
    ext['yaxis' + a] = { showticklabels: false };
  });
  Plotly.react('plot-marg', mt, layout({ ...ext, grid: { rows: 3, columns: 4, pattern: 'independent', ygap: 0.55, xgap: 0.12 }, shapes, margin: { l: 20, r: 10, t: 10, b: 40 } }), CFG);

  // summary table
  $('#tab-post').innerHTML = '<tr><th>Parameter</th><th style="text-align:right">Median</th><th style="text-align:right">68 % interval</th><th style="text-align:right">Truth</th><th style="text-align:right">σ</th><th style="text-align:right">z</th><th style="text-align:right">τ</th></tr>' +
    stats.map((s, p) => `<tr class="${Math.abs(s.z) > 3 ? 'flag' : ''}"><td>${tex(C.labels[p])}</td><td class="num">${fmt(s.med, 6)}</td><td class="num">[${fmt(s.lo, 5)}, ${fmt(s.hi, 5)}]</td><td class="num">${fmt(C.truth[p], 6)}</td><td class="num">${fmt(s.sd, 3)}</td><td class="num">${s.z.toFixed(2)}</td><td class="num">${tau[p].toFixed(1)}</td></tr>`).join('');
}

window.addEventListener('load', init);
