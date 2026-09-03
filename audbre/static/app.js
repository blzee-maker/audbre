/* AudBre — front end.
   The waveform is drawn by hand rather than pulled from a library: the design
   needs an inverted selection band and a grey ghost of the original sitting
   behind the cleaned signal, which is easier to own than to configure. */

const $ = (id) => document.getElementById(id);
const INK = '#000000', DIM = '#767676', FAINT = '#C9C9C9', PAPER = '#FFFFFF';

const state = {
  project: null,
  peaksBase: [],
  peaksCurrent: [],
  sel: null,        // {from, to} in seconds
  job: null,        // finished job awaiting keep/discard
  source: 'current',
  format: 'wav',
  audio: new Audio(),
  busy: false,
};

/* ---------------- helpers ---------------- */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function tc(s, decimals = 1) {
  if (!isFinite(s)) return '--:--';
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, '0')}:${rest.toFixed(decimals).padStart(decimals ? 4 + decimals : 2, '0')}`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res;
}

function resample(peaks, n) {
  if (!peaks.length || n <= 0) return new Array(Math.max(n, 0)).fill(0);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * peaks.length / n);
    const b = Math.max(a + 1, Math.floor((i + 1) * peaks.length / n));
    let peak = 0;
    for (let j = a; j < b && j < peaks.length; j++) peak = Math.max(peak, peaks[j]);
    out[i] = peak;
  }
  return out;
}

/* ---------------- waveform ---------------- */

const canvas = $('wave');
const ctx = canvas.getContext('2d');

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const pitch = 3, barW = 2;
  const count = Math.max(1, Math.floor(w / pitch));
  const base = resample(state.peaksBase, count);
  const cur = resample(state.peaksCurrent.length ? state.peaksCurrent : state.peaksBase, count);
  const dur = state.project?.duration || 0;
  const mid = h / 2;

  // selection band, painted first so bars can invert on top of it
  let selPx = null;
  if (state.sel && dur > 0) {
    const x1 = (state.sel.from / dur) * w, x2 = (state.sel.to / dur) * w;
    selPx = [Math.min(x1, x2), Math.max(x1, x2)];
    ctx.fillStyle = INK;
    ctx.fillRect(selPx[0], 0, selPx[1] - selPx[0], h);
  }
  const inSel = (x) => selPx && x >= selPx[0] && x < selPx[1];

  for (let i = 0; i < count; i++) {
    const x = i * pitch;
    const sel = inSel(x);

    // ghost of the original — what used to be there
    const gh = Math.max(1, base[i] * (h - 4));
    if (gh > 1.5) {
      ctx.fillStyle = sel ? DIM : FAINT;
      ctx.fillRect(x, mid - gh / 2, barW, gh);
    }
    // the signal as it stands now
    const ch = Math.max(1, cur[i] * (h - 4));
    ctx.fillStyle = sel ? PAPER : INK;
    ctx.fillRect(x, mid - ch / 2, barW, ch);
  }

  // playhead
  const a = state.audio;
  if (a.duration && a.currentTime > 0) {
    const x = (a.currentTime / a.duration) * w;
    ctx.fillStyle = inSel(x) ? PAPER : INK;
    ctx.fillRect(Math.round(x), 0, 1, h);
  }
}

function ruler() {
  const dur = state.project?.duration || 0;
  const marks = 9;
  $('ruler').innerHTML = Array.from({ length: marks }, (_, i) => {
    const t = (dur * i) / (marks - 1);
    return `<span>${tc(t, 0)}</span>`;
  }).join('');
}

/* selection by dragging; a click without a drag seeks instead */
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (!state.project) return;
  canvas.setPointerCapture(e.pointerId);
  const x = e.offsetX;
  drag = { x0: x, moved: false };
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dur = state.project.duration;
  const w = canvas.clientWidth;
  if (Math.abs(e.offsetX - drag.x0) > 3) drag.moved = true;
  if (!drag.moved) return;
  const a = clamp(drag.x0 / w, 0, 1) * dur;
  const b = clamp(e.offsetX / w, 0, 1) * dur;
  state.sel = { from: Math.min(a, b), to: Math.max(a, b) };
  paintSelection();
  draw();
});
canvas.addEventListener('pointerup', (e) => {
  if (!drag) return;
  if (!drag.moved) {
    const t = clamp(e.offsetX / canvas.clientWidth, 0, 1) * state.project.duration;
    if (state.audio.duration) state.audio.currentTime = t;
    draw();
  }
  drag = null;
});

function paintSelection() {
  const has = !!state.sel && (state.sel.to - state.sel.from) > 0.02;
  $('marktag').hidden = !has;
  $('clearmark').hidden = !has;
  $('markrange').textContent = has ? `${tc(state.sel.from)}  →  ${tc(state.sel.to)}` : '';
  $('markdur').textContent = has ? `(${(state.sel.to - state.sel.from).toFixed(1)}s)` : '';
  if (!has) state.sel = null;
}

/* ---------------- audio ---------------- */

function sourceUrl(kind) {
  const p = state.project;
  if (!p) return '';
  if (state.job && kind === 'current') return `/api/jobs/${state.job.id}/audio/residual?t=${Date.now()}`;
  if (kind === 'target') {
    return state.job ? `/api/jobs/${state.job.id}/audio/target?t=${Date.now()}` : '';
  }
  return `/api/projects/${p.id}/audio/${kind}?t=${Date.now()}`;
}

function setSource(kind, { keepTime = true } = {}) {
  const at = keepTime ? state.audio.currentTime : 0;
  const playing = !state.audio.paused;
  state.source = kind;
  state.audio.src = sourceUrl(kind);
  state.audio.addEventListener('loadedmetadata', function once() {
    state.audio.removeEventListener('loadedmetadata', once);
    if (at && at < state.audio.duration) state.audio.currentTime = at;
    if (playing) state.audio.play().catch(() => {});
  });
  document.querySelectorAll('.ab .chip').forEach((c) => c.classList.toggle('on', c.dataset.src === kind));
}

state.audio.addEventListener('play', () => { $('play').innerHTML = '&#9646;&#9646; PAUSE'; tick(); });
state.audio.addEventListener('pause', () => { $('play').innerHTML = '&#9654; PLAY'; });
state.audio.addEventListener('ended', () => { $('play').innerHTML = '&#9654; PLAY'; draw(); });

function tick() {
  if (state.audio.paused) return;
  draw();
  requestAnimationFrame(tick);
}

$('play').onclick = () => {
  if (!state.project) return;
  if (state.audio.paused) state.audio.play().catch(() => {}); else state.audio.pause();
};
document.querySelectorAll('.ab .chip').forEach((chip) => {
  chip.onclick = () => { if (!chip.disabled) setSource(chip.dataset.src); };
});
$('clearmark').onclick = () => { state.sel = null; paintSelection(); draw(); };

/* ---------------- project ---------------- */

async function openProject(data) {
  state.project = data;
  state.peaksBase = data.peaks || [];
  state.job = null;
  state.sel = null;
  $('empty').hidden = true;
  $('editor').hidden = false;
  $('fname').textContent = data.name;
  $('fmeta').textContent = `${tc(data.duration, 0)}  ·  ${(data.sample_rate / 1000).toFixed(1)} kHz  ·  ${data.has_video ? 'VIDEO' : 'AUDIO'}`;
  $('fmt-video').hidden = !data.has_video;
  $('saved').hidden = true;
  hideResult();
  paintSelection();
  ruler();
  await refreshCurrent();
  setSource('current', { keepTime: false });
  renderLayers();
}

async function refreshCurrent() {
  try {
    const { peaks } = await api(`/api/projects/${state.project.id}/waveform/current`);
    state.peaksCurrent = peaks;
  } catch { state.peaksCurrent = []; }
  draw();
}

function renderLayers() {
  const p = state.project;
  const ul = $('layers');
  const active = p.layers.filter((l) => l.enabled).length;
  $('stackcount').textContent = p.layers.length ? `${active} ACTIVE` : '';

  if (!p.layers.length) {
    ul.innerHTML = '<li class="empty">NOTHING REMOVED YET</li>';
    return;
  }
  ul.innerHTML = p.layers.map((l, i) => {
    const at = l.anchors?.length ? tc(l.anchors[0][1]) : 'ALL';
    const len = l.anchors?.length
      ? `${l.anchors.reduce((s, a) => s + (a[2] - a[1]), 0).toFixed(1)}s`
      : 'cont.';
    return `<li class="${l.enabled ? 'on' : ''}" data-id="${l.id}">
      <span class="n">${String(i + 1).padStart(2, '0')}</span>
      <button class="box" title="mute or restore">${l.enabled ? '[x]' : '[ ]'}</button>
      <span class="desc">${escapeHtml(l.description || '(marked span)')}</span>
      <span class="leader"></span>
      <span class="at">${at}</span>
      <span class="len">${len}</span>
      <button class="del" title="delete">&times;</button>
    </li>`;
  }).join('');

  ul.querySelectorAll('li').forEach((li) => {
    const id = li.dataset.id;
    li.querySelector('.box').onclick = () => toggleLayer(id);
    li.querySelector('.del').onclick = () => deleteLayer(id);
  });
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function toggleLayer(id) {
  const layer = state.project.layers.find((l) => l.id === id);
  state.project = await api(`/api/projects/${state.project.id}/layers/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: !layer.enabled }),
  });
  renderLayers();
  await refreshCurrent();
  if (state.source === 'current') setSource('current');
}

async function deleteLayer(id) {
  state.project = await api(`/api/projects/${state.project.id}/layers/${id}`, { method: 'DELETE' });
  renderLayers();
  await refreshCurrent();
  if (state.source === 'current') setSource('current');
}

/* ---------------- separation ---------------- */

function showStatus(head, msg, sub = '', pct = null, loud = false) {
  $('status').hidden = false;
  $('status').classList.toggle('loud', loud);
  $('status-head').textContent = head;
  $('status-msg').textContent = msg;
  $('status-sub').textContent = sub;
  $('status-bar').parentElement.style.display = pct === null ? 'none' : '';
  if (pct !== null) $('status-bar').style.width = `${Math.round(pct * 100)}%`;
}
const hideStatus = () => { $('status').hidden = true; };
const hideResult = () => {
  $('result').hidden = true;
  $('chip-target').disabled = true;
};

function setBusy(busy) {
  state.busy = busy;
  $('run').disabled = busy;
  $('run').textContent = busy ? 'WORKING' : 'PREVIEW';
}

$('run').onclick = async () => {
  if (state.busy || !state.project) return;
  const description = $('desc').value.trim();
  const anchors = state.sel ? [['+', +state.sel.from.toFixed(3), +state.sel.to.toFixed(3)]] : [];
  if (!description && !anchors.length) {
    showStatus('NOTHING TO DO', 'Describe the sound, mark it on the timeline, or both.');
    return;
  }
  hideResult();
  setBusy(true);
  showStatus('WORKING', 'Sending to the GPU…', '', 0);

  try {
    const job = await api(`/api/projects/${state.project.id}/separate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description, anchors }),
    });
    await poll(job.id, description);
  } catch (err) {
    showStatus('STOPPED', err.message, 'Nothing was changed.', null, true);
    setBusy(false);
  }
};

async function poll(jobId, description) {
  const started = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch (err) { showStatus('STOPPED', err.message, '', null, true); setBusy(false); return; }

    if (job.status === 'running' || job.status === 'queued') {
      showStatus('WORKING', 'Separating…', `${Math.round((Date.now() - started) / 1000)}s ELAPSED`, job.progress);
      continue;
    }
    setBusy(false);

    if (job.status === 'error') {
      showStatus('STOPPED', job.error, 'Nothing was changed.', null, true);
      return;
    }

    hideStatus();
    state.job = job;
    const secs = ((Date.now() - started) / 1000).toFixed(1);

    if (!job.found) {
      $('result').hidden = false;
      $('result-head').textContent = 'NOTHING FOUND';
      $('result-head').classList.add('quiet');
      $('result-title').textContent = `No “${description || 'sound'}” in that span.`;
      $('result-sub').textContent = 'Widen the marks, or describe it the way it sounds.';
      $('keep').hidden = true;
      $('discard').textContent = 'DISMISS';
      return;
    }

    $('result').hidden = false;
    $('result-head').textContent = 'FOUND IT';
    $('result-head').classList.remove('quiet');
    $('result-title').textContent = `“${description || 'marked span'}”`;
    $('result-sub').textContent =
      `PEAK ${job.target_peak}  ·  ${secs}s ON ${document.body.dataset.engine || 'GPU'}  ·  PREVIEWING`;
    $('keep').hidden = false;
    $('discard').textContent = 'DISCARD';
    $('chip-target').disabled = false;
    setSource('current');   // now points at the job's residual — the preview
    return;
  }
}

$('discard').onclick = () => {
  state.job = null;
  hideResult();
  setSource('current');
};

$('keep').onclick = async () => {
  if (!state.job) return;
  const jobId = state.job.id;
  state.job = null;
  hideResult();
  showStatus('WORKING', 'Writing it into the stack…');
  try {
    state.project = await api(`/api/projects/${state.project.id}/layers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
    });
    hideStatus();
    $('desc').value = '';
    state.sel = null;
    paintSelection();
    renderLayers();
    await refreshCurrent();
    setSource('current');
  } catch (err) {
    showStatus('STOPPED', err.message, '', null, true);
  }
};

$('desc').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('run').click(); });

/* ---------------- upload ---------------- */

async function upload(file) {
  const body = new FormData();
  body.append('file', file);
  $('drop').classList.remove('over');
  $('drop').querySelector('.drop-title').textContent = 'READING…';
  try {
    const data = await api('/api/projects', { method: 'POST', body });
    await openProject(data);
  } catch (err) {
    $('drop').querySelector('.drop-title').textContent = 'COULD NOT READ THAT';
    $('drop').querySelector('.drop-formats').textContent = err.message.toUpperCase();
  }
}

$('browse').onclick = () => $('file').click();
$('file').onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); };

const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});

/* ---------------- export ---------------- */

document.querySelectorAll('#formats .chip').forEach((chip) => {
  chip.onclick = () => {
    state.format = chip.dataset.fmt;
    document.querySelectorAll('#formats .chip').forEach((c) => c.classList.toggle('on', c === chip));
  };
});

$('export').onclick = () => {
  if (!state.project) return;
  const url = `/api/projects/${state.project.id}/export?format=${state.format}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  a.click();
  $('saved').hidden = false;
  const active = state.project.layers.filter((l) => l.enabled).length;
  $('saved-msg').textContent =
    `${active} sound${active === 1 ? '' : 's'} removed — written beside the original.`;
};

$('close').onclick = () => {
  state.audio.pause();
  state.project = null;
  $('editor').hidden = true;
  $('empty').hidden = false;
  $('drop').querySelector('.drop-title').textContent = 'DROP A RECORDING';
  loadRecent();
};

/* ---------------- boot ---------------- */

async function loadRecent() {
  try {
    const list = await api('/api/projects');
    $('recent-wrap').hidden = !list.length;
    $('recent').innerHTML = list.slice(0, 6).map((p, i) => `
      <li data-id="${p.id}">
        <span class="n">${String(i + 1).padStart(2, '0')}</span>
        <span>${escapeHtml(p.name)}</span>
        <span class="leader"></span>
        <span class="meta">${tc(p.duration, 0)}  ·  ${p.layers} REMOVED</span>
      </li>`).join('');
    $('recent').querySelectorAll('li').forEach((li) => {
      li.onclick = async () => openProject(await api(`/api/projects/${li.dataset.id}`));
    });
  } catch { /* first run, nothing stored yet */ }
}

async function boot() {
  try {
    const cfg = await api('/api/config');
    document.body.dataset.engine = cfg.engine === 'local' ? 'CPU' : 'GPU';
    const model = cfg.model.split('/').pop().toUpperCase();
    $('engine').textContent = `ENGINE ${cfg.engine.toUpperCase()}  ·  ${model}`;
    if (!cfg.configured) {
      $('engine').textContent = 'NO ENGINE CONFIGURED — SET AUDBRE_MODAL_URL';
      $('engine').classList.add('bad');
    }
  } catch {
    $('engine').textContent = 'SERVER UNREACHABLE';
    $('engine').classList.add('bad');
  }
  loadRecent();
}

window.addEventListener('resize', draw);
boot();
