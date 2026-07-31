/* glasshouse — the browser half.
 *
 * No framework and no build step. The whole interface is a projection of one
 * event stream, so the code is organised the same way: `state` holds what the
 * server has told us so far, each event folds into it, and `render*` functions
 * draw from it. Nothing draws from an event directly, which is why a late
 * `verdicts` event can repaint sentences that were rendered plain seconds
 * earlier without any special-casing.
 */

'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  question: '',
  claims: [],       // {text, verdict, note, support, noise_floor, memory}
  chunks: [],       // retrieved, in rank order
  runs: [],
  projection: null,
  summary: null,
  selected: null,   // index of the claim being inspected
  busy: false,
};

/* ------------------------------------------------------------------ startup */

async function boot() {
  try {
    const corpus = await (await fetch('/api/corpus')).json();
    renderCorpus(corpus);
  } catch (err) {
    notice('Could not reach the server.', true);
  }
  $('asker').addEventListener('submit', onSubmit);
}

function renderCorpus(corpus) {
  const chip = $('corpus-chip');
  const bits = [
    `<b>${escape(corpus.title)}</b>`,
    `${corpus.documents.length} docs`,
    `${corpus.chunks} chunks`,
    `top-${corpus.top_k}`,
    `${escape(corpus.model)}`,
  ];
  if (corpus.recorded) bits.push('<b>recorded demo</b>');
  chip.innerHTML = bits.join('<span>·</span>');
  chip.hidden = false;

  if (corpus.questions && corpus.questions.length) {
    const box = $('suggestions');
    box.innerHTML = '';
    for (const q of corpus.questions) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = q;
      button.addEventListener('click', () => {
        $('question').value = q;
        $('asker').requestSubmit();
      });
      box.appendChild(button);
    }
    box.hidden = false;
  }

  if (corpus.recorded) {
    notice(
      'This is a recorded run: the answers and vectors below are real output ' +
      'from a live model, replayed so the demo needs no API key. Only the ' +
      'questions above are recorded.'
    );
  }
}

/* -------------------------------------------------------------------- asking */

function onSubmit(event) {
  event.preventDefault();
  if (state.busy) return;
  const question = $('question').value.trim();
  if (!question) return;
  ask(question);
}

function ask(question) {
  reset(question);
  setBusy(true);

  const source = new EventSource('/api/ask?q=' + encodeURIComponent(question));

  source.addEventListener('retrieved', (e) => {
    const data = JSON.parse(e.data);
    state.chunks = data.chunks;
    state.projection = data.projection;
    renderSources();
    renderMatrix();
    drawSpace();
    if (!state.chunks.length) {
      notice('Nothing in the corpus matched this question closely enough to retrieve.');
    }
  });

  source.addEventListener('answer', (e) => {
    const data = JSON.parse(e.data);
    state.claims = data.claims.map((text, i) => ({
      text,
      verdict: data.checkable[i] ? null : 'no_claim',
      support: [],
    }));
    renderAnswer();
    renderMatrix();
  });

  source.addEventListener('run', (e) => {
    state.runs.push(JSON.parse(e.data));
    renderMeters();
  });

  source.addEventListener('verdicts', (e) => {
    const data = JSON.parse(e.data);
    state.claims = data.claims;
    renderAnswer();
    renderSources();
    renderMatrix();
  });

  source.addEventListener('report', (e) => {
    const data = JSON.parse(e.data);
    state.summary = data.summary;
    state.claims = data.claims;
    state.chunks = data.retrieved;
    renderAnswer();
    renderSources();
    renderMatrix();
    renderMeters();
    if (state.summary.checkable && !state.summary.corpus_contributed) {
      notice(
        'No claim in this answer depended on your documents — withholding ' +
        'every chunk in turn changed nothing. Retrieval found no evidence the ' +
        'model actually used.'
      );
    } else if (state.summary.truncated) {
      notice('The run budget was reached, so some claims are unresolved.');
    }
  });

  source.addEventListener('error', (e) => {
    // A server-sent `error` event carries data; a transport failure does not.
    if (e.data) {
      notice(JSON.parse(e.data).message, true);
    } else if (state.busy) {
      notice('The connection dropped before the analysis finished.', true);
    }
    source.close();
    setBusy(false);
  });

  // Closing on the pipeline's own `done` would be a race: it fires before the
  // report is serialised. `complete` is sent last, after everything.
  source.addEventListener('complete', () => {
    source.close();
    setBusy(false);
  });
}

function reset(question) {
  state.question = question;
  state.claims = [];
  state.chunks = [];
  state.runs = [];
  state.projection = null;
  state.summary = null;
  state.selected = null;

  $('stage').hidden = false;
  $('meters').hidden = false;
  $('notice').hidden = true;
  $('verdict-detail').hidden = true;
  $('answer').innerHTML = '<span class="pending">retrieving evidence…</span>';
  $('sources').innerHTML = '';
  $('matrix').innerHTML = '';
  renderMeters();
  drawSpace();
}

function setBusy(busy) {
  state.busy = busy;
  $('submit').disabled = busy;
  $('question').disabled = busy;
}

function notice(message, isError) {
  const el = $('notice');
  el.textContent = message;
  el.className = 'notice' + (isError ? ' error' : '');
  el.hidden = false;
}

/* ------------------------------------------------------------------ answer */

function renderAnswer() {
  const box = $('answer');
  if (!state.claims.length) {
    box.innerHTML = '<span class="pending">generating…</span>';
    return;
  }

  box.innerHTML = '';
  state.claims.forEach((claim, i) => {
    const span = document.createElement('span');
    span.className = 'claim' + (claim.verdict ? '' : ' unjudged');
    span.dataset.verdict = claim.verdict || '';
    span.dataset.index = i;
    span.textContent = claim.text;
    span.title = claim.verdict ? claim.note : 'measuring…';

    if (claim.verdict && claim.verdict !== 'no_claim') {
      span.tabIndex = 0;
      span.setAttribute('role', 'button');
      span.setAttribute('aria-pressed', state.selected === i ? 'true' : 'false');
      span.setAttribute('aria-label', `Inspect sentence ${i + 1}: ${claim.verdict}`);
      span.addEventListener('click', () => select(i));
      span.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select(i);
        }
      });
      span.addEventListener('mouseenter', () => highlight(i));
      span.addEventListener('focus', () => highlight(i));
      span.addEventListener('mouseleave', () => highlight(state.selected));
      span.addEventListener('blur', () => highlight(state.selected));
    }
    box.appendChild(span);
    box.appendChild(document.createTextNode(' '));
  });
}

function select(index) {
  state.selected = state.selected === index ? null : index;
  renderDetail();
  highlight(state.selected);
  document.querySelectorAll('.claim').forEach((el) => {
    const active = Number(el.dataset.index) === state.selected;
    el.classList.toggle('active', active);
    if (el.hasAttribute('aria-pressed')) {
      el.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
  });
}

function renderDetail() {
  const box = $('verdict-detail');
  if (state.selected === null) { box.hidden = true; return; }

  const claim = state.claims[state.selected];
  const labels = {
    grounded: 'grounded in your documents',
    model_memory: 'model memory, not your documents',
    unsupported: 'not attributable',
    undetermined: 'unresolved',
  };
  box.innerHTML =
    `<div class="headline" style="color: var(--${cssVerdict(claim.verdict)})">` +
      escape(labels[claim.verdict] || claim.verdict) +
    `</div>` +
    `<div>${escape(claim.note)}</div>` +
    `<div class="numbers">` +
      `<span>noise floor ${fmt(claim.noise_floor)}</span>` +
      `<span>closed-book match ${fmt(claim.memory)}</span>` +
      (claim.support.length
        ? `<span>strongest effect ${fmt(claim.support[0].effect)}</span>`
        : '') +
    `</div>`;
  box.hidden = false;
}

/* ----------------------------------------------------------------- sources */

function renderSources() {
  const list = $('sources');
  list.innerHTML = '';

  $('retrieval-hint').textContent = state.chunks.length
    ? `${state.chunks.length} chunks`
    : '';

  state.chunks.forEach((chunk) => {
    const li = document.createElement('li');
    li.className = 'source';
    li.dataset.chunk = chunk.chunk_id;

    const found = [];
    if (chunk.lexical_rank !== null) found.push('keyword');
    if (chunk.dense_rank !== null) found.push('vector');

    li.innerHTML =
      `<div class="source-head">` +
        `<span class="source-title">${escape(chunk.doc_title)}</span>` +
        `<span class="source-id">${escape(chunk.chunk_id)}</span>` +
      `</div>` +
      `<div class="source-text">${escape(trim(chunk.text, 240))}</div>` +
      `<div class="source-meta">` +
        found.map((f) => `<span class="tag">${f}</span>`).join('') +
        `<span class="bar"><span data-fill="${chunk.chunk_id}"></span></span>` +
      `</div>`;
    list.appendChild(li);
  });
}

/* Light the chunks that supported one sentence, dim the rest. */
function highlight(index) {
  const effects = new Map();
  if (index !== null && state.claims[index]) {
    for (const s of state.claims[index].support || []) {
      effects.set(s.chunk_id, s);
    }
  }

  document.querySelectorAll('.source').forEach((el) => {
    const support = effects.get(el.dataset.chunk);
    const lit = support && support.credited;
    el.classList.toggle('lit', Boolean(lit));
    el.classList.toggle('dimmed', index !== null && !lit);

    const fill = el.querySelector('[data-fill]');
    if (fill) {
      const effect = support ? support.effect : 0;
      fill.style.width = Math.min(100, effect * 160) + '%';
    }
  });

  document.querySelectorAll('.cell').forEach((el) => {
    el.classList.toggle(
      'highlight',
      index !== null && Number(el.dataset.claim) === index
    );
  });

  drawSpace(index);
}

/* ------------------------------------------------------------------ matrix */

function renderMatrix() {
  const grid = $('matrix');
  if (!state.chunks.length || !state.claims.length) { grid.innerHTML = ''; return; }

  const judged = state.claims
    .map((c, i) => ({ ...c, i }))
    .filter((c) => c.verdict !== 'no_claim');

  grid.innerHTML = '';
  grid.style.gridTemplateColumns =
    `minmax(34px, auto) repeat(${state.chunks.length}, minmax(30px, 1fr))`;

  grid.appendChild(cellDiv('corner', ''));
  state.chunks.forEach((chunk) => {
    grid.appendChild(cellDiv('col-label', shortId(chunk.chunk_id)));
  });

  judged.forEach((claim) => {
    grid.appendChild(cellDiv('row-label', 'S' + (claim.i + 1)));
    const byChunk = new Map((claim.support || []).map((s) => [s.chunk_id, s]));

    state.chunks.forEach((chunk) => {
      const support = byChunk.get(chunk.chunk_id);
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.claim = claim.i;
      cell.dataset.chunk = chunk.chunk_id;

      if (!support) {
        cell.classList.add('waiting');
      } else {
        cell.classList.add('filled');
        if (support.credited) cell.classList.add('credited');
        cell.style.background = heat(support.effect);
        cell.title =
          `S${claim.i + 1} × ${chunk.chunk_id}\n` +
          `effect ${fmt(support.effect)} (raw ${fmt(support.raw_drop)})` +
          (support.joint ? '\ncredited jointly' : '');
      }
      grid.appendChild(cell);
    });
  });
}

function cellDiv(className, text) {
  const el = document.createElement('div');
  el.className = className;
  el.textContent = text;
  return el;
}

/* Effect size to colour. Zero is the panel background rather than a colour, so
 * "no effect" reads as absence rather than as a low reading. */
function heat(effect) {
  const t = Math.max(0, Math.min(1, effect / 0.6));
  if (t < 0.02) return '#ffffff08';
  return `rgba(61, 220, 151, ${0.10 + t * 0.75})`;
}

/* --------------------------------------------------------- embedding space */

function drawSpace(highlightIndex) {
  const canvas = $('space');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  const projection = state.projection;
  if (!projection || !projection.chunks.length) {
    ctx.fillStyle = '#56687e';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('waiting for retrieval…', W / 2, H / 2);
    return;
  }

  $('variance-hint').textContent =
    `PCA · ${(projection.explained_variance * 100).toFixed(0)}% of variance`;

  const points = projection.chunks.slice();
  if (projection.query) points.push(projection.query);

  const pad = 34;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const scale = (v, lo, hi, a, b) =>
    hi - lo < 1e-9 ? (a + b) / 2 : a + ((v - lo) / (hi - lo)) * (b - a);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (p) => scale(p.x, x0, x1, pad, W - pad);
  const py = (p) => scale(p.y, y0, y1, H - pad, pad);

  const retrieved = new Set(state.chunks.map((c) => c.chunk_id));
  const effects = new Map();
  if (highlightIndex != null && state.claims[highlightIndex]) {
    for (const s of state.claims[highlightIndex].support || []) {
      effects.set(s.chunk_id, s);
    }
  }

  // Lines from the query to the chunks that supported the hovered sentence.
  if (projection.query && effects.size) {
    const q = projection.query;
    for (const chunk of projection.chunks) {
      const support = effects.get(chunk.chunk_id);
      if (!support || !support.credited) continue;
      ctx.strokeStyle = 'rgba(61, 220, 151, 0.5)';
      ctx.lineWidth = 1 + support.effect * 4;
      ctx.beginPath();
      ctx.moveTo(px(q), py(q));
      ctx.lineTo(px(chunk), py(chunk));
      ctx.stroke();
    }
  }

  for (const chunk of projection.chunks) {
    const isRetrieved = retrieved.has(chunk.chunk_id);
    const support = effects.get(chunk.chunk_id);
    const credited = support && support.credited;

    const r = credited ? 8 : isRetrieved ? 5.5 : 3;
    ctx.beginPath();
    ctx.arc(px(chunk), py(chunk), r, 0, Math.PI * 2);

    if (credited) {
      ctx.fillStyle = '#3ddc97';
      ctx.shadowColor = '#3ddc97';
      ctx.shadowBlur = 14;
    } else if (isRetrieved) {
      ctx.fillStyle = effects.size ? '#3a4a5e' : '#5ac8fa';
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = '#2a3746';
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  if (projection.query) {
    const q = projection.query;
    ctx.beginPath();
    ctx.arc(px(q), py(q), 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffc857';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#ffc857';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('question', px(q), py(q) - 12);
  }
}

/* ------------------------------------------------------------------ meters */

function renderMeters() {
  const s = state.summary;
  const runs = s ? s.runs : state.runs.length + 1;
  const tokens = s
    ? s.input_tokens + s.output_tokens
    : state.runs.reduce((n, r) => n + r.input_tokens + r.output_tokens, 0);
  const cost = s ? s.cost_usd : state.runs.reduce((n, r) => n + r.cost_usd, 0);

  $('m-runs').textContent = runs;
  $('m-tokens').textContent = tokens.toLocaleString();
  $('m-cost').textContent = s && s.cached ? 'recorded' : '$' + cost.toFixed(4);
  $('m-time').textContent = s ? s.elapsed_s.toFixed(1) + 's' : '—';
  $('m-grounded').textContent = s ? `${s.grounded}/${s.checkable}` : '–';
}

/* ------------------------------------------------------------------ helpers */

function escape(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function trim(text, limit) {
  return text.length <= limit ? text : text.slice(0, limit - 1) + '…';
}

function shortId(id) {
  const [doc, ordinal] = id.split('#');
  return (doc.length > 7 ? doc.slice(0, 6) + '…' : doc) + '#' + ordinal;
}

function fmt(value) {
  return value == null ? '—' : Number(value).toFixed(2);
}

function cssVerdict(verdict) {
  return verdict === 'model_memory' ? 'memory'
    : verdict === 'grounded' ? 'grounded'
    : verdict === 'unsupported' ? 'unsupported'
    : 'undetermined';
}

boot();
