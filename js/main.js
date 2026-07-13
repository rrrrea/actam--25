/* Codec Explorer — H4–H8
 * Architecture:
 *  - All variants (original + every bitrate) of the current sample are decoded
 *    into AudioBuffers and played SIMULTANEOUSLY through per-variant GainNodes.
 *  - "Switching bitrate" only flips gains -> perfectly time-aligned A/B.
 *  - Token grid: canvas heatmap of codes [n_q x n_frames]; playhead driven by
 *    AudioContext time via requestAnimationFrame.
 */

const BITRATES = [1.5, 3.0, 6.0, 12.0, 24.0]; // fallback; manifest may override
const NQ = { 1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32 };

const el = id => document.getElementById(id);
const tag = (stem, br) => `${stem}_${br.toFixed(1)}kbps`.replace(/\./g, "_");

const state = {
  ctx: null,
  stem: null,
  bitrates: BITRATES,
  variants: {},      // key -> {buffer, gain, source}
  tokens: {},        // key -> codes[][]
  current: "original",
  lastBitrate: null,   // last non-original key, target of spacebar A/B
  mode: "recon",       // 'recon' | 'residual' | 'contrib'
  playing: false,
  startedAt: 0,
  duration: 0,
};

/* ---------- boot ---------- */

async function boot() {
  let manifest = { samples: ["sample"], bitrates: BITRATES };
  try { manifest = await (await fetch("assets/manifest.json")).json(); } catch (e) {}
  state.bitrates = manifest.bitrates || BITRATES;

  const sel = el("sampleSelect");
  manifest.samples.forEach(s => sel.add(new Option(s, s)));
  sel.onchange = () => loadSample(sel.value);

  buildLadder();
  el("playBtn").onclick = togglePlay;
  const setMode = m => {
    state.mode = state.mode === m ? "recon" : m;
    el("residBtn").setAttribute("aria-pressed", state.mode === "residual");
    el("contribBtn").setAttribute("aria-pressed", state.mode === "contrib");
    applyGains();
    selectVariant(state.current); // refresh readout
  };
  el("residBtn").onclick = () => setMode("residual");
  el("contribBtn").onclick = () => setMode("contrib");
  window.addEventListener("keydown", e => {
    if (e.code !== "Space" || e.target.tagName === "SELECT") return;
    e.preventDefault();
    if (state.current === "original" && state.lastBitrate) selectVariant(state.lastBitrate);
    else if (state.current !== "original") selectVariant("original");
  });
  await loadSample(manifest.samples[0]);
}

function buildLadder() {
  const ladder = el("ladder");
  ladder.innerHTML = "";
  const mk = (label, key, small) => {
    const b = document.createElement("button");
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", key === state.current);
    b.innerHTML = `${label}<small>${small}</small>`;
    b.onclick = () => selectVariant(key);
    b.dataset.key = key;
    ladder.appendChild(b);
  };
  mk("orig", "original", "reference");
  state.bitrates.forEach(br =>
    mk(`${br}k`, tag("", br).slice(1), `${NQ[br] || "?"} layers`));
}

/* ---------- loading ---------- */

async function loadSample(stem) {
  stopPlayback();
  state.stem = stem;
  state.variants = {};
  state.tokens = {};
  el("playBtn").disabled = true;
  el("playBtn").textContent = "load\u2026";

  if (!state.ctx) state.ctx = new (window.AudioContext || window.webkitAudioContext)();

  const jobs = [fetchBuffer(`assets/audio/${stem}_original.wav`, "original")];
  for (const br of state.bitrates) {
    const key = tag(stem, br).slice(stem.length + 1); // e.g. "1_5kbps"
    jobs.push(fetchBuffer(`assets/audio/${tag(stem, br)}.wav`, key));
    jobs.push(fetchBuffer(`assets/audio/${tag(stem, br)}_residual.wav`, key + "_res").catch(() => {}));
    jobs.push(fetchBuffer(`assets/audio/${tag(stem, br)}_contrib.wav`, key + "_ctb").catch(() => {}));
    jobs.push(
      fetch(`assets/tokens/${tag(stem, br)}.json`)
        .then(r => r.json())
        .then(j => { state.tokens[key] = j; })
        .catch(() => {})
    );
  }
  await Promise.all(jobs);

  state.duration = state.variants.original.buffer.duration;
  el("axisEnd").textContent = `${state.duration.toFixed(1)} s`;
  el("specOriginal").src = `assets/spectrograms/${stem}_original.png`;
  selectVariant(state.current === "original" ? tag(stem, state.bitrates[0]).slice(stem.length + 1) : state.current);
  el("playBtn").disabled = false;
  el("playBtn").textContent = "play";
}

async function fetchBuffer(url, key) {
  const buf = await (await fetch(url)).arrayBuffer();
  const audio = await state.ctx.decodeAudioData(buf);
  state.variants[key] = { buffer: audio, gain: null, source: null };
}

/* ---------- playback: simultaneous sources, gain switching ---------- */

function audibleKey() {
  if (state.current !== "original") {
    if (state.mode === "residual" && state.variants[state.current + "_res"]) return state.current + "_res";
    if (state.mode === "contrib" && state.variants[state.current + "_ctb"]) return state.current + "_ctb";
  }
  return state.current;
}

function applyGains() {
  if (!state.playing) return;
  const now = state.ctx.currentTime, target = audibleKey();
  for (const [k, v] of Object.entries(state.variants)) {
    if (!v.gain) continue;
    v.gain.gain.setTargetAtTime(k === target ? 1 : 0, now, 0.008); // click-free
  }
}

function startPlayback() {
  const ctx = state.ctx;
  if (ctx.state === "suspended") ctx.resume();
  const t0 = ctx.currentTime + 0.05;
  const target = audibleKey();
  for (const [key, v] of Object.entries(state.variants)) {
    const src = ctx.createBufferSource();
    src.buffer = v.buffer;
    const g = ctx.createGain();
    g.gain.value = key === target ? 1 : 0;
    src.connect(g).connect(ctx.destination);
    src.start(t0);
    v.source = src; v.gain = g;
  }
  state.variants.original.source.onended = stopPlayback;
  state.startedAt = t0;
  state.playing = true;
  el("playBtn").textContent = "stop";
  requestAnimationFrame(drawPlayhead);
}

function stopPlayback() {
  if (!state.playing) return;
  state.playing = false;
  for (const v of Object.values(state.variants)) {
    try { v.source && (v.source.onended = null, v.source.stop()); } catch (e) {}
    v.source = null; v.gain = null;
  }
  el("playBtn").textContent = "play";
  drawTokens(); // clear playhead
}

function togglePlay() { state.playing ? stopPlayback() : startPlayback(); }

function selectVariant(key) {
  state.current = key;
  if (key !== "original") state.lastBitrate = key;
  document.querySelectorAll(".ladder button").forEach(b =>
    b.setAttribute("aria-checked", b.dataset.key === key));
  applyGains();
  // readout + spectrogram + tokens
  if (key === "original") {
    el("readoutKbps").textContent = "reference";
    el("readoutNq").textContent = "no quantization";
    el("specRecon").src = el("specOriginal").src || "";
    el("specReconCap").textContent = "original (reference)";
    el("tokensNote").textContent = "select a bitrate to see its discrete representation";
  } else {
    const j = state.tokens[key];
    const res = state.mode === "residual" ? " \u00b7 residual" : state.mode === "contrib" ? " \u00b7 layer contribution" : "";
    el("readoutKbps").textContent = (j ? `${j.kbps} kbps` : key) + res;
    el("readoutNq").textContent = j ? `${j.n_q} codebook layers \u00d7 ${j.n_frames} frames` : "";
    el("specRecon").src = `assets/spectrograms/${state.stem}_${key}.png`;
    el("specReconCap").textContent = j ? `reconstruction \u00b7 ${j.kbps} kbps` : "reconstruction";
    el("tokensNote").textContent = j
      ? `${j.n_q * j.n_frames} integers are the entire signal at this bitrate`
      : "";
  }
  drawTokens();
}

/* ---------- token grid ---------- */

function inferno(v) { // v in [0,1] -> css color, spectrogram-like ramp
  const stops = [[13,8,58],[84,15,109],[156,36,97],[217,80,57],[250,153,42],[252,244,163]];
  const x = Math.min(0.999, Math.max(0, v)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(",")})`;
}

function drawTokens() {
  const cv = el("tokenCanvas");
  const ctx2 = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = 220 * devicePixelRatio;
  ctx2.clearRect(0, 0, W, H);

  const j = state.tokens[state.current];
  if (!j) {
    ctx2.fillStyle = "#6d7169";
    ctx2.font = `${12 * devicePixelRatio}px IBM Plex Mono, monospace`;
    ctx2.fillText("original — continuous signal, no tokens", 12 * devicePixelRatio, 24 * devicePixelRatio);
    return;
  }
  const { codes, n_q, n_frames } = { codes: j.codes, n_q: j.n_q, n_frames: j.n_frames };
  const cw = W / n_frames, ch = H / n_q;
  for (let q = 0; q < n_q; q++) {
    for (let t = 0; t < n_frames; t++) {
      ctx2.fillStyle = inferno(codes[q][t] / 1024);
      ctx2.fillRect(t * cw, q * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}

function drawPlayhead() {
  if (!state.playing) return;
  drawTokens();
  const cv = el("tokenCanvas");
  const ctx2 = cv.getContext("2d");
  const elapsed = state.ctx.currentTime - state.startedAt;
  const x = Math.max(0, Math.min(1, elapsed / state.duration)) * cv.width;
  ctx2.strokeStyle = "#d5451b";
  ctx2.lineWidth = 2 * devicePixelRatio;
  ctx2.beginPath(); ctx2.moveTo(x, 0); ctx2.lineTo(x, cv.height); ctx2.stroke();
  requestAnimationFrame(drawPlayhead);
}

window.addEventListener("resize", drawTokens);

// hover readout: which integer is under the cursor
el("tokenCanvas").addEventListener("mousemove", e => {
  const j = state.tokens[state.current];
  if (!j) return;
  const r = e.target.getBoundingClientRect();
  const t = Math.min(j.n_frames - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * j.n_frames)));
  const q = Math.min(j.n_q - 1, Math.max(0, Math.floor((e.clientY - r.top) / r.height * j.n_q)));
  el("tokensNote").textContent =
    `layer ${q + 1}/${j.n_q} \u00b7 frame ${t} (${(t / 75).toFixed(2)} s) \u00b7 token #${j.codes[q][t]}`;
});
el("tokenCanvas").addEventListener("mouseleave", () => selectVariant(state.current));

boot();
