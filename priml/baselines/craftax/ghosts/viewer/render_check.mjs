// Load a packed ghost-run site in headless Chrome and check that it renders.
//
//   node render_check.mjs [--chrome PATH] [--only CASES] OUT_DIR SITE_DIR
//
// --only runs just the named cases, comma-separated, each on a fresh page:
// timing, pages (load, base maps, pickers and every tier, set and count),
// phone, oneRow, reward, twoRows, follow, wins, loop, items, projectiles, dark,
// replay, shade, sleep, embed,
// fit (the fitted figure: its geometry, resizing, the plot scrubber and
// restart), autoplay (the fitted figure's autoplay and the reader's pause),
// outlines (its markers' outlines by kind).
//
// Serves SITE_DIR (pack.mjs's output) over local HTTP as the Artifact host
// would: index.html wrapped in its document skeleton, and only the file types
// it serves.
//
// First, on a fresh page, it plays every tier's sets at their largest count,
// skipping quiet stretches and in real time at 3,000 decisions/s and at 30
// and 300 in real time, and records the page's work per frame, the browser's
// frame intervals and seek times (timingRuns). While playing it checks that
// the heat on screen is the heat at the decision on screen and covers every
// living player ghost's tile (HEAT_STATE). It runs before any check reads
// pixels back: Chrome moves a canvas read back often to the CPU, which would
// slow every later frame.
//
// Then, on a fresh page: the default tier reports "Loaded"; every floor's base
// map draws many colours; clicking a tier, set and count loads them; for every
// tier, set and count, the page loads it and, at the first, middle and last
// position, its ghost and heat layers draw; a 390 px phone width has one
// column and no horizontal scroll (on a fresh page); "One row" lays the nine
// boards edge to edge across 1,440, 1,920 and 390 px windows and survives a
// reload (rowLayout); "Two rows" lays nine boards and the win tally as ten
// equal squares, five to a row, across the same widths, and the tally counts
// the wins the timeline and readout count (twoRows); the reward plot draws
// and its cursor follows the seek bar (rewardPlot); the wins view follows its
// pinned run unbroken, with the policy view on that run's decision
// (winsView). Screenshots each set's largest count, the phone view (in the
// light theme), One row at 1,440 and 1,920 px, Two rows on the last tier's
// runs mid-playback and on the wins view early, midway and at the end, and
// the timing run. "Follow" shows three equal squares at the same widths: a
// window of three floors that keeps the followed run in view and moves only
// when it must; in playback it slides as a conveyor when the run goes below
// it (mid-move four boards show, the leaving one partly out of the strip, and
// it comes to rest on the new floors), stays put when the run climbs within
// it, and slides back when a run climbs above it; then all of it again with
// two boards, as the blog's embed shows them (follow).
//
// Last, the embed (embedCase): a page holding only an embedded figure of the
// last tier's short games at their largest count, two screens down, with the
// placeholder policy view (policy_stub.js). It must not load until scrolled
// near, play once its boards show, follow the shortest win, keep the panel
// on the decision the ghosts show, pause when scrolled away, stay still for a
// reader who prefers reduced motion, and fit a 390 px phone as 3 x 3 boards.
// Screenshots it mid-playback and on the phone.
//
// Writes OUT_DIR/render-check.json; exits 1 on any failed check, console
// error, uncaught exception or crash.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
// The types the Artifact host serves for this site's files; it refuses any other.
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.txt': 'text/plain' };
const SKELETON = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">',
  '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#fafaf7}img{max-width:100%}[hidden]{display:none!important}</style></head><body>',
  '</body></html>'];

// The embed's test page: the figure two screens down, between spacers.
function embedPage(manifest) {
  const tier = manifest.tiers.at(-1), set = tier.sets.find(s => s.name === 'short') ?? tier.sets[0];
  const floors = ['#4f9a3d', '#8b7b67', '#b48b3b', '#3c8e86', '#c9a227', '#7f5b9c', '#d5582b', '#5aa6d5', '#6f7192'].map((c, f) => `--floor-${f}:${c}`).join(';');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Embed check</title>
<style>:root{--surface:#14171b;--ink:#f5f5f1;--ink-3:#bec2c7;--accent:#3fb89c;--rule:#20242a;--rule-strong:#2c3137;--death:#ff5a4c;${floors}}
body{margin:0;background:#0a0c0e;color:#f5f5f1;font:15px/1.5 system-ui,sans-serif}.spacer{height:250vh;padding:16px}figure{margin:0 16px}</style></head><body>
<div class="spacer">The figure is two screens down.</div>
<figure><div data-ghost-embed data-data-base="./" data-transport="b64" data-tier="${tier.name}" data-set="${set.name}" data-count="${set.counts.at(-1)}"></div><div data-ghost-policy></div></figure>
<div class="spacer"></div>
<script src="decode.js"></script><script src="__policy_stub.js"></script><script src="ghosts.js"></script></body></html>`;
}

// The fitted figure's test page, as the post sets it: a sticky 60 px header,
// a text column 640 px wide at most with a screen of intro (the figure is
// read once scrolled under the header), then in that column
// the figure, its card bleeding --bleed (12 px, 8 px on phones) past the
// column on each side with its content back in the column, with its title,
// the policy view (the site's own bundle and script, when it has them) and
// the embed with the controls on top, fitted to its column, scrubbed by its
// plot, following the pinned run of the last tier's unbroken (else wins) set
// in the Follow layout with `boards` boards (?boards=3, else 2, as the blog).
function embedFitPage(manifest, root, boards) {
  const tier = manifest.tiers.find(t => t.sets.some(s => s.time_map?.unbroken != null)) ?? manifest.tiers.at(-1);
  const set = tier.sets.find(s => s.name === 'unbroken') ?? tier.sets.find(s => s.time_map?.unbroken != null) ?? tier.sets[0];
  const panel = fs.existsSync(path.join(root, 'policy-view.json')) && fs.existsSync(path.join(root, 'policy_view.js'));
  const floors = ['#4f9a3d', '#8b7b67', '#b48b3b', '#3c8e86', '#c9a227', '#7f5b9c', '#d5582b', '#5aa6d5', '#6f7192'].map((c, f) => `--floor-${f}:${c}`).join(';');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fitted embed check</title>
<style>:root{--surface:#ffffff;--ink:#0b0d10;--ink-3:#4a5058;--accent:#1d8a73;--rule:#e4e6e2;--rule-strong:#cfd2cc;--death:#d93a30;--bg:#fafaf7;${floors}}
body{margin:0;background:#fafaf7;color:#0b0d10;font:15px/1.5 system-ui,sans-serif}nav{position:sticky;top:0;z-index:50;height:60px;background:#fafaf7;border-bottom:1px solid #e4e6e2}
main{max-width:640px;margin:0 auto;padding:0 16px}.intro{margin:0;height:100vh}figure{--bleed:12px;margin:0 calc(-1 * var(--bleed));padding:12px calc(var(--bleed) - 1px);border:1px solid #e4e6e2;border-radius:10px;background:#f4f4f0}@media (max-width:600px){figure{--bleed:8px}}figure h3{margin:0 0 8px;text-align:center}figure > [data-ghost-policy]{display:none}</style></head><body>
<nav>Header</nav><main><p class="intro">Intro.</p>
<figure id="fig"><h3>Winning games in one world</h3>${panel ? '<div data-ghost-policy data-bundle="policy-view.bin.gz.b64.txt" data-layout="side"></div>' : ''}
<div data-ghost-embed data-data-base="./" data-transport="b64" data-tier="${tier.name}" data-set="${set.name}" data-count="${set.counts.at(-1)}" data-follow="pinned" data-layout="follow" data-follow-boards="${boards}" data-controls="top" data-fit="column" data-scrubber="plot" data-speeds="10,50,100,250" data-speed="50"></div></figure>
<p class="intro"></p></main>
<script src="decode.js"></script>${panel ? '<script src="policy_view.js"></script>' : ''}<script src="ghosts.js"></script></body></html>`;
}

function serve(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8')), here = path.dirname(new URL(import.meta.url).pathname);
  const server = http.createServer((request, response) => {
    const name = decodeURIComponent(new URL(request.url, 'http://x').pathname).replace(/^\/$/, '/index.html');
    if (name === '/__embed.html') { response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' }); response.end(embedPage(manifest)); return; }
    if (name === '/__embed-fit.html') { response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' }); response.end(embedFitPage(manifest, root, new URL(request.url, 'http://x').searchParams.get('boards') === '3' ? 3 : 2)); return; }
    const file = name === '/__policy_stub.js' ? path.join(here, 'policy_stub.js') : path.join(root, path.normalize(name));
    if (!(file.startsWith(root) || file.startsWith(here)) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { response.writeHead(404); response.end(); return; }
    let body = fs.readFileSync(file);
    if (name === '/index.html') body = Buffer.from(SKELETON[0] + SKELETON[1] + body.toString('utf8') + SKELETON[2]);
    if (!TYPES[path.extname(file)]) { response.writeHead(415); response.end(); return; }
    const headers = { 'Content-Type': TYPES[path.extname(file)], 'Cache-Control': 'no-store' };
    response.writeHead(200, headers);
    response.end(body);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function launch(binary) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'ghosts-check-'));
  // Port 0, read back from the profile's DevToolsActivePort: a random port can be
  // taken. No deadline of our own: a live Chrome that has not opened DevTools yet
  // is slow, not broken (a cold first launch took up to 35 s on a CI runner),
  // and one that exited fails at once.
  const proc = spawn(binary, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--window-size=1440,1000', 'about:blank'], { stdio: 'ignore' });
  let exited = null;
  proc.on('exit', (code, signal) => { exited = signal ?? code; });
  proc.on('error', error => { exited = error.message; });
  let target;
  while (!target) {
    if (exited !== null) throw Error(`Chrome exited (${exited}) before opening a debugging target.`);
    await sleep(200);
    try {
      const port = Number(fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]);
      if (port) target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page');
    } catch { /* not up */ }
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const pending = new Map(), errors = [];
  let id = 0;
  ws.onmessage = event => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
    else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
    else if (m.method === 'Inspector.targetCrashed') errors.push('The page crashed.');
  };
  // A command unanswered in 5 minutes fails the check instead of stalling it.
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const n = ++id, timer = setTimeout(() => { pending.delete(n); reject(Error(`${method} got no answer in 300 s.`)); }, 300_000);
    pending.set(n, m => { clearTimeout(timer); if (m.error) reject(Error(`${method}: ${m.error.message}`)); else resolve(m.result); });
    ws.send(JSON.stringify({ id: n, method, params }));
  });
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw Error(`${expression.slice(0, 80)}: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
    return r.result.value;
  };
  const close = async () => { ws.close(); proc.kill(); await sleep(500); fs.rmSync(profile, { recursive: true, force: true }); };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Inspector.enable');
  return { send, evaluate, close, errors };
}

async function until(chrome, expression, seconds) {
  for (const t0 = Date.now(); Date.now() - t0 < seconds * 1000; await sleep(50)) {
    if (await chrome.evaluate(expression)) return true;
  }
  return false;
}

// Distinct colours on each base canvas, and painted pixels on each layer.
const PIXELS = `(() => {
  const count = (canvas, colours) => { const d = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    const seen = new Set(); let painted = 0;
    for (let i = 0; i < d.length; i += 16) { if (colours) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]); else if (d[i + 3]) painted++; }
    return colours ? seen.size : painted; };
  const all = name => [...document.querySelectorAll('canvas.' + name)];
  return { base: all('base').map(c => count(c, true)), layer: [...all('layer'), ...all('heat')].map(c => count(c, false)), ghosts: all('ghosts').map(c => count(c, false)) };
})()`;
const LOADED = `(tier, set, count) => document.getElementById('status').textContent.startsWith('Loaded') && ghosts.tier.name === tier && ghosts.collection.id === set && ghosts.count === count`;
const click = (container, key) => `document.querySelector('#${container} [data-key="${key}"]').click()`;
const WINDOWS = `[...ghosts.windows.values()].every(w => w.index)`;

async function seekTo(chrome, fraction) {
  await chrome.evaluate(`(() => { const s = document.getElementById('seek'); s.value = String(Math.round(Number(s.max) * ${fraction}));
    s.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await sleep(100);
  await until(chrome, WINDOWS, 30);
  await sleep(50);
  return chrome.evaluate(PIXELS);
}

async function screenshot(chrome, file) {
  const { cssContentSize: size } = await chrome.send('Page.getLayoutMetrics');
  const { data } = await chrome.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 } });
  fs.writeFileSync(file, Buffer.from(data, 'base64'));
}

const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(q * values.length))];
const timing = values => ({ n: values.length, mean: +(values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)).toFixed(2), p50: +quantile(values, 0.5)?.toFixed(2), p95: +quantile(values, 0.95)?.toFixed(2), max: +Math.max(...values).toFixed(2) });

// Load the page afresh; the marker keeps the previous document's status from passing.
async function open(chrome, url) {
  await chrome.evaluate('window.previous = true');
  await chrome.send('Page.navigate', { url });
  return until(chrome, `!window.previous && document.getElementById('status')?.textContent.startsWith('Loaded')`, 180);
}

async function check(chrome, url, out, failures, only) {
  const report = { url, combos: [] }, run = name => !only || only.includes(name);
  if (run('timing') && await open(chrome, url)) report.timing = await timingRuns(chrome, out, failures);
  if (run('pages')) await pages(chrome, url, out, failures, report);
  if (run('phone') && await open(chrome, url)) report.phone = await phone(chrome, out, failures);
  if (run('oneRow') && await open(chrome, url)) report.oneRow = await rowLayout(chrome, url, out, failures);
  if (run('reward') && await open(chrome, url)) report.reward = await rewardPlot(chrome, failures);
  if (run('twoRows') && await open(chrome, url)) report.twoRows = await twoRows(chrome, url, out, failures);
  if (run('follow') && await open(chrome, url)) report.follow = await followLayout(chrome, url, out, failures, 3);
  if (run('follow') && await open(chrome, url)) report.followTwo = await followLayout(chrome, url, out, failures, 2);
  if (run('wins') && await open(chrome, url)) report.wins = await winsView(chrome, out, failures);
  if (run('loop') && await open(chrome, url)) report.loop = await loopCase(chrome, failures);
  if (run('items') && await open(chrome, url)) report.items = await itemsCase(chrome, failures);
  if (run('projectiles') && await open(chrome, url)) report.projectiles = await projectileCase(chrome, failures);
  if (run('dark') && await open(chrome, url)) report.dark = await darkCase(chrome, failures);
  if (run('replay') && await open(chrome, url)) report.replay = await replayCase(chrome, failures);
  if (run('shade') && await open(chrome, url)) report.shade = await shadeCase(chrome, out, failures);
  if (run('sleep') && await open(chrome, url)) report.sleep = await sleepCase(chrome, out, failures);
  if (run('embed')) report.embed = await embedCase(chrome, `${url}__embed.html`, out, failures);
  if (run('fit')) report.fit = await fitCase(chrome, `${url}__embed-fit.html`, out, failures);
  if (run('autoplay')) report.autoplay = await autoplayCase(chrome, `${url}__embed-fit.html`, failures);
  if (run('outlines')) report.outlines = await outlineCase(chrome, `${url}__embed-fit.html`, out, failures);
  return report;
}

// The page loads, its base maps draw, its pickers load what they name, and
// every tier, set and count loads and draws.
async function pages(chrome, url, out, failures, report) {
  report.opened = await open(chrome, url);
  if (!report.opened) failures.push(`never loaded: ${await chrome.evaluate(`document.getElementById('status')?.textContent`)}`);
  if (!report.opened) return;
  const first = await chrome.evaluate(PIXELS);
  report.baseColours = first.base;
  if (first.base.some(c => c < 8)) failures.push(`a base map is blank: ${first.base}`);
  report.columnsDesktop = await chrome.evaluate(`getComputedStyle(document.getElementById('floors')).gridTemplateColumns.split(' ').length`);
  if (report.columnsDesktop !== 3) failures.push(`desktop shows ${report.columnsDesktop} floor columns, not 3`);
  // The buttons work: the first tier's last count, picked by clicks.
  const [tier0, set0, count0] = await chrome.evaluate(`(() => { const t = ghosts.manifest.tiers[0], s = ghosts.setsOf(t)[0]; return [t.name, s.id, s.counts.at(-1)]; })()`);
  await chrome.evaluate(click('tiers', tier0));
  await until(chrome, `ghosts.tier.name === ${JSON.stringify(tier0)} && !!document.querySelector('#counts [data-key="${count0}"]')`, 60);
  await chrome.evaluate(click('sets', set0));
  await chrome.evaluate(click('counts', count0));
  report.clicked = await until(chrome, `(${LOADED})(${JSON.stringify(tier0)}, ${JSON.stringify(set0)}, ${count0})`, 300);
  if (!report.clicked) failures.push(`clicking ${tier0}, ${set0} and ${count0} did not load them`);
  const combos = await chrome.evaluate(`ghosts.manifest.tiers.flatMap(t => ghosts.setsOf(t).flatMap(s => s.counts.map((c, i) => [t.name, s.id, c, i === s.counts.length - 1])))`);
  for (const [tier, set, count, last] of combos) report.combos.push(await combination(chrome, tier, set, count, last, out, failures));
}

const EMBED_STATE = `(() => { const g = window.ghosts; return { loaded: !!g?.pb, playing: !!g?.playing, t: g?.pb?.t ?? -1, selected: g?.selected ?? -1,
  followed: g?.pb && g.selected >= 0 ? g.pb.decision[g.selected] : -1,
  shortest: g?.set ? GhostDecode.shortestWin(g.set) : -2, outcome: g?.set && g.selected >= 0 ? GhostDecode.OUTCOMES[g.set.outcome[g.selected]] : null,
  panel: g?.policyShown ?? null, scrollWidth: document.documentElement.scrollWidth, innerWidth,
  columns: getComputedStyle(document.querySelector('.ghost-embed .floors')).gridTemplateColumns.split(' ').length }; })()`;

// The fitted figure's geometry at window sizes, scrolled under the header and
// held still (no conveyor move caught mid-slide; followLayout checks the
// moves), with `boards` Follow boards; every size from the column's width.
// Two boards: both squares of the agent view's height P, half the column,
// edge to edge; the timeline and the reward plot exactly as wide as the wider
// of the boards and the view, centred in the column. Three: the boards span
// the column and the reward plot, the view 1.2 boards tall unless the width
// caps it. Both: the view as wide as its cell and above the boards, the badge
// in bounds, no slider and no sideways scroll; and a change of the window's
// height alone (FIT_HEIGHTS) leaves the figure's layout and every pixel its
// canvases hold unchanged.
const FIT_SIZES = [[1440, 900, false], [1920, 1080, false], [2560, 1440, false], [1536, 864, false], [1280, 720, false], [768, 1024, true], [390, 844, true]];
const FIT_HEIGHTS = [[1440, 900, 600, false], [390, 844, 600, true]];
const FIT_ALIGN = `(() => { const fig = document.getElementById('fig'), header = document.querySelector('nav').getBoundingClientRect().bottom; window.scrollTo({ top: fig.getBoundingClientRect().top + scrollY - header, behavior: 'instant' }); })()`;
const FIT_GEOMETRY = `(() => { const fig = document.getElementById('fig'), r = el => el?.getBoundingClientRect(), edges = b => b && [+b.left.toFixed(1), +b.right.toFixed(1)];
  const boards = [...fig.querySelectorAll('.floor.in-strip .board')].map(r).filter(b => b.width).sort((a, b) => a.left - b.left), canvas = fig.querySelector('.agent canvas'), cell = r(fig.querySelector('.agent'));
  const badge = fig.querySelector('.won-badge'), side = boards[0]?.width ?? 0, strip = r(fig.querySelector('[data-g="floors"]')), stage = r(fig.querySelector('[data-g="stage"]'));
  return { innerWidth, innerHeight, count: boards.length,
    side: +side.toFixed(1), square: boards.every(b => Math.abs(b.width - b.height) < 1 && Math.abs(b.width - side) < 1), joined: boards.every((b, k) => !k || Math.abs(b.left - boards[k - 1].right) < 1),
    canvas: canvas && [+r(canvas).width.toFixed(1), +r(canvas).height.toFixed(1)], cell: cell && [+cell.width.toFixed(1), +cell.height.toFixed(1)], view: edges(canvas && r(canvas)),
    strip: boards.length ? [+boards[0].left.toFixed(1), +boards.at(-1).right.toFixed(1)] : null, reward: edges(r(fig.querySelector('[data-g="reward"]'))), timeline: edges(r(fig.querySelector('[data-g="occupancy"]'))),
    content: edges(r(fig.querySelector('[data-ghost-embed]'))), column: edges(r(document.querySelector('.intro'))), fit: ghosts.fit, overflow: document.documentElement.scrollWidth - innerWidth,
    above: canvas ? r(canvas).bottom <= boards[0].top + 1 : null, centred: Math.abs((strip.left + strip.right) / 2 - (stage.left + stage.right) / 2) < 2,
    badge: { font: parseFloat(getComputedStyle(badge).fontSize), share: +(r(badge).width / side).toFixed(3) }, slider: !!fig.querySelector('input[type=range]') }; })()`;
// The figure as a height-only resize must leave it: each element's box
// relative to the figure's, and a digest of every canvas's pixels.
const FIT_PRINT = `(async () => { const fig = document.getElementById('fig'), f = fig.getBoundingClientRect(), boxes = [];
  for (const n of fig.querySelectorAll('*')) { const b = n.getBoundingClientRect(); if (b.width) boxes.push([b.left - f.left, b.top - f.top, b.width, b.height].map(v => +v.toFixed(2)).join()); }
  const pixels = [];
  for (const c of fig.querySelectorAll('canvas')) { const data = new TextEncoder().encode(c.toDataURL()), hash = await crypto.subtle.digest('SHA-256', data); pixels.push([c.width, c.height, [...new Uint8Array(hash)].slice(0, 8).join('.')].join(':')); }
  return { size: [f.width, f.height].map(v => +v.toFixed(2)), boxes, pixels }; })()`;
async function fitSizes(chrome, url, boards, report, fail, out) {
  Object.assign(report, { sizes: [], heights: [] });
  await chrome.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await chrome.send('Page.navigate', { url });
  await until(chrome, 'document.readyState === "complete" && !!window.ghosts', 30);
  await chrome.evaluate(FIT_ALIGN);
  if (!await until(chrome, 'window.ghosts?.pb && (!document.querySelector(\'[data-ghost-policy]\') || ghosts.policyView)', 180)) { fail(`the fitted figure with ${boards} boards did not load`); return false; }
  await chrome.evaluate('(() => { ghosts.held = true; ghosts.stop(); ghosts.goTo(Math.round(ghosts.set.maxSteps / 3)); })()');
  const at = async (width, height, mobile) => {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(600);
    await chrome.evaluate(FIT_ALIGN);
    await sleep(300);
  };
  for (const [width, height, mobile] of FIT_SIZES) {
    await at(width, height, mobile);
    const g = await chrome.evaluate(FIT_GEOMETRY);
    report.sizes.push(g);
    const id = `${width}x${height} with ${boards} boards`, near = (a, b) => !!a && !!b && Math.abs(a[0] - b[0]) <= 1 && Math.abs(a[1] - b[1]) <= 1;
    const column = g.column[1] - g.column[0];
    if (g.count !== boards || !g.square || !g.joined || !g.centred || g.above === false) fail(`at ${id} the boards are not ${boards} equal squares edge to edge, centred under the agent's view: ${JSON.stringify(g)}`);
    if (!g.canvas || Math.abs(g.canvas[0] - g.cell[0]) > 1) fail(`at ${id} the agent's view is not as wide as its cell: ${JSON.stringify(g)}`);
    if (boards === 2) {
      const widest = g.view[1] - g.view[0] > g.strip[1] - g.strip[0] ? g.view : g.strip;
      if (Math.abs(g.side - Math.floor(column / 2)) > 1 || Math.abs(g.canvas[1] - g.side) > 1) fail(`at ${id} the boards and the agent's view are not half the column (${column} px) tall: ${JSON.stringify(g)}`);
      if (!near(g.timeline, widest) || !near(g.reward, widest)) fail(`at ${id} the timeline and the reward plot are not as wide as the widest element: ${JSON.stringify(g)}`);
      if (Math.abs(g.content[0] - g.column[0] - (g.column[1] - g.content[1])) > 1 || g.content[1] - g.content[0] > column + 1) fail(`at ${id} the figure is not centred in the column: ${JSON.stringify(g)}`);
    } else {
      if (!near(g.strip, g.reward) || !near(g.strip, g.column)) fail(`at ${id} the boards do not span the column and the reward plot's width: ${JSON.stringify(g)}`);
      if (Math.abs(g.canvas[1] - Math.floor(Math.min(1.2 * g.side, column * 144 / 260))) > 1.5) fail(`at ${id} the agent's view is not 1.2 boards tall: ${JSON.stringify(g)}`);
    }
    if (g.overflow > 0) fail(`at ${id} the page scrolls sideways by ${g.overflow} px`);
    if (g.badge.font < 7 || g.badge.font > 18 || (width < 600 && g.badge.share > 0.16)) fail(`at ${id} the win badge is out of bounds: ${JSON.stringify(g.badge)}`);
    if (g.slider) fail(`at ${id} the figure has a slider though its plot scrubs`);
    if (width === 1440 || width === 390) await screenshot(chrome, path.join(out, `${path.basename(out)}-fit-${boards}-${width}.png`));
  }
  for (const [width, tall, short, mobile] of FIT_HEIGHTS) {
    await at(width, tall, mobile);
    const before = await chrome.evaluate(FIT_PRINT);
    await at(width, short, mobile);
    const after = await chrome.evaluate(FIT_PRINT);
    await at(width, tall, mobile);
    const back = await chrome.evaluate(FIT_PRINT);
    report.heights.push({ width, tall, short, size: [before.size, after.size] });
    const moved = before.boxes.filter((b, k) => b !== after.boxes[k]).length + Math.abs(before.boxes.length - after.boxes.length);
    if (before.size.join() !== after.size.join() || moved || before.pixels.join() !== after.pixels.join() || after.pixels.join() !== back.pixels.join()) fail(`with ${boards} boards, resizing the window from ${width}x${tall} to ${width}x${short} changed the figure: size ${before.size} -> ${after.size}, ${moved} boxes moved, pixels ${before.pixels.join() === after.pixels.join() ? 'kept' : 'changed'}`);
  }
  return true;
}

// The fitted figure: its geometry at seven window sizes once scrolled under
// the header and across height-only resizes (fitSizes), with three boards and then two, as the blog sets
// it; with two, its bounds through a resize sequence; the plot as scrubber,
// by drag and by keys, with no slider; and restart while playing and paused.
async function fitCase(chrome, url, out, failures) {
  const report = { three: {}, two: {}, resizes: [] }, fail = message => failures.push(`fit: ${message}`);
  if (!await fitSizes(chrome, `${url}?boards=3`, 3, report.three, fail, out) || !await fitSizes(chrome, url, 2, report.two, fail, out)) return report;
  // Resizes: everything stays in the figure's box, the plots' backing stores
  // follow their size, the boards stay square, the content is centred in the
  // text column and no wider, the card (bordered all round, rounded) bleeds --bleed past it on
  // each side, and the page does not scroll sideways.
  const BOUNDS = `(() => { const fig = document.getElementById('fig'), cs = getComputedStyle(fig), box = fig.getBoundingClientRect();
    const left = box.left + parseFloat(cs.paddingLeft) + parseFloat(cs.borderLeftWidth) - 1, right = box.right - parseFloat(cs.paddingRight) - parseFloat(cs.borderRightWidth) + 1;
    const out = [...fig.querySelectorAll('*')].filter(n => { const b = n.getBoundingClientRect(); return b.width && !n.closest('.floor:not(.in-strip)') && (b.right > right || b.left < left); }).map(n => n.tagName + '.' + n.className);
    const stores = [...fig.querySelectorAll('[data-g="occupancy"], [data-g="reward"]')].map(c => [c.dataset.g, c.width, Math.round(c.clientWidth * devicePixelRatio)]).filter(([, a, b]) => Math.abs(a - b) > 1);
    const boards = [...fig.querySelectorAll('.floor.in-strip .board')].map(b => b.getBoundingClientRect()).filter(b => b.width);
    const column = document.querySelector('.intro').getBoundingClientRect(), content = fig.querySelector('[data-ghost-embed]').getBoundingClientRect(), bleed = parseFloat(cs.getPropertyValue('--bleed'));
    const borders = ['Top', 'Right', 'Bottom', 'Left'].every(e => parseFloat(cs['border' + e + 'Width']) >= 1 && cs['border' + e + 'Style'] !== 'none');
    return { innerWidth, out: out.slice(0, 5), stores, square: boards.length === ghosts.followBoards && boards.every(b => Math.abs(b.width - b.height) < 1), side: Math.round(boards[0]?.width ?? 0),
      column: Math.abs(content.left - column.left - (column.right - content.right)) <= 1 && content.width <= column.width + 1,
      card: bleed > 0 && Math.abs(column.left - box.left - bleed) <= 1 && Math.abs(box.right - column.right - bleed) <= 1 && parseFloat(cs.borderTopLeftRadius) > 0 && borders,
      bleed, overflow: document.documentElement.scrollWidth - innerWidth, width: Math.round(box.width) }; })()`;
  for (const [width, height, mobile] of [[1920, 1080, false], [1280, 720, false], [1440, 900, false], [390, 844, true], [961, 900, false], [1600, 900, false], [1920, 1080, false], [2560, 1440, false]]) {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(700);
    const b = await chrome.evaluate(BOUNDS);
    report.resizes.push({ width, ...b });
    if (b.out.length || b.stores.length || !b.square || !b.column || !b.card || b.overflow > 0) fail(`after resizing to ${width}x${height}: ${JSON.stringify(b)}`);
  }
  await chrome.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(600);
  await chrome.evaluate(FIT_ALIGN);
  await sleep(300);
  report.speeds = await chrome.evaluate(`(() => { const s = document.querySelector('[data-g="speed"]'); return { options: [...s.options].map(o => Number(o.value)), value: Number(s.value) }; })()`);
  if (report.speeds.options.join() !== '10,50,100,250' || report.speeds.value !== 50) fail(`the speed menu is not 10, 50, 100, 250 at 50: ${JSON.stringify(report.speeds)}`);
  report.scrub = await scrubCase(chrome, fail);
  report.cursors = [];
  for (const [width, height, mobile] of [[1440, 900, false], [390, 844, true]]) {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(600);
    for (const fraction of [0, 0.5, 1]) {
      await chrome.evaluate(`(() => { ghosts.stop(); ghosts.goTo(ghosts.timeline.toDecision(Math.round(ghosts.timeline.length * ${fraction}))); })()`);
      await sleep(150);
      const c = await chrome.evaluate(CURSORS);
      report.cursors.push({ width, fraction, ...c });
      for (const [name, plot] of Object.entries(c)) {
        if (plot.caret === null || plot.line === null || Math.abs(plot.caret - plot.line) > 0.5 || plot.line < plot.left - 0.5 || plot.line > plot.right + 0.5) fail(`at ${width} px, ${fraction} of the way, the ${name} play head's caret (${plot.caret}) is not centred on its line (${plot.line}) inside the plot area: ${JSON.stringify(plot)}`);
      }
    }
  }
  await chrome.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(600);
  report.restart = await restartCase(chrome, fail);
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  console.log(`fit: side/view ${[3, 2].map(n => `${n} boards ${report[n === 3 ? 'three' : 'two'].sizes.map(g => `${g.innerWidth}x${g.innerHeight} ${g.side}/${g.canvas?.[1]}`).join(', ')}`).join('; ')}; height-only resizes ${report.two.heights.map(h => `${h.width}x${h.tall}->${h.short} ${h.size[0].join('x')}->${h.size[1].join('x')}`).join(', ')}; resizes ${report.resizes.map(r => `${r.width}:${r.side}`).join(' ')}; scrub ${JSON.stringify(report.scrub)}; restart ${JSON.stringify(report.restart)}`);
  return report;
}

// The fitted figure's autoplay, at 1,440 x 900 and 390 x 844: (a) it plays
// once its stage shows, the policy view alone in view with the boards below
// the window; (b) once the reader pauses it, it stays paused at that step
// through a scroll away and back and a resize; (c) once they press play
// again, a scroll away pauses it and a scroll back plays it again.
const STAGE_STATE = `(() => { const r = el => el.getBoundingClientRect(), agent = r(document.querySelector('.agent')), floors = r(document.querySelector('[data-g="floors"]'));
  return { playing: ghosts.playing, t: ghosts.pb?.t ?? -1, agentShows: agent.top < innerHeight && agent.bottom > 0, boardsShow: floors.top < innerHeight && floors.bottom > 0 }; })()`;
async function autoplayCase(chrome, url, failures) {
  const report = [];
  for (const [width, height, mobile] of [[1440, 900, false], [390, 844, true]]) {
    const fail = message => failures.push(`autoplay at ${width}x${height}: ${message}`), state = () => chrome.evaluate(STAGE_STATE);
    const scroll = expression => chrome.evaluate(`window.scrollTo({ top: ${expression}, behavior: 'instant' })`);
    // The policy view's bottom at the window's: the boards below it, unseen.
    const toAgent = `document.querySelector('.agent').getBoundingClientRect().bottom + scrollY - innerHeight + 2`;
    const toStage = `document.querySelector('[data-g="stage"]').getBoundingClientRect().top + scrollY - 70`;
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await chrome.send('Page.navigate', { url });
    if (!await until(chrome, 'document.readyState === "complete" && !!window.ghosts?.pb && !!document.querySelector(".agent canvas")', 120)) { fail('the figure did not load'); continue; }
    const r = { width };
    r.before = await state();
    if (r.before.playing) fail('played before its stage showed');
    await scroll(toAgent);
    r.agentOnly = await state();
    r.agentPlays = await until(chrome, 'ghosts.playing && ghosts.pb.t > 0', 10);
    if (!r.agentOnly.agentShows || r.agentOnly.boardsShow) fail(`the scroll did not show the policy view alone: ${JSON.stringify(r.agentOnly)}`);
    else if (!r.agentPlays) fail(`did not play with its policy view in view: ${JSON.stringify(await state())}`);
    await scroll(toStage);
    await sleep(500);
    await chrome.evaluate(`document.querySelector('[data-g="play"]').click()`);
    await sleep(200);
    r.paused = await state();
    if (r.paused.playing) fail('the play button did not pause it');
    await scroll(0);
    await sleep(1000);
    await scroll(toStage);
    await sleep(1500);
    await chrome.send('Emulation.setDeviceMetricsOverride', { width: width - 40, height: height - 100, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(800);
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await scroll(toStage);
    await sleep(1500);
    r.stillPaused = await state();
    if (r.stillPaused.playing || r.stillPaused.t !== r.paused.t) fail(`after the reader's pause, a scroll away and back and a resize, it is ${r.stillPaused.playing ? 'playing' : 'paused'} at ${r.stillPaused.t}, not paused at ${r.paused.t}`);
    await chrome.evaluate(`document.querySelector('[data-g="play"]').click()`);
    r.replays = await until(chrome, `ghosts.playing && ghosts.pb.t > ${r.paused.t}`, 10);
    if (!r.replays) fail('the play button did not play it again');
    await scroll(0);
    r.awayPauses = await until(chrome, '!ghosts.playing', 5);
    await scroll(toStage);
    r.backPlays = await until(chrome, 'ghosts.playing', 10);
    if (!r.awayPauses || !r.backPlays) fail(`after the reader's play, scrolled away it ${r.awayPauses ? 'paused' : 'kept playing'} and back it ${r.backPlays ? 'played' : 'stayed paused'}`);
    report.push(r);
  }
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  console.log(`autoplay: ${report.map(r => `${r.width}: policy view alone ${r.agentPlays ? 'plays' : 'paused'}; reader pause held at ${r.stillPaused?.t} (${r.stillPaused?.playing ? 'playing' : 'paused'}); play again ${r.replays}, away ${r.awayPauses}, back ${r.backPlays}`).join('; ')}`);
  return report;
}

// The markers' outlines on the fitted figure's boards, at 1,440 x 900 and
// 390 x 844, with trails and end marks off so only markers draw on the ghost
// layer (but the followed run's trail), and lighting and fog off: at steps
// through the set, a followed
// player, a ghost player and an enemy each drawn two tiles or more from any
// other marker (app.marks), read along the four lines from its tile's edge
// outward, taking the least of them, and its opacity at the tile's centre.
// Prominence must rank followed player > ghost player > enemy in ring width
// (CSS pixels), ring brightness (its green, or an enemy's red, as drawn,
// times its opacity) and marker opacity; an enemy's ring pixels must be red.
// Every marker drawn at those steps must show its sprite on its disk (a
// black bat on a black disk reads as an empty disk): in its atlas cell, at
// least MARK_LEGIBLE of the sprite's opaque pixels differ from the disk's
// colour (read where the sprite is clear) by more than 48 in some channel.
const MARK_LEGIBLE = 0.25;
const OUTLINES = `(async () => {
  const g = ghosts, wait = ms => new Promise(r => setTimeout(r, ms)), found = {}, cells = new Set();
  // Lighting and fog off: a marker's own style, not its tile's shade (darkCase).
  Object.assign(g.show, { trails: false, deaths: false, light: false, fog: false }); g.marks = []; g.held = true; g.stop();
  const green = (r, gr, b) => gr > 100 && gr > r + 40 && gr > b + 30, red = (r, gr, b) => r > 150 && r > gr + 60 && r > b + 50;
  const scan = ([kind, floor, row, col], test) => {
    const canvas = g.panels[floor].ghosts.canvas, scale = 528 / canvas.clientWidth, data = canvas.getContext('2d').getImageData(0, 0, 528, 528).data;
    const x0 = col * 11 + 5.5, y0 = row * 11 + 5.5, reach = Math.ceil(4 * scale) + 2, lines = [];
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      let count = 0, bright = 0;
      for (let d = 5; d <= 5 + reach; d++) {
        const x = Math.floor(x0 + dx * d), y = Math.floor(y0 + dy * d);
        if (x < 0 || y < 0 || x >= 528 || y >= 528) continue;
        const i = (y * 528 + x) * 4, a = data[i + 3] / 255;
        if (a > 0.2 && test(data[i], data[i + 1], data[i + 2])) { count++; bright = Math.max(bright, (test === red ? data[i] : data[i + 1]) * a); }
      }
      lines.push([count / scale, bright]);
    }
    // The least of the four lines: the followed run's own trail may cross one.
    const centre = ((row * 11 + 5) * 528 + col * 11 + 5) * 4;
    return { kind, floor, row, col, scale: +scale.toFixed(2), width: +Math.min(...lines.map(l => l[0])).toFixed(2), bright: Math.round(Math.min(...lines.map(l => l[1]))), opacity: +(data[centre + 3] / 255).toFixed(2), lines };
  };
  for (let k = 1; k < 40 && Object.keys(found).length < 3; k++) {
    g.goTo(Math.round(k / 40 * g.set.maxSteps)); await wait(80);
    const shown = new Set(g.panels.map((p, f) => [p, f]).filter(([p]) => p.article.classList.contains('in-strip') && p.ghosts.canvas.clientWidth).map(([, f]) => f));
    const marks = g.marks.slice();
    for (const m of marks) if (m[0] !== 'followed') cells.add(m[4] + ',' + m[5]);
    const alone = m => marks.every(o => o === m || o[1] !== m[1] || (m[0] === 'followed' && o[2] === m[2] && o[3] === m[3]) || Math.max(Math.abs(o[2] - m[2]), Math.abs(o[3] - m[3])) > 2);
    for (const [kind, test] of [['followed', green], ['ghost', green], ['enemy', red]]) {
      if (found[kind]) continue;
      const m = marks.find(m => m[0] === kind && shown.has(m[1]) && alone(m) && m[2] > 2 && m[2] < 45 && m[3] > 2 && m[3] < 45);
      if (m) found[kind] = { step: g.pb.t, ...scan(m, test) };
    }
  }
  // Each drawn atlas cell's sprite against its disk.
  const { pad, cell } = g.atlas, atlas = g.atlas.getContext('2d').getImageData(0, 0, g.atlas.width, g.atlas.height).data, aw = g.atlas.width;
  // Projectile rows (4 on, a row per class and facing) turn their sprite;
  // this measures only facing 0, unturned (ghosts.js mapSprite).
  const spriteOf = (row, col) => row === 0 ? [39, 40, 38, 37, 41][col] : row === 1 ? 80 + col : row === 2 ? 88 + col : row === 3 ? 91 + col : (row - 4) % 5 ? -1 : [50, 99, 100, 101, 50, 102, 100, 101][col];
  const mask = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  mask.canvas.width = mask.canvas.height = 9; mask.imageSmoothingEnabled = false;
  const legible = [];
  for (const key of cells) {
    const [row, col] = key.split(',').map(Number), id = spriteOf(row, col);
    if (id < 0) continue;
    mask.clearRect(0, 0, 9, 9); mask.drawImage(g.sprites, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, 0, 0, 9, 9);
    const alpha = mask.getImageData(0, 0, 9, 9).data, at = (x, y) => ((row * cell + pad + 1 + y) * aw + col * cell + pad + 1 + x) * 4;
    const clear = [], solid = [];
    for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
      const i = at(x, y), px = [atlas[i], atlas[i + 1], atlas[i + 2]];
      if (alpha[(y * 9 + x) * 4 + 3] > 200) solid.push(px);
      else if (alpha[(y * 9 + x) * 4 + 3] === 0 && Math.hypot(x - 4, y - 4) < 3) clear.push(px);
    }
    if (!clear.length || !solid.length) continue;
    const disk = [0, 1, 2].map(c => clear.map(p => p[c]).sort((a, b) => a - b)[clear.length >> 1]);
    const share = solid.filter(p => Math.max(...p.map((v, c) => Math.abs(v - disk[c]))) > 48).length / solid.length;
    legible.push([key, id, +share.toFixed(2)]);
  }
  return { ...found, legible }; })()`;
async function outlineCase(chrome, url, out, failures) {
  const report = [];
  for (const [width, height, mobile] of [[1440, 900, false], [390, 844, true]]) {
    const fail = message => failures.push(`outlines at ${width}x${height}: ${message}`);
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await chrome.send('Page.navigate', { url });
    if (!await until(chrome, 'document.readyState === "complete" && !!window.ghosts?.pb', 120)) { fail('the figure did not load'); continue; }
    const found = await chrome.evaluate(OUTLINES), { followed, ghost, enemy } = found;
    if (!found.legible) { fail('the markers were not measured'); continue; }
    report.push({ width, ...found });
    if (!followed || !ghost || !enemy) { fail(`no lone ${['followed', 'ghost', 'enemy'].filter(k => !found[k]).join(', ')} marker to measure`); continue; }
    // followed > ghost > enemy, each by a margin, in every measure.
    for (const [measure, margin] of [['width', 0.3], ['bright', 25], ['opacity', 0.1]]) {
      const [a, b, c] = [followed, ghost, enemy].map(m => m[measure]);
      if (!(a >= b + (measure === 'opacity' ? 0 : margin) && b >= c + margin)) fail(`${measure} does not rank followed (${a}) > ghost (${b}) > enemy (${c})`);
    }
    if (!(ghost.width >= 0.5 && enemy.width >= 0.5 && enemy.lines.every(([count]) => count > 0))) fail(`a ghost's or an enemy's outline is missing or not red: ghost ${JSON.stringify(ghost)}, enemy ${JSON.stringify(enemy)}`);
    const hidden = found.legible.filter(([, , share]) => share < MARK_LEGIBLE);
    if (!found.legible.length || hidden.length) fail(`markers whose sprite does not show on its disk (atlas cell, sprite, share of its pixels that do): ${JSON.stringify(hidden.length ? hidden : 'none measured')}`);
  }
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  console.log(`outlines: ${report.map(r => `${r.width}: ${['followed', 'ghost', 'enemy'].map(k => r[k] ? `${k} ${r[k].width} px, ${r[k].bright}, opacity ${r[k].opacity}` : `${k} none`).join('; ')}; sprites shown ${(r.legible ?? []).map(([key, , share]) => `${key}:${share}`).join(' ')}`).join(' | ')}`);
  return report;
}

// Where each plot's play head shows, in CSS pixels: the centre of the red
// caret's run of pixels along the top and of the line's near the bottom, and
// the plot area it must stay in.
const CURSORS = `(() => { const out = {}, dpr = devicePixelRatio, red = (d, i) => d[i] > 230 && d[i + 1] < 70 && d[i + 2] < 70;
  const centre = (c, y) => { const d = c.getContext('2d').getImageData(0, y, c.width, 1).data; let a = -1, b = -1; for (let x = 0; x < c.width; x++) if (red(d, 4 * x)) { if (a < 0) a = x; b = x; } return a < 0 ? null : +((a + b) / 2 / dpr).toFixed(2); };
  for (const name of ['occupancy', 'reward']) {
    const c = document.querySelector('[data-g="' + name + '"]'), area = name === 'reward' ? ghosts.rewardArea : { left: 0, right: c.width };
    out[name] = { caret: centre(c, Math.round(3 * dpr)), line: centre(c, c.height - Math.round(2 * dpr)), left: +(area.left / dpr).toFixed(1), right: +(area.right / dpr).toFixed(1) };
  }
  return out; })()`;

// The plot as scrubber: a press at 40% and a drag to 75% of its width seek
// there; keys move the play head; aria-valuenow follows.
async function scrubCase(chrome, fail) {
  const box = await chrome.evaluate(`(() => { const b = document.querySelector('.scrub canvas').getBoundingClientRect(); return { x: b.left, y: b.top + b.height / 2, w: b.width }; })()`);
  const at = f => box.x + f * box.w, state = `({ u: ghosts.u, t: ghosts.pb.t, span: ghosts.timeline.length, now: Number(document.querySelector('.scrub').getAttribute('aria-valuenow')), playing: ghosts.playing })`;
  await chrome.evaluate('ghosts.stop()');
  await chrome.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: at(0.4), y: box.y, button: 'left', buttons: 1, clickCount: 1 });
  await sleep(150);
  const pressed = await chrome.evaluate(state);
  await chrome.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: at(0.75), y: box.y, button: 'left', buttons: 1 });
  await sleep(150);
  await chrome.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: at(0.75), y: box.y, button: 'left', buttons: 0, clickCount: 1 });
  await sleep(150);
  const dragged = await chrome.evaluate(state);
  const near = (s, f) => Math.abs(s.u - f * s.span) <= Math.max(2, s.span / box.w * 2);
  if (!near(pressed, 0.4) || !near(dragged, 0.75) || dragged.now !== dragged.u) fail(`dragging the plot does not seek: ${JSON.stringify({ pressed, dragged })}`);
  await chrome.evaluate(`document.querySelector('.scrub').focus()`);
  const key = async (k, code, modifiers = 0) => {
    await chrome.send('Input.dispatchKeyEvent', { type: 'keyDown', key: k, code, modifiers, windowsVirtualKeyCode: { Home: 36, End: 35, ArrowRight: 39, ' ': 32 }[k] });
    await chrome.send('Input.dispatchKeyEvent', { type: 'keyUp', key: k, code, modifiers });
    await sleep(120);
    return chrome.evaluate(state);
  };
  const home = await key('Home', 'Home'), right = await key('ArrowRight', 'ArrowRight'), far = await key('ArrowRight', 'ArrowRight', 8), end = await key('End', 'End'), space = await key(' ', 'Space');
  if (home.u !== 0 || right.u !== 1 || far.u !== 101 || end.u !== end.span || !space.playing) fail(`keys on the plot do not step it: ${JSON.stringify({ home, right, far, end, space })}`);
  await chrome.evaluate('ghosts.stop()');
  return { pressed: pressed.u, dragged: dragged.u, span: dragged.span, keys: [home.u, right.u, far.u, end.u], space: space.playing };
}

// Restart while playing keeps playing from step 0; while paused stays at 0;
// either way the trails, heat, win count, Follow strip and policy view start
// over.
async function restartCase(chrome, fail) {
  const STATE = `(() => { const g = ghosts, longest = Math.max(...Array.from(g.pb.trailLength)); let heat = 0; for (const v of g.pb.heat) heat += v;
    return { t: g.pb.t, playing: g.playing, won: g.won, strip: g.strip.at, panel: g.policyShown ?? null, trail: longest, heat }; })()`;
  const click = `document.querySelector('[data-g="restart"]').click()`;
  await chrome.evaluate(`(() => { const s = document.querySelector('[data-g="speed"]'); s.value = s.options[s.options.length - 1].value; ghosts.goTo(Math.round(ghosts.set.maxSteps * 0.6)); ghosts.play(); })()`);
  await sleep(500);
  const before = await chrome.evaluate(STATE);
  await chrome.evaluate(click);
  const playing = await chrome.evaluate(STATE);
  await sleep(300);
  const after = await chrome.evaluate(STATE);
  await chrome.evaluate(`(() => { ghosts.stop(); ghosts.goTo(Math.round(ghosts.set.maxSteps * 0.6)); })()`);
  await sleep(200);
  await chrome.evaluate(click);
  await sleep(200);
  const paused = await chrome.evaluate(STATE);
  const fresh = s => s.won === 0 && s.strip === 0 && s.trail <= 1 && (s.panel === null || s.panel === 0);
  if (!(playing.t === 0 && fresh(playing) && after.playing && after.t > 0 && after.t < before.t)) fail(`restart while playing: ${JSON.stringify({ before, playing, after })}`);
  if (!(paused.t === 0 && !paused.playing && fresh(paused))) fail(`restart while paused: ${JSON.stringify(paused)}`);
  return { before: before.t, playing: [playing.t, after.playing, after.t], paused: [paused.t, paused.playing] };
}

async function embedCase(chrome, url, out, failures) {
  const report = {}, fail = message => failures.push(`embed: ${message}`), shot = name => screenshot(chrome, path.join(out, `${path.basename(out)}-embed-${name}.png`));
  // Instant: a page with smooth scrolling would still be on its way.
  const reveal = `document.querySelector('[data-ghost-embed]').scrollIntoView({ block: 'center', behavior: 'instant' })`;
  await chrome.send('Page.navigate', { url });
  await until(chrome, 'document.readyState === "complete" && !!window.ghosts', 30);
  await sleep(1500);
  report.beforeScroll = await chrome.evaluate(EMBED_STATE);
  if (report.beforeScroll.loaded) fail('loaded its data while two screens away');
  await chrome.evaluate(reveal);
  report.playing = await until(chrome, 'window.ghosts?.playing && ghosts.pb.t > 0', 120);
  if (!report.playing) fail(`did not play once in view: ${JSON.stringify(await chrome.evaluate(EMBED_STATE))}`);
  const samples = [];
  for (let i = 0; i < 5; i++) { await sleep(300); samples.push(await chrome.evaluate(EMBED_STATE)); }
  report.samples = samples;
  const last = samples.at(-1);
  if (last.shortest >= 0 ? last.selected !== last.shortest || last.outcome !== 'win' : last.selected < 0) fail(`follows run ${last.selected} (${last.outcome}), not the shortest win ${last.shortest}`);
  if (samples.some(s => s.panel !== s.followed)) fail(`the panel left the followed run's decision: ${JSON.stringify(samples.map(s => [s.t, s.followed, s.panel]))}`);
  await shot('playing');
  await chrome.evaluate(`window.scrollTo({ top: 0, behavior: 'instant' })`);
  report.pausedAway = await until(chrome, '!ghosts.playing', 5);
  if (!report.pausedAway) fail('kept playing when scrolled away');
  await chrome.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await chrome.send('Page.navigate', { url });
  await until(chrome, 'document.readyState === "complete" && !!window.ghosts', 30);
  await chrome.evaluate(reveal);
  await until(chrome, 'window.ghosts?.pb && ghosts.pb.t > 300', 120);
  report.phone = await chrome.evaluate(EMBED_STATE);
  if (report.phone.columns !== 3 || report.phone.scrollWidth > report.phone.innerWidth) fail(`at 390 px: ${report.phone.columns} columns, ${report.phone.scrollWidth} px wide`);
  await chrome.evaluate(reveal);
  await shot('phone');
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  await chrome.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
  await chrome.send('Page.navigate', { url });
  await until(chrome, 'document.readyState === "complete" && !!window.ghosts', 30);
  await chrome.evaluate(reveal);
  await until(chrome, '!!window.ghosts?.pb', 120);
  await sleep(1000);
  report.reducedMotion = await chrome.evaluate(EMBED_STATE);
  if (report.reducedMotion.playing || report.reducedMotion.t !== 0) fail('played for a reader who prefers reduced motion');
  await chrome.send('Emulation.setEmulatedMedia', { features: [] });
  console.log(`embed: ${failures.filter(f => f.startsWith('embed')).length ? 'failed' : 'ok'}; follows ${last.selected} (${last.outcome}); panel in step at ${samples.map(s => s.t).join(', ')}`);
  return report;
}

// Load one tier, set and count; seek to its first, middle and last position.
async function combination(chrome, tier, set, count, last, out, failures) {
  const t0 = Date.now();
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`);
  const id = `${tier}/${set}/${count}`, loaded = await until(chrome, `(${LOADED})(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`, 300);
  const combo = { tier, set, count, loaded, loadSeconds: (Date.now() - t0) / 1000, status: await chrome.evaluate(`document.getElementById('status').textContent`) };
  if (!loaded) {
    failures.push(`${id} did not load: ${combo.status}`);
    return combo;
  }
  Object.assign(combo, await chrome.evaluate(`({ decisions: ghosts.stats.decisions, longest: ghosts.set.maxDecisions, shown: ghosts.timeline.length })`), { seeks: [] });
  for (const fraction of [0, 0.5, 1]) {
    const pixels = await seekTo(chrome, fraction), ghostPixels = pixels.ghosts.reduce((a, b) => a + b), layerPixels = pixels.layer.reduce((a, b) => a + b);
    combo.seeks.push({ fraction, ghostPixels, layerPixels });
    if (!ghostPixels) failures.push(`${id} at ${fraction}: no ghosts or end marks drawn`);
    if (fraction > 0 && !layerPixels) failures.push(`${id} at ${fraction}: no heat or map changes drawn`);
  }
  if (last) {
    await seekTo(chrome, 0.3);
    await screenshot(chrome, path.join(out, `${path.basename(out)}-${tier}-${set}-${count}.png`));
  }
  console.log(`${id}: loaded in ${combo.loadSeconds.toFixed(1)} s, ${combo.decisions} decisions, ${combo.shown + 1} of ${combo.longest + 1} states shown`);
  return combo;
}

// Play every tier's sets at their largest count from decision 0 at 3,000
// decisions/s for 4 s, skipping quiet stretches and in real time; record the
// page's work per frame (seek plus draw) and the browser's frame intervals,
// and time 20 random seeks. Screenshots the heaviest set mid-play.
async function timingRuns(chrome, out, failures) {
  const sets = await chrome.evaluate(`ghosts.manifest.tiers.flatMap(t => t.sets.map(s => [t.name, s.name, s.counts.at(-1), s.decisions])).sort((a, b) => b[3] - a[3])`);
  const results = [];
  for (const [tier, set, count] of sets) {
    // Each set loads cold here: download, base64 and gzip decoding, then decode.js.
    const t0 = Date.now();
    await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`);
    await until(chrome, `(${LOADED})(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`, 300);
    const loadSeconds = (Date.now() - t0) / 1000, id = `${tier}/${set}/${count}`;
    console.log(`cold load ${id}: ${loadSeconds.toFixed(2)} s`);
    for (const skip of [true, false]) {
      const played = await playFor(chrome, { skip, speed: 3000, from: 0, seconds: 4 }, `${id} skip ${skip ? 'on' : 'off'}`, failures);
      if (!results.length) await screenshot(chrome, path.join(out, `${path.basename(out)}-playing.png`));
      const seeks = skip ? [] : await chrome.evaluate(`(() => { const out = []; for (let i = 0; i < 20; i++) { ghosts.goTo(Math.floor(Math.random() * ghosts.set.maxDecisions)); out.push(ghosts.frameTimes.at(-1)); } return out; })()`);
      const result = { tier, set, count, skip, loadSeconds, reachedDecision: played.t, heatSamples: played.samples, workMs: timing(played.work), frameIntervalMs: timing(played.intervals), ...(seeks.length ? { seekMs: timing(seeks) } : {}) };
      if (result.workMs.p95 > 33) failures.push(`${id}: p95 frame work ${result.workMs.p95} ms exceeds 33 ms`);
      console.log(`timing ${id} skip ${skip ? 'on' : 'off'} to decision ${played.t}: work ${JSON.stringify(result.workMs)}; intervals ${JSON.stringify(result.frameIntervalMs)}${seeks.length ? `; seeks ${JSON.stringify(result.seekMs)}` : ''}`);
      results.push(result);
    }
    // Slower speeds, from a third of the way in, for the heat check only.
    for (const speed of [30, 300]) {
      const from = await chrome.evaluate('Math.round(ghosts.set.maxDecisions / 3)');
      results.push({ tier, set, count, speed, heatSamples: (await playFor(chrome, { skip: false, speed, from, seconds: 1.5 }, `${id} at ${speed}/s`, failures)).samples });
    }
  }
  return results;
}

// Whether the heat on screen is the heat at the decision on screen, and
// whether it covers every living player ghost's tile: read from the page's
// state between frames, not from pixels.
const HEAT_STATE = `(() => { const { pb, set } = ghosts; let alive = 0, missing = 0;
  for (let e = 0; e < set.n; e++) if (pb.alive(e)) { alive++; missing += !(pb.heat[pb.cur.floor[e] * 2304 + pb.cur.row[e] * 48 + pb.cur.col[e]] > 0); }
  return { t: pb.t, drawnAt: ghosts.heatDrawnAt, alive, missing }; })()`;

// Play from decision `from` at `speed` for `seconds`, sampling HEAT_STATE five
// times; returns the samples, the page's work per frame and the frame intervals.
async function playFor(chrome, { skip, speed, from, seconds }, label, failures) {
  await chrome.evaluate(`(() => { const box = document.getElementById('skip-quiet'); if (box.checked !== ${skip}) box.click(); ghosts.goTo(${from});
    document.getElementById('speed').value = '${speed}'; ghosts.frameTimes.length = 0;
    window.intervals = []; let last = 0; const tick = now => { if (last) window.intervals.push(now - last); last = now; if (ghosts.playing) requestAnimationFrame(tick); };
    ghosts.play(); requestAnimationFrame(tick); })()`);
  const samples = [];
  for (let i = 0; i < 5; i++) {
    await sleep(seconds * 200);
    const sample = await chrome.evaluate(HEAT_STATE);
    samples.push(sample);
    if (sample.missing || sample.drawnAt !== sample.t) failures.push(`${label}: at decision ${sample.t} the heat shown is for ${sample.drawnAt} and misses ${sample.missing} of ${sample.alive} living ghosts' tiles`);
  }
  return { samples, ...await chrome.evaluate(`(() => { ghosts.stop(); return { work: ghosts.frameTimes.slice(), intervals: window.intervals.slice(1), t: ghosts.pb.t }; })()`) };
}

// "One row" at 1,440 and 1,920 px and on a 390 px phone: nine square boards
// side by side across the whole window, touching, with no sideways scroll;
// the choice survives a reload. Restores the grid.
async function rowLayout(chrome, url, out, failures) {
  const report = [];
  const ROW = `(() => { const boards = [...document.querySelectorAll('.floor:not(.tally) .board')].map(b => b.getBoundingClientRect());
    return { innerWidth, scrollWidth: document.documentElement.scrollWidth, tops: new Set(boards.map(b => Math.round(b.top))).size,
      width: boards[0].width, height: boards[0].height, left: boards[0].left, right: boards.at(-1).right }; })()`;
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'row'; s.dispatchEvent(new Event('change')); })()`);
  for (const [width, height, mobile] of [[1440, 1000, false], [1920, 1080, false], [390, 844, true]]) {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(400);
    const row = await chrome.evaluate(ROW);
    report.push({ width, ...row });
    if (row.scrollWidth > row.innerWidth || row.tops !== 1 || Math.abs(row.left) > 0.5 || Math.abs(row.right - row.innerWidth) > 0.5 || Math.abs(row.width - row.height) > 0.5) {
      failures.push(`one row at ${width} px is not nine square boards across the window: ${JSON.stringify(row)}`);
    }
    if (!mobile) await screenshot(chrome, path.join(out, `${path.basename(out)}-one-row-${width}.png`));
  }
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  const kept = await open(chrome, url) && await chrome.evaluate(`document.getElementById('page').classList.contains('one-row')`);
  if (!kept) failures.push('one row did not survive a reload');
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'grid'; s.dispatchEvent(new Event('change')); })()`);
  return { sizes: report, keptAfterReload: kept };
}

// The reward plot: drawn, with many colours, and its cursor where the seek
// bar's thumb is after each of three seeks.
async function rewardPlot(chrome, failures) {
  const STATE = `(() => { const c = document.getElementById('reward'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, w = c.width, h = c.height;
    const rgb = [255, 42, 42];
    const colours = new Set(); for (let i = 0; i < d.length; i += 28) colours.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    const near = x => { const at = 4 * (Math.round(h / 3) * w + Math.round(x)); return Math.abs(d[at] - rgb[0]) + Math.abs(d[at + 1] - rgb[1]) + Math.abs(d[at + 2] - rgb[2]) < 40; };
    // The plot's time axis spans its plot area, between the axis labels.
    const { left, right } = ghosts.rewardArea, half = 6.5 * devicePixelRatio, x = Math.min(right - half - devicePixelRatio, Math.max(left + half + devicePixelRatio, left + ghosts.u / Math.max(1, ghosts.timeline.length) * (right - left)));
    return { colours: colours.size, cursor: near(x), u: ghosts.u, x: Math.round(x), runs: ghosts.reward.runs.length }; })()`;
  const report = [];
  for (const fraction of [0.2, 0.55, 0.9]) {
    await seekTo(chrome, fraction);
    report.push(await chrome.evaluate(STATE));
  }
  if (report.some(r => r.colours < 20 || !r.cursor || !r.runs) || new Set(report.map(r => r.x)).size !== 3) failures.push(`the reward plot is blank or its cursor does not follow the seek bar: ${JSON.stringify(report)}`);
  console.log(`reward plot: ${report.map(r => `cursor at ${r.x} px for step ${r.u}, ${r.colours} colours`).join('; ')}`);
  return report;
}

// "Two rows" at 1,440 and 1,920 px and on a 390 px phone: ten equal squares,
// floors 0-4 over floors 5-8 and the win tally, across the whole window with
// no sideways scroll; the tally's count is the readout's and the timeline's
// wins at every seek. Screenshots it mid-playback on the last tier's runs.
async function twoRows(chrome, url, out, failures) {
  const report = { sizes: [], counts: [] };
  const GEOMETRY = `(() => { const boards = [...document.querySelectorAll('.board')].map(b => b.getBoundingClientRect()).filter(b => b.width);
    const tops = [...new Set(boards.map(b => Math.round(b.top)))];
    return { innerWidth, scrollWidth: document.documentElement.scrollWidth, boards: boards.length, tops: tops.length, perRow: tops.map(t => boards.filter(b => Math.round(b.top) === t).length),
      widths: [...new Set(boards.map(b => Math.round(b.width)))], square: boards.every(b => Math.abs(b.width - b.height) <= 0.5), left: Math.min(...boards.map(b => b.left)), right: Math.max(...boards.map(b => b.right)),
      tallyLast: [...document.querySelectorAll('.floors .floor')].at(-1).classList.contains('tally') }; })()`;
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'two-rows'; s.dispatchEvent(new Event('change')); })()`);
  for (const [width, height, mobile] of [[1440, 1000, false], [1920, 1080, false], [390, 844, true]]) {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(400);
    const g = await chrome.evaluate(GEOMETRY);
    report.sizes.push({ width, ...g });
    if (g.scrollWidth > g.innerWidth || g.boards !== 10 || g.tops !== 2 || g.perRow.join() !== '5,5' || g.widths.length !== 1 || !g.square || Math.abs(g.left) > 0.5 || Math.abs(g.right - g.innerWidth) > 0.5 || !g.tallyLast) {
      failures.push(`two rows at ${width} px are not ten equal squares, five to a row, across the window: ${JSON.stringify(g)}`);
    }
  }
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  const COUNT = `(() => ({ u: ghosts.pb.t, tally: ghosts.tally.won, readout: Number(document.getElementById('position').textContent.match(/([\\d,]+) won/)[1].replace(/,/g, '')),
    brute: Array.from({ length: ghosts.set.n }, (_, e) => ghosts.set.outcome[e] === 2 && ghosts.set.length[e] <= ghosts.pb.t).filter(Boolean).length }))()`;
  for (const fraction of [0, 0.35, 0.7, 1]) {
    await seekTo(chrome, fraction);
    report.counts.push(await chrome.evaluate(COUNT));
  }
  if (report.counts.some(c => c.tally !== c.brute || c.readout !== c.brute)) failures.push(`the tally, readout and wins disagree: ${JSON.stringify(report.counts)}`);
  if (!report.counts.at(-1).brute) failures.push('two rows: no run has won by the end of the last tier');
  await seekTo(chrome, 0.45);
  await screenshot(chrome, path.join(out, `${path.basename(out)}-two-rows.png`));
  report.keptAfterReload = await open(chrome, url) && await chrome.evaluate(`document.getElementById('page').classList.contains('two-rows')`);
  if (!report.keptAfterReload) failures.push('two rows did not survive a reload');
  console.log(`two rows: ${report.sizes.map(s => `${s.width} px boards ${s.widths[0]} px`).join(', ')}; tally ${report.counts.map(c => c.tally).join(', ')}`);
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'grid'; s.dispatchEvent(new Event('change')); })()`);
  return report;
}

// The first floor of the Follow window of `boards` floors at every state of
// the followed run, by the rule (keep the run in view, move as little as
// needed, start on floor 0), computed here apart from the page's own.
const STRIP_WINDOWS = boards => `(() => { const st = ghosts.selectedStates, end = ghosts.set.decisions[ghosts.selected], out = [];
  for (let s = 0, first = 0; s <= end; s++) { const f = st[4 * s]; first = f > first + ${boards - 1} ? f - ${boards - 1} : f < first ? f : first; out.push(first); } return out; })()`;
const floorsFrom = (first, boards) => Array.from({ length: boards }, (_, k) => first + k).join();

// "Follow" with `boards` boards (3 as the page offers it, 2 as the blog's
// embed sets it) on the last tier's runs, following the shortest win (or the
// median run): at 1,440, 1,920 and 390 px that many equal squares edge to
// edge with no sideways scroll; at sampled steps, reached by seeks in both
// directions and including either side of a step where the run climbs back
// up, the boards are the window the rule gives and the run's board is
// outlined. Then the conveyor in playback (conveyor). Screenshots it at
// 1,440 px.
async function followLayout(chrome, url, out, failures, boards) {
  const report = { boards, sizes: [], steps: [] }, name = boards === 3 ? 'follow' : `follow-${boards}`;
  await chrome.evaluate(`(() => { ghosts.setFollowBoards(${boards}); const s = document.getElementById('layout'); s.value = 'follow'; s.dispatchEvent(new Event('change'));
    const f = document.getElementById('follow'); f.value = 'shortest-win'; f.dispatchEvent(new Event('change')); })()`);
  for (const [width, height, mobile] of [[1440, 1000, false], [1920, 1080, false], [390, 844, true]]) {
    await chrome.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await sleep(400);
    const g = await chrome.evaluate(STRIP);
    report.sizes.push({ width, ...g });
    if (g.scrollWidth > g.innerWidth || g.floors.length !== boards || g.tops !== 1 || g.widths.length !== 1 || !g.square || Math.abs(g.left) > 0.5 || Math.abs(g.right - g.innerWidth) > 0.5) {
      failures.push(`${name} at ${width} px is not ${boards} equal squares across the window: ${JSON.stringify(g)}`);
    }
  }
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  const steps = await chrome.evaluate(`(() => { const st = ghosts.selectedStates, e = ghosts.selected, end = ghosts.set.decisions[e], out = [0, end, Math.round(end / 3), Math.round(2 * end / 3)];
    for (let t = 1; t <= end; t++) if (st[4 * t] < st[4 * (t - 1)] && st[4 * (t - 1)] >= 3) { out.push(t - 1, t, t + 1); break; }
    return out.map(t => GhostDecode.stepOf(ghosts.set, e, Math.min(t, end))); })()`);
  await strips(chrome, steps, report.steps, failures, boards);
  if (steps.length < 7) failures.push(`${name}: the followed run never climbs back up from floor 3 or below, so no climb was checked`);
  report.moves = await conveyor(chrome, out, failures, boards);
  await chrome.evaluate(`(() => { const f = document.getElementById('follow'); f.value = 'shortest-win'; f.dispatchEvent(new Event('change')); })()`);
  await chrome.evaluate(`ghosts.goTo(${steps[2]})`);
  await sleep(600);
  await screenshot(chrome, path.join(out, `${path.basename(out)}-${name}.png`));
  console.log(`${name}: ${report.sizes.map(s => `${s.width} px boards ${s.widths[0]} px`).join(', ')}; floors ${report.steps.map(s => `${s.u}:${s.floor}->${s.floors.join('')}`).join(' ')}`);
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'grid'; s.dispatchEvent(new Event('change')); })()`);
  return report;
}

// Seek to each step and check the strip shows the rule's window with the run
// outlined on its board.
async function strips(chrome, steps, report, failures, boards) {
  const windows = await chrome.evaluate(STRIP_WINDOWS(boards));
  for (const u of steps) {
    await chrome.evaluate(`ghosts.goTo(${u})`);
    await sleep(300);
    const g = await chrome.evaluate(STRIP), first = windows[g.decision];
    report.push(g);
    if (g.floors.join() !== floorsFrom(first, boards)) failures.push(`follow (${boards}) at step ${u}: the run is on floor ${g.floor} and the window starts at ${first}, but the boards show ${g.floors}`);
    if (g.outlined !== g.floor) failures.push(`follow (${boards}) at step ${u}: the run on floor ${g.floor} is not the outlined board (${g.outlined})`);
    // The followed run's trail is drawn at the board's size, not while its
    // board was hidden (where its strokes would flood the board in green).
    if (!(g.flood >= 0 && g.flood < 0.15)) failures.push(`follow (${boards}) at step ${u}: ${Math.round(100 * g.flood)}% of the followed board is the followed run's green`);
  }
}

// The Follow strip as the page shows it: its boards' floors left to right,
// their geometry and the board outlined for the followed run.
const STRIP = `(() => { const shown = [...document.querySelectorAll('.floor')].map((a, f) => [a, f]).filter(([a]) => a.getBoundingClientRect().width > 0)
    .sort((a, b) => a[0].getBoundingClientRect().left - b[0].getBoundingClientRect().left), boards = shown.map(([a]) => a.querySelector('.board').getBoundingClientRect());
  const e = ghosts.selected, d = ghosts.pb.decision[e];
  return { u: ghosts.pb.t, decision: d, floor: ghosts.selectedStates[4 * d], floors: shown.map(([, f]) => f), innerWidth, scrollWidth: document.documentElement.scrollWidth,
    widths: [...new Set(boards.map(b => Math.round(b.width)))], square: boards.every(b => Math.abs(b.width - b.height) <= 0.5), tops: new Set(boards.map(b => Math.round(b.top))).size,
    left: Math.min(...boards.map(b => b.left)), right: Math.max(...boards.map(b => b.right)), badge: document.querySelector('.won-badge')?.textContent ?? '',
    outlined: shown.find(([a]) => a.classList.contains('followed'))?.[1] ?? -1, flood: (c => { if (!c) return -1; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0, green = 0;
      for (let i = 0; i < d.length; i += 4 * 97) { n++; if (d[i + 1] > 200 && d[i] < 120 && d[i + 2] < 160 && d[i + 3] > 100) green++; } return +(green / n).toFixed(3); })(shown.find(([a]) => a.classList.contains('followed'))?.[0].querySelector('canvas.ghosts')) }; })()`;

// The Follow strip of `boards` boards in playback, standing before a floor
// change and playing one step over it, read 150 ms in and again once it
// rests: on the followed run, its first move on (one board more than the
// window mid-move, the leftmost partly out) and a climb within the window (no
// move); then, on a run of the middle tier that climbs above its window, the
// slide back (the rightmost partly out, the run on the left board at rest)
// and its next move on. Screenshots the first move mid-slide.
async function conveyor(chrome, out, failures, boards) {
  const report = [];
  const FIND = `(() => { const w = ${STRIP_WINDOWS(boards)}, st = ghosts.selectedStates, e = ghosts.selected, end = ghosts.set.decisions[e], found = {};
    for (let t = 0; t < end; t++) {
      if (found.on === undefined && w[t + 1] > w[t]) found.on = t;
      if (found.within === undefined && st[4 * (t + 1)] < st[4 * t] && w[t + 1] === w[t]) found.within = t;
      if (found.back === undefined && w[t + 1] < w[t]) found.back = t;
      if (found.back !== undefined && found.again === undefined && t > found.back && w[t + 1] > w[t]) found.again = t; }
    return Object.entries(found).map(([k, t]) => [k, GhostDecode.stepOf(ghosts.set, e, t), GhostDecode.stepOf(ghosts.set, e, t + 1)]); })()`;
  const STATE = `(() => { const box = document.querySelector('.floors').getBoundingClientRect(), shown = [...document.querySelectorAll('.floor.in-strip')].map(a => [a, a.getBoundingClientRect()])
      .filter(([, r]) => r.width && r.right > box.left + 0.5 && r.left < box.right - 0.5).sort((x, y) => x[1].left - y[1].left), index = a => [...a.parentNode.children].indexOf(a);
    const d = ghosts.pb.decision[ghosts.selected], followed = shown.find(([a]) => a.classList.contains('followed'));
    return { boards: shown.length, floors: shown.map(([a]) => index(a)), lefts: shown.map(([, r]) => Math.round(r.left - box.left)), width: Math.round(box.width),
      moving: shown.some(([a]) => a.getAnimations().length), decision: d, floor: ghosts.selectedStates[4 * d], outlined: followed ? index(followed[0]) : -1 }; })()`;
  const play = async (kind, before, after, windows) => {
    await chrome.evaluate(`ghosts.goTo(${before})`);
    await sleep(500);
    const start = await chrome.evaluate(STATE);
    await chrome.evaluate(`ghosts.goTo(${after}, true)`);
    await sleep(150);
    const mid = await chrome.evaluate(STATE);
    if (kind === 'on') await screenshot(chrome, path.join(out, `${path.basename(out)}-follow${boards === 3 ? '' : `-${boards}`}-moving.png`));
    await sleep(900);
    const end = await chrome.evaluate(STATE), first = windows[end.decision];
    report.push({ kind, before, after, start, mid, end });
    const fail = message => failures.push(`follow (${boards}) ${kind} at step ${after}: ${message}: ${JSON.stringify({ start, mid, end })}`);
    if (kind === 'within' && (mid.moving || mid.floors.join() !== start.floors.join())) fail('the strip moved while the run stayed in view');
    if (kind !== 'within' && (mid.boards < boards + 1 || !mid.moving)) fail(`mid-move the strip is not ${boards + 1} or more moving boards`);
    if ((kind === 'on' || kind === 'again') && mid.lefts[0] >= 0) fail('mid-move the leftmost board is not partly out of the strip');
    if (kind === 'back' && mid.lefts.at(-1) + mid.width / boards <= mid.width) fail('mid-move the rightmost board is not partly out of the strip');
    const rest = Array.from({ length: boards }, (_, k) => Math.round(k * end.width / boards)).join();
    if (end.boards !== boards || end.moving || end.floors.join() !== floorsFrom(first, boards) || end.lefts.join() !== rest) fail(`the strip does not rest on floors ${first}-${first + boards - 1}`);
    if (end.outlined !== end.floor) fail(`the run on floor ${end.floor} is not the outlined board`);
    if (kind === 'back' && end.floors[0] !== end.floor) fail('after sliding back the run is not on the left board');
  };
  let windows = await chrome.evaluate(STRIP_WINDOWS(boards));
  const moves = await chrome.evaluate(FIND);
  for (const [kind, before, after] of moves.filter(([k]) => k === 'on' || k === 'within')) await play(kind, before, after, windows);
  // A run of the middle tier that climbs above its window, followed by hand.
  const tier = await chrome.evaluate(`(() => { const t = ghosts.manifest.tiers.find(t => t.name === 'medium') ?? ghosts.manifest.tiers[0]; return [t.name, t.sets.find(s => s.name === 'all').counts[0]]; })()`);
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier[0])}, 'all', ${tier[1]})`);
  await until(chrome, `(${LOADED})(${JSON.stringify(tier[0])}, 'all', ${tier[1]})`, 300);
  const climber = await chrome.evaluate(`(() => { const set = ghosts.set; for (let e = 0; e < set.n; e++) { const st = GhostDecode.decodeEpisode(set, e).states;
      for (let s = 1, first = 0; s <= set.decisions[e]; s++) { const f = st[4 * s]; if (f < first) return e; first = f > first + ${boards - 1} ? f - ${boards - 1} : first; } } return -1; })()`);
  if (climber < 0) failures.push(`follow (${boards}): no ${tier[0]} run climbs above its window, so no slide back was checked`);
  else {
    await chrome.evaluate(`ghosts.select(${climber})`);
    windows = await chrome.evaluate(STRIP_WINDOWS(boards));
    for (const [kind, before, after] of (await chrome.evaluate(FIND)).filter(([k]) => k === 'back' || k === 'again')) await play(kind, before, after, windows);
  }
  const kinds = report.map(r => r.kind);
  for (const kind of ['on', 'within', 'back', 'again']) if (!kinds.includes(kind)) failures.push(`follow (${boards}): no ${kind} move was found to check`);
  console.log(`follow (${boards}) conveyor: ${report.map(r => `${r.kind} at step ${r.after}: ${r.start.floors.join('')} -> mid ${r.mid.floors.join('')} at ${r.mid.lefts.join('/')} -> ${r.end.floors.join('')}, run on ${r.end.floor}`).join('; ')}`);
  return report;
}

// Playback that reaches the end holds it, then starts again from step 0 and
// keeps playing; a reader who stopped it at the end stays there.
async function loopCase(chrome, failures) {
  await chrome.evaluate(`(() => { document.getElementById('speed').value = '3000'; ghosts.goTo(Math.max(0, ghosts.set.maxSteps - 600)); ghosts.play(); })()`);
  const reachedEnd = await until(chrome, 'ghosts.pb.t === ghosts.set.maxSteps', 20);
  const wrapped = await until(chrome, 'ghosts.playing && ghosts.pb.t > 0 && ghosts.pb.t < 2000', 20);
  const after = await chrome.evaluate('({ t: ghosts.pb.t, playing: ghosts.playing, trails: Array.from(ghosts.pb.trailLength).reduce((a, b) => Math.max(a, b), 0) })');
  await chrome.evaluate('(() => { ghosts.stop(); ghosts.goTo(ghosts.set.maxSteps); })()');
  await sleep(2500);
  const held = await chrome.evaluate('({ t: ghosts.pb.t, playing: ghosts.playing, end: ghosts.set.maxSteps })');
  if (!reachedEnd || !wrapped) failures.push(`loop: playback did not start again from the start: ${JSON.stringify(after)}`);
  if (held.playing || held.t !== held.end) failures.push(`loop: a stopped viewer at the end moved on: ${JSON.stringify(held)}`);
  console.log(`loop: wrapped to step ${after.t}, still playing ${after.playing}; stopped at the end stays at ${held.t}`);
  return { reachedEnd, wrapped, after, held };
}

// Projectiles face the way they fly, as original Craftax draws them
// (craftax/renderer.py): each kind's texture points up (constants.py: an
// arrow, a dagger, a fireball, an iceball, an arrow, a slimeball, a fireball,
// an iceball, by type; sheet sprites 51, 99, 100, 101, 51, 102, 100, 101) and
// is flipped top to bottom when it flies down or right (drow > 0 or dcol >
// 0), then transposed when it flies left or right (dcol != 0). For each
// kind and facing (LEFT, RIGHT, UP, DOWN: 1-4), GhostViewer.drawTurned at 16
// px must equal those array operations on the sprite, pixel for pixel; and
// each projectile cell of the marker atlas (both classes, every type and
// facing; ghosts.js markRow) must show its sprite at 9 px turned to its own
// facing: of the sprite drawn at 9 px turned to each of the four, its own
// must match the cell best (by the mean colour difference over the turned
// sprite's opaque pixels; a 16-to-9 scale samples alike only up to a pixel,
// so a match is the closest, not an exact one), ties allowed where turns
// look alike.
const UPSTREAM_PROJECTILES = [51, 99, 100, 101, 51, 102, 100, 101];
const FACINGS = [[1, 0, -1], [2, 0, 1], [3, -1, 0], [4, 1, 0]];
async function projectileCase(chrome, failures) {
  const report = await chrome.evaluate(`(() => { const g = ghosts, sheet = g.sprites, out = { turned: [], atlas: [] };
    const read = (size, draw) => { const x = Object.assign(document.createElement('canvas'), { width: size, height: size }).getContext('2d'); x.imageSmoothingEnabled = false; draw(x); return x.getImageData(0, 0, size, size).data; };
    for (const id of new Set(${JSON.stringify(UPSTREAM_PROJECTILES)})) {
      const base = read(16, x => x.drawImage(sheet, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, 0, 0, 16, 16));
      for (const [facing, drow, dcol] of ${JSON.stringify(FACINGS)}) {
        const want = new Uint8ClampedArray(1024);
        for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
          let [sr, sc] = dcol !== 0 ? [c, r] : [r, c];
          if (drow > 0 || dcol > 0) sr = 15 - sr;
          want.set(base.subarray((sr * 16 + sc) * 4, (sr * 16 + sc) * 4 + 4), (r * 16 + c) * 4);
        }
        const got = read(16, x => GhostViewer.drawTurned(x, sheet, id, 0, 0, 16, facing));
        let diff = 0;
        for (let i = 0; i < 1024; i++) diff = Math.max(diff, Math.abs(got[i] - want[i]));
        out.turned.push([id, facing, diff]);
      }
    }
    const { pad, cell } = g.atlas, atlas = g.atlas.getContext('2d').getImageData(0, 0, g.atlas.width, g.atlas.height).data, aw = g.atlas.width;
    for (const klass of [3, 4]) for (const [facing] of ${JSON.stringify(FACINGS)}) for (let type = 0; type < 8; type++) {
      const row = 4 + 5 * (klass - 3) + facing, scores = [];
      for (const [turn] of ${JSON.stringify(FACINGS)}) {
        const sprite = read(9, x => GhostViewer.drawTurned(x, sheet, ${JSON.stringify(UPSTREAM_PROJECTILES)}[type], 0, 0, 9, turn));
        let sum = 0, opaque = 0;
        for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
          const i = (y * 9 + x) * 4, j = ((row * cell + pad + 1 + y) * aw + type * cell + pad + 1 + x) * 4;
          if (sprite[i + 3] < 255) continue;
          opaque++;
          for (let c = 0; c < 3; c++) sum += Math.abs(atlas[j + c] - sprite[i + c]);
        }
        scores.push(opaque ? +(sum / opaque / 3).toFixed(1) : Infinity);
      }
      out.atlas.push([klass, facing, type, scores]);
    }
    return out; })()`);
  const bad = report.turned.filter(([, , diff]) => diff > 0), off = report.atlas.filter(([, facing, , scores]) => !(scores[facing - 1] <= Math.min(...scores)) || !Number.isFinite(scores[facing - 1]));
  if (bad.length) failures.push(`projectiles: turned sprites differ from Craftax's flip-then-transpose (sprite, facing, max difference): ${JSON.stringify(bad)}`);
  if (off.length) failures.push(`projectiles: atlas cells that match another facing better (class, facing, type, mean difference to each facing 1-4): ${JSON.stringify(off.slice(0, 12))}`);
  console.log(`projectiles: ${report.turned.length} turned sprites, ${bad.length} wrong; ${report.atlas.length} atlas cells, ${off.length} wrong`);
  return report;
}

// Creatures and projectiles in the dark: on the boss tier's runs, with
// trails and end marks off and the fog off, one lone marker of each kind (a
// projectile, a melee, a passive and a ranged creature, a ghost player, and
// the followed player), drawn two tiles or more from any other (app.marks),
// its tile's light set by hand. At light 12 of 255 (original Craftax's 0.05
// and below; the agent sees nothing there) no pixel of a creature or
// projectile is drawn, disk and ring included; at 128 each is drawn, its
// marker's pixels darkened as its tile is, by the shade 0.7 (1 - 128 / 255)
// of SHADE_RGB over them. Players show at both, unshaded.
const DARK_KINDS = { projectile: 'm[4] >= 4', melee: 'm[4] === 1', passive: 'm[4] === 2', ranged: 'm[4] === 3', ghost: "m[0] === 'ghost'", followed: "m[0] === 'followed'" };
async function darkCase(chrome, failures) {
  const fail = message => failures.push(`dark: ${message}`), [tier, set, count] = ['boss', 'all', 100];
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`);
  if (!await until(chrome, `(${LOADED})(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`, 300)) { fail('the boss tier did not load'); return null; }
  const report = await chrome.evaluate(`(async () => {
    const g = ghosts, wait = () => new Promise(r => setTimeout(r, 60)), kinds = { ${Object.entries(DARK_KINDS).map(([k, test]) => `${k}: m => ${test}`).join(', ')} }, found = {};
    Object.assign(g.show, { trails: false, deaths: false, fog: false }); g.marks = []; g.held = true; g.stop();
    for (let k = 1; k < 200 && Object.keys(found).length < Object.keys(kinds).length; k++) {
      g.goTo(Math.round(k / 200 * g.set.maxSteps)); await wait();
      const marks = g.marks.slice(), followed = marks.find(m => m[0] === 'followed');
      // The followed player's marker is also listed as a player's (drawMark).
      const same = (o, m) => o[1] === m[1] && o[2] === m[2] && o[3] === m[3];
      const lone = m => m[2] > 1 && m[2] < 46 && m[3] > 1 && m[3] < 46 && marks.every(o => o === m || same(o, m) && (m[0] === 'followed' || o[0] === 'followed') || o[1] !== m[1] || Math.max(Math.abs(o[2] - m[2]), Math.abs(o[3] - m[3])) > 2);
      for (const [kind, test] of Object.entries(kinds)) {
        if (found[kind]) continue;
        const m = marks.find(m => test(m) && lone(m) && (kind !== 'ghost' || !followed || !same(m, followed)));
        if (m) found[kind] = { step: g.pb.t, mark: kind === 'followed' ? marks.find(o => o[0] === 'ghost' && same(o, m)) ?? m : m };
      }
    }
    const out = {};
    for (const [kind, { step, mark }] of Object.entries(found)) {
      g.goTo(step); await wait();
      const [, floor, row, col, arow = 0, acol = 0] = mark, tile = floor * 2304 + row * 48 + col, { pad, cell } = g.atlas, before = g.pb.light[tile];
      const region = () => g.panels[floor].ghosts.getImageData(col * 11 - pad, row * 11 - pad, cell, cell).data;
      const atlas = g.atlas.getContext('2d').getImageData(acol * cell, arow * cell, cell, cell).data;
      g.pb.light[tile] = 12; g.goTo(step); await wait();
      const dark = region();
      g.pb.light[tile] = 128; g.goTo(step); await wait();
      const lit = region(), player = kind === 'ghost' || kind === 'followed', shade = player ? 0 : Math.round(255 * 0.7 * (1 - 128 / 255)) / 255, rgb = [7, 13, 17];
      g.pb.light[tile] = before; g.goTo(step); await wait();
      let darkPixels = 0, litPixels = 0, litDiff = 0, unchanged = 0;
      for (let i = 0; i < dark.length; i += 4) if (dark[i + 3]) darkPixels++;
      for (let i = 0; i < dark.length; i++) unchanged = Math.max(unchanged, Math.abs(dark[i] - lit[i]));
      for (let i = 0; i < atlas.length; i += 4) {
        // The marker's opaque pixels where it drew opaquely enough to read
        // its colour back, ring and casing included. (A player is checked
        // instead by its region not changing with the light: the followed
        // player's own ring and trail cover its cell.)
        if (atlas[i + 3] < 255 || lit[i + 3] < 64) continue;
        litPixels++;
        for (let c = 0; c < 3; c++) litDiff = Math.max(litDiff, Math.abs(lit[i + c] - (atlas[i + c] * (1 - shade) + rgb[c] * shade)));
      }
      out[kind] = { step, floor, row, col, darkPixels, litPixels, litDiff: Math.round(litDiff), unchanged, opacity: +(lit[(Math.floor(cell / 2) * cell + Math.floor(cell / 2)) * 4 + 3] / 255).toFixed(2) };
    }
    return out;
  })()`);
  for (const kind of Object.keys(DARK_KINDS)) {
    const r = report[kind];
    if (!r) { fail(`no lone ${kind} marker was found to check`); continue; }
    const player = kind === 'ghost' || kind === 'followed';
    if (player) {
      if (!r.darkPixels || r.unchanged) fail(`a ${kind} player is hidden or shaded by its tile's light (${r.darkPixels} pixels at light 12, ${r.unchanged} the largest change from light 128): ${JSON.stringify(r)}`);
      continue;
    }
    if (r.darkPixels) fail(`a ${kind} on a tile of light 12 has ${r.darkPixels} pixels drawn: ${JSON.stringify(r)}`);
    if (!r.litPixels || r.litDiff > 3) fail(`a ${kind} on a tile of light 128 is not darkened as its tile is (max channel difference ${r.litDiff}): ${JSON.stringify(r)}`);
  }
  console.log(`dark: ${Object.entries(report).map(([kind, r]) => `${kind} (step ${r.step}, floor ${r.floor}): light 12 ${r.darkPixels} px; ${kind === 'ghost' || kind === 'followed' ? `unchanged at 128 (max ${r.unchanged})` : `light 128 ${r.litPixels} px within ${r.litDiff}`}, opacity ${r.opacity}`).join('; ')}`);
  return report;
}

// A frame depends only on its step, not on how playback reached it: on the
// boss tier's unbroken set, the boards' ghost layers at its last step, sought
// there straight from step 0, equal them played there from step 0 two steps
// a frame, pixel for pixel (every run has ended, so no ghost trail, whose
// history a seek replays only from a keyframe, is drawn).
async function replayCase(chrome, failures) {
  const [tier, set] = ['boss', 'unbroken'];
  const count = await chrome.evaluate(`ghosts.manifest.tiers.find(t => t.name === ${JSON.stringify(tier)})?.sets.find(s => s.name === ${JSON.stringify(set)})?.counts.at(-1) ?? 0`);
  if (!count) { failures.push(`replay: no ${tier}/${set} set`); return null; }
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`);
  if (!await until(chrome, `(${LOADED})(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`, 300)) { failures.push('replay: the set did not load'); return null; }
  const report = await chrome.evaluate(`(async () => {
    const g = ghosts, last = g.set.maxSteps, layers = () => g.panels.map(p => p.ghosts.getImageData(0, 0, 528, 528).data);
    g.held = true; g.stop();
    g.goTo(0); g.goTo(last);
    const sought = layers();
    g.goTo(0);
    for (let t = 2; t < last; t += 2) g.goTo(t, true);
    g.goTo(last, true);
    const played = layers();
    return sought.map((a, f) => { let n = 0; for (let i = 0; i < a.length; i += 4) if (a[i] !== played[f][i] || a[i + 1] !== played[f][i + 1] || a[i + 2] !== played[f][i + 2] || a[i + 3] !== played[f][i + 3]) n++; return n; });
  })()`);
  if (report.some(n => n)) failures.push(`replay: at the last step the ghost layers sought there and played there differ (pixels per floor ${report.join(', ')})`);
  console.log(`replay: ${tier}/${set} at its last step, sought vs played, differing pixels per floor ${report.join(', ')}`);
  return report;
}

// Fog of war and lighting (drawShade) on the loaded set: at the start, midway
// and at the end, each tile's shade on its floor's shade layer, read at the
// tile's centre, is the rule's from the viewer's state: FOG (0.9) where no
// run has had the tile in view by then (pre.reveal), else DARK (0.7) times
// 1 - light / 255 of the living runs' torch light (pb.light). A torch a run
// places brightens its tile from the step it shows, and the run's end
// darkens it again. With Fog of war off, unrevealed tiles take the light's
// shade; with Lighting off too, none. Screenshots both on at the three steps,
// in the light and dark themes.
const SHADE_STATE = `(() => { const g = ghosts, fogged = Math.round(255 * 0.9), out = { sampled: 0, mismatches: 0, fogged: 0, dimmed: 0, first: null };
  for (let f = 0; f < 9; f++) {
    const data = g.panels[f].shade.getImageData(0, 0, 528, 528).data;
    for (let cell = 0; cell < 2304; cell++) {
      const tile = f * 2304 + cell, fog = g.show.fog && g.pre.reveal[tile] > g.pb.t;
      const want = fog ? fogged : g.show.light ? Math.round(255 * 0.7 * (1 - g.pb.light[tile] / 255)) : 0;
      const got = data[((Math.floor(cell / 48) * 11 + 5) * 528 + (cell % 48) * 11 + 5) * 4 + 3];
      out.sampled++; out.fogged += fog; out.dimmed += !fog && want > 0;
      if (Math.abs(got - want) > 1) { out.mismatches++; out.first ??= { tile, got, want }; }
    }
  }
  return out; })()`;
async function shadeCase(chrome, out, failures) {
  const report = { steps: [] }, fail = message => failures.push(`shade: ${message}`);
  const at = async u => { await chrome.evaluate(`(() => { ghosts.stop(); ghosts.goTo(${u}); })()`); await sleep(150); };
  const maxSteps = await chrome.evaluate('ghosts.set.maxSteps');
  for (const [name, u] of [['start', 0], ['mid', Math.round(maxSteps / 2)], ['late', maxSteps]]) {
    await at(u);
    const g = await chrome.evaluate(SHADE_STATE);
    report.steps.push({ name, u, ...g });
    if (g.mismatches) fail(`at step ${u}, ${g.mismatches} of ${g.sampled} tiles are not shaded by the rule, first ${JSON.stringify(g.first)}`);
    for (const scheme of ['light', 'dark']) {
      await chrome.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: scheme }] });
      await sleep(200);
      await screenshot(chrome, path.join(out, `${path.basename(out)}-shade-${name}-${scheme}.png`));
    }
  }
  await chrome.send('Emulation.setEmulatedMedia', { features: [] });
  if (!report.steps[0].fogged || report.steps[0].fogged <= report.steps[2].fogged) fail(`the fog does not lift over time: ${report.steps.map(s => s.fogged).join(' -> ')} fogged tiles`);
  if (!report.steps.some(s => s.dimmed)) fail('no tile is dimmed by the light');
  // A torch: a light change on a tile no other run lights, so it brightens
  // the tile at the step it shows and the run's end takes the light away.
  report.torch = await chrome.evaluate(`(async () => { const g = ghosts, D = GhostDecode, L = g.pre.light, wait = () => new Promise(r => setTimeout(r, 60)), alpha = tile => { const c = tile % 2304;
      return g.panels[Math.floor(tile / 2304)].shade.getImageData((c % 48) * 11 + 5, Math.floor(c / 48) * 11 + 5, 1, 1).data[3]; };
    const only = (tile, e) => { for (let i = L.tileStart[tile]; i < L.tileStart[tile + 1]; i++) if (L.tileRun[i] !== e) return false; return true; };
    for (let e = 0, tries = 0; e < g.set.n && tries < 40; e++) {
      const run = L.runs[e];
      for (let k = 0; k < run.tile.length && tries < 40; k++) {
        const tile = run.tile[k], shows = D.stepOf(g.set, e, run.decision[k] + 1);
        if (shows < 1 || shows >= g.set.length[e] || !only(tile, e)) continue;
        tries++;
        g.goTo(shows - 1); await wait(); const before = [g.pb.light[tile], alpha(tile)];
        g.goTo(shows); await wait(); const lit = [g.pb.light[tile], alpha(tile)];
        g.goTo(g.set.length[e]); await wait(); const ended = [g.pb.light[tile], alpha(tile)];
        if (lit[0] > before[0] && ended[0] < lit[0] && g.pre.reveal[tile] < shows - 1) return { e, k, tile, shows, end: g.set.length[e], before, lit, ended };
      }
    }
    return null; })()`);
  const t = report.torch;
  if (!t) fail('no torch brightened a tile and went dark again with its run');
  else if (!(t.lit[1] < t.before[1] && t.ended[1] > t.lit[1])) fail(`a torch's light does not show in the shade: ${JSON.stringify(t)}`);
  // The toggles.
  await at(Math.round(maxSteps / 4));
  for (const off of ['show-fog', 'show-light']) {
    await chrome.evaluate(`document.getElementById('${off}').click()`);
    await sleep(150);
    const g = await chrome.evaluate(SHADE_STATE);
    report[off] = g;
    if (g.mismatches || (off === 'show-light' && g.dimmed)) fail(`with ${off} off the shade is wrong: ${JSON.stringify(g)}`);
  }
  console.log(`shade: ${report.steps.map(s => `${s.name} (step ${s.u}) ${s.fogged} fogged, ${s.dimmed} dimmed, ${s.mismatches} wrong`).join('; ')}; torch ${t ? `of run ${t.e} at step ${t.shows}: light ${t.before[0]} -> ${t.lit[0]} -> ${t.ended[0]} at its end, alpha ${t.before[1]} -> ${t.lit[1]} -> ${t.ended[1]}` : 'none'}`);
  return report;
}

// Items draw as the sheet's item sprites, fixed here apart from the viewer's
// rule: the game's ItemType (game/state.py: torch 1, ladder down 2, ladder up
// 3) and the sheet (sprites.png: 43 a torch, 44 a ladder down, 45 a ladder
// up). Floor 0's ladder down and floor 1's ladder up on the base maps, and a
// torch some run placed on the map-change layer, must each equal the same tile
// drawn here from its sprite, pixel for pixel.
const ITEM_SPRITES = { 1: 43, 2: 44, 3: 45 };
async function itemsCase(chrome, failures) {
  const CHECK = `(async () => { const g = ghosts, D = GhostDecode, T = 11, sheet = g.sprites, out = [], sprite = ${JSON.stringify(ITEM_SPRITES)};
    const expect = (block, sprite, alpha, base) => { const c = Object.assign(document.createElement('canvas'), { width: T, height: T }), x = c.getContext('2d');
      x.imageSmoothingEnabled = false; if (base) { x.fillStyle = '#070d11'; x.fillRect(0, 0, T, T); } x.globalAlpha = alpha;
      const tile = id => x.drawImage(sheet, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, 0, 0, T, T);
      if (block) tile(block); tile(sprite); x.globalAlpha = 1;
      return x.getImageData(0, 0, T, T).data; };
    const same = (canvas, cell, want) => { const got = canvas.getContext('2d').getImageData((cell % 48) * T, Math.floor(cell / 48) * T, T, T).data;
      let diff = 0; for (let i = 0; i < got.length; i++) diff = Math.max(diff, Math.abs(got[i] - want[i])); return diff; };
    for (const [floor, cell, item] of [[0, g.world.down[0], 2], [1, g.world.up[1], 3]]) {
      const tile = floor * 2304 + cell;
      out.push({ what: item === 2 ? 'ladder down' : 'ladder up', floor, item: g.world.item[tile], sprite: sprite[item],
        diff: same(g.panels[floor].base.canvas, cell, expect(g.world.block[tile], sprite[item], 1, true)) });
    }
    let torch = -1;
    for (const u of [0.3, 0.5, 0.7, 0.9].map(f => Math.round(f * g.set.maxSteps))) {
      g.goTo(u); await new Promise(r => setTimeout(r, 50));
      for (let tile = 0; tile < D.TILES && torch < 0; tile++) if (g.pb.tileCount[tile] && (g.pb.tileCode[tile] - 1) % D.ITEMS === 1) torch = tile;
      if (torch >= 0) break;
    }
    if (torch >= 0) { const code = g.pb.tileCode[torch] - 1, k = g.pb.tileCount[torch], floor = Math.floor(torch / 2304);
      out.push({ what: 'torch', floor, item: code % D.ITEMS, sprite: sprite[1], runs: k,
        diff: same(g.panels[floor].layer.canvas, torch % 2304, expect(Math.floor(code / D.ITEMS), sprite[1], g.changeAlpha[k], false)) }); }
    return out; })()`;
  const tiles = await chrome.evaluate(CHECK);
  for (const t of tiles) if (t.diff > 2) failures.push(`items: the ${t.what} on floor ${t.floor} does not draw as sprite ${t.sprite} (max channel difference ${t.diff})`);
  if (!tiles.some(t => t.what === 'torch')) failures.push('items: no run placed a torch to check');
  if (tiles.some(t => t.what !== 'torch' && t.item !== (t.what === 'ladder down' ? 2 : 3))) failures.push(`items: the world's ladders are not where its ladder table says: ${JSON.stringify(tiles)}`);
  console.log(`items: ${tiles.map(t => `${t.what} on floor ${t.floor} as sprite ${t.sprite}, max difference ${t.diff}`).join('; ')}`);
  return tiles;
}

// "Show sleep" on the unbroken set (else the wins set), where the manifest
// has sleep maps: the set plays its sleep_map's steps; the pinned run still
// shows every decision; through one of its sleeps the run holds its tile and
// decision while the steps pass the sleep's samples, and the policy view
// draws another frame than the decision's. Screenshots it mid-sleep in the
// Follow layout.
async function sleepCase(chrome, out, failures) {
  const found = await chrome.evaluate(`(() => { const t = ghosts.manifest.tiers.find(t => t.sets.some(s => s.sleep_map)); if (!t) return null;
    const s = t.sets.find(s => s.name === 'unbroken') ?? t.sets.find(s => s.sleep_map); return [t.name, s.name, s.counts.at(-1)]; })()`);
  if (!found) return null;
  const [tier, set, count] = found, loaded = `(${LOADED})(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`;
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'follow'; s.dispatchEvent(new Event('change')); })()`);
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, ${JSON.stringify(set)}, ${count})`);
  await until(chrome, loaded, 300);
  await chrome.evaluate(`(() => { const s = document.getElementById('show-sleep'); s.checked = true; s.dispatchEvent(new Event('change')); })()`);
  if (!await until(chrome, `${loaded} && ghosts.set.sleepView`, 300)) {
    failures.push(`sleep: the view that shows sleep did not load: ${await chrome.evaluate(`document.getElementById('status').textContent`)}`);
    return null;
  }
  const PANEL = `(() => { const c = document.querySelector('#policy-panel canvas'); if (!c || document.getElementById('policy-panel').hidden) return null;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 7) h = (h * 31 + d[i]) >>> 0; return h; })()`;
  const state = await chrome.evaluate(`(() => { const g = ghosts, s = g.set, e = g.selected, D = GhostDecode, steps = s.length[e];
    let at = -1; for (let u = 0; u < steps && at < 0; u++) if (D.sampleAt(s, e, u) === 2) at = u;
    const decisions = new Set(Array.from({ length: steps }, (_, u) => D.decisionAt(s, e, u)));
    return { maxSteps: s.maxSteps, mapSteps: g.collection.sleep_map.steps, decisions: s.decisions[e], every: decisions.size === s.decisions[e], at }; })()`);
  const report = { tier, set, ...state, samples: [] };
  if (state.maxSteps !== state.mapSteps) failures.push(`sleep: the view plays ${state.maxSteps} steps, not its sleep_map's ${state.mapSteps}`);
  if (!state.every) failures.push('sleep: the pinned run skips a decision when sleep shows');
  if (state.at < 0) failures.push('sleep: the pinned run never sleeps through two samples');
  else {
    for (const u of [state.at - 2, state.at - 1, state.at]) {
      await chrome.evaluate(`ghosts.goTo(${u})`);
      await until(chrome, WINDOWS, 30);
      await sleep(300);
      report.samples.push({ u, ...(await chrome.evaluate(`({ decision: ghosts.pb.decision[ghosts.selected], sample: GhostDecode.sampleAt(ghosts.set, ghosts.selected, ghosts.pb.t), tile: ghosts.pb.cur.row[ghosts.selected] * 48 + ghosts.pb.cur.col[ghosts.selected], info: document.getElementById('episode-info').textContent })`)), panel: await chrome.evaluate(PANEL) });
    }
    const [before, , asleep] = report.samples;
    if (asleep.decision !== before.decision + 0 && asleep.decision !== report.samples[1].decision) failures.push(`sleep: the run left its decision mid-sleep: ${JSON.stringify(report.samples)}`);
    if (asleep.tile !== report.samples[1].tile || !asleep.info.includes('asleep')) failures.push(`sleep: the run moved or does not read as asleep mid-sleep: ${JSON.stringify(report.samples)}`);
    if (asleep.panel === null || asleep.panel === before.panel) failures.push(`sleep: the policy view did not draw a sleep frame: ${JSON.stringify(report.samples)}`);
    await screenshot(chrome, path.join(out, `${path.basename(out)}-sleep.png`));
  }
  console.log(`sleep: ${tier}/${set} ${state.maxSteps} steps (sleep_map ${state.mapSteps}); pinned run every decision ${state.every}; mid-sleep ${JSON.stringify(report.samples.map(s => [s.u, s.decision, s.sample]))}`);
  await chrome.evaluate(`(() => { const s = document.getElementById('show-sleep'); s.checked = false; s.dispatchEvent(new Event('change')); })()`);
  return report;
}

// The wins view, where the manifest has one: every win of the last tier in
// steps of at most its time map's budget, following its pinned run, which
// shows every decision and wins at its own length; the quiet-stretch toggle
// gone; the policy view, when shipped, on the followed run's decision. Two
// rows, screenshotted early, midway and at the end.
async function winsView(chrome, out, failures) {
  const found = await chrome.evaluate(`(() => { const t = ghosts.manifest.tiers.find(t => t.sets.some(s => s.time_map)); return t ? [t.name, t.sets.find(s => s.time_map).counts.at(-1)] : null; })()`);
  if (!found) return null;
  const [tier, count] = found, report = { tier, count, seeks: [] };
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'two-rows'; s.dispatchEvent(new Event('change')); })()`);
  await chrome.evaluate(`ghosts.load(${JSON.stringify(tier)}, 'wins', ${count})`);
  if (!await until(chrome, `(${LOADED})(${JSON.stringify(tier)}, 'wins', ${count})`, 300)) {
    failures.push(`wins view did not load: ${await chrome.evaluate(`document.getElementById('status').textContent`)}`);
    return report;
  }
  const STATE = `(() => { const g = ghosts, e = g.selected, s = g.set; return { u: g.pb.t, steps: s.maxSteps, budget: g.collection.time_map.rule.steps, selected: e, seed: s.seeds[e],
    want: document.getElementById('page').dataset.followSeedWins, unbroken: s.shown[e] === null && s.length[e] === s.decisions[e], decisions: s.decisions[e], decision: g.pb.decision[e],
    alive: g.pb.alive(e), panel: document.getElementById('policy-panel').hidden ? null : g.policyShown, policy: !!g.policy, skipHidden: document.getElementById('skip-quiet').closest('label').hidden,
    follow: document.getElementById('follow').value, won: g.won }; })()`;
  for (const [fraction, name] of [[0.1, 'early'], [0.5, 'mid'], [1, 'end']]) {
    await seekTo(chrome, fraction);
    await sleep(300);
    const state = await chrome.evaluate(STATE);
    report.seeks.push(state);
    await screenshot(chrome, path.join(out, `${path.basename(out)}-wins-${name}.png`));
  }
  const [first, , last] = report.seeks;
  if (first.steps > first.budget) failures.push(`wins view plays ${first.steps} steps, over its ${first.budget}-step budget`);
  if (first.follow !== 'pinned' || (first.want && first.seed !== first.want) || !first.unbroken) failures.push(`wins view does not follow its pinned unbroken run: ${JSON.stringify(first)}`);
  if (report.seeks.some(s => s.alive && s.decision !== s.u)) failures.push(`the pinned run skips a decision: ${JSON.stringify(report.seeks.map(s => [s.u, s.decision]))}`);
  if (last.alive || last.decision !== last.decisions) failures.push(`the pinned run has not ended at the end: ${JSON.stringify(last)}`);
  if (!first.skipHidden) failures.push('wins view shows the quiet-stretch toggle');
  if (first.policy && report.seeks.some(s => s.panel !== s.decision)) failures.push(`the policy view left the followed run's decision: ${JSON.stringify(report.seeks.map(s => [s.decision, s.panel]))}`);
  console.log(`wins view: ${tier} ${count} runs in ${first.steps} steps, follows seed ${first.seed} (${first.decisions} decisions, unbroken ${first.unbroken}), policy view ${first.policy ? 'in step' : 'absent'}; ${last.won} won at the end`);
  await chrome.evaluate(`(() => { const s = document.getElementById('layout'); s.value = 'grid'; s.dispatchEvent(new Event('change')); })()`);
  return report;
}

async function phone(chrome, out, failures) {
  // The phone view also shows the light theme; the desktop screenshots show Chrome's default.
  await chrome.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });
  await chrome.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await sleep(400);
  const layout = await chrome.evaluate(`({ scrollWidth: document.documentElement.scrollWidth, innerWidth, columns: getComputedStyle(document.getElementById('floors')).gridTemplateColumns.split(' ').length })`);
  if (layout.scrollWidth > layout.innerWidth) failures.push(`phone width scrolls sideways: ${layout.scrollWidth} > ${layout.innerWidth}`);
  if (layout.columns !== 1) failures.push(`phone width shows ${layout.columns} floor columns`);
  await screenshot(chrome, path.join(out, `${path.basename(out)}-phone.png`));
  await chrome.send('Emulation.setEmulatedMedia', { features: [] });
  await chrome.send('Emulation.clearDeviceMetricsOverride');
  return layout;
}

async function main() {
  const args = process.argv.slice(2);
  let binary = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', only = null;
  if (args[0] === '--chrome') binary = args.splice(0, 2)[1];
  if (args[0] === '--only') only = args.splice(0, 2)[1].split(',');
  const [out, site] = args;
  if (!binary || !out || !site || args.length !== 2) {
    console.error('Usage: node render_check.mjs [--chrome PATH] [--only CASES] OUT_DIR SITE_DIR');
    process.exit(1);
  }
  fs.mkdirSync(out, { recursive: true });
  const server = await serve(path.resolve(site)), chrome = await launch(binary), failures = [];
  let report;
  try {
    report = await check(chrome, `http://127.0.0.1:${server.address().port}/`, out, failures, only);
  } finally {
    Object.assign(report ??= {}, { errors: chrome.errors, failures });
    fs.writeFileSync(path.join(out, 'render-check.json'), JSON.stringify(report, null, 1) + '\n');
    await chrome.close();
    server.close();
  }
  console.log(`failures ${failures.length}; page errors ${chrome.errors.length}`);
  for (const line of [...failures, ...chrome.errors]) console.log(`  ${line}`);
  process.exit(failures.length || chrome.errors.length ? 1 : 0);
}

await main();
