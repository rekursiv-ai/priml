// Load every game of pages built by games.mjs in headless Chrome and check each renders.
//
//   node render_check.mjs [--chrome PATH] OUT_DIR PAGE.html...
//
// For each page: open it from file://, wait until #status leaves "Loading", then
// for every entry of #game-picker select it, wait until #status reports
// "Loaded", seek to the first, middle and last decision, and count distinct
// colours on the #game canvas at each (a blank or failed board has one or two),
// in every view the game offers: the policy view, and an exact game's full map.
// Writes OUT_DIR/render-check.json and a screenshot of each page's first game.
// Exits 1 when a page never reports "Loaded" on opening, has no games, throws
// or logs an error, or when any game fails to load or draws a blank board.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function launch(binary) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'viewer-check-'));
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
  const pending = new Map();
  const errors = [];
  let id = 0;
  ws.onmessage = event => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
    else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map(a => a.value ?? a.description).join(' '));
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const n = ++id;
    pending.set(n, m => (m.error ? reject(Error(`${method}: ${m.error.message}`)) : resolve(m.result)));
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
  return { send, evaluate, close, errors };
}

async function until(chrome, expression, seconds) {
  for (const t0 = Date.now(); Date.now() - t0 < seconds * 1000; await sleep(50)) {
    if (await chrome.evaluate(expression)) return true;
  }
  return false;
}

const COLOURS = `(() => { const c = document.getElementById('game'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const seen = new Set(); for (let i = 0; i < d.length; i += 16) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]); return seen.size; })()`;

async function seekColours(chrome, fraction) {
  await chrome.evaluate(`(() => { const s = document.getElementById('seek'); s.value = String(Math.round(Number(s.max) * ${fraction}));
    s.dispatchEvent(new Event('input', { bubbles: true })); s.dispatchEvent(new Event('change', { bubbles: true })); })()`);
  await sleep(120);
  return chrome.evaluate(COLOURS);
}

async function main() {
  const args = process.argv.slice(2);
  let binary = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  if (args[0] === '--chrome') binary = args.splice(0, 2)[1];
  const [out, ...pages] = args;
  if (!binary || !out || !pages.length) {
    console.error('Usage: node render_check.mjs [--chrome PATH] OUT_DIR PAGE.html...');
    process.exit(1);
  }
  fs.mkdirSync(out, { recursive: true });
  const chrome = await launch(binary);
  const report = { pages: [] };
  let failures = 0;
  try {
    for (const page of pages) {
      // The location check keeps the previous page's status from passing for this one.
      const url = pathToFileURL(path.resolve(page)).href, onPage = `location.href === ${JSON.stringify(url)}`;
      await chrome.send('Page.navigate', { url });
      await until(chrome, `${onPage} && !(document.getElementById('status')?.textContent ?? 'Loading').startsWith('Loading')`, 120);
      const opened = await chrome.evaluate(`${onPage} && !!document.getElementById('status')?.textContent.startsWith('Loaded')`);
      const count = await chrome.evaluate(`document.getElementById('game-picker')?.options.length ?? 0`);
      const games = [];
      for (let i = 0; i < count; i++) {
        await chrome.evaluate(`(() => { const p = document.getElementById('game-picker'); p.value = '${i}'; p.dispatchEvent(new Event('change')); })()`);
        const loaded = await until(chrome, `(() => { const s = document.getElementById('status').textContent; return s.startsWith('Loaded') || s.startsWith('Could not'); })()`, 60);
        const status = await chrome.evaluate(`document.getElementById('status').textContent`);
        const decisions = await chrome.evaluate(`Number(document.getElementById('seek').max)`);
        // A model game hides the map tab; the view checked last, the map, stays open.
        const views = await chrome.evaluate(`document.getElementById('map-tab').hidden`) ? ['policy'] : ['policy', 'map'];
        const colours = {};
        for (const view of views) {
          await chrome.evaluate(`document.getElementById('${view}-tab').click()`);
          colours[view] = [await seekColours(chrome, 0), await seekColours(chrome, 0.5), await seekColours(chrome, 1)];
        }
        const ok = loaded && status.startsWith('Loaded') && decisions > 0 && Object.values(colours).flat().every(c => c > 8);
        failures += ok ? 0 : 1;
        games.push({ index: i, title: await chrome.evaluate(`document.getElementById('game-picker').selectedOptions[0].text`), ok, status, decisions, colours });
        if (i === 0) {
          await seekColours(chrome, 0.5);
          const { data } = await chrome.send('Page.captureScreenshot', { format: 'png' });
          fs.writeFileSync(path.join(out, `${path.basename(page, '.html')}-game0.png`), Buffer.from(data, 'base64'));
        }
      }
      // A page that never loaded, or offered nothing to check, fails even when every game it offered rendered.
      failures += opened && games.length ? 0 : 1;
      const rendered = games.filter(g => g.ok).length, decisions = games.map(g => g.decisions);
      report.pages.push({ page, opened, games: games.length, ok: rendered, entries: games });
      console.log(`${path.basename(page)}: ${rendered}/${games.length} games render; decisions ${games.length ? `${Math.min(...decisions)}-${Math.max(...decisions)}` : 'none'}${opened ? '' : '; the page never reported Loaded'}`);
    }
  } finally {
    report.errors = chrome.errors;
    report.failures = failures;
    fs.writeFileSync(path.join(out, 'render-check.json'), JSON.stringify(report, null, 1) + '\n');
    await chrome.close();
  }
  console.log(`failures ${failures}; page errors ${report.errors.length}`);
  process.exit(failures || report.errors.length ? 1 : 0);
}

await main();
