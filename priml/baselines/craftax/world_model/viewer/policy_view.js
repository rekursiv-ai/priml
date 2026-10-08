'use strict';
// One exact game as its policy saw it, for a page that shows it beside other
// figures: before each decision, the 9x11 window and the HUD (hud.js) on one
// canvas in the game's own colours, scaled by a whole number of device pixels
// to fit the container's width. The bundle is policy_view.py's, served gzipped,
// as base64 text of the gzip, or already decoded by the server.
//
// games.mjs panel wraps hud.js and this file in one function: the script it
// writes defines CraftaxPolicyView and no other name, and loads nothing until
// mount, which loads the bundle and the sprite sheet and nothing else. The
// bundle comes through one function, loadBundle(url), which a page whose host
// fetch() cannot reach passes in place of fetchBundle; it resolves to the
// bundle file's bytes (an ArrayBuffer or a Uint8Array) or its base64 text.
//
//   const view = await CraftaxPolicyView.mount(container, { bundleUrl, spritesUrl });
//   view.show(t);  // the frame before decision t; from t = view.decisions on, the last
//   view.show(t, k);  // decision t's sleep's k-th frame (1-based), with sleep frames
//   view.caption.textContent = '…';  // the line under the canvas
//   view.fit();  // size the canvas again, after the page moved what it fits to
//
// Three layouts. 'column' (the default) stacks the window over the HUD,
// meters over items, and fits the container's width at a whole number of
// device pixels per game pixel. 'row' sets them side by side, the window,
// then the meters, then the items in columns; 'side' sets beside the window
// one narrow column, meters stacked over a five-wide grid of items. Both fit
// the height that height() gives (CSS pixels): the canvas is drawn at the
// whole number of device pixels per game pixel that covers it and shown at
// exactly that height, scaled down nearest-neighbour when it is not a whole
// number (or less tall, if that height would make it wider than the
// container). A row or side panel narrower than wrapBelow CSS pixels (600
// by default) is too small to read, so the panel falls back to the column
// layout there; pass 0 to keep it.
//
// Sleep frames: a bundle written from a replay with sleep frames has
// policy-sleep.bin.gz beside it, its manifest's sleeps listing each sleep's
// decision, ticks and first frame there. mount(..., { sleepUrl, sleeps })
// loads it (through loadBundle too); show(t, k) then draws the k-th frame of
// decision t's sleep, the world as it stood every sleep_stride ticks of it.
//
// A won game: mount(..., { beatenFrom }) names the first frame with the
// necromancer beaten, a won game's last (its decisions, as the ghosts' win
// ends at the first DEFEAT_NECROMANCER); from it on the window crosses out
// the necromancer, which stays on the map (hud.js drawWindow).

// policy_view.py's record: exact.py's frame up to its map, same offsets
// (HUD_END bytes), then from schema v2 on its view's 99 projectile facings.
// mount tells a v2 bundle from a v1 one by which record size its final
// frame's marks sit at.
const HUD_END = 885, RECORDS = [HUD_END + 99, HUD_END];
// Each layout's canvas in game pixels: the window's 16-pixel tiles at the top
// left; the HUD's area, cleared before it is drawn; its five meters, each
// meterStep below the last from meterY: a label at labelX, a value ending at
// valueEnd and a bar at barX, barDy below the label, barWidth long; and its
// items in rows of itemColumns from (itemX, itemY), at most items of them.
const METER = { labelX: 3, valueEnd: 49, barX: 52, barDy: 1, barWidth: 121 };
const COLUMN = { width: 176, height: 236, hud: [0, 144, 176, 92], ...METER, meterX: 0, meterY: 148, meterStep: 7, itemX: 0, itemY: 186, itemColumns: 11, items: 33 };
const ROW = { width: 425, height: 144, hud: [176, 0, 249, 144], ...METER, meterX: 182, meterY: 12, meterStep: 27, itemX: 361, itemY: 0, itemColumns: 4, items: 36 };
const SIDE = {
  width: 260, height: 144, hud: [176, 0, 84, 144], labelX: 2, valueEnd: 83, barX: 2, barDy: 7, barWidth: 80,
  meterX: 176, meterY: 3, meterStep: 12, itemX: 178, itemY: 64, itemColumns: 5, items: 25,
};
// A 3x5 pixel font: per glyph, five rows of three bits, the high bit leftmost.
const GLYPHS = {
  0: '75557', 1: '26227', 2: '71747', 3: '71317', 4: '55711', 5: '74717', 6: '74757', 7: '71111',
  8: '75757', 9: '75717', '.': '00002', '-': '00700', A: '25755', D: '65556', E: '74647', F: '74644',
  G: '34553', H: '55755', I: '72227', K: '55655', L: '44447', M: '57755', N: '57775', O: '25552',
  R: '65655', T: '72222', Y: '55222',
};

async function mount(container, { bundleUrl, spritesUrl, loadBundle = fetchBundle, layout = 'column', height = null, wrapBelow = 600, sleepUrl = null, sleeps = [], beatenFrom = null }) {
  const sprites = new Image();
  sprites.src = spritesUrl;
  const [bytes, slept] = await Promise.all([loadBundle(bundleUrl).then(unpack), sleepUrl ? loadBundle(sleepUrl).then(unpack) : null, sprites.decode()]);
  const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // The record size whose final frame is marked (action 255, its step the
  // decision count).
  const RECORD = RECORDS.find(size => {
    const decisions = bytes.length / size - 1;
    return Number.isInteger(decisions) && decisions >= 1 && bytes[decisions * size + 12] === 255 && data.getUint32(decisions * size, true) === decisions;
  });
  if (!RECORD) throw Error(`${bundleUrl} holds no policy-view frames (${bytes.length} bytes).`);
  const decisions = bytes.length / RECORD - 1, facings = RECORD > HUD_END ? HUD_END : -1;
  // Per sleep decision, its first frame in slept and how many it has.
  const sleepAt = new Map(sleeps.map(([decision, , first], k) => [decision, [first, (sleeps[k + 1]?.[2] ?? (slept?.length ?? 0) / RECORD) - first]]));
  if (slept && (slept.length % RECORD || sleeps.some(([, , first]) => first * RECORD > slept.length))) throw Error(`${sleepUrl} does not hold the sleep frames its manifest lists.`);
  const sleptData = slept && new DataView(slept.buffer, slept.byteOffset, slept.byteLength);
  const figure = document.createElement('figure'), canvas = document.createElement('canvas'), caption = document.createElement('figcaption');
  figure.className = 'cpv';
  figure.style.cssText = 'margin:0;min-width:0';
  canvas.className = 'cpv-canvas';
  canvas.style.cssText = 'display:block;image-rendering:pixelated';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'What the policy saw: its 9 by 11 window of the floor and its HUD');
  caption.className = 'cpv-caption';
  caption.style.cssText = 'margin:6px 0 0;font-size:13px;line-height:1.4;color:inherit';
  figure.append(canvas, caption);
  container.append(figure);
  const ctx = canvas.getContext('2d');
  let geometry = COLUMN, scale = 0, size = '', shown = '', hud = null, again = [0, 0];
  // hud holds the frame whose HUD is on the canvas: its bytes and offset.
  const show = (t, k = 0) => {
    const i = Math.max(0, Math.min(decisions, Math.floor(t) || 0)), sleep = k > 0 && sleepAt.get(i);
    const [source, view, base, key] = sleep && k <= sleep[1] ? [slept, sleptData, (sleep[0] + k - 1) * RECORD, `${i}/${k}`] : [bytes, data, i * RECORD, `${i}`];
    again = [t, k];
    if (key === shown) return;
    drawWindow(ctx, sprites, source, base + 34, source[base + 17], source[base + 18], 16, beatenFrom !== null && i >= beatenFrom, facings < 0 ? -1 : base + facings);
    if (!hud || hud[0] !== source || hudChanged(source, hud[1], base)) {
      drawHud(ctx, sprites, view, base, geometry);
      hud = [source, base];
    }
    shown = key;
  };
  // A new size clears the canvas, so the frame on it is drawn again.
  const fit = () => {
    const ratio = window.devicePixelRatio || 1, row = layout !== 'column' && height && figure.clientWidth >= wrapBelow;
    const g = row ? (layout === 'side' ? SIDE : ROW) : COLUMN, tall = row ? Math.min(height(), figure.clientWidth * g.height / g.width) : 0;
    const next = Math.max(1, row ? Math.ceil(tall * ratio / g.height) : Math.floor(figure.clientWidth * ratio / g.width));
    const [cssWidth, cssHeight] = row ? [tall * g.width / g.height, tall] : [g.width * next / ratio, g.height * next / ratio];
    const key = `${cssWidth}x${cssHeight}`;
    if (g === geometry && next === scale && key === size) return;
    [geometry, scale, size] = [g, next, key];
    canvas.width = g.width * scale;
    canvas.height = g.height * scale;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    [shown, hud] = ['', null];
    show(...again);
  };
  fit();
  new ResizeObserver(fit).observe(figure);
  return Object.freeze({ decisions, caption, show, fit });
}

// The meters and held items of the frame at byte base, where the layout g puts
// the HUD.
function drawHud(ctx, sprites, data, base, g) {
  ctx.fillStyle = '#0b1519';
  ctx.fillRect(...g.hud);
  hudMeters(data, base).forEach(([label, value, max, colour], k) => {
    const x = g.meterX, y = g.meterY + g.meterStep * k, number = label === 'Health' ? value.toFixed(1) : String(value);
    const bar = x + g.barX, width = g.barWidth, length = Math.round(width * Math.max(0, Math.min(1, value / Math.max(1, max))));
    drawText(ctx, label.toUpperCase(), x + g.labelX, y, '#9fb2ba');
    drawText(ctx, number, x + g.valueEnd - 4 * number.length, y, '#f4f7f5');
    ctx.fillStyle = '#24343b';
    ctx.fillRect(bar, y + g.barDy, width, 3);
    ctx.fillStyle = colour;
    ctx.fillRect(bar, y + g.barDy, length, 3);
    ctx.fillStyle = '#0b1519';
    for (let unit = 1; unit < max; unit++) ctx.fillRect(bar + Math.round(width * unit / max), y + g.barDy, 1, 3);
  });
  hudItems(data, base).slice(0, g.items).forEach(([, icon, count], k) => {
    const x = g.itemX + 16 * (k % g.itemColumns), y = g.itemY + 16 * Math.floor(k / g.itemColumns), width = 4 * count.length - 1;
    ctx.drawImage(sprites, (icon % 16) * 16, Math.floor(icon / 16) * 16, 16, 16, x, y, 16, 16);
    if (!count) return;
    ctx.fillStyle = '#070d11';
    ctx.fillRect(x + 14 - width, y + 9, width + 2, 7);
    drawText(ctx, count, x + 15 - width, y + 10, '#f4f7f5');
  });
}

function drawText(ctx, text, x, y, colour) {
  ctx.fillStyle = colour;
  [...text].forEach((glyph, i) => {
    for (let row = 0; row < 5; row++) {
      const bits = Number(GLYPHS[glyph][row]);
      for (let col = 0; col < 3; col++) if (bits >> (2 - col) & 1) ctx.fillRect(x + 4 * i + col, y + row, 1, 1);
    }
  });
}

// Whether the frames at bytes a and b differ in what drawHud reads: health,
// mana, and every field from the meters' maxima on.
function hudChanged(bytes, a, b) {
  return [[22, 26], [30, 32], [826, HUD_END]].some(([start, stop]) => {
    for (let k = start; k < stop; k++) if (bytes[a + k] !== bytes[b + k]) return true;
    return false;
  });
}

async function fetchBundle(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`${url} did not load (HTTP ${response.status}).`);
  return response.arrayBuffer();
}

// The frames a loaded bundle holds: gzip, base64 text of the gzip, or the
// frames themselves, which a server that decodes gzip hands over.
async function unpack(loaded) {
  const text = typeof loaded === 'string' ? loaded : '';
  let bytes = text ? new TextEncoder().encode(text) : ArrayBuffer.isView(loaded) ? new Uint8Array(loaded.buffer, loaded.byteOffset, loaded.byteLength) : new Uint8Array(loaded);
  // 'H4sI' is the base64 of gzip's first three bytes, 1f 8b 08.
  if (String.fromCharCode(...bytes.subarray(0, 4)) === 'H4sI') bytes = fromBase64(text || new TextDecoder().decode(bytes));
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function fromBase64(text) {
  text = text.trim();
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(text);
  const binary = atob(text), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

window.CraftaxPolicyView = Object.freeze({ mount });
