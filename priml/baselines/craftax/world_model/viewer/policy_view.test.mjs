// Runs the panel script games.mjs panel writes on a fake page: a canvas whose
// context records what it draws, and fetch, Image and ResizeObserver that
// record what they are asked for.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import zlib from 'node:zlib';

const here = path.dirname(fileURLToPath(import.meta.url));
const RECORD = 885;
let root, script;

before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'craftax-policy-view-test-'));
  const output = path.join(root, 'policy_view.js');
  const built = spawnSync(process.execPath, [path.join(here, 'games.mjs'), 'panel', '--output', output], { encoding: 'utf8' });
  assert.equal(built.status, 0, built.stderr);
  script = fs.readFileSync(output, 'utf8');
});

after(() => fs.rmSync(root, { recursive: true, force: true }));

// Five frames, so four decisions. Every cell is lit and holds the frame's block
// but the top-left one, which is dark, and each frame adds its own items and
// creatures as [row, col, channel, value], so a drawing names its frame.
const blocks = [2, 3, 4, 5, 6];
const extras = [
  [],
  [[1, 1, 1, 2], [2, 3, 3, 1], [0, 2, 1, 3], [0, 3, 1, 4], [0, 4, 1, 5]],
  [[3, 4, 7, 1], [6, 7, 4, 1], [8, 10, 5, 2]],
  [[7, 2, 6, 2]],
  [[5, 5, 3, 3]],
];
// The sprites those draw: a torch and a zombie; a player arrow, a cow and a
// gnome archer; a mob fireball; an orc. Then the player, by facing or asleep.
// Frame 1 also holds, as the observation stores items (the item plus one), a
// ladder down, a ladder up and a blocked ladder down: sprites 44, 45 and 46,
// as the torch is 43 (hud.js itemSprite; game/state.py ItemType).
const sprites = [{}, { '1,1': 43, '2,3': 80, '0,2': 44, '0,3': 45, '0,4': 46 }, { '3,4': 50, '6,7': 88, '8,10': 92 }, { '7,2': 99 }, { '5,5': 82 }];
const players = [37, 39, 40, 41, 37];

function frames() {
  const bytes = Buffer.alloc(5 * RECORD);
  for (let t = 0; t < 5; t++) {
    const base = t * RECORD;
    bytes.writeUInt32LE(t, base);
    bytes[base + 12] = t < 4 ? 1 : 255;
    bytes[base + 17] = [4, 1, 2, 3, 4][t];
    bytes[base + 18] = t === 3 ? 1 : 0;
    for (let cell = 1; cell < 99; cell++) bytes.set([blocks[t], 1, 1], base + 34 + cell * 8);
    for (const [row, col, channel, value] of extras[t]) bytes[base + 34 + (row * 11 + col) * 8 + channel] = value;
    bytes.writeFloatLE([9, 7.25, 7.25, 3.5, 3.5][t], base + 22);
    bytes.writeInt16LE(t < 3 ? 5 : 2, base + 30);
    bytes.set([9, 9, 9, ...(t < 3 ? [9, 8, 7] : [6, 5, 4])], base + 826);
    const held = [{}, { 0: 5 }, { 0: 12, 6: 2, 9: 3 }][Math.min(t, 2)];
    for (const [slot, count] of Object.entries(held)) bytes.writeUInt16LE(count, base + 833 + 2 * Number(slot));
    bytes[base + 881] = t >= 3 ? 1 : 0;
  }
  return bytes;
}

// A canvas context that records what it draws. save, restore, translate and
// transform keep the current matrix [a, b, c, d, e, f] (setTransform, the
// panel's device-pixel scale, is only recorded, so positions stay in game
// pixels); an image is recorded with its destination's bounding box under
// that matrix, and the matrix and destination themselves.
function recorder() {
  const ctx = { ops: [], fillStyle: '', imageSmoothingEnabled: true, matrix: [1, 0, 0, 1, 0, 0], stack: [] };
  const at = ([a, b, c, d, e, f], x, y) => [a * x + c * y + e, b * x + d * y + f];
  ctx.save = () => ctx.stack.push(ctx.matrix);
  ctx.restore = () => { ctx.matrix = ctx.stack.pop(); };
  ctx.transform = (a, b, c, d, e, f) => {
    const [A, B, C, D, E, F] = ctx.matrix;
    ctx.matrix = [A * a + C * b, B * a + D * b, A * c + C * d, B * c + D * d, A * e + C * f + E, B * e + D * f + F];
  };
  ctx.translate = (x, y) => ctx.transform(1, 0, 0, 1, x, y);
  // A sheet sprite at (sx, sy) of 16-pixel cells is sprite sy + sx / 16.
  ctx.drawImage = (image, sx, sy, sw, sh, dx, dy, dw, dh) => {
    const corners = [[dx, dy], [dx + dw, dy], [dx, dy + dh], [dx + dw, dy + dh]].map(([x, y]) => at(ctx.matrix, x, y));
    const [left, top] = [0, 1].map(k => Math.min(...corners.map(p => p[k]))), [right, bottom] = [0, 1].map(k => Math.max(...corners.map(p => p[k])));
    ctx.ops.push(['image', sy + sx / 16, left, top, right - left, bottom - top, ctx.matrix, [dx, dy, dw, dh]]);
  };
  ctx.fillRect = (x, y, w, h) => ctx.ops.push(['rect', ctx.fillStyle, x, y, w, h]);
  ctx.setTransform = (...matrix) => ctx.ops.push(['transform', ...matrix]);
  return ctx;
}

function element(tag, width) {
  const node = { tag, children: [], style: {}, attributes: {}, className: '', textContent: '' };
  node.append = (...children) => node.children.push(...children);
  node.setAttribute = (name, value) => { node.attributes[name] = value; };
  Object.defineProperty(node, 'clientWidth', { get: width });
  if (tag === 'canvas') {
    const ctx = recorder();
    node.getContext = () => ctx;
  }
  return node;
}

function page(files = {}) {
  const log = { fetched: [], images: [], created: [], observers: [] };
  const context = {
    layoutWidth: 700, devicePixelRatio: 2, Blob, Response, DecompressionStream, TextDecoder, TextEncoder, atob,
    fetch: async url => {
      log.fetched.push(url);
      return url in files ? new Response(files[url]) : new Response('', { status: 404 });
    },
    Image: class { set src(url) { log.images.push(url); } decode() { return Promise.resolve(); } },
    ResizeObserver: class { constructor(callback) { log.observers.push(callback); } observe() {} },
    document: { createElement: tag => { log.created.push(tag); return element(tag, () => context.layoutWidth); } },
  };
  context.window = context;
  vm.createContext(context);
  return { context, log };
}

async function mounted(served = zlib.gzipSync(frames()), bundleUrl = 'bundle.bin.gz', options = {}, more = {}) {
  const { context, log } = page({ 'bundle.bin.gz': served, ...more });
  vm.runInContext(script, context);
  const container = element('div', () => context.layoutWidth);
  const view = await context.CraftaxPolicyView.mount(container, { bundleUrl, spritesUrl: 'sprites.png', ...options });
  const canvas = container.children[0].children[0];
  return { context, log, view, container, canvas, ctx: canvas.getContext('2d') };
}

// Each window cell's sprites in draw order, keyed 'row,col', and the dark cells.
function windowOf(ops) {
  const cells = {}, dark = [];
  for (const [op, id, x, y, w, h] of ops) {
    if (op === 'image' && y < 144) (cells[`${y / 16},${x / 16}`] ??= []).push(id);
    if (op === 'rect' && id === '#111719') dark.push(`${y / 16},${x / 16}`);
  }
  return { cells, dark };
}

function assertFrame(ops, t) {
  const expected = {};
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 11; c++) {
      const key = `${r},${c}`;
      if (key !== '0,0') expected[key] = [blocks[t], ...(key in sprites[t] ? [sprites[t][key]] : []), ...(key === '4,5' ? [players[t]] : [])];
    }
  }
  assert.deepEqual(windowOf(ops), { cells: expected, dark: ['0,0'] }, `frame ${t}`);
}

const hudOps = ops => ops.filter(([op, , , y]) => (op === 'image' || op === 'rect') && y >= 144);

test('the script defines one global and loads nothing until mount', () => {
  const { context, log } = page();
  const names = new Set(Object.getOwnPropertyNames(context));
  vm.runInContext(script, context);
  assert.deepEqual(Object.getOwnPropertyNames(context).filter(name => !names.has(name)), ['CraftaxPolicyView']);
  for (const name of ['inventoryIcon', 'drawWindow', 'hudMeters', 'mount', 'RECORD', 'GLYPHS']) {
    assert.equal(vm.runInContext(`typeof ${name}`, context), 'undefined', name);
  }
  assert.deepEqual([log.fetched, log.images, log.created], [[], [], []]);
});

test('mount loads the bundle and the sprites, at a whole number of device pixels', async () => {
  const { log, view, container, canvas, ctx } = await mounted();
  assert.deepEqual([log.fetched, log.images], [['bundle.bin.gz'], ['sprites.png']]);
  assert.equal(view.decisions, 4);
  const [figure] = container.children;
  assert.deepEqual(figure.children.map(node => [node.tag, node.className]), [['canvas', 'cpv-canvas'], ['figcaption', 'cpv-caption']]);
  assert.equal(figure.className, 'cpv');
  assert.equal(view.caption, figure.children[1]);
  // 700 CSS pixels of 2 device pixels hold 7 device pixels per game pixel of 176.
  assert.deepEqual([canvas.width, canvas.height, canvas.style.width, canvas.style.height], [176 * 7, 236 * 7, '616px', '826px']);
  assert.deepEqual(ctx.ops[0], ['transform', 7, 0, 0, 7, 0, 0]);
  assertFrame(ctx.ops, 0);
});

test('show(t) draws the frame recorded before decision t, and the last from decisions on', async () => {
  const { view, ctx } = await mounted();
  for (const [t, frame] of [[2, 2], [0, 0], [3, 3], [1, 1], [4, 4], [-3, 0], [99, 4], [2.7, 2], [Number.NaN, 0]]) {
    ctx.ops.length = 0;
    assert.equal(view.show(t), undefined);
    assertFrame(ctx.ops, frame);
  }
});

test('show redraws nothing for the frame shown, and the HUD only when it changes', async () => {
  const { view, ctx } = await mounted();
  view.show(1);
  ctx.ops.length = 0;
  view.show(1);
  assert.deepEqual(ctx.ops, []);
  view.show(3);
  ctx.ops.length = 0;
  view.show(4);
  assertFrame(ctx.ops, 4);
  assert.deepEqual(hudOps(ctx.ops), []);
  ctx.ops.length = 0;
  view.show(2);
  assert.ok(hudOps(ctx.ops).length);
});

test('the HUD draws each meter against its maximum and every held item', async () => {
  const { view, ctx } = await mounted();
  ctx.ops.length = 0;
  view.show(2);
  const bars = Object.fromEntries(['#42b65a', '#438ed0', '#d5a43e', '#4cabc5', '#ad79d6'].map(colour => [colour, ctx.ops.find(op => op[1] === colour).slice(2)]));
  assert.deepEqual(bars, {
    '#42b65a': [52, 149, Math.round(121 * 7.25 / 9), 3],
    '#438ed0': [52, 156, Math.round(121 * 5 / 9), 3],
    '#d5a43e': [52, 163, 121, 3],
    '#4cabc5': [52, 170, Math.round(121 * 8 / 9), 3],
    '#ad79d6': [52, 177, Math.round(121 * 7 / 9), 3],
  });
  // The H of HEALTH: its top row is 1 0 1.
  assert.ok(ctx.ops.some(op => op.join() === 'rect,#9fb2ba,3,148,1,1') && ctx.ops.some(op => op.join() === 'rect,#9fb2ba,5,148,1,1'));
  const items = () => ctx.ops.filter(([op, , , y]) => op === 'image' && y >= 186).map(([, id, x, y]) => [id, x, y]);
  // Wood, a stone pickaxe (tier 2, uncounted) and arrows.
  assert.deepEqual(items(), [[6, 0, 186], [63, 16, 186], [51, 32, 186]]);
  const counted = x => ctx.ops.some(([op, colour, left, top]) => op === 'rect' && colour === '#f4f7f5' && left >= x && left < x + 16 && top >= 196);
  assert.deepEqual([counted(0), counted(16), counted(32)], [true, false, true]);
  ctx.ops.length = 0;
  view.show(3);
  assert.deepEqual(items(), [[6, 0, 186], [63, 16, 186], [51, 32, 186], [100, 48, 186]]);
});

test('a resize draws the frame again at the new whole-number scale', async () => {
  const { context, log, view, canvas, ctx } = await mounted();
  view.show(2);
  context.layoutWidth = 300;
  ctx.ops.length = 0;
  log.observers[0]();
  // 600 device pixels hold 3 of 176.
  assert.deepEqual([canvas.width, canvas.style.width], [528, '264px']);
  assert.deepEqual(ctx.ops[0], ['transform', 3, 0, 0, 3, 0, 0]);
  assertFrame(ctx.ops, 2);
  assert.ok(hudOps(ctx.ops).length);
  ctx.ops.length = 0;
  log.observers[0]();
  assert.deepEqual(ctx.ops, []);
});

test('the bundle may come decoded or as base64 text, or through the page loader', async () => {
  for (const served of [frames(), Buffer.from(`${zlib.gzipSync(frames()).toString('base64')}\n`)]) {
    assert.equal((await mounted(served)).view.decisions, 4);
  }
  const { context, log } = page();
  vm.runInContext(script, context);
  const asked = [];
  const loadBundle = async url => {
    asked.push(url);
    return zlib.gzipSync(frames()).toString('base64');
  };
  const container = element('div', () => 700);
  const view = await context.CraftaxPolicyView.mount(container, { bundleUrl: 'blog:panel', spritesUrl: 'sprites.png', loadBundle });
  assert.deepEqual([asked, log.fetched, view.decisions], [['blog:panel'], [], 4]);
});

test('mount refuses a bundle that is missing or holds no policy-view frames', async () => {
  await assert.rejects(mounted(undefined, 'elsewhere.bin.gz'), /elsewhere\.bin\.gz did not load \(HTTP 404\)/);
  await assert.rejects(mounted(frames().subarray(0, 4 * RECORD + 3)), /holds no policy-view frames/);
  const unmarked = frames();
  unmarked[4 * RECORD + 12] = 0;
  await assert.rejects(mounted(unmarked), /holds no policy-view frames/);
});

test('the row layout sets window, meters and items side by side at the height asked', async () => {
  let tall = 288;
  const { context, log, view, canvas, ctx } = await mounted(undefined, undefined, { layout: 'row', height: () => tall });
  context.layoutWidth = 1000;
  log.observers[0]();
  // 288 CSS pixels of 2 device pixels hold 4 per game pixel of 144, exactly.
  assert.deepEqual([canvas.width, canvas.height, canvas.style.width, canvas.style.height], [425 * 4, 144 * 4, '850px', '288px']);
  ctx.ops.length = 0;
  view.show(2);
  assertFrame(ctx.ops.filter(([, , x]) => x < 176), 2);
  // The meters right of the window, one row each, 27 game pixels apart.
  const bars = ['#42b65a', '#438ed0', '#d5a43e', '#4cabc5', '#ad79d6'].map(colour => ctx.ops.find(op => op[1] === colour).slice(2, 4));
  assert.deepEqual(bars, [[234, 13], [234, 40], [234, 67], [234, 94], [234, 121]]);
  // The items right of the meters, four to a row: wood, a stone pickaxe and arrows.
  const items = ctx.ops.filter(([op, , x]) => op === 'image' && x >= 361).map(([, id, x, y]) => [id, x, y]);
  assert.deepEqual(items, [[6, 361, 0], [63, 377, 0], [51, 393, 0]]);
  // A height that is no whole number of game pixels: drawn at the next whole
  // scale, shown at exactly that height.
  tall = 250;
  view.fit();
  assert.deepEqual([canvas.width, canvas.height, canvas.style.height], [425 * 4, 144 * 4, '250px']);
  // Too narrow to read in a row: the column layout, fitting the width.
  context.layoutWidth = 500;
  log.observers[0]();
  assert.deepEqual([canvas.width, canvas.height, canvas.style.width], [176 * 5, 236 * 5, '440px']);
});

// Three sleep frames: decision 1's sleep of 9 ticks (two frames) and decision
// 3's of 5 (one), each a block of its own and the player asleep.
function sleepFrames() {
  const bytes = Buffer.alloc(3 * RECORD);
  for (let k = 0; k < 3; k++) {
    const base = k * RECORD;
    bytes.writeUInt32LE([1, 1, 3][k], base);
    bytes[base + 12] = 6;
    bytes[base + 17] = 4;
    bytes[base + 18] = 1;
    for (let cell = 1; cell < 99; cell++) bytes.set([20 + k, 1, 1], base + 34 + cell * 8);
    bytes.writeFloatLE(9, base + 22);
  }
  return bytes;
}

test('show(t, k) draws the k-th frame of decision t\'s sleep, and the decision\'s frame otherwise', async () => {
  const sleeps = [[1, 9, 0], [3, 5, 2]];
  const { view, ctx } = await mounted(undefined, undefined, { sleepUrl: 'sleep.bin.gz', sleeps }, { 'sleep.bin.gz': zlib.gzipSync(sleepFrames()) });
  const window = () => windowOf(ctx.ops).cells;
  for (const [t, k, block, player] of [[1, 1, 20, 41], [1, 2, 21, 41], [3, 1, 22, 41], [1, 3, blocks[1], players[1]], [2, 1, blocks[2], players[2]], [1, 0, blocks[1], players[1]]]) {
    ctx.ops.length = 0;
    view.show(t, k);
    const cells = window();
    assert.equal(cells['2,2'][0], block, `show(${t}, ${k})`);
    assert.equal(cells['4,5'].at(-1), player, `show(${t}, ${k}) player`);
  }
  ctx.ops.length = 0;
  view.show(1, 0);
  assert.deepEqual(ctx.ops, [], 'the frame on screen is not drawn again');
});

test('the side layout sets one narrow column beside the window: meters over five items a row', async () => {
  const { context, log, view, canvas, ctx } = await mounted(undefined, undefined, { layout: 'side', height: () => 288, wrapBelow: 0 });
  context.layoutWidth = 520;
  log.observers[0]();
  // 288 CSS pixels of 2 device pixels hold 4 per game pixel of 144; the
  // panel is 260 game pixels wide, the window's 176 and a column of 84.
  assert.deepEqual([canvas.width, canvas.height, canvas.style.width, canvas.style.height], [260 * 4, 144 * 4, '520px', '288px']);
  ctx.ops.length = 0;
  view.show(2);
  assertFrame(ctx.ops.filter(([, , x]) => x < 176), 2);
  // Each meter's bar under its label, 12 game pixels apart, 80 long at most.
  const bars = ['#42b65a', '#438ed0', '#d5a43e', '#4cabc5', '#ad79d6'].map(colour => ctx.ops.find(op => op[1] === colour).slice(2, 5));
  assert.deepEqual(bars, [[178, 10, Math.round(80 * 7.25 / 9)], [178, 22, Math.round(80 * 5 / 9)], [178, 34, 80], [178, 46, Math.round(80 * 8 / 9)], [178, 58, Math.round(80 * 7 / 9)]]);
  // The items under the meters, five to a row, inside the column.
  const items = ctx.ops.filter(([op, , x, y]) => op === 'image' && x >= 176 && y >= 64).map(([, id, x, y]) => [id, x, y]);
  assert.deepEqual(items, [[6, 178, 64], [63, 194, 64], [51, 210, 64]]);
  assert.ok(ctx.ops.every(([op, , x, y, w = 0, h = 0]) => op !== 'rect' || (x + w <= 260 && y + h <= 144)), 'nothing drawn outside the panel');
  // A width narrower than the height asks: fit the width instead.
  context.layoutWidth = 390;
  log.observers[0]();
  assert.deepEqual([canvas.style.width, canvas.style.height], ['390px', `${390 * 144 / 260}px`]);
});

test('from beatenFrom on, a red X on a dark casing crosses out each necromancer in sight', async () => {
  // The necromancer (block 32) in sight at (3, 5) in frames 3 and 4, and in
  // the dark at (0, 0) in frame 4.
  const bytes = frames();
  for (const t of [3, 4]) bytes[t * RECORD + 34 + (3 * 11 + 5) * 8] = 32;
  bytes[4 * RECORD + 34] = 32;
  const strokes = (ops, colour) => ops.filter(op => op[0] === 'rect' && op[1] === colour).map(([, , x, y, w, h]) => [x, y, w, h]);
  // Both diagonals of the tile at (row, col), squares of width from pixel 3 - width / 2.
  const cross = (row, col, width) => Array.from({ length: 11 }, (_, i) => {
    const from = 3 - width / 2;
    return [[col * 16 + from + i, row * 16 + from + i, width, width], [col * 16 + 16 - from - width - i, row * 16 + from + i, width, width]];
  }).flat();
  for (const [beatenFrom, t, crossed] of [[4, 3, false], [4, 4, true], [null, 4, false], [3, 3, true], [3, 4, true]]) {
    const { view, ctx } = await mounted(zlib.gzipSync(bytes), undefined, beatenFrom === null ? {} : { beatenFrom });
    ctx.ops.length = 0;
    view.show(t);
    const label = `beatenFrom ${beatenFrom}, frame ${t}`;
    assert.deepEqual(strokes(ctx.ops, '#ff2a2a'), crossed ? cross(3, 5, 2) : [], label);
    assert.deepEqual(strokes(ctx.ops, 'rgba(8,10,12,.9)'), crossed ? cross(3, 5, 4) : [], label);
    if (!crossed) continue;
    // Over the necromancer's tile, casing first, and inside it.
    const tileAt = ctx.ops.findIndex(([op, id, x, y]) => op === 'image' && id === 32 && x === 80 && y === 48);
    const casingAt = ctx.ops.findIndex(op => op[1] === 'rgba(8,10,12,.9)'), redAt = ctx.ops.findIndex(op => op[1] === '#ff2a2a');
    assert.ok(tileAt >= 0 && tileAt < casingAt && casingAt < redAt, label);
    assert.ok(cross(3, 5, 4).every(([x, y, w, h]) => x >= 80 && x + w <= 96 && y >= 48 && y + h <= 64), label);
  }
});

test('a v2 bundle turns each projectile the way it flies, as original Craftax does', async () => {
  // Five frames of 984-byte records: the cut frame, then its 99 facings. In
  // frame 1, at view cells in a row, an enemy's projectile of each kind
  // (channel 6) and the player's (channel 7), each kind in each of the four
  // facings, LEFT 1, RIGHT 2, UP 3, DOWN 4; frame 2 shows a v2 frame without
  // projectiles.
  const kinds = [[1, 51], [2, 99], [3, 100], [4, 101], [6, 102]], cut = frames(), V2 = RECORD + 99;
  const bytes = Buffer.alloc(5 * V2);
  for (let t = 0; t < 5; t++) cut.copy(bytes, t * V2, t * RECORD, (t + 1) * RECORD);
  const placed = [];
  kinds.forEach(([value, sprite], kind) => [1, 2, 3, 4].forEach((facing, k) => {
    for (const [channel, shift] of [[6, 0], [7, 4]]) {
      const cell = 1 + (kind * 8 + k * 2 + (channel - 6)) % 98;
      bytes[V2 + 34 + cell * 8 + 3] = 0;
      bytes[V2 + 34 + cell * 8 + channel] = value;
      bytes[V2 + RECORD + cell] |= facing << shift;
      placed.push({ cell, sprite, facing });
    }
  }));
  const { view, ctx } = await mounted(zlib.gzipSync(bytes));
  assert.equal(view.decisions, 4);
  ctx.ops.length = 0;
  view.show(1);
  // Upstream's arrays: a texture flipped top to bottom (axis 0) when drow > 0
  // or dcol > 0, then transposed when dcol != 0. Source pixel (row v, column
  // u) therefore lands at row v, column u when flying up; row 15 - v when
  // down; row u, column v when left; row u, column 15 - v when right.
  const lands = (facing, v, u) => {
    let [row, col] = [v, u];
    if (facing === 4 || facing === 2) row = 15 - row;
    return facing === 1 || facing === 2 ? [col, row] : [row, col];
  };
  for (const { cell, sprite, facing } of placed) {
    const r = Math.floor(cell / 11), c = cell % 11;
    const ops = ctx.ops.filter(([op, id, x, y]) => op === 'image' && id === sprite && x === c * 16 && y === r * 16);
    assert.equal(ops.length, 1, `sprite ${sprite} facing ${facing} at cell ${cell}`);
    const [, , , , , , [a, b, cc, d, e, f], [dx, dy, dw, dh]] = ops[0];
    for (let v = 0; v < 16; v++) for (let u = 0; u < 16; u++) {
      const x = dx + (u + 0.5) * dw / 16, y = dy + (v + 0.5) * dh / 16;
      const got = [Math.floor(b * x + d * y + f) - r * 16, Math.floor(a * x + cc * y + e) - c * 16];
      assert.deepEqual(got, lands(facing, v, u), `sprite ${sprite} facing ${facing}, source pixel (${v}, ${u})`);
    }
  }
  // Each kind in each facing, from either side: 5 x 4 x 2 drawn.
  assert.equal(placed.length, 40);
});
