// Drives headless Chrome at render_check.mjs's default path, one launch per test.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const here = path.dirname(fileURLToPath(import.meta.url));
let root, html;

// An exact game whose full map holds a different block in every cell, so its
// board draws many colours.
before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'craftax-render-test-'));
  const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
  const frames = Buffer.alloc(2 * 7858);
  for (let i = 0; i < 2; i++) for (let cell = 0; cell < 48 * 48; cell++) frames.set([1 + cell % 40, 0, 255], i * 7858 + 885 + 3 * cell);
  frames.writeUInt8(255, 7858 + 12);
  const gzip = zlib.gzipSync(frames), game = path.join(root, 'game');
  fs.mkdirSync(game);
  fs.writeFileSync(path.join(game, 'frames.bin.gz'), gzip);
  fs.writeFileSync(path.join(game, 'manifest.json'), JSON.stringify({
    schema: 'craftax-exact-game/v1', title: 'Map', description: '1 recorded action', endLabel: 'End',
    provenance: 'Hand-written frames', frames: 2, actions: 1, tick: 0, score: 0, achievements: 0,
    framesSha256: sha(frames), gzipSha256: sha(gzip),
  }));
  const page = path.join(root, 'page.html');
  const built = spawnSync(process.execPath, [path.join(here, 'games.mjs'), 'build', '--output', page, game], { encoding: 'utf8' });
  assert.equal(built.status, 0, built.stderr);
  html = fs.readFileSync(page, 'utf8');
});

after(() => fs.rmSync(root, { recursive: true, force: true }));

function check(name, text) {
  const page = path.join(root, `${name}.html`), out = path.join(root, name);
  fs.writeFileSync(page, text);
  const result = spawnSync(process.execPath, [path.join(here, 'render_check.mjs'), out, page], { encoding: 'utf8' });
  const report = JSON.parse(fs.readFileSync(path.join(out, 'render-check.json'), 'utf8'));
  return { status: result.status, errors: report.errors, ...report.pages[0] };
}

test('a page whose games all render passes', () => {
  const result = check('good', html);
  assert.equal(result.status, 0);
  assert.deepEqual([result.opened, result.games, result.ok, result.errors], [true, 1, 1, []]);
  assert.deepEqual(Object.keys(result.entries[0].colours), ['policy', 'map']);
});

test('an exact game whose policy view fails to draw fails, though its map draws', () => {
  const result = check('board', html.replace('function renderBoard(i){', 'function renderBoard(i){throw Error("no board");'));
  assert.equal(result.status, 1);
  assert.match(result.errors.join('\n'), /no board/);
  assert.deepEqual([result.opened, result.games], [true, 1]);
});

test('a script error fails the page, though its games render', () => {
  const result = check('thrown', html.replace('</body>', '<script>throw Error("injected")</script></body>'));
  assert.equal(result.status, 1);
  assert.match(result.errors.join('\n'), /injected/);
  assert.deepEqual([result.opened, result.games, result.ok], [true, 1, 1]);
});

test('a page whose first load fails is a failure, though each game loads when picked', () => {
  const result = check('unopened', html.replace('window.CRAFTAX_DEFAULT_GAME=0;', 'window.CRAFTAX_DEFAULT_GAME=9;'));
  assert.equal(result.status, 1);
  assert.deepEqual([result.opened, result.games, result.ok, result.errors], [false, 1, 1, []]);
});

test('a page with no games to check is a failure', () => {
  const result = check('empty', '<p id="status">Loaded</p><select id="game-picker"></select>');
  assert.equal(result.status, 1);
  assert.deepEqual([result.opened, result.games, result.errors], [true, 0, []]);
});
