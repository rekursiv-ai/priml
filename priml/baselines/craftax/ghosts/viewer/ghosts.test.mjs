// node --test ghosts.test.mjs
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import zlib from 'node:zlib';

import { addPolicy, writeData } from './blog.mjs';
import { pack } from './pack.mjs';
import { writeSynthetic } from './synthetic.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.join(here, '..', 'testdata', 'fixture');
const context = {};
vm.runInNewContext(fs.readFileSync(path.join(here, 'decode.js'), 'utf8'), context);
const D = context.GhostDecode;
let root, synthetic, truth;

before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'ghosts-test-'));
  synthetic = path.join(root, 'synthetic');
  ({ truth } = writeSynthetic(synthetic));
});

after(() => fs.rmSync(root, { recursive: true, force: true }));

// Typed arrays from decode.js's context have that context's prototypes, which
// deepStrictEqual would report as a difference.
const same = (actual, expected, message) => assert.deepEqual(Array.from(actual), Array.from(expected), message);
const read = (dir, name) => {
  const bytes = fs.readFileSync(path.join(dir, name));
  return new Uint8Array(name.endsWith('.gz') ? zlib.gunzipSync(bytes) : bytes);
};
const json = (dir, name) => JSON.parse(Buffer.from(read(dir, name)).toString('utf8'));

// The fixture's expected values, each episode a full entry: an entry that
// repeats an earlier one's ghost names it (same_as) and holds only its place
// and what differs (fixture.py shared).
function expectedValues() {
  const values = json(fixture, 'expected.json'), episodes = [];
  for (const { same_as: base, ...entry } of values.episodes) {
    if (base === undefined) episodes.push(entry);
    else { const { tier, set, index, ...body } = episodes[base]; void tier; void set; void index; episodes.push({ ...body, ...entry }); }
  }
  return { ...values, episodes };
}

// The page's load path: a tier's set's first `groups` groups, then the
// precompute pass; with `sleep`, the view that shows sleep.
function load(dir, tierName, setName, groups, sleep = false) {
  const manifest = json(dir, 'manifest.json'), world = D.parseWorld(read(dir, 'world.bin.gz'));
  const tier = manifest.tiers.find(t => t.name === tierName).sets.find(s => s.name === setName);
  const loaded = tier.groups.slice(0, groups).map(g => ({
    listing: json(dir, `${g.path}/episodes.json`), players: read(dir, `${g.path}/players.bin.gz`), events: read(dir, `${g.path}/events.bin.gz`),
    keeps: tier.time_map ? read(dir, `${g.path}/${sleep ? 'keeps-sleep' : 'keeps'}.bin.gz`) : null,
    sleep: sleep ? read(dir, `${g.path}/sleep.bin.gz`) : null,
  }));
  const set = D.episodeSet(world, loaded, manifest), run = D.precompute(set);
  let step;
  while (!(step = run.next()).done);
  return { manifest, tier, world, set, pre: step.value };
}

// Every creature window of the set's groups, indexed: per episode, its samples.
function creatureSamples(dir, tier, set) {
  const out = Array.from({ length: set.n }, () => []), groups = tier.groups.slice(0, set.groupSizes.length);
  for (let j = 0; j < Math.max(...groups.map(g => g.windows)); j++) {
    const files = groups.map(g => (j < g.windows ? read(dir, `${g.path}/creatures-w${j}.bin.gz`) : null));
    const win = D.indexWindow(set, j, files);
    for (let e = 0; e < set.n; e++) {
      for (let s = 0; s < win.per; s++) {
        const at = win.index[e * win.per + s], bytes = files[set.group[e]];
        if (at < 0) continue;
        const list = [];
        for (let c = 0, k = at + 1; c < bytes[at]; c++, k += set.creatureBytes) list.push([bytes[k] >> 4, bytes[k] & 15, bytes[k + 1], bytes[k + 2], ...(set.creatureBytes > 3 ? [bytes[k + 3]] : [])]);
        out[e].push([j * set.window + s * set.stride, list]);
      }
    }
  }
  return out;
}

test('the fixture decodes to layout.py\'s values, byte for byte', () => {
  const site = path.join(fixture, 'site'), expected = expectedValues();
  for (const [tierName, name] of [['fixture', 'all'], ['fixture', 'short'], ['fixture-wins', 'wins']]) {
    const { tier, world, set } = load(site, tierName, name, 1), episodes = expected.episodes.filter(e => e.tier === tierName && e.set === name);
    assert.equal(set.n, episodes.length);
    checkEpisodes(episodes, set, world, creatureSamples(site, tier, set));
  }
});

test('the fixture\'s time maps decode to layout.py\'s displayed decisions, the pinned run unbroken', () => {
  const site = path.join(fixture, 'site'), expected = expectedValues();
  const { tier, set } = load(site, 'fixture-wins', 'wins', 1), episodes = expected.episodes.filter(e => e.set === 'wins');
  assert.ok(set.mapped && episodes.length === 2);
  episodes.forEach((want, e) => {
    assert.equal(set.length[e], want.displayed.length);
    same(Array.from({ length: set.length[e] }, (_, u) => D.decisionAt(set, e, u)), want.displayed, `episode ${e}`);
    assert.equal(D.decisionAt(set, e, set.length[e]), set.decisions[e]);
  });
  const pinned = tier.time_map.unbroken;
  assert.equal(pinned, 0);
  assert.equal(set.shown[pinned], null);
  assert.equal(set.length[pinned], set.decisions[pinned]);
  assert.ok(set.length[1] < set.decisions[1]);
  assert.equal(set.maxSteps, tier.time_map.steps);
});

function checkEpisodes(episodes, set, world, samples) {
  episodes.forEach((want, e) => {
    const { states, interactions } = D.decodeEpisode(set, e);
    same(states, want.path.flat(), `path of episode ${e}`);
    assert.deepEqual(JSON.parse(JSON.stringify(interactions)), want.interactions, `interactions of episode ${e}`);
    assert.deepEqual([set.decisions[e], D.OUTCOMES[set.outcome[e]], set.ret[e]], [want.decisions, want.outcome, want.achievement_return]);
    assert.equal(D.returnAt(set, e, set.decisions[e]), want.achievement_return, `return of episode ${e}`);
    assert.deepEqual([set.endTile[e] / D.CELLS | 0, (set.endTile[e] % D.CELLS) / D.MAP | 0, set.endTile[e] % D.MAP, set.endFacing[e]], want.end);
    // The final map: the world with each tile's last change applied.
    const block = world.block.slice(), item = world.item.slice(), map = set.map[e];
    for (let k = 0; k < map.decision.length; k++) {
      const code = map.code[k], tile = map.tile[k];
      block[tile] = code ? (code - 1) / D.ITEMS | 0 : world.block[tile];
      item[tile] = code ? (code - 1) % D.ITEMS : world.item[tile];
    }
    const changes = [];
    for (let tile = 0; tile < D.TILES; tile++) {
      if (block[tile] !== world.block[tile] || item[tile] !== world.item[tile]) changes.push([tile / D.CELLS | 0, (tile % D.CELLS) / D.MAP | 0, tile % D.MAP, block[tile], item[tile]]);
    }
    assert.deepEqual(changes, want.final_changes, `final changes of episode ${e}`);
    assert.equal(crypto.createHash('sha256').update(block).update(item).digest('hex'), want.final_map_sha256);
    assert.deepEqual(samples[e], want.creatures, `creatures of episode ${e}`);
  });
}

test('the view that shows sleep steps through each sleep\'s samples as layout.py does', () => {
  const site = path.join(fixture, 'site'), values = expectedValues();
  const { set, pre } = load(site, 'fixture-wins', 'wins', 1, true), episodes = values.episodes.filter(e => e.set === 'wins');
  assert.ok(set.sleepView && episodes.some(e => e.sleeps.length));
  episodes.forEach((want, e) => {
    same(Array.from({ length: set.length[e] }, (_, u) => [D.decisionAt(set, e, u), D.sampleAt(set, e, u)]).flat(), want.displayed_sleep.flat(), `episode ${e}`);
    const runs = set.sleepRuns[set.group[e]], at = Array.from(set.runAt[e]).filter(k => k >= 0);
    const size = set.creatureBytes, decoded = at.map(k => Array.from({ length: runs[k] }, (_, c) => {
      const record = runs.subarray(k + 1 + size * c, k + 1 + size * (c + 1));
      return [record[0] >> 4, record[0] & 15, ...record.subarray(1)];
    }));
    assert.deepEqual(JSON.parse(JSON.stringify(decoded)), want.sleep_creatures, `sleep samples of episode ${e}`);
  });
  // Through a sleep's samples every ghost holds its decision's state.
  const pb = new D.Playback(set, pre);
  for (let u = 0; u <= set.maxSteps; u++) {
    pb.seek(u);
    const want = expected(set, u);
    same(Array.from({ length: set.n }, (_, e) => pb.cur.floor[e] * D.CELLS + pb.cur.row[e] * 48 + pb.cur.col[e]), want.at, `positions at ${u}`);
    same(pb.heat, want.heat, `heat at ${u}`);
  }
});

test('the fixture\'s timelines decode to layout.py\'s activity and segments', () => {
  const site = path.join(fixture, 'site'), expected = expectedValues();
  assert.ok(expected.timelines.some(t => t.segments.length > 1));
  for (const want of expected.timelines) {
    const sets = json(site, 'manifest.json').tiers.find(t => t.name === want.tier).sets;
    const { set } = load(site, want.tier, want.set, sets.find(s => s.name === want.set).counts.indexOf(want.count) + 1);
    const quiet = D.parseTimeline(read(site, `${want.tier}/${want.set}/timeline-n${want.count}.bin.gz`), set.maxDecisions);
    same(quiet.activity, want.activity, `activity of ${want.set} ${want.count}`);
    assert.deepEqual(Array.from(quiet.starts, (start, i) => [start, quiet.stops[i]]), want.segments);
    const shown = Array.from({ length: quiet.timeline.length + 1 }, (_, u) => quiet.timeline.toDecision(u));
    assert.deepEqual(shown, [...want.segments.flatMap(([a, b]) => Array.from({ length: b - a }, (_, i) => a + i)), set.maxDecisions]);
  }
});

test('the summary of each set and count equals the manifest\'s stats', () => {
  const site = path.join(fixture, 'site');
  for (const { name, stats } of json(site, 'manifest.json').tiers[0].sets) {
    stats.forEach((want, k) => {
      const { decisions, ...got } = D.summary(load(site, 'fixture', name, k + 1).set);
      assert.ok(decisions > 0);
      assert.deepEqual(JSON.parse(JSON.stringify(got)), want);
    });
  }
});

test('every decoded synthetic state equals the generator\'s, escapes included', () => {
  for (const tier of ['early', 'high']) {
    const { set } = load(synthetic, tier, 'all', 2);
    truth[tier].forEach((episode, e) => same(D.decodeEpisode(set, e).states, episode.states, `${tier} episode ${e}`));
  }
  assert.ok(truth.high.some(e => e.escapes.length));
});

// Brute force from the full state arrays: what the playback must hold at
// display step u, each episode at its own decision.
function expected(set, u) {
  const heat = new Uint32Array(2 * D.TILES), at = [], decision = [], changes = new Map();
  for (let e = 0; e < set.n; e++) {
    const { states, interactions } = D.decodeEpisode(set, e), t = D.decisionAt(set, e, u);
    const tile = s => states[4 * s] * D.CELLS + states[4 * s + 1] * D.MAP + states[4 * s + 2];
    heat[tile(0)]++;
    for (let s = 1; s <= t; s++) if (tile(s) !== tile(s - 1)) heat[tile(s)]++;
    for (const [d, floor, r, c] of interactions) if (d < t && r >= 0 && r < 48 && c >= 0 && c < 48) heat[D.TILES + floor * D.CELLS + r * 48 + c]++;
    at.push(tile(t));
    decision.push(t);
    if (u >= set.length[e]) continue;
    const map = set.map[e];
    for (let k = 0; k < map.decision.length && map.decision[k] < t; k++) changes.set(`${map.tile[k]}/${e}`, map.code[k]);
  }
  const count = new Uint16Array(D.TILES);
  for (const [key, code] of changes) if (code) count[Number(key.split('/')[0])]++;
  return { heat, at, decision, count };
}

for (const name of ['all', 'wins']) {
  test(`seeking anywhere and playing forward agree with a full replay (${name})`, () => {
    const { set, pre } = load(synthetic, 'high', name, 2);
    assert.equal(set.mapped, name === 'wins');
    if (set.mapped) assert.ok(set.shown.slice(1).every(shown => shown && shown.length < 0.8 * shown.at(-1)), 'the synthetic time maps skip little');
    const forward = new D.Playback(set, pre), jumping = new D.Playback(set, pre);
    // Display steps of map changes, where applying one a step early would show.
    const changes = set.map.flatMap((m, e) => Array.from(m.decision.subarray(0, 3), d => D.stepOf(set, e, d)));
    const times = [0, 1, 5, 1023, 1024, 1025, 3000, 2047, 9000, 4100, set.maxSteps, 17, set.maxSteps - 1, ...changes].filter(u => u <= set.maxSteps);
    for (const u of [...times].sort((a, b) => a - b)) {
      forward.seek(u);
      jumping.seek(u);
      const want = expected(set, u);
      for (const pb of [forward, jumping]) {
        pb.summarize();
        same(pb.heat, want.heat, `heat at ${u}`);
        same(Array.from({ length: set.n }, (_, e) => pb.cur.floor[e] * D.CELLS + pb.cur.row[e] * 48 + pb.cur.col[e]), want.at, `positions at ${u}`);
        same(pb.decision, want.decision, `decisions at ${u}`);
        same(pb.tileCount, want.count, `map changes at ${u}`);
      }
    }
    for (const u of times) {
      jumping.seek(u);
      jumping.summarize();
      same(jumping.tileCount, expected(set, u).count, `map changes after seeking back to ${u}`);
    }
  });
}

test('each run\'s return and the wins so far equal a brute-force count at every step', () => {
  for (const [dir, tierName, name, groups] of [[synthetic, 'high', 'all', 2], [synthetic, 'high', 'wins', 2], [path.join(fixture, 'site'), 'fixture-wins', 'wins', 1]]) {
    const { set } = load(dir, tierName, name, groups), curves = D.rewardCurves(set), wins = D.winSteps(set);
    assert.equal(curves.top, Math.max(...Array.from(set.ret)));
    const steps = new Set([0, 1, set.maxSteps, set.maxSteps + 5]);
    for (let e = 0; e < set.n; e++) {
      for (const d of set.score[e].decision) for (const s of [d, d + 1, d + 2]) if (s <= set.decisions[e]) { const u = D.stepOf(set, e, s); steps.add(u); steps.add(u - 1); }
      steps.add(set.length[e]);
      steps.add(set.length[e] - 1);
    }
    for (const u of [...steps].filter(u => u >= 0)) {
      let won = 0;
      for (let e = 0; e < set.n; e++) {
        const value = D.returnAt(set, e, D.decisionAt(set, e, u));
        assert.equal(D.curveAt(curves.runs[e], u), value, `${name} run ${e} at ${u}`);
        won += set.outcome[e] === 2 && set.length[e] <= u;
      }
      assert.equal(D.wonBy(wins, u), won, `${name} wins by ${u}`);
    }
    assert.ok(wins.length, `${tierName}/${name} has no win to count`);
  }
});

test('a ghost trail holds its last 240 moves on its floor since the replay start', () => {
  assert.equal(D.TRAIL, 241);
  let full = 0;
  for (const [name, u] of [['all', 600], ['all', 2048], ['all', 5000], ['all', 12_000], ['wins', 2048], ['wins', 5000]]) {
    const { set, pre } = load(synthetic, 'high', name, 2);
    // A fresh playback replays from 0 up to two keyframes ahead, else from
    // the keyframe before u's.
    const pb = new D.Playback(set, pre), start = u <= 2 * D.KEYFRAME ? 0 : Math.max(0, Math.floor(u / D.KEYFRAME) - 1) * D.KEYFRAME;
    pb.seek(u);
    for (let e = 0; e < set.n; e++) {
      if (!pb.alive(e)) continue;
      const states = truth.high[e].states, cell = s => states[4 * s + 1] * 48 + states[4 * s + 2], from = D.decisionAt(set, e, start), t = D.decisionAt(set, e, u);
      let points = [cell(from)];
      for (let s = from + 1; s <= t; s++) {
        if (states[4 * s] !== states[4 * (s - 1)]) points = [cell(s)];
        else if (cell(s) !== cell(s - 1)) points.push(cell(s));
      }
      same(pb.trailOf(e), points.slice(-D.TRAIL), `${name} episode ${e} at ${u}`);
      full += pb.trailLength[e] === D.TRAIL;
    }
  }
  assert.ok(full, 'no trail reached 240 moves');
});

test('the followed run\'s tail holds its last 800 moves on its floor', () => {
  const { set } = load(synthetic, 'high', 'all', 2);
  let full = 0;
  for (let e = 0; e < set.n; e++) {
    const { states } = D.decodeEpisode(set, e), cell = s => states[4 * s + 1] * 48 + states[4 * s + 2];
    for (const s of [0, 500, 3000, set.decisions[e]].filter(s => s <= set.decisions[e])) {
      let points = [cell(0)];
      for (let j = 1; j <= s; j++) {
        if (states[4 * j] !== states[4 * (j - 1)]) points = [cell(j)];
        else if (cell(j) !== cell(j - 1)) points.push(cell(j));
      }
      same(D.pathTail(states, s, 800), points.slice(-801), `episode ${e} at ${s}`);
      full += points.length > 801;
    }
  }
  assert.ok(full, 'no tail reached 800 moves');
});

test('the fog lifts where a run has had a tile in its 9 x 11 view, from the first display step any did', () => {
  for (const [tier, name] of [['early', 'all'], ['high', 'all'], ['high', 'wins']]) {
    const { set, pre } = load(synthetic, tier, name, 2), expected = new Int32Array(D.TILES).fill(D.NEVER);
    for (let e = 0; e < set.n; e++) {
      const { states } = D.decodeEpisode(set, e);
      for (let s = 0; s <= set.decisions[e]; s++) {
        const u = D.stepOf(set, e, s), [floor, row, col] = states.subarray(4 * s, 4 * s + 3);
        for (let r = Math.max(0, row - 4); r <= Math.min(D.MAP - 1, row + 4); r++) {
          for (let c = Math.max(0, col - 5); c <= Math.min(D.MAP - 1, col + 5); c++) {
            const tile = floor * D.CELLS + r * D.MAP + c;
            if (u < expected[tile]) expected[tile] = u;
          }
        }
      }
    }
    same(pre.reveal, expected, `${tier}/${name}`);
    assert.ok(expected.some(u => u === D.NEVER) && expected.some(u => u > 0 && u !== D.NEVER), `${tier}/${name} has tiles revealed late and never`);
  }
});

test('a torch lights its 9 x 9 square by the game\'s float32 rule, as numpy computes it', () => {
  // [light before, rows off, columns off, light after], from game/rules.py's
  // arithmetic in numpy float32.
  for (const [before, dr, dc, after] of [
    [0, 0, 0, 255], [0, 0, 1, 204], [0, 1, 1, 182], [0, 0, 2, 153], [0, 1, 2, 140], [0, 3, 3, 38], [0, 4, 4, 0], [0, 0, 4, 50],
    [100, 0, 2, 253], [100, 1, 2, 240], [100, 3, 3, 138], [100, 4, 4, 100], [100, 0, 4, 151], [200, 3, 3, 238], [200, 0, 4, 251], [254, 4, 4, 254],
  ]) assert.equal(D.torchLit(before, (dr + 4) * 9 + dc + 4), after, `${before} at (${dr}, ${dc})`);
});

test('the light is the brightest of the world\'s and each living run\'s own, through seeks and as runs end', () => {
  const { set, pre } = load(synthetic, 'high', 'all', 2), pb = new D.Playback(set, pre);
  const expected = u => {
    const out = set.world.light.slice();
    for (let e = 0; e < set.n; e++) {
      if (u >= set.length[e]) continue;
      const own = D.lightAt(set, pre.light, e, D.decisionAt(set, e, u));
      for (let tile = 0; tile < D.TILES; tile++) if (own[tile] > out[tile]) out[tile] = own[tile];
    }
    return out;
  };
  const ends = Array.from(set.length).filter(l => l < set.maxSteps), seen = new Map();
  const times = [0, 1, 40, 900, 5000, 4990, 2000, 2100, ...ends.flatMap(l => [l - 1, l, l + 1]), set.maxSteps, 3, 12_000, 300];
  for (const u of times.filter(u => u <= set.maxSteps)) {
    pb.seek(u);
    if (!seen.has(u)) seen.set(u, expected(u));
    same(pb.light, seen.get(u), `light at step ${u}`);
  }
  assert.ok(pre.light.runs.every(r => r.tile.length), 'every run placed a torch that lit something');
  // A run that ends takes its light away.
  assert.ok(ends.some(l => seen.get(l - 1).some((v, tile) => v > seen.get(l)[tile])), 'no run\'s end darkened a tile');
});

test('creature windows index every synthetic sample and reject a corrupt run', () => {
  const { tier, set } = load(synthetic, 'high', 'all', 2);
  const samples = creatureSamples(synthetic, tier, set);
  truth.high.forEach((episode, e) => assert.deepEqual(samples[e], episode.samples, `episode ${e}`));
  assert.ok(samples.flat().some(([, list]) => list.some(([klass, , , , facing]) => klass >= 3 && facing >= 1 && facing <= 4)), 'no projectile with its facing');
  const files = tier.groups.map(g => read(synthetic, `${g.path}/creatures-w0.bin.gz`)), { index } = D.indexWindow(set, 0, files);
  // A facing past the four, on the first sampled creature of group 0.
  const per = index.length / set.n, at = index.find((k, i) => k >= 0 && set.group[i / per | 0] === 0 && files[0][k] > 0);
  files[0][at + 4] = 5;
  assert.throws(() => D.indexWindow(set, 0, files), /malformed/);
  files[0][at + 4] = 0;
  files[0][index[0]] += 1;
  assert.throws(() => D.indexWindow(set, 0, files), /malformed|does not hold/);
});

test('an end or first floor entry the decoder does not reproduce is rejected', () => {
  const site = path.join(fixture, 'site'), manifest = json(site, 'manifest.json'), world = D.parseWorld(read(site, 'world.bin.gz'));
  const group = { listing: json(site, 'fixture/all/g0/episodes.json'), players: read(site, 'fixture/all/g0/players.bin.gz'), events: read(site, 'fixture/all/g0/events.bin.gz') };
  const drain = g => { const run = D.precompute(D.episodeSet(world, [g], manifest)); while (!run.next().done); };
  drain(group);
  const turned = structuredClone(group.listing);
  turned.episodes[0].end[3] = turned.episodes[0].end[3] % 4 + 1;
  assert.throws(() => drain({ ...group, listing: turned }), /decodes to an end/);
  const entered = structuredClone(group.listing);
  entered.episodes[0].floor_first[1] += 1;
  assert.throws(() => drain({ ...group, listing: entered }), /first stands on floor 1/);
  const players = group.players.slice();
  players[0] = 19 | 0x80;
  assert.throws(() => drain({ ...group, players }), /leaves the floors/);
});

test('pack ships binary data as base64 text and checks every file against the manifest', () => {
  const site = path.join(fixture, 'site'), manifest = json(site, 'manifest.json'), out = path.join(root, 'site');
  const written = pack({ data: site, out }).map(f => f.name).sort();
  const shipped = name => (name.endsWith('.bin.gz') ? `${name}.b64.txt` : name);
  assert.deepEqual(written, [...Object.keys(manifest.files).map(shipped), 'manifest.json', 'index.html', 'ghosts.js', 'decode.js', 'policy_view.js', 'sprites.png'].sort());
  for (const name of Object.keys(manifest.files)) {
    const text = fs.readFileSync(path.join(out, shipped(name)), 'utf8');
    if (name.endsWith('.bin.gz')) assert.match(text, /^[A-Za-z0-9+/]*={0,2}$/, name);
    const bytes = name.endsWith('.bin.gz') ? Buffer.from(text, 'base64') : Buffer.from(text);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), manifest.files[name], name);
  }
  assert.throws(() => pack({ data: site, out: path.join(root, 'site') }), /not empty/);
  const tampered = path.join(root, 'tampered');
  fs.cpSync(site, tampered, { recursive: true });
  fs.appendFileSync(path.join(tampered, 'fixture/all/g0/episodes.json'), ' ');
  assert.throws(() => pack({ data: tampered, out: path.join(root, 'out-tampered') }), /does not match the manifest/);
  fs.writeFileSync(path.join(tampered, 'stray.bin'), 'x');
  assert.throws(() => pack({ data: tampered, out: path.join(root, 'out-stray') }), /unlisted stray\.bin/);
});

test('a timeline maps shown positions to decisions', () => {
  const real = D.realTime(10);
  assert.deepEqual([real.length, real.toDecision(7), real.toIndex(7), real.skipped().length], [10, 7, 7, 0]);
  const timeline = new D.Timeline([[0, 5], [100, 101], [200, 210]]);
  assert.equal(timeline.length, 15);
  assert.deepEqual([0, 4, 5, 6, 15, 99].map(u => timeline.toDecision(u)), [0, 4, 100, 200, 209, 209]);
  assert.deepEqual([0, 4, 5, 50, 100, 150, 209, 500].map(t => timeline.toIndex(t)), [0, 4, 5, 5, 5, 6, 15, 15]);
  assert.deepEqual(JSON.parse(JSON.stringify(timeline.skipped())), [[5, 100], [101, 200]]);
  assert.throws(() => new D.Timeline([[0, 5], [4, 8]]), /disjoint/);
});

test('the shortest win is the winning run with the fewest decisions, the lowest index on a tie', () => {
  const set = (outcomes, decisions) => ({ n: outcomes.length, outcome: outcomes.map(o => D.OUTCOMES.indexOf(o)), decisions });
  assert.equal(D.shortestWin(set(['death', 'win', 'win', 'win'], [3, 90, 40, 40])), 2);
  assert.equal(D.shortestWin(set(['death', 'timeout', 'win'], [3, 10, 500])), 2);
  assert.equal(D.shortestWin(set(['death', 'timeout'], [3, 10])), -1);
  // On real data: the synthetic high tier has wins, the fixture none.
  const { set: high } = load(synthetic, 'high', 'all', 2), wins = truth.high.map((e, i) => [e.decisions, i]).filter((_, i) => truth.high[i].outcome === 'win').sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  assert.ok(wins.length);
  assert.equal(D.shortestWin(high), wins[0][1]);
  assert.equal(D.shortestWin(load(path.join(fixture, 'site'), 'fixture', 'all', 1).set), -1);
});

test('an embed reads its options from data attributes and rejects ones it cannot use', () => {
  const plain = { ...D.embedOptions({}) };
  assert.deepEqual(plain, {
    tier: 'boss', set: 'short', count: 1000, follow: 'shortest-win', transport: 'raw', layout: 'row', skipQuiet: true, speed: 300, speeds: null, dataBase: '', followSeed: null, loop: true, cap: null, controls: 'bottom', fit: 'none', scrubber: 'slider', followBoards: 3,
  });
  const set = { ...D.embedOptions({ tier: 'high', set: 'all', count: '250', follow: 'median', transport: 'b64', layout: 'grid', skipQuiet: 'false', speed: '1000', dataBase: 'https://example.test/ghosts', followSeed: '42', loop: 'false', cap: '6000', controls: 'top', fit: 'column', scrubber: 'plot', followBoards: '2' }) };
  assert.deepEqual(set, {
    tier: 'high', set: 'all', count: 250, follow: 'median', transport: 'b64', layout: 'grid', skipQuiet: false, speed: 1000, speeds: null, dataBase: 'https://example.test/ghosts/', followSeed: '42', loop: false, cap: 6000, controls: 'top', fit: 'column', scrubber: 'plot', followBoards: 2,
  });
  const menu = { ...D.embedOptions({ speeds: '10, 50,100,250', speed: '50' }) };
  assert.deepEqual([[...menu.speeds], menu.speed], [[10, 50, 100, 250], 50]);
  assert.equal(D.embedOptions({ speeds: '10,50' }).speed, 10);
  assert.equal(D.embedOptions({ dataBase: 'http://127.0.0.1:8737/' }).dataBase, 'http://127.0.0.1:8737/');
  for (const [data, message] of [
    [{ transport: 'gzip' }, /data-transport="gzip" is not one of raw, b64/], [{ follow: 'shortest' }, /data-follow/], [{ skipQuiet: 'yes' }, /data-skip-quiet/],
    [{ layout: 'column' }, /data-layout/], [{ count: '0' }, /data-count/], [{ count: '2.5' }, /data-count/], [{ speed: 'fast' }, /data-speed/], [{ loop: 'yes' }, /data-loop/], [{ cap: '0' }, /data-cap/], [{ controls: 'left' }, /data-controls/], [{ fit: 'width' }, /data-fit/], [{ scrubber: 'bar' }, /data-scrubber/], [{ followBoards: '4' }, /data-follow-boards="4" is not one of 2, 3/], [{ speeds: '10,50', speed: '300' }, /data-speed="300" is not one of data-speeds/], [{ speeds: '10,fast' }, /data-speeds/], [{ speeds: '0,5' }, /data-speeds/],
  ]) assert.throws(() => D.embedOptions(data), message, JSON.stringify(data));
});

test('the blog data follows the shortest win and refuses a policy view of another run', () => {
  const out = path.join(root, 'blog-data'), { files, followed: shortestWin } = writeData({ site: synthetic, out, tier: 'high', set: 'all', count: 3 });
  const truthWin = truth.high.map((e, i) => [e.decisions, i]).filter(([, i]) => truth.high[i].outcome === 'win').sort((a, b) => a[0] - b[0] || a[1] - b[1])[0][1];
  assert.equal(shortestWin.index, truthWin);
  assert.ok(files.includes('sprites.png') && files.includes('high/all/timeline-n3.bin.gz'));
  const trimmed = json(out, 'manifest.json');
  assert.deepEqual(trimmed.tiers.map(t => [t.name, t.sets.map(s => s.name), t.sets[0].timelines.map(l => l.count)]), [['high', ['all'], [3]]]);
  for (const name of Object.keys(trimmed.files)) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(out, name))).digest('hex'), trimmed.files[name], name);
  const bundle = path.join(root, 'policy'), bytes = zlib.gzipSync(Buffer.from('frames'));
  fs.mkdirSync(bundle);
  fs.writeFileSync(path.join(bundle, 'policy-view.bin.gz'), bytes);
  const policy = seed => fs.writeFileSync(path.join(bundle, 'manifest.json'), JSON.stringify({
    schema: 'craftax-policy-view/v1', gzipSha256: crypto.createHash('sha256').update(bytes).digest('hex'), samplingSeed: seed, decisions: shortestWin.decisions,
  }));
  policy('12345');
  assert.throws(() => addPolicy({ data: out, bundle }), /sampling seed 12345.*but the embed follows seed/);
  assert.ok(!fs.existsSync(path.join(out, 'policy-view.bin.gz')));
  policy(shortestWin.sampling_seed);
  assert.equal(addPolicy({ data: out, bundle }).index, shortestWin.index);
  assert.ok(fs.readFileSync(path.join(out, 'policy-view.bin.gz')).equals(bytes));
});

test('the blog data of a time-mapped set carries its time maps and follows its pinned run', () => {
  const out = path.join(root, 'blog-wins'), { files, followed } = writeData({ site: synthetic, out, tier: 'high', set: 'wins', count: 3 });
  assert.equal(followed.index, 0);
  assert.ok(files.includes('high/wins/g0/keeps.bin.gz') && !files.some(name => name.includes('timeline')));
  assert.deepEqual(json(out, 'manifest.json').tiers[0].sets[0].timelines, []);
  const { set } = load(out, 'high', 'wins', 2);
  assert.equal(set.mapped, true);
});
