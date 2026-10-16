// Write a synthetic ghost-run data directory in ../FORMAT.md's format for the
// decoder's seek tests, which need what the fixture lacks: episodes past a
// keyframe, on many floors, with escapes, over two groups and several
// creature windows. Episodes are seeded random walks
// drifting toward each floor's down ladder; nothing here plays Craftax. Every
// tier has an `all` set, whose timelines keep every decision; the high tier
// also has a time-mapped `wins` set of the same episodes (its runs need not
// win: the decoder does not ask), the first unbroken and the others keeping
// random runs of about half their decisions.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const MAP = 48, FLOORS = 9, CELLS = MAP * MAP, TILES = FLOORS * CELLS;
const STRIDE = 4;
const STEP_ROW = [0, 0, 0, -1, 1], STEP_COL = [0, -1, 1, 0, 0];
const INTERACTS = new Set([5, 7, 8, 9, 10, 24, 26, 27, 28]);
const SPECIES = [8, 3, 8, 8, 8];
const REWARDS = [...Array(25).fill(1), 3, 3, 3, 3, 3, 5, 5, 5, 8, 8, 8, 3, 3, 3, 3, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 5, 5, 5, 5];

const SPEC = {
  seed: 7, counts: [2, 3], window: 2048,
  tiers: [
    { name: 'early', episodes: 3, meanDecisions: 900, maxDecisions: 3000, maxFloor: 2, wins: 0 },
    { name: 'high', episodes: 3, meanDecisions: 9000, maxDecisions: 20000, maxFloor: 8, wins: 0.3 },
  ],
};

function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let x = Math.imul(a ^ (a >>> 15), a | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

class Bytes {
  constructor() { this.buffer = new Uint8Array(256); this.length = 0; }
  push(...values) { this.append(values); }
  append(values) {
    if (this.length + values.length > this.buffer.length) {
      const grown = new Uint8Array(Math.max(this.buffer.length * 2, this.length + values.length));
      grown.set(this.buffer.subarray(0, this.length));
      this.buffer = grown;
    }
    this.buffer.set(values, this.length);
    this.length += values.length;
  }
  bytes() { return this.buffer.subarray(0, this.length); }
}

function world(rand) {
  const bytes = new Uint8Array(3 * TILES + 4 * FLOORS);
  const floorBlocks = [[2, 2, 2, 5, 3, 4], [7, 7, 17, 4], [7, 4, 8, 9], [7, 17, 3], [7, 17, 23], [7, 4, 20], [25, 14, 28, 7], [26, 29, 7], [27, 33, 34, 7]];
  for (let f = 0; f < FLOORS; f++) {
    for (let cell = 0; cell < CELLS; cell++) {
      const blocks = floorBlocks[f];
      bytes[f * CELLS + cell] = blocks[Math.floor(rand() * blocks.length)];
      bytes[2 * TILES + f * CELLS + cell] = f ? 90 + Math.floor(rand() * 165) : 255;
    }
    const down = Math.floor(rand() * CELLS), up = Math.floor(rand() * CELLS);
    if (f < FLOORS - 1) bytes[TILES + f * CELLS + down] = 2;
    if (f) bytes[TILES + f * CELLS + up] = 3;
    bytes.set([down / MAP | 0, down % MAP], 3 * TILES + 2 * f);
    bytes.set([up / MAP | 0, up % MAP], 3 * TILES + 2 * FLOORS + 2 * f);
  }
  return bytes;
}

// One episode as the builder's Ghost holds it, with every state and creature
// sample kept when asked, for the decoder tests.
function episode(rand, base, tier, window) {
  const ladder = (f, which) => base.subarray(3 * TILES + which * 2 * FLOORS + 2 * f, 3 * TILES + which * 2 * FLOORS + 2 * f + 2);
  const length = Math.min(tier.maxDecisions, Math.max(8, Math.round(-Math.log(1 - rand()) * tier.meanDecisions)));
  const deepest = Math.floor(rand() * (tier.maxFloor + 1));
  const players = new Uint8Array(length), map = [], achievements = [], escapes = [], windows = [], samples = [];
  const states = new Uint8Array((length + 1) * 4), floorFirst = new Array(FLOORS).fill(-1), unlocked = new Set();
  let floor = 0, row = 24, col = 24, facing = 3, creatures = [];
  const escape = Math.floor(rand() * length);
  floorFirst[0] = 0;
  for (let t = 0; ; t++) {
    states.set([floor, row, col, facing], t * 4);
    if (t % STRIDE === 0 && t < length) {
      // A projectile (class 3 or 4) flies one of the four facings, 1-4, taken
      // from its tile so the draws, and the episodes, stay as they were.
      if (rand() < 0.05 || creatures.length === 0) creatures = Array.from({ length: Math.floor(rand() * 5) }, () => {
        const klass = Math.floor(rand() * 5), species = Math.floor(rand() * SPECIES[klass]);
        const r = Math.max(0, Math.min(47, row + Math.floor(rand() * 11) - 5)), c = Math.max(0, Math.min(47, col + Math.floor(rand() * 11) - 5));
        return [klass, species, r, c, klass >= 3 ? 1 + (r + c) % 4 : 0];
      });
      for (const c of creatures) { c[2] = Math.max(0, Math.min(47, c[2] + Math.floor(rand() * 3) - 1)); c[3] = Math.max(0, Math.min(47, c[3] + Math.floor(rand() * 3) - 1)); }
      const into = windows[Math.floor(t / window)] ??= new Bytes();
      into.push(creatures.length);
      for (const [klass, species, r, c, facing] of creatures) into.push(klass << 4 | species, r, c, facing);
      samples.push([t, creatures.map(c => [...c])]);
    }
    if (t === length) break;
    const down = ladder(floor, 0);
    let action, byte;
    if (row === down[0] && col === down[1] && floor < deepest) {
      action = 18;
      floor += 1;
      [row, col] = ladder(floor, 1);
      byte = action | 0x80;
      if (floorFirst[floor] < 0) floorFirst[floor] = t + 1;
    } else {
      if (rand() < 0.6) {
        const toward = floor < deepest && rand() < 0.35, dr = Math.sign(down[0] - row), dc = Math.sign(down[1] - col);
        action = toward ? (dc < 0 ? 1 : dc > 0 ? 2 : dr < 0 ? 3 : 4) : 1 + Math.floor(rand() * 4);
      } else action = [5, 5, 5, 7, 0, 6, 24, 28][Math.floor(rand() * 8)];
      byte = action;
      if (INTERACTS.has(action)) {
        // A torch (28) keeps the world's block and puts item 1 on the tile.
        const tr = row + STEP_ROW[facing], tc = col + STEP_COL[facing], at = floor * CELLS + tr * MAP + tc;
        if (tr >= 0 && tr < MAP && tc >= 0 && tc < MAP && rand() < 0.08) map.push(action === 28 ? [t, floor, tr, tc, base[at], 1] : [t, floor, tr, tc, action === 7 ? 4 : 7, 0]);
      }
      if (action >= 1 && action <= 4) {
        facing = action;
        const nr = row + STEP_ROW[action], nc = col + STEP_COL[action];
        if (nr >= 0 && nr < MAP && nc >= 0 && nc < MAP && rand() < 0.9) { row = nr; col = nc; byte |= 0x40; }
      }
      if (t === escape) {
        row = Math.floor(rand() * MAP); col = Math.floor(rand() * MAP); facing = 1 + Math.floor(rand() * 4);
        escapes.push([t, floor, row, col, facing]);
      }
    }
    if (rand() < 0.003) {
      const achievement = Math.floor(rand() * REWARDS.length);
      if (!unlocked.has(achievement)) { unlocked.add(achievement); achievements.push([t, achievement]); }
    }
    players[t] = byte;
  }
  const outcome = length === tier.maxDecisions ? 'timeout' : rand() < tier.wins ? 'win' : 'death';
  return {
    decisions: length, players, map, achievements, escapes, windows, samples, states, outcome, floorFirst,
    achievementReturn: [...unlocked].reduce((sum, a) => sum + REWARDS[a], 0), end: [floor, row, col, facing],
  };
}

// A table as events.bin holds them: u32 row count, then each column of
// `width`-byte values, padded to 4 bytes.
function table(rows, widths) {
  const out = new Bytes(), bytes = (v, width) => Array.from({ length: width }, (_, i) => (v >>> (8 * i)) & 255);
  out.append(bytes(rows.length, 4));
  widths.forEach((width, k) => {
    for (const row of rows) out.append(bytes(row[k], width));
    out.append(new Uint8Array((4 - (rows.length * width) % 4) % 4));
  });
  return Buffer.from(out.bytes());
}

function windowFile(runs) {
  const header = new DataView(new ArrayBuffer(4 * (runs.length + 2)));
  header.setUint32(0, runs.length, true);
  let end = 0;
  runs.forEach((run, i) => { header.setUint32(4 + 4 * i, end, true); end += run.length; });
  header.setUint32(4 + 4 * runs.length, end, true);
  return Buffer.concat([Buffer.from(header.buffer), ...runs.map(run => Buffer.from(run))]);
}

const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

// Kept runs over `decisions`, from 0 to the end: alternating kept and skipped
// stretches of up to 400 decisions, about half of them kept.
function keptRuns(rand, decisions) {
  const runs = [];
  for (let at = 0, keep = true; at < decisions; keep = !keep) {
    const stop = Math.min(decisions, at + 1 + Math.floor(rand() * 400));
    if (keep || stop === decisions) runs.push([at, stop]);
    at = stop;
  }
  return runs.reduce((merged, run) => (merged.length && merged.at(-1)[1] === run[0] ? (merged.at(-1)[1] = run[1], merged) : [...merged, run]), []);
}

function stats(episodes) {
  const n = episodes.length, deaths = new Array(FLOORS).fill(0), escapes = episodes.reduce((sum, e) => sum + e.escapes.length, 0);
  for (const e of episodes) if (e.outcome === 'death') deaths[e.end[0]]++;
  return {
    count: n, mean_return: episodes.reduce((s, e) => s + e.achievementReturn, 0) / n, mean_decisions: episodes.reduce((s, e) => s + e.decisions, 0) / n,
    reached: Array.from({ length: FLOORS }, (_, f) => episodes.filter(e => e.floorFirst[f] >= 0).length),
    deaths, timeouts: episodes.filter(e => e.outcome === 'timeout').length, wins: episodes.filter(e => e.outcome === 'win').length, escapes,
  };
}

// Write the data directory; return per tier its episodes, with every state
// and creature sample.
export function writeSynthetic(dir) {
  const spec = SPEC, rand = random(spec.seed), files = {}, sizes = {}, truth = {};
  const write = (name, bytes) => {
    const file = path.join(dir, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, bytes);
    files[name] = sha(bytes);
    sizes[name] = bytes.length;
  };
  const gzip = bytes => zlib.gzipSync(bytes, { level: 9 }), base = world(rand), tiers = [];
  write('world.bin.gz', gzip(base));
  for (const tier of spec.tiers) {
    const episodes = Array.from({ length: tier.episodes }, () => episode(rand, base, tier, spec.window));
    truth[tier.name] = episodes;
    const sets = [setOf(tier, episodes, 'all', null)];
    if (tier.name === 'high') sets.push(setOf(tier, episodes, 'wins', episodes.map((e, i) => (i ? keptRuns(rand, e.decisions) : [[0, e.decisions]]))));
    tiers.push({ name: tier.name, arm: tiers.length, sources: [{ root: 'synthetic', capped: false, episodes: tier.episodes, provenance: { policy: `random walks to floor ${tier.maxFloor}` } }], sets });
  }

  // One set's groups, and its timelines or, with `keeps`, its time maps.
  function setOf(tier, episodes, name, keeps) {
    const groups = [];
    spec.counts.forEach((stop, g) => {
      const first = g ? spec.counts[g - 1] : 0, members = episodes.slice(first, stop), prefix = `${tier.name}/${name}/g${g}`;
      const windows = Math.ceil(Math.max(...members.map(e => e.decisions)) / spec.window);
      const players = new Bytes(), starts = [0, 0, 0], listing = [];
      members.forEach((e, i) => {
        listing.push({
          index: first + i, source: 0, ordinal: first + i, sampling_seed: String(first + i), decisions: e.decisions, outcome: e.outcome, end: e.end,
          achievement_return: e.achievementReturn, floor_first: e.floorFirst, players: players.length,
          map: [starts[0], e.map.length], achievements: [starts[1], e.achievements.length], escapes: [starts[2], e.escapes.length],
        });
        players.append(e.players);
        starts[0] += e.map.length; starts[1] += e.achievements.length; starts[2] += e.escapes.length;
      });
      write(`${prefix}/episodes.json`, Buffer.from(JSON.stringify({ tier: tier.name, set: name, group: g, first, episodes: listing }) + '\n'));
      if (keeps) {
        const runs = keeps.slice(first, stop);
        write(`${prefix}/keeps.bin.gz`, gzip(Buffer.concat([table(runs.map(r => [r.length]), [4]), table(runs.flat(), [4, 4])])));
      }
      write(`${prefix}/players.bin.gz`, gzip(players.bytes()));
      write(`${prefix}/events.bin.gz`, gzip(Buffer.concat([
        table(members.flatMap(e => e.map), [4, 1, 1, 1, 1, 1]), table(members.flatMap(e => e.achievements), [4, 1]), table(members.flatMap(e => e.escapes), [4, 1, 1, 1, 1]),
      ])));
      for (let j = 0; j < windows; j++) write(`${prefix}/creatures-w${j}.bin.gz`, gzip(windowFile(members.map(e => e.windows[j]?.bytes() ?? new Uint8Array()))));
      groups.push({ path: prefix, first, count: members.length, decisions: members.reduce((s, e) => s + e.decisions, 0), windows });
    });
    const timelines = keeps ? [] : spec.counts.map(count => {
      const decisions = Math.max(...episodes.slice(0, count).map(e => e.decisions)), file = `${tier.name}/${name}/timeline-n${count}.bin.gz`;
      write(file, gzip(Buffer.concat([table(Array.from({ length: decisions }, () => [0]), [2]), table([[0, decisions]], [4, 4])])));
      return { count, path: file, decisions, kept: decisions, segments: 1 };
    });
    const outcomes = kind => episodes.filter(e => e.outcome === kind).length, lengths = keeps?.map(runs => runs.reduce((sum, [a, b]) => sum + b - a, 0));
    return {
      name, rule: 'Synthetic.', counts: spec.counts, episodes: tier.episodes,
      decisions: episodes.reduce((s, e) => s + e.decisions, 0), max_decisions: Math.max(...episodes.map(e => e.decisions)),
      composition: {
        run: tier.episodes, qualified: tier.episodes, deaths: outcomes('death'), wins: outcomes('win'), timeouts: outcomes('timeout'), truncated: 0,
        added_wins: 0, natural_win_share: 0, death_decisions: [], win_decisions: [],
      },
      groups, stats: spec.counts.map(count => stats(episodes.slice(0, count))), timelines,
      time_map: keeps ? {
        rule: { steps: Math.max(...lengths), levels: [[0, 0]] }, steps: Math.max(...lengths), kept: lengths.reduce((a, b) => a + b, 0), shortest: Math.min(...lengths),
        median: [...lengths].sort((a, b) => a - b)[lengths.length >> 1], levels: [lengths.length - 1, 0], unbroken: 0,
      } : null,
    };
  }
  const manifest = {
    format: 'craftax-ghosts/4', git_commit: 'synthetic', game_package_digest: 'synthetic', platform: 'synthetic', world_seed: spec.seed,
    start: [0, 24, 24, 3], creature_stride: STRIDE, window_decisions: spec.window, counts: spec.counts, short_decisions: 10_000,
    quiet: { per_live: 50, min_run: 64, keep: 8 }, achievement_rewards: REWARDS,
    tiers, files, sizes,
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');
  return { manifest, truth };
}
