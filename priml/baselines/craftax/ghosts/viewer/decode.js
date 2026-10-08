'use strict';
// Ghost-run decoding and the playback state at a decision, without the DOM,
// and the rules the viewer picks runs and reads its options by. ghosts.js
// draws what this computes; ghosts.test.mjs runs it in node against the
// builder's fixture. ../FORMAT.md specifies every file read here, and
// ../layout.py is the reference decoder this must agree with.
//
// Time is the decision index t. State t is the game before decision t; an
// episode of T decisions has states 0..T and has ended at t >= T.
//
// Playback runs in display steps u. In a set without time maps a display step
// is a decision: u shows state min(u, T) of every episode. In a time-mapped
// set (the wins set, FORMAT.md keeps.bin) episode e shows, at u < L_e, the
// state before the u-th decision its kept runs cover, and has ended at
// u >= L_e; decisionAt is that map, stepOf its inverse.
var GhostDecode = (() => {
  const MAP = 48, FLOORS = 9, CELLS = MAP * MAP, TILES = FLOORS * CELLS;
  // Position and heat keyframes every 1,024 decisions bound a seek's replay.
  const KEY_SHIFT = 10, KEYFRAME = 1 << KEY_SHIFT;
  const OCCUPANCY_SHIFT = 6;
  // A ghost's trail ring: its last 240 moves, 241 tiles.
  const TRAIL = 241;
  const MOVED = 0x40, FLOOR_CHANGE = 0x80, ACTION_MASK = 0x3f;
  const DESCEND = 18;
  const ENTERED = 1, INTERACTED = 2, NEW_FLOOR = 4;
  const STEP_ROW = Int8Array.of(0, 0, 0, -1, 1), STEP_COL = Int8Array.of(0, -1, 1, 0, 0);
  const INTERACTS = new Uint8Array(64);
  for (const action of [5, 7, 8, 9, 10, 24, 26, 27, 28]) INTERACTS[action] = 1;
  const OUTCOMES = ['death', 'timeout', 'win', 'truncated'];
  const ITEMS = 5;
  // Creature slots on a floor: 3 melee, 3 passive, 2 ranged, 3 and 3 projectiles.
  const MAX_CREATURES = 14;
  const WORLD_BYTES = 3 * TILES + 4 * FLOORS;
  // The policy's view: 9 rows by 11 columns about the player, cut at the
  // map's edges (game/observation.py).
  const VIEW_ROWS = 4, VIEW_COLS = 5, NEVER = 0x7fffffff, WORLD = 0xffff;
  // A placed torch (game/rules.py place, PLACE_TORCH) lights each tile of the
  // 9 x 9 square about it, in float32 as the game does: the tile's light
  // l / 255 plus max(0, 1 - sqrt(dr^2 + dc^2) / 5), clamped to [0, 1], times
  // 255 truncated back to a byte. TORCH_KERNEL holds the added term.
  const TORCH = 1, TORCH_REACH = 4, f32 = Math.fround;
  const TORCH_KERNEL = Float32Array.from({ length: 81 }, (_, k) => {
    const dr = (k / 9 | 0) - TORCH_REACH, dc = k % 9 - TORCH_REACH;
    return Math.max(0, f32(1 - f32(f32(Math.sqrt(dr * dr + dc * dc)) / 5)));
  });
  const torchLit = (light, k) => Math.trunc(f32(Math.min(1, Math.max(0, f32(f32(light / 255) + TORCH_KERNEL[k]))) * 255));

  // world.bin: block, item and light planes, each [floor][row][col], then the
  // down ladders' and the up ladders' (row, col) per floor.
  function parseWorld(bytes) {
    if (bytes.length !== WORLD_BYTES) throw Error(`world.bin holds ${bytes.length} bytes, not ${WORLD_BYTES}.`);
    const ladders = bytes.subarray(3 * TILES), down = new Int32Array(FLOORS), up = new Int32Array(FLOORS);
    for (let f = 0; f < FLOORS; f++) {
      down[f] = ladders[2 * f] * MAP + ladders[2 * f + 1];
      up[f] = ladders[2 * FLOORS + 2 * f] * MAP + ladders[2 * FLOORS + 2 * f + 1];
    }
    return { block: bytes.subarray(0, TILES), item: bytes.subarray(TILES, 2 * TILES), light: bytes.subarray(2 * TILES, 3 * TILES), down, up };
  }

  // Tables back to back, each a u32 row count then its columns, each padded
  // to 4 bytes, so every column is a typed-array view. `tables` lists each
  // table's column types; returns each table's columns.
  function parseTables(bytes, name, tables) {
    if (bytes.byteOffset % 4) bytes = bytes.slice();
    let at = 0;
    const out = tables.map(types => {
      const n = new DataView(bytes.buffer, bytes.byteOffset + at, 4).getUint32(0, true);
      at += 4;
      return types.map(Type => {
        const column = new Type(bytes.buffer, bytes.byteOffset + at, n);
        at += Math.ceil(n * Type.BYTES_PER_ELEMENT / 4) * 4;
        return column;
      });
    });
    if (at !== bytes.length) throw Error(`${name} has ${bytes.length - at} bytes after its tables.`);
    return out;
  }

  // events.bin: the map, achievement and escape tables.
  function parseEvents(bytes) {
    const [map, achievements, escapes] = parseTables(bytes, 'events.bin', [
      [Uint32Array, Uint8Array, Uint8Array, Uint8Array, Uint8Array, Uint8Array], [Uint32Array, Uint8Array], [Uint32Array, Uint8Array, Uint8Array, Uint8Array, Uint8Array],
    ]);
    return { map, achievements, escapes };
  }

  // timeline-n<count>.bin: per-decision activity of the first `count`
  // episodes and the [start, stop) segments kept when quiet stretches are
  // skipped, as a Timeline that also shows the end state `decisions`.
  function parseTimeline(bytes, decisions) {
    const [[activity], [starts, stops]] = parseTables(bytes, 'timeline.bin', [[Uint16Array], [Uint32Array, Uint32Array]]);
    if (activity.length !== decisions) throw Error(`A timeline covers ${activity.length} decisions, not the episodes' ${decisions}.`);
    return { activity, starts, stops, timeline: new Timeline([...Array.from(starts, (start, i) => [start, stops[i]]), [decisions, decisions + 1]]) };
  }

  // keeps.bin: per episode its count of kept runs, then every run's [start,
  // stop) back to back; returns each episode's runs as [starts, stops].
  function parseKeeps(bytes, episodes) {
    const [[counts], [starts, stops]] = parseTables(bytes, 'keeps.bin', [[Uint32Array], [Uint32Array, Uint32Array]]);
    if (counts.length !== episodes) throw Error(`keeps.bin holds ${counts.length} episodes' time maps, not ${episodes}.`);
    let at = 0;
    const out = Array.from(counts, count => {
      const runs = [starts.subarray(at, at + count), stops.subarray(at, at + count)];
      at += count;
      return runs;
    });
    if (at !== starts.length) throw Error(`keeps.bin's run counts cover ${at} runs, not its ${starts.length}.`);
    return out;
  }

  // The decisions an episode of `decisions` decisions shows, one per display
  // step, from its kept runs; null when they keep every decision.
  function shownDecisions([starts, stops], decisions, index) {
    const n = starts.length;
    if (!n || starts[0] !== 0 || stops[n - 1] !== decisions) throw Error(`Episode ${index}'s time map does not run from decision 0 to its end, ${decisions}.`);
    let length = 0;
    for (let i = 0; i < n; i++) {
      if (stops[i] <= starts[i] || (i && starts[i] <= stops[i - 1])) throw Error(`Episode ${index}'s kept runs are not increasing, disjoint and non-empty.`);
      length += stops[i] - starts[i];
    }
    if (length === decisions) return null;
    const shown = new Int32Array(length);
    for (let i = 0, u = 0; i < n; i++) for (let d = starts[i]; d < stops[i]; d++) shown[u++] = d;
    return shown;
  }

  // sleep.bin: per episode its sleeps (decision, ticks) and, per sleep, its
  // samples' creature runs, as offsets into the group's runs; returns
  // {episodes: [{decisions, ticks, samples: [[run offsets], ...]}], runs}.
  // A sleep of k ticks has floor((k - 1) / stride) samples (FORMAT.md).
  function parseSleep(bytes, episodes, stride) {
    const [[counts], [decisions, ticks], [runAt, changes], , [runs]] = parseTables(bytes, 'sleep.bin', [
      [Uint32Array], [Uint32Array, Uint32Array], [Uint32Array, Uint32Array], [Uint8Array, Uint8Array, Uint8Array, Uint8Array, Uint8Array], [Uint8Array],
    ]);
    if (counts.length !== episodes) throw Error(`sleep.bin holds ${counts.length} episodes' sleeps, not ${episodes}.`);
    let sleep = 0, sample = 0;
    const out = Array.from(counts, count => {
      const own = { decisions: decisions.subarray(sleep, sleep + count), ticks: ticks.subarray(sleep, sleep + count), samples: [] };
      for (let k = 0; k < count; k++, sleep++) {
        const m = Math.floor((ticks[sleep] - 1) / stride);
        own.samples.push(runAt.subarray(sample, sample + m));
        sample += m;
      }
      return own;
    });
    if (sleep !== decisions.length || sample !== runAt.length) throw Error('sleep.bin\'s sleeps and samples do not add up.');
    void changes;
    return { episodes: out, runs };
  }

  // The view that shows sleep: each kept decision a step, a kept sleep's
  // decision followed by a step per sample of it. Returns the decision each
  // step shows (repeated through a sleep), each step's sample (1-based, 0 for
  // the state before the decision) and each sample step's creature run in
  // the group's runs (-1 elsewhere).
  function sleepSteps([starts, stops], decisions, sleeps, index) {
    shownDecisions([starts, stops], decisions, index);
    const asleep = new Map(Array.from(sleeps.decisions, (d, k) => [d, sleeps.samples[k]]));
    let length = 0;
    for (let i = 0; i < starts.length; i++) for (let d = starts[i]; d < stops[i]; d++) length += 1 + (asleep.get(d)?.length ?? 0);
    const shown = new Int32Array(length), sub = new Uint16Array(length), runAt = new Int32Array(length).fill(-1);
    for (let i = 0, u = 0; i < starts.length; i++) {
      for (let d = starts[i]; d < stops[i]; d++) {
        shown[u++] = d;
        const runs = asleep.get(d);
        for (let k = 0; runs && k < runs.length; k++, u++) [shown[u], sub[u], runAt[u]] = [d, k + 1, runs[k]];
      }
    }
    for (const d of sleeps.decisions) if (!shown.includes(d)) throw Error(`Episode ${index}'s view that shows sleep skips its sleep at decision ${d}.`);
    return { shown, sub, runAt };
  }

  // Episode e's sleep sample at display step u: 0 when the step shows a
  // decision's state, else its sample's 1-based index in the sleep.
  function sampleAt(set, e, u) {
    const sub = set.sub[e];
    return sub && u < sub.length ? sub[u] : 0;
  }

  // The decision episode e shows at display step u: state u of a set without
  // time maps; its end, T, from u >= L_e on.
  function decisionAt(set, e, u) {
    const shown = set.shown[e];
    if (!shown) return Math.min(u, set.decisions[e]);
    return u < shown.length ? shown[u] : set.decisions[e];
  }

  // The first display step at which episode e shows state s or later, s <= T.
  function stepOf(set, e, s) {
    const shown = set.shown[e];
    return shown ? lowerBound(shown, s) : Math.min(s, set.decisions[e]);
  }

  // An item's sprite in sprites.png: the sheet holds the items from id 42 in
  // the game's ItemType order (game/state.py), 42 for none (blank), 43 a
  // torch, 44 a ladder down, 45 a ladder up, 46 a blocked ladder down, so an
  // item byte as the State holds it (world.bin, map events) is sprite 42 +
  // item. The policy's observation stores item + 1 (hud.js draws it so).
  const ITEM_SPRITE = 42;
  const itemSprite = item => ITEM_SPRITE + item;

  // A tile's block and item as one code, 0 when it equals the world's.
  function tileCode(world, tile, block, item) {
    return block === world.block[tile] && item === world.item[tile] ? 0 : block * ITEMS + item + 1;
  }

  // The displayed episodes: groups 0..k of one tier, in index order. Each
  // group is {listing: episodes.json, players, events, keeps, sleep},
  // decompressed; keeps (keeps.bin) only in a time-mapped set, where every
  // group has one, and in the view that shows sleep keeps-sleep.bin with its
  // sleep.bin. length[e] is L_e, the display steps episode e plays (T
  // without a time map), and maxSteps the largest.
  function episodeSet(world, groups, manifest) {
    const n = groups.reduce((sum, g) => sum + g.listing.episodes.length, 0), mapped = groups.some(g => g.keeps);
    if (groups.some(g => !g.keeps === mapped)) throw Error('Some groups of the set have time maps and some do not.');
    const set = {
      n, world, start: manifest.start, window: manifest.window_decisions, stride: manifest.creature_stride, maxDecisions: 0, maxSteps: 0, mapped,
      // Format 4 adds each creature's facing (FORMAT.md, creatures-w<j>.bin).
      creatureBytes: manifest.format === 'craftax-ghosts/4' ? 4 : 3,
      decisions: new Int32Array(n), length: new Int32Array(n), outcome: new Uint8Array(n), ret: new Float64Array(n), endTile: new Int32Array(n), endFacing: new Uint8Array(n),
      floorFirst: new Int32Array(n * FLOORS), group: new Uint8Array(n), local: new Int32Array(n), seeds: [], shown: [],
      sleepView: groups.some(g => g.sleep), sub: [], runAt: [], sleepRuns: [], sleeps: [],
      players: [], map: [], score: [], escape: [], groupSizes: groups.map(g => g.listing.episodes.length),
    };
    let e = 0;
    groups.forEach((group, g) => {
      const events = parseEvents(group.events), [decision, floor, row, col, block, item] = events.map;
      const keeps = mapped ? parseKeeps(group.keeps, group.listing.episodes.length) : null;
      const sleep = group.sleep ? parseSleep(group.sleep, group.listing.episodes.length, manifest.sleep_stride) : null;
      set.sleepRuns.push(sleep?.runs ?? null);
      const tile = Int32Array.from(decision, (_, k) => floor[k] * CELLS + row[k] * MAP + col[k]);
      const code = Uint8Array.from(decision, (_, k) => tileCode(world, tile[k], block[k], item[k]));
      const [when, which] = events.achievements, [escaped, ...after] = events.escapes;
      group.listing.episodes.forEach((episode, i) => {
        const { decisions, players } = episode, [endFloor, endRow, endCol, endFacing] = episode.end;
        if (players + decisions > group.players.length) throw Error(`Episode ${episode.index}'s player bytes run past players.bin.`);
        if (!OUTCOMES.includes(episode.outcome)) throw Error(`Episode ${episode.index} has unknown outcome ${episode.outcome}.`);
        set.decisions[e] = decisions;
        set.outcome[e] = OUTCOMES.indexOf(episode.outcome);
        set.ret[e] = episode.achievement_return;
        set.endTile[e] = endFloor * CELLS + endRow * MAP + endCol;
        set.endFacing[e] = endFacing;
        set.floorFirst.set(episode.floor_first, e * FLOORS);
        set.seeds.push(episode.sampling_seed);
        const steps = sleep ? sleepSteps(keeps[i], decisions, sleep.episodes[i], episode.index) : null;
        const shown = steps ? steps.shown : keeps ? shownDecisions(keeps[i], decisions, episode.index) : null;
        set.shown.push(shown);
        set.sub.push(steps?.sub ?? null);
        set.runAt.push(steps?.runAt ?? null);
        set.sleeps.push(sleep?.episodes[i] ?? null);
        set.length[e] = shown ? shown.length : decisions;
        set.maxSteps = Math.max(set.maxSteps, set.length[e]);
        set.group[e] = g;
        set.local[e] = i;
        set.players.push(group.players.subarray(players, players + decisions));
        const [ms, mc] = episode.map, [as, ac] = episode.achievements, [es, ec] = episode.escapes;
        set.map.push({ decision: decision.subarray(ms, ms + mc), tile: tile.subarray(ms, ms + mc), code: code.subarray(ms, ms + mc) });
        let total = 0;
        set.score.push({ decision: when.subarray(as, as + ac), value: Int32Array.from(which.subarray(as, as + ac), a => (total += manifest.achievement_rewards[a])) });
        set.escape.push({ decision: escaped.subarray(es, es + ec), state: after.map(column => column.subarray(es, es + ec)) });
        set.maxDecisions = Math.max(set.maxDecisions, decisions);
        e++;
      });
    });
    return set;
  }

  function cursors(n) {
    return { floor: new Uint8Array(n), row: new Uint8Array(n), col: new Uint8Array(n), facing: new Uint8Array(n), escape: new Int32Array(n), targetRow: 0, targetCol: 0 };
  }

  function begin(set, cur, e) {
    [cur.floor[e], cur.row[e], cur.col[e], cur.facing[e]] = set.start;
    cur.escape[e] = 0;
  }

  const tileOf = (cur, e) => cur.floor[e] * CELLS + cur.row[e] * MAP + cur.col[e];
  const onMap = (row, col) => row >= 0 && row < MAP && col >= 0 && col < MAP;

  // Play decision t of episode e on its cursor (FORMAT.md, players.bin).
  // Returns ENTERED when the player reached another tile, NEW_FLOOR when on
  // another floor, and INTERACTED with cur.targetRow/targetCol set to the
  // faced tile, which may lie off the map.
  function step(set, cur, e, t) {
    const byte = set.players[e][t], action = byte & ACTION_MASK;
    let floor = cur.floor[e], row = cur.row[e], col = cur.col[e], flags = 0;
    if (INTERACTS[action]) {
      const facing = cur.facing[e];
      cur.targetRow = row + STEP_ROW[facing];
      cur.targetCol = col + STEP_COL[facing];
      flags = INTERACTED;
    }
    const escape = set.escape[e], k = cur.escape[e];
    if (k < escape.decision.length && escape.decision[k] === t) {
      const [floors, rows, cols, facings] = escape.state;
      if (floors[k] !== floor) flags |= NEW_FLOOR | ENTERED;
      else if (rows[k] !== row || cols[k] !== col) flags |= ENTERED;
      cur.floor[e] = floors[k]; cur.row[e] = rows[k]; cur.col[e] = cols[k]; cur.facing[e] = facings[k];
      cur.escape[e] = k + 1;
      return flags;
    }
    if (action >= 1 && action <= 4) cur.facing[e] = action;
    if (byte & MOVED) {
      if (action < 1 || action > 4) throw Error(`Episode ${e} moves at decision ${t} without a move action.`);
      row += STEP_ROW[action]; col += STEP_COL[action]; flags |= ENTERED;
    } else if (byte & FLOOR_CHANGE) {
      floor += action === DESCEND ? 1 : -1;
      if (floor < 0 || floor >= FLOORS) throw Error(`Episode ${e} leaves the floors at decision ${t}.`);
      const tile = action === DESCEND ? set.world.up[floor] : set.world.down[floor];
      row = tile / MAP | 0; col = tile % MAP;
      flags |= ENTERED | NEW_FLOOR;
    }
    cur.floor[e] = floor; cur.row[e] = row; cur.col[e] = col;
    return flags;
  }

  // One episode in full: states [T + 1][floor, row, col, facing] and its
  // interaction targets [t, floor, row, col], as layout.py decodes them.
  function decodeEpisode(set, e) {
    const cur = cursors(set.n), end = set.decisions[e], states = new Uint8Array((end + 1) * 4), interactions = [];
    begin(set, cur, e);
    for (let t = 0; ; t++) {
      states.set([cur.floor[e], cur.row[e], cur.col[e], cur.facing[e]], t * 4);
      if (t === end) return { states, interactions };
      const floor = cur.floor[e];
      if (step(set, cur, e, t) & INTERACTED) interactions.push([t, floor, cur.targetRow, cur.targetCol]);
    }
  }

  // One pass over every decision: position keyframes, cumulative heat
  // keyframes and floor occupancy every 64 display steps; checks each decoded
  // end and first floor entry against episodes.json. A generator, so the page
  // can yield to the browser between episodes.
  //
  // reveal[tile] is the first display step at which any episode's view (9 x
  // 11 about the player) has held the tile, NEVER if none has: the fog of war
  // lifts there. light (torchEvents) is each episode's light changes.
  //
  // heat[k] holds, through display step k * KEYFRAME, the summed tile entries
  // (movement, first TILES) and interactions on the faced tile (next TILES).
  // Decision t's entry shows from the first display step past it, the count
  // of shown decisions up to t, which is t + 1 without a time map.
  function* precompute(set) {
    const n = set.n, last = set.maxSteps, keys = (last >> KEY_SHIFT) + 1;
    const key = new Uint8Array(n * keys * 4), heat = new Uint32Array((keys + 1) * 2 * TILES);
    const occupancy = new Uint16Array(((last >> OCCUPANCY_SHIFT) + 1) * FLOORS);
    const firstEntry = new Int32Array(FLOORS), cur = cursors(n), reveal = new Int32Array(TILES).fill(NEVER);
    let work = 0;
    for (let e = 0; e < n; e++) {
      begin(set, cur, e);
      firstEntry.fill(-1);
      firstEntry[cur.floor[e]] = 0;
      heat[tileOf(cur, e)] += 1;
      see(reveal, cur.floor[e], cur.row[e], cur.col[e], 0, -9, -9);
      const end = set.decisions[e], length = set.length[e], shown = set.shown[e];
      // u: the display steps whose decisions are before t, then up to t.
      for (let t = 0, u = 0; t < end; t++) {
        // A sleep's sample steps repeat its decision: the player holds its tile.
        while (u < length && (shown ? shown[u] : u) === t) {
          if ((u & (KEYFRAME - 1)) === 0) key.set([cur.floor[e], cur.row[e], cur.col[e], cur.facing[e]], (e * keys + (u >> KEY_SHIFT)) * 4);
          if ((u & ((1 << OCCUPANCY_SHIFT) - 1)) === 0) occupancy[(u >> OCCUPANCY_SHIFT) * FLOORS + cur.floor[e]]++;
          u++;
        }
        const floor = cur.floor[e], row = cur.row[e], col = cur.col[e], flags = step(set, cur, e, t);
        if (!flags) continue;
        // The state after decision t first shows at display step u.
        if (flags & ENTERED) see(reveal, cur.floor[e], cur.row[e], cur.col[e], u, flags & NEW_FLOOR ? -9 : row, col);
        const slot = (((u - 1) >> KEY_SHIFT) + 1) * 2 * TILES;
        if (flags & INTERACTED && onMap(cur.targetRow, cur.targetCol)) heat[slot + TILES + floor * CELLS + cur.targetRow * MAP + cur.targetCol]++;
        if (flags & ENTERED) heat[slot + tileOf(cur, e)]++;
        if (flags & NEW_FLOOR && firstEntry[cur.floor[e]] < 0) firstEntry[cur.floor[e]] = t + 1;
      }
      if (tileOf(cur, e) !== set.endTile[e] || cur.facing[e] !== set.endFacing[e]) throw Error(`Episode ${e} decodes to an end other than the one episodes.json records.`);
      for (let f = 0; f < FLOORS; f++) {
        if (firstEntry[f] !== set.floorFirst[e * FLOORS + f]) throw Error(`Episode ${e} first stands on floor ${f} at ${firstEntry[f]}, but episodes.json says ${set.floorFirst[e * FLOORS + f]}.`);
      }
      work += end;
      if (work > 1 << 21) { work = 0; yield (e + 1) / n; }
    }
    for (let k = 1; k <= keys; k++) {
      const into = k * 2 * TILES, from = into - 2 * TILES;
      for (let i = 0; i < 2 * TILES; i++) heat[into + i] += heat[from + i];
    }
    return { keys, key, heat, occupancy, reveal, light: torchEvents(set) };
  }

  // Mark the view about (row, col) on `floor` as seen by display step u in
  // reveal, or, after a one-tile move from (fromRow, fromCol), only the edge
  // the move brought into view. A function of its own: inside precompute's
  // generator it ran some 50 times slower.
  function see(reveal, floor, row, col, u, fromRow, fromCol) {
    const dr = row - fromRow, dc = col - fromCol, step = Math.abs(dr) + Math.abs(dc) === 1;
    const r0 = step && dr ? row + dr * VIEW_ROWS : row - VIEW_ROWS, r1 = step && dr ? r0 : row + VIEW_ROWS;
    const c0 = step && dc ? col + dc * VIEW_COLS : col - VIEW_COLS, c1 = step && dc ? c0 : col + VIEW_COLS;
    for (let r = Math.max(0, r0); r <= Math.min(MAP - 1, r1); r++) {
      for (let c = Math.max(0, c0), tile = floor * CELLS + r * MAP + c; c <= Math.min(MAP - 1, c1); c++, tile++) if (reveal[tile] > u) reveal[tile] = u;
    }
  }

  // Each episode's light changes from the torches it places, by the game's
  // rule (TORCH_KERNEL) on its own light map, the world's plus its earlier
  // torches': per episode, in order, the decision that placed the torch and
  // each tile it changed with its new light. A torch is a map row that puts
  // item 1 on a tile that held none; the game never takes one away. Also, per
  // tile, the (episode, change) pairs that touch it (tileStart, tileRun,
  // tileChange), to recount a tile when an episode ends.
  function torchEvents(set) {
    const { world } = set, light = world.light.slice(), item = world.item.slice(), touched = [], runs = [];
    let total = 0;
    for (let e = 0; e < set.n; e++) {
      const map = set.map[e], decision = [], tile = [], value = [];
      for (let k = 0; k < map.tile.length; k++) {
        const at = map.tile[k], code = map.code[k], now = code ? (code - 1) % ITEMS : world.item[at];
        const placed = now === TORCH && item[at] !== TORCH;
        touched.push(at);
        item[at] = now;
        if (!placed) continue;
        const floor = at / CELLS | 0, row = (at % CELLS) / MAP | 0, col = at % MAP;
        for (let r = Math.max(0, row - TORCH_REACH); r <= Math.min(MAP - 1, row + TORCH_REACH); r++) {
          for (let c = Math.max(0, col - TORCH_REACH); c <= Math.min(MAP - 1, col + TORCH_REACH); c++) {
            const lit = floor * CELLS + r * MAP + c, next = torchLit(light[lit], (r - row + TORCH_REACH) * 9 + c - col + TORCH_REACH);
            if (next === light[lit]) continue;
            touched.push(lit);
            light[lit] = next;
            decision.push(map.decision[k]); tile.push(lit); value.push(next);
          }
        }
      }
      for (const at of touched) { light[at] = world.light[at]; item[at] = world.item[at]; }
      touched.length = 0;
      runs.push({ decision: Int32Array.from(decision), tile: Uint16Array.from(tile), value: Uint8Array.from(value) });
      total += tile.length;
    }
    // Uint16 tiles and episodes (TILES and a set's runs are under 65,536): a
    // thousand runs of the boss tier hold some 3 million changes.
    if (set.n >= WORLD) throw Error(`A set of ${set.n} runs is too many to light.`);
    const tileStart = new Int32Array(TILES + 1), tileRun = new Uint16Array(total), tileChange = new Int32Array(total);
    for (const { tile } of runs) for (const at of tile) tileStart[at + 1]++;
    for (let at = 0; at < TILES; at++) tileStart[at + 1] += tileStart[at];
    const fill = tileStart.slice(0, TILES);
    runs.forEach(({ tile }, e) => tile.forEach((at, k) => { tileRun[fill[at]] = e; tileChange[fill[at]++] = k; }));
    return { runs, tileStart, tileRun, tileChange };
  }

  // Episode e's light map at state s: the world's, then its torches' changes
  // from decisions before s.
  function lightAt(set, light, e, s) {
    const out = set.world.light.slice(), run = light.runs[e];
    for (let k = 0; k < run.tile.length && run.decision[k] < s; k++) out[run.tile[k]] = run.value[k];
    return out;
  }

  // The world at display step t for every episode: each at its own decision
  // (decision[e], decisionAt), positions, heat through them, each living
  // ghost's last 240 moves on its floor, and the map changes of the living
  // episodes summarised per tile (how many changed it, and to what most
  // often). An ended episode keeps only its end tile.
  class Playback {
    constructor(set, pre) {
      const n = set.n;
      Object.assign(this, {
        set, pre, n, t: 0, cur: cursors(n), decision: new Int32Array(n), heat: new Uint32Array(2 * TILES),
        trail: new Int16Array(n * TRAIL), trailLength: new Uint8Array(n), trailHead: new Uint8Array(n),
        sampleFloor: new Uint8Array(n), changes: new Uint8Array(TILES * n), mapCursor: new Int32Array(n),
        tileCount: new Uint16Array(TILES), tileCode: new Uint8Array(TILES), dirty: new Uint8Array(TILES), dirtyList: [],
        histogram: new Uint16Array(ITEMS * 37 + 1),
        // The light the living episodes' torches have shed by now: per tile the
        // brightest of the world's and each living episode's own (lightAt),
        // and the episode it comes from (WORLD for the world's), so an
        // episode's end recounts only the tiles it lit brightest.
        light: set.world.light.slice(), lightOwner: new Uint16Array(TILES).fill(WORLD), lightCursor: new Int32Array(n), lightVersion: 0,
      });
      this._positions(0);
      this._rebuildMap(0);
    }

    alive(e) { return this.t < this.set.length[e]; }

    // Move to display step t: forward by replaying, anywhere else from keyframes.
    seek(t) {
      t = Math.max(0, Math.min(this.set.maxSteps, Math.round(t)));
      if (t >= this.t && t - this.t <= 2 * KEYFRAME) return this._advance(t);
      this._positions(t);
      if (t >= this.t) this._forwardMap(t);
      else this._rebuildMap(t);
      this.t = t;
    }

    // Recount the dirty tiles' map-change summaries; returns those tiles.
    summarize() {
      const { n, set, changes, histogram } = this, alive = [], updated = this.dirtyList;
      if (!updated.length) return updated;
      for (let e = 0; e < n; e++) if (this.t < set.length[e]) alive.push(e);
      for (const tile of this.dirtyList) {
        let count = 0, best = 0;
        histogram.fill(0);
        for (const e of alive) {
          const code = changes[tile * n + e];
          if (!code) continue;
          count++;
          if (++histogram[code] > histogram[best] || (histogram[code] === histogram[best] && code < best)) best = code;
        }
        this.tileCount[tile] = count;
        this.tileCode[tile] = best;
        this.dirty[tile] = 0;
      }
      this.dirtyList = [];
      return updated;
    }

    // The trail's points, oldest first, as tiles within the ghost's floor.
    trailOf(e) {
      const length = this.trailLength[e], head = this.trailHead[e], out = new Int16Array(length);
      for (let i = 0; i < length; i++) out[i] = this.trail[e * TRAIL + (head - length + 1 + i + TRAIL) % TRAIL];
      return out;
    }

    _advance(to) {
      const { set } = this;
      for (let e = 0; e < this.n; e++) {
        if (this.t >= set.length[e]) continue;
        const stop = decisionAt(set, e, to);
        for (let t = this.decision[e]; t < stop; t++) this._play(e, t, true);
        this.decision[e] = stop;
      }
      this._forwardMap(to);
      this.t = to;
    }

    // Play one decision for the heat, trail and creature-sample floor.
    _play(e, t, heat) {
      const cur = this.cur, floor = cur.floor[e], flags = step(this.set, cur, e, t);
      if (flags & INTERACTED && heat && onMap(cur.targetRow, cur.targetCol)) this.heat[TILES + floor * CELLS + cur.targetRow * MAP + cur.targetCol]++;
      if (flags & ENTERED) {
        const tile = tileOf(cur, e);
        if (heat) this.heat[tile]++;
        this._push(e, tile % CELLS, flags & NEW_FLOOR);
      }
      if ((t + 1) % this.set.stride === 0) this.sampleFloor[e] = cur.floor[e];
    }

    _push(e, cell, restart) {
      if (restart) this.trailLength[e] = 0;
      const head = (this.trailHead[e] + 1) % TRAIL;
      this.trailHead[e] = head;
      this.trail[e * TRAIL + head] = cell;
      if (this.trailLength[e] < TRAIL) this.trailLength[e]++;
    }

    // Positions and heat at display step t from the keyframe at or before
    // it. Replay starts one keyframe earlier, without heat, to give trails a
    // history.
    _positions(t) {
      const { set, pre, cur } = this, k = t >> KEY_SHIFT, from = Math.max(0, k - 1);
      this.heat.set(pre.heat.subarray(k * 2 * TILES, (k + 1) * 2 * TILES));
      for (let e = 0; e < this.n; e++) {
        if (set.length[e] <= from * KEYFRAME) {
          const tile = set.endTile[e];
          cur.floor[e] = tile / CELLS | 0; cur.row[e] = (tile % CELLS) / MAP | 0; cur.col[e] = tile % MAP; cur.facing[e] = set.endFacing[e];
          this.trailLength[e] = 0;
          this.decision[e] = set.decisions[e];
          continue;
        }
        const at = (e * pre.keys + from) * 4, start = decisionAt(set, e, from * KEYFRAME), counted = decisionAt(set, e, k * KEYFRAME);
        cur.floor[e] = pre.key[at]; cur.row[e] = pre.key[at + 1]; cur.col[e] = pre.key[at + 2]; cur.facing[e] = pre.key[at + 3];
        cur.escape[e] = lowerBound(set.escape[e].decision, start);
        this.sampleFloor[e] = cur.floor[e];
        this.trailLength[e] = 0;
        this._push(e, tileOf(cur, e) % CELLS, false);
        const stop = decisionAt(set, e, t);
        for (let d = start; d < stop; d++) this._play(e, d, d >= counted);
        this.decision[e] = stop;
      }
    }

    _forwardMap(to) {
      const { set } = this, ended = [];
      for (let e = 0; e < this.n; e++) {
        if (this.t >= set.length[e]) continue;
        const before = decisionAt(set, e, to);
        this._applyMap(e, before);
        if (to >= set.length[e]) {
          for (const tile of set.map[e].tile) this._mark(tile);
          ended.push(e);
        } else this._applyLight(e, before);
      }
      // An episode that ends takes its torches' light with it.
      const light = this.pre.light;
      for (const e of ended) {
        const run = light.runs[e];
        for (let k = 0; k < this.lightCursor[e]; k++) if (this.lightOwner[run.tile[k]] === e) this._recountLight(run.tile[k], to);
        this.lightCursor[e] = 0;
      }
    }

    _rebuildMap(t) {
      for (let tile = 0; tile < TILES; tile++) if (this.tileCount[tile]) this._mark(tile);
      this.changes.fill(0);
      this.mapCursor.fill(0);
      this.light.set(this.set.world.light);
      this.lightOwner.fill(WORLD);
      this.lightCursor.fill(0);
      this.lightVersion++;
      for (let e = 0; e < this.n; e++) {
        if (t >= this.set.length[e]) continue;
        const before = decisionAt(this.set, e, t);
        this._applyMap(e, before);
        this._applyLight(e, before);
      }
    }

    // Apply episode e's light changes from decisions before `before`.
    _applyLight(e, before) {
      const run = this.pre.light.runs[e], light = this.light;
      let k = this.lightCursor[e];
      for (; k < run.tile.length && run.decision[k] < before; k++) {
        const tile = run.tile[k];
        if (run.value[k] > light[tile]) { light[tile] = run.value[k]; this.lightOwner[tile] = e; this.lightVersion++; }
      }
      this.lightCursor[e] = k;
    }

    // A tile's light again from the world's and the episodes living at t.
    _recountLight(tile, t) {
      const { tileStart, tileRun, tileChange, runs } = this.pre.light, set = this.set;
      let value = set.world.light[tile], owner = WORLD;
      for (let i = tileStart[tile]; i < tileStart[tile + 1]; i++) {
        const e = tileRun[i], k = tileChange[i];
        if (t < set.length[e] && k < this.lightCursor[e] && runs[e].value[k] > value) { value = runs[e].value[k]; owner = e; }
      }
      this.lightOwner[tile] = owner;
      if (value !== this.light[tile]) { this.light[tile] = value; this.lightVersion++; }
    }

    // Apply episode e's map events from decisions before `before`.
    _applyMap(e, before) {
      const events = this.set.map[e];
      let k = this.mapCursor[e];
      for (; k < events.decision.length && events.decision[k] < before; k++) {
        const tile = events.tile[k];
        this.changes[tile * this.n + e] = events.code[k];
        this._mark(tile);
      }
      this.mapCursor[e] = k;
    }

    _mark(tile) {
      if (this.dirty[tile]) return;
      this.dirty[tile] = 1;
      this.dirtyList.push(tile);
    }
  }

  // The decisions playback shows, as sorted, disjoint [start, stop) spans in
  // decisions; index u counts shown decisions, so 0..length covers them all.
  // A decision inside a skipped stretch maps to the next shown one.
  class Timeline {
    constructor(spans) {
      const n = spans.length;
      if (!n || spans.some(([a, b], i) => b <= a || (i && a < spans[i - 1][1]))) throw Error('A timeline needs sorted, disjoint, non-empty spans.');
      this.starts = Int32Array.from(spans, s => s[0]);
      this.stops = Int32Array.from(spans, s => s[1]);
      this.offsets = new Int32Array(n + 1);
      for (let i = 0; i < n; i++) this.offsets[i + 1] = this.offsets[i] + this.stops[i] - this.starts[i];
      this.length = this.offsets[n] - 1;
    }

    toDecision(u) {
      u = Math.max(0, Math.min(this.length, Math.round(u)));
      const k = upperBound(this.offsets, u) - 1;
      return this.starts[k] + u - this.offsets[k];
    }

    toIndex(t) {
      const k = upperBound(this.stops, t);
      if (k === this.stops.length) return this.length;
      return this.offsets[k] + Math.max(0, t - this.starts[k]);
    }

    // The skipped stretches, [start, stop) in decisions.
    skipped() {
      const out = this.starts[0] > 0 ? [[0, this.starts[0]]] : [];
      for (let i = 1; i < this.starts.length; i++) if (this.stops[i - 1] < this.starts[i]) out.push([this.stops[i - 1], this.starts[i]]);
      return out;
    }
  }

  // Every decision 0..last, shown in real time.
  const realTime = last => new Timeline([[0, last + 1]]);

  function upperBound(sorted, value) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= value) lo = mid + 1; else hi = mid; }
    return lo;
  }

  function lowerBound(sorted, value) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < value) lo = mid + 1; else hi = mid; }
    return lo;
  }

  // The tiles (row * 48 + col) of an episode's last `moves` moves on the floor
  // it stands on at state s, oldest first, from decodeEpisode's states.
  function pathTail(states, s, moves) {
    const floor = states[4 * s], out = [];
    for (let j = s, last = -1; j >= 0 && out.length <= moves && states[4 * j] === floor; j--) {
      const cell = states[4 * j + 1] * MAP + states[4 * j + 2];
      if (cell !== last) out.push(last = cell);
    }
    return Int32Array.from(out.reverse());
  }

  // The shortest winning run: of the episodes that won, the one with the
  // fewest decisions, the lowest index among equals; -1 when none won.
  function shortestWin(set) {
    const win = OUTCOMES.indexOf('win');
    let best = -1;
    for (let e = 0; e < set.n; e++) if (set.outcome[e] === win && (best < 0 || set.decisions[e] < set.decisions[best])) best = e;
    return best;
  }

  // An embedded viewer's options from its element's data attributes
  // (dataset): which runs it shows, how it reaches its data, and how it plays.
  // Throws on a value the viewer cannot use, so a typo fails where it is made.
  function embedOptions(data) {
    const pick = (name, allowed, fallback) => {
      const value = data[name] ?? fallback;
      if (!allowed.includes(value)) throw Error(`data-${name.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}="${value}" is not one of ${allowed.join(', ')}.`);
      return value;
    };
    const number = (name, fallback) => {
      const value = Number(data[name] ?? fallback);
      if (!Number.isInteger(value) || value < 1) throw Error(`data-${name}="${data[name]}" is not a positive whole number.`);
      return value;
    };
    const base = data.dataBase ?? '';
    // data-speeds: the speed menu's choices, a comma list of positive whole
    // numbers; data-speed must be one of them (the first if not given).
    let speeds = null;
    if (data.speeds !== undefined) {
      speeds = String(data.speeds).split(',').map(s => Number(s.trim()));
      if (!speeds.length || speeds.some(v => !Number.isInteger(v) || v < 1)) throw Error(`data-speeds="${data.speeds}" is not a comma list of positive whole numbers.`);
      if (data.speed !== undefined && !speeds.includes(Number(data.speed))) throw Error(`data-speed="${data.speed}" is not one of data-speeds="${data.speeds}".`);
    }
    return {
      tier: data.tier ?? 'boss', set: data.set ?? 'short', count: number('count', 1000),
      follow: pick('follow', FOLLOWS, 'shortest-win'), transport: pick('transport', ['raw', 'b64'], 'raw'),
      layout: pick('layout', LAYOUTS, 'row'), skipQuiet: pick('skipQuiet', ['true', 'false'], 'true') === 'true',
      speed: number('speed', speeds ? speeds[0] : 300), speeds, dataBase: base && !base.endsWith('/') ? `${base}/` : base, followSeed: data.followSeed ?? null,
      loop: pick('loop', ['true', 'false'], 'true') === 'true', cap: data.cap === undefined ? null : number('cap', 1),
      controls: pick('controls', ['bottom', 'top'], 'bottom'), fit: pick('fit', ['none', 'column'], 'none'), scrubber: pick('scrubber', ['slider', 'plot'], 'slider'),
      followBoards: Number(pick('followBoards', ['2', '3'], '3')),
    };
  }

  // The runs a viewer can follow, as its follow option names them.
  // 'pinned' is a time-mapped set's unbroken run (time_map.unbroken).
  const FOLLOWS = ['shortest-win', 'median', 'best', 'worst', 'longest', 'random', 'pinned'];
  // Grid, one row of nine, two rows (floors 0-4, then 5-8 and the win tally),
  // or Follow: three boards (an embed's data-follow-boards="2": two) on the
  // followed run's floor.
  const LAYOUTS = ['grid', 'row', 'two-rows', 'follow'];

  // Episode e's achievement return at state t.
  function returnAt(set, e, t) {
    const score = set.score[e], k = lowerBound(score.decision, t);
    return k ? score.value[k - 1] : 0;
  }

  // Each run's achievement return as a step function of the display step:
  // run e's return is runs[e].values[k] from display step runs[e].steps[k]
  // until the next, 0 before the first; an ended run keeps its final return.
  // top is the largest return.
  function rewardCurves(set) {
    const runs = [];
    let top = 0;
    for (let e = 0; e < set.n; e++) {
      const score = set.score[e], steps = [], values = [];
      for (let k = 0; k < score.decision.length; k++) {
        const u = stepOf(set, e, score.decision[k] + 1), value = score.value[k];
        if (steps.at(-1) === u) values[values.length - 1] = value;
        else { steps.push(u); values.push(value); }
      }
      top = Math.max(top, values.at(-1) ?? 0);
      runs.push({ steps: Int32Array.from(steps), values: Int32Array.from(values) });
    }
    return { runs, top };
  }

  // A step function's value at display step u, as rewardCurves gives them.
  function curveAt(curve, u) {
    const k = upperBound(curve.steps, u);
    return k ? curve.values[k - 1] : 0;
  }

  // The display steps at which the winning runs end, in order: by step u,
  // upperBound(winSteps, u) runs have won.
  function winSteps(set) {
    const win = OUTCOMES.indexOf('win'), steps = [];
    for (let e = 0; e < set.n; e++) if (set.outcome[e] === win) steps.push(set.length[e]);
    return Int32Array.from(steps).sort();
  }

  // The runs that have won by display step u.
  const wonBy = (steps, u) => upperBound(steps, u);

  // Creature window j: per episode and sample, the byte offset of the sample
  // (a count, then class << 4 | species, row, col and, from format 4, facing
  // per creature: set.creatureBytes) in its group's file, files[g], or -1. A group whose files stop before window j
  // passes null; its episodes all ended before it.
  function indexWindow(set, j, files) {
    const per = set.window / set.stride, index = new Int32Array(set.n * per).fill(-1);
    const heads = files.map((bytes, g) => {
      if (!bytes) return null;
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length), count = view.getUint32(0, true);
      if (count !== set.groupSizes[g]) throw Error(`Creature window ${j} of group ${g} lists ${count} episodes, not ${set.groupSizes[g]}.`);
      const body = 4 * (count + 2), ends = Array.from({ length: count + 1 }, (_, i) => view.getUint32(4 + 4 * i, true));
      if (body + ends[count] !== bytes.length) throw Error(`Creature window ${j} of group ${g} does not end where its offsets do.`);
      return { body, ends };
    });
    for (let e = 0; e < set.n; e++) {
      const g = set.group[e], head = heads[g], stop = Math.min(set.decisions[e], (j + 1) * set.window);
      if (!head) {
        if (stop > j * set.window) throw Error(`Episode ${e} has no creature window ${j}.`);
        continue;
      }
      const bytes = files[g], i = set.local[e];
      let at = head.body + head.ends[i];
      for (let s = 0, d = j * set.window; d < stop; s++, d += set.stride) {
        index[e * per + s] = at;
        const count = bytes[at];
        if (at >= bytes.length || count > MAX_CREATURES) throw Error(`Episode ${e}'s creature sample at decision ${d} is malformed.`);
        for (let c = 0, k = at + 1; c < count; c++, k += set.creatureBytes) {
          const kind = bytes[k], facing = set.creatureBytes > 3 ? bytes[k + 3] : 0;
          if (kind >> 4 > 4 || (kind & 15) > 7 || bytes[k + 1] >= MAP || bytes[k + 2] >= MAP || facing > (kind >> 4 >= 3 ? 4 : 0)) throw Error(`Episode ${e}'s creature sample at decision ${d} is malformed.`);
        }
        at += 1 + set.creatureBytes * count;
      }
      if (at !== head.body + head.ends[i + 1]) throw Error(`Episode ${e}'s creature run in window ${j} does not hold its samples.`);
    }
    return { j, per, index, files };
  }

  // Statistics over the displayed episodes, named as the manifest's stats.
  function summary(set) {
    const reached = new Array(FLOORS).fill(0), deaths = new Array(FLOORS).fill(0), outcomes = [0, 0, 0, 0];
    let total = 0, decisions = 0, escapes = 0;
    for (let e = 0; e < set.n; e++) {
      escapes += set.escape[e].decision.length;
      for (let f = 0; f < FLOORS; f++) if (set.floorFirst[e * FLOORS + f] >= 0) reached[f]++;
      outcomes[set.outcome[e]]++;
      if (set.outcome[e] === 0) deaths[set.endTile[e] / CELLS | 0]++;
      total += set.ret[e];
      decisions += set.decisions[e];
    }
    return { count: set.n, mean_return: total / set.n, mean_decisions: decisions / set.n, reached, deaths, timeouts: outcomes[1], wins: outcomes[2], escapes, decisions };
  }

  return {
    MAP, FLOORS, CELLS, TILES, KEYFRAME, TRAIL, ITEMS, MAX_CREATURES, OCCUPANCY_SHIFT, OUTCOMES,
    parseWorld, parseEvents, parseKeeps, episodeSet, decodeEpisode, precompute, Playback, pathTail, returnAt, indexWindow, summary,
    Timeline, realTime, parseTimeline, shortestWin, embedOptions, FOLLOWS, LAYOUTS,
    decisionAt, stepOf, rewardCurves, curveAt, winSteps, wonBy, itemSprite, parseSleep, sampleAt, lightAt, torchLit, NEVER,
  };
})();
