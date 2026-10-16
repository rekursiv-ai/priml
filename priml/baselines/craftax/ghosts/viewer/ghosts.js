'use strict';
// The Craftax ghost-run viewer: loads a tier's episodes, asks decode.js for
// the world at the playhead and draws the nine floors. Each floor stacks four
// canvases: the base world (drawn once), map changes (the tiles whose summary
// changed, each frame), heat and ghosts (redrawn whenever the decision does),
// so every layer shows the decision on screen. Under the floors, a plot of
// every run's achievement return; in the Two rows layout a tenth square
// tallies the wins. The playhead is a display step (decode.js): a decision,
// except in a time-mapped set, where each run shows its own decision.
//
// One renderer, two hosts, both mounted from markup:
//   [data-ghost-page]  the full page (index.html): every picker, the stats and
//                      the legend, data packed as base64 text beside it.
//   [data-ghost-embed] a figure in another page (the blog post): fixed runs,
//                      its boards, the plots and play/pause only, loaded when
//                      near the viewport and played while any of its stage
//                      shows. decode.js's embedOptions reads its data
//                      attributes. GhostEmbed.mount(container, options) mounts
//                      one by hand.
// A page hosts one viewer; the script is safe to load on pages with neither,
// and re-mounts on the site router's rekursiv:route event.
(() => {
  const D = GhostDecode;
  const { MAP, FLOORS, CELLS, TILES } = D;
  const TILE = 11, SIDE = MAP * TILE, RECENT_ENDS = 500;
  const FLOOR_NAMES = ['Overworld', 'Dungeon', 'Gnomish Mines', 'Sewers', 'Vault', 'Troll Mines', 'Fire Realm', 'Ice Realm', 'Graveyard'];
  // Each creature class's disk under its sprite: melee and ranged enemies,
  // passive creatures, enemy and player projectiles. Enemies are told apart
  // by their red outline (RINGS) on a dark disk; a passive creature keeps its
  // light green disk and no outline, as the bat's black sprite needs a light
  // disk to show (on a dark one a bat reads as an empty disk).
  const CLASS_COLORS = ['#05090c', '#8bc34a', '#05090c', '#ce93d8', '#4dd0e1'];
  // The classes that are enemies: melee and ranged.
  const ENEMY = [true, false, true, false, false];
  // The player facing left, right, up and down, then asleep (atlas cell 4).
  const PLAYER_SPRITES = [39, 40, 38, 37], ASLEEP = 41;
  const SPEEDS = [10, 30, 100, 300, 1000, 3000];
  // The opacity of a marker that one run in N shows; k runs on a tile draw at
  // floor + (top - floor) sqrt(k / N). Ghost players stand out (PLAYER_OPACITY),
  // enemies recede (ENEMY_OPACITY) under the players and trails; passive
  // creatures and projectiles take GHOST_FLOOR to 1. Changed map tiles start
  // a little fainter, so the world under them reads.
  const GHOST_FLOOR = 0.6, CHANGE_FLOOR = GHOST_FLOOR - 0.25, PLAYER_OPACITY = [0.85, 1], ENEMY_OPACITY = [0.5, 0.6];
  // The followed run's colour: bright green, cased in black so it reads over
  // grass, heat and the green win checks (which are check-shaped and smaller).
  // Its trail, ring and win mark keep at least about their size on a 528 px
  // board when boards shrink (the embed's one row is about 150 px a board),
  // and the board it is on is outlined in it.
  const FOLLOWED_MOVES = 800, FOLLOW = '61,255,127';
  // Wins on the timeline: a band of the runs that have won, stacked on the
  // living runs' floors, deeper than the followed run's green and edged in a
  // brighter one, and ticks along the top in the win checks' green.
  const WON_FILL = 'rgba(28,122,58,.9)', WON_EDGE = '#6fe08a', WIN_TICK = '#2fbf4a';
  const FORMATS = ['craftax-ghosts/2', 'craftax-ghosts/3', 'craftax-ghosts/4'];
  // CSS pixels on the reward plot's left for its axis title.
  const REWARD_GUTTER = 18;
  const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)');
  // How much the followed run's strokes widen on a board shown smaller than
  // its canvas, so a 1-pixel stroke stays 0.7 CSS pixels wide or more; a
  // hidden board (no width) draws them at 1, to be redrawn at its size once
  // shown.
  const followScale = ctx => (ctx.canvas.clientWidth ? Math.max(1, 0.7 * SIDE / ctx.canvas.clientWidth) : 1);
  // Heat colormaps, cool to hot, as [position, hex] stops: movement runs blue,
  // purple, red, orange, yellow to near-white; interactions teal to pale lime,
  // so the two read apart when both are shown. Both start bright enough that a
  // single entry shows on a dark floor; heat stays at most HEAT_ALPHA opaque so
  // the ghosts above it read.
  const MOVE_STOPS = [[0, '#3d6df2'], [0.2, '#7b3fd1'], [0.42, '#c22f86'], [0.6, '#e8473a'], [0.76, '#f5862a'], [0.9, '#fbd148'], [1, '#fff8e0']];
  const INTERACT_STOPS = [[0, '#17a2b0'], [0.3, '#16a37f'], [0.55, '#3cc06a'], [0.78, '#a6dc4a'], [1, '#f4ffd0']];
  const HEAT_ALPHA = [0.3, 0.5];
  const MOVE_RAMP = colormap(MOVE_STOPS), INTERACT_RAMP = colormap(INTERACT_STOPS);
  const fmt = value => Math.round(value).toLocaleString('en-US');

  // Data files by URL. Kept files stay cached until a load of another set
  // drops them.
  const files = new Map();
  // The mounted viewer, exposed as window.ghosts for render_check.mjs.
  let app = null;
  // Every load and mount takes the next generation; a load that finds a newer
  // one has been superseded and drops its result.
  let generation = 0;

  // An element of the mounted viewer, by its data-g name; null if this host
  // has none (an embed has no pickers).
  const $ = name => app.root.querySelector(`[data-g="${name}"]`);
  const text = (name, value) => { const node = $(name); if (node) node.textContent = value; };

  // A data file's bytes, gunzipped. transport 'b64' fetches each .bin.gz file
  // as NAME.b64.txt, its bytes in base64 (pack.mjs), for hosts that refuse
  // binary types; 'raw' fetches it as it is.
  async function loadFile(name, keep = true) {
    const url = app.o.dataBase + name;
    if (files.has(url)) return files.get(url);
    const packed = app.o.transport === 'b64' && name.endsWith('.bin.gz');
    const loading = (async () => {
      const source = packed ? `${url}.b64.txt` : url, response = await fetch(source);
      if (!response.ok) throw Error(`${source} did not load (HTTP ${response.status}).`);
      let body = packed ? new Blob([fromBase64(await response.text())]).stream() : response.body;
      if (name.endsWith('.gz')) body = body.pipeThrough(new DecompressionStream('gzip'));
      return new Uint8Array(await new Response(body).arrayBuffer());
    })();
    if (keep) files.set(url, loading);
    loading.catch(() => files.delete(url));
    return loading;
  }

  function fromBase64(encoded) {
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(encoded);
    const binary = atob(encoded), bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  const loadJson = async name => JSON.parse(new TextDecoder().decode(await loadFile(name)));

  // 256 RGB levels interpolated between the stops, as [r, g, b] * 256.
  function colormap(stops) {
    const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)), out = new Uint8Array(256 * 3);
    for (let level = 0, k = 0; level < 256; level++) {
      const x = level / 255;
      while (stops[k + 1][0] < x) k++;
      const [[x0, a], [x1, b]] = [stops[k], stops[k + 1]], f = (x - x0) / (x1 - x0), [ra, rb] = [rgb(a), rgb(b)];
      for (let c = 0; c < 3; c++) out[3 * level + c] = Math.round(ra[c] + f * (rb[c] - ra[c]));
    }
    return out;
  }

  const gradient = stops => `linear-gradient(90deg, ${stops.map(([x, hex]) => `${hex} ${100 * x}%`).join(', ')})`;

  // A projectile's sprite with its facing known (format 4) is original
  // Craftax's texture, pointing up (51 the arrow pointing up), which
  // drawTurned turns; unknown (facing 0), it is the one this viewer always
  // drew, an arrow pointing down (50).
  function mapSprite(klass, type, facing = 0) {
    if (klass === 0) return 80 + type;
    if (klass === 1) return 88 + type;
    if (klass === 2) return 91 + type;
    return (facing ? [51, 99, 100, 101, 51, 102, 100, 101] : [50, 99, 100, 101, 50, 102, 100, 101])[type] ?? 50;
  }

  // Draw sheet sprite `id` in the size x size square at (x, y), turned to
  // `facing` (a move action; 0 or UP as it is) by original Craftax's rule for
  // projectiles (craftax/renderer.py): flipped top to bottom when it flies
  // down or right, then transposed, rows for columns, when it flies left or
  // right. The transforms compose right to left, so the transpose is set
  // first.
  const [LEFT, RIGHT, DOWN] = [1, 2, 4];
  function drawTurned(ctx, sprites, id, x, y, size, facing) {
    ctx.save();
    ctx.translate(x, y);
    if (facing === LEFT || facing === RIGHT) ctx.transform(0, 1, 1, 0, 0, 0);
    if (facing === DOWN || facing === RIGHT) ctx.transform(1, 0, 0, -1, 0, size);
    ctx.drawImage(sprites, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, 0, 0, size, size);
    ctx.restore();
  }

  // The atlas row of a creature of `klass` (rows 1-3, then for each
  // projectile class 3 and 4 a row per facing 0-4) and of the players (row 0).
  const markRow = (klass, facing) => (klass < 3 ? 1 + klass : 4 + 5 * (klass - 3) + facing);

  function drawMapTile(ctx, id, row, col) {
    ctx.drawImage(app.sprites, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, col * TILE, row * TILE, TILE, TILE);
  }

  // Outlines by kind, as [colour, width, dark casing on each side], widths in
  // CSS pixels at the board's size, ranked by prominence: the followed
  // player's (drawFollowed) thickest and brightest green, a ghost player's
  // thinner green, an enemy's (melee or ranged) thin red over a thin casing,
  // drawn at the enemy's lower opacity. Each is a ring just outside its tile;
  // the casing lets it read on every floor, the Fire Realm's reds and the
  // Graveyard's darks alike.
  const RINGS = { enemy: ['#ff4b3e', 0.75, 0.3], ghost: ['#4cd86a', 1.5, 0.6], followed: [`rgb(${FOLLOW})`, 2.5, 0.6] }, RING_DARK = 'rgba(5,9,12,.92)';
  // Stroke a ring of the given kind about (x, y) on ctx, at `scale` backing
  // pixels per CSS pixel.
  function ring(ctx, x, y, kind, scale) {
    const [color, width, casing] = RINGS[kind], w = width * scale, radius = TILE / 2 + w / 2;
    for (const [style, line] of [[RING_DARK, w + 2 * casing * scale], [color, w]]) {
      ctx.strokeStyle = style;
      ctx.lineWidth = line;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  // The margin a cell of the atlas leaves about its tile for a ring.
  const ringPad = scale => Math.ceil((RINGS.followed[1] + RINGS.followed[2]) * scale);

  // One cell per player facing (row 0), per creature class and type (rows
  // 1-3) and per projectile class, facing and type (markRow): a dark disk and
  // a coloured one under a 9 px sprite, as the single-game viewer, in an 11 px tile with ringPad(scale) around it for
  // the outline of its kind, drawn for boards shown at `scale` backing
  // pixels per CSS pixel (markScale).
  function buildAtlas(scale = 1) {
    const pad = ringPad(scale), cell = TILE + 2 * pad, atlas = document.createElement('canvas');
    atlas.width = 8 * cell;
    atlas.height = 14 * cell;
    const ctx = atlas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    // A dark rim around the disk keeps a ghost legible over any heat colour.
    const mark = (row, col, color, id, kind, facing = 0) => {
      const x = col * cell + pad, y = row * cell + pad;
      for (const [fill, radius] of [['#05090c', 5.5], [color, 4.2]]) {
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(x + 5.5, y + 5.5, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
      drawTurned(ctx, app.sprites, id, x + 1, y + 1, 9, facing);
      if (kind) ring(ctx, x + 5.5, y + 5.5, kind, scale);
    };
    PLAYER_SPRITES.forEach((id, facing) => mark(0, facing, '#fff', id, 'ghost'));
    mark(0, 4, '#b8c4ff', ASLEEP, 'ghost');
    for (let klass = 0; klass < 5; klass++) {
      for (let facing = 0; facing < (klass < 3 ? 1 : 5); facing++) {
        for (let type = 0; type < 8; type++) mark(markRow(klass, facing), type, CLASS_COLORS[klass], mapSprite(klass, type, facing), ENEMY[klass] ? 'enemy' : null, facing);
      }
    }
    return Object.assign(atlas, { scale, pad, cell });
  }

  // Draw the atlas cell at (row, col) over the tile at board (tileRow,
  // tileCol). With app.marks set (render_check sets it), each drawn marker is
  // listed there as [kind, floor, tile row, tile col, atlas row, atlas col]
  // for the frame on screen. A shade (0-255) darkens the marker's own pixels
  // as the shade layer darkens a tile (drawShade).
  function drawMark(ctx, row, col, tileRow, tileCol, shade = 0) {
    const { atlas } = app, { pad, cell } = atlas;
    let [image, x, y] = [atlas, col * cell, row * cell];
    if (shade > 0) {
      const scratch = app.shadeScratch ??= document.createElement('canvas').getContext('2d');
      scratch.canvas.width = scratch.canvas.height = cell;
      scratch.drawImage(atlas, x, y, cell, cell, 0, 0, cell, cell);
      scratch.globalCompositeOperation = 'source-atop';
      scratch.fillStyle = `rgba(${SHADE_RGB},${shade / 255})`;
      scratch.fillRect(0, 0, cell, cell);
      [image, x, y] = [scratch.canvas, 0, 0];
    }
    ctx.drawImage(image, x, y, cell, cell, tileCol * TILE - pad, tileRow * TILE - pad, cell, cell);
    app.marks?.push([row === 0 ? 'ghost' : row === 1 || row === 3 ? 'enemy' : 'creature', app.panels.findIndex(p => p.ghosts === ctx), tileRow, tileCol, row, col]);
  }

  // A creature (melee, passive or ranged) or projectile shows only where its
  // tile is lit, as the game shows it to its agent (observation._visible_tile_numba:
  // light above VISIBLE_LIGHT, 12 of 255, original Craftax's 0.05), and dims
  // with its tile, disk and ring too: it takes the tile's shade (drawShade),
  // as original Craftax's renderer multiplies what it draws by its tile's
  // light. Returns that shade, or -1 to hide it. With Lighting off it always
  // shows, darkened only by the fog. Players are not shaded: they always show.
  const VISIBLE_LIGHT = 12;
  function creatureShade(tile) {
    if (app.show.light && app.pb.light[tile] <= VISIBLE_LIGHT) return -1;
    return Math.max(0, app.shade?.drawn[tile] ?? 0);
  }

  // Backing pixels per CSS pixel of the boards on screen (1 if none shows);
  // the atlas is drawn again when it changes by more than 5%.
  function markScale() {
    const width = app.panels.map(p => p.ghosts.canvas.clientWidth).find(w => w > 0);
    return width ? SIDE / width : 1;
  }

  function buildPanels() {
    const floors = $('floors');
    for (let f = 0; f < FLOORS; f++) {
      const article = document.createElement('article');
      article.className = 'floor';
      article.style.setProperty('--floor-color', `var(--floor-${f})`);
      article.innerHTML = `<header><h2><span>${f}</span>${FLOOR_NAMES[f]}</h2><p class="floor-stats"></p><button type="button" class="enlarge" aria-pressed="false">Enlarge</button></header><div class="board"></div>`;
      const board = article.querySelector('.board');
      const [base, layer, shade, heat, ghosts] = ['base', 'layer', 'shade', 'heat', 'ghosts'].map(name => {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = SIDE;
        canvas.className = name;
        board.append(canvas);
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        return ctx;
      });
      ghosts.canvas.setAttribute('aria-label', app.page ? `Floor ${f}, ${FLOOR_NAMES[f]}; click a ghost to follow its run` : `Floor ${f}, ${FLOOR_NAMES[f]}`);
      if (app.page) ghosts.canvas.addEventListener('click', event => followAt(f, event));
      const enlarge = article.querySelector('.enlarge');
      enlarge.onclick = () => {
        const wide = article.classList.toggle('wide');
        enlarge.setAttribute('aria-pressed', String(wide));
        enlarge.textContent = wide ? 'Shrink' : 'Enlarge';
      };
      floors.append(article);
      app.panels.push({ article, base, layer, shade, heat, ghosts, stats: article.querySelector('.floor-stats'), text: '' });
    }
    floorInks();
    // The win tally, the tenth square of the Two rows layout.
    const tally = document.createElement('article');
    tally.className = 'floor tally';
    tally.innerHTML = '<header><h2><span>✓</span>Won</h2></header><div class="board"></div>';
    const canvas = Object.assign(document.createElement('canvas'), { width: SIDE, height: SIDE, className: 'tally-marks' });
    canvas.setAttribute('aria-label', 'Runs that have won by the decision on screen');
    tally.querySelector('.board').append(canvas);
    floors.append(tally);
    app.tally = { ctx: canvas.getContext('2d'), drawn: '' };
    // The Follow layout's win count, over its right board.
    const badge = Object.assign(document.createElement('span'), { className: 'won-badge', role: 'status' });
    badge.dataset.g = 'won-badge';
    badge.innerHTML = '<b data-g="won-count"></b><span class="won-word"> won</span>';
    floors.append(badge);
  }

  // Each floor's band colour is the host's --floor-f, the timeline's own, at
  // 60% over the board; its ink is near-black or white, whichever contrasts
  // with the colour more (WCAG), with a halo of the other, so it reads over
  // whatever terrain shows through.
  const DARK_INK = '#0b0d10', LIGHT_INK = '#ffffff';
  function floorInks() {
    const probe = document.createElement('canvas').getContext('2d');
    for (const { article } of app.panels) {
      probe.fillStyle = '#000';
      probe.fillStyle = getComputedStyle(article).getPropertyValue('--floor-color').trim() || '#05090c';
      const dark = contrast(probe.fillStyle, DARK_INK) >= contrast(probe.fillStyle, LIGHT_INK);
      article.style.setProperty('--floor-ink', dark ? DARK_INK : LIGHT_INK);
      article.style.setProperty('--floor-halo', dark ? 'rgba(255,255,255,.85)' : 'rgba(0,0,0,.85)');
    }
  }

  // A CSS colour scaled toward black by `factor`, as #rrggbb.
  function darker(color, factor) {
    const probe = document.createElement('canvas').getContext('2d');
    probe.fillStyle = '#000';
    probe.fillStyle = color;
    const hex = probe.fillStyle;
    return `#${[1, 3, 5].map(i => Math.round(parseInt(hex.slice(i, i + 2), 16) * factor).toString(16).padStart(2, '0')).join('')}`;
  }

  // The WCAG contrast ratio of two #rrggbb colours.
  function contrast(a, b) {
    const luminance = hex => {
      const [r, g, bl] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
    };
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + 0.05) / (low + 0.05);
  }

  // The world's blocks and items as they stand at the reset, undimmed: the
  // shade layer (drawShade) darkens them by the light.
  function drawBase(f) {
    const { world } = app, ctx = app.panels[f].base;
    ctx.fillStyle = '#070d11';
    ctx.fillRect(0, 0, SIDE, SIDE);
    for (let cell = 0; cell < CELLS; cell++) {
      const tile = f * CELLS + cell, row = cell / MAP | 0, col = cell % MAP;
      if (world.block[tile]) drawMapTile(ctx, world.block[tile], row, col);
      if (world.item[tile]) drawMapTile(ctx, D.itemSprite(world.item[tile]), row, col);
    }
  }

  const title = name => name.charAt(0).toUpperCase() + name.slice(1);
  const SET_LABELS = { short: 'Short games', all: 'All', wins: 'Wins only', unbroken: 'Unbroken wins' };

  // A tier's episode sets, short games first: the page opens on them.
  function setsOf(tier) {
    return [...tier.sets].sort((a, b) => (b.name === 'short') - (a.name === 'short')).map(s => ({ ...s, id: s.name, label: SET_LABELS[s.name] ?? title(s.name) }));
  }

  function buildPickers() {
    const { manifest } = app, facts = $('facts');
    const fact = (term, value) => {
      const [dt, dd] = [document.createElement('dt'), document.createElement('dd')];
      dt.textContent = term;
      dd.textContent = value;
      facts.append(dt, dd);
    };
    fact('World', `pool world ${manifest.world_seed}`);
    fact('Tiers', manifest.tiers.map(t => title(t.name)).join(', '));
    fact('Built from', `${manifest.git_commit.slice(0, 10)} · ${manifest.platform}`);
    $('provenance').textContent = [
      `format ${manifest.format}`, `commit ${manifest.git_commit}`, `game ${manifest.game_package_digest}`, `platform ${manifest.platform}`,
      ...manifest.tiers.flatMap(t => t.sources.map(source => `${t.name}: ${Object.entries(source.provenance).map(([k, v]) => `${k} ${v}`).join(', ')}`)),
    ].join(' · ');
  }

  function picker(container, entries, pressed, onPick) {
    container.replaceChildren(...entries.map(({ key, label, note }) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.key = key;
      button.innerHTML = note ? '<strong></strong><span></span>' : '';
      if (note) {
        button.className = 'choice';
        button.querySelector('strong').textContent = label;
        button.querySelector('span').textContent = note;
      } else {
        button.className = 'count';
        button.textContent = label;
      }
      button.setAttribute('aria-pressed', String(key === pressed));
      button.onclick = () => onPick(key);
      return button;
    }));
  }

  function updatePickers() {
    const { tier, collection } = app, sets = setsOf(tier);
    picker($('tiers'), app.manifest.tiers.map(t => ({ key: t.name, label: title(t.name), note: t.sources[0].provenance.checkpoint?.split('/').pop() ?? `${fmt(t.sources[0].episodes)} runs` })), tier.name, name => show(name, collection.id, app.count));
    $('sets-field').hidden = sets.length < 2;
    picker($('sets'), sets.map(s => ({ key: s.id, label: s.label, note: `${fmt(s.composition.qualified)} ${s.time_map ? 'wins' : ''} of ${fmt(s.composition.run)} runs`.replace('  ', ' ') })), collection.id, id => show(tier.name, id, app.count));
    picker($('counts'), collection.counts.map(c => ({ key: String(c), label: fmt(c) })), String(app.count), c => show(tier.name, collection.id, Number(c)));
  }

  // Load a tier's set's first `count` episodes and decode them, then follow
  // the run the follow option names and stand at display step `at`. A
  // time-mapped set (wins) plays each run by its time map: it has no quiet
  // stretches to skip, and follows its unbroken run unless told otherwise.
  async function show(tierName, setId, count, at = null) {
    const version = ++generation;
    stop();
    const tier = app.manifest.tiers.find(t => t.name === tierName);
    if (!tier) throw Error(`manifest.json has no tier ${tierName}.`);
    const sets = setsOf(tier), collection = sets.find(s => s.id === setId) ?? sets[0], counts = collection.counts;
    count = counts.includes(count) ? count : counts.reduce((a, b) => (Math.abs(b - count) < Math.abs(a - count) ? b : a));
    const entering = app.collection?.id !== collection.id, mapped = !!collection.time_map;
    Object.assign(app, { tier, collection, count });
    if (app.page) updatePickers();
    useFollows(mapped, entering);
    const note = $('set-note');
    if (note) note.hidden = !mapped;
    if (note && mapped) {
      note.textContent = collection.id === 'unbroken'
        ? `Unbroken wins: ${fmt(collection.episodes)} runs that beat the game in at most ${fmt(collection.time_map.rule.steps)} decisions, the pinned run and others spread evenly over their lengths, each played whole: every decision shown, none skipped.`
        : `Wins only: every run that beat the game, each on its own clock, aligned by its progress. Idle stretches are skipped, so each win plays in at most ${fmt(collection.time_map.rule.steps)} steps; the pinned run plays unbroken, every decision shown.`;
    }
    setStatus(`Loading ${title(tier.name)}, ${collection.label.toLowerCase()}, ${fmt(count)} runs…`);
    try {
      const groups = collection.groups.slice(0, counts.indexOf(count) + 1), urls = groups.flatMap(g => ['episodes.json', 'players.bin.gz', 'events.bin.gz', 'keeps.bin.gz'].map(f => app.o.dataBase + `${g.path}/${f}`));
      for (const url of files.keys()) if (/\/g\d+\/(episodes\.json|players\.bin\.gz|events\.bin\.gz|keeps\.bin\.gz)$/.test(url) && !urls.includes(url)) files.delete(url);
      const sleep = mapped && app.showSleep && !!collection.sleep_map;
      const [timeline, ...loaded] = await Promise.all([mapped ? null : loadFile(collection.timelines.find(t => t.count === count).path, false), ...groups.map(g => loadGroup(g, mapped, sleep))]);
      if (version !== generation) return;
      const set = D.episodeSet(app.world, loaded, app.manifest), run = D.precompute(set);
      const quiet = mapped ? null : D.parseTimeline(timeline, set.maxDecisions);
      let step;
      while (!(step = run.next()).done) {
        setStatus(`Decoding ${title(tier.name)}: ${Math.round(100 * step.value)}%`);
        await new Promise(resolve => setTimeout(resolve));
        if (version !== generation) return;
      }
      const pre = step.value;
      if (app.o.cap && set.maxSteps > app.o.cap) throw Error(`the set plays ${fmt(set.maxSteps)} steps, more than the figure's cap of ${fmt(app.o.cap)}.`);
      Object.assign(app, {
        set, pre, groups, quiet, pb: new D.Playback(set, pre), stats: D.summary(set), windows: new Map(), stale: true, heatDrawnAt: -1, timeline: null,
        reward: D.rewardCurves(set), winSteps: D.winSteps(set),
      });
      allocate(set);
      follow($('follow')?.value ?? app.o.follow);
      useTimeline();
      if (app.page) renderStats();
      // The page opens where the runs have spread out: a tenth of the median
      // run, at most 1,000 steps.
      const median = [...set.length].sort((a, b) => a - b)[set.n >> 1];
      goTo(at ?? Math.min(1000, Math.round(median / 10)));
      const longest = mapped ? `${fmt(set.maxSteps)} steps, each run aligned by its progress` : `longest ${fmt(set.maxDecisions)}`;
      setStatus(`Loaded ${title(tier.name)}: ${fmt(set.n)} runs, ${fmt(app.stats.decisions)} decisions, ${longest}.`);
      app.o.onReady?.(app);
    } catch (error) {
      if (version === generation) fail(error);
    }
  }

  // A group's files; a time-mapped set's view that shows sleep (`sleep`)
  // reads keeps-sleep.bin and sleep.bin instead of keeps.bin.
  async function loadGroup(group, mapped, sleep = false) {
    const [listing, players, events, keeps, slept] = await Promise.all([
      loadJson(`${group.path}/episodes.json`), loadFile(`${group.path}/players.bin.gz`), loadFile(`${group.path}/events.bin.gz`),
      mapped ? loadFile(`${group.path}/${sleep ? 'keeps-sleep' : 'keeps'}.bin.gz`) : null, sleep ? loadFile(`${group.path}/sleep.bin.gz`) : null,
    ]);
    return { listing, players, events, keeps, sleep: slept };
  }

  // The follow choices and quiet-stretch toggle a set offers: a time-mapped
  // set adds its pinned unbroken run, the default on entering it, and has no
  // quiet stretches to skip.
  function useFollows(mapped, entering) {
    const select = $('follow'), pinned = select?.querySelector('option[value="pinned"]'), skip = $('skip-quiet')?.closest('label');
    const sleep = $('show-sleep')?.closest('label');
    if (skip) skip.hidden = mapped;
    if (sleep) sleep.hidden = !(mapped && app.collection.sleep_map);
    if (!pinned) return;
    pinned.hidden = pinned.disabled = !mapped;
    if (mapped && entering) select.value = 'pinned';
    else if (!mapped && select.value === 'pinned') select.value = app.o.follow;
  }

  // Per-frame scratch, sized once per loaded set.
  function allocate(set) {
    const n = set.n;
    const share = ([floor, top]) => Float32Array.from({ length: n + 1 }, (_, k) => floor + (top - floor) * Math.sqrt(k / n));
    app.alpha = share([GHOST_FLOOR, 1]);
    app.playerAlpha = share(PLAYER_OPACITY);
    app.enemyAlpha = share(ENEMY_OPACITY);
    app.changeAlpha = Float32Array.from({ length: n + 1 }, (_, k) => CHANGE_FLOOR + (1 - CHANGE_FLOOR) * Math.sqrt(k / n));
    app.playerCount = new Uint16Array(TILES);
    app.playerFacing = new Uint16Array(TILES * 5);
    app.playerTouched = new Int32Array(n);
    app.creatureCount = new Uint16Array(TILES * 5 * 40);
    app.creatureTouched = new Int32Array(n * D.MAX_CREATURES);
    app.endCount = new Uint16Array(4 * TILES);
    app.endRecent = new Uint8Array(4 * TILES);
    app.endTouched = new Int32Array(n);
    app.heatLevels = new Int16Array(CELLS);
    app.endOrder = Int32Array.from({ length: n }, (_, e) => e).sort((a, b) => set.length[a] - set.length[b]);
    // The winners in the order they win, which the tally fills its marks in.
    app.winOrder = app.endOrder.filter(e => set.outcome[e] === 2);
  }

  // Every display step, or the builder's timeline with quiet stretches
  // collapsed; a time-mapped set has every step.
  function useTimeline() {
    app.timeline = app.skipQuiet && app.quiet ? app.quiet.timeline : D.realTime(app.set.maxSteps);
    const seek = $('seek'), scrub = $('scrub');
    if (seek) seek.max = String(app.timeline.length);
    if (scrub) scrub.setAttribute('aria-valuemax', String(app.timeline.length));
    drawChart();
    drawReward();
  }

  // The decision the followed run shows, and its sleep sample (0 but mid-sleep):
  // what the policy view draws.
  const followedDecision = () => (app.selected >= 0 ? app.pb.decision[app.selected] : app.pb.t);
  const followedSample = () => (app.selected >= 0 ? D.sampleAt(app.set, app.selected, app.pb.t) : 0);

  function goTo(t, playing = false) {
    const start = performance.now();
    app.pb.seek(t);
    if (!playing) app.u = app.timeline.toIndex(app.pb.t);
    app.stripAnimate = playing;
    render();
    app.stripAnimate = false;
    ensureWindows();
    app.frameTimes.push(performance.now() - start);
    if (app.frameTimes.length > 4000) app.frameTimes.splice(0, 2000);
    if (app.policy?.view && !$('policy-panel').hidden) app.policy.view.show(app.policyShown = followedDecision(), followedSample());
    app.o.onDecision?.(followedDecision(), { playing, sample: followedSample() });
  }

  function render() {
    if (!app.pb || !app.timeline) return;
    // The strip first, so the followed run's board shows, at its size, before
    // its trail and the markers are drawn to that size (followScale,
    // markScale).
    updateStrip(app.stripAnimate);
    const scale = markScale();
    if (Math.abs(scale - app.atlas.scale) > 0.05 * scale) app.atlas = buildAtlas(scale);
    drawChanges();
    drawShade();
    if (app.heatDrawnAt !== app.pb.t) drawHeat();
    drawGhosts();
    drawPlayhead();
    drawRewardCursor();
    drawTally();
    updateReadouts();
  }

  // Map changes: redraw only the tiles whose summary changed since the last
  // frame, or every changed tile when the layer is stale (a new set, a toggle).
  function drawChanges() {
    const { pb } = app, updated = pb.summarize();
    let tiles = updated;
    if (app.stale) {
      for (const panel of app.panels) panel.layer.clearRect(0, 0, SIDE, SIDE);
      tiles = [];
      for (let tile = 0; tile < TILES; tile++) if (pb.tileCount[tile]) tiles.push(tile);
      app.stale = false;
    }
    for (const tile of tiles) {
      const ctx = app.panels[tile / CELLS | 0].layer, row = (tile % CELLS) / MAP | 0, col = tile % MAP, k = pb.tileCount[tile];
      ctx.clearRect(col * TILE, row * TILE, TILE, TILE);
      if (!k || !app.show.changes) continue;
      const code = pb.tileCode[tile] - 1, block = code / D.ITEMS | 0, item = code % D.ITEMS;
      ctx.globalAlpha = app.changeAlpha[k];
      if (block) drawMapTile(ctx, block, row, col);
      if (item) drawMapTile(ctx, D.itemSprite(item), row, col);
      ctx.globalAlpha = 1;
    }
  }

  // Fog of war and torchlight over each floor's map, under the heat and the
  // ghosts: a tile no run has had in view by now (pre.reveal) near-black, at
  // FOG; with the light on, every other tile darkened by DARK (1 - light /
  // 255), the light the living runs' torches have shed by now (pb.light), as
  // the strategy clips shade the game's light map. A floor redraws only when
  // a tile's shade changes: its shades go to a 48 x 48 image, drawn up to the
  // board a tile a cell.
  const FOG = 0.9, DARK = 0.7, SHADE_RGB = [7, 13, 17];
  function drawShade() {
    const { pb, pre } = app, fog = app.show.fog, lit = app.show.light;
    if (!app.shade) {
      const canvas = Object.assign(document.createElement('canvas'), { width: MAP, height: MAP });
      app.shade = { drawn: new Int16Array(TILES).fill(-1), image: canvas.getContext('2d').createImageData(MAP, MAP), canvas };
    }
    const { drawn, image, canvas } = app.shade, pixels = image.data;
    for (let f = 0; f < FLOORS; f++) {
      let changed = false;
      for (let tile = f * CELLS; tile < (f + 1) * CELLS; tile++) {
        const alpha = fog && pre.reveal[tile] > pb.t ? Math.round(255 * FOG) : lit ? Math.round(255 * DARK * (1 - pb.light[tile] / 255)) : 0;
        if (alpha !== drawn[tile]) { drawn[tile] = alpha; changed = true; }
      }
      if (!changed) continue;
      for (let cell = 0; cell < CELLS; cell++) pixels.set([...SHADE_RGB, drawn[f * CELLS + cell]], 4 * cell);
      canvas.getContext('2d').putImageData(image, 0, 0);
      const ctx = app.panels[f].shade;
      ctx.clearRect(0, 0, SIDE, SIDE);
      ctx.drawImage(canvas, 0, 0, SIDE, SIDE);
    }
  }

  // Heat through the decision on screen: player tile entries (movement) and
  // interactions on the faced tile, summed over the shown runs.
  function drawHeat() {
    const mode = app.heatMode;
    for (const panel of app.panels) panel.heat.clearRect(0, 0, SIDE, SIDE);
    if (mode === 'movement' || mode === 'both') fillHeat(0, MOVE_RAMP);
    if (mode === 'interactions') fillHeat(TILES, INTERACT_RAMP);
    if (mode === 'both') outlineHeat(TILES, INTERACT_RAMP);
    app.heatDrawnAt = app.pb.t;
  }

  // A count's level, 0-255, on a log scale whose top is its floor's largest
  // count: 1 run-visit sits at the cool end, the busiest tile at the hot end.
  function heatLevels(offset, f) {
    const { heat } = app.pb, levels = app.heatLevels;
    let max = 0;
    for (let cell = 0; cell < CELLS; cell++) max = Math.max(max, heat[offset + f * CELLS + cell]);
    const top = Math.log1p(max);
    for (let cell = 0; cell < CELLS; cell++) {
      const count = heat[offset + f * CELLS + cell];
      levels[cell] = count ? Math.round(255 * Math.log1p(count) / top) : -1;
    }
    return max;
  }

  // Fill each floor's heated tiles with the colormap, more opaque as it heats:
  // all floors in one 48 x 432 image, put once and scaled up without smoothing.
  function fillHeat(offset, ramp) {
    const image = app.heatImage, pixels = image.data, heated = [];
    for (let f = 0; f < FLOORS; f++) {
      if (!heatLevels(offset, f)) continue;
      heated.push(f);
      for (let cell = 0, at = 4 * f * CELLS; cell < CELLS; cell++, at += 4) {
        const level = app.heatLevels[cell];
        if (level < 0) { pixels[at + 3] = 0; continue; }
        pixels[at] = ramp[3 * level];
        pixels[at + 1] = ramp[3 * level + 1];
        pixels[at + 2] = ramp[3 * level + 2];
        pixels[at + 3] = 255 * (HEAT_ALPHA[0] + (HEAT_ALPHA[1] - HEAT_ALPHA[0]) * level / 255);
      }
    }
    app.heatCanvas.getContext('2d').putImageData(image, 0, 0);
    for (const f of heated) app.panels[f].heat.drawImage(app.heatCanvas, 0, f * MAP, MAP, MAP, 0, 0, SIDE, SIDE);
  }

  // Outline each floor's heated tiles in the colormap, over a movement fill;
  // one path per band of 32 levels.
  function outlineHeat(offset, ramp) {
    for (let f = 0; f < FLOORS; f++) {
      if (!heatLevels(offset, f)) continue;
      const ctx = app.panels[f].heat, bands = Array.from({ length: 8 }, () => new Path2D());
      for (let cell = 0; cell < CELLS; cell++) {
        const level = app.heatLevels[cell];
        if (level >= 0) bands[level >> 5].rect((cell % MAP) * TILE + 1.5, (cell / MAP | 0) * TILE + 1.5, TILE - 3, TILE - 3);
      }
      ctx.lineWidth = 1.5;
      bands.forEach((band, b) => {
        const level = 32 * b + 16;
        ctx.strokeStyle = `rgb(${ramp[3 * level]},${ramp[3 * level + 1]},${ramp[3 * level + 2]})`;
        ctx.stroke(band);
      });
    }
  }

  // Bottom to top: trails, deaths and timeouts, creatures, players, the followed
  // run, then wins, which nothing may cover.
  function drawGhosts() {
    const ctxs = app.panels.map(panel => panel.ghosts);
    if (app.marks) app.marks.length = 0;
    // Round caps and joins every frame: drawEnd sets them and a context keeps
    // them, so without this a trail's corners would depend on whether an end
    // mark was drawn on its board in some earlier frame, and a seek would
    // draw a step otherwise than playback reaching it.
    for (const ctx of ctxs) {
      ctx.clearRect(0, 0, SIDE, SIDE);
      ctx.lineCap = ctx.lineJoin = 'round';
    }
    // Creatures first, so the trails and the players draw over them.
    if (app.show.creatures) drawCreatures(ctxs);
    if (app.show.trails) drawTrails(ctxs);
    const wins = app.show.deaths ? drawEnds(ctxs) : [];
    drawPlayers(ctxs);
    drawFollowed(ctxs);
    drawWins(ctxs, wins);
  }

  // Each living run's last 240 moves as 1 px lines that add up where they overlap.
  function drawTrails(ctxs) {
    const { pb, set } = app, alpha = Math.min(0.5, 4 / Math.sqrt(set.n));
    for (const ctx of ctxs) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = `rgba(150,215,255,${alpha.toFixed(3)})`;
      ctx.lineWidth = 1;
    }
    for (let e = 0; e < set.n; e++) {
      const length = pb.trailLength[e];
      if (length < 2 || !pb.alive(e)) continue;
      const ctx = ctxs[pb.cur.floor[e]], head = pb.trailHead[e];
      let previous = -1;
      ctx.beginPath();
      for (let i = 0; i < length; i++) {
        const cell = pb.trail[e * D.TRAIL + (head - length + 1 + i + D.TRAIL) % D.TRAIL], row = cell / MAP | 0, col = cell % MAP;
        const adjacent = previous >= 0 && Math.abs((previous / MAP | 0) - row) + Math.abs(previous % MAP - col) === 1;
        if (adjacent) ctx.lineTo(col * TILE + 5.5, row * TILE + 5.5);
        else ctx.moveTo(col * TILE + 5.5, row * TILE + 5.5);
        previous = cell;
      }
      ctx.stroke();
    }
    for (const ctx of ctxs) ctx.globalCompositeOperation = 'source-over';
  }

  // One mark per tile for the runs that ended there by now: a red X for deaths
  // and a green check for wins, each thicker for more runs and brighter for
  // ends in the last RECENT_ENDS decisions, and a grey dot for timeouts and
  // cut-off runs. Draws all but the wins; returns those as [tile, runs, recent].
  function drawEnds(ctxs) {
    const { pb, set } = app, { endCount: count, endRecent: recent, endTouched: touched, endOrder: order } = app;
    let m = 0;
    for (let i = 0; i < order.length && set.length[order[i]] <= pb.t; i++) {
      const e = order[i], tile = set.endTile[e], at = 4 * tile + set.outcome[e];
      if (!(count[4 * tile] | count[4 * tile + 1] | count[4 * tile + 2] | count[4 * tile + 3])) touched[m++] = tile;
      count[at]++;
      if (pb.t - set.length[e] < RECENT_ENDS) recent[at] = 1;
    }
    const wins = [];
    for (let i = 0; i < m; i++) {
      const tile = touched[i], ctx = ctxs[tile / CELLS | 0], row = (tile % CELLS) / MAP | 0, col = tile % MAP, at = 4 * tile;
      if (count[at + 1] || count[at + 3]) drawDot(ctx, row, col);
      if (count[at]) drawEnd(ctx, row, col, END_MARKS.death, count[at], recent[at], false, labelled());
      if (count[at + 2]) wins.push([tile, count[at + 2], recent[at + 2]]);
      count.fill(0, at, at + 4);
      recent.fill(0, at, at + 4);
    }
    return wins;
  }

  // Counts print on end marks only where a tile is at least 10 px on screen.
  const labelled = () => app.panels[0].ghosts.canvas.clientWidth / MAP >= 10;

  // The wins, then the followed run's win, above everything else.
  function drawWins(ctxs, wins) {
    for (const [tile, k, recent] of wins) drawEnd(ctxs[tile / CELLS | 0], (tile % CELLS) / MAP | 0, tile % MAP, END_MARKS.win, k, recent, false, labelled());
    const e = app.selected, { set, pb } = app;
    if (e < 0 || pb.t < set.length[e] || D.OUTCOMES[set.outcome[e]] !== 'win') return;
    const tile = set.endTile[e], ctx = ctxs[tile / CELLS | 0], row = (tile % CELLS) / MAP | 0, col = tile % MAP, z = followScale(ctx);
    const [x, y] = [col * TILE + 5.5, row * TILE + 5.5];
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(z, z);
    ctx.translate(-x, -y);
    drawEnd(ctx, row, col, END_MARKS.win, 1, true, true, false);
    ctx.restore();
  }

  // A death's X and a win's check: the stroke within an 11 px tile, inset by
  // `inset`, and the colours of an old and a recent end.
  const END_MARKS = {
    death: { path: (p, x, y, inset) => { p.moveTo(x + inset, y + inset); p.lineTo(x + TILE - inset, y + TILE - inset); p.moveTo(x + TILE - inset, y + inset); p.lineTo(x + inset, y + TILE - inset); }, color: '#c2271d', recent: '#ff4433' },
    win: { path: (p, x, y, inset) => { p.moveTo(x + inset, y + 6); p.lineTo(x + 4.5, y + TILE - inset); p.lineTo(x + TILE - inset, y + inset); }, color: '#2fbf4a', recent: '#7dff6e' },
  };

  function drawEnd(ctx, row, col, mark, k, recent, followed, numbered) {
    const x = col * TILE, y = row * TILE, width = Math.min(4, 1.5 + 0.75 * Math.log2(k)), path = new Path2D();
    mark.path(path, x, y, followed ? -1 : 1.5);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = width + 2.5;
    ctx.strokeStyle = followed ? '#fff' : 'rgba(8,10,12,.9)';
    ctx.stroke(path);
    ctx.lineWidth = width;
    ctx.strokeStyle = recent ? mark.recent : mark.color;
    ctx.stroke(path);
    if (k < 3 || !numbered) return;
    ctx.font = '700 7px system-ui, sans-serif';
    ctx.textBaseline = 'top';
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000';
    ctx.strokeText(String(k), x + 0.5, y + 0.5);
    ctx.fillStyle = '#fff';
    ctx.fillText(String(k), x + 0.5, y + 0.5);
  }

  function drawDot(ctx, row, col) {
    ctx.fillStyle = '#b0bec5';
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(col * TILE + 5.5, row * TILE + 5.5, 2.6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  // Creatures of the living runs, bucketed by (tile, class, facing, type):
  // each run's sample at its own decision.
  function drawCreatures(ctxs) {
    const { pb, set } = app, { creatureCount: counts, creatureTouched: touched } = app, size = set.creatureBytes;
    let m = 0;
    for (let e = 0; e < set.n; e++) {
      if (!pb.alive(e)) continue;
      const [bytes, at] = creatureSample(e);
      if (at < 0) continue;
      const base = pb.sampleFloor[e] * CELLS;
      for (let c = 0, k = at + 1; c < bytes[at]; c++, k += size) {
        const facing = size > 3 ? bytes[k + 3] : 0, key = ((base + bytes[k + 1] * MAP + bytes[k + 2]) * 5 + (bytes[k] >> 4)) * 40 + facing * 8 + (bytes[k] & 15);
        if (!counts[key]++) touched[m++] = key;
      }
    }
    for (let i = 0; i < m; i++) {
      const key = touched[i], type = key & 7, facing = (key % 40) >> 3, klass = (key / 40 | 0) % 5, tile = (key / 40 | 0) / 5 | 0, ctx = ctxs[tile / CELLS | 0];
      const shade = creatureShade(tile), count = counts[key];
      counts[key] = 0;
      if (shade < 0) continue;
      ctx.globalAlpha = (ENEMY[klass] ? app.enemyAlpha : app.alpha)[count];
      drawMark(ctx, markRow(klass, facing), type, (tile % CELLS) / MAP | 0, tile % MAP, shade);
    }
    for (const ctx of ctxs) ctx.globalAlpha = 1;
  }

  // Run e's creature sample on screen, as [bytes, offset of its run] (offset
  // -1 for none): mid-sleep, the sleep's sample, the creatures as they were
  // at that tick; else its decision's sample from its creature window.
  function creatureSample(e) {
    const { pb, set } = app, u = pb.t;
    if (D.sampleAt(set, e, u)) return [set.sleepRuns[set.group[e]], set.runAt[e][u]];
    const d = pb.decision[e], win = app.windows.get(Math.floor(d / set.window));
    if (!win?.index) return [null, -1];
    return [win.files[set.group[e]], win.index[e * win.per + ((d - win.j * set.window) / set.stride | 0)]];
  }

  // Living players bucketed by tile, drawn facing their bucket's majority,
  // or asleep where most of a tile's runs sleep.
  function drawPlayers(ctxs) {
    const { pb, set } = app, { playerCount: counts, playerFacing: facing, playerTouched: touched } = app, cur = pb.cur;
    let m = 0;
    for (let e = 0; e < set.n; e++) {
      if (!pb.alive(e)) continue;
      const tile = cur.floor[e] * CELLS + cur.row[e] * MAP + cur.col[e];
      if (!counts[tile]++) touched[m++] = tile;
      facing[tile * 5 + (D.sampleAt(set, e, pb.t) ? 4 : cur.facing[e] - 1)]++;
    }
    for (let i = 0; i < m; i++) {
      const tile = touched[i], ctx = ctxs[tile / CELLS | 0];
      let best = 0;
      for (let f = 1; f < 5; f++) if (facing[tile * 5 + f] > facing[tile * 5 + best]) best = f;
      ctx.globalAlpha = app.playerAlpha[counts[tile]];
      drawMark(ctx, 0, best, (tile % CELLS) / MAP | 0, tile % MAP);
      counts[tile] = 0;
      facing.fill(0, tile * 5, tile * 5 + 5);
    }
    for (const ctx of ctxs) ctx.globalAlpha = 1;
  }

  // The followed run: its last 800 moves in FOLLOW over a dark casing, fading
  // with age in 16 bands, its creatures and itself opaque with its thicker,
  // brighter ring (RINGS.followed), or its end mark outlined in white.
  function drawFollowed(ctxs) {
    const e = app.selected, states = app.selectedStates;
    if (e < 0 || !states) return;
    const { pb, set } = app, s = pb.decision[e];
    const floor = states[4 * s], row = states[4 * s + 1], col = states[4 * s + 2], ctx = ctxs[floor], z = followScale(ctx);
    app.panels.forEach((panel, f) => panel.article.classList.toggle('followed', f === floor));
    const points = D.pathTail(states, s, FOLLOWED_MOVES), bands = Array.from({ length: 16 }, () => new Path2D());
    for (let i = 1; i < points.length; i++) {
      const [a, b] = [points[i - 1], points[i]];
      if (Math.abs((a / MAP | 0) - (b / MAP | 0)) + Math.abs(a % MAP - b % MAP) !== 1) continue;
      const band = bands[Math.floor(16 * i / points.length)];
      band.moveTo((a % MAP) * TILE + 5.5, (a / MAP | 0) * TILE + 5.5);
      band.lineTo((b % MAP) * TILE + 5.5, (b / MAP | 0) * TILE + 5.5);
    }
    bands.forEach((band, k) => {
      const alpha = 0.15 + 0.85 * (k + 1) / 16;
      ctx.lineWidth = 4 * z;
      ctx.strokeStyle = `rgba(0,0,0,${(0.7 * alpha).toFixed(3)})`;
      ctx.stroke(band);
      ctx.lineWidth = 2 * z;
      ctx.strokeStyle = `rgba(${FOLLOW},${alpha.toFixed(3)})`;
      ctx.stroke(band);
    });
    if (!pb.alive(e)) {
      const outcome = set.outcome[e];
      if (outcome === 0) drawEnd(ctx, row, col, END_MARKS.death, 1, true, true, false);
      else if (outcome !== 2) drawDot(ctx, row, col);
      return;
    }
    const [bytes, at] = creatureSample(e), asleep = D.sampleAt(set, e, pb.t) > 0;
    if (at >= 0) {
      const sampleCtx = ctxs[pb.sampleFloor[e]];
      for (let c = 0, k = at + 1; c < bytes[at]; c++, k += set.creatureBytes) {
        const klass = bytes[k] >> 4, shade = creatureShade(pb.sampleFloor[e] * CELLS + bytes[k + 1] * MAP + bytes[k + 2]);
        if (shade < 0) continue;
        // Its enemies as faint as any run's most-seen ones.
        sampleCtx.globalAlpha = ENEMY[klass] ? ENEMY_OPACITY[1] : 1;
        drawMark(sampleCtx, markRow(klass, set.creatureBytes > 3 ? bytes[k + 3] : 0), bytes[k] & 15, bytes[k + 1], bytes[k + 2], shade);
      }
      sampleCtx.globalAlpha = 1;
    }
    drawMark(ctx, 0, asleep ? 4 : states[4 * s + 3] - 1, row, col);
    if (asleep) {
      // A cased "z" over the followed sleeper, so its sleep reads at a glance.
      ctx.font = `700 ${Math.round(9 * z)}px system-ui, sans-serif`;
      ctx.lineWidth = 3 * z;
      ctx.strokeStyle = '#000';
      ctx.fillStyle = `rgb(${FOLLOW})`;
      ctx.strokeText('z', col * TILE + 7, row * TILE - 1);
      ctx.fillText('z', col * TILE + 7, row * TILE - 1);
    }
    ring(ctx, col * TILE + 5.5, row * TILE + 5.5, 'followed', ctx.canvas.clientWidth ? SIDE / ctx.canvas.clientWidth : app.atlas.scale);
    app.marks?.push(['followed', floor, row, col]);
  }

  function updateReadouts() {
    const { pb, set } = app, t = pb.t, here = new Array(FLOORS).fill(0), reached = new Array(FLOORS).fill(0), died = new Array(FLOORS).fill(0);
    for (let e = 0; e < set.n; e++) {
      if (pb.alive(e)) here[pb.cur.floor[e]]++;
      else if (set.outcome[e] === 0) died[set.endTile[e] / CELLS | 0]++;
      for (let f = 0; f < FLOORS; f++) {
        const entry = set.floorFirst[e * FLOORS + f];
        if (entry >= 0 && entry <= pb.decision[e]) reached[f]++;
      }
    }
    app.panels.forEach((panel, f) => {
      const stats = `here ${fmt(here[f])} · reached ${fmt(reached[f])} · died ${fmt(died[f])}`;
      if (stats !== panel.text) panel.stats.textContent = panel.article.title = panel.text = stats;
    });
    app.won = D.wonBy(app.winSteps, t);
    if (app.layout === 'follow') text('won-count', `✓${fmt(app.won)}`);
    text('position', `${set.mapped ? 'Step' : 'Decision'} ${fmt(t)} of ${fmt(set.maxSteps)} · ${fmt(app.won)} won · ${fmt(here.reduce((a, b) => a + b))} of ${fmt(set.n)} alive`);
    const seek = $('seek'), scrub = $('scrub');
    if (seek) seek.value = String(app.u);
    if (scrub) {
      scrub.setAttribute('aria-valuenow', String(app.u));
      scrub.setAttribute('aria-valuetext', `${set.mapped ? 'Step' : 'Decision'} ${fmt(t)} of ${fmt(set.maxSteps)}`);
    }
    const e = app.selected;
    if (e >= 0) {
      const s = pb.decision[e], floor = app.selectedStates[4 * s], at = set.mapped ? `decision ${fmt(s)} · ` : '';
      const k = D.sampleAt(set, e, t), sleep = k && set.sleeps[e], ticks = sleep ? sleep.ticks[Array.prototype.indexOf.call(sleep.decisions, s)] : 0;
      const where = pb.alive(e) ? `on ${FLOOR_NAMES[floor]}` : `${['died', 'timed out', 'won', 'was cut off'][set.outcome[e]]} on ${FLOOR_NAMES[floor]} at decision ${fmt(set.decisions[e])}`;
      const state = k ? `asleep on ${FLOOR_NAMES[floor]}, tick ${fmt(k * app.manifest.sleep_stride)} of ${fmt(ticks)}` : where;
      text('episode-info', `${at}return ${D.returnAt(set, e, s)} · ${state}`);
    }
  }

  function renderStats() {
    const { stats, collection } = app, pct = value => `${(100 * value).toFixed(1)}%`, best = app.manifest.achievement_rewards.reduce((a, b) => a + b, 0);
    const card = (label, value, note) => `<div class="stat"><span>${label}</span><strong>${value}</strong>${note ? `<span>${note}</span>` : ''}</div>`;
    const bars = stats.reached.map((count, f) => `<div class="reach-bar" title="${FLOOR_NAMES[f]}: ${fmt(count)} of ${fmt(stats.count)} runs" style="--bar:var(--floor-${f})"><b>${Math.round(100 * count / stats.count)}</b><i style="height:${(30 * count / stats.count).toFixed(1)}px"></i></div>`).join('');
    const deaths = stats.deaths.reduce((a, b) => a + b, 0), { composition: c } = collection, range = ([low, high]) => `${fmt(low)} to ${fmt(high)}`;
    const cards = [
      card('Runs shown', fmt(stats.count), `${fmt(c.qualified)} ${collection.id === 'all' ? 'captured' : 'qualify'} of ${fmt(c.run)} run`),
      card('Mean return', stats.mean_return.toFixed(1), `${pct(stats.mean_return / best)} of ${best}`),
      card('Deaths', fmt(deaths), pct(deaths / stats.count)),
      card('Timeouts', fmt(stats.timeouts), pct(stats.timeouts / stats.count)),
      card('Wins', fmt(stats.wins), pct(stats.wins / stats.count)),
      card('Mean length', fmt(stats.mean_decisions), 'decisions'),
    ];
    if (collection.id !== 'all') {
      const lengths = [c.death_decisions.length && `deaths in ${range(c.death_decisions)}`, c.win_decisions.length && `wins in ${range(c.win_decisions)}`].filter(Boolean).join(', ');
      const added = c.added_wins ? `; ${fmt(c.added_wins)} longer wins added to keep wins at their ${pct(c.natural_win_share)} share` : '';
      cards.push(card(collection.label, `${fmt(c.deaths)} deaths, ${fmt(c.wins)} wins`, `${lengths} decisions${added}`));
    }
    cards.push(`<div class="stat reach"><span>Reach per floor, % of runs (floor 0 to 8)</span><div class="reach-bars">${bars}</div></div>`);
    $('stats').innerHTML = cards.join('');
  }

  // Runs on each floor every 64 display steps, stacked, and on top the runs
  // that have won by then; along the top a strip of ticks, wins above and
  // deaths below, each more opaque where more fall in one pixel column; the
  // followed run's win as a cased check over a line; and, when quiet
  // stretches are skipped, a mark where each was cut. The x axis is the
  // timeline: what playback shows, in order. Colours come from the host's
  // tokens (--surface, --floor-0..8, --death, --ink) as the viewer inherits them.
  function drawChart() {
    const canvas = $('occupancy'), scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
    const { width: w, height: h } = canvas, image = document.createElement('canvas');
    image.width = w;
    image.height = h;
    const ctx = image.getContext('2d'), style = getComputedStyle(app.root);
    ctx.fillStyle = style.getPropertyValue('--surface');
    ctx.fillRect(0, 0, w, h);
    app.chart = image;
    if (!app.pre || !app.timeline) return;
    const { set, pre, timeline } = app, samples = pre.occupancy.length / FLOORS, span = Math.max(1, timeline.length);
    // The tick bands over the plot, flush with it: wins, then deaths, each
    // only when some run ends so.
    const band = 5 * scale, winBand = set.outcome.includes(2) ? band : 0, deathBand = set.outcome.includes(0) ? band : 0, strip = winBand + deathBand;
    const x = t => timeline.toIndex(t) / span * w, y = v => h - v / set.n * (h - strip);
    const below = new Float64Array(samples);
    for (let f = 0; f < FLOORS; f++) {
      ctx.beginPath();
      for (let i = 0; i < samples; i++) ctx.lineTo(x(i << D.OCCUPANCY_SHIFT), y(below[i] + pre.occupancy[i * FLOORS + f]));
      for (let i = samples - 1; i >= 0; i--) ctx.lineTo(x(i << D.OCCUPANCY_SHIFT), y(below[i]));
      ctx.closePath();
      ctx.fillStyle = style.getPropertyValue(`--floor-${f}`);
      ctx.fill();
      for (let i = 0; i < samples; i++) below[i] += pre.occupancy[i * FLOORS + f];
    }
    if (app.winSteps.length) {
      const won = i => below[i] + D.wonBy(app.winSteps, i << D.OCCUPANCY_SHIFT), edge = new Path2D();
      ctx.beginPath();
      for (let i = 0; i < samples; i++) {
        ctx.lineTo(x(i << D.OCCUPANCY_SHIFT), y(won(i)));
        edge.lineTo(x(i << D.OCCUPANCY_SHIFT), y(won(i)));
      }
      for (let i = samples - 1; i >= 0; i--) ctx.lineTo(x(i << D.OCCUPANCY_SHIFT), y(below[i]));
      ctx.closePath();
      ctx.fillStyle = WON_FILL;
      ctx.fill();
      ctx.strokeStyle = WON_EDGE;
      ctx.lineWidth = scale;
      ctx.stroke(edge);
    }
    ticks(ctx, set, x, w, e => set.outcome[e] === 2, WIN_TICK, 0, winBand);
    // Deaths in a darker red than the play head, short ticks at the edge.
    ticks(ctx, set, x, w, e => set.outcome[e] === 0, darker(style.getPropertyValue('--death').trim() || '#d93a30', 0.7), winBand, deathBand);
    ctx.strokeStyle = style.getPropertyValue('--ink');
    ctx.setLineDash([2 * scale, 2 * scale]);
    ctx.lineWidth = scale;
    for (const [from] of timeline.skipped()) {
      ctx.beginPath();
      ctx.moveTo(Math.round(x(from)) + 0.5, 0);
      ctx.lineTo(Math.round(x(from)) + 0.5, h);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const e = app.selected;
    if (e >= 0 && set.outcome[e] === 2) {
      const at = Math.round(x(set.length[e])) + 0.5;
      for (const [color, width] of [['rgba(0,0,0,.75)', 3], [`rgb(${FOLLOW})`, 1]]) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width * scale;
        ctx.beginPath();
        ctx.moveTo(at, strip);
        ctx.lineTo(at, h);
        ctx.stroke();
      }
      const r = Math.max(3.5 * scale, Math.min(6 * scale, winBand * 0.8));
      check(ctx, Math.min(w - r - scale, Math.max(r + scale, at)), Math.max(r * 0.8, winBand / 2), r);
    }
    updateFollowWin();
  }

  // One tick per pixel column holding an end of the kind `pick` takes, more
  // opaque for more ends there.
  function ticks(ctx, set, x, w, pick, color, top, height) {
    const columns = new Uint16Array(w + 1);
    for (let e = 0; e < set.n; e++) if (pick(e)) columns[Math.min(w, Math.max(0, Math.floor(x(set.length[e]))))]++;
    ctx.fillStyle = color;
    for (let c = 0; c <= w; c++) {
      if (!columns[c]) continue;
      ctx.globalAlpha = Math.min(1, 0.4 + 0.2 * Math.log2(columns[c]));
      ctx.fillRect(c, top, 1, height);
    }
    ctx.globalAlpha = 1;
  }

  // The followed run's win check, cased in black, centred on (x, y).
  function check(ctx, x, y, r) {
    const path = new Path2D();
    path.moveTo(x - r, y);
    path.lineTo(x - r / 3, y + r * 0.7);
    path.lineTo(x + r, y - r * 0.8);
    ctx.lineCap = ctx.lineJoin = 'round';
    for (const [color, width] of [['#000', r * 0.75], [`rgb(${FOLLOW})`, r * 0.4]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke(path);
    }
  }

  // The followed run's win on the seek bar's track, where the thumb will stand
  // when it wins; hidden for a run that does not win.
  function updateFollowWin() {
    const mark = $('follow-win'), e = app.selected, { set, timeline } = app;
    if (!mark) return;
    mark.hidden = !(e >= 0 && set && timeline && set.outcome[e] === 2);
    if (!mark.hidden) mark.style.setProperty('--at', String(timeline.toIndex(set.length[e]) / Math.max(1, timeline.length)));
  }

  function drawPlayhead() {
    const canvas = $('occupancy');
    if (!app.chart) return;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(app.chart, 0, 0);
    drawCursor(ctx, app.u / Math.max(1, app.timeline.length) * canvas.width, canvas);
  }

  // The play head of both plots, from one definition: a bright red line,
  // 3 px wide and cased in near-black, the plot's full height, with a caret
  // along the top edge centred on it, the two clamped together at the ends
  // so the caret shows whole and the line stays under its tip. It reads on
  // every floor colour, the won band and either theme; the death ticks are a
  // darker red, short and at the edge.
  const CURSOR = { color: '#ff2a2a', casing: 'rgba(8,10,12,.9)', line: 3, caret: 13, tall: 8 };
  function drawCursor(ctx, x, canvas, left = 0, right = canvas.width) {
    const scale = window.devicePixelRatio || 1, half = CURSOR.caret * scale / 2;
    const at = Math.min(right - half - scale, Math.max(left + half + scale, x));
    ctx.save();
    ctx.lineCap = 'butt';
    for (const [color, width] of [[CURSOR.casing, CURSOR.line + 2], [CURSOR.color, CURSOR.line]]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width * scale;
      ctx.beginPath();
      ctx.moveTo(at, 0);
      ctx.lineTo(at, canvas.height);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(at - half, scale / 2);
    ctx.lineTo(at + half, scale / 2);
    ctx.lineTo(at, CURSOR.tall * scale);
    ctx.closePath();
    ctx.fillStyle = CURSOR.color;
    ctx.strokeStyle = CURSOR.casing;
    ctx.lineWidth = scale;
    ctx.lineJoin = 'round';
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    app.cursors ??= {};
    app.cursors[canvas.dataset.g] = at / scale;
  }

  // The reward plot's scale: gridlines every `step` points up to `top`, a
  // round number above the largest return, or the best possible return when
  // that is nearer.
  function rewardScale() {
    const best = app.manifest.achievement_rewards.reduce((a, b) => a + b, 0), most = Math.max(10, app.reward?.top ?? 0);
    const step = [5, 10, 20, 25, 50, 100].find(s => most / s <= 5) ?? 100;
    return { best, top: Math.min(best, Math.ceil(most / step) * step), step };
  }

  // Every run's achievement return over the timeline, thin and faint so they
  // add up where runs agree, and the followed run's, in its green. Drawn once per
  // set, timeline, followed run, size or theme; each frame adds only the cursor.
  function drawReward() {
    const canvas = $('reward');
    if (!canvas) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
    const { width: w, height: h } = canvas, image = document.createElement('canvas');
    image.width = w;
    image.height = h;
    const ctx = image.getContext('2d'), style = getComputedStyle(app.root);
    ctx.fillStyle = style.getPropertyValue('--surface');
    ctx.fillRect(0, 0, w, h);
    app.rewardImage = image;
    if (!app.reward || !app.timeline) return;
    const { set, timeline, reward } = app, span = Math.max(1, timeline.length), { best, top, step } = rewardScale();
    // Left of the plot area: the axis title, "Reward", turned up its edge, and
    // the points; right of it the shares. Nothing of the plot crosses them.
    const ink = style.getPropertyValue('--ink'), muted = style.getPropertyValue('--muted') || ink;
    ctx.font = `${10 * scale}px ${style.getPropertyValue('--font-data') || 'ui-monospace, monospace'}`;
    ctx.textBaseline = 'bottom';
    const gutter = REWARD_GUTTER * scale, left = gutter + ctx.measureText(String(top)).width + 7 * scale, right = w - ctx.measureText('100%').width - 7 * scale;
    const pad = 7 * scale, x = u => left + timeline.toIndex(u) / span * (right - left), y = v => h - pad - v / top * (h - 2 * pad);
    app.rewardY = y;
    app.rewardArea = { left, right };
    const lines = Array.from({ length: Math.floor(top / step) + 1 }, (_, k) => k * step);
    if (top % step) lines.push(top);
    for (const v of lines) {
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = ink;
      ctx.fillRect(left, Math.round(y(v)), right - left, scale);
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = muted;
      // A label sits above its line, or below it at the top; a line too near
      // the top's keeps no label.
      const at = v === top ? y(v) + 11 * scale : y(v) - scale;
      if (v !== top && y(v) - y(top) < 24 * scale) continue;
      ctx.textAlign = 'right';
      ctx.fillText(String(v), left - 4 * scale, at);
      ctx.textAlign = 'left';
      ctx.fillText(`${Math.round(100 * v / best)}%`, right + 4 * scale, at);
    }
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = muted;
    ctx.save();
    ctx.translate(gutter / 2, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = `600 ${11 * scale}px ${style.getPropertyValue('--font-data') || 'ui-monospace, monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Reward', 0, 0);
    ctx.restore();
    ctx.globalAlpha = 1;
    const line = curve => {
      const path = new Path2D();
      path.moveTo(left, y(0));
      let value = 0;
      for (let k = 0; k < curve.steps.length; k++) {
        const at = x(curve.steps[k]);
        path.lineTo(at, y(value));
        value = curve.values[k];
        path.lineTo(at, y(value));
      }
      path.lineTo(right, y(value));
      return path;
    };
    ctx.strokeStyle = ink;
    ctx.globalAlpha = Math.min(0.5, 2.5 / Math.sqrt(set.n));
    ctx.lineWidth = scale;
    for (const run of reward.runs) ctx.stroke(line(run));
    ctx.globalAlpha = 1;
    const e = app.selected;
    if (e >= 0) {
      const path = line(reward.runs[e]);
      for (const [color, width] of [['rgba(0,0,0,.7)', 3.5], [`rgb(${FOLLOW})`, 2]]) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width * scale;
        ctx.stroke(path);
      }
    }
  }

  // The reward plot at the playhead: the static plot, a cursor in step with
  // the timeline's, and a dot on the followed run's return.
  function drawRewardCursor() {
    const canvas = $('reward');
    if (!canvas || !app.rewardImage) return;
    const ctx = canvas.getContext('2d'), scale = window.devicePixelRatio || 1, { left, right } = app.rewardArea ?? { left: 0, right: canvas.width };
    const x = left + app.u / Math.max(1, app.timeline.length) * (right - left);
    ctx.drawImage(app.rewardImage, 0, 0);
    drawCursor(ctx, x, canvas, left, right);
    const e = app.selected;
    if (e < 0 || !app.rewardY) return;
    ctx.beginPath();
    ctx.arc(x, app.rewardY(D.curveAt(app.reward.runs[e], app.pb.t)), 3.5 * scale, 0, 2 * Math.PI);
    ctx.fillStyle = `rgb(${FOLLOW})`;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1.5 * scale;
    ctx.fill();
    ctx.stroke();
  }

  // The Two rows layout's tenth square: how many runs have won by now, the
  // number large over one mark per winner, filled in the order they win; the
  // followed run's mark bright green and cased once it has won, outlined
  // before. With no winner it says how far the tier gets instead. Redrawn
  // only when what it shows changes.
  function drawTally() {
    const { tally, set, pb } = app;
    if (!tally || app.layout !== 'two-rows') return;
    const won = D.wonBy(app.winSteps, pb.t), total = app.winOrder.length, key = `${set.n}/${total}/${won}/${app.selected}/${app.winSteps === tally.steps}`;
    if (key === tally.drawn) return;
    Object.assign(tally, { drawn: key, steps: app.winSteps, won });
    const ctx = tally.ctx, followed = app.winOrder.indexOf(app.selected);
    ctx.fillStyle = '#070d11';
    ctx.fillRect(0, 0, SIDE, SIDE);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = won ? '#7dff6e' : '#e6edf0';
    ctx.font = '700 120px system-ui, sans-serif';
    ctx.fillText(fmt(won), 22, 150);
    const number = ctx.measureText(fmt(won)).width;
    ctx.fillStyle = '#e6edf0';
    ctx.font = '600 44px system-ui, sans-serif';
    ctx.fillText('won', 40 + number, 150);
    ctx.fillStyle = '#9fb0b6';
    ctx.font = '600 34px system-ui, sans-serif';
    ctx.fillText(`of ${fmt(set.n)} runs`, 22, 200);
    if (!total) {
      const furthest = app.stats.reached.findLastIndex(count => count > 0);
      ctx.fillText('None beats the game.', 22, 280);
      ctx.fillText('The furthest any run gets:', 22, 330);
      ctx.fillStyle = '#e6edf0';
      ctx.fillText(`floor ${furthest}, ${FLOOR_NAMES[furthest]}`, 22, 380);
      return;
    }
    ctx.fillText(`${fmt(total)} win by the end`, 22, 244);
    const area = { x: 22, y: 270, w: SIDE - 44, h: SIDE - 292 }, cols = Math.ceil(Math.sqrt(total * area.w / area.h)), rows = Math.ceil(total / cols);
    const cell = Math.min(area.w / cols, area.h / rows), gap = cell >= 6 ? Math.max(1, cell / 6) : 0;
    for (let k = 0; k < total; k++) {
      const cx = area.x + (k % cols) * cell, cy = area.y + Math.floor(k / cols) * cell;
      ctx.fillStyle = k < won ? WIN_TICK : 'rgba(255,255,255,.1)';
      ctx.fillRect(cx, cy, cell - gap, cell - gap);
    }
    if (followed < 0) return;
    const cx = area.x + (followed % cols) * cell + (cell - gap) / 2, cy = area.y + Math.floor(followed / cols) * cell + (cell - gap) / 2, r = Math.max(cell, 14) / 2 + 2;
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#000';
    ctx.strokeRect(cx - r, cy - r, 2 * r, 2 * r);
    if (followed < won) {
      ctx.fillStyle = `rgb(${FOLLOW})`;
      ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);
    } else {
      ctx.lineWidth = 2;
      ctx.strokeStyle = `rgb(${FOLLOW})`;
      ctx.strokeRect(cx - r, cy - r, 2 * r, 2 * r);
    }
  }

  // The creature windows the living runs' decisions fall in, the next one
  // when a run is past a window's middle, and the one before, kept.
  function ensureWindows() {
    const { pb, set, groups } = app, want = new Set(), windows = Math.max(...groups.map(g => g.windows));
    for (let e = 0; e < set.n; e++) {
      if (!pb.alive(e)) continue;
      const d = pb.decision[e], j = Math.floor(d / set.window);
      want.add(j);
      if (d - j * set.window > set.window / 2) want.add(j + 1);
    }
    for (const k of [...app.windows.keys()]) if (!want.has(k) && !want.has(k + 1)) app.windows.delete(k);
    for (const k of want) if (k < windows && !app.windows.has(k)) loadWindow(k);
  }

  async function loadWindow(j) {
    const entry = {}, { set, groups } = app, version = generation;
    app.windows.set(j, entry);
    try {
      const parts = await Promise.all(groups.map(g => (j < g.windows ? loadFile(`${g.path}/creatures-w${j}.bin.gz`, false) : null)));
      if (version !== generation || app.windows.get(j) !== entry) return;
      Object.assign(entry, D.indexWindow(set, j, parts));
      if (!app.playing) render();
    } catch (error) {
      if (version === generation) fail(error);
    }
  }

  // Follow a run by rule: the shortest win (the median-return run when none
  // won), by return or length, or, in a time-mapped set, its pinned run.
  function follow(choice) {
    const { set } = app, order = Int32Array.from({ length: set.n }, (_, e) => e).sort((a, b) => set.ret[a] - set.ret[b] || a - b);
    const pick = {
      'shortest-win': () => D.shortestWin(set), median: () => order[order.length >> 1], best: () => order[order.length - 1], worst: () => order[0],
      longest: () => set.decisions.indexOf(set.maxDecisions), random: () => Math.floor(Math.random() * set.n), pinned,
    }[choice]();
    select(pick >= 0 ? pick : order[order.length >> 1]);
  }

  // A time-mapped set's pinned run: the one its time map leaves unbroken
  // (time_map.unbroken), which must be the run of the sampling seed the host
  // names (data-follow-seed-wins) when it names one, and must show every
  // decision. Anything else stops the viewer rather than follow another run.
  function pinned() {
    const { set, collection } = app, e = collection.time_map?.unbroken ?? -1, seed = app.o.winsFollowSeed;
    if (e < 0 || e >= set.n) throw Error(`the ${collection.label.toLowerCase()} set pins no unbroken run to follow.`);
    if (seed && set.seeds[e] !== seed) throw Error(`the pinned run has sampling seed ${set.seeds[e]}, not ${seed}${set.seeds.includes(seed) ? '' : `, which is not among its ${fmt(set.n)} runs`}.`);
    // Its time map keeps every decision: none, or, with sleep shown, only its
    // sleeps' samples between them.
    const shown = set.shown[e], sub = set.sub[e], kept = !shown ? set.decisions[e] : sub ? sub.reduce((n, k) => n + !k, 0) : shown.length;
    if (kept !== set.decisions[e]) throw Error(`the pinned run of seed ${set.seeds[e]} skips decisions; its time map must keep all ${fmt(set.decisions[e])}.`);
    return e;
  }

  function select(e) {
    const { set } = app;
    if (!set || !(e >= 0 && e < set.n)) return;
    app.selected = e;
    app.selectedStates = D.decodeEpisode(set, e).states;
    app.selectedWindow = stripWindows(app.selectedStates, set.decisions[e], app.followBoards);
    const input = $('episode');
    if (input) Object.assign(input, { max: String(set.n - 1), value: String(e) });
    showPolicy().catch(fail);
    if (!app.pb || !app.timeline) return;
    drawChart();
    drawReward();
    render();
  }

  // Follow the run under a click: a living ghost on that tile, else one that ended there.
  function followAt(f, event) {
    const { pb, set } = app;
    if (!pb) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left) / rect.width * MAP), row = Math.floor((event.clientY - rect.top) / rect.height * MAP);
    const tile = f * CELLS + row * MAP + col, cur = pb.cur, living = [], ended = [];
    for (let e = 0; e < set.n; e++) {
      if (pb.alive(e) && cur.floor[e] * CELLS + cur.row[e] * MAP + cur.col[e] === tile) living.push(e);
      else if (!pb.alive(e) && set.endTile[e] === tile) ended.push(e);
    }
    const candidates = living.length ? living : ended;
    if (candidates.length) select(candidates[(candidates.indexOf(app.selected) + 1) % candidates.length]);
  }

  function play() {
    if (app.playing) return stop();
    if (app.pb.t >= app.set.maxSteps) goTo(0);
    Object.assign(app, { playing: true, last: 0, carry: 0, endedAt: 0 });
    playLabel(true);
    requestAnimationFrame(frame);
  }

  function stop() {
    if (!app) return;
    app.playing = false;
    playLabel(false);
  }

  // The play button's label: its word, or, on an icon button, a play
  // triangle or pause bars with the word as its accessible name and tooltip.
  const PLAY_ICON = '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" focusable="false"><path d="M4.5 2.5v11l9-5.5z" fill="currentColor"/></svg>';
  const RESTART_ICON = '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" focusable="false"><path d="M3 2.5h2.2v11H3zM13.5 2.5v11L5.8 8z" fill="currentColor"/></svg>';
  const PAUSE_ICON = '<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" focusable="false"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z" fill="currentColor"/></svg>';
  function playLabel(playing) {
    const button = $('play'), word = playing ? 'Pause' : 'Play';
    if (!button) return;
    if (!button.classList.contains('icon-play')) { button.textContent = word; return; }
    button.innerHTML = playing ? PAUSE_ICON : PLAY_ICON;
    button.setAttribute('aria-label', word);
    button.title = word;
  }

  // Playback walks the timeline at the chosen speed; a skipped stretch is one
  // step, so quiet stretches pass in a blink. At the end it holds the last
  // step for LOOP_HOLD ms and, looping (the default), starts again from step
  // 0, trails and heat cleared, still playing; a reader's pause or seek stops
  // it until they press play.
  const LOOP_HOLD = 1200;
  function frame(now) {
    if (!app?.playing) return;
    // A hidden tab pauses frames; cap the catch-up so it resumes where it was.
    const elapsed = app.last ? Math.min(100, now - app.last) : 0;
    app.last = now;
    if (app.u >= app.timeline.length) {
      app.endedAt ||= now;
      if (!app.o.loop) return stop();
      if (now - app.endedAt >= LOOP_HOLD) {
        Object.assign(app, { endedAt: 0, carry: 0 });
        goTo(0);
      }
      return requestAnimationFrame(frame);
    }
    app.carry += elapsed * Number($('speed').value) / 1000;
    const steps = Math.floor(app.carry);
    if (steps) {
      app.carry -= steps;
      app.u = Math.min(app.timeline.length, app.u + steps);
      goTo(app.timeline.toDecision(app.u), true);
    }
    requestAnimationFrame(frame);
  }

  // Seek at most once per frame while the scrubber or chart is dragged; both
  // work in timeline positions. A reader's seek holds playback until they
  // press play.
  function requestSeek(u) {
    stop();
    app.held = true;
    if (app.pendingSeek < 0) requestAnimationFrame(() => { if (!app) return; const to = app.pendingSeek; app.pendingSeek = -1; goTo(app.timeline.toDecision(to)); });
    app.pendingSeek = u;
  }

  // A drag on a plot seeks to where it is over the plot's time axis; the
  // reward plot's starts after its gutter.
  function seekFromChart(event) {
    if (!app.pb || !(event.buttons & 1)) return;
    const rect = event.currentTarget.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
    const area = event.currentTarget === $('reward') && app.rewardArea ? { left: app.rewardArea.left / scale, right: app.rewardArea.right / scale } : { left: 0, right: rect.width };
    requestSeek(Math.round(Math.max(0, Math.min(1, (event.clientX - rect.left - area.left) / (area.right - area.left))) * app.timeline.length));
  }

  // Back to step 0 as the loop's wrap does: trails, heat, tally, the Follow
  // strip and the policy view start over, and playback keeps its state.
  function restart() {
    Object.assign(app, { carry: 0, endedAt: 0, last: 0 });
    goTo(0);
  }

  function nudge(delta) {
    stop();
    goTo(app.timeline.toDecision(app.timeline.toIndex(app.pb.t) + delta));
  }

  function setStatus(message) { text('status', message); }

  function fail(error) {
    stop();
    setStatus(`Could not load: ${error.message}`);
    console.error(error);
  }

  // Wire what this host has: an embed has no pickers, layer toggles or keys.
  function wire() {
    const on = (name, event, handler) => { const node = $(name); if (node) node[`on${event}`] = handler; };
    on('play', 'click', () => { if (!app.pb) return; app.held = app.playing; play(); });
    on('restart', 'click', () => app.pb && restart());
    on('back', 'click', () => app.pb && nudge(-1));
    on('forward', 'click', () => app.pb && nudge(1));
    on('skip', 'click', () => app.pb && nudge(100));
    on('seek', 'input', event => app.pb && requestSeek(Number(event.target.value)));
    for (const chart of [$('scrub') ? 'scrub' : 'occupancy', 'reward']) {
      on(chart, 'pointerdown', event => { event.currentTarget.setPointerCapture(event.pointerId); seekFromChart(event); });
      on(chart, 'pointermove', seekFromChart);
    }
    // The plot as the scrubber, by keyboard: arrows a step (Shift, a hundred),
    // Page Up and Down a tenth, Home and End the ends, Space play or pause.
    on('scrub', 'keydown', event => {
      if (!app.pb) return;
      const span = app.timeline.length, u = app.timeline.toIndex(app.pb.t), far = Math.max(1, Math.round(span / 10));
      const to = {
        ArrowRight: u + (event.shiftKey ? 100 : 1), ArrowUp: u + (event.shiftKey ? 100 : 1), ArrowLeft: u - (event.shiftKey ? 100 : 1), ArrowDown: u - (event.shiftKey ? 100 : 1),
        PageUp: u + far, PageDown: u - far, Home: 0, End: span,
      }[event.key];
      if (event.key === ' ') { event.preventDefault(); app.held = app.playing; play(); return; }
      if (to === undefined) return;
      event.preventDefault();
      stop();
      app.held = true;
      goTo(app.timeline.toDecision(Math.max(0, Math.min(span, to))));
    });
    on('skip-quiet', 'change', event => {
      app.skipQuiet = event.target.checked;
      if (!app.pb) return;
      useTimeline();
      goTo(app.pb.t);
    });
    on('show-sleep', 'change', event => {
      app.showSleep = event.target.checked;
      if (app.collection?.time_map) show(app.tier.name, app.collection.id, app.count, app.pb ? null : 0);
    });
    on('layout', 'change', event => layout(event.target.value));
    on('heat', 'change', event => { app.heatMode = event.target.value; app.heatDrawnAt = -1; render(); });
    for (const name of Object.keys(app.show)) on(`show-${name}`, 'change', event => { app.show[name] = event.target.checked; app.stale = true; render(); });
    on('follow', 'change', event => app.set && follow(event.target.value));
    on('episode', 'change', event => select(Number(event.target.value)));
    for (const speed of app.o.speeds ?? SPEEDS) $('speed').add(new Option(`${fmt(speed)} / s`, speed));
    $('speed').value = String(app.o.speed);
    if (app.page) {
      const keys = event => {
        if (!app?.pb || event.target.closest('input, select, button')) return;
        if (event.key === 'ArrowRight') nudge(1);
        else if (event.key === 'ArrowLeft') nudge(-1);
        else if (event.key === ' ') { event.preventDefault(); play(); }
      };
      document.addEventListener('keydown', keys);
      app.cleanup.push(() => document.removeEventListener('keydown', keys));
    }
    const redraw = () => { if (app?.panels.length) floorInks(); if (app?.pb && app.timeline) { drawChart(); drawReward(); render(); } }, scheme = window.matchMedia('(prefers-color-scheme: dark)');
    scheme.addEventListener('change', redraw);
    const theme = new MutationObserver(redraw);
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    let width = 0, rewardSize = '';
    const resize = new ResizeObserver(() => {
      if (!app.pb || !app.timeline) return;
      const now = $('occupancy').clientWidth, reward = $('reward'), size = reward ? `${reward.clientWidth}x${reward.clientHeight}` : '';
      if (now !== width) {
        width = now;
        drawChart();
        drawPlayhead();
      }
      if (size !== rewardSize) {
        rewardSize = size;
        drawReward();
        drawRewardCursor();
      }
    });
    resize.observe($('occupancy'));
    if ($('reward')) resize.observe($('reward'));
    // A move to a screen of another pixel ratio redraws the plots' backing
    // stores at the new ratio and refits the figure.
    let ratio = null;
    const rescale = () => {
      ratio?.removeEventListener('change', rescale);
      if (ratio) { redraw(); fitStage(); }
      ratio = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      ratio.addEventListener('change', rescale);
    };
    rescale();
    app.cleanup.push(() => scheme.removeEventListener('change', redraw), () => theme.disconnect(), () => resize.disconnect(), () => ratio?.removeEventListener('change', rescale));
  }

  // The Follow layout's boards, three (two in an embed with
  // data-follow-boards="2"; app.followBoards): a window of that many floors
  // that keeps the followed run in view and moves only when it must, as little
  // as it must (stripWindows): it starts on floor 0, moves on when the run goes
  // below it, the run's floor on the right, so a new deepest floor is always
  // the right board, and moves back when the run climbs above it, the run's
  // floor on the left, keeping as much of the deeper floors as it can. The
  // boards sit in slots of a strip clipped to the window; during playback the strip
  // slides like a conveyor, the boards drawing as they move, a move of n
  // boards in one slide of up to 800 ms; moves queue and run back to back. A
  // seek, a layout change or reduced motion places the strip at once.
  const STRIP_MS = [0, 400, 600, 800];
  function updateStrip(animate) {
    if (app.layout !== 'follow') return;
    const e = app.selected, target = e >= 0 && app.selectedWindow ? app.selectedWindow[app.pb.decision[e]] : 0, strip = app.strip;
    if (!animate || strip.at < 0 || REDUCED_MOTION.matches) {
      if (target === strip.at && !strip.moving) return;
      for (const animation of strip.moving ?? []) animation.cancel();
      Object.assign(strip, { at: target, queue: [], moving: null });
      placeStrip(target);
      return;
    }
    if (target !== (strip.queue.at(-1) ?? strip.at)) strip.queue.push(target);
    if (!strip.moving) nextMove();
  }

  // The first floor of a window of `boards` floors at each state of a run
  // whose states (floor, row, col, facing) are given: updateStrip's rule.
  function stripWindows(states, decisions, boards) {
    const out = new Uint8Array(decisions + 1);
    for (let s = 0, first = 0; s <= decisions; s++) {
      const floor = states[4 * s];
      if (floor > first + boards - 1) first = floor - boards + 1;
      else if (floor < first) first = floor;
      out[s] = first;
    }
    return out;
  }

  // Run the next queued move of the strip: every board from the old window
  // to the new one slides together, the ones leaving out of the clip.
  function nextMove() {
    const strip = app.strip, to = strip.queue.shift();
    if (to === undefined) { strip.moving = null; return; }
    const from = strip.at, low = Math.min(from, to), high = Math.max(from, to), slide = (f, first) => `translateX(${(f - first) * 100}%)`;
    const duration = STRIP_MS[Math.min(3, high - low)];
    strip.at = to;
    strip.moving = [];
    for (let f = low; f < high + app.followBoards; f++) {
      const article = app.panels[f].article;
      article.classList.add('in-strip');
      article.style.transform = slide(f, to);
      strip.moving.push(article.animate([{ transform: slide(f, from) }, { transform: slide(f, to) }], { duration, easing: 'ease-in-out' }));
    }
    strip.moving[0].finished.then(() => {
      if (app?.strip !== strip || strip.at !== to) return;
      placeStrip(to);
      nextMove();
    }, () => {});
  }

  // Show the window's floors from first on in the strip's slots, at rest.
  function placeStrip(first) {
    app.panels.forEach((panel, f) => {
      const shown = f >= first && f < first + app.followBoards;
      panel.article.classList.toggle('in-strip', shown);
      panel.article.style.transform = shown ? `translateX(${(f - first) * 100}%)` : '';
    });
  }

  // The Follow strip's board count (updateStrip): a change recomputes the
  // followed run's windows and sets the layout again, the strip at rest.
  function setFollowBoards(n) {
    app.followBoards = n;
    if (app.selectedStates) app.selectedWindow = stripWindows(app.selectedStates, app.set.decisions[app.selected], n);
    layout(app.layout);
  }

  // Grid; all nine floors in one row across the window; two rows, floors 0-4
  // over 5-8 and the win tally, ten equal squares five to a row; or Follow,
  // three boards (or two) on the followed run (updateStrip). The page remembers the
  // choice in this browser when storage allows. One row holds on phones too,
  // with boards about 43 px wide at 390 px: small, but the whole run at a
  // glance; two rows keep about 78 px boards there, Follow about 130 px.
  function layout(value) {
    const select = $('layout');
    if (select) select.value = value;
    app.layout = value;
    app.root.classList.toggle('one-row', value === 'row');
    app.root.classList.toggle('two-rows', value === 'two-rows');
    app.root.classList.toggle('follow-three', value === 'follow');
    app.root.style.setProperty('--follow-boards', String(app.followBoards));
    for (const animation of app.strip.moving ?? []) animation.cancel();
    app.strip = { at: -1, queue: [], moving: null };
    for (const panel of app.panels) panel.article.style.transform = '';
    if (app.tally) app.tally.drawn = '';
    fitStage();
    if (app.pb && app.timeline) render();
    if (app.page) try { localStorage.setItem('craftax-ghosts-layout', value); } catch { /* storage blocked: the choice lasts this visit */ }
  }

  // An embed's figure fitted to its column (data-fit="column"), the content
  // box of the element the embed sits in: the agent's view centred in a row
  // above the Follow strip, every size from the column's width alone, so a
  // change of the window's height alone changes nothing.
  // - Two boards (data-follow-boards="2"): the view and both boards are P, half
  //   the column, tall, the boards square and edge to edge, so the figure is
  //   2P wide (the view, about 1.8 P, is narrower), centred in the column,
  //   with the controls, the timeline and the reward plot as wide.
  // - Three boards: they span the column, each side H a third of it, and the
  //   view is 1.2 H tall, less only if the width asks.
  // The win count over the right board scales with its side, and drops its
  // word on small boards.
  const AGENT_ASPECT = 260 / 144, AGENT_SCALE = 1.2, STACK_GAP = 6;
  function fitStage() {
    const stage = $('stage'), floors = $('floors'), agent = $('agent'), host = app.root.parentElement;
    if (!stage || !host || app.o.fit !== 'column') return;
    const box = getComputedStyle(host), column = host.clientWidth - parseFloat(box.paddingLeft) - parseFloat(box.paddingRight);
    const follow = app.layout === 'follow', withAgent = agent && !agent.hidden, two = follow && app.followBoards === 2;
    if (!(column > 0)) return;
    const board = two ? Math.floor(column / 2) : column / 3;
    const view = two ? board : Math.floor(Math.min(AGENT_SCALE * board, column / AGENT_ASPECT)), width = two ? 2 * board : column;
    Object.assign(app.root.style, { maxWidth: `${width}px`, marginInline: 'auto' });
    stage.classList.add('stacked');
    floors.style.width = follow ? `${Math.floor(app.followBoards * board)}px` : '';
    // The view's cell is exactly as wide as the panel drawn at its height.
    if (withAgent) Object.assign(agent.style, { width: `${view * AGENT_ASPECT}px`, height: `${view}px` });
    app.fit = { column, width, board: Math.floor(board), view };
    app.agentHeight = view;
    const side = follow ? Math.floor(board) : floors.clientWidth / 9;
    floors.style.setProperty('--badge-font', `${Math.max(7, Math.min(18, 0.055 * side)).toFixed(1)}px`);
    floors.classList.toggle('badge-short', side < 220);
    app.policyView?.fit?.();
  }

  // Mount the viewer on `root`, replacing any viewer already mounted.
  // Options: mode ('page' or 'embed'), dataBase (URL prefix of the data, ''
  // for beside the page), transport ('b64' or 'raw'), spritesUrl, tier, set,
  // count, follow, layout, skipQuiet, speed, loop (start again at the end,
  // default true), winsFollowSeed (the sampling seed
  // a time-mapped set's pinned run must have), onDecision(d, {playing}) with
  // the followed run's decision after every drawn step and onReady(viewer)
  // once a set has loaded.
  function mount(root, options) {
    unmount();
    const o = { mode: 'page', dataBase: '', transport: 'b64', follow: 'median', layout: 'grid', skipQuiet: true, speed: 300, loop: true, ...options };
    generation++;
    app = window.ghosts = {
      root, o, page: o.mode === 'page', manifest: null, world: null, tier: null, collection: null, count: 0, set: null, pre: null, pb: null, stats: null, timeline: null,
      playing: false, held: false, last: 0, carry: 0, u: 0, pendingSeek: -1, stale: true, heatDrawnAt: -1,
      heatMode: 'movement', show: { trails: true, creatures: true, changes: true, deaths: true, fog: true, light: true }, shade: null, shadeScratch: null, skipQuiet: o.skipQuiet,
      selected: -1, selectedStates: null, selectedWindow: null, followBoards: o.followBoards ?? 3, windows: new Map(), panels: [], atlas: null, chart: null, frameTimes: [], cleanup: [],
      reward: null, rewardImage: null, rewardY: null, winSteps: new Int32Array(0), winOrder: new Int32Array(0), won: 0, tally: null, layout: o.layout, policy: null, strip: { at: -1, queue: [], moving: null }, stripAnimate: false, showSleep: false,
      sprites: Object.assign(new Image(), { crossOrigin: 'anonymous', src: o.spritesUrl ?? `${o.dataBase}sprites.png` }),
      load: show, goTo, play, stop, setsOf, follow, select, unmount, setFollowBoards,
    };
    wire();
    layout(o.layout);
    return app;
  }

  function unmount() {
    if (!app) return;
    stop();
    generation++;
    for (const undo of app.cleanup) undo();
    app = null;
  }

  // Sprites, the world and the manifest: what every set needs.
  async function boot() {
    await app.sprites.decode();
    app.atlas = buildAtlas();
    app.heatCanvas = Object.assign(document.createElement('canvas'), { width: MAP, height: FLOORS * MAP });
    app.heatImage = new ImageData(MAP, FLOORS * MAP);
    buildPanels();
    app.manifest = await loadJson('manifest.json');
    if (!FORMATS.includes(app.manifest.format)) throw Error(`manifest.json is ${app.manifest.format}, not one of ${FORMATS.join(', ')}.`);
    app.world = D.parseWorld(await loadFile('world.bin.gz'));
    for (let f = 0; f < FLOORS; f++) drawBase(f);
  }

  // The full page: every tier, set and count, opening on the last tier listed,
  // the strongest, at 250 runs where it has them.
  async function page(root) {
    let saved = 'grid';
    try { saved = localStorage.getItem('craftax-ghosts-layout') ?? 'grid'; } catch { /* storage blocked */ }
    const viewer = mount(root, { mode: 'page', layout: D.LAYOUTS.includes(saved) ? saved : 'grid', winsFollowSeed: root.dataset.followSeedWins ?? null });
    $('ramp-movement').style.background = gradient(MOVE_STOPS);
    $('ramp-interactions').style.background = gradient(INTERACT_STOPS);
    $('ramp-opacity').style.background = `linear-gradient(90deg, rgba(255,255,255,${PLAYER_OPACITY[0]}), #fff)`;
    try {
      await boot();
      buildPickers();
      app.policy = await loadPolicy();
      const tier = app.manifest.tiers.at(-1), counts = setsOf(tier)[0].counts;
      await show(tier.name, setsOf(tier)[0].id, counts.includes(250) ? 250 : counts.at(-1));
    } catch (error) {
      fail(error);
    }
    return viewer;
  }

  // The policy view's bundle beside the page, if pack.mjs shipped one:
  // policy-view.json names the run (its sampling seed) and how many decisions
  // its frames hold. The panel shows when that run is followed.
  async function loadPolicy() {
    const panel = $('policy');
    if (!panel || !window.CraftaxPolicyView) return null;
    const response = await fetch(`${app.o.dataBase}policy-view.json`);
    if (!response.ok) return null;
    return { ...(await response.json()), view: null, panel };
  }

  // The policy view's first frame with the necromancer beaten: a won run's
  // last, as a run ends at its first DEFEAT_NECROMANCER (FORMAT.md); null for
  // any other run.
  const beatenFrom = (set, e) => (D.OUTCOMES[set.outcome[e]] === 'win' ? set.decisions[e] : null);

  // Show the policy view while the run it was made from is followed, in step
  // with the ghosts: it draws the followed run's decision.
  async function showPolicy() {
    const policy = app.policy, e = app.selected, wrap = $('policy-panel');
    if (!policy || !wrap) return;
    const on = e >= 0 && app.set.seeds[e] === policy.samplingSeed;
    wrap.hidden = !on;
    if (!on) return;
    if (app.set.decisions[e] !== policy.decisions) throw Error(`the policy view's run has ${policy.decisions} decisions, but the followed run of its seed has ${app.set.decisions[e]}.`);
    text('policy-caption', `What the followed run's agent saw: run ${fmt(e)}, sampling seed ${policy.samplingSeed}, every one of its ${fmt(policy.decisions)} decisions.`);
    if (!policy.view) {
      const file = name => `${app.o.dataBase}${name}${app.o.transport === 'b64' ? '.b64.txt' : ''}`;
      // A bundle with sleep frames shows the followed run's sleeps as they played.
      const sleep = policy.sleepFrames ? { sleepUrl: file('policy-sleep.bin.gz'), sleeps: policy.sleeps } : {};
      policy.view = await window.CraftaxPolicyView.mount(policy.panel, { bundleUrl: file('policy-view.bin.gz'), spritesUrl: `${app.o.dataBase}sprites.png`, beatenFrom: beatenFrom(app.set, e), ...sleep });
      if (policy.view.decisions !== policy.decisions) throw Error(`the policy view holds ${policy.view.decisions} decisions, not the ${policy.decisions} its manifest says.`);
    }
    if (app.pb) policy.view.show(app.policyShown = followedDecision(), followedSample());
  }

  // The structure an embed's markup needs, injected once; the host page's
  // stylesheet sets its width and its tokens (--surface, --ink, --accent,
  // --floor-0..8, --death). Under 600 px the floors stack 3 x 3: nine in a row
  // would be about 43 px each, under one pixel per tile.
  const EMBED_CSS = `
.ghost-embed { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
.ghost-embed > *, .ghost-embed .stage > * { min-width: 0; max-width: 100%; }
.ghost-embed canvas { max-width: 100%; }
.ghost-embed [data-g="occupancy"], .ghost-embed [data-g="reward"] { box-sizing: border-box; }
.ghost-embed .stage { display: flex; justify-content: center; align-items: flex-start; gap: 0; }
.ghost-embed .stage.stacked { flex-direction: column; align-items: center; gap: ${STACK_GAP}px; }
.ghost-embed .stage > .floors { flex: 0 0 auto; width: 100%; }
.ghost-embed .agent { flex: 0 0 auto; overflow: hidden; background: #070d11; }
.ghost-embed .agent[hidden] { display: none; }
.ghost-embed .agent > [data-ghost-policy], .ghost-embed .agent .cpv { width: 100%; height: 100%; }
.ghost-embed .agent .cpv { margin: 0; padding: 0; background: none; border: 0; }
.ghost-embed .agent .cpv-caption { display: none; }
.ghost-embed .floors { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 0; }
.ghost-embed.two-rows .floors { grid-template-columns: repeat(5, minmax(0, 1fr)); }
${FOLLOW_STRIP_CSS('.ghost-embed')}
.ghost-embed .floor { position: relative; min-width: 0; }
.ghost-embed .floor.tally { display: none; }
.ghost-embed.two-rows .floor.tally { display: block; }
.ghost-embed .floor header { position: absolute; z-index: 1; inset: 0 0 auto; padding: 1px 4px; background: color-mix(in srgb, var(--floor-color, #05090c) 60%, transparent); pointer-events: none; }
.ghost-embed .floor h2 { margin: 0; color: var(--floor-ink, #fff); text-shadow: 0 0 2px var(--floor-halo, #000), 0 0 1px var(--floor-halo, #000), 0 0 1px var(--floor-halo, #000); font: 600 11px/1.3 var(--font-mono, ui-monospace, monospace); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0; }
.ghost-embed .floor h2 span { margin-right: 4px; opacity: .8; }
.ghost-embed .floor.followed::after { content: ''; position: absolute; z-index: 2; inset: 0; border: 2px solid rgb(61, 255, 127); pointer-events: none; }
.ghost-embed .floor-stats, .ghost-embed .enlarge { display: none; }
.ghost-embed .board { position: relative; aspect-ratio: 1; background: #070d11; }
.ghost-embed .board canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; }
.ghost-embed .board canvas.tally-marks { image-rendering: auto; }

.ghost-embed [data-g="reward"] { display: block; width: 100%; height: 120px; border: 1px solid var(--rule, transparent); border-radius: 4px; cursor: crosshair; touch-action: none; }
.ghost-embed .reward-key { margin: 0; font: 11px/1.4 var(--font-mono, ui-monospace, monospace); color: var(--ink-3, var(--ink)); }
.ghost-embed .reward-key i { display: inline-block; width: 14px; height: 3px; margin: 0 4px 0 2px; vertical-align: middle; border-radius: 1px; }
.ghost-embed .deck { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; font: 12px/1.3 var(--font-mono, ui-monospace, monospace); color: var(--ink-3, var(--ink)); }
.ghost-embed .deck .transport { display: contents; }
.ghost-embed .deck button, .ghost-embed .deck select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--rule-strong, currentColor); border-radius: 6px; padding: 4px 10px; }
.ghost-embed .deck button { min-width: 5.5em; cursor: pointer; }
.ghost-embed [data-g="position"] { margin-left: auto; font-variant-numeric: tabular-nums; }
.ghost-embed [data-g="status"]:empty, .ghost-embed [data-g="status"][data-done] { display: none; }
.ghost-embed [data-g="occupancy"] { display: block; width: 100%; height: 64px; border: 1px solid var(--rule, transparent); border-radius: 4px; cursor: crosshair; touch-action: none; }
.ghost-embed .seek-wrap { position: relative; }
.ghost-embed [data-g="seek"] { display: block; width: 100%; margin: 0; accent-color: var(--accent); }
${FOLLOW_WIN_CSS('.ghost-embed ')}
${CONTROLS_TOP_CSS()}
@media (max-width: 600px) { .ghost-embed:not(.two-rows):not(.follow-three) .floors { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
`;

  // The Follow layout's strip, under the host's `scope`: --follow-boards
  // slots clipped to the width, each board placed in its slot by a transform
  // (updateStrip), and the win count fixed over the right slot.
  function FOLLOW_STRIP_CSS(scope) {
    return `${scope} .won-badge { display: none; }
${scope}.follow-three .floors { display: block; position: relative; aspect-ratio: var(--follow-boards, 3) / 1; overflow: hidden; }
${scope}.follow-three .floor { position: absolute; top: 0; left: 0; width: calc(100% / var(--follow-boards, 3)); }
${scope}.follow-three .floor:not(.in-strip), ${scope}.follow-three .floor.tally { display: none; }
${scope}.follow-three .won-badge { display: block; position: absolute; z-index: 2; top: 18px; right: 4px; padding: .15em .45em; border-radius: .35em; background: rgba(5, 9, 12, .78); color: #7dff6e; font: 600 var(--badge-font, 13px)/1.25 ui-monospace, monospace; white-space: nowrap; pointer-events: none; }
${scope}.follow-three .won-badge b { font-weight: 600; }
${scope}.follow-three .floors.badge-short .won-word { display: none; }
${scope}.follow-three .floors.badge-short .won-badge { top: 17px; right: 2px; padding: .1em .3em; }`;
  }

  // data-controls="top": the play button and speed centred as one group in
  // the post's accent, the play button a square icon (a darker accent with
  // white in the light theme, the accent with near-black in the dark, both
  // above 4.5:1), the readout to the right; in a figure under 700 px wide
  // the readout centred on its own line below. data-scrubber="plot": the timeline plot half as
  // tall, dragged or keyed, its hit area reaching past it.
  function CONTROLS_TOP_CSS() {
    return `
.ghost-embed.controls-top { --ctl-bg: color-mix(in srgb, var(--accent, #1d8a73) 80%, #000); --ctl-hover: color-mix(in srgb, var(--accent, #1d8a73) 68%, #000); --ctl-ink: #fff; }
:root[data-theme="dark"] .ghost-embed.controls-top { --ctl-bg: var(--accent, #3fb89c); --ctl-hover: color-mix(in srgb, var(--accent, #3fb89c) 82%, #fff); --ctl-ink: #0a0c0e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .ghost-embed.controls-top { --ctl-bg: var(--accent, #3fb89c); --ctl-hover: color-mix(in srgb, var(--accent, #3fb89c) 82%, #fff); --ctl-ink: #0a0c0e; } }
.ghost-embed.controls-top .deck { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 6px 12px; }
.ghost-embed.controls-top .deck .transport { grid-column: 2; display: flex; align-items: center; gap: 10px; }
.ghost-embed.controls-top [data-g="position"] { grid-column: 3; justify-self: end; margin-left: 0; text-align: right; }
.ghost-embed.controls-top .deck button.icon-play { display: inline-grid; place-items: center; min-width: 0; width: 36px; height: 36px; padding: 0; border: 0; border-radius: 8px; background: var(--ctl-bg); color: var(--ctl-ink); }
.ghost-embed.controls-top .deck button.icon-play:hover { background: var(--ctl-hover); }
.ghost-embed.controls-top .deck button.icon-play:focus-visible, .ghost-embed.controls-top .deck select:focus-visible, .ghost-embed .scrub:focus-visible { outline: 2px solid var(--ctl-bg, var(--accent)); outline-offset: 2px; }
.ghost-embed.controls-top .deck label { display: inline-flex; align-items: center; gap: 6px; color: var(--ink); }
.ghost-embed.controls-top .deck select { border: 1.5px solid var(--ctl-bg); background: color-mix(in srgb, var(--ctl-bg) 10%, var(--surface, #fff)); color: var(--ink); padding: 6px 10px; }
.ghost-embed.controls-top .deck select:hover { border-color: var(--ctl-hover); }
.ghost-embed.controls-top { container-type: inline-size; }
@container (max-width: 700px) {
  .ghost-embed.controls-top .deck { grid-template-columns: 1fr; justify-items: center; }
  .ghost-embed.controls-top .deck .transport, .ghost-embed.controls-top [data-g="position"] { grid-column: 1; justify-self: center; text-align: center; }
  .ghost-embed.controls-top .deck button.icon-play { width: 40px; height: 40px; }
}
.ghost-embed .scrub { position: relative; z-index: 1; padding: 4px 0 12px; margin: -4px 0 -12px; cursor: ew-resize; touch-action: pan-y; border-radius: 4px; }
.ghost-embed .scrub [data-g="occupancy"] { height: 32px; cursor: inherit; pointer-events: none; }`;
  }

  // The followed run's win on the seek bar: a cased green check over the
  // track where the thumb will stand when it wins.
  function FOLLOW_WIN_CSS(scope) {
    const svg = "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M2.5 8.5l3.5 4L13.5 3' fill='none' stroke='%23000' stroke-width='5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M2.5 8.5l3.5 4L13.5 3' fill='none' stroke='%233dff7f' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E";
    return `${scope}.follow-win { position: absolute; top: -8px; left: calc(8px + (100% - 16px) * var(--at, 0)); width: 15px; height: 15px; transform: translateX(-50%); pointer-events: none; background: url("data:image/svg+xml,${svg}") center / contain no-repeat; }
${scope}.follow-win[hidden] { display: none; }`;
  }

  const REWARD_KEY = '<p class="reward-key">Reward over the timeline: each run in grey <i style="background:var(--ink);opacity:.5"></i>, the highlighted run in green <i style="background:rgb(61,255,127)"></i>; points on the left, % of all 226 on the right.</p>';

  // The controls: the transport, then the timeline, scrubbed with its slider
  // or, with data-scrubber="plot", by the plot itself; with data-controls="top"
  // the play button is an icon.
  const embedControls = o => `
<div class="deck"><div class="transport">${o.controls === 'top'
    ? `<button type="button" data-g="restart" class="icon-play" aria-label="Restart" title="Restart">${RESTART_ICON}</button><button type="button" data-g="play" class="icon-play" aria-label="Play" title="Play">${PLAY_ICON}</button>`
    : '<button type="button" data-g="play">Play</button>'}<label>Speed <select data-g="speed"></select></label></div><span data-g="position"></span></div>
${o.scrubber === 'plot'
    ? '<div class="scrub" data-g="scrub" role="slider" tabindex="0" aria-label="Timeline" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"><canvas data-g="occupancy" aria-hidden="true"></canvas></div>'
    : `<canvas data-g="occupancy" aria-label="Runs on each floor over time and the runs that have won, with ticks where runs win and die; drag to seek"></canvas>
<div class="seek-wrap"><input data-g="seek" type="range" min="0" max="0" value="0" aria-label="Seek to decision"><span class="follow-win" data-g="follow-win" hidden></span></div>`}`;
  const EMBED_STAGE = '<div class="stage" data-g="stage"><div class="agent" data-g="agent" hidden></div><div class="floors" data-g="floors"></div></div>';
  const EMBED_REWARD = `<canvas data-g="reward" aria-label="Each run's reward over time and the highlighted run's; drag to seek"></canvas>
${REWARD_KEY}`;

  // An embed's markup: the boards, the reward plot, then the controls; or,
  // with data-controls="top", the controls first and the reward plot last.
  const embedHtml = o => [...(o.controls === 'top' ? [embedControls(o), EMBED_STAGE] : [EMBED_STAGE]), EMBED_REWARD, ...(o.controls === 'top' ? [] : [embedControls(o)]), '<p data-g="status" role="status"></p>'].join('\n');

  // An embed: fixed runs that load when the figure comes within a screen of
  // the viewport and play while any of its stage (the policy view and the
  // boards) shows, unless the reader prefers reduced motion or has paused or
  // sought.
  function embed(container, options) {
    if (!document.getElementById('ghost-embed-style')) document.head.append(Object.assign(document.createElement('style'), { id: 'ghost-embed-style', textContent: EMBED_CSS }));
    container.classList.add('ghost-embed');
    container.classList.toggle('controls-top', options.controls === 'top');
    container.classList.toggle('plot-scrubber', options.scrubber === 'plot');
    container.innerHTML = embedHtml(options);
    const viewer = mount(container, { ...options, mode: 'embed' });
    // A fitted figure sets its policy view in the stage, beside or above the boards.
    if (options.fit === 'column' && options.panel) {
      $('agent').append(options.panel);
      $('agent').hidden = false;
    }
    if (options.fit === 'column') {
      // One fit per frame whenever the figure's box or the embed's changes;
      // every size comes from the column's width at that moment.
      let pending = 0;
      const refit = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; if (viewer === app) fitStage(); }); };
      const watch = new ResizeObserver(refit);
      watch.observe(container);
      const figure = container.closest('figure');
      if (figure) watch.observe(figure);
      viewer.cleanup.push(() => watch.disconnect(), () => cancelAnimationFrame(pending));
      fitStage();
    }
    // Autoplay starts and resumes playback whenever the stage shows, unless the
    // reader holds it (app.held: they paused or sought, until they press play
    // again; it lasts the page load, never stored); leaving view pauses it
    // without a hold, so coming back plays again.
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const autoplay = () => { if (viewer === app && app.pb && app.visible && !app.held && !app.playing && !reduced.matches) play(); };
    const near = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      near.disconnect();
      load(viewer, autoplay);
    }, { rootMargin: '100% 0px' });
    const seen = new IntersectionObserver(entries => {
      viewer.visible = entries.at(-1).isIntersecting;
      if (viewer !== app) return;
      if (viewer.visible) autoplay();
      else if (app.playing) stop();
    }, { threshold: 0 });
    near.observe(container);
    // The stage (the policy view, when set there, and the boards), not the
    // charts around it, is what must show; any of it will do, the policy
    // view alone too, as a reader scrolling down sees it first.
    seen.observe(container.querySelector('[data-g="stage"]'));
    viewer.cleanup.push(() => near.disconnect(), () => seen.disconnect());
    setStatus('');
    return viewer;
  }

  async function load(viewer, then) {
    try {
      setStatus('Loading 1,000 runs…');
      await boot();
      if (viewer !== app) return;
      await show(app.o.tier, app.o.set, app.o.count, 0);
      if (viewer !== app || !app.pb) return;
      $('status').dataset.done = '';
      then();
    } catch (error) {
      if (viewer === app) fail(error);
    }
  }

  // The embed a page declares in markup: [data-ghost-embed] with its options as
  // data attributes (decode.js embedOptions), and beside it in the same figure
  // an optional [data-ghost-policy] panel, where CraftaxPolicyView, if loaded,
  // shows what the followed run's agent saw, in step with the ghosts. With
  // data-follow-seed set, the followed run must be the episode of that sampling
  // seed, the one the panel was made from; a mismatch stops the figure.
  function autoMount() {
    if (app && !app.root.isConnected) unmount();
    const root = document.querySelector('[data-ghost-embed]');
    if (!root || app?.root === root) return;
    let options;
    try {
      options = D.embedOptions(root.dataset);
    } catch (error) {
      root.textContent = `This figure is misconfigured: ${error.message}`;
      console.error(error);
      return;
    }
    if (!options.dataBase) {
      root.textContent = 'This interactive figure loads its data once it is published.';
      return;
    }
    const panel = root.closest('figure')?.querySelector('[data-ghost-policy]');
    let view = null, shown = 0;
    embed(root, {
      ...options,
      panel,
      onDecision: (t, { sample = 0 } = {}) => { shown = t; if (view) { view.show(t, sample); app.policyShown = t; } },
      onReady: async viewer => {
        const e = viewer.selected, seed = viewer.set.seeds[e];
        if (options.followSeed && seed !== options.followSeed) return fail(Error(`the followed run is episode ${e} with sampling seed ${seed}, but its panel shows the run of seed ${options.followSeed}.`));
        if (!panel || !window.CraftaxPolicyView) return;
        const resolve = url => (/^[a-z]+:|^\//i.test(url) ? url : options.dataBase + url);
        // data-layout="row" or "side" sets the panel beside itself at the
        // boards' height, or in a fitted figure the height fitStage chose.
        const fitted = options.fit === 'column';
        const board = () => (fitted ? app?.agentHeight : null) ?? app?.panels.find(p => p.article.getBoundingClientRect().width)?.article.querySelector('.board').clientHeight ?? 0;
        const mounted = await window.CraftaxPolicyView.mount(panel, {
          bundleUrl: resolve(panel.dataset.bundle ?? ''), spritesUrl: `${options.dataBase}sprites.png`, decisions: viewer.set.decisions[e], beatenFrom: beatenFrom(viewer.set, e),
          layout: panel.dataset.layout ?? (fitted ? 'side' : 'column'), height: board, ...(fitted ? { wrapBelow: 0 } : {}),
        });
        if (viewer !== app) return;
        app.policyView = mounted;
        if (mounted.fit) {
          const refit = new ResizeObserver(() => mounted.fit());
          refit.observe(root.querySelector('[data-g="floors"]'));
          viewer.cleanup.push(() => refit.disconnect());
        }
        if (mounted.decisions !== viewer.set.decisions[e]) return fail(Error(`the panel's run has ${mounted.decisions} decisions, but the followed run has ${viewer.set.decisions[e]}.`));
        view = mounted;
        view.show(shown);
        app.policyShown = shown;
      },
    });
  }

  // drawTurned is here for render_check, which checks its pixels.
  window.GhostViewer = { mount, unmount, page, drawTurned };
  window.GhostEmbed = { mount: embed, autoMount };
  const start = () => {
    const pageRoot = document.querySelector('[data-ghost-page]');
    if (pageRoot && app?.root !== pageRoot) page(pageRoot);
    else autoMount();
  };
  document.addEventListener('rekursiv:route', start);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
