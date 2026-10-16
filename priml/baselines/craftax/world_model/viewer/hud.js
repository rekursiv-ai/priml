'use strict';
// What the policy saw of one frame (exact.py's layout, read at its byte
// offsets): its 9x11 window and its HUD. viewer.html draws the window on its
// board and the HUD as cards beside it; policy_view.js draws both on one canvas.
const inventoryNames = [
  'Wood', 'Stone', 'Coal', 'Iron', 'Diamond', 'Sapling', 'Pickaxe', 'Sword',
  'Bow', 'Arrows', 'Helmet', 'Chestplate', 'Leggings', 'Boots', 'Torches',
  'Ruby', 'Sapphire', 'Red potion', 'Green potion', 'Blue potion',
  'Pink potion', 'Cyan potion', 'Yellow potion', 'Books',
];
let lastInventory = '';
function inventoryIcon(slot, quantity) {
  if (slot < 5) return [6, 4, 8, 9, 10][slot];
  if (slot === 5) return 77;
  if (slot === 6) return 61 + quantity;
  if (slot === 7) return 65 + quantity;
  if (slot === 8) return 70;
  if (slot === 9) return 51;
  if (slot >= 10 && slot <= 13) return (quantity >= 2 ? 58 : 54) + slot - 10;
  if (slot === 14) return 78;
  if (slot === 15) return 22;
  if (slot === 16) return 21;
  if (slot >= 17 && slot <= 22) return 71 + slot - 17;
  return 79;
}
// An item's sprite in sprites.png, from the State's item byte (game/state.py
// ItemType): the sheet holds the items from id 42 in that order, 42 blank for
// none, 43 a torch, 44 a ladder down, 45 a ladder up, 46 a blocked ladder down.
// The policy's observation stores the item plus one, so its x is item x - 1.
function itemSprite(item) {
  return 42 + item;
}
// The necromancer's block (game/state.py BlockType.NECROMANCER). It stays on
// the map once beaten; the map never holds NECROMANCER_VULNERABLE (36).
const NECROMANCER = 32;
// A projectile's sprite by its type plus one, the observation's channel 6
// (an enemy's) and 7 (the player's) value: original Craftax's textures
// (constants.py), each pointing up, 51 the arrow pointing up; turned to the
// way it flies when its facing is known (projectileSprite), else as the
// single-game viewer has always drawn them, an arrow pointing down (50).
const PROJECTILE_SPRITES = [51, 99, 100, 101, 51, 102, 100, 101];
const UNTURNED_PROJECTILE_SPRITES = [50, 99, 100, 101, 50, 102, 100, 101];
// Draw sheet sprite id in the size x size tile at (row, col), turned to
// facing (a move action: LEFT 1, RIGHT 2, UP 3, DOWN 4; 0 or UP as it is) by
// original Craftax's rule for projectiles (craftax/renderer.py): flipped top
// to bottom when it flies down or right, then transposed, rows for columns,
// when it flies left or right. The transforms compose right to left, so the
// transpose is set first.
function drawTurned(ctx, sprites, id, row, col, size, facing) {
  ctx.save();
  ctx.translate(col * size, row * size);
  if (facing === 1 || facing === 2) ctx.transform(0, 1, 1, 0, 0, 0);
  if (facing === 4 || facing === 2) ctx.transform(1, 0, 0, -1, 0, size);
  ctx.drawImage(sprites, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, 0, 0, size, size);
  ctx.restore();
}
// The window's 99 cells from cells[base] on, 8 channels each (block, item plus
// one, lit, then one creature class per channel), and the player at its centre,
// in square tiles of size pixels; with beaten set, a red X over each
// necromancer in sight. With facings at cells[facings] (one byte per cell, the
// enemy projectile's facing in bits 0-3, the player's in bits 4-7; -1 for
// none), each projectile is turned the way it flies.
function drawWindow(ctx, sprites, cells, base, direction, sleeping, size, beaten = false, facings = -1) {
  const tile = (id, row, col) => ctx.drawImage(sprites, (id % 16) * 16, Math.floor(id / 16) * 16, 16, 16, col * size, row * size, size, size);
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#070d11';
  ctx.fillRect(0, 0, 11 * size, 9 * size);
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 11; c++) {
      const b = base + (r * 11 + c) * 8;
      if (!cells[b + 2]) {
        ctx.fillStyle = '#111719';
        ctx.fillRect(c * size, r * size, size, size);
        continue;
      }
      tile(cells[b], r, c);
      let x = cells[b + 1];
      if (x > 1) tile(itemSprite(x - 1), r, c);
      for (let k = 3; k < 6; k++) {
        x = cells[b + k];
        if (x) tile([80, 88, 91][k - 3] + x - 1, r, c);
      }
      for (let k = 6; k < 8; k++) {
        x = cells[b + k];
        const facing = facings < 0 ? 0 : cells[facings + r * 11 + c] >> 4 * (k - 6) & 15;
        if (x) drawTurned(ctx, sprites, (facing ? PROJECTILE_SPRITES : UNTURNED_PROJECTILE_SPRITES)[x - 1], r, c, size, facing);
      }
      if (beaten && cells[b] === NECROMANCER) crossOut(ctx, r, c, size);
    }
  }
  tile(sleeping ? 41 : [39, 40, 38, 37][Math.max(0, Math.min(3, direction - 1))], 4, 5);
}
// A red X over the tile at (row, col), in the ghost viewer's play-head red: a
// 2-pixel stroke on a 4-pixel dark casing (in sixteenths of the tile), from
// pixel 1 to 14 on both diagonals, so it reads on any block.
function crossOut(ctx, row, col, size) {
  const p = size / 16, x = col * size, y = row * size;
  for (const [colour, width] of [['rgba(8,10,12,.9)', 4], ['#ff2a2a', 2]]) {
    ctx.fillStyle = colour;
    const from = 3 - width / 2;
    for (let i = 0; i <= 10; i++) {
      ctx.fillRect(x + (from + i) * p, y + (from + i) * p, width * p, width * p);
      ctx.fillRect(x + (16 - from - width - i) * p, y + (from + i) * p, width * p, width * p);
    }
  }
}
// The frame at byte base of view: its five meters as label, value, maximum and
// colour.
function hudMeters(view, base) {
  const u8 = offset => view.getUint8(base + offset);
  return [
    ['Health', view.getFloat32(base + 22, true), u8(826), '#42b65a'],
    ['Mana', view.getInt16(base + 30, true), u8(827), '#438ed0'],
    ['Food', u8(829), u8(828), '#d5a43e'],
    ['Drink', u8(830), u8(828), '#4cabc5'],
    ['Energy', u8(831), u8(828), '#ad79d6'],
  ];
}
// What the player of the frame at byte base of view holds, as label, sprite and
// count text: each held item (gear by its tier, uncounted), then each spell.
function hudItems(view, base) {
  const items = [];
  for (let slot = 0; slot < 24; slot++) {
    const count = view.getUint16(base + 833 + slot * 2, true);
    if (!count) continue;
    const gear = slot === 6 || slot === 7 || (slot >= 10 && slot <= 13);
    items.push([inventoryNames[slot] + (gear ? ` · tier ${count}` : ''), inventoryIcon(slot, count), gear ? '' : String(count)]);
  }
  if (view.getUint8(base + 881)) items.push(['Fireball', 100, '']);
  if (view.getUint8(base + 882)) items.push(['Iceball', 101, '']);
  return items;
}
function inventoryItem(label, icon, count) {
  const card = document.createElement('div');
  card.className = 'inv-item';
  const picture = document.createElement('div');
  picture.className = 'item-icon';
  picture.style.backgroundPosition = `-${(icon % 16) * 16}px -${Math.floor(icon / 16) * 16}px`;
  picture.setAttribute('aria-label', label);
  card.title = count ? `${label}: ${count}` : label;
  card.append(picture);
  if (count) {
    const quantity = document.createElement('span');
    quantity.className = 'item-count';
    quantity.textContent = count;
    card.append(quantity);
  }
  const name = document.createElement('div');
  name.className = 'item-label';
  name.textContent = label;
  card.append(name);
  return card;
}
function renderHud(i) {
  const base = at(i, 0);
  $('resources').replaceChildren(...hudMeters(view, base).map(([label, value, max, colour]) => {
    const box = document.createElement('div');
    box.className = 'resource';
    const head = document.createElement('div');
    head.className = 'resource-head';
    const title = document.createElement('span');
    title.textContent = label;
    const count = document.createElement('span');
    count.textContent = `${Number(value).toFixed(label === 'Health' ? 1 : 0)} / ${max}`;
    head.append(title, count);
    const meter = document.createElement('div');
    meter.className = 'meter';
    const fill = document.createElement('div');
    fill.className = 'meter-fill';
    fill.style.background = colour;
    fill.style.width = `${Math.max(0, Math.min(100, 100 * Number(value) / Math.max(1, Number(max))))}%`;
    meter.append(fill);
    box.append(head, meter);
    return box;
  }));
  $('achievements').textContent = `${u8(i, 832)} / 67`;
  const items = hudItems(view, base);
  const signature = JSON.stringify(items);
  if (signature === lastInventory) return;
  lastInventory = signature;
  const cards = items.map(([label, icon, count]) => inventoryItem(label, icon, count));
  if (!cards.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No items yet';
    cards.push(empty);
  }
  $('inventory').replaceChildren(...cards);
}
