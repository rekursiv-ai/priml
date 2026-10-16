import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const here = path.dirname(fileURLToPath(import.meta.url));
const recordBytes = 7858;

function argumentsFor(command, values) {
  const flags = new Map();
  const positional = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i].startsWith('--')) {
      if (!values[i + 1] || values[i + 1].startsWith('--')) {
        throw Error(`Missing value for ${values[i]}`);
      }
      flags.set(values[i++], values[i]);
    } else {
      positional.push(values[i]);
    }
  }
  if (command === 'build' && (!flags.has('--output') || !positional.length)) {
    throw Error('build requires --output FILE and at least one game directory');
  }
  if (command === 'panel' && (!flags.has('--output') || positional.length)) {
    throw Error('panel requires --output FILE and nothing else');
  }
  return { flags, positional };
}

function frameSummary(frames) {
  if (frames.length % recordBytes || frames.length < 2 * recordBytes) {
    throw Error(`Invalid frame stream: ${frames.length} bytes`);
  }
  const count = frames.length / recordBytes;
  const last = (count - 1) * recordBytes;
  if (frames.readUInt8(last + 12) !== 255) throw Error('Missing final frame marker');
  return {
    frames: count,
    actions: count - 1,
    tick: frames.readUInt32LE(last + 4),
    score: frames.readUInt16LE(last + 20),
    achievements: frames.readUInt8(last + 832),
  };
}

function verifiedPayload(directory, name, label, rawSha, gzipSha, bytes) {
  const compressed = fs.readFileSync(path.join(directory, `${name}.bin.gz`));
  if (crypto.createHash('sha256').update(compressed).digest('hex') !== gzipSha) {
    throw Error(`${label} payload hash mismatch: ${directory}`);
  }
  const raw = zlib.gunzipSync(compressed);
  if (raw.length !== bytes || crypto.createHash('sha256').update(raw).digest('hex') !== rawSha) {
    throw Error(`${label}/frame mismatch: ${directory}`);
  }
  return compressed.toString('base64');
}

// A model bundle (written by bundle.py, beside this file)
// adds the exact token frames, optional real token frames for comparison,
// and per-decision annotations to the viewer frames.
function modelPayload(directory, manifest, frames) {
  const notes = manifest.annotations;
  if (notes.invalid.length !== frames || notes.frameSource.length !== frames
      || notes.overrides.length !== frames || notes.actionMarks.length !== frames - 1
      || notes.rewardProb.length !== frames - 1 || notes.doneProb.length !== frames - 1) {
    throw Error(`Annotation/frame mismatch: ${directory}`);
  }
  const tokensGzip = verifiedPayload(directory, 'tokens', 'Tokens', manifest.tokensSha256,
    manifest.tokensGzipSha256, frames * 894);
  const referenceGzip = manifest.referenceGzipSha256 === null ? null : verifiedPayload(directory,
    'reference', 'Reference', manifest.referenceSha256, manifest.referenceGzipSha256,
    manifest.referenceFrames * 894);
  return { source: manifest.source, absent: manifest.absent, annotations: notes, tokensGzip, referenceGzip };
}

function build(flags, directories) {
  const defaultGame = flags.get('--default-game') || 'first';
  if (defaultGame !== 'first' && defaultGame !== 'last') {
    throw Error('--default-game must be first or last');
  }
  const games = directories.map(directory => {
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
    const model = manifest.schema === 'craftax-model-game/v1';
    if (!model && manifest.schema !== 'craftax-exact-game/v1') throw Error(`Wrong schema: ${directory}`);
    const compressed = fs.readFileSync(path.join(directory, 'frames.bin.gz'));
    if (crypto.createHash('sha256').update(compressed).digest('hex') !== manifest.gzipSha256) {
      throw Error(`Compressed payload hash mismatch: ${directory}`);
    }
    const frames = zlib.gunzipSync(compressed);
    if (crypto.createHash('sha256').update(frames).digest('hex') !== manifest.framesSha256) {
      throw Error(`Frame payload hash mismatch: ${directory}`);
    }
    const summary = frameSummary(frames);
    for (const key of ['frames', 'actions', 'tick', 'score', 'achievements']) {
      if (summary[key] !== manifest[key]) throw Error(`${key} mismatch: ${directory}`);
    }
    let healthGzip = null;
    if (manifest.healthGzipSha256) {
      healthGzip = fs.readFileSync(path.join(directory, 'health.bin.gz'));
      if (crypto.createHash('sha256').update(healthGzip).digest('hex') !== manifest.healthGzipSha256) {
        throw Error(`Health payload hash mismatch: ${directory}`);
      }
      const health = zlib.gunzipSync(healthGzip);
      if (health.length !== summary.frames * 120 || crypto.createHash('sha256').update(health).digest('hex') !== manifest.healthSha256) {
        throw Error(`Health/frame mismatch: ${directory}`);
      }
    }
    return {
      kind: model ? 'model' : 'exact',
      ...(model ? modelPayload(directory, manifest, summary.frames) : {}),
      title: manifest.title,
      description: manifest.description,
      endLabel: manifest.endLabel,
      provenance: manifest.provenance,
      frames: manifest.frames,
      tick: manifest.tick,
      score: manifest.score,
      meanScore: manifest.meanScore ?? null,
      meanReturnPct: manifest.meanReturnPct ?? null,
      gzip: compressed.toString('base64'),
      healthGzip: healthGzip?.toString('base64') ?? null,
    };
  });
  const sprite = fs.readFileSync(path.join(here, 'sprites.png')).toString('base64');
  const spriteUrl = `data:image/png;base64,${sprite}`;
  let html = fs.readFileSync(path.join(here, 'viewer.html'), 'utf8');
  const escapeHtml = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  const pageTitle = escapeHtml(flags.get('--page-title') || 'Saved Craftax games');
  const pageNote = escapeHtml(flags.get('--page-note') || '');
  const pickerLabel = escapeHtml(flags.get('--picker-label') || 'Game');
  const selected = defaultGame === 'last' ? games.length - 1 : 0;
  const payload = `window.CRAFTAX_GAMES=${JSON.stringify(games).replaceAll('<', '\\u003c')};window.CRAFTAX_DEFAULT_GAME=${selected};`;
  const replacements = [
    ['/* CRAFTAX_GAMES_PAYLOAD */', payload],
    ['<!-- PAGE_TITLE -->Saved Craftax games', pageTitle],
    ['<!-- PAGE_NOTE -->', pageNote],
    ['<label for="game-picker">Game </label>', `<label for="game-picker">${pickerLabel} </label>`],
    ['aria-label="Choose game"', `aria-label="Choose ${pickerLabel}"`],
    ['<script src="hud.js"></script>', `<script>${fs.readFileSync(path.join(here, 'hud.js'), 'utf8')}</script>`],
    ['<script src="model.js"></script>', `<script>${fs.readFileSync(path.join(here, 'model.js'), 'utf8')}</script>`],
    ["url('sprites.png')", `url('${spriteUrl}')`],
    ["sprites.src='sprites.png'", `sprites.src='${spriteUrl}'`],
  ];
  for (const [before, after] of replacements) {
    if (!html.includes(before)) throw Error(`Missing template marker: ${before.slice(0, 70)}`);
    // A function, not the string: replaceAll expands $&, $` and $' in a string,
    // which a bundle's title or an inlined script may hold.
    html = html.replaceAll(before, () => after);
  }
  if (/<(?:script|img)[^>]*\bsrc=|url\('sprites\.png'\)/.test(html)) {
    throw Error('Standalone page retains an external asset');
  }
  const output = path.resolve(flags.get('--output'));
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, html);
  process.stdout.write(`${output}: ${games.length} games, ${Buffer.byteLength(html)} bytes\n`);
}

// The policy-view panel's script: hud.js and policy_view.js in one function, so
// the page that loads it gains CraftaxPolicyView and no other name.
function panel(flags) {
  const sources = ['hud.js', 'policy_view.js'].map(name => fs.readFileSync(path.join(here, name), 'utf8'));
  const script = `(() => {\n${sources.join('\n')}})();\n`;
  const output = path.resolve(flags.get('--output'));
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, script);
  process.stdout.write(`${output}: ${Buffer.byteLength(script)} bytes\n`);
}

const [command, ...values] = process.argv.slice(2);
try {
  const { flags, positional } = argumentsFor(command, values);
  if (command === 'build') build(flags, positional);
  else if (command === 'panel') panel(flags);
  else throw Error('Usage: node games.mjs build --output FILE GAME_DIR... | node games.mjs panel --output FILE');
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
