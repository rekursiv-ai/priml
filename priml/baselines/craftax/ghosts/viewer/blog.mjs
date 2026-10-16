// The ghost figure in the Craftax blog post: its data, its vendored scripts
// and a local server to preview it with.
//
//   node blog.mjs data --site SITE_DATA --out DATA_DIR [--tier boss --set short --count 1000]
//     Copy one tier's set at one count out of build.py's site data (raw
//     .bin.gz, checked against the manifest's sha256s), the world, sprites.png
//     and a manifest trimmed to them, and print the run it follows: a
//     time-mapped set's pinned run (time_map.unbroken), else the shortest win.
//   node blog.mjs policy --data DATA_DIR --bundle POLICY_DIR
//     Add the policy view's bundle (POLICY_DIR/policy-view.bin.gz, checked
//     against POLICY_DIR/manifest.json) to the data, after checking it is the
//     run the embed follows, by sampling seed.
//   node blog.mjs vendor --post POST_DIR [--policy-view FILE --policy-origin TEXT]
//     Write decode.js, ghosts.js and the policy view (policy_stub.js unless
//     the built one is given, with TEXT naming where it was built from) into
//     the post as *.vendored.js, headed with their source; the post must not
//     edit them.
//   node blog.mjs serve --data DATA_DIR [--port 8737]
//     Serve DATA_DIR with CORS for a local preview of the post. It sends .gz
//     files as application/octet-stream with no Content-Encoding, as the
//     embed's DecompressionStream needs them.
//   node blog.mjs publish --data DATA_DIR
//     Upload DATA_DIR to R2 under craftax/ghosts/<sha256 of its manifest, 24
//     hex>/ with the website's wrangler command (tools/publish-graph-video.cjs
//     putObject, under this prefix; CLOUDFLARE_ACCOUNT_ID names the account),
//     and print each public URL. The edge cache
//     keeps the first response to a key for a year and does not vary it by
//     Origin, so the first request to each key is made here, right after its
//     upload, with Origin: https://rekursiv.ai, and must carry that CORS
//     header and the manifest's bytes; a second request must carry it too,
//     from the cache (HIT) or, for types the edge does not cache (DYNAMIC:
//     .json), from R2 again. Nothing requests a key before its upload: the
//     edge would cache the 404. DATA_DIR/published.json records each key as
//     it is uploaded and verified; a rerun skips the uploaded ones, verifying
//     any not yet verified, so no key is written twice.
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = path.dirname(fileURLToPath(import.meta.url));
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

// The data subset; returns its files and the followed run's episode entry.
export function writeData({ site, out, tier: tierName = 'boss', set: setName = 'short', count = 1000 }) {
  if (fs.existsSync(out) && fs.readdirSync(out).length) throw Error(`${out} is not empty; write the data into a new directory.`);
  const manifest = JSON.parse(fs.readFileSync(path.join(site, 'manifest.json'), 'utf8'));
  const tier = manifest.tiers.find(t => t.name === tierName), set = tier?.sets.find(s => s.name === setName);
  if (!set?.counts.includes(count)) throw Error(`${site} has no ${tierName}/${setName} set with ${count} episodes.`);
  const groups = set.groups.slice(0, set.counts.indexOf(count) + 1), timeline = set.timelines.find(t => t.count === count);
  const names = ['world.bin.gz', ...(timeline ? [timeline.path] : []), ...groups.flatMap(g => [
    ...['episodes.json', 'players.bin.gz', 'events.bin.gz', ...(set.time_map ? ['keeps.bin.gz'] : [])].map(f => `${g.path}/${f}`),
    ...Array.from({ length: g.windows }, (_, j) => `${g.path}/creatures-w${j}.bin.gz`),
  ])];
  const files = {}, sizes = {};
  for (const name of names) {
    const bytes = fs.readFileSync(path.join(site, name));
    if (sha(bytes) !== manifest.files[name] || bytes.length !== manifest.sizes[name]) throw Error(`${name} does not match ${site}/manifest.json.`);
    fs.mkdirSync(path.dirname(path.join(out, name)), { recursive: true });
    fs.writeFileSync(path.join(out, name), bytes);
    files[name] = manifest.files[name];
    sizes[name] = bytes.length;
  }
  const trimmed = { ...manifest, tiers: [{ ...tier, sets: [{ ...set, timelines: timeline ? [timeline] : [] }] }], files, sizes };
  fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(trimmed, null, 1) + '\n');
  fs.copyFileSync(path.join(here, '../../world_model/viewer/sprites.png'), path.join(out, 'sprites.png'));
  const episodes = groups.flatMap(g => JSON.parse(fs.readFileSync(path.join(out, g.path, 'episodes.json'), 'utf8')).episodes);
  return { files: [...names, 'manifest.json', 'sprites.png'], followed: episodes[followedIndex(set, episodes)] ?? null };
}

// The policy-view bundle schemas the panel reads: v2 adds each frame's
// projectile facings.
const POLICY_SCHEMAS = ['craftax-policy-view/v1', 'craftax-policy-view/v2'];

// The policy view's bundle, added to the data once it is the followed run's.
export function addPolicy({ data, bundle }) {
  const policy = JSON.parse(fs.readFileSync(path.join(bundle, 'manifest.json'), 'utf8')), bytes = fs.readFileSync(path.join(bundle, 'policy-view.bin.gz'));
  if (!POLICY_SCHEMAS.includes(policy.schema) || sha(bytes) !== policy.gzipSha256) throw Error(`${bundle} does not hold the policy-view bundle its manifest describes.`);
  const manifest = JSON.parse(fs.readFileSync(path.join(data, 'manifest.json'), 'utf8')), set = manifest.tiers[0].sets[0];
  const episodes = set.groups.flatMap(g => JSON.parse(fs.readFileSync(path.join(data, g.path, 'episodes.json'), 'utf8')).episodes);
  const followed = episodes[followedIndex(set, episodes)];
  if (!followed || followed.sampling_seed !== policy.samplingSeed || followed.decisions !== policy.decisions) {
    throw Error(`The policy view shows the run of sampling seed ${policy.samplingSeed} (${policy.decisions} decisions), but the embed follows ${followed ? `seed ${followed.sampling_seed} (${followed.decisions} decisions)` : 'no run'}.`);
  }
  fs.writeFileSync(path.join(data, 'policy-view.bin.gz'), bytes);
  return followed;
}

// The run the embed follows among a set's episodes.json entries: a
// time-mapped set's pinned run, else the shortest win by decode.js's rule.
function followedIndex(set, episodes) {
  if (set.time_map?.unbroken != null) return set.time_map.unbroken;
  const decode = {};
  vm.runInNewContext(fs.readFileSync(path.join(here, 'decode.js'), 'utf8'), decode);
  const D = decode.GhostDecode;
  return D.shortestWin({ n: episodes.length, outcome: episodes.map(e => D.OUTCOMES.indexOf(e.outcome)), decisions: episodes.map(e => e.decisions) });
}

// Copy the viewer's scripts into the post under a header naming their source.
export function vendor({ post, policyView = path.join(here, 'policy_stub.js'), policyOrigin = null }) {
  const repo = spawnSync('git', ['-C', here, 'rev-parse', '--show-toplevel'], { encoding: 'utf8' }).stdout.trim();
  const commit = spawnSync('git', ['-C', here, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).stdout.trim();
  const written = [];
  for (const [source, name] of [[path.join(here, 'decode.js'), 'ghosts-decode.vendored.js'], [path.join(here, 'ghosts.js'), 'ghosts-viewer.vendored.js'], [policyView, 'craftax-policy-view.vendored.js']]) {
    const relative = path.relative(repo, path.resolve(source));
    const dirty = spawnSync('git', ['-C', repo, 'status', '--porcelain', '--', relative], { encoding: 'utf8' }).stdout.trim();
    const origin = source === policyView && policyOrigin ? policyOrigin : `${relative} at ${commit}${dirty ? ' with uncommitted changes' : ''}`;
    const header = `// Vendored from ${origin}.\n// Do not edit here; regenerate with ${path.relative(repo, path.join(here, 'blog.mjs'))} vendor.\n`;
    fs.writeFileSync(path.join(post, name), header + fs.readFileSync(source, 'utf8'));
    written.push(name);
  }
  return { commit, written };
}

const TYPES = { '.json': 'application/json', '.png': 'image/png', '.gz': 'application/octet-stream' };

export function serve({ data, port = 8737 }) {
  const root = path.resolve(data);
  return http.createServer((request, response) => {
    const file = path.join(root, path.normalize(decodeURIComponent(new URL(request.url, 'http://x').pathname)));
    const headers = { 'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store' };
    if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { response.writeHead(404, headers); response.end(); return; }
    response.writeHead(200, { ...headers, 'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream' });
    response.end(fs.readFileSync(file));
  }).listen(port, '127.0.0.1');
}

const MEDIA = 'https://media.rekursiv.ai/', BUCKET = 'rekursiv-public-assets', ORIGIN = 'https://rekursiv.ai';
const CONTENT = { '.json': 'application/json', '.png': 'image/png', '.gz': 'application/octet-stream' };

export async function publish({ data }) {
  const recordFile = path.join(data, 'published.json');
  const manifestBytes = fs.readFileSync(path.join(data, 'manifest.json')), manifest = JSON.parse(manifestBytes.toString('utf8'));
  const prefix = `craftax/ghosts/${sha(manifestBytes).slice(0, 24)}/`;
  const record = fs.existsSync(recordFile) ? JSON.parse(fs.readFileSync(recordFile, 'utf8')) : { base: `${MEDIA}${prefix}`, files: {} };
  if (record.base !== `${MEDIA}${prefix}`) throw Error(`${recordFile} is for ${record.base}, not ${MEDIA}${prefix}.`);
  const save = () => fs.writeFileSync(recordFile, JSON.stringify(record, null, 1) + '\n');
  const extras = ['policy-view.bin.gz'].filter(name => fs.existsSync(path.join(data, name)));
  for (const name of [...Object.keys(manifest.files), 'manifest.json', 'sprites.png', ...extras]) {
    const file = path.join(data, name), bytes = fs.readFileSync(file), type = CONTENT[path.extname(name)], url = `${MEDIA}${prefix}${name}`;
    if (record.files[name]?.verified) continue;
    if (!record.files[name]) {
      // The website's putObject (tools/publish-graph-video.cjs), under this prefix;
      // wrangler reads the account from CLOUDFLARE_ACCOUNT_ID.
      const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'r2-publisher-'));
      const put = spawnSync('npx', ['--yes', 'wrangler@4.131.1', 'r2', 'object', 'put', `${BUCKET}/${prefix}${name}`, '--remote', '--file', file,
        '--content-type', type, '--cache-control', 'public, max-age=31536000, immutable'], {
        cwd, encoding: 'utf8', timeout: 180000,
      });
      if (put.error || put.status !== 0) throw Error(`R2 upload failed for ${name}: ${put.error?.message ?? put.stderr}`);
      record.files[name] = { url, bytes: bytes.length, sha256: sha(bytes), type, verified: false };
      save();
    }
    const first = await get(url), again = await get(url), row = record.files[name];
    Object.assign(row, { first: first.summary, again: again.summary });
    const problems = [first, again].flatMap((r, i) => [
      r.status !== 200 && `status ${r.status}`, r.headers['access-control-allow-origin'] !== ORIGIN && `allow-origin ${r.headers['access-control-allow-origin']}`,
      r.headers['content-encoding'] && `content-encoding ${r.headers['content-encoding']}`, r.headers['content-type'] !== type && `content-type ${r.headers['content-type']}`,
      (r.body.length !== bytes.length || sha(r.body) !== row.sha256) && `${r.body.length} bytes, sha256 ${sha(r.body)}`,
      i === 1 && !['HIT', 'DYNAMIC'].includes(r.headers['cf-cache-status']) && `cf-cache-status ${r.headers['cf-cache-status']}`,
    ].filter(Boolean).map(p => `${['first', 'second'][i]} request: ${p}`));
    save();
    if (problems.length) throw Error(`${url}: ${problems.join('; ')}`);
    row.verified = true;
    save();
  }
  return { base: record.base, rows: Object.values(record.files) };
}

// GET with the site's Origin, as the post's fetch() makes it.
function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { Origin: ORIGIN } }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => {
        const { headers, statusCode: status } = response, body = Buffer.concat(chunks);
        const summary = { status, allowOrigin: headers['access-control-allow-origin'] ?? null, cache: headers['cf-cache-status'] ?? null, type: headers['content-type'], encoding: headers['content-encoding'] ?? null, bytes: body.length };
        resolve({ status, headers, body, summary });
      });
    }).on('error', reject);
  });
}

async function main() {
  const [command, ...rest] = process.argv.slice(2), flags = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i].startsWith('--') || rest[i + 1] === undefined) throw Error(`Expected --flag value pairs, not ${rest.slice(i).join(' ')}.`);
    flags[rest[i].slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = rest[i + 1];
  }
  if (command === 'data' && flags.site && flags.out) {
    const { files, followed: win } = writeData({ ...flags, count: Number(flags.count ?? 1000) });
    const bytes = files.reduce((sum, name) => sum + fs.statSync(path.join(flags.out, name)).size, 0);
    process.stdout.write(`${flags.out}: ${files.length} files, ${bytes} bytes\n`);
    process.stdout.write(win ? `follows: index ${win.index}, source ${win.source}, ordinal ${win.ordinal}, sampling seed ${win.sampling_seed}, ${win.decisions} decisions\n` : 'no episode won\n');
  } else if (command === 'policy' && flags.data && flags.bundle) {
    const followed = addPolicy(flags);
    process.stdout.write(`${flags.data}/policy-view.bin.gz: the run of sampling seed ${followed.sampling_seed}, ${followed.decisions} decisions, set index ${followed.index}\n`);
  } else if (command === 'vendor' && flags.post) {
    const { commit, written } = vendor(flags);
    process.stdout.write(`${flags.post}: ${written.join(', ')} from ${commit}\n`);
  } else if (command === 'publish' && flags.data) {
    const { base, rows } = await publish(flags);
    for (const row of rows) process.stdout.write(`${row.url} ${row.bytes} ${row.sha256} first ${JSON.stringify(row.first)} second ${JSON.stringify(row.again)}\n`);
    process.stdout.write(`Published and verified ${rows.length} files at ${base}\n`);
  } else if (command === 'serve' && flags.data) {
    const port = Number(flags.port ?? 8737);
    serve({ data: flags.data, port });
    process.stdout.write(`Serving ${flags.data} with CORS at http://127.0.0.1:${port}/\n`);
  } else {
    throw Error('Usage: node blog.mjs data --site SITE_DATA --out DATA_DIR [--tier T --set S --count N] | policy --data DATA_DIR --bundle POLICY_DIR | vendor --post POST_DIR [--policy-view FILE --policy-origin TEXT] | serve --data DATA_DIR [--port P] | publish --data DATA_DIR');
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  });
}
