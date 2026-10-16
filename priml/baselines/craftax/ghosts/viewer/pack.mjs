// Assemble the ghost-run site: the page's files beside the builder's data
// (../FORMAT.md), each data file checked against the manifest's size and
// SHA-256, with no file the manifest does not list and none over the
// Artifact host's limits.
//
// The host serves only web types by extension and refuses gzip and other
// binary data, so each binary data file PATH is written as PATH.b64.txt, its
// bytes in base64 (standard alphabet, no line breaks); ghosts.js decodes it.
//
// With --policy, the policy view's bundle (games.py panel) of the run a wins
// set pins goes beside the page as policy-view.json (its manifest),
// policy-view.bin.gz.b64.txt and, when it has sleep frames,
// policy-sleep.bin.gz.b64.txt, once its sampling seed and length are checked
// against that run's; the page shows it while that run is followed. With
// --policy-view, the panel's script is that built file (games.mjs panel)
// rather than one built here from this tree's hud.js and policy_view.js.
//
//   node pack.mjs --data DATA_DIR --out SITE_DIR [--policy POLICY_DIR] [--policy-view FILE]
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = { 'index.html': 'index.html', 'ghosts.js': 'ghosts.js', 'decode.js': 'decode.js', 'sprites.png': '../../world_model/viewer/sprites.png' };
// The policy view's script as world_model/viewer/games.mjs panel writes it:
// hud.js and policy_view.js in one function, adding only CraftaxPolicyView.
const POLICY_VIEW = () => Buffer.from(`(() => {\n${['hud.js', 'policy_view.js'].map(name => fs.readFileSync(path.join(here, '../../world_model/viewer', name), 'utf8')).join('\n')}})();\n`);
const LIMIT = { binary: 15 * 2 ** 20, text: 16 * 2 ** 20, total: 256 * 2 ** 20, files: 511 };
const BINARY = /\.bin\.gz$/;

function listFiles(dir, prefix = '') {
  return fs.readdirSync(path.join(dir, prefix), { withFileTypes: true }).flatMap(entry => {
    const name = path.posix.join(prefix, entry.name);
    return entry.isDirectory() ? listFiles(dir, name) : [name];
  });
}

// The policy bundle's files, once it is the run a wins set pins.
function policyFiles(data, manifest, bundle) {
  const policy = JSON.parse(fs.readFileSync(path.join(bundle, 'manifest.json'), 'utf8')), bytes = fs.readFileSync(path.join(bundle, 'policy-view.bin.gz'));
  if (!['craftax-policy-view/v1', 'craftax-policy-view/v2'].includes(policy.schema) || crypto.createHash('sha256').update(bytes).digest('hex') !== policy.gzipSha256) throw Error(`${bundle} does not hold the policy-view bundle its manifest describes.`);
  const pinned = manifest.tiers.flatMap(t => t.sets).filter(s => s.time_map?.unbroken != null).map(s => {
    const group = s.groups.find(g => g.first <= s.time_map.unbroken && s.time_map.unbroken < g.first + g.count);
    return JSON.parse(fs.readFileSync(path.join(data, group.path, 'episodes.json'), 'utf8')).episodes[s.time_map.unbroken - group.first];
  });
  if (!pinned.some(e => e.sampling_seed === policy.samplingSeed && e.decisions === policy.decisions)) {
    throw Error(`The policy view shows the run of sampling seed ${policy.samplingSeed} (${policy.decisions} decisions), which no wins set pins: ${pinned.map(e => `${e.sampling_seed} (${e.decisions})`).join(', ') || 'none'}.`);
  }
  const files = [['policy-view.json', Buffer.from(JSON.stringify(policy, null, 1) + '\n')], ['policy-view.bin.gz.b64.txt', Buffer.from(bytes.toString('base64'))]];
  // A bundle with sleep frames carries them beside its decision frames.
  if (policy.sleepFrames) {
    const slept = fs.readFileSync(path.join(bundle, 'policy-sleep.bin.gz'));
    if (crypto.createHash('sha256').update(slept).digest('hex') !== policy.sleepGzipSha256) throw Error(`${bundle}/policy-sleep.bin.gz does not match its manifest.`);
    files.push(['policy-sleep.bin.gz.b64.txt', Buffer.from(slept.toString('base64'))]);
  }
  return files;
}

export function pack({ data, out, policy = null, policyView = null }) {
  if (fs.existsSync(out) && fs.readdirSync(out).length) throw Error(`${out} is not empty; pack into a new directory.`);
  const manifest = JSON.parse(fs.readFileSync(path.join(data, 'manifest.json'), 'utf8'));
  const listed = Object.keys(manifest.files), present = listFiles(data).filter(name => name !== 'manifest.json');
  const unlisted = present.filter(name => !(name in manifest.files)), missing = listed.filter(name => !present.includes(name));
  if (unlisted.length || missing.length) throw Error(`Data and manifest disagree: unlisted ${unlisted.join(', ') || 'none'}; missing ${missing.join(', ') || 'none'}.`);
  const sources = [
    ...['manifest.json', ...listed].map(name => [name, path.join(data, name)]),
    ...Object.entries(PAGE).map(([name, source]) => [name, path.join(here, source)]),
  ];
  const extra = [['policy_view.js', policyView ? fs.readFileSync(policyView) : POLICY_VIEW()], ...(policy ? policyFiles(data, manifest, policy) : [])];
  const written = sources.map(([source, file]) => {
    let bytes = fs.readFileSync(file), name = source;
    if (source in manifest.files) {
      const sha = crypto.createHash('sha256').update(bytes).digest('hex');
      if (bytes.length !== manifest.sizes[source] || sha !== manifest.files[source]) throw Error(`${source} does not match the manifest: ${bytes.length} bytes, sha256 ${sha}.`);
    }
    if (BINARY.test(source)) [name, bytes] = [`${source}.b64.txt`, Buffer.from(bytes.toString('base64'))];
    const kind = name.endsWith('.png') ? 'binary' : 'text';
    if (bytes.length > LIMIT[kind]) throw Error(`${name} is ${bytes.length} bytes, over the host's ${LIMIT[kind]}-byte limit for a ${kind} file.`);
    return { name, bytes };
  }).concat(extra.map(([name, bytes]) => ({ name, bytes })));
  const total = written.reduce((sum, f) => sum + f.bytes.length, 0);
  if (total > LIMIT.total || written.length > LIMIT.files) throw Error(`The site holds ${written.length} files and ${total} bytes, over the host's ${LIMIT.files} files or ${LIMIT.total} bytes.`);
  for (const { name, bytes } of written) {
    fs.mkdirSync(path.dirname(path.join(out, name)), { recursive: true });
    fs.writeFileSync(path.join(out, name), bytes);
  }
  return written.map(({ name, bytes }) => ({ name, bytes: bytes.length }));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), at = flag => args[args.indexOf(flag) + 1];
  const flags = new Set(args.filter((_, i) => i % 2 === 0));
  if (args.length % 2 || !flags.has('--data') || !flags.has('--out') || [...flags].some(f => !['--data', '--out', '--policy', '--policy-view'].includes(f))) {
    process.stderr.write('Usage: node pack.mjs --data DATA_DIR --out SITE_DIR [--policy POLICY_DIR] [--policy-view FILE]\n');
    process.exit(1);
  }
  try {
    const written = pack({ data: at('--data'), out: at('--out'), policy: flags.has('--policy') ? at('--policy') : null, policyView: flags.has('--policy-view') ? at('--policy-view') : null });
    const total = written.reduce((sum, f) => sum + f.bytes, 0), largest = written.reduce((a, b) => (b.bytes > a.bytes ? b : a));
    process.stdout.write(`${at('--out')}: ${written.length} files, ${total} bytes; largest ${largest.name} ${largest.bytes} bytes\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
  }
}
