import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(resolve(root, 'app/package.json'));
const { transform } = require('esbuild');
const dependencies = JSON.parse(await readFile(resolve(root, 'app/package.json'), 'utf8')).dependencies;
const assets = resolve(root, 'src/ml_stack/ui/assets');
const modules = resolve(root, 'app/node_modules');
const files = [
  ['marked', 'lib/marked.umd.js', 'marked.umd.js'],
  ['dompurify', 'dist/purify.min.js', 'purify.min.js'],
  ['three', 'build/three.module.js', 'three.module.js'],
  ['three', 'build/three.core.js', 'three.core.js'],
  ['three', 'examples/jsm/controls/OrbitControls.js', 'OrbitControls.js'],
];
await mkdir(assets, { recursive: true });
for (const [name, source, target] of files) {
  const text = (await readFile(resolve(modules, name, source), 'utf8'))
    .replaceAll("from 'three'", "from './three.module.js'");
  const { code } = await transform(text, {
    minify: true, legalComments: 'inline', target: 'es2022',
  });
  await writeFile(resolve(assets, target), code);
}
for (const name of Object.keys(dependencies)) {
  await copyFile(resolve(modules, name, 'LICENSE'), resolve(assets, `${name}.LICENSE`));
}
await writeFile(resolve(assets, 'vendor.json'), JSON.stringify({ dependencies, files }, null, 2) + '\n');
