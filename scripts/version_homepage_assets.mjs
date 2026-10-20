#!/usr/bin/env node
// Refresh local asset URLs without bundling or changing the offline page.
import fs from 'node:fs';
import crypto from 'node:crypto';

const root = new URL('../examples/code-research-demo/', import.meta.url);
const file = new URL('index.html', root);
const html = fs.readFileSync(file, 'utf8');
const headEnd = html.indexOf('</head>');
if (headEnd < 0) throw new Error('Homepage head not found');
const head = html.slice(0, headEnd).replace(
  /((?:href|src)=")([\w./-]+\.(?:css|js|svg))(?:\?v=[^"]*)?(")/g,
  (_, prefix, asset, suffix) => {
    const digest = crypto.createHash('sha256')
      .update(fs.readFileSync(new URL(asset, root))).digest('hex').slice(0, 12);
    return `${prefix}${asset}?v=${digest}${suffix}`;
  },
);
fs.writeFileSync(file, head + html.slice(headEnd));
console.log('Homepage asset versions match their contents.');
