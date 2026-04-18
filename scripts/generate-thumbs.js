const puppeteer = require('puppeteer');
const fs = require('fs/promises');
const path = require('path');
const { globby } = require('globby');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'assets/thumbs');
const MANIFEST = path.join(ROOT, '_data/thumbs.json');

const VIEWPORT = { width: 1200, height: 750, deviceScaleFactor: 2 };
const SETTLE_MS = 1600;

function slugFrom(file) {
  return path.basename(path.dirname(file));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(path.dirname(MANIFEST), { recursive: true });

  const files = await globby(['_site/blog/**/index.html'], { cwd: ROOT, absolute: true });
  if (files.length === 0) {
    console.error('No posts found under _site/blog. Did you run `jekyll build` first?');
    process.exit(1);
  }

  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport(VIEWPORT);

  const manifest = {};
  for (const file of files) {
    const slug = slugFrom(file);
    await page.goto(`file://${file}`, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, SETTLE_MS));

    const el = await page.$('[data-thumb]');
    if (!el) { console.warn(`skip ${slug}: no [data-thumb]`); continue; }

    const out = path.join(OUT_DIR, `${slug}.png`);
    await el.screenshot({ path: out, type: 'png', omitBackground: false });
    manifest[slug] = `/assets/thumbs/${slug}.png`;
    console.log(`\u2713 ${slug}`);
  }
  await browser.close();
  await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
  console.log(`\n${Object.keys(manifest).length} thumbs written; manifest at ${path.relative(ROOT, MANIFEST)}`);
}

main().catch(err => { console.error(err); process.exit(1); });
