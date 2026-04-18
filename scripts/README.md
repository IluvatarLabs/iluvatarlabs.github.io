# Build scripts

## Thumbnail generation

Generates `/assets/thumbs/<slug>.png` for every post that has an element
tagged `data-thumb` in its rendered HTML.

Requires a fresh Jekyll build, then the script, then another Jekyll build
so the new manifest data is picked up:

```
npm install
npm run thumbs:full
```

Or manually:
```
bundle exec jekyll build
node scripts/generate-thumbs.js
bundle exec jekyll build
```

Commit generated `assets/thumbs/*.png` and `_data/thumbs.json` so CI builds
don't need Puppeteer.
