# Iluvatar Labs site — agent guide

Excluded from Jekyll build (see `_config.yml > exclude`). Notes for
future agents working on this repo.

## Repo layout

- `index.html` — homepage (layout: null, owns its full markup)
- `actuator/`, `marvin/`, `iori/`, `about/` — product / org pages,
  each `index.html` is `layout: null` and owns its full markup
- `blog/index.html` — Discover index (post list)
- `_posts/` — blog posts in markdown, served via `_layouts/post.html`
- `_projects/` — IORI project case studies, `_layouts/iori-project.html`
- `_includes/nav.html` — shared nav across landing pages
- `_includes/thumb-icon.html` — blog post thumbnail pictograph SVGs

## Copy / style rules (HARD)

- **No em-dashes (`—`)** in prose, frontmatter descriptions, post bodies,
  or hero copy. Use commas, periods, or rephrase. Em-dashes read as an
  LLM tell.
- **No emojis** unless the user explicitly asks.
- **No eyebrow labels** above section headings (small mono "OUR THESIS"
  / "WHAT WE BUILD" tags). Section headings stand alone.
- **No post chrome**: no prev/next post footers, no breadcrumbs above
  post titles.
- **No marketing pills** ("Closed beta · 2026", beta badges, status
  footer lines).
- **Product pages** reuse the existing product source verbatim. Don't
  invent blog-post-style intros for product pages.
- **"Less dense"** means more whitespace and padding, not smaller text
  or cut content.
- **Heroes / nav must structurally match the homepage** across pages
  (same items, icons, sizes, fonts). Only colors adapt per palette.

## Adding a new blog post

Drop a file in `_posts/YYYY-MM-DD-slug.md` with this frontmatter:

```yaml
---
title: "Post Title"
subtitle: "Optional one-line subtitle"
date: YYYY-MM-DD
category: Product   # or Research
thumbnail: marvin   # one of: marvin | actuator | dca | elastic | indigo (fallback)
author: "Iluvatar Labs"
excerpt: "1-2 sentence summary used in the post list and as og:description."
image: /assets/images/<slug>/social-card.png   # optional, enables Twitter summary_large_image
---

Body markdown here.
```

The `thumbnail` value drives both the background palette
(`.lab-thumb-{value}` CSS class on the homepage and Discover index) and
the pictograph SVG (`_includes/thumb-icon.html` Liquid case). Both
must know about the value or you'll get an unstyled card or no glyph.

## Thumbnail system

### Visual language (HARD)

Every thumbnail follows the same rules. Don't break them.

- **Aspect**: 4:3, `viewBox="0 0 480 360"`
- **One pictograph**: one dominant bold geometric symbol, centered or
  off-center with generous negative space
- **Two colors**: post family color (background) + post accent color
  (glyph). No third color, no gradients on the glyph itself.
- **Stroke discipline**: where strokes are used, 12-14px is the main
  weight, 6-8px for secondary detail (tick marks, etc.). Always
  `stroke-linecap="round"` and `stroke-linejoin="round"`.
- **No text** inside the thumb. Title sits beside it on the index.
- **No labels, no arrowheads** unless they're load-bearing semantics
  (e.g. Elastic Speculation arrows show forward direction).
- **No graphs of dots-and-edges**. Pictograph means symbol, not diagram.
- **Background**: full bleed in family color, with two corner radial
  gradient glows (top-right + bottom-left, low alpha, accent palette)
  and SVG-turbulence grain via `::before` pseudo-element.

### Existing pictographs

| Post family | Background | Accent | Glyph | What it represents |
|-------------|-----------|--------|-------|-------------------|
| `marvin` | aegean `#16415A` | cyan `#6EC8E6` | atom (3 crossing ellipses + nucleus) | Marvin is a research agent. Atom = research/science. |
| `actuator` | cream `#FAFAF5` | coral `#CC6A5B` | gauge (arc + ticks + needle + hub) | Actuator measures and controls. Gauge = measure-and-control. |
| `dca` | indigo `#2E2A5A` | lavender `#B8A6FF` | hourglass (two filled triangles meeting at a point) | The architecture's literal shape: tokens diverge → converge → diverge. |
| `elastic` | ink-dark `#14141A` | cyan `#6EC8E6` | three forward arrows of varying length, one dashed | Variable-depth lookahead with retraction. |
| `indigo` | indigo `#2E2A5A` | lavender (no glyph) | (fallback only) | Used when no thumbnail value is recognized. |

### Background palette reference

```css
/* Two corner radial gradients (low alpha, accent palette) +
   solid base color. Cream-on-dark grain or dark-on-cream grain
   layered above via ::before. */

/* Marvin */
background:
    radial-gradient(ellipse 92% 92% at 95% 10%, rgba(110,200,230,0.24) 0%, rgba(110,200,230,0) 62%),
    radial-gradient(ellipse 92% 82% at 5% 100%, rgba(224,129,74,0.16) 0%, rgba(224,129,74,0) 65%),
    #16415A;

/* Actuator (cream — uses dark-on-cream grain) */
background:
    radial-gradient(ellipse 100% 90% at 95% 8%, rgba(233,123,106,0.18) 0%, rgba(233,123,106,0) 60%),
    radial-gradient(ellipse 95% 85% at 5% 100%, rgba(79,70,229,0.14) 0%, rgba(79,70,229,0) 65%),
    #FAFAF5;

/* DCA */
background:
    radial-gradient(ellipse 95% 90% at 95% 10%, rgba(184,166,255,0.22) 0%, rgba(184,166,255,0) 60%),
    radial-gradient(ellipse 90% 80% at 5% 100%, rgba(233,123,106,0.14) 0%, rgba(233,123,106,0) 65%),
    #2E2A5A;

/* Elastic */
background:
    radial-gradient(ellipse 95% 90% at 95% 10%, rgba(110,200,230,0.22) 0%, rgba(110,200,230,0) 60%),
    radial-gradient(ellipse 90% 80% at 5% 100%, rgba(224,129,74,0.12) 0%, rgba(224,129,74,0) 65%),
    #14141A;
```

Grain (use exactly these values — earlier overcorrections were rejected):
- baseFrequency `0.95`, multiplier `1.4`, offset `-1.95`
- Cream-on-dark grain opacity: `0.35`
- Dark-on-cream grain opacity: `0.25`

### Adding a new thumbnail variant

When a new post needs a new pictograph (the existing ones don't fit):

**1. Pick the symbol.** Apply these tests in order:
   - Is there a literal architecture shape? (DCA hourglass passed this.)
   - Is there a universal pictograph for the domain? (Marvin atom for
     "research", Actuator gauge for "measure-and-control".)
   - Is the symbol identifiable in 0.5 seconds at thumbnail size? If
     not, it's wrong.
   - Avoid: graphs-of-dots-and-edges, flowcharts with boxes/arrows,
     branded clip art, anything custom-illustrated freehand (heads,
     anatomy, etc.) — these have all been rejected before.

**2. Pick a palette.** Either reuse one of the existing five palettes,
   or define a new one. New palettes need:
   - Base background color (solid)
   - Accent color used for the glyph
   - Two corner radial gradient glows with low-alpha colors
   - Choose grain variant: cream-on-dark for dark backgrounds,
     dark-on-cream for cream backgrounds

**3. Add the SVG to `_includes/thumb-icon.html`.** Add a new
   `{% when 'your-value' %}` case. Follow the visual language rules
   above. Use `viewBox="0 0 480 360"`, single accent color, round
   caps/joins.

**4. Add CSS variants in two places** (different files, both needed):

   - `blog/index.html` — `.lab-thumb-<value>` selector + matching
     `.lab-thumb-<value>::before` for grain. Used on Discover index.
   - `index.html` — `.lab-thumb-<value>` selector. Used in the
     homepage's "From the team" section. Note: the homepage uses a
     separate `.lab-thumb-grain` div (not `::before`); if your new
     palette needs dark-on-cream grain, also add an
     `.lab-thumb-<value> + .lab-thumb-grain` override (see how
     `.lab-thumb-actuator + .lab-thumb-grain` does it).

**5. Set `thumbnail: <value>` in the post's frontmatter.**

**6. Verify.** Build (`bundle exec jekyll build`) and load both `/`
   (homepage featured + grid) and `/blog/` (Discover index featured)
   to make sure the SVG renders inside the colored card with the
   grain layer and the palette matches.

### Mockup workflow

Mockups live as `mockup-blog-thumbs-vN.html` at the repo root (excluded
from Jekyll build via the wildcard in `_config.yml`). When iterating
on a new variant, work in a fresh mockup file rather than editing the
production includes. The current production version is whatever shipped
last (check git log: `git log --oneline -- _includes/thumb-icon.html`).

## Hero sizing

Heroes use a `min-height: <vh>` baseline + a laptop-height media query
floor so 16" MBPs and similar shorter viewports don't render squished:

```css
.hero { min-height: 58vh; /* or 65vh, etc. */ }

/* Laptop-height floor: activates only on viewports ≤ 1399px tall,
   so 1440p+ desktops keep their vh-based sizing untouched. */
@media (min-width: 961px) and (max-height: 1399px) {
    .hero { min-height: 820px; /* or whatever pixel floor fits */ }
}
```

Don't use `max(<vh>, <px>)` — different MBP scaling modes (Default
1117 vh / More Space 1329 vh) need the media query to consistently
activate, and `max()` skips the floor when vh exceeds it.

## Color tokens (site-wide)

Defined per-page in `:root`. Common ones:

- `--ink-cream` `#F5F1EB`, soft `0.72`, mute `0.46`, faint `0.28`
- `--ink-dark` `#14141A`, soft `0.72`, mute `0.46`
- `--page-bg` `#FAFAF5`
- `--accent` `#E97B6A` (coral), `--accent-deep` `#CC6A5B`
- `--accent-cyan` `#6EC8E6`
- Page palettes: aegean `#16415A`, pine `#1F3D2C`, indigo `#2E2A5A`,
  lavender `#B8A6FF`
