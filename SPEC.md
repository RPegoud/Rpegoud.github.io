# Portfolio site — build spec

## 0. Context

Personal site for an ML researcher (RL, inference efficiency, model compression, GPU kernels). Primary audience: hiring engineers and research leads at frontier labs and ML-heavy startups, scanning for 40–90 seconds. Secondary: readers arriving from Towards Data Science / Medium.

The site's single job: make it obvious within one screen that this person works on a **specific** technical problem area and produces rigorous, self-contained artifacts about it.

Start populating the website with articles in: `/Users/ryanpegoud/Documents/research_vault 2/01 -  Projects/Articles/Learning Triton One Kernel At A Time`: `.md` articles plus source files for animations (Manim) and illustrations (matplotlib, Excalidraw, etc.). Other articles will be added manually later.

---

## 1. Phase 0 — audit before building

**Do this first and report back before writing any application code.**

Inventory the source folder and produce a table covering:

1. Each `.md` article: title, word count, whether math is present as raw LaTeX (`$...$` / `$$...$$`) or as image references.
2. **If any article's equations exist only as images, stop and flag it.** Do not attempt OCR without asking. Report the count of affected articles and the approximate number of image-equations.
3. Every figure and animation referenced: file path, whether a regenerable source exists (`.py`, `.excalidraw`, Manim scene), and current resolution/format.
4. Any asset whose only copy came from `miro.medium.com` — these are recompressed and must be re-rendered from source or dropped.
5. Frontmatter present vs. missing per article.

Then propose a normalized frontmatter schema and wait for approval.

---

## 2. Stack — decided, do not re-litigate

| Concern | Choice |
|---|---|
| Framework | Astro (latest stable), static output |
| Content | Astro Content Collections, Zod-validated frontmatter |
| Article format | `.md` by default; `.mdx` **only** for articles needing an embedded widget |
| Math | `remark-math` + `rehype-katex`, build-time render, KaTeX CSS self-hosted |
| Styling | Tailwind v4 + `@tailwindcss/typography` (customized, not stock prose) |
| Islands | React via `@astrojs/react` |
| Code blocks | Shiki (Astro built-in), one light + one dark theme |
| Deploy | Cloudflare Pages |
| Large media | Cloudflare R2, referenced by URL — **never** committed to the repo |

**Constraints:**

- Zero JavaScript on any page that has no widget. Verify this; it is an acceptance criterion.
- No client-side math rendering. No layout shift from equations.
- No CMS, no database, no auth, no analytics beyond a single privacy-preserving snippet if requested later.

---

## 3. Content model

Two collections.

**`articles`** — full posts hosted on this site.

```ts
{
  title: string
  claim: string          // ONE sentence stating the technical finding. Not a teaser.
  date: Date
  updated?: Date
  venue?: 'Towards Data Science' | 'Medium' | 'Self-published'
  accolade?: string      // e.g. "Editor's Pick"
  originalUrl?: string   // canonical link if first published elsewhere
  theme: 'inference-efficiency' | 'compression' | 'rl' | 'systems'   // extend as needed
  featured: boolean
  heroAsset?: string
  draft: boolean
}
```

**`externalWriting`** — articles that stay on their original platform.

```ts
{ title, claim, date, venue, accolade?, url, theme }
```

`claim` is mandatory in both. If an article has no one-line finding, that is a signal it should not be featured — surface that in the audit rather than inventing one.

---

## 4. Homepage

Asymmetric hierarchy. **No carousel, no slider, no auto-advancing anything.** Featured work must be visible simultaneously without interaction.

```
┌──────────────────────────────────────────────────────┐
│  Name · one line on what they work on                │
│  (no photo, no "welcome to my blog")                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│   FLAGSHIP — full width                              │
│   Title, claim, live widget or animation preview     │
│                                                      │
├────────────────┬────────────────┬────────────────────┤
│  Featured 1    │  Featured 2    │  Featured 3        │
│  claim         │  claim         │  claim             │
│  venue · date  │  venue · date  │  venue · date      │
├────────────────┴────────────────┴────────────────────┤
│  WRITING ELSEWHERE                                   │
│  Dense table, grouped by theme (not chronological)   │
├──────────────────────────────────────────────────────┤
│  Publications · minimal, links out                   │
└──────────────────────────────────────────────────────┘
```

Card rules:

- Card body is the `claim` field. Never an excerpt, never a truncated first paragraph.
- Provenance is a single muted line: `Towards Data Science · Editor's Pick · Mar 2025`. Secondary type size, no badge, no coloured pill, no icon. It should read as a fact, not a trophy.
- Thumbnails only where a re-rendered high-DPI asset exists. Otherwise text-only cards with generous whitespace — that is the better outcome, not the fallback.
- Grid stacks to one column on mobile. Featured grid is `grid`, not `flex`, so card heights align.

External writing table: title, theme, venue, date. Muted metadata columns. Grouped under theme headings so the through-line is legible at a glance.

---

## 5. Article layout

Distill-style reading experience:

- Body measure ~68ch, centered.
- **Margin sidenotes** that flow inline below the paragraph on narrow viewports. Reference implementation: Tufte CSS. Expose as an MDX component `<Sidenote>` and a markdown-friendly footnote fallback.
- **Full-bleed and wide figures** that break the text column. Provide `<Figure width="column|wide|full">` with a caption slot.
- Sticky-free table of contents in the left margin on wide viewports, collapsed to a top disclosure on mobile. Highlight the current section.
- Reading time and last-updated in the header.
- Canonical `<link rel="canonical">` pointing to `originalUrl` where present.

---

## 6. Math

- `$inline$` and `$$block$$` via remark-math.
- KaTeX renders at build time. Self-host the fonts; do not hotlink the CDN.
- KaTeX does not support `\label` / `\eqref`. If any article needs numbered, referenceable equations, implement a small MDX `<Equation id="...">` component with a counter and an `<EqRef>` — do **not** swap the whole pipeline to MathJax for a handful of cases.
- Verify every article renders without KaTeX errors. Fail the build on parse errors rather than silently emitting red text.

---

## 7. Widgets

Islands, loaded with `client:visible`. Each widget lives in `src/components/widgets/`, is self-contained, and has no dependency on page state.

**First widget — memory calculator (vocabulary pruning article).** Inputs: vocabulary size, hidden dimension, number of layers, dtype, batch size, sequence length. Outputs: total parameter count, embedding + LM-head share of parameters, activation memory during fine-tuning, and bytes moved per decode step. Show the pruned-vs-baseline delta as the headline number, since that is the article's actual claim.

Requirements for all widgets:

- Sensible defaults matching a real model, so the widget is informative before the reader touches it.
- Keyboard operable; visible focus states.
- Degrades to a static table of representative values if JS is disabled.
- Charts: `uPlot` or Observable Plot. Do not pull in full d3 for a line chart.

---

## 8. Media pipeline

Write a script (`scripts/build-media.*`) that regenerates derivatives from source. Do not hand-encode.

**Manim animations** — render at 1440p60 from source, then encode two derivatives:
- AV1 (`libsvtav1`) and H.264 (`libx264`, `yuv420p`, `+faststart`) in MP4.
- Serve via `<video muted loop playsinline preload="metadata" poster>`. Poster is a WebP first frame.
- Respect `prefers-reduced-motion`: do not autoplay; show the poster with a play control.

**Scrubbable sequences** (a slider stepping through frames, e.g. a diffusion or pruning schedule) — do not use `<video>`. Export frames as WebP, preload, drive from a range input.

**Static figures** — regenerate as SVG from matplotlib source where possible. Raster only where SVG is impractical, at 2x with a WebP derivative.

**Sizing:** Cloudflare Pages rejects individual assets over 25 MB. Anything above ~5 MB goes to R2 from the start. The build script should error loudly if a repo-committed asset exceeds 5 MB.

---

## 9. Design direction

The brief is: **precise, quiet, instrument-like.** The subject matter is measurement — roofline plots, bandwidth numbers, ablations. The design should feel like a well-made technical instrument, not a magazine and not a startup landing page.

Follow the two-pass process: produce a token plan (4–6 named hex values, typefaces for display / body / utility-mono roles, a layout concept, and one signature element), critique it against this brief, then build from the revised plan.

Hard constraints:

- **Do not** use warm cream (~`#F4F1EA`) with a terracotta/clay accent (~`#D97757`). Do not use near-black with a single acid-green or vermilion accent. Do not use a hairline-ruled broadsheet pastiche. These are the current generic defaults and all three would be wrong here.
- Body face: a text serif with real typographic quality at 19–20px (Newsreader, Source Serif 4, and Spectral are all acceptable starting points; justify whatever you pick).
- A monospace face earns a genuine role — data, units, kernel names, table figures — not decoration.
- Spend boldness in exactly one place. The signature element should be the thing the flagship article's widget or animation does; everything around it stays disciplined.
- No gradient hero, no glassmorphism, no scroll-jacking, no ambient particle background.
- Motion: page-load and scroll-reveal only if it serves comprehension. Default to none.
- Self-host all fonts. Subset them.

Copy is design material. Write in active voice, sentence case, no filler. Empty states and 404 give direction rather than mood.

---

## 10. Quality floor

- Responsive to 360px.
- Visible keyboard focus throughout; `prefers-reduced-motion` respected.
- Lighthouse: 100 accessibility, 100 SEO, performance ≥95 on the heaviest article.
- Zero JS on non-widget pages — verify in the network panel, not by assumption.
- RSS feed, sitemap, per-article OG images generated at build (Satori).
- Dark mode via `prefers-color-scheme`, with a manual toggle. KaTeX and Shiki must both be themed correctly in each.

---

## 11. Phasing

1. **Audit** (§1) — report and stop.
2. Scaffold, content collections, one article rendering end-to-end with math and a figure.
3. Article layout: sidenotes, wide figures, TOC.
4. Homepage.
5. Memory calculator widget.
6. Media pipeline script, remaining article ports.
7. Deploy, OG images, RSS.

Ship after step 4 if time is short. Steps 5–7 are improvements to a live site, not prerequisites.

---

## 12. Non-goals

- Comments, newsletter signup, social feeds.
- A general-purpose design system or component library.
- Search (10–15 articles do not need it).
- Any framework migration or "while I was in there" refactor.
- Porting every article. Featured set is 4–5. The rest are links.
