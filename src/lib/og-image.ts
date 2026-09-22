import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Static (non-variable) woff files — satori can't parse variable fonts or
// woff2 (brotli) compression, only plain woff/ttf/otf. Resolved through
// Node's own module resolution (not a relative path from this file) since
// the build bundles this module somewhere under dist/.prerender/, where a
// '../../node_modules' style path no longer points anywhere real.
function resolveFontPath(specifier: string): string {
  return fileURLToPath(import.meta.resolve(specifier));
}
const newsreaderRegular = readFileSync(
  resolveFontPath('@fontsource/newsreader/files/newsreader-latin-400-normal.woff'),
);
const newsreaderMedium = readFileSync(
  resolveFontPath('@fontsource/newsreader/files/newsreader-latin-500-normal.woff'),
);
const plexMonoRegular = readFileSync(
  resolveFontPath('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff'),
);
const plexMonoMedium = readFileSync(
  resolveFontPath('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-600-normal.woff'),
);

const fonts = [
  { name: 'Newsreader', data: newsreaderRegular, weight: 400 as const, style: 'normal' as const },
  { name: 'Newsreader', data: newsreaderMedium, weight: 500 as const, style: 'normal' as const },
  { name: 'IBM Plex Mono', data: plexMonoRegular, weight: 400 as const, style: 'normal' as const },
  { name: 'IBM Plex Mono', data: plexMonoMedium, weight: 600 as const, style: 'normal' as const },
];

// Same tokens as global.css's light palette — OG images render once at
// build time with no theme toggle, so they're fixed to light.
const INK = '#161b22';
const INK_MUTED = '#5b6660';
const ACCENT = '#c2410c';
const BG = '#ffffff';
const RULE = '#c9cdc5';

interface OgImageOptions {
  eyebrow: string;
  title: string;
  claim: string;
  meta?: string;
}

export async function renderOgImage({ eyebrow, title, claim, meta }: OgImageOptions): Promise<Buffer> {
  const svg = await satori(
    {
      type: 'div',
      props: {
        style: {
          width: '1200px',
          height: '630px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          backgroundColor: BG,
          padding: '72px',
          fontFamily: 'Newsreader',
        },
        children: [
          {
            type: 'div',
            props: {
              style: {
                display: 'flex',
                fontFamily: 'IBM Plex Mono',
                fontSize: '22px',
                fontWeight: 600,
                letterSpacing: '2px',
                textTransform: 'uppercase',
                color: ACCENT,
              },
              children: eyebrow,
            },
          },
          {
            type: 'div',
            props: {
              style: { display: 'flex', flexDirection: 'column', gap: '28px' },
              children: [
                {
                  type: 'div',
                  props: {
                    style: {
                      display: 'flex',
                      fontSize: '58px',
                      fontWeight: 500,
                      lineHeight: 1.15,
                      color: INK,
                      maxWidth: '1000px',
                    },
                    children: title,
                  },
                },
                {
                  type: 'div',
                  props: {
                    style: {
                      display: 'flex',
                      fontSize: '28px',
                      lineHeight: 1.45,
                      color: INK_MUTED,
                      maxWidth: '920px',
                    },
                    children: claim,
                  },
                },
              ],
            },
          },
          {
            type: 'div',
            props: {
              style: {
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderTop: `2px solid ${RULE}`,
                paddingTop: '28px',
                fontFamily: 'IBM Plex Mono',
                fontSize: '22px',
                color: INK_MUTED,
              },
              children: [
                { type: 'div', props: { style: { display: 'flex' }, children: 'Ryan Pégoud' } },
                ...(meta ? [{ type: 'div', props: { style: { display: 'flex' }, children: meta } }] : []),
              ],
            },
          },
        ],
      },
    },
    { width: 1200, height: 630, fonts },
  );

  const resvg = new Resvg(svg, { fitTo: { mode: 'width', value: 1200 } });
  return resvg.render().asPng();
}
