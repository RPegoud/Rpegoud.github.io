// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
import { remarkSidenotes } from './src/lib/remark-sidenotes.mjs';
import { remarkFigures } from './src/lib/remark-figures.mjs';
import { hastHandlers } from './src/lib/hast-handlers.mjs';

import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  // GitHub user page (rpegoud/rpegoud.github.io) — serves at the domain
  // root, so no `base` path is needed (unlike a project page).
  site: 'https://rpegoud.github.io',
  integrations: [react(), sitemap(), mdx()],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkFigures, remarkSidenotes],
      rehypePlugins: [
        [
          rehypeKatex,
          {
            // Fail the build on parse errors instead of emitting red error text
            // (SPEC §6: "Fail the build on parse errors rather than silently emitting red text.")
            strict: 'error',
            trust: false,
            macros: {
              // TeX shrinks fraction contents to textstyle by default, which
              // crams \sum/\prod limits to the side instead of stacking them
              // above/below — technically correct TeX behavior, but hard to
              // read for the sum-over-a-fraction shapes this site uses
              // constantly (softmax, cross-entropy, ...). Force full display
              // style in every fraction instead.
              '\\frac': '\\dfrac',
            },
          },
        ],
      ],
      remarkRehype: { handlers: hastHandlers },
    }),
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
});