import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// SPEC §3. `theme` is extended as new writing doesn't fit the original
// placeholder set: 'kernels' for the Triton kernel-writing series (distinct
// from broader 'systems' work), 'autonomous-driving' and 'llms' for
// externalWriting entries outside the site's core kernels/RL focus.
const theme = z.enum([
  'inference-efficiency',
  'compression',
  'rl',
  'systems',
  'kernels',
  'autonomous-driving',
  'llms',
]);

const venue = z.enum(['Towards Data Science', 'Medium', 'Self-published']);

const articles = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    // ONE sentence stating the technical finding. Not a teaser.
    claim: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    venue: venue.optional(),
    accolade: z.string().optional(),
    originalUrl: z.string().url().optional(),
    theme,
    featured: z.boolean().default(false),
    heroAsset: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

const externalWriting = defineCollection({
  loader: glob({ pattern: '**/*.{md,yaml,yml}', base: './src/content/externalWriting' }),
  schema: z.object({
    title: z.string(),
    claim: z.string(),
    date: z.coerce.date(),
    venue,
    accolade: z.string().optional(),
    url: z.string().url(),
    theme,
  }),
});

export const collections = { articles, externalWriting };
