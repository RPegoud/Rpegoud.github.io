import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { renderOgImage } from '../../../lib/og-image';

export async function getStaticPaths() {
  const articles = await getCollection('articles', ({ data }) => !data.draft);
  return articles.map((entry) => ({ params: { slug: entry.id }, props: { entry } } as const));
}

export const GET: APIRoute = async ({ props }) => {
  const { entry } = props as any;
  const meta = [entry.data.venue, entry.data.date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' })]
    .filter(Boolean)
    .join(' · ');

  const png = await renderOgImage({
    eyebrow: 'Article',
    title: entry.data.title,
    claim: entry.data.claim,
    meta,
  });

  return new Response(png, {
    headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' },
  });
};
