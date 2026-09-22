import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const articles = (await getCollection('articles', ({ data }) => !data.draft)).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf(),
  );

  return rss({
    title: 'Ryan Pégoud',
    description: 'Writing on inference efficiency, compression, and GPU kernels.',
    site: context.site!,
    items: articles.map((entry) => ({
      title: entry.data.title,
      description: entry.data.claim,
      pubDate: entry.data.date,
      link: `/articles/${entry.id}/`,
    })),
  });
}
