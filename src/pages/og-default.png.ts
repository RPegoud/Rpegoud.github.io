import type { APIRoute } from 'astro';
import { renderOgImage } from '../lib/og-image';

export const GET: APIRoute = async () => {
  const png = await renderOgImage({
    eyebrow: 'Ryan Pégoud',
    title: 'Research Engineer',
    claim: 'RL, inference efficiency, and GPU kernels.',
    meta: 'rpegoud.github.io',
  });

  return new Response(png, {
    headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' },
  });
};
