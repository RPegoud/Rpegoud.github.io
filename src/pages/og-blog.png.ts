import type { APIRoute } from 'astro';
import { renderOgImage } from '../lib/og-image';

export const GET: APIRoute = async () => {
  const png = await renderOgImage({
    eyebrow: 'Blog',
    title: 'Writing on inference efficiency and GPU kernels',
    claim: 'Triton kernels, model compression, and reinforcement learning — from first principles.',
    meta: 'rpegoud.github.io/blog',
  });

  return new Response(png, {
    headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' },
  });
};
