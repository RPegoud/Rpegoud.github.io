const WORDS_PER_MINUTE = 200; // slower default: technical/math-dense prose

/** Rough reading time estimate from raw markdown body text. */
export function readingTime(body: string): string {
  const words = body.trim().split(/\s+/).filter(Boolean).length;
  const minutes = Math.max(1, Math.round(words / WORDS_PER_MINUTE));
  return `${minutes} min read`;
}
