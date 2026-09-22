import { visit } from 'unist-util-visit';

const WIDTHS = new Set(['column', 'wide', 'full']);

/**
 * Turns a standalone `![alt](src "wide")` image — optionally followed on the
 * very next line by an `_italic caption_` — into a `figure` mdast node
 * carrying a width mode (SPEC §5: `<Figure width="column|wide|full">`).
 * Plain markdown stays the authoring format; the width mode rides in the
 * image's title slot (`"wide"` / `"full"`), defaulting to `"column"` (no
 * special treatment) when omitted or unrecognised.
 *
 * Because there's no blank line between the image and its caption line,
 * remark-parse doesn't give two paragraphs here — it gives ONE paragraph
 * whose children are `[image, text("\n"), emphasis]`. This plugin matches
 * that shape directly rather than looking for a second paragraph sibling.
 *
 * The original `image` node is kept unchanged (just re-parented) so Astro's
 * own local-image collection/optimization — which walks the tree generically
 * for `image` nodes and later matches `<img>` tags in the hast output — still
 * finds and optimizes it normally.
 */
export function remarkFigures() {
  return (tree) => {
    visit(tree, 'paragraph', (node, index, parent) => {
      if (!parent || index === undefined) return;
      const [first, ...rest] = node.children;
      if (!first || first.type !== 'image') return;

      const width = WIDTHS.has(first.title) ? first.title : 'column';
      // The title slot is repurposed as the width-mode signal — clear it so
      // it doesn't leak onto the rendered <img> as a native tooltip.
      if (WIDTHS.has(first.title)) first.title = null;
      const nonWhitespace = rest.filter((n) => !(n.type === 'text' && n.value.trim() === ''));

      let figureChildren;
      if (nonWhitespace.length === 0) {
        figureChildren = [first];
      } else if (nonWhitespace.length === 1 && nonWhitespace[0].type === 'emphasis') {
        figureChildren = [first, { type: 'figureCaption', children: nonWhitespace[0].children }];
      } else {
        return; // unrecognised trailing content — leave the paragraph as plain markdown
      }

      parent.children[index] = {
        type: 'figure',
        data: { width },
        children: figureChildren,
      };
    });
  };
}
