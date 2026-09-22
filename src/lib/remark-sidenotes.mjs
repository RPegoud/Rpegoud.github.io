import { visit } from 'unist-util-visit';

/**
 * Turns GFM footnotes (`text[^1]` ... `[^1]: note`) into Tufte-style margin
 * sidenotes instead of end-of-document endnotes (SPEC §5: "Margin sidenotes
 * that flow inline below the paragraph on narrow viewports... Expose as an
 * MDX component <Sidenote> and a markdown-friendly footnote fallback.").
 *
 * Standard `[^1]` markdown IS the fallback here — this plugin just changes
 * where the definition's content ends up (inline at the reference point, as
 * a `sidenote` mdast node) instead of collected at the bottom of the page.
 * Pair with the `sidenote` remark-rehype handler to turn that node into
 * actual markup, and with the CSS in global.css for the margin/inline
 * placement.
 */
export function remarkSidenotes() {
  return (tree) => {
    const definitions = new Map();

    // Collect definitions and drop them from the tree — their content moves
    // to live inline at the reference site instead.
    const toRemove = [];
    visit(tree, 'footnoteDefinition', (node, index, parent) => {
      definitions.set(node.identifier, node.children);
      toRemove.push({ parent, index });
    });
    for (const { parent, index } of toRemove.reverse()) {
      parent.children.splice(index, 1);
    }

    let counter = 0;
    visit(tree, 'footnoteReference', (node, index, parent) => {
      let children = definitions.get(node.identifier);
      if (!children) return; // dangling reference — leave remark-rehype's default handling
      // A footnote definition is almost always a single paragraph. Unwrap it
      // so the sidenote's content flows inline instead of nesting a <p>
      // inside the <span> the rehype handler emits.
      if (children.length === 1 && children[0].type === 'paragraph') {
        children = children[0].children;
      }
      counter += 1;
      parent.children[index] = {
        type: 'sidenote',
        data: { number: counter },
        children,
      };
    });
  };
}
