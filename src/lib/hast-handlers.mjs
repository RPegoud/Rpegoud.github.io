/**
 * remark-rehype `handlers` for the custom mdast node types produced by
 * remark-sidenotes.mjs and remark-figures.mjs. See those files for what
 * produces these nodes and why.
 */
export const hastHandlers = {
  figure(state, node) {
    return {
      type: 'element',
      tagName: 'figure',
      properties: { className: ['figure', `figure-${node.data.width}`] },
      children: state.all(node),
    };
  },
  figureCaption(state, node) {
    return {
      type: 'element',
      tagName: 'figcaption',
      properties: {},
      children: state.all(node),
    };
  },
  sidenote(state, node) {
    const number = String(node.data.number);
    return {
      type: 'element',
      tagName: 'span',
      properties: { className: ['sidenote-wrapper'] },
      children: [
        {
          type: 'element',
          tagName: 'sup',
          properties: { className: ['sidenote-marker'] },
          children: [{ type: 'text', value: number }],
        },
        {
          type: 'element',
          tagName: 'span',
          properties: { className: ['sidenote-content'] },
          children: [
            {
              type: 'element',
              tagName: 'span',
              properties: { className: ['sidenote-number'] },
              children: [{ type: 'text', value: number }],
            },
            { type: 'text', value: ' ' },
            ...state.all(node),
          ],
        },
      ],
    };
  },
};
