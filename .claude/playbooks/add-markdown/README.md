# Add Markdown Rendering

Adds markdown rendering with GitHub Flavored Markdown (GFM), math equations (KaTeX), and syntax-highlighted code blocks.

## Features

- GitHub Flavored Markdown (tables, strikethrough, task lists, etc.)
- Math equations with KaTeX (`$inline$` and `$$block$$`)
- Syntax-highlighted code blocks with copy/download buttons
- Dark/light theme support
- Line wrapping toggle for code blocks

## Prerequisites

- Next.js project created from nextjs-ventures-starter
- `src/components/` directory exists
- `@/lib/utils` with `cn()` utility (included in starter)
- `@/components/ui/button` (included in starter)

## Usage

Run from the project root:

```bash
bash .claude/playbooks/add-markdown/install.sh
```

## What It Installs

### Dependencies

```
react-markdown
remark-gfm
remark-math
rehype-katex
katex
react-syntax-highlighter
@types/react-syntax-highlighter (dev)
@tailwindcss/typography (dev)
```

### Components

- `src/components/markdown/Markdown.tsx` — Main component
- `src/components/markdown/MarkdownComponents.tsx` — Custom renderers
- `src/components/markdown/CodeBlock.tsx` — Syntax-highlighted code block

## Example

```tsx
import { Markdown } from '@/components/markdown/Markdown'

export function MyComponent() {
  const content = `
# Hello World

This is **bold** and this is *italic*.

\`\`\`typescript
const greeting = "Hello, world!"
console.log(greeting)
\`\`\`

Math: $E = mc^2$
`

  return <Markdown content={content} />
}
```

## Customization

The `Markdown` component accepts a `className` prop for custom styling:

```tsx
<Markdown content={content} className="text-sm" />
```

To modify the default styles, edit `MarkdownComponents.tsx`.
