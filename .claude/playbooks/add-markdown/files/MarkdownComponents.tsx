import type { Components } from 'react-markdown'
import { memo } from 'react'
import { CodeBlock } from './CodeBlock'

const InlineCode = memo(function InlineCode({ children, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <code
      className="rounded bg-gray-100 px-1 py-0.5 text-sm dark:bg-gray-800"
      {...props}
    >
      {children}
    </code>
  )
})

export const markdownComponents: Components = {
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    const isInline = !match

    if (isInline) return <InlineCode {...props}>{children}</InlineCode>

    const language = match ? match[1] : undefined
    const codeContent = String(children).replace(/\n$/, '')

    return (
      <CodeBlock language={language} className="my-2">
        {codeContent}
      </CodeBlock>
    )
  },
  pre({ children }) {
    return <>{children}</>
  },
  ul({ children }) {
    return <ul className="my-2 list-disc pl-6">{children}</ul>
  },
  ol({ children }) {
    return <ol className="my-2 list-decimal pl-6">{children}</ol>
  },
  blockquote({ children }) {
    return (
      <blockquote className="my-2 border-l-4 border-gray-300 pl-4 italic dark:border-gray-600">
        {children}
      </blockquote>
    )
  },
  h1({ children }) {
    return <h1 className="mb-2 mt-4 text-xl font-bold">{children}</h1>
  },
  h2({ children }) {
    return <h2 className="mb-2 mt-3 text-lg font-semibold">{children}</h2>
  },
  h3({ children }) {
    return <h3 className="mb-1 mt-2 text-base font-semibold">{children}</h3>
  },
  p({ children }) {
    return <p className="my-2">{children}</p>
  },
  a({ href, children }) {
    return (
      <a
        href={href}
        className="text-blue-600 underline hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    )
  },
}
