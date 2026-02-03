'use client'

import { useState, useEffect } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useTheme } from 'next-themes'

type CodeBlockProps = {
  code: string
  language?: string
  filename?: string
}

/**
 * Syntax-highlighted code block using Prism via react-syntax-highlighter
 *
 * Features:
 * - Line numbers (fainter, separated from code)
 * - Theme-aware (dark/light mode)
 * - Optional filename header
 */
export function CodeBlock({
  code,
  language = 'python',
  filename,
}: CodeBlockProps) {
  const { resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Default to dark until mounted to avoid hydration mismatch
  const isDark = !mounted || resolvedTheme === 'dark'

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      {filename && (
        <div className="px-4 py-2 border-b border-border bg-muted/30 text-xs text-muted-foreground font-mono">
          {filename}
        </div>
      )}
      <div className="text-xs overflow-x-auto">
        <SyntaxHighlighter
          language={language}
          style={isDark ? oneDark : oneLight}
          showLineNumbers
          lineNumberStyle={{
            minWidth: '2.5em',
            paddingRight: '1em',
            color: isDark ? 'rgb(156 163 175 / 0.5)' : 'rgb(107 114 128 / 0.5)',
            userSelect: 'none',
          }}
          customStyle={{
            margin: 0,
            padding: '0.5rem',
            background: 'transparent',
            fontSize: 'inherit',
          }}
          codeTagProps={{
            style: {
              fontFamily: 'inherit',
            },
          }}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  )
}
