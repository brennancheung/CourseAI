'use client'

import { useState, useEffect } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useTheme } from 'next-themes'
import { ClipboardCopy } from '@/components/common/ClipboardCopy'

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
 * - Copy to clipboard button
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
    <div className="group relative rounded-lg border border-border overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <span className="text-xs text-muted-foreground font-mono">
          {filename ?? language}
        </span>
        <ClipboardCopy
          text={code.trim()}
          className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
        />
      </div>
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
