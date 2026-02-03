'use client'

import { useEffect, useState } from 'react'
import { codeToHtml } from 'shiki'

type CodeBlockProps = {
  code: string
  language?: string
  filename?: string
}

/**
 * Syntax-highlighted code block using Shiki
 *
 * Uses VS Code's TextMate grammars for accurate highlighting.
 * Renders asynchronously since Shiki loads grammars on demand.
 */
export function CodeBlock({
  code,
  language = 'python',
  filename,
}: CodeBlockProps) {
  const [html, setHtml] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function highlight() {
      try {
        const result = await codeToHtml(code.trim(), {
          lang: language,
          theme: 'github-dark',
        })
        if (!cancelled) {
          setHtml(result)
          setIsLoading(false)
        }
      } catch (err) {
        console.error('Shiki highlighting failed:', err)
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    highlight()
    return () => {
      cancelled = true
    }
  }, [code, language])

  return (
    <div className="rounded-lg border border-border overflow-hidden bg-[#0d1117]">
      {filename && (
        <div className="px-4 py-2 border-b border-border bg-muted/30 text-xs text-muted-foreground font-mono">
          {filename}
        </div>
      )}
      <div className="overflow-x-auto">
        {isLoading ? (
          <pre className="p-4 font-mono text-sm text-muted-foreground">
            <code>{code.trim()}</code>
          </pre>
        ) : (
          <div
            className="[&>pre]:p-4 [&>pre]:m-0 [&>pre]:bg-transparent [&>pre]:text-sm [&>pre]:leading-relaxed [&_code]:font-mono"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}
      </div>
    </div>
  )
}
