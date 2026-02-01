'use client'

import { useState, memo, useMemo, useEffect } from 'react'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import { useTheme } from 'next-themes'
import { Check, Copy, Download, WrapText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

import js from 'react-syntax-highlighter/dist/esm/languages/hljs/javascript'
import ts from 'react-syntax-highlighter/dist/esm/languages/hljs/typescript'
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python'
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json'
import css from 'react-syntax-highlighter/dist/esm/languages/hljs/css'
import xml from 'react-syntax-highlighter/dist/esm/languages/hljs/xml'
import bash from 'react-syntax-highlighter/dist/esm/languages/hljs/bash'
import yaml from 'react-syntax-highlighter/dist/esm/languages/hljs/yaml'
import markdown from 'react-syntax-highlighter/dist/esm/languages/hljs/markdown'
import sql from 'react-syntax-highlighter/dist/esm/languages/hljs/sql'

SyntaxHighlighter.registerLanguage('javascript', js)
SyntaxHighlighter.registerLanguage('typescript', ts)
SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('css', css)
SyntaxHighlighter.registerLanguage('xml', xml)
SyntaxHighlighter.registerLanguage('html', xml)
SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('shell', bash)
SyntaxHighlighter.registerLanguage('yaml', yaml)
SyntaxHighlighter.registerLanguage('yml', yaml)
SyntaxHighlighter.registerLanguage('markdown', markdown)
SyntaxHighlighter.registerLanguage('md', markdown)
SyntaxHighlighter.registerLanguage('sql', sql)

const getStyles = async (isDark: boolean) => {
  if (isDark) {
    const { default: vscDarkPlus } = await import('react-syntax-highlighter/dist/esm/styles/hljs/vs2015')
    return vscDarkPlus
  }
  const { default: github } = await import('react-syntax-highlighter/dist/esm/styles/hljs/github')
  return github
}

interface CodeBlockProps {
  language?: string
  children: string
  className?: string
}

export const CodeBlock = memo(function CodeBlock({ language, children, className }: CodeBlockProps) {
  const { theme } = useTheme()
  const [copied, setCopied] = useState(false)
  const [lineWrap, setLineWrap] = useState(false)
  const [syntaxStyle, setSyntaxStyle] = useState<Record<string, React.CSSProperties>>({})

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleDownload = () => {
    const blob = new Blob([children], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `code.${language || 'txt'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleToggleWrap = () => setLineWrap(!lineWrap)

  useEffect(() => {
    getStyles(theme === 'dark').then(setSyntaxStyle)
  }, [theme])

  const customStyle = useMemo(() => ({
    margin: 0,
    padding: '1rem',
    fontSize: '0.875rem',
    lineHeight: '1.5',
  }), [])

  const codeTagStyle = useMemo(() => ({
    fontSize: 'inherit',
    lineHeight: 'inherit',
  }), [])

  return (
    <div className={cn('group relative my-4 overflow-hidden rounded-lg border', className)}>
      <div className="flex items-center justify-between border-b bg-muted/50 px-4 py-2">
        <span className="text-xs font-medium text-muted-foreground">
          {language || 'plaintext'}
        </span>
        <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleToggleWrap}
            title={lineWrap ? 'Disable line wrap' : 'Enable line wrap'}
          >
            <WrapText className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleDownload}
            title="Download"
          >
            <Download className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleCopy}
            title={copied ? 'Copied!' : 'Copy to clipboard'}
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-green-600" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>

      <div className={cn('overflow-x-auto bg-muted/30 dark:bg-muted/50', lineWrap && 'whitespace-pre-wrap')}>
        <SyntaxHighlighter
          language={language}
          style={syntaxStyle}
          customStyle={customStyle}
          codeTagProps={{ style: codeTagStyle }}
          wrapLongLines={lineWrap}
        >
          {children}
        </SyntaxHighlighter>
      </div>
    </div>
  )
})
