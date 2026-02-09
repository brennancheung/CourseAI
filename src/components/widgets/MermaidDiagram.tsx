'use client'

import { useEffect, useId, useRef, useState } from 'react'

interface MermaidDiagramProps {
  chart: string
  className?: string
}

export function MermaidDiagram({ chart, className }: MermaidDiagramProps) {
  const id = useId().replace(/:/g, '-')
  const containerRef = useRef<HTMLDivElement>(null)
  const [svg, setSvg] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function render() {
      try {
        const mermaid = (await import('mermaid')).default

        mermaid.initialize({
          startOnLoad: false,
          theme: 'dark',
          darkMode: true,
          fontFamily: 'ui-sans-serif, system-ui, sans-serif',
          themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#e2e8f0',
            primaryBorderColor: '#4f46e5',
            lineColor: '#64748b',
            secondaryColor: '#1e293b',
            tertiaryColor: '#0f172a',
            background: '#0f172a',
            mainBkg: '#1e293b',
            nodeBorder: '#475569',
            clusterBkg: '#1e293b',
            titleColor: '#e2e8f0',
            edgeLabelBackground: '#1e293b',
          },
        })

        const diagramId = `mermaid-${id}`
        const { svg: renderedSvg } = await mermaid.render(diagramId, chart.trim())

        if (!cancelled) {
          setSvg(renderedSvg)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to render diagram')
          setSvg('')
        }
      }
    }

    render()

    return () => {
      cancelled = true
    }
  }, [chart, id])

  if (error) {
    return (
      <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-400">
        <p className="font-medium">Diagram render error</p>
        <pre className="mt-2 text-xs whitespace-pre-wrap">{error}</pre>
      </div>
    )
  }

  if (!svg) {
    return (
      <div className="flex items-center justify-center rounded-lg border bg-card p-8">
        <div className="text-sm text-muted-foreground">Loading diagram...</div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={className}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
