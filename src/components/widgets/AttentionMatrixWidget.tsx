'use client'

import { useMemo, useState, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * AttentionMatrixWidget
 *
 * Interactive heatmap showing raw dot-product attention weights.
 * Student types a sentence, sees the attention weight matrix (tokens on both axes).
 * Color intensity = attention weight. Hover shows exact value.
 *
 * Key pedagogical purpose: make SYMMETRY visible. Raw dot-product attention
 * (without Q/K projections) produces a symmetric matrix because dot(a,b) = dot(b,a).
 * The student should discover this property by exploring the widget.
 *
 * Does NOT use Q, K, V terminology. Uses raw embedding dot products only.
 */

// ---------------------------------------------------------------------------
// Preset sentences for exploration
// ---------------------------------------------------------------------------

const PRESET_SENTENCES: { label: string; text: string }[] = [
  { label: 'Bank (river)', text: 'The bank was steep and muddy' },
  { label: 'Bank (money)', text: 'The bank raised interest rates' },
  { label: 'Cat chased mouse', text: 'The cat chased the mouse' },
  { label: 'Dog bites man', text: 'Dog bites man' },
  { label: 'Man bites dog', text: 'Man bites dog' },
]

// ---------------------------------------------------------------------------
// Simulated embedding vectors
// We use deterministic pseudo-embeddings derived from token characters.
// This gives consistent, reproducible results that show meaningful patterns.
// ---------------------------------------------------------------------------

function hashToken(token: string): number[] {
  const dims = 16
  const vec: number[] = []
  for (let d = 0; d < dims; d++) {
    let val = 0
    for (let i = 0; i < token.length; i++) {
      // Mix character code with dimension to get varied values
      val += Math.sin((token.charCodeAt(i) * (d + 1) * 0.37) + (d * 2.1)) * Math.cos((token.charCodeAt(i) * 0.73) + (i * d * 0.19))
    }
    vec.push(val / Math.max(token.length, 1))
  }
  return vec
}

// Semantic clusters: words that should have similar embeddings get a shared base
const SEMANTIC_GROUPS: Record<string, number[]> = {
  // Articles / function words
  the: [0.9, 0.1, -0.2, 0.5, -0.1, 0.3, -0.4, 0.2, 0.1, -0.3, 0.4, -0.1, 0.2, 0.3, -0.2, 0.1],
  a: [0.85, 0.15, -0.18, 0.45, -0.12, 0.28, -0.38, 0.22, 0.08, -0.28, 0.38, -0.08, 0.18, 0.28, -0.18, 0.12],
  an: [0.87, 0.12, -0.19, 0.47, -0.11, 0.29, -0.39, 0.21, 0.09, -0.29, 0.39, -0.09, 0.19, 0.29, -0.19, 0.11],
  // Verbs of action
  chased: [0.1, 0.8, 0.6, -0.3, 0.7, -0.1, 0.2, 0.5, -0.4, 0.3, -0.2, 0.6, -0.1, 0.4, 0.3, -0.5],
  bites: [0.15, 0.75, 0.55, -0.25, 0.65, -0.05, 0.25, 0.45, -0.35, 0.25, -0.15, 0.55, -0.05, 0.35, 0.25, -0.45],
  raised: [0.2, 0.7, 0.3, -0.1, 0.5, 0.2, 0.1, 0.6, -0.2, 0.4, -0.3, 0.4, 0.1, 0.3, 0.4, -0.3],
  sat: [0.05, 0.6, 0.4, -0.2, 0.5, 0.0, 0.15, 0.35, -0.3, 0.2, -0.1, 0.45, -0.15, 0.25, 0.2, -0.35],
  was: [0.7, 0.3, -0.1, 0.3, 0.0, 0.2, -0.2, 0.3, 0.0, -0.2, 0.3, 0.0, 0.15, 0.2, -0.1, 0.05],
  // Animals
  cat: [-0.3, 0.2, 0.8, 0.6, -0.1, 0.4, 0.5, -0.2, 0.3, 0.7, -0.4, 0.1, 0.6, -0.3, 0.2, 0.4],
  dog: [-0.25, 0.25, 0.75, 0.55, -0.15, 0.35, 0.45, -0.15, 0.35, 0.65, -0.35, 0.15, 0.55, -0.25, 0.25, 0.35],
  mouse: [-0.2, 0.15, 0.7, 0.5, -0.05, 0.3, 0.4, -0.1, 0.25, 0.6, -0.3, 0.05, 0.5, -0.2, 0.15, 0.3],
  man: [-0.1, 0.3, 0.6, 0.4, 0.1, 0.5, 0.3, -0.3, 0.4, 0.5, -0.2, 0.2, 0.4, -0.1, 0.3, 0.2],
  // Financial
  bank: [0.3, -0.2, 0.4, 0.1, 0.6, -0.3, 0.5, 0.2, -0.1, 0.3, 0.4, -0.5, 0.2, 0.6, -0.1, 0.3],
  interest: [0.4, -0.1, 0.2, 0.0, 0.7, -0.2, 0.4, 0.3, -0.2, 0.2, 0.5, -0.4, 0.3, 0.5, 0.0, 0.2],
  rates: [0.35, -0.15, 0.15, -0.05, 0.65, -0.15, 0.35, 0.25, -0.15, 0.15, 0.45, -0.35, 0.25, 0.45, 0.05, 0.15],
  // Terrain
  steep: [-0.4, 0.1, 0.3, 0.7, -0.2, 0.6, -0.1, 0.4, 0.5, -0.1, 0.2, 0.3, -0.4, 0.1, 0.5, 0.6],
  muddy: [-0.35, 0.05, 0.25, 0.65, -0.15, 0.55, -0.05, 0.35, 0.45, -0.05, 0.15, 0.25, -0.35, 0.05, 0.45, 0.55],
  and: [0.8, 0.2, -0.1, 0.4, -0.05, 0.25, -0.3, 0.15, 0.05, -0.25, 0.35, -0.05, 0.15, 0.25, -0.15, 0.08],
  on: [0.75, 0.18, -0.12, 0.35, -0.08, 0.22, -0.28, 0.18, 0.03, -0.22, 0.32, -0.03, 0.12, 0.22, -0.12, 0.06],
}

function getEmbedding(token: string): number[] {
  const lower = token.toLowerCase()
  const known = SEMANTIC_GROUPS[lower]
  if (known) return known
  return hashToken(lower)
}

// ---------------------------------------------------------------------------
// Attention computation: raw dot products -> softmax -> weights
// ---------------------------------------------------------------------------

function dotProduct(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return sum
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

function computeAttentionMatrix(tokens: string[]): {
  scores: number[][]
  weights: number[][]
} {
  const embeddings = tokens.map(getEmbedding)
  const n = tokens.length

  // Compute raw dot-product scores (XX^T)
  const scores: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      row.push(dotProduct(embeddings[i], embeddings[j]))
    }
    scores.push(row)
  }

  // Apply softmax to each row to get attention weights
  const weights: number[][] = scores.map((row) => softmax(row))

  return { scores, weights }
}

// ---------------------------------------------------------------------------
// Color interpolation for attention weights (0 to 1)
// Dark/transparent (low attention) -> bright violet (high attention)
// ---------------------------------------------------------------------------

function weightToColor(w: number): string {
  // 0 = dark background, 1 = bright violet
  const clamped = Math.max(0, Math.min(1, w))
  const r = Math.round(30 + clamped * (139 - 30))
  const g = Math.round(30 + clamped * (92 - 30))
  const b = Math.round(46 + clamped * (246 - 46))
  return `rgb(${r},${g},${b})`
}

function weightToTextColor(w: number): string {
  if (w > 0.3) return 'rgb(255, 255, 255)'
  return 'rgb(160, 160, 180)'
}

// ---------------------------------------------------------------------------
// Tokenize simply by splitting on whitespace
// ---------------------------------------------------------------------------

function tokenize(text: string): string[] {
  return text.trim().split(/\s+/).filter((t) => t.length > 0)
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type AttentionMatrixWidgetProps = {
  width?: number
  height?: number
}

export function AttentionMatrixWidget({ width: widthOverride }: AttentionMatrixWidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(500)
  const width = widthOverride ?? measuredWidth

  const [inputText, setInputText] = useState('The cat chased the mouse')
  const [hoveredCell, setHoveredCell] = useState<{
    row: number
    col: number
    weight: number
    score: number
  } | null>(null)
  const [showScores, setShowScores] = useState(false)

  const tokens = useMemo(() => tokenize(inputText), [inputText])
  const { scores, weights } = useMemo(() => {
    if (tokens.length === 0) return { scores: [] as number[][], weights: [] as number[][] }
    return computeAttentionMatrix(tokens)
  }, [tokens])

  // Check symmetry: compare weight[i][j] vs weight[j][i]
  const isSymmetric = useMemo(() => {
    if (tokens.length < 2) return true
    for (let i = 0; i < tokens.length; i++) {
      for (let j = i + 1; j < tokens.length; j++) {
        if (Math.abs(scores[i][j] - scores[j][i]) > 0.001) return false
      }
    }
    return true
  }, [tokens.length, scores])

  // Layout
  const padding = 16
  const labelSpace = Math.min(80, Math.max(50, tokens.length > 0 ? Math.max(...tokens.map((t) => t.length)) * 8 : 50))
  const availableSize = width - labelSpace - padding * 2
  const maxCellSize = 64
  const minCellSize = 28
  const cellSize = tokens.length > 0
    ? Math.max(minCellSize, Math.min(maxCellSize, Math.floor(availableSize / tokens.length)))
    : maxCellSize
  const handleCellHover = useCallback((row: number, col: number, weight: number, score: number) => {
    setHoveredCell({ row, col, weight, score })
  }, [])

  const handleMouseLeave = useCallback(() => {
    setHoveredCell(null)
  }, [])

  const handlePreset = useCallback((text: string) => {
    setInputText(text)
    setHoveredCell(null)
  }, [])

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Input area */}
      <div className="space-y-2">
        <label className="text-xs text-muted-foreground/70 block">
          Type a sentence to see its attention pattern:
        </label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value)
            setHoveredCell(null)
          }}
          className="w-full px-3 py-2 rounded-md bg-muted/50 border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          placeholder="Type a sentence..."
        />
        <div className="flex flex-wrap gap-1.5">
          {PRESET_SENTENCES.map((preset) => (
            <button
              key={preset.label}
              onClick={() => handlePreset(preset.text)}
              className="px-2 py-1 rounded text-xs bg-muted/50 hover:bg-muted text-muted-foreground/70 hover:text-muted-foreground transition-colors cursor-pointer"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Toggle: weights vs raw scores */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setShowScores(false)}
          className={`px-3 py-1 rounded text-xs transition-colors cursor-pointer ${
            !showScores
              ? 'bg-violet-500/20 text-violet-400 font-medium'
              : 'bg-muted/50 text-muted-foreground/70 hover:text-muted-foreground'
          }`}
        >
          Attention Weights (after softmax)
        </button>
        <button
          onClick={() => setShowScores(true)}
          className={`px-3 py-1 rounded text-xs transition-colors cursor-pointer ${
            showScores
              ? 'bg-violet-500/20 text-violet-400 font-medium'
              : 'bg-muted/50 text-muted-foreground/70 hover:text-muted-foreground'
          }`}
        >
          Raw Dot Products (before softmax)
        </button>
      </div>

      {/* Hover info */}
      <div className="h-6 text-xs text-muted-foreground/70">
        {hoveredCell ? (
          <span>
            &ldquo;<span className="font-medium text-foreground">{tokens[hoveredCell.row]}</span>&rdquo;
            {' '}&rarr;{' '}
            &ldquo;<span className="font-medium text-foreground">{tokens[hoveredCell.col]}</span>&rdquo;
            {': '}
            {showScores ? (
              <span>
                raw score = <span className="font-mono font-medium text-foreground">{hoveredCell.score.toFixed(3)}</span>
              </span>
            ) : (
              <span>
                weight = <span className="font-mono font-medium text-foreground">{hoveredCell.weight.toFixed(4)}</span>
                {' '}(score: {hoveredCell.score.toFixed(3)})
              </span>
            )}
            {hoveredCell.row !== hoveredCell.col && (
              <span className="ml-2 text-muted-foreground/50">
                | reverse: {showScores
                  ? scores[hoveredCell.col]?.[hoveredCell.row]?.toFixed(3)
                  : weights[hoveredCell.col]?.[hoveredCell.row]?.toFixed(4)
                }
                {!showScores && Math.abs(scores[hoveredCell.row][hoveredCell.col] - scores[hoveredCell.col][hoveredCell.row]) < 0.001 && (
                  <span className="text-amber-400 ml-1">(same score!)</span>
                )}
              </span>
            )}
          </span>
        ) : (
          <span>Hover over the matrix to see attention values. Rows = &ldquo;this token is looking at...&rdquo;, Columns = &ldquo;...this token&rdquo;</span>
        )}
      </div>

      {/* Heatmap */}
      {tokens.length > 0 && (
        <div className="overflow-x-auto">
          <div className="inline-block">
            {/* Column labels (top) */}
            <div className="flex" style={{ marginLeft: labelSpace }}>
              {tokens.map((token, j) => (
                <div
                  key={j}
                  className="text-xs font-mono text-muted-foreground/70 overflow-hidden text-ellipsis"
                  style={{
                    width: cellSize,
                    textAlign: 'center',
                    transform: tokens.length > 6 ? 'rotate(-45deg) translateX(-4px)' : undefined,
                    transformOrigin: 'bottom center',
                    height: tokens.length > 6 ? 40 : 20,
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'center',
                    paddingBottom: 2,
                  }}
                >
                  {token}
                </div>
              ))}
            </div>

            {/* Grid with row labels */}
            <div
              className="relative"
              onMouseLeave={handleMouseLeave}
            >
              {tokens.map((rowToken, i) => (
                <div key={i} className="flex items-center">
                  {/* Row label */}
                  <div
                    className="text-xs font-mono text-muted-foreground/70 text-right pr-2 overflow-hidden text-ellipsis whitespace-nowrap"
                    style={{ width: labelSpace }}
                  >
                    {rowToken}
                  </div>
                  {/* Cells */}
                  {tokens.map((_, j) => {
                    const displayValue = showScores ? scores[i][j] : weights[i][j]
                    const weight = weights[i][j]
                    const score = scores[i][j]
                    const isHovered = hoveredCell?.row === i && hoveredCell?.col === j
                    const isMirror = hoveredCell?.row === j && hoveredCell?.col === i && hoveredCell?.row !== hoveredCell?.col

                    // For raw scores, normalize to 0-1 for coloring
                    const colorWeight = showScores
                      ? (() => {
                          const allScores = scores.flat()
                          const minS = Math.min(...allScores)
                          const maxS = Math.max(...allScores)
                          const range = maxS - minS
                          if (range === 0) return 0.5
                          return (score - minS) / range
                        })()
                      : weight

                    return (
                      <div
                        key={j}
                        className="relative cursor-crosshair transition-all duration-75"
                        style={{
                          width: cellSize,
                          height: cellSize,
                          backgroundColor: weightToColor(colorWeight),
                          outline: isHovered
                            ? '2px solid var(--foreground)'
                            : isMirror
                              ? '2px dashed rgba(251, 191, 36, 0.6)'
                              : 'none',
                          outlineOffset: '-1px',
                          zIndex: isHovered ? 10 : isMirror ? 5 : 0,
                        }}
                        onMouseEnter={() => handleCellHover(i, j, weight, score)}
                      >
                        {/* Show value in cell if cells are big enough */}
                        {cellSize >= 40 && (
                          <div
                            className="absolute inset-0 flex items-center justify-center text-xs font-mono"
                            style={{ color: weightToTextColor(colorWeight), fontSize: cellSize >= 52 ? 11 : 9 }}
                          >
                            {displayValue.toFixed(showScores ? 1 : 2)}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Symmetry indicator */}
      {tokens.length >= 2 && (
        <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded-md ${
          isSymmetric
            ? 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
            : 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
        }`}>
          <span className="font-medium">
            {isSymmetric ? 'Symmetric matrix' : 'Asymmetric matrix'}
          </span>
          <span className="text-muted-foreground">
            {isSymmetric
              ? '- The raw dot-product scores are the same in both directions. Hover a cell to see: the dashed outline shows the mirror cell.'
              : '- Scores differ depending on direction.'
            }
          </span>
        </div>
      )}

      {/* Color legend */}
      <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground/60">
        <span className="font-mono">{showScores ? 'low' : '0'}</span>
        <div
          className="h-3 rounded-sm"
          style={{
            width: 120,
            background: `linear-gradient(to right, ${weightToColor(0)}, ${weightToColor(0.5)}, ${weightToColor(1)})`,
          }}
        />
        <span className="font-mono">{showScores ? 'high' : '1'}</span>
      </div>

      <p className="text-xs text-muted-foreground/50 text-center">
        Each row shows how much one token &ldquo;attends to&rdquo; every other token. Brighter = higher weight.
      </p>
    </div>
  )
}
