'use client'

import { useMemo, useState, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * PositionalEncodingHeatmap
 *
 * Visualizes sinusoidal positional encoding as a heatmap.
 * Rows = positions (0..maxPos-1), Columns = encoding dimensions (0..dims-1).
 * Color: diverging blue (negative) → white (zero) → red (positive).
 *
 * Shows the characteristic pattern: high-frequency waves on the left (low dimension
 * indices) that oscillate rapidly, low-frequency waves on the right (high dimension
 * indices) that change slowly.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MAX_POSITIONS = 20
const NUM_DIMENSIONS = 64
const CELL_HEIGHT = 18

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding computation
// ---------------------------------------------------------------------------

function computePositionalEncoding(maxPos: number, dims: number): number[][] {
  const pe: number[][] = []
  for (let pos = 0; pos < maxPos; pos++) {
    const row: number[] = []
    for (let i = 0; i < dims; i++) {
      const dimIndex = Math.floor(i / 2)
      const denominator = Math.pow(10000, (2 * dimIndex) / dims)
      const value = i % 2 === 0
        ? Math.sin(pos / denominator)
        : Math.cos(pos / denominator)
      row.push(value)
    }
    pe.push(row)
  }
  return pe
}

// ---------------------------------------------------------------------------
// Color interpolation: blue (-1) → white (0) → red (+1)
// ---------------------------------------------------------------------------

function valueToColor(v: number): string {
  const clamped = Math.max(-1, Math.min(1, v))

  if (clamped < 0) {
    // blue to white
    const t = 1 + clamped // 0 at -1, 1 at 0
    const r = Math.round(59 + t * (255 - 59))
    const g = Math.round(130 + t * (255 - 130))
    const b = Math.round(246 + t * (255 - 246))
    return `rgb(${r},${g},${b})`
  }

  // white to red
  const t = clamped // 0 at 0, 1 at 1
  const r = 255
  const g = Math.round(255 - t * (255 - 100))
  const b = Math.round(255 - t * (255 - 100))
  return `rgb(${r},${g},${b})`
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type PositionalEncodingHeatmapProps = {
  width?: number
  height?: number
}

export function PositionalEncodingHeatmap({ width: widthOverride }: PositionalEncodingHeatmapProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(500)
  const width = widthOverride ?? measuredWidth

  const [hoveredCell, setHoveredCell] = useState<{ pos: number; dim: number; value: number } | null>(null)

  const pe = useMemo(() => computePositionalEncoding(MAX_POSITIONS, NUM_DIMENSIONS), [])

  // Layout calculations
  const labelWidth = 32
  const availableWidth = width - labelWidth - 16 // 16 for padding
  const cellWidth = Math.max(2, Math.floor(availableWidth / NUM_DIMENSIONS))
  const gridWidth = cellWidth * NUM_DIMENSIONS
  const totalHeight = CELL_HEIGHT * MAX_POSITIONS

  const handleCellHover = useCallback((pos: number, dim: number, value: number) => {
    setHoveredCell({ pos, dim, value })
  }, [])

  const handleMouseLeave = useCallback(() => {
    setHoveredCell(null)
  }, [])

  return (
    <div ref={containerRef} className="space-y-3">
      {/* Hover info */}
      <div className="h-6 text-xs text-muted-foreground/70">
        {hoveredCell ? (
          <span>
            Position <span className="font-mono font-medium text-foreground">{hoveredCell.pos}</span>,
            Dimension <span className="font-mono font-medium text-foreground">{hoveredCell.dim}</span>
            {' = '}
            <span className="font-mono font-medium text-foreground">{hoveredCell.value.toFixed(4)}</span>
            {' '}
            ({hoveredCell.dim % 2 === 0 ? 'sin' : 'cos'})
          </span>
        ) : (
          <span>Hover over the heatmap to see values</span>
        )}
      </div>

      {/* Heatmap */}
      <div className="overflow-x-auto">
        <div className="inline-flex items-start gap-1">
          {/* Y-axis labels (positions) */}
          <div className="flex flex-col" style={{ width: labelWidth }}>
            {Array.from({ length: MAX_POSITIONS }, (_, pos) => (
              <div
                key={pos}
                className="flex items-center justify-end pr-1 text-xs font-mono text-muted-foreground/60"
                style={{ height: CELL_HEIGHT }}
              >
                {pos}
              </div>
            ))}
          </div>

          {/* Heatmap grid */}
          <div
            className="relative"
            style={{ width: gridWidth, height: totalHeight }}
            onMouseLeave={handleMouseLeave}
          >
            <svg width={gridWidth} height={totalHeight}>
              {pe.map((row, pos) =>
                row.map((value, dim) => (
                  <rect
                    key={`${pos}-${dim}`}
                    x={dim * cellWidth}
                    y={pos * CELL_HEIGHT}
                    width={cellWidth}
                    height={CELL_HEIGHT}
                    fill={valueToColor(value)}
                    stroke={
                      hoveredCell && hoveredCell.pos === pos && hoveredCell.dim === dim
                        ? 'var(--foreground)'
                        : 'none'
                    }
                    strokeWidth={hoveredCell && hoveredCell.pos === pos && hoveredCell.dim === dim ? 1.5 : 0}
                    onMouseEnter={() => handleCellHover(pos, dim, value)}
                    className="cursor-crosshair"
                  />
                ))
              )}
            </svg>
          </div>
        </div>

        {/* X-axis labels */}
        <div className="flex items-start gap-1" style={{ marginLeft: labelWidth + 4 }}>
          <div className="flex justify-between text-xs font-mono text-muted-foreground/60" style={{ width: gridWidth }}>
            <span>dim 0</span>
            <span>dim {NUM_DIMENSIONS - 1}</span>
          </div>
        </div>
      </div>

      {/* Annotations */}
      <div className="flex justify-between text-xs text-muted-foreground/60 px-8">
        <div className="flex items-center gap-1.5">
          <span className="text-red-400">&larr;</span>
          <span>Changes rapidly (fine position)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span>Changes slowly (coarse position)</span>
          <span className="text-blue-400">&rarr;</span>
        </div>
      </div>

      {/* Color legend */}
      <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground/60">
        <span className="font-mono">-1</span>
        <div
          className="h-3 rounded-sm"
          style={{
            width: 120,
            background: 'linear-gradient(to right, rgb(59,130,246), rgb(255,255,255), rgb(255,100,100))',
          }}
        />
        <span className="font-mono">+1</span>
      </div>

      <p className="text-xs text-muted-foreground/50 text-center">
        Each row is a unique encoding for that position. Nearby rows (positions) have similar patterns.
      </p>
    </div>
  )
}
