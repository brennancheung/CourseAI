'use client'

import { useState } from 'react'

// ─── Grid ───────────────────────────────────────────────────────────
const COLS = 8 // channels
const ROWS = 3 // batch examples
const CELL = 46
const GAP = 3

const GRID_W = COLS * CELL + (COLS - 1) * GAP
const GRID_H = ROWS * CELL + (ROWS - 1) * GAP
const LABEL_W = 74
const GRID_X = LABEL_W + 6
const CH_Y = 14
const GRID_Y = 28
const SVG_W = GRID_X + GRID_W + 10
const SVG_H = GRID_Y + GRID_H + 8

// ─── Colors ─────────────────────────────────────────────────────────
const C = {
  batch: '#f59e0b',
  layer: '#818cf8',
  group: '#22c55e',
  base: '#64748b',
}

// ─── Types ──────────────────────────────────────────────────────────
interface CellPos { row: number; col: number }

// For a given cell, determine its relationship to the selected cell
// under each normalization type. The categories are mutually exclusive.
type CellCategory = 'selected' | 'batch' | 'layer' | 'group' | 'inactive'

function categorize(
  row: number, col: number, sel: CellPos, numGroups: number,
): CellCategory {
  if (row === sel.row && col === sel.col) return 'selected'

  const gs = COLS / numGroups
  const sameGroup = Math.floor(col / gs) === Math.floor(sel.col / gs)

  // Same row + same group = group norm peer (takes precedence over layer)
  if (row === sel.row && sameGroup) return 'group'
  // Same row + different group = layer norm peer only
  if (row === sel.row) return 'layer'
  // Same column = batch norm peer
  if (col === sel.col) return 'batch'

  return 'inactive'
}

const FILLS: Record<CellCategory, string> = {
  selected: '#ffffff',
  batch: C.batch,
  layer: C.layer,
  group: C.group,
  inactive: C.base,
}

const OPACITIES: Record<CellCategory, number> = {
  selected: 0.95,
  batch: 0.65,
  layer: 0.50,
  group: 0.65,
  inactive: 0.1,
}

// ─── Description builder ────────────────────────────────────────────
// Always returns 3 rows; dynamic parts swap when a cell is active.
function buildDescriptions(sel: CellPos | null, numGroups: number) {
  const gs = COLS / numGroups

  if (!sel) {
    return [
      { color: C.batch, label: 'Batch Norm', detail: `${ROWS} values — per channel, across batch` },
      { color: C.layer, label: 'Layer Norm', detail: `${COLS} values — all channels, per example` },
      { color: C.group, label: 'Group Norm', detail: `${gs} values — per channel group, per example` },
    ]
  }

  const g = Math.floor(sel.col / gs)
  const start = g * gs
  const end = start + gs - 1
  const rangeStr = start === end ? `C${start}` : `C${start}–C${end}`

  return [
    { color: C.batch, label: 'Batch Norm', detail: `${ROWS} values — C${sel.col} across all examples` },
    { color: C.layer, label: 'Layer Norm', detail: `${COLS} values — all channels in Example ${sel.row}` },
    { color: C.group, label: 'Group Norm', detail: `${gs} value${gs > 1 ? 's' : ''} — ${rangeStr} in Example ${sel.row}` },
  ]
}

// ─── Component ──────────────────────────────────────────────────────
export function NormalizationComparisonWidget() {
  const [locked, setLocked] = useState<CellPos | null>(null)
  const [hovered, setHovered] = useState<CellPos | null>(null)
  const [numGroups, setNumGroups] = useState(2)

  // Hover previews; click locks; mouse-out snaps back to locked cell
  const active = hovered ?? locked
  const groupSize = COLS / numGroups

  function handleClick(row: number, col: number) {
    if (locked?.row === row && locked?.col === col) {
      setLocked(null)
      return
    }
    setLocked({ row, col })
  }

  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full"
        style={{ maxWidth: 580 }}
        onMouseLeave={() => setHovered(null)}
        role="img"
        aria-label="Interactive normalization comparison — click a cell to see how each norm groups it"
      >
        {/* Channel labels */}
        {Array.from({ length: COLS }, (_, c) => (
          <text
            key={c}
            x={GRID_X + c * (CELL + GAP) + CELL / 2}
            y={CH_Y}
            textAnchor="middle"
            fill="currentColor"
            opacity={0.5}
            fontSize={11}
          >
            C{c}
          </text>
        ))}

        <g transform={`translate(0, ${GRID_Y})`}>
          {/* Example labels */}
          {Array.from({ length: ROWS }, (_, r) => (
            <text
              key={r}
              x={LABEL_W}
              y={r * (CELL + GAP) + CELL / 2 + 4}
              textAnchor="end"
              fill="currentColor"
              opacity={0.5}
              fontSize={11}
            >
              Example {r}
            </text>
          ))}

          {/* Group dividers (always visible) */}
          {numGroups > 1 &&
            Array.from({ length: numGroups - 1 }, (_, i) => {
              const col = (i + 1) * groupSize
              const x = GRID_X + col * (CELL + GAP) - GAP / 2
              return (
                <line
                  key={i}
                  x1={x} y1={-2} x2={x} y2={GRID_H + 2}
                  className="stroke-stone-400 dark:stroke-stone-500"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
              )
            })}

          {/* Cells */}
          {Array.from({ length: ROWS }, (_, r) =>
            Array.from({ length: COLS }, (_, c) => {
              const cat = active ? categorize(r, c, active, numGroups) : null
              const fill = cat ? FILLS[cat] : C.base
              const opacity = cat ? OPACITIES[cat] : 0.3

              return (
                <rect
                  key={`${r}-${c}`}
                  x={GRID_X + c * (CELL + GAP)}
                  y={r * (CELL + GAP)}
                  width={CELL}
                  height={CELL}
                  rx={4}
                  fill={fill}
                  opacity={opacity}
                  className="cursor-pointer"
                  style={{ transition: 'fill 200ms, opacity 200ms' }}
                  onClick={() => handleClick(r, c)}
                  onMouseEnter={() => setHovered({ row: r, col: c })}
                  onMouseLeave={() => setHovered(null)}
                />
              )
            }),
          )}

          {/* Norm outlines — only when a cell is active */}
          {active && (
            <>
              {/* Batch norm: column (amber) */}
              <rect
                x={GRID_X + active.col * (CELL + GAP) - 3}
                y={-3}
                width={CELL + 6}
                height={GRID_H + 6}
                rx={6}
                fill="none"
                stroke={C.batch}
                strokeWidth={2}
                strokeOpacity={0.7}
              />

              {/* Layer norm: full row (indigo, subtle) */}
              <rect
                x={GRID_X - 3}
                y={active.row * (CELL + GAP) - 3}
                width={GRID_W + 6}
                height={CELL + 6}
                rx={6}
                fill="none"
                stroke={C.layer}
                strokeWidth={1.5}
                strokeOpacity={0.3}
              />

              {/* Group norm: row segment (green, strongest) */}
              {(() => {
                const g = Math.floor(active.col / groupSize)
                const startCol = g * groupSize
                return (
                  <rect
                    x={GRID_X + startCol * (CELL + GAP) - 3}
                    y={active.row * (CELL + GAP) - 3}
                    width={groupSize * (CELL + GAP) - GAP + 6}
                    height={CELL + 6}
                    rx={6}
                    fill="none"
                    stroke={C.group}
                    strokeWidth={2.5}
                    strokeOpacity={0.9}
                    pointerEvents="none"
                  />
                )
              })()}

              {/* Selected cell ring */}
              <rect
                x={GRID_X + active.col * (CELL + GAP)}
                y={active.row * (CELL + GAP)}
                width={CELL}
                height={CELL}
                rx={4}
                fill="none"
                stroke="#ffffff"
                strokeWidth={2.5}
                pointerEvents="none"
              />
            </>
          )}
        </g>
      </svg>

      {/* Legend — always present, dynamic parts swap on hover/click */}
      <div className="space-y-1">
        {buildDescriptions(active, numGroups).map((d) => (
          <div key={d.label} className="flex items-center gap-2.5 text-sm">
            <span
              className="inline-block w-3 h-3 rounded-sm shrink-0"
              style={{ backgroundColor: d.color }}
            />
            <span className="font-medium text-foreground w-[5.5rem]">{d.label}</span>
            <span className="text-muted-foreground">{d.detail}</span>
          </div>
        ))}
      </div>

      {/* Group count selector */}
      <div className="flex items-center justify-center gap-2 text-sm">
        <span className="text-muted-foreground">Groups:</span>
        {[1, 2, 4, 8].map((n) => (
          <button
            key={n}
            onClick={() => setNumGroups(n)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              numGroups === n
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:text-foreground'
            }`}
          >
            {n}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground text-center">
        1 group = Layer Norm. {COLS} groups = each channel alone (Instance Norm).
      </p>
    </div>
  )
}
