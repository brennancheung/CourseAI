'use client'

import { useState, useMemo, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Point = { x: number; y: number; label: 0 | 1 }

type Mode = 'discriminative' | 'generative'

// ---------------------------------------------------------------------------
// Constants: two Gaussian clusters (class 0 = blue, class 1 = orange)
// ---------------------------------------------------------------------------

const CLUSTER_A = { cx: -1.5, cy: 0.8, sx: 0.6, sy: 0.7, label: 0 as const }
const CLUSTER_B = { cx: 1.5, cy: -0.5, sx: 0.7, sy: 0.6, label: 1 as const }

// Seeded pseudo-random number generator for deterministic data
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function boxMullerFromRng(rng: () => number): [number, number] {
  const u1 = rng()
  const u2 = rng()
  const mag = Math.sqrt(-2.0 * Math.log(u1 + 1e-10))
  return [mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)]
}

type Cluster = { cx: number; cy: number; sx: number; sy: number; label: 0 | 1 }

function generateClusterPoints(
  cluster: Cluster,
  count: number,
  rng: () => number,
): Point[] {
  const points: Point[] = []
  for (let i = 0; i < count; i++) {
    const [z1, z2] = boxMullerFromRng(rng)
    points.push({
      x: cluster.cx + z1 * cluster.sx,
      y: cluster.cy + z2 * cluster.sy,
      label: cluster.label,
    })
  }
  return points
}

const SEED_RNG = mulberry32(42)
const TRAINING_DATA: Point[] = [
  ...generateClusterPoints(CLUSTER_A, 40, SEED_RNG),
  ...generateClusterPoints(CLUSTER_B, 40, SEED_RNG),
]

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

// Gaussian density (unnormalized for heatmap)
function gaussianDensity(
  x: number,
  y: number,
  cx: number,
  cy: number,
  sx: number,
  sy: number,
): number {
  const dx = (x - cx) / sx
  const dy = (y - cy) / sy
  return Math.exp(-0.5 * (dx * dx + dy * dy))
}

// Linear decision boundary for these two clusters
// The boundary bisects the line between the two cluster centers
function decisionBoundaryY(xVal: number): number {
  // Midpoint between cluster centers
  const mx = (CLUSTER_A.cx + CLUSTER_B.cx) / 2
  const my = (CLUSTER_A.cy + CLUSTER_B.cy) / 2
  // Direction vector between centers
  const dx = CLUSTER_B.cx - CLUSTER_A.cx
  const dy = CLUSTER_B.cy - CLUSTER_A.cy
  // Boundary is perpendicular to the direction vector, passing through midpoint
  // dy * (x - mx) + dx * (y - my) = 0  => y = my - (dy/dx) * (x - mx)
  if (Math.abs(dx) < 1e-6) return my
  return my - (dy / dx) * (xVal - mx)
}

function classifyPoint(x: number, y: number): 0 | 1 {
  // Which side of the decision boundary? Use signed distance
  const mx = (CLUSTER_A.cx + CLUSTER_B.cx) / 2
  const my = (CLUSTER_A.cy + CLUSTER_B.cy) / 2
  const dx = CLUSTER_B.cx - CLUSTER_A.cx
  const dy = CLUSTER_B.cy - CLUSTER_A.cy
  // Normal direction: (dy, -dx) pointing toward cluster A
  const dist = dy * (x - mx) - dx * (y - my)
  if (dist > 0) return 0
  return 1
}

// Sample from a mixture of two Gaussians
function sampleFromDensity(rng: () => number): Point {
  // Pick a cluster with equal probability
  const cluster = rng() < 0.5 ? CLUSTER_A : CLUSTER_B
  const [z1, z2] = boxMullerFromRng(rng)
  return {
    x: cluster.cx + z1 * cluster.sx,
    y: cluster.cy + z2 * cluster.sy,
    label: cluster.label,
  }
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

const DATA_RANGE = { xMin: -4, xMax: 4, yMin: -3.5, yMax: 3.5 }
const PADDING = { top: 20, right: 20, bottom: 30, left: 40 }

function dataToSvg(
  x: number,
  y: number,
  plotWidth: number,
  plotHeight: number,
): { sx: number; sy: number } {
  const innerW = plotWidth - PADDING.left - PADDING.right
  const innerH = plotHeight - PADDING.top - PADDING.bottom
  return {
    sx: Math.round((PADDING.left + ((x - DATA_RANGE.xMin) / (DATA_RANGE.xMax - DATA_RANGE.xMin)) * innerW) * 100) / 100,
    sy: Math.round((PADDING.top + ((DATA_RANGE.yMax - y) / (DATA_RANGE.yMax - DATA_RANGE.yMin)) * innerH) * 100) / 100,
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function HeatmapLayer({
  plotWidth,
  plotHeight,
}: {
  plotWidth: number
  plotHeight: number
}) {
  const resolution = 40
  const innerW = plotWidth - PADDING.left - PADDING.right
  const innerH = plotHeight - PADDING.top - PADDING.bottom
  const cellW = innerW / resolution
  const cellH = innerH / resolution

  const cells = useMemo(() => {
    const result: { rx: number; ry: number; dA: number; dB: number }[] = []
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const dataX =
          DATA_RANGE.xMin +
          ((i + 0.5) / resolution) * (DATA_RANGE.xMax - DATA_RANGE.xMin)
        const dataY =
          DATA_RANGE.yMax -
          ((j + 0.5) / resolution) * (DATA_RANGE.yMax - DATA_RANGE.yMin)
        const dA = gaussianDensity(
          dataX,
          dataY,
          CLUSTER_A.cx,
          CLUSTER_A.cy,
          CLUSTER_A.sx,
          CLUSTER_A.sy,
        )
        const dB = gaussianDensity(
          dataX,
          dataY,
          CLUSTER_B.cx,
          CLUSTER_B.cy,
          CLUSTER_B.sx,
          CLUSTER_B.sy,
        )
        result.push({
          rx: PADDING.left + i * cellW,
          ry: PADDING.top + j * cellH,
          dA,
          dB,
        })
      }
    }
    return result
  }, [cellW, cellH])

  return (
    <g>
      {cells.map((cell, idx) => {
        const totalDensity = cell.dA + cell.dB
        if (totalDensity < 0.01) return null
        // Blend: class A is blue, class B is orange
        const ratioA = cell.dA / (totalDensity + 1e-10)
        const opacity = Math.min(totalDensity * 0.7, 0.65)
        // Color interpolation: blue (220, 80%, 60%) to orange (25, 80%, 55%)
        const r = Math.round(59 + (249 - 59) * (1 - ratioA))
        const g = Math.round(130 + (115 - 130) * (1 - ratioA))
        const b = Math.round(246 + (22 - 246) * (1 - ratioA))
        return (
          <rect
            key={idx}
            x={cell.rx}
            y={cell.ry}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={`rgb(${r}, ${g}, ${b})`}
            opacity={opacity}
          />
        )
      })}
    </g>
  )
}

function DecisionBoundaryLayer({
  plotWidth,
  plotHeight,
}: {
  plotWidth: number
  plotHeight: number
}) {
  // Draw the decision boundary as a line
  const x1 = DATA_RANGE.xMin
  const x2 = DATA_RANGE.xMax
  const y1 = decisionBoundaryY(x1)
  const y2 = decisionBoundaryY(x2)

  const p1 = dataToSvg(x1, y1, plotWidth, plotHeight)
  const p2 = dataToSvg(x2, y2, plotWidth, plotHeight)

  // Also shade regions
  const innerW = plotWidth - PADDING.left - PADDING.right
  const innerH = plotHeight - PADDING.top - PADDING.bottom
  const resolution = 30
  const cellW = innerW / resolution
  const cellH = innerH / resolution

  // Annotation position: midpoint of the boundary line, offset slightly
  const labelX = (p1.sx + p2.sx) / 2 + 8
  const labelY = (p1.sy + p2.sy) / 2 - 10

  const regionCells = useMemo(() => {
    const result: { rx: number; ry: number; cls: 0 | 1 }[] = []
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const dataX =
          DATA_RANGE.xMin +
          ((i + 0.5) / resolution) * (DATA_RANGE.xMax - DATA_RANGE.xMin)
        const dataY =
          DATA_RANGE.yMax -
          ((j + 0.5) / resolution) * (DATA_RANGE.yMax - DATA_RANGE.yMin)
        result.push({
          rx: PADDING.left + i * cellW,
          ry: PADDING.top + j * cellH,
          cls: classifyPoint(dataX, dataY),
        })
      }
    }
    return result
  }, [cellW, cellH])

  return (
    <g>
      {regionCells.map((cell, idx) => (
        <rect
          key={idx}
          x={cell.rx}
          y={cell.ry}
          width={cellW + 0.5}
          height={cellH + 0.5}
          fill={cell.cls === 0 ? '#3b82f6' : '#f97316'}
          opacity={0.08}
        />
      ))}
      <line
        x1={p1.sx}
        y1={p1.sy}
        x2={p2.sx}
        y2={p2.sy}
        stroke="#ffffff"
        strokeWidth={2.5}
        strokeDasharray="8,4"
        opacity={0.8}
      />
      {/* Inline annotation */}
      <text
        x={labelX}
        y={labelY}
        fill="#ffffff"
        fontSize={11}
        fontStyle="italic"
        opacity={0.85}
      >
        Decision boundary&mdash;the only thing this model learned
      </text>
    </g>
  )
}

function DataPoints({
  points,
  plotWidth,
  plotHeight,
  radius = 4,
  opacity = 0.8,
  isSampled = false,
}: {
  points: Point[]
  plotWidth: number
  plotHeight: number
  radius?: number
  opacity?: number
  isSampled?: boolean
}) {
  return (
    <g>
      {points.map((pt, i) => {
        const { sx, sy } = dataToSvg(pt.x, pt.y, plotWidth, plotHeight)
        const fill = pt.label === 0 ? '#3b82f6' : '#f97316'
        return (
          <circle
            key={`${isSampled ? 'sampled' : 'train'}-${i}`}
            cx={sx}
            cy={sy}
            r={radius}
            fill={fill}
            opacity={opacity}
            stroke={isSampled ? '#ffffff' : 'none'}
            strokeWidth={isSampled ? 1.5 : 0}
          />
        )
      })}
    </g>
  )
}

function AxisLabels({
  plotWidth,
  plotHeight,
}: {
  plotWidth: number
  plotHeight: number
}) {
  const innerW = plotWidth - PADDING.left - PADDING.right
  const innerH = plotHeight - PADDING.top - PADDING.bottom

  // X-axis ticks
  const xTicks = [-3, -2, -1, 0, 1, 2, 3]
  const yTicks = [-3, -2, -1, 0, 1, 2, 3]

  return (
    <g>
      {/* Axes */}
      <line
        x1={PADDING.left}
        y1={PADDING.top + innerH}
        x2={PADDING.left + innerW}
        y2={PADDING.top + innerH}
        stroke="var(--border)"
        strokeWidth={1}
      />
      <line
        x1={PADDING.left}
        y1={PADDING.top}
        x2={PADDING.left}
        y2={PADDING.top + innerH}
        stroke="var(--border)"
        strokeWidth={1}
      />
      {/* X-axis ticks */}
      {xTicks.map((tick) => {
        const { sx, sy } = dataToSvg(tick, DATA_RANGE.yMin, plotWidth, plotHeight)
        return (
          <g key={`xtick-${tick}`}>
            <line x1={sx} y1={sy - 3} x2={sx} y2={sy} stroke="var(--border)" strokeWidth={1} />
            <text x={sx} y={sy + 14} textAnchor="middle" fill="var(--muted-foreground)" fontSize={10}>
              {tick}
            </text>
          </g>
        )
      })}
      {/* Y-axis ticks */}
      {yTicks.map((tick) => {
        const { sx, sy } = dataToSvg(DATA_RANGE.xMin, tick, plotWidth, plotHeight)
        return (
          <g key={`ytick-${tick}`}>
            <line x1={sx} y1={sy} x2={sx + 3} y2={sy} stroke="var(--border)" strokeWidth={1} />
            <text x={sx - 6} y={sy + 4} textAnchor="end" fill="var(--muted-foreground)" fontSize={10}>
              {tick}
            </text>
          </g>
        )
      })}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Main widget component
// ---------------------------------------------------------------------------

type WidgetProps = {
  width?: number
  height?: number
}

export function GenerativeVsDiscriminativeWidget({
  width: widthOverride,
  height: heightOverride,
}: WidgetProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth
  const height = heightOverride ?? 380

  const [mode, setMode] = useState<Mode>('discriminative')
  const [sampledPoints, setSampledPoints] = useState<Point[]>([])
  const [sampleCounter, setSampleCounter] = useState(0)

  const handleSample = useCallback(() => {
    const rng = mulberry32(100 + sampleCounter + 1)
    const newPoints: Point[] = []
    for (let i = 0; i < 5; i++) {
      newPoints.push(sampleFromDensity(rng))
    }
    setSampledPoints((prev) => [...prev, ...newPoints])
    setSampleCounter((prev) => prev + 1)
  }, [sampleCounter])

  const handleClearSamples = useCallback(() => {
    setSampledPoints([])
    setSampleCounter(0)
  }, [])

  const handleModeChange = useCallback((newMode: Mode) => {
    setMode(newMode)
    setSampledPoints([])
    setSampleCounter(0)
  }, [])

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Mode toggle */}
      <div className="flex items-center justify-center gap-2">
        <button
          onClick={() => handleModeChange('discriminative')}
          className={cn(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer',
            mode === 'discriminative'
              ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          Discriminative
        </button>
        <button
          onClick={() => handleModeChange('generative')}
          className={cn(
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer',
            mode === 'generative'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          Generative
        </button>
      </div>

      {/* Description */}
      <div className="px-4 py-2 bg-muted/30 rounded-lg text-center">
        {mode === 'discriminative' && (
          <p className="text-sm text-muted-foreground">
            <strong className="text-foreground">Discriminative:</strong> The model
            learns a <em>decision boundary</em> that separates the two classes. It
            says nothing about what the data looks like&mdash;only where the boundary
            is.
          </p>
        )}
        {mode === 'generative' && (
          <p className="text-sm text-muted-foreground">
            <strong className="text-foreground">Generative:</strong> The model learns
            the <em>density</em> of each class&mdash;where data is likely to appear.
            You can <strong>sample</strong> new points from the learned distribution.
          </p>
        )}
      </div>

      {/* SVG plot */}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="rounded-lg bg-[#0f0f1a]"
      >
        {/* Background layers */}
        {mode === 'generative' && (
          <>
            <HeatmapLayer plotWidth={width} plotHeight={height} />
            {/* Inline annotations for generative mode */}
            <text
              x={dataToSvg(CLUSTER_A.cx, CLUSTER_A.cy, width, height).sx}
              y={dataToSvg(CLUSTER_A.cx, CLUSTER_A.cy, width, height).sy - 28}
              textAnchor="middle"
              fill="#ffffff"
              fontSize={11}
              fontStyle="italic"
              opacity={0.85}
            >
              High density&mdash;sample new points from here
            </text>
          </>
        )}
        {mode === 'discriminative' && (
          <DecisionBoundaryLayer plotWidth={width} plotHeight={height} />
        )}

        {/* Axes */}
        <AxisLabels plotWidth={width} plotHeight={height} />

        {/* Training data */}
        <DataPoints
          points={TRAINING_DATA}
          plotWidth={width}
          plotHeight={height}
          radius={3.5}
          opacity={0.6}
        />

        {/* Sampled points (generative mode) */}
        {mode === 'generative' && sampledPoints.length > 0 && (
          <DataPoints
            points={sampledPoints}
            plotWidth={width}
            plotHeight={height}
            radius={5}
            opacity={1}
            isSampled
          />
        )}

        {/* Legend */}
        <g transform={`translate(${width - 130}, 30)`}>
          <circle cx={0} cy={0} r={5} fill="#3b82f6" opacity={0.8} />
          <text x={10} y={4} fill="var(--muted-foreground)" fontSize={11}>
            Class A
          </text>
          <circle cx={0} cy={20} r={5} fill="#f97316" opacity={0.8} />
          <text x={10} y={24} fill="var(--muted-foreground)" fontSize={11}>
            Class B
          </text>
          {mode === 'discriminative' && (
            <>
              <line x1={-8} y1={40} x2={8} y2={40} stroke="#ffffff" strokeWidth={2} strokeDasharray="4,2" opacity={0.8} />
              <text x={14} y={44} fill="var(--muted-foreground)" fontSize={11}>
                Boundary
              </text>
            </>
          )}
          {mode === 'generative' && sampledPoints.length > 0 && (
            <>
              <circle cx={0} cy={40} r={5} fill="#3b82f6" stroke="#ffffff" strokeWidth={1.5} />
              <text x={10} y={44} fill="var(--muted-foreground)" fontSize={11}>
                Sampled
              </text>
            </>
          )}
        </g>
      </svg>

      {/* Generative controls */}
      {mode === 'generative' && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={handleSample}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors cursor-pointer"
          >
            Sample 5 Points
          </button>
          {sampledPoints.length > 0 && (
            <button
              onClick={handleClearSamples}
              className="px-3 py-2 rounded-lg text-sm text-muted-foreground bg-muted hover:bg-muted/80 transition-colors cursor-pointer"
            >
              Clear ({sampledPoints.length} sampled)
            </button>
          )}
        </div>
      )}

      {/* Stats */}
      <div className="flex gap-3 flex-wrap">
        <div className="flex-1 min-w-[140px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Training points</p>
          <p className="text-sm font-medium text-foreground">{TRAINING_DATA.length}</p>
        </div>
        <div className="flex-1 min-w-[140px] px-3 py-2 bg-muted/50 rounded-md">
          <p className="text-xs text-muted-foreground">Mode</p>
          <p className="text-sm font-medium text-foreground">
            {mode === 'discriminative' ? (
              <span>
                Learn P(y|x)&mdash;<span className="text-violet-400">boundary</span>
              </span>
            ) : (
              <span>
                Learn P(x)&mdash;<span className="text-emerald-400">density</span>
              </span>
            )}
          </p>
        </div>
        {mode === 'generative' && (
          <div className="flex-1 min-w-[140px] px-3 py-2 bg-muted/50 rounded-md">
            <p className="text-xs text-muted-foreground">Sampled points</p>
            <p className="text-sm font-medium text-foreground">
              <span className="text-emerald-400">{sampledPoints.length}</span>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
