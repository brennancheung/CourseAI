'use client'

import { useState, useMemo, useRef, useEffect, useCallback } from 'react'

/**
 * LossSurfaceExplorer - 2D contour heatmap of loss landscape
 *
 * Shows how MSE varies as slope and intercept change.
 * Features:
 * - 2D heatmap with contour lines
 * - Click/drag to move position on surface
 * - Connected 2D line fitting view
 *
 * Used in:
 * - Lesson 1.1.3: Loss Functions
 * - Lesson 1.1.4: Gradient Descent (with gradient arrows)
 */

interface DataPoint {
  x: number
  y: number
}

// Fixed data points for consistency
const DATA_POINTS: DataPoint[] = [
  { x: -2, y: -0.8 },
  { x: -1.5, y: -0.3 },
  { x: -1, y: 0.2 },
  { x: -0.5, y: 0.8 },
  { x: 0, y: 0.5 },
  { x: 0.5, y: 1.2 },
  { x: 1, y: 1.5 },
  { x: 1.5, y: 1.8 },
  { x: 2, y: 2.3 },
]

const SLOPE_RANGE: [number, number] = [-1, 2]
const INTERCEPT_RANGE: [number, number] = [-1, 2]

function calculateMSE(slope: number, intercept: number, points: DataPoint[]): number {
  let sum = 0
  for (const p of points) {
    const predicted = slope * p.x + intercept
    sum += (p.y - predicted) ** 2
  }
  return sum / points.length
}

function findOptimalParams(points: DataPoint[]): { slope: number; intercept: number } {
  const n = points.length
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumX2 = 0

  for (const p of points) {
    sumX += p.x
    sumY += p.y
    sumXY += p.x * p.y
    sumX2 += p.x * p.x
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
  const intercept = (sumY - slope * sumX) / n

  return { slope, intercept }
}

// Color palette: dark indigo (low) → teal → green → yellow (high)
// Works well on dark backgrounds
function lossToRGB(t: number): [number, number, number] {
  // t is 0..1 (normalized loss, 0 = minimum, 1 = maximum)
  // Apply sqrt for perceptual spread — gives more color range to lower loss values
  const s = Math.sqrt(Math.max(0, Math.min(1, t)))

  // 5-stop gradient
  const stops: Array<{ pos: number; r: number; g: number; b: number }> = [
    { pos: 0.0, r: 15, g: 20, b: 80 },    // deep indigo (minimum)
    { pos: 0.25, r: 20, g: 70, b: 140 },   // blue
    { pos: 0.5, r: 15, g: 130, b: 130 },   // teal
    { pos: 0.75, r: 50, g: 170, b: 80 },   // green
    { pos: 1.0, r: 220, g: 200, b: 50 },   // yellow (maximum)
  ]

  let lower = stops[0]
  let upper = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (s >= stops[i].pos && s <= stops[i + 1].pos) {
      lower = stops[i]
      upper = stops[i + 1]
      break
    }
  }

  const range = upper.pos - lower.pos
  const f = range === 0 ? 0 : (s - lower.pos) / range

  return [
    Math.round(lower.r + (upper.r - lower.r) * f),
    Math.round(lower.g + (upper.g - lower.g) * f),
    Math.round(lower.b + (upper.b - lower.b) * f),
  ]
}

// Precompute loss grid for heatmap and contour detection
function computeLossGrid(resolution: number): {
  grid: Float64Array
  minLoss: number
  maxLoss: number
} {
  const grid = new Float64Array(resolution * resolution)
  let minLoss = Infinity
  let maxLoss = -Infinity

  for (let iy = 0; iy < resolution; iy++) {
    for (let ix = 0; ix < resolution; ix++) {
      const slope = SLOPE_RANGE[0] + (SLOPE_RANGE[1] - SLOPE_RANGE[0]) * (ix / (resolution - 1))
      const intercept = INTERCEPT_RANGE[1] - (INTERCEPT_RANGE[1] - INTERCEPT_RANGE[0]) * (iy / (resolution - 1))
      const loss = calculateMSE(slope, intercept, DATA_POINTS)
      grid[iy * resolution + ix] = loss
      minLoss = Math.min(minLoss, loss)
      maxLoss = Math.max(maxLoss, loss)
    }
  }

  return { grid, minLoss, maxLoss }
}

function LossHeatmap({
  slope,
  intercept,
  onPositionChange,
  optimal,
}: {
  slope: number
  intercept: number
  onPositionChange: (s: number, i: number) => void
  optimal: { slope: number; intercept: number }
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)
  const resolution = 200

  const { grid, minLoss, maxLoss } = useMemo(() => computeLossGrid(resolution), [resolution])

  // Contour levels — evenly spaced in sqrt-space for better visual distribution
  const contourLevels = useMemo(() => {
    const levels: number[] = []
    const numContours = 12
    for (let i = 1; i < numContours; i++) {
      const t = i / numContours
      // Invert the sqrt mapping: if sqrt(normalized) = t, then loss = minLoss + t^2 * range
      const loss = minLoss + t * t * (maxLoss - minLoss)
      levels.push(loss)
    }
    return levels
  }, [minLoss, maxLoss])

  // Render heatmap
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = resolution
    canvas.height = resolution

    const imageData = ctx.createImageData(resolution, resolution)
    const range = maxLoss - minLoss

    for (let iy = 0; iy < resolution; iy++) {
      for (let ix = 0; ix < resolution; ix++) {
        const loss = grid[iy * resolution + ix]
        const t = range === 0 ? 0 : (loss - minLoss) / range

        // Check if this pixel is near a contour line
        let isContour = false
        for (const level of contourLevels) {
          // Check adjacent pixels
          const checkNeighbor = (dx: number, dy: number): boolean => {
            const nx = ix + dx
            const ny = iy + dy
            if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution) return false
            const neighborLoss = grid[ny * resolution + nx]
            return (loss - level) * (neighborLoss - level) < 0
          }

          if (checkNeighbor(1, 0) || checkNeighbor(0, 1)) {
            isContour = true
            break
          }
        }

        const [r, g, b] = lossToRGB(t)
        const idx = (iy * resolution + ix) * 4

        if (isContour) {
          // Contour lines: semi-transparent white
          imageData.data[idx] = Math.min(255, r + 60)
          imageData.data[idx + 1] = Math.min(255, g + 60)
          imageData.data[idx + 2] = Math.min(255, b + 60)
          imageData.data[idx + 3] = 255
        } else {
          imageData.data[idx] = r
          imageData.data[idx + 1] = g
          imageData.data[idx + 2] = b
          imageData.data[idx + 3] = 255
        }
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [grid, minLoss, maxLoss, contourLevels, resolution])

  // Convert pixel coordinates to parameter space
  const pixelToParams = useCallback(
    (clientX: number, clientY: number): { slope: number; intercept: number } | null => {
      const container = containerRef.current
      if (!container) return null
      const rect = container.getBoundingClientRect()
      const px = (clientX - rect.left) / rect.width
      const py = (clientY - rect.top) / rect.height

      const s = SLOPE_RANGE[0] + (SLOPE_RANGE[1] - SLOPE_RANGE[0]) * Math.max(0, Math.min(1, px))
      const i = INTERCEPT_RANGE[1] - (INTERCEPT_RANGE[1] - INTERCEPT_RANGE[0]) * Math.max(0, Math.min(1, py))
      return { slope: s, intercept: i }
    },
    []
  )

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      isDragging.current = true
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
      const params = pixelToParams(e.clientX, e.clientY)
      if (params) {
        onPositionChange(
          Math.round(params.slope * 20) / 20,
          Math.round(params.intercept * 20) / 20
        )
      }
    },
    [pixelToParams, onPositionChange]
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging.current) return
      const params = pixelToParams(e.clientX, e.clientY)
      if (params) {
        onPositionChange(
          Math.round(params.slope * 20) / 20,
          Math.round(params.intercept * 20) / 20
        )
      }
    },
    [pixelToParams, onPositionChange]
  )

  const handlePointerUp = useCallback(() => {
    isDragging.current = false
  }, [])

  // Convert parameter values to percentage position
  const slopeToPercent = (s: number) =>
    ((s - SLOPE_RANGE[0]) / (SLOPE_RANGE[1] - SLOPE_RANGE[0])) * 100
  const interceptToPercent = (i: number) =>
    ((INTERCEPT_RANGE[1] - i) / (INTERCEPT_RANGE[1] - INTERCEPT_RANGE[0])) * 100

  const currentX = slopeToPercent(slope)
  const currentY = interceptToPercent(intercept)
  const optimalX = slopeToPercent(optimal.slope)
  const optimalY = interceptToPercent(optimal.intercept)

  // Tick values for axis labels
  const slopeTicks = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
  const interceptTicks = [-1, -0.5, 0, 0.5, 1, 1.5, 2]

  return (
    <div className="space-y-1">
      {/* Y-axis label */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground -rotate-90 w-4 flex-shrink-0 whitespace-nowrap">
          intercept (b)
        </span>
        <div className="flex-1">
          {/* Heatmap with markers */}
          <div
            ref={containerRef}
            className="relative aspect-square w-full cursor-crosshair rounded-lg overflow-hidden border border-border/50"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
          >
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full"
              style={{ imageRendering: 'auto' }}
            />

            {/* SVG overlay for markers and labels */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Crosshair at current position */}
              <line
                x1={currentX}
                y1="0"
                x2={currentX}
                y2="100"
                stroke="white"
                strokeWidth="0.3"
                opacity="0.3"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1="0"
                y1={currentY}
                x2="100"
                y2={currentY}
                stroke="white"
                strokeWidth="0.3"
                opacity="0.3"
                vectorEffect="non-scaling-stroke"
              />
            </svg>

            {/* Optimal point (green) — positioned with CSS % */}
            <div
              className="absolute w-3 h-3 rounded-full bg-emerald-400 border-2 border-white shadow-lg pointer-events-none"
              style={{
                left: `${optimalX}%`,
                top: `${optimalY}%`,
                transform: 'translate(-50%, -50%)',
              }}
            />
            <div
              className="absolute text-[10px] text-emerald-400 font-medium pointer-events-none whitespace-nowrap"
              style={{
                left: `${optimalX}%`,
                top: `${optimalY}%`,
                transform: 'translate(-50%, -150%)',
              }}
            >
              minimum
            </div>

            {/* Current point (orange) */}
            <div
              className="absolute w-4 h-4 rounded-full bg-orange-500 border-2 border-white shadow-lg shadow-orange-500/30 pointer-events-none"
              style={{
                left: `${currentX}%`,
                top: `${currentY}%`,
                transform: 'translate(-50%, -50%)',
              }}
            />

            {/* Y-axis tick labels (intercept) */}
            {interceptTicks
              .filter((v) => v > INTERCEPT_RANGE[0] && v < INTERCEPT_RANGE[1])
              .map((v) => (
                <div
                  key={`y${v}`}
                  className="absolute left-1 text-[9px] text-white/50 pointer-events-none"
                  style={{
                    top: `${interceptToPercent(v)}%`,
                    transform: 'translateY(-50%)',
                  }}
                >
                  {v}
                </div>
              ))}

            {/* X-axis tick labels (slope) */}
            {slopeTicks
              .filter((v) => v > SLOPE_RANGE[0] && v < SLOPE_RANGE[1])
              .map((v) => (
                <div
                  key={`x${v}`}
                  className="absolute bottom-0.5 text-[9px] text-white/50 pointer-events-none"
                  style={{
                    left: `${slopeToPercent(v)}%`,
                    transform: 'translateX(-50%)',
                  }}
                >
                  {v}
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* X-axis label */}
      <div className="text-center">
        <span className="text-xs text-muted-foreground">slope (w)</span>
      </div>
    </div>
  )
}

// 2D line preview
function LinePreview({ slope, intercept }: { slope: number; intercept: number }) {
  const width = 300
  const height = 200
  const padding = 30

  const xScale = (x: number) => padding + ((x + 3) / 6) * (width - 2 * padding)
  const yScale = (y: number) => height - padding - ((y + 1) / 4) * (height - 2 * padding)

  const lineY1 = slope * -3 + intercept
  const lineY2 = slope * 3 + intercept

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full bg-muted/30 rounded-lg">
      {/* Grid lines */}
      {[-2, -1, 0, 1, 2].map((x) => (
        <line
          key={`v${x}`}
          x1={xScale(x)}
          y1={padding}
          x2={xScale(x)}
          y2={height - padding}
          stroke="#333"
          strokeWidth={1}
          opacity={0.3}
        />
      ))}
      {[-1, 0, 1, 2, 3].map((y) => (
        <line
          key={`h${y}`}
          x1={padding}
          y1={yScale(y)}
          x2={width - padding}
          y2={yScale(y)}
          stroke="#333"
          strokeWidth={1}
          opacity={0.3}
        />
      ))}

      {/* Fitted line */}
      <line
        x1={xScale(-3)}
        y1={yScale(lineY1)}
        x2={xScale(3)}
        y2={yScale(lineY2)}
        stroke="#22c55e"
        strokeWidth={2}
      />

      {/* Data points */}
      {DATA_POINTS.map((p, i) => (
        <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={4} fill="#3b82f6" />
      ))}

      {/* Residual lines */}
      {DATA_POINTS.map((p, i) => {
        const predicted = slope * p.x + intercept
        return (
          <line
            key={`r${i}`}
            x1={xScale(p.x)}
            y1={yScale(p.y)}
            x2={xScale(p.x)}
            y2={yScale(predicted)}
            stroke="#ef4444"
            strokeWidth={1}
            opacity={0.5}
          />
        )
      })}
    </svg>
  )
}

export function LossSurfaceExplorer() {
  const [slope, setSlope] = useState(0.5)
  const [intercept, setIntercept] = useState(0.5)

  const mse = calculateMSE(slope, intercept, DATA_POINTS)
  const optimal = useMemo(() => findOptimalParams(DATA_POINTS), [])
  const optimalMSE = calculateMSE(optimal.slope, optimal.intercept, DATA_POINTS)

  const handlePositionChange = useCallback((s: number, i: number) => {
    setSlope(s)
    setIntercept(i)
  }, [])

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        {/* 2D Contour Heatmap */}
        <LossHeatmap
          slope={slope}
          intercept={intercept}
          onPositionChange={handlePositionChange}
          optimal={optimal}
        />

        {/* Line Preview + Sliders */}
        <div className="space-y-4">
          <LinePreview slope={slope} intercept={intercept} />

          {/* Sliders */}
          <div className="space-y-3">
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Slope (w)</span>
                <span className="font-mono">{slope.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="2"
                step="0.05"
                value={slope}
                onChange={(e) => setSlope(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Intercept (b)</span>
                <span className="font-mono">{intercept.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-1"
                max="2"
                step="0.05"
                value={intercept}
                onChange={(e) => setIntercept(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-4 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Current MSE: </span>
          <span className="font-mono">{mse.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-emerald-500/10 text-emerald-400">
          <span>Optimal MSE: </span>
          <span className="font-mono">{optimalMSE.toFixed(3)}</span>
          <span className="text-xs ml-2">
            (w={optimal.slope.toFixed(2)}, b={optimal.intercept.toFixed(2)})
          </span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Click or drag on the heatmap to explore the loss surface. Darker regions = lower loss.
        The orange point is your current position. The green point marks the minimum.
      </p>
    </div>
  )
}
