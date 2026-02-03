'use client'

import { useState, useMemo, useCallback, useRef } from 'react'
import { Line, Circle, Text, Arrow } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * LinearFitExplorer - Interactive widget for exploring line fitting
 *
 * Built with react-konva for reliable canvas rendering.
 * Features:
 * - Draggable numbers in the equation to control slope and intercept
 * - Data points scattered around a "true" line
 * - Always shows MSE with color feedback
 * - Optional: Show residuals
 */

type DataPoint = {
  x: number
  y: number
}

type LinearFitExplorerProps = {
  /** Show residual lines from points to the fitted line */
  showResiduals?: boolean
  /** Show MSE calculation */
  showMSE?: boolean
  /** Initial slope */
  initialSlope?: number
  /** Initial intercept */
  initialIntercept?: number
  /** Fixed data points (optional - will generate random if not provided) */
  dataPoints?: DataPoint[]
  /** Width of the visualization */
  width?: number
  /** Height of the visualization */
  height?: number
}

// Viewport config
const VIEW = {
  xMin: -4,
  xMax: 4,
  yMin: -2,
  yMax: 4,
}

// Generate deterministic data points around a line with noise
function generateDataPoints(): DataPoint[] {
  return [
    { x: -3, y: -1.5 },
    { x: -2.5, y: -0.8 },
    { x: -2, y: 0.2 },
    { x: -1.5, y: 0.1 },
    { x: -1, y: 0.5 },
    { x: -0.5, y: 0.8 },
    { x: 0, y: 0.6 },
    { x: 0.5, y: 1.2 },
    { x: 1, y: 0.9 },
    { x: 1.5, y: 1.8 },
    { x: 2, y: 1.5 },
    { x: 2.5, y: 2.2 },
    { x: 3, y: 2.0 },
  ]
}

// Calculate MSE
function calculateMSE(points: DataPoint[], slope: number, intercept: number): number {
  const errors = points.map((p) => {
    const predicted = slope * p.x + intercept
    return Math.pow(p.y - predicted, 2)
  })
  return errors.reduce((sum, e) => sum + e, 0) / points.length
}

// Find optimal parameters analytically
function findOptimalParams(points: DataPoint[]): { slope: number; intercept: number } {
  const n = points.length
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0

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

/**
 * DraggableNumber - A number that can be dragged left/right to change value
 */
type DraggableNumberProps = {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  color?: string
  label?: string
}

function DraggableNumber({ value, onChange, min, max, step, color = '#f97316', label }: DraggableNumberProps) {
  const isDragging = useRef(false)
  const startX = useRef(0)
  const startValue = useRef(0)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true
    startX.current = e.clientX
    startValue.current = value
    document.body.style.cursor = 'ew-resize'
    document.body.style.userSelect = 'none'

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      const dx = e.clientX - startX.current
      const range = max - min
      const delta = (dx / 150) * range
      const newValue = Math.max(min, Math.min(max, startValue.current + delta))
      const rounded = Math.round(newValue / step) * step
      onChange(rounded)
    }

    const handleMouseUp = () => {
      isDragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [value, onChange, min, max, step])

  return (
    <span
      onMouseDown={handleMouseDown}
      className="cursor-ew-resize px-1 py-0.5 rounded font-mono font-bold select-none hover:bg-white/10 transition-colors"
      style={{ color }}
      title={label ? `Drag to adjust ${label}` : 'Drag left/right to adjust'}
    >
      {value >= 0 ? value.toFixed(2) : value.toFixed(2)}
    </span>
  )
}

export function LinearFitExplorer({
  showResiduals = false,
  showMSE = true,
  initialSlope = 0.3,
  initialIntercept = 0,
  dataPoints: providedPoints,
  width = 600,
  height = 350,
}: LinearFitExplorerProps) {

  const [slope, setSlope] = useState(initialSlope)
  const [intercept, setIntercept] = useState(initialIntercept)

  // Coordinate transforms
  const toPixelX = (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width
  const toPixelY = (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height

  // Use provided points or generate fixed ones
  const dataPoints = useMemo(() => {
    if (providedPoints) return providedPoints
    return generateDataPoints()
  }, [providedPoints])

  // Calculate MSE and optimal values
  const mse = useMemo(() => calculateMSE(dataPoints, slope, intercept), [dataPoints, slope, intercept])
  const optimal = useMemo(() => findOptimalParams(dataPoints), [dataPoints])
  const optimalMSE = useMemo(() => calculateMSE(dataPoints, optimal.slope, optimal.intercept), [dataPoints, optimal])

  // MSE quality indicator
  const mseRatio = optimalMSE > 0 ? mse / optimalMSE : 1
  const getMSEColor = () => {
    if (mseRatio < 1.1) return 'text-green-400'
    if (mseRatio < 1.5) return 'text-yellow-400'
    if (mseRatio < 2.5) return 'text-orange-400'
    return 'text-red-400'
  }

  // Grid lines
  const gridLines: { points: number[]; key: string }[] = []
  for (let x = Math.ceil(VIEW.xMin); x <= VIEW.xMax; x++) {
    gridLines.push({ key: `v${x}`, points: [toPixelX(x), 0, toPixelX(x), height] })
  }
  for (let y = Math.ceil(VIEW.yMin); y <= VIEW.yMax; y++) {
    gridLines.push({ key: `h${y}`, points: [0, toPixelY(y), width, toPixelY(y)] })
  }

  // Line endpoints (extend beyond viewport)
  const lineY1 = slope * VIEW.xMin + intercept
  const lineY2 = slope * VIEW.xMax + intercept

  return (
    <div className="space-y-4">
      {/* Graph */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
            {/* Grid */}
            {gridLines.map((line) => (
              <Line
                key={line.key}
                points={line.points}
                stroke="#333355"
                strokeWidth={1}
              />
            ))}

            {/* Axes */}
            <Arrow
              points={[0, toPixelY(0), width, toPixelY(0)]}
              stroke="#666688"
              strokeWidth={2}
              fill="#666688"
              pointerLength={8}
              pointerWidth={6}
            />
            <Arrow
              points={[toPixelX(0), height, toPixelX(0), 0]}
              stroke="#666688"
              strokeWidth={2}
              fill="#666688"
              pointerLength={8}
              pointerWidth={6}
            />

            {/* Axis labels */}
            <Text x={width - 20} y={toPixelY(0) + 10} text="x" fontSize={14} fill="#888" />
            <Text x={toPixelX(0) + 10} y={10} text="y" fontSize={14} fill="#888" />

            {/* X axis numbers */}
            {[-3, -2, -1, 1, 2, 3].map((x) => (
              <Text
                key={`xl${x}`}
                x={toPixelX(x) - 6}
                y={toPixelY(0) + 8}
                text={x.toString()}
                fontSize={11}
                fill="#888"
              />
            ))}

            {/* Y axis numbers */}
            {[-1, 1, 2, 3].map((y) => (
              <Text
                key={`yl${y}`}
                x={toPixelX(0) + 8}
                y={toPixelY(y) - 5}
                text={y.toString()}
                fontSize={11}
                fill="#888"
              />
            ))}

            {/* Residual lines */}
            {showResiduals &&
              dataPoints.map((point, i) => {
                const predicted = slope * point.x + intercept
                return (
                  <Line
                    key={`residual-${i}`}
                    points={[
                      toPixelX(point.x),
                      toPixelY(point.y),
                      toPixelX(point.x),
                      toPixelY(predicted),
                    ]}
                    stroke="#ef4444"
                    strokeWidth={2}
                    dash={[4, 4]}
                    opacity={0.7}
                  />
                )
              })}

            {/* The fitted line */}
            <Line
              points={[toPixelX(VIEW.xMin), toPixelY(lineY1), toPixelX(VIEW.xMax), toPixelY(lineY2)]}
              stroke="#22c55e"
              strokeWidth={3}
            />

            {/* Data points */}
            {dataPoints.map((point, i) => (
              <Circle
                key={i}
                x={toPixelX(point.x)}
                y={toPixelY(point.y)}
                radius={8}
                fill="#6366f1"
                shadowColor="black"
                shadowBlur={4}
                shadowOpacity={0.3}
              />
            ))}
        </ZoomableCanvas>
      </div>

      {/* Interactive Equation */}
      <div className="p-4 rounded-lg bg-muted/50 border">
        <div className="text-lg flex items-center gap-1 flex-wrap">
          <span className="text-muted-foreground">y =</span>
          <DraggableNumber
            value={slope}
            onChange={setSlope}
            min={-2}
            max={2}
            step={0.01}
            color="#f97316"
            label="slope"
          />
          <span className="text-muted-foreground">x</span>
          <span className="text-muted-foreground">{intercept >= 0 ? '+' : '−'}</span>
          <DraggableNumber
            value={Math.abs(intercept)}
            onChange={(v) => setIntercept(intercept >= 0 ? v : -v)}
            min={0}
            max={3}
            step={0.01}
            color="#8b5cf6"
            label="intercept"
          />
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Drag the <span className="text-orange-400 font-medium">orange</span> and{' '}
          <span className="text-violet-400 font-medium">purple</span> numbers to adjust the line
        </p>
      </div>

      {/* Sliders for fine control */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Slope (m)</span>
            <span className="font-mono text-orange-400">{slope.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.01"
            value={slope}
            onChange={(e) => setSlope(parseFloat(e.target.value))}
            className="w-full accent-orange-500"
          />
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Intercept (b)</span>
            <span className="font-mono text-violet-400">{intercept.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="-3"
            max="3"
            step="0.01"
            value={intercept}
            onChange={(e) => setIntercept(parseFloat(e.target.value))}
            className="w-full accent-violet-500"
          />
        </div>
      </div>

      {/* MSE Display */}
      {showMSE && (
        <div className="flex flex-wrap gap-4 text-sm justify-center">
          <div className="px-3 py-2 rounded-md bg-muted flex items-center gap-2">
            <span className="text-muted-foreground">MSE:</span>
            <span className={`font-mono font-bold ${getMSEColor()}`}>{mse.toFixed(3)}</span>
            {mseRatio < 1.05 && <span className="text-green-400 text-xs">✓ Optimal!</span>}
          </div>
          <div className="px-3 py-2 rounded-md bg-muted/50 text-xs">
            <span className="text-muted-foreground">Best possible: </span>
            <span className="font-mono text-green-400">{optimalMSE.toFixed(3)}</span>
            <span className="text-muted-foreground ml-1">
              (m={optimal.slope.toFixed(2)}, b={optimal.intercept.toFixed(2)})
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
