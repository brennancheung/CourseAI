'use client'

import { useState, useMemo, useCallback, useRef } from 'react'
import { Mafs, Coordinates, Plot, Point, Line, Theme } from 'mafs'
import 'mafs/core.css'

/**
 * LinearFitExplorer - Interactive widget for exploring line fitting
 *
 * Features:
 * - Draggable numbers in the equation to control slope and intercept
 * - Data points scattered around a "true" line
 * - Always shows MSE with color feedback
 * - Optional: Show residuals
 *
 * Used in:
 * - Lesson 1.1.2: Linear Regression from Scratch
 * - Lesson 1.1.3: Loss Functions (with residuals)
 */

interface DataPoint {
  x: number
  y: number
}

interface LinearFitExplorerProps {
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
  /** Height of the visualization */
  height?: number
}

// Generate deterministic data points around a line with noise
// Using a seed-like approach for consistent data across renders
function generateDataPoints(): DataPoint[] {
  // Fixed "random" data that looks scattered but is deterministic
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
interface DraggableNumberProps {
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
      // Sensitivity: 100px drag = full range
      const range = max - min
      const delta = (dx / 150) * range
      const newValue = Math.max(min, Math.min(max, startValue.current + delta))
      // Round to step
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
  height = 350,
}: LinearFitExplorerProps) {
  const [slope, setSlope] = useState(initialSlope)
  const [intercept, setIntercept] = useState(initialIntercept)

  // Use provided points or generate fixed ones
  const dataPoints = useMemo(() => {
    if (providedPoints) return providedPoints
    return generateDataPoints()
  }, [providedPoints])

  // Calculate MSE and optimal values
  const mse = useMemo(() => calculateMSE(dataPoints, slope, intercept), [dataPoints, slope, intercept])
  const optimal = useMemo(() => findOptimalParams(dataPoints), [dataPoints])
  const optimalMSE = useMemo(() => calculateMSE(dataPoints, optimal.slope, optimal.intercept), [dataPoints, optimal])

  // MSE quality indicator (green when close to optimal)
  const mseRatio = optimalMSE > 0 ? mse / optimalMSE : 1
  const getMSEColor = () => {
    if (mseRatio < 1.1) return 'text-green-400'
    if (mseRatio < 1.5) return 'text-yellow-400'
    if (mseRatio < 2.5) return 'text-orange-400'
    return 'text-red-400'
  }

  return (
    <div className="space-y-4">
      {/* Graph */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <Mafs
          height={height}
          viewBox={{ x: [-4, 4], y: [-3, 4] }}
          preserveAspectRatio={false}
        >
          <Coordinates.Cartesian />

          {/* Data points */}
          {dataPoints.map((point, i) => (
            <Point key={i} x={point.x} y={point.y} color={Theme.blue} />
          ))}

          {/* Residual lines */}
          {showResiduals &&
            dataPoints.map((point, i) => {
              const predicted = slope * point.x + intercept
              return (
                <Line.Segment
                  key={`residual-${i}`}
                  point1={[point.x, point.y]}
                  point2={[point.x, predicted]}
                  color={Theme.red}
                  opacity={0.6}
                  style="dashed"
                />
              )
            })}

          {/* The fitted line */}
          <Plot.OfX y={(x) => slope * x + intercept} color={Theme.green} />
        </Mafs>
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
          <button
            onClick={() => setIntercept(-intercept)}
            className="ml-1 text-xs text-muted-foreground hover:text-foreground px-1 rounded hover:bg-muted"
            title="Toggle sign"
          >
            ±
          </button>
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
        <div className="flex flex-wrap gap-4 text-sm">
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
