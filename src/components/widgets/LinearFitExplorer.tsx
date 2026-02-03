'use client'

import { useState, useMemo } from 'react'
import { Mafs, Coordinates, Plot, Point, Line, Text, useMovablePoint, Theme } from 'mafs'
import 'mafs/core.css'

/**
 * LinearFitExplorer - Interactive widget for exploring line fitting
 *
 * Features:
 * - Draggable points to control slope and intercept
 * - Data points scattered around a "true" line
 * - Visual feedback on how well the line fits
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
  /** Whether to allow dragging the line */
  interactive?: boolean
  /** Height of the visualization */
  height?: number
}

// Generate random data points around a line with noise
function generateDataPoints(
  trueSlope: number,
  trueIntercept: number,
  count: number,
  noise: number
): DataPoint[] {
  const points: DataPoint[] = []
  for (let i = 0; i < count; i++) {
    const x = -3 + (6 * i) / (count - 1) // Spread from -3 to 3
    const y = trueSlope * x + trueIntercept + (Math.random() - 0.5) * noise
    points.push({ x, y })
  }
  return points
}

// Calculate MSE
function calculateMSE(points: DataPoint[], slope: number, intercept: number): number {
  const errors = points.map((p) => {
    const predicted = slope * p.x + intercept
    return Math.pow(p.y - predicted, 2)
  })
  return errors.reduce((sum, e) => sum + e, 0) / points.length
}

export function LinearFitExplorer({
  showResiduals = false,
  showMSE = false,
  initialSlope = 0.5,
  initialIntercept = 0,
  dataPoints: providedPoints,
  interactive = true,
  height = 400,
}: LinearFitExplorerProps) {
  // Use provided points or generate random ones
  const [dataPoints] = useState<DataPoint[]>(() => {
    if (providedPoints) return providedPoints
    // True line: y = 0.7x + 0.5, with noise
    return generateDataPoints(0.7, 0.5, 10, 1.5)
  })

  // Movable point to control the line
  // We'll use two points: one for intercept (at x=0) and one for slope
  const interceptPoint = useMovablePoint([0, initialIntercept], {
    constrain: ([, y]) => [0, y], // Lock to y-axis
  })

  const slopePoint = useMovablePoint([2, initialSlope * 2 + initialIntercept], {
    constrain: 'horizontal',
  })

  // Calculate slope and intercept from the two control points
  const intercept = interceptPoint.point[1]
  const slope = useMemo(() => {
    const dx = slopePoint.point[0] - 0
    const dy = slopePoint.point[1] - intercept
    return dy / dx
  }, [slopePoint.point, intercept])

  // Keep slope point on the line when intercept changes
  const adjustedSlopePointY = slope * slopePoint.point[0] + intercept

  // Calculate MSE
  const mse = useMemo(() => calculateMSE(dataPoints, slope, intercept), [dataPoints, slope, intercept])

  return (
    <div className="space-y-4">
      <div className="rounded-lg border bg-card overflow-hidden">
        <Mafs
          height={height}
          viewBox={{ x: [-4, 4], y: [-3, 5] }}
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
                  opacity={0.5}
                  style="dashed"
                />
              )
            })}

          {/* The fitted line */}
          <Plot.OfX y={(x) => slope * x + intercept} color={Theme.green} />

          {/* Control points (only if interactive) */}
          {interactive && (
            <>
              {/* Intercept control point */}
              <Point
                x={interceptPoint.point[0]}
                y={interceptPoint.point[1]}
                color={Theme.orange}
              />
              {interceptPoint.element}

              {/* Slope control point */}
              <Point
                x={slopePoint.point[0]}
                y={adjustedSlopePointY}
                color={Theme.orange}
              />
              {slopePoint.element}

              {/* Labels for control points */}
              <Text x={0.3} y={intercept + 0.3} size={12} color={Theme.orange}>
                intercept
              </Text>
              <Text x={slopePoint.point[0] + 0.3} y={adjustedSlopePointY + 0.3} size={12} color={Theme.orange}>
                slope
              </Text>
            </>
          )}
        </Mafs>
      </div>

      {/* Equation and MSE display */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Equation: </span>
          <span className="font-mono">
            y = {slope.toFixed(2)}x {intercept >= 0 ? '+' : ''} {intercept.toFixed(2)}
          </span>
        </div>
        {showMSE && (
          <div className="px-3 py-2 rounded-md bg-muted">
            <span className="text-muted-foreground">MSE: </span>
            <span className="font-mono">{mse.toFixed(3)}</span>
          </div>
        )}
      </div>

      {interactive && (
        <p className="text-xs text-muted-foreground">
          Drag the orange points to adjust the line. The <strong>intercept</strong> point
          moves up/down. The <strong>slope</strong> point changes the angle.
        </p>
      )}
    </div>
  )
}
