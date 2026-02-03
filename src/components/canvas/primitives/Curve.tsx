'use client'

import { Line } from 'react-konva'
import { useCoordinates } from '../MathCanvas'

type CurveProps = {
  /** Function to plot y = f(x) */
  fn: (x: number) => number
  /** Color of the curve */
  color?: string
  /** Line width */
  strokeWidth?: number
  /** Number of sample points */
  samples?: number
  /** Dashed line */
  dashed?: boolean
}

export function Curve({
  fn,
  color = '#6366f1',
  strokeWidth = 2,
  samples = 200,
  dashed = false,
}: CurveProps) {
  const { viewBox, toPixelX, toPixelY } = useCoordinates()
  const { xMin, xMax, yMin, yMax } = viewBox

  // Generate points
  const points: number[] = []
  const step = (xMax - xMin) / samples

  for (let i = 0; i <= samples; i++) {
    const x = xMin + i * step
    const y = fn(x)

    // Only add points within viewport
    if (y >= yMin && y <= yMax) {
      points.push(toPixelX(x), toPixelY(y))
    }
  }

  return (
    <Line
      points={points}
      stroke={color}
      strokeWidth={strokeWidth}
      lineCap="round"
      lineJoin="round"
      dash={dashed ? [8, 4] : undefined}
    />
  )
}
