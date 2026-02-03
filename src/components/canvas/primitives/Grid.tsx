'use client'

import { ReactElement } from 'react'
import { Line } from 'react-konva'
import { useCoordinates } from '../MathCanvas'

type GridProps = {
  /** Grid line spacing in math units */
  spacing?: number
  /** Color of grid lines */
  color?: string
  /** Opacity of grid lines */
  opacity?: number
}

export function Grid({
  spacing = 1,
  color = '#444466',
  opacity = 0.5,
}: GridProps) {
  const { viewBox, toPixelX, toPixelY, width, height } = useCoordinates()
  const { xMin, xMax, yMin, yMax } = viewBox

  const lines: ReactElement[] = []

  // Vertical lines
  const startX = Math.ceil(xMin / spacing) * spacing
  for (let x = startX; x <= xMax; x += spacing) {
    const px = toPixelX(x)
    lines.push(
      <Line
        key={`v-${x}`}
        points={[px, 0, px, height]}
        stroke={color}
        strokeWidth={1}
        opacity={opacity}
      />
    )
  }

  // Horizontal lines
  const startY = Math.ceil(yMin / spacing) * spacing
  for (let y = startY; y <= yMax; y += spacing) {
    const py = toPixelY(y)
    lines.push(
      <Line
        key={`h-${y}`}
        points={[0, py, width, py]}
        stroke={color}
        strokeWidth={1}
        opacity={opacity}
      />
    )
  }

  return <>{lines}</>
}
