'use client'

import { ReactElement } from 'react'
import { Line, Text, Arrow } from 'react-konva'
import { useCoordinates } from '../MathCanvas'

type AxisProps = {
  /** Show axis arrows */
  showArrows?: boolean
  /** Color of axes */
  color?: string
  /** Label spacing (math units) */
  labelSpacing?: number
  /** Show labels */
  showLabels?: boolean
  /** Font size for labels */
  fontSize?: number
  /** X-axis label */
  xLabel?: string
  /** Y-axis label */
  yLabel?: string
}

export function Axis({
  showArrows = true,
  color = '#888899',
  labelSpacing = 1,
  showLabels = true,
  fontSize = 12,
  xLabel,
  yLabel,
}: AxisProps) {
  const { viewBox, toPixelX, toPixelY, width, height } = useCoordinates()
  const { xMin, xMax, yMin, yMax } = viewBox

  const elements: ReactElement[] = []

  // X axis (y = 0)
  const y0 = toPixelY(0)
  if (showArrows) {
    elements.push(
      <Arrow
        key="x-axis"
        points={[0, y0, width, y0]}
        stroke={color}
        strokeWidth={2}
        fill={color}
        pointerLength={8}
        pointerWidth={6}
      />
    )
  } else {
    elements.push(
      <Line
        key="x-axis"
        points={[0, y0, width, y0]}
        stroke={color}
        strokeWidth={2}
      />
    )
  }

  // Y axis (x = 0)
  const x0 = toPixelX(0)
  if (showArrows) {
    elements.push(
      <Arrow
        key="y-axis"
        points={[x0, height, x0, 0]}
        stroke={color}
        strokeWidth={2}
        fill={color}
        pointerLength={8}
        pointerWidth={6}
      />
    )
  } else {
    elements.push(
      <Line
        key="y-axis"
        points={[x0, height, x0, 0]}
        stroke={color}
        strokeWidth={2}
      />
    )
  }

  // X axis labels
  if (showLabels) {
    const startX = Math.ceil(xMin / labelSpacing) * labelSpacing
    for (let x = startX; x <= xMax; x += labelSpacing) {
      if (Math.abs(x) < 0.01) continue // Skip 0
      const px = toPixelX(x)
      elements.push(
        <Text
          key={`xlabel-${x}`}
          x={px - 10}
          y={y0 + 8}
          text={x.toString()}
          fontSize={fontSize}
          fill={color}
          width={20}
          align="center"
        />
      )
    }

    // Y axis labels
    const startY = Math.ceil(yMin / labelSpacing) * labelSpacing
    for (let y = startY; y <= yMax; y += labelSpacing) {
      if (Math.abs(y) < 0.01) continue // Skip 0
      const py = toPixelY(y)
      elements.push(
        <Text
          key={`ylabel-${y}`}
          x={x0 + 8}
          y={py - 6}
          text={y.toString()}
          fontSize={fontSize}
          fill={color}
        />
      )
    }
  }

  // Axis labels
  if (xLabel) {
    elements.push(
      <Text
        key="x-label"
        x={width - 30}
        y={y0 + 20}
        text={xLabel}
        fontSize={fontSize + 2}
        fill={color}
        fontStyle="italic"
      />
    )
  }

  if (yLabel) {
    elements.push(
      <Text
        key="y-label"
        x={x0 + 15}
        y={15}
        text={yLabel}
        fontSize={fontSize + 2}
        fill={color}
        fontStyle="italic"
      />
    )
  }

  return <>{elements}</>
}
