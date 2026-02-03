'use client'

import { Circle, Text, Group } from 'react-konva'
import { useCoordinates } from '../MathCanvas'

type BallProps = {
  /** X position in math coordinates */
  x: number
  /** Y position in math coordinates */
  y: number
  /** Radius in pixels */
  radius?: number
  /** Fill color */
  color?: string
  /** Whether the ball is draggable */
  draggable?: boolean
  /** Callback when dragged (returns math coordinates) */
  onDrag?: (x: number, y: number) => void
  /** Optional label below the ball */
  label?: string
  /** Label color */
  labelColor?: string
}

export function Ball({
  x,
  y,
  radius = 10,
  color = '#f97316',
  draggable = false,
  onDrag,
  label,
  labelColor = '#22c55e',
}: BallProps) {
  const { toPixelX, toPixelY, toMathX, toMathY } = useCoordinates()

  const px = toPixelX(x)
  const py = toPixelY(y)

  return (
    <Group>
      <Circle
        x={px}
        y={py}
        radius={radius}
        fill={color}
        shadowColor="black"
        shadowBlur={4}
        shadowOpacity={0.3}
        draggable={draggable}
        onDragMove={(e) => {
          if (onDrag) {
            const newX = toMathX(e.target.x())
            const newY = toMathY(e.target.y())
            onDrag(newX, newY)
          }
        }}
      />
      {label && (
        <Text
          x={px - 30}
          y={py + radius + 8}
          text={label}
          fontSize={12}
          fill={labelColor}
          width={60}
          align="center"
        />
      )}
    </Group>
  )
}
