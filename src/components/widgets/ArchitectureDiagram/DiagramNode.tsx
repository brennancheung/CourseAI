import { memo } from 'react'
import { Group, Rect, Text } from 'react-konva'
import type { DiagramNodeDef } from './types'

const DEFAULT_WIDTH = 140
const DEFAULT_HEIGHT = 48

type DiagramNodeProps = {
  node: DiagramNodeDef
  isHighlighted: boolean
  isDimmed: boolean
  isSelected: boolean
  onMouseEnter: () => void
  onMouseLeave: () => void
  onClick: () => void
}

export const DiagramNode = memo(function DiagramNode({
  node,
  isHighlighted,
  isDimmed,
  isSelected,
  onMouseEnter,
  onMouseLeave,
  onClick,
}: DiagramNodeProps) {
  const w = node.width ?? DEFAULT_WIDTH
  const h = node.height ?? DEFAULT_HEIGHT
  const x = node.x - w / 2
  const y = node.y - h / 2

  const fillColor = node.fillColor ?? '#1e293b'
  const baseStroke = node.strokeColor ?? '#475569'

  const getStroke = () => {
    if (isSelected) return '#f59e0b'
    if (isHighlighted) return '#fbbf24'
    return baseStroke
  }

  const getStrokeWidth = () => {
    if (isSelected) return 2.5
    if (isHighlighted) return 2
    return 1.5
  }

  const opacity = isDimmed ? 0.3 : 1

  return (
    <Group
      x={x}
      y={y}
      opacity={opacity}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      onTap={onClick}
    >
      <Rect
        width={w}
        height={h}
        fill={fillColor}
        stroke={getStroke()}
        strokeWidth={getStrokeWidth()}
        cornerRadius={6}
      />
      <Text
        width={w}
        height={h}
        text={node.label}
        fontSize={12}
        fontFamily="ui-monospace, monospace"
        fill="#e2e8f0"
        align="center"
        verticalAlign="middle"
        padding={4}
        listening={false}
      />
    </Group>
  )
})
