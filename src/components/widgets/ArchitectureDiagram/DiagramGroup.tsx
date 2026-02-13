import { memo } from 'react'
import { Rect, Text } from 'react-konva'
import type { DiagramGroupDef } from './types'

type DiagramGroupProps = {
  group: DiagramGroupDef
}

const DASH_PATTERN = [8, 4] as const

export const DiagramGroup = memo(function DiagramGroup({ group }: DiagramGroupProps) {
  const dash = group.strokeStyle === 'dashed' ? [...DASH_PATTERN] : undefined

  return (
    <>
      <Rect
        x={group.x}
        y={group.y}
        width={group.width}
        height={group.height}
        fill={group.fillColor ?? 'rgba(255,255,255,0.03)'}
        stroke={group.strokeColor ?? '#334155'}
        strokeWidth={1}
        dash={dash}
        cornerRadius={8}
        listening={false}
      />
      <Text
        x={group.x + 10}
        y={group.y + 8}
        text={group.title}
        fontSize={11}
        fontFamily="ui-monospace, monospace"
        fill="#64748b"
        listening={false}
      />
    </>
  )
})
