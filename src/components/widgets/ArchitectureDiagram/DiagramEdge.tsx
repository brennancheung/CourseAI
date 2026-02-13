import { memo } from 'react'
import { Arrow, Text } from 'react-konva'
import type { DiagramNodeDef, DiagramEdgeDef } from './types'
import { getEdgePoint } from './edge-utils'

const DASH_PATTERN = [6, 4] as const

type RectBounds = { x: number; y: number; width: number; height: number }

type DiagramEdgeProps = {
  edge: DiagramEdgeDef
  fromNode: DiagramNodeDef
  toNode: DiagramNodeDef
  fromBounds: RectBounds
  toBounds: RectBounds
  isHighlighted: boolean
  isDimmed: boolean
}

function computePoints(
  edge: DiagramEdgeDef,
  fromBounds: RectBounds,
  toBounds: RectBounds,
): number[] {
  const waypoints = edge.waypoints ?? []

  if (waypoints.length === 0) {
    // Direct connection
    const from = getEdgePoint(fromBounds, { x: toBounds.x, y: toBounds.y })
    const to = getEdgePoint(toBounds, { x: fromBounds.x, y: fromBounds.y })
    return [from.x, from.y, to.x, to.y]
  }

  // With waypoints: source → first wp, intermediate wps, last wp → target
  const firstWp = waypoints[0]
  const lastWp = waypoints[waypoints.length - 1]
  const from = getEdgePoint(fromBounds, firstWp)
  const to = getEdgePoint(toBounds, lastWp)

  const points = [from.x, from.y]
  for (const wp of waypoints) {
    points.push(wp.x, wp.y)
  }
  points.push(to.x, to.y)
  return points
}

export const DiagramEdge = memo(function DiagramEdge({
  edge,
  fromBounds,
  toBounds,
  isHighlighted,
  isDimmed,
}: DiagramEdgeProps) {
  const points = computePoints(edge, fromBounds, toBounds)
  const dash = edge.style === 'dashed' ? [...DASH_PATTERN] : undefined

  const baseColor = edge.color ?? '#475569'
  const color = isHighlighted ? '#fbbf24' : baseColor
  const opacity = isDimmed ? 0.15 : 1
  const strokeWidth = isHighlighted ? 2.5 : 1.5

  // Label position: midpoint between first and last point pair
  const labelX = (points[0] + points[points.length - 2]) / 2
  const labelY = (points[1] + points[points.length - 1]) / 2

  return (
    <>
      <Arrow
        points={points}
        stroke={color}
        strokeWidth={strokeWidth}
        fill={color}
        dash={dash}
        opacity={opacity}
        pointerLength={6}
        pointerWidth={5}
        hitStrokeWidth={12}
        tension={edge.tension ?? 0}
        listening={false}
      />
      {edge.label && (
        <Text
          x={labelX - 30}
          y={labelY - 14}
          width={60}
          text={edge.label}
          fontSize={9}
          fontFamily="ui-monospace, monospace"
          fill={isHighlighted ? '#fbbf24' : '#64748b'}
          opacity={isDimmed ? 0.15 : 0.8}
          align="center"
          listening={false}
        />
      )}
    </>
  )
})
