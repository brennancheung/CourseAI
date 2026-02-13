/**
 * Compute the point where a ray from a rectangle's center to a target
 * exits the rectangle boundary.
 */
export function getEdgePoint(
  rect: { x: number; y: number; width: number; height: number },
  target: { x: number; y: number },
): { x: number; y: number } {
  const hw = rect.width / 2
  const hh = rect.height / 2
  const dx = target.x - rect.x
  const dy = target.y - rect.y

  // Target is at center — return center
  if (dx === 0 && dy === 0) {
    return { x: rect.x, y: rect.y }
  }

  // Calculate intersection with each edge
  const absDx = Math.abs(dx)
  const absDy = Math.abs(dy)

  // Check if ray hits left/right edge or top/bottom edge
  const scaleX = absDx === 0 ? Infinity : hw / absDx
  const scaleY = absDy === 0 ? Infinity : hh / absDy

  const scale = Math.min(scaleX, scaleY)

  return {
    x: rect.x + dx * scale,
    y: rect.y + dy * scale,
  }
}
