'use client'

import { useMemo, useCallback, useRef, useEffect, useState } from 'react'
import { Circle, Line, Text, Shape } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * XORSolverWidget - Shows how two decision boundaries solve XOR
 *
 * The key insight: a neural network with 2 hidden neurons learns TWO lines.
 * The region BETWEEN the lines correctly classifies XOR.
 *
 * - Line 1: x + y = 0.5 (separates bottom-left corner)
 * - Line 2: x + y = 1.5 (separates top-right corner)
 * - Between them: the XOR "true" region (orange)
 */

type XORSolverWidgetProps = {
  /** Width override */
  width?: number
  /** Height override */
  height?: number
}

// XOR training data
const XOR_POINTS = [
  { x: 0, y: 0, label: 0 },
  { x: 1, y: 0, label: 1 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 1, label: 0 },
]

// The two decision boundaries learned by hidden neurons
// Line 1: x + y = 0.5 (below this = class 0)
// Line 2: x + y = 1.5 (above this = class 0)
// Between them = class 1
const LINE_1_INTERCEPT = 0.5 // x + y = 0.5
const LINE_2_INTERCEPT = 1.5 // x + y = 1.5

// Viewport with padding
const VIEW = {
  xMin: -0.4,
  xMax: 1.4,
  yMin: -0.4,
  yMax: 1.4,
}

export function XORSolverWidget({
  width: widthOverride,
  height: heightOverride,
}: XORSolverWidgetProps) {
  // Measure container
  const containerRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(400)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateWidth = () => {
      const rect = container.getBoundingClientRect()
      setMeasuredWidth(rect.width)
    }

    updateWidth()

    const observer = new ResizeObserver(updateWidth)
    observer.observe(container)

    return () => observer.disconnect()
  }, [])

  const width = widthOverride ?? measuredWidth
  const height = heightOverride ?? Math.min(400, width)

  // Coordinate transforms
  const toPixelX = useCallback(
    (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width,
    [width]
  )
  const toPixelY = useCallback(
    (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height,
    [height]
  )

  // Calculate line endpoints
  // Line equation: x + y = c, so y = c - x
  const getLinePoints = useCallback(
    (intercept: number) => {
      // Find where line intersects the viewport edges
      const points: number[] = []

      // Try left edge (x = VIEW.xMin)
      const yAtLeft = intercept - VIEW.xMin
      if (yAtLeft >= VIEW.yMin && yAtLeft <= VIEW.yMax) {
        points.push(toPixelX(VIEW.xMin), toPixelY(yAtLeft))
      }

      // Try bottom edge (y = VIEW.yMin)
      const xAtBottom = intercept - VIEW.yMin
      if (xAtBottom >= VIEW.xMin && xAtBottom <= VIEW.xMax) {
        points.push(toPixelX(xAtBottom), toPixelY(VIEW.yMin))
      }

      // Try right edge (x = VIEW.xMax)
      const yAtRight = intercept - VIEW.xMax
      if (yAtRight >= VIEW.yMin && yAtRight <= VIEW.yMax) {
        points.push(toPixelX(VIEW.xMax), toPixelY(yAtRight))
      }

      // Try top edge (y = VIEW.yMax)
      const xAtTop = intercept - VIEW.yMax
      if (xAtTop >= VIEW.xMin && xAtTop <= VIEW.xMax) {
        points.push(toPixelX(xAtTop), toPixelY(VIEW.yMax))
      }

      return points.slice(0, 4) // Only need 2 points (4 values)
    },
    [toPixelX, toPixelY]
  )

  const line1Points = useMemo(() => getLinePoints(LINE_1_INTERCEPT), [getLinePoints])
  const line2Points = useMemo(() => getLinePoints(LINE_2_INTERCEPT), [getLinePoints])

  // Grid lines at 0 and 1
  const gridLines = useMemo(() => {
    return [
      { points: [toPixelX(0), 0, toPixelX(0), height], key: 'v0' },
      { points: [toPixelX(1), 0, toPixelX(1), height], key: 'v1' },
      { points: [0, toPixelY(0), width, toPixelY(0)], key: 'h0' },
      { points: [0, toPixelY(1), width, toPixelY(1)], key: 'h1' },
    ]
  }, [toPixelX, toPixelY, width, height])

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      {/* Canvas */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Shaded region between the two lines (the XOR "true" region) */}
          <Shape
            sceneFunc={(context, shape) => {
              context.beginPath()
              // Create a polygon for the band between the two lines
              // Line 1: x + y = 0.5 (from bottom-left to top-right-ish)
              // Line 2: x + y = 1.5 (from bottom-left-ish to top-right)

              // Band corners (clockwise from bottom)
              // Bottom edge intersection with line 1: (0.5, 0) - but clamped
              // Left edge intersection with line 1: (0, 0.5) - but clamped
              // Left edge intersection with line 2: (0, 1.5) - clamped to (0, 1.4)
              // Top edge intersection with line 2: (0.1, 1.4) approximate
              // etc.

              // Simpler: trace the polygon
              // Points where line 1 (x+y=0.5) intersects viewport
              const l1_left = Math.max(VIEW.yMin, 0.5 - VIEW.xMin)
              const l1_bottom = Math.max(VIEW.xMin, 0.5 - VIEW.yMin)
              // Points where line 2 (x+y=1.5) intersects viewport
              const l2_top = Math.min(VIEW.xMax, 1.5 - VIEW.yMax)
              const l2_right = Math.min(VIEW.yMax, 1.5 - VIEW.xMax)

              context.moveTo(toPixelX(VIEW.xMin), toPixelY(l1_left))
              context.lineTo(toPixelX(l1_bottom), toPixelY(VIEW.yMin))
              context.lineTo(toPixelX(VIEW.xMax), toPixelY(VIEW.yMin))
              context.lineTo(toPixelX(VIEW.xMax), toPixelY(l2_right))
              context.lineTo(toPixelX(l2_top), toPixelY(VIEW.yMax))
              context.lineTo(toPixelX(VIEW.xMin), toPixelY(VIEW.yMax))
              context.closePath()
              context.fillStrokeShape(shape)
            }}
            fill="rgba(249, 115, 22, 0.15)"
          />

          {/* Grid lines */}
          {gridLines.map((line) => (
            <Line
              key={line.key}
              points={line.points}
              stroke="#444466"
              strokeWidth={1}
              dash={[4, 4]}
            />
          ))}

          {/* Axis labels */}
          <Text x={toPixelX(0) - 15} y={toPixelY(0) + 8} text="0" fontSize={14} fill="#888" />
          <Text x={toPixelX(1) - 3} y={toPixelY(0) + 8} text="1" fontSize={14} fill="#888" />
          <Text x={toPixelX(0) - 20} y={toPixelY(1) - 5} text="1" fontSize={14} fill="#888" />
          <Text x={width - 15} y={toPixelY(0) + 8} text="A" fontSize={12} fill="#666" fontStyle="italic" />
          <Text x={toPixelX(0) + 8} y={8} text="B" fontSize={12} fill="#666" fontStyle="italic" />

          {/* Decision boundary line 1 */}
          <Line
            points={line1Points}
            stroke="#22c55e"
            strokeWidth={3}
            lineCap="round"
          />

          {/* Decision boundary line 2 */}
          <Line
            points={line2Points}
            stroke="#22c55e"
            strokeWidth={3}
            lineCap="round"
          />

          {/* Line labels */}
          <Text
            x={toPixelX(0.6)}
            y={toPixelY(0.1)}
            text="A + B = 0.5"
            fontSize={11}
            fill="#22c55e"
            rotation={-45}
          />
          <Text
            x={toPixelX(1.1)}
            y={toPixelY(0.6)}
            text="A + B = 1.5"
            fontSize={11}
            fill="#22c55e"
            rotation={-45}
          />

          {/* XOR points */}
          {XOR_POINTS.map((point, i) => (
            <Circle
              key={i}
              x={toPixelX(point.x)}
              y={toPixelY(point.y)}
              radius={18}
              fill={point.label === 1 ? '#f97316' : '#3b82f6'}
              stroke="white"
              strokeWidth={3}
            />
          ))}

          {/* Point labels */}
          {XOR_POINTS.map((point, i) => (
            <Text
              key={`label-${i}`}
              x={toPixelX(point.x) - 5}
              y={toPixelY(point.y) - 7}
              text={point.label.toString()}
              fontSize={14}
              fontStyle="bold"
              fill="white"
            />
          ))}
        </ZoomableCanvas>
      </div>

      {/* Legend and explanation */}
      <div className="space-y-3">
        <div className="flex items-center justify-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500 border-2 border-white" />
            <span className="text-muted-foreground">Output = 0</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-orange-500 border-2 border-white" />
            <span className="text-muted-foreground">Output = 1</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-3 bg-green-500 rounded" />
            <span className="text-muted-foreground">Decision boundaries</span>
          </div>
        </div>

        <p className="text-sm text-muted-foreground text-center">
          <strong>Two lines</strong> create a diagonal band. Points <em>between</em> the
          lines are classified as 1 (orange). Points <em>outside</em> are classified as 0 (blue).
        </p>
      </div>
    </div>
  )
}
