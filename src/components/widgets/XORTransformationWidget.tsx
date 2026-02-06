'use client'

import { useRef, useEffect, useState } from 'react'
import { Circle, Line, Text } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * XORTransformationWidget - Shows how the hidden layer transforms space
 *
 * The key insight: neural networks don't "draw multiple lines" in input space.
 * They TRANSFORM the input space so that a single line works.
 *
 * Left panel: Original (A, B) space - XOR points NOT linearly separable
 * Right panel: Transformed (h₁, h₂) space - same points, NOW separable
 */

// XOR data points
const XOR_POINTS = [
  { a: 0, b: 0, label: 0 },
  { a: 0, b: 1, label: 1 },
  { a: 1, b: 0, label: 1 },
  { a: 1, b: 1, label: 0 },
]

// Hidden layer transformation:
// h1 = ReLU(A + B - 0.5)  -- fires when at least one input is ~1
// h2 = ReLU(A + B - 1.5)  -- fires only when both inputs are ~1
function transform(a: number, b: number): { h1: number; h2: number } {
  const h1 = Math.max(0, a + b - 0.5)
  const h2 = Math.max(0, a + b - 1.5)
  return { h1, h2 }
}

// Pre-compute transformed points
const TRANSFORMED_POINTS = XOR_POINTS.map((p) => ({
  ...p,
  ...transform(p.a, p.b),
}))

export function XORTransformationWidget() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateWidth = () => {
      setWidth(container.getBoundingClientRect().width)
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  const panelWidth = Math.min(280, (width - 40) / 2)
  const panelHeight = 240

  // Coordinate system helpers for left panel (0-1 range)
  const leftMargin = 40
  const leftScale = (panelWidth - 50) / 1.2
  const toLeftX = (a: number) => leftMargin + (a + 0.1) * leftScale
  const toLeftY = (b: number) => panelHeight - 30 - (b + 0.1) * leftScale

  // Coordinate system helpers for right panel (0-2 range for h1, 0-1 for h2)
  const rightMargin = 40
  const rightScaleX = (panelWidth - 50) / 2
  const rightScaleY = (panelHeight - 60) / 1
  const toRightX = (h1: number) => rightMargin + h1 * rightScaleX
  const toRightY = (h2: number) => panelHeight - 30 - h2 * rightScaleY

  const labelColor = (label: number) => (label === 1 ? '#f97316' : '#3b82f6')
  const labelStroke = (label: number) => (label === 1 ? '#fdba74' : '#93c5fd')

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      <div className="flex flex-wrap justify-center gap-4">
        {/* Left Panel: Original Space */}
        <div className="flex flex-col items-center">
          <p className="text-sm font-medium text-rose-400 mb-2">
            Original Space (A, B)
          </p>
          <div className="rounded-lg border bg-card overflow-hidden">
            <ZoomableCanvas
              width={panelWidth}
              height={panelHeight}
              backgroundColor="#1a1a2e"
            >
              {/* Axes */}
              <Line
                points={[leftMargin, panelHeight - 30, panelWidth - 10, panelHeight - 30]}
                stroke="#4a4a6a"
                strokeWidth={1}
              />
              <Line
                points={[leftMargin, panelHeight - 30, leftMargin, 20]}
                stroke="#4a4a6a"
                strokeWidth={1}
              />
              <Text x={panelWidth - 20} y={panelHeight - 25} text="A" fontSize={12} fill="#6b7280" />
              <Text x={leftMargin - 5} y={10} text="B" fontSize={12} fill="#6b7280" />

              {/* Grid lines at 0 and 1 */}
              <Line
                points={[toLeftX(0), panelHeight - 30, toLeftX(0), 20]}
                stroke="#4a4a6a"
                strokeWidth={1}
                dash={[2, 4]}
              />
              <Line
                points={[toLeftX(1), panelHeight - 30, toLeftX(1), 20]}
                stroke="#4a4a6a"
                strokeWidth={1}
                dash={[2, 4]}
              />
              <Line
                points={[leftMargin, toLeftY(0), panelWidth - 10, toLeftY(0)]}
                stroke="#4a4a6a"
                strokeWidth={1}
                dash={[2, 4]}
              />
              <Line
                points={[leftMargin, toLeftY(1), panelWidth - 10, toLeftY(1)]}
                stroke="#4a4a6a"
                strokeWidth={1}
                dash={[2, 4]}
              />

              {/* Axis labels */}
              <Text x={toLeftX(0) - 3} y={panelHeight - 18} text="0" fontSize={10} fill="#6b7280" />
              <Text x={toLeftX(1) - 3} y={panelHeight - 18} text="1" fontSize={10} fill="#6b7280" />
              <Text x={leftMargin - 15} y={toLeftY(0) - 5} text="0" fontSize={10} fill="#6b7280" />
              <Text x={leftMargin - 15} y={toLeftY(1) - 5} text="1" fontSize={10} fill="#6b7280" />

              {/* Show a failed attempt at drawing a line */}
              <Line
                points={[toLeftX(-0.1), toLeftY(0.5), toLeftX(1.1), toLeftY(0.5)]}
                stroke="#ef4444"
                strokeWidth={2}
                dash={[6, 4]}
                opacity={0.5}
              />
              <Text
                x={toLeftX(0.3)}
                y={toLeftY(0.7)}
                text="No line works!"
                fontSize={11}
                fill="#ef4444"
              />

              {/* XOR Points */}
              {XOR_POINTS.map((p, i) => (
                <Circle
                  key={i}
                  x={toLeftX(p.a)}
                  y={toLeftY(p.b)}
                  radius={18}
                  fill={labelColor(p.label)}
                  stroke={labelStroke(p.label)}
                  strokeWidth={2}
                />
              ))}
              {XOR_POINTS.map((p, i) => (
                <Text
                  key={`label-${i}`}
                  x={toLeftX(p.a) - 4}
                  y={toLeftY(p.b) - 6}
                  text={String(p.label)}
                  fontSize={12}
                  fill="white"
                  fontStyle="bold"
                />
              ))}
            </ZoomableCanvas>
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-center max-w-[260px]">
            Can&apos;t draw one line to separate orange from blue
          </p>
        </div>

        {/* Arrow between panels */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-2xl text-emerald-400">→</div>
          <p className="text-xs text-emerald-400 text-center max-w-[80px]">
            Hidden layer transforms
          </p>
        </div>

        {/* Right Panel: Transformed Space */}
        <div className="flex flex-col items-center">
          <p className="text-sm font-medium text-emerald-400 mb-2">
            Transformed Space (h₁, h₂)
          </p>
          <div className="rounded-lg border bg-card overflow-hidden">
            <ZoomableCanvas
              width={panelWidth}
              height={panelHeight}
              backgroundColor="#1a1a2e"
            >
              {/* Axes */}
              <Line
                points={[rightMargin, panelHeight - 30, panelWidth - 10, panelHeight - 30]}
                stroke="#4a4a6a"
                strokeWidth={1}
              />
              <Line
                points={[rightMargin, panelHeight - 30, rightMargin, 20]}
                stroke="#4a4a6a"
                strokeWidth={1}
              />
              <Text x={panelWidth - 20} y={panelHeight - 25} text="h₁" fontSize={12} fill="#6b7280" />
              <Text x={rightMargin - 5} y={10} text="h₂" fontSize={12} fill="#6b7280" />

              {/* Separating line: h1 - 3*h2 = 0.25 */}
              {/* At h2=0: h1=0.25. At h2=0.5: h1=1.75 */}
              <Line
                points={[
                  toRightX(0.25),
                  toRightY(0),
                  toRightX(1.75),
                  toRightY(0.5),
                ]}
                stroke="#22c55e"
                strokeWidth={3}
              />
              <Text
                x={toRightX(0.8)}
                y={toRightY(0.35)}
                text="One line works!"
                fontSize={11}
                fill="#22c55e"
              />

              {/* Transformed Points */}
              {TRANSFORMED_POINTS.map((p, i) => (
                <Circle
                  key={i}
                  x={toRightX(p.h1)}
                  y={toRightY(p.h2)}
                  radius={18}
                  fill={labelColor(p.label)}
                  stroke={labelStroke(p.label)}
                  strokeWidth={2}
                />
              ))}
              {TRANSFORMED_POINTS.map((p, i) => (
                <Text
                  key={`label-${i}`}
                  x={toRightX(p.h1) - 4}
                  y={toRightY(p.h2) - 6}
                  text={String(p.label)}
                  fontSize={12}
                  fill="white"
                  fontStyle="bold"
                />
              ))}

              {/* Point labels showing original coordinates */}
              <Text
                x={toRightX(0) + 12}
                y={toRightY(0) + 8}
                text="(0,0)"
                fontSize={9}
                fill="#6b7280"
              />
              <Text
                x={toRightX(0.5) + 12}
                y={toRightY(0) + 8}
                text="(0,1) & (1,0)"
                fontSize={9}
                fill="#6b7280"
              />
              <Text
                x={toRightX(1.5) - 35}
                y={toRightY(0.5) - 25}
                text="(1,1)"
                fontSize={9}
                fill="#6b7280"
              />
            </ZoomableCanvas>
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-center max-w-[260px]">
            Same points, new positions — now separable!
          </p>
        </div>
      </div>

      {/* Transformation formulas */}
      <div className="p-4 rounded-lg bg-muted/30 border">
        <p className="text-sm font-medium mb-3">The Hidden Layer Transformation:</p>
        <div className="grid gap-2 md:grid-cols-2 text-sm">
          <div className="p-2 rounded bg-violet-500/10 border border-violet-500/20">
            <p className="font-mono text-violet-300">h₁ = ReLU(A + B - 0.5)</p>
            <p className="text-xs text-muted-foreground mt-1">
              Fires when A + B {'>'} 0.5 (at least one input is 1)
            </p>
          </div>
          <div className="p-2 rounded bg-violet-500/10 border border-violet-500/20">
            <p className="font-mono text-violet-300">h₂ = ReLU(A + B - 1.5)</p>
            <p className="text-xs text-muted-foreground mt-1">
              Fires only when A + B {'>'} 1.5 (both inputs are 1)
            </p>
          </div>
        </div>
        <div className="mt-3 text-sm text-muted-foreground">
          <p>
            <strong>Notice:</strong> (0,1) and (1,0) land at the same spot — that&apos;s fine,
            they have the same label! The key is that <span className="text-emerald-400">(1,1)
            moved away</span> from where (0,0) is, making them separable.
          </p>
        </div>
      </div>

      {/* The insight */}
      <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
        <p className="text-sm">
          <strong className="text-emerald-400">The insight:</strong> The hidden layer
          doesn&apos;t &quot;draw multiple lines.&quot; It <em>transforms the geometry</em> of
          the problem. Points that were tangled together get pulled apart. Then the output
          layer draws <strong>one line</strong> in the new space.
        </p>
      </div>
    </div>
  )
}
