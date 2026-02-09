'use client'

import { useState, useMemo, useCallback, useEffect } from 'react'
import { Line, Circle, Text, Arrow, Rect } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * XORClassifierExplorer - Interactive widget demonstrating the XOR problem
 *
 * Shows the classic XOR dataset and lets the user try (and fail) to
 * separate the two classes with a single line. This motivates the need
 * for nonlinearity and multi-layer networks.
 *
 * XOR data:
 * (0,0) → 0 (class A)
 * (0,1) → 1 (class B)
 * (1,0) → 1 (class B)
 * (1,1) → 0 (class A)
 */

type XORClassifierExplorerProps = {
  /** Width of the visualization */
  width?: number
  /** Height of the visualization */
  height?: number
}

// XOR data points with their labels
const XOR_DATA = [
  { x: 0, y: 0, label: 0 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 0, label: 1 },
  { x: 1, y: 1, label: 0 },
]

// Viewport config - centered around the unit square
const VIEW = {
  xMin: -0.5,
  xMax: 1.5,
  yMin: -0.5,
  yMax: 1.5,
}

// Check which side of the line a point is on
// Returns true if point is on the "positive" side (above the line)
function isAboveLine(
  px: number,
  py: number,
  slope: number,
  intercept: number
): boolean {
  const lineY = slope * px + intercept
  return py > lineY
}

// Calculate classification accuracy
function calculateAccuracy(slope: number, intercept: number): number {
  let correct = 0

  for (const point of XOR_DATA) {
    const isAbove = isAboveLine(point.x, point.y, slope, intercept)
    // Let's say "above" = class 1, "below" = class 0
    const predictedLabel = isAbove ? 1 : 0
    if (predictedLabel === point.label) {
      correct++
    }
  }

  return correct / XOR_DATA.length
}

// Check if we need to flip the classification
function getBestAccuracy(slope: number, intercept: number): { accuracy: number; flipped: boolean } {
  const acc1 = calculateAccuracy(slope, intercept)
  const acc2 = 1 - acc1 // Flipping which side is which class

  if (acc2 > acc1) {
    return { accuracy: acc2, flipped: true }
  }
  return { accuracy: acc1, flipped: false }
}

export function XORClassifierExplorer({
  width: widthOverride,
  height = 400,
}: XORClassifierExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(500)
  const width = widthOverride ?? measuredWidth
  const [slope, setSlope] = useState(0)
  const [intercept, setIntercept] = useState(0.5)
  const [attempts, setAttempts] = useState(0)

  // Track attempts when user changes the line
  useEffect(() => {
    setAttempts((prev) => prev + 1)
  }, [slope, intercept])

  // Coordinate transforms
  const toPixelX = useCallback(
    (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width,
    [width]
  )
  const toPixelY = useCallback(
    (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height,
    [height]
  )

  // Calculate accuracy
  const { accuracy, flipped } = useMemo(
    () => getBestAccuracy(slope, intercept),
    [slope, intercept]
  )

  // Determine classification for each point based on current line
  const classifications = useMemo(() => {
    return XOR_DATA.map((point) => {
      let isAbove = isAboveLine(point.x, point.y, slope, intercept)
      if (flipped) isAbove = !isAbove
      const predicted = isAbove ? 1 : 0
      const correct = predicted === point.label
      return { ...point, predicted, correct }
    })
  }, [slope, intercept, flipped])

  // Line endpoints (extend beyond viewport)
  const lineY1 = slope * VIEW.xMin + intercept
  const lineY2 = slope * VIEW.xMax + intercept

  // Shading regions
  const abovePoints = [
    [VIEW.xMin, lineY1],
    [VIEW.xMax, lineY2],
    [VIEW.xMax, VIEW.yMax],
    [VIEW.xMin, VIEW.yMax],
  ]
    .map(([x, y]) => [toPixelX(x), toPixelY(y)])
    .flat()

  const belowPoints = [
    [VIEW.xMin, lineY1],
    [VIEW.xMax, lineY2],
    [VIEW.xMax, VIEW.yMin],
    [VIEW.xMin, VIEW.yMin],
  ]
    .map(([x, y]) => [toPixelX(x), toPixelY(y)])
    .flat()

  // Message based on accuracy and attempts
  const getMessage = () => {
    if (accuracy === 1) return 'Perfect! Wait... that should be impossible!'
    if (accuracy === 0.75) return 'Close! But one point is always wrong...'
    if (accuracy === 0.5) return 'Two points on each side — not separating by class'
    return 'Keep trying!'
  }

  const getMessageColor = () => {
    if (accuracy === 0.75) return 'text-amber-400'
    if (accuracy === 0.5) return 'text-rose-400'
    return 'text-muted-foreground'
  }

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Graph */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Region shading */}
          <Line
            points={abovePoints}
            closed
            fill={flipped ? 'rgba(99, 102, 241, 0.1)' : 'rgba(249, 115, 22, 0.1)'}
          />
          <Line
            points={belowPoints}
            closed
            fill={flipped ? 'rgba(249, 115, 22, 0.1)' : 'rgba(99, 102, 241, 0.1)'}
          />

          {/* Grid lines */}
          <Line
            points={[toPixelX(0), 0, toPixelX(0), height]}
            stroke="#333355"
            strokeWidth={1}
          />
          <Line
            points={[toPixelX(1), 0, toPixelX(1), height]}
            stroke="#333355"
            strokeWidth={1}
          />
          <Line
            points={[0, toPixelY(0), width, toPixelY(0)]}
            stroke="#333355"
            strokeWidth={1}
          />
          <Line
            points={[0, toPixelY(1), width, toPixelY(1)]}
            stroke="#333355"
            strokeWidth={1}
          />

          {/* Axes */}
          <Arrow
            points={[toPixelX(VIEW.xMin), toPixelY(0), toPixelX(VIEW.xMax), toPixelY(0)]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />
          <Arrow
            points={[toPixelX(0), toPixelY(VIEW.yMin), toPixelX(0), toPixelY(VIEW.yMax)]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />

          {/* Axis labels */}
          <Text x={toPixelX(1) - 4} y={toPixelY(0) + 12} text="1" fontSize={12} fill="#888" />
          <Text x={toPixelX(0) - 16} y={toPixelY(1) - 4} text="1" fontSize={12} fill="#888" />
          <Text x={width - 20} y={toPixelY(0) + 10} text="x" fontSize={14} fill="#888" />
          <Text x={toPixelX(0) + 10} y={10} text="y" fontSize={14} fill="#888" />

          {/* The decision boundary line */}
          <Line
            points={[
              toPixelX(VIEW.xMin),
              toPixelY(lineY1),
              toPixelX(VIEW.xMax),
              toPixelY(lineY2),
            ]}
            stroke="#22c55e"
            strokeWidth={3}
          />

          {/* Data points */}
          {classifications.map((point, i) => {
            const baseColor = point.label === 0 ? '#6366f1' : '#f97316'
            const ringColor = point.correct ? '#22c55e' : '#ef4444'

            return (
              <Circle
                key={i}
                x={toPixelX(point.x)}
                y={toPixelY(point.y)}
                radius={20}
                fill={baseColor}
                stroke={ringColor}
                strokeWidth={4}
                shadowColor="black"
                shadowBlur={8}
                shadowOpacity={0.4}
              />
            )
          })}

          {/* Point labels */}
          {XOR_DATA.map((point, i) => (
            <Text
              key={`label-${i}`}
              x={toPixelX(point.x) - 4}
              y={toPixelY(point.y) - 6}
              text={point.label.toString()}
              fontSize={14}
              fontStyle="bold"
              fill="white"
            />
          ))}

          {/* Legend */}
          <Rect x={10} y={10} width={100} height={60} fill="rgba(0,0,0,0.5)" cornerRadius={5} />
          <Circle x={25} y={30} radius={8} fill="#6366f1" />
          <Text x={40} y={24} text="Class 0" fontSize={12} fill="#ccc" />
          <Circle x={25} y={50} radius={8} fill="#f97316" />
          <Text x={40} y={44} text="Class 1" fontSize={12} fill="#ccc" />
        </ZoomableCanvas>
      </div>

      {/* Controls */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Slope</span>
            <span className="font-mono text-green-400">{slope.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="-5"
            max="5"
            step="0.1"
            value={slope}
            onChange={(e) => setSlope(parseFloat(e.target.value))}
            className="w-full accent-green-500"
          />
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Intercept</span>
            <span className="font-mono text-green-400">{intercept.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="-2"
            max="3"
            step="0.1"
            value={intercept}
            onChange={(e) => setIntercept(parseFloat(e.target.value))}
            className="w-full accent-green-500"
          />
        </div>
      </div>

      {/* Accuracy display */}
      <div className="flex flex-wrap items-center justify-center gap-4">
        <div className="px-4 py-2 rounded-lg bg-muted flex items-center gap-3">
          <span className="text-muted-foreground">Accuracy:</span>
          <span
            className={`font-mono font-bold text-lg ${
              accuracy === 1
                ? 'text-green-400'
                : accuracy >= 0.75
                  ? 'text-amber-400'
                  : 'text-rose-400'
            }`}
          >
            {(accuracy * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-muted-foreground">
            ({Math.round(accuracy * 4)}/4 correct)
          </span>
        </div>
      </div>

      {/* Message */}
      <div className="text-center">
        <p className={`text-sm ${getMessageColor()}`}>{getMessage()}</p>
        {attempts > 10 && accuracy < 1 && (
          <p className="text-xs text-muted-foreground mt-2">
            No matter how you position the line, you can&apos;t separate both classes perfectly.
            That&apos;s the XOR problem!
          </p>
        )}
      </div>

      {/* Legend explanation */}
      <div className="text-xs text-muted-foreground text-center space-y-1">
        <p>
          <span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-1 align-middle" />
          Green ring = correctly classified
          <span className="mx-2">|</span>
          <span className="inline-block w-3 h-3 rounded-full bg-rose-500 mr-1 align-middle" />
          Red ring = misclassified
        </p>
      </div>
    </div>
  )
}
