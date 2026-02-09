'use client'

import { useState } from 'react'
import { Circle, Line, Text } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * OverfittingWidget - Visualizes underfitting, good fit, and overfitting
 *
 * Shows the same data points with three different model complexities:
 * - Underfitting: straight line (too simple)
 * - Good fit: smooth curve that captures the pattern
 * - Overfitting: wiggly line that hits every point but doesn't generalize
 *
 * This helps build intuition for bias-variance tradeoff.
 */

// True underlying pattern: a quadratic with strong curvature for visual clarity
const TRUE_FUNCTION = (x: number) => 3.0 * (x - 0.5) ** 2 + 0.1

// Data points generated from TRUE_FUNCTION + small noise
// TRUE_FUNCTION values: 0.08→0.63, 0.25→0.29, 0.4→0.13, 0.55→0.11, 0.72→0.25, 0.9→0.58
const DATA_POINTS = [
  { x: 0.08, y: 0.65 },  // true 0.63, noise +0.02
  { x: 0.25, y: 0.32 },  // true 0.29, noise +0.03
  { x: 0.40, y: 0.11 },  // true 0.13, noise -0.02
  { x: 0.55, y: 0.15 },  // true 0.11, noise +0.04
  { x: 0.72, y: 0.28 },  // true 0.25, noise +0.03
  { x: 0.90, y: 0.62 },  // true 0.58, noise +0.04
]

// Model predictions for different complexity levels
const UNDERFIT_LINE = (x: number) => -0.025 * x + 0.37 // least-squares line—nearly flat, misses U-shape
const GOOD_FIT = (x: number) => 3.0 * (x - 0.5) ** 2 + 0.12 // smooth curve close to true pattern
const OVERFIT_CURVE = (x: number) => {
  // Lagrange interpolation - passes through every point exactly
  // This creates a wiggly curve that memorizes the data
  let result = 0
  for (let i = 0; i < DATA_POINTS.length; i++) {
    let term = DATA_POINTS[i].y
    for (let j = 0; j < DATA_POINTS.length; j++) {
      if (i !== j) {
        term *= (x - DATA_POINTS[j].x) / (DATA_POINTS[i].x - DATA_POINTS[j].x)
      }
    }
    result += term
  }
  return result
}

type FitType = 'underfit' | 'good' | 'overfit'

const FIT_CONFIG: Record<FitType, { label: string; color: string; fn: (x: number) => number; description: string }> = {
  underfit: {
    label: 'Underfitting',
    color: '#f97316', // orange
    fn: UNDERFIT_LINE,
    description: 'Too simple—misses the pattern',
  },
  good: {
    label: 'Good Fit',
    color: '#22c55e', // green
    fn: GOOD_FIT,
    description: 'Just right—captures the pattern',
  },
  overfit: {
    label: 'Overfitting',
    color: '#ef4444', // red
    fn: OVERFIT_CURVE,
    description: 'Too complex—memorizes noise',
  },
}

export function OverfittingWidget() {
  const [selectedFit, setSelectedFit] = useState<FitType>('underfit')
  const { containerRef, width } = useContainerWidth(500)

  const height = 450
  const margin = { left: 40, right: 20, top: 20, bottom: 40 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  // Scale functions
  const toX = (x: number) => margin.left + x * plotWidth
  const toY = (y: number) => margin.top + plotHeight - y * plotHeight

  // Generate curve points
  const generateCurvePoints = (fn: (x: number) => number, numPoints = 50) => {
    const points: number[] = []
    for (let i = 0; i <= numPoints; i++) {
      const x = i / numPoints
      const y = Math.max(0, Math.min(1, fn(x))) // clamp to visible range
      points.push(toX(x), toY(y))
    }
    return points
  }

  const config = FIT_CONFIG[selectedFit]

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      {/* Fit selector */}
      <div className="flex justify-center gap-2">
        {(Object.keys(FIT_CONFIG) as FitType[]).map((fit) => (
          <button
            key={fit}
            onClick={() => setSelectedFit(fit)}
            className={`cursor-pointer px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedFit === fit
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            }`}
          >
            {FIT_CONFIG[fit].label}
          </button>
        ))}
      </div>

      {/* Visualization */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Axes */}
          <Line
            points={[margin.left, height - margin.bottom, width - margin.right, height - margin.bottom]}
            stroke="#4a4a6a"
            strokeWidth={1}
          />
          <Line
            points={[margin.left, margin.top, margin.left, height - margin.bottom]}
            stroke="#4a4a6a"
            strokeWidth={1}
          />
          <Text x={width / 2} y={height - 12} text="Input (x)" fontSize={11} fill="#6b7280" />
          <Text x={8} y={height / 2 - 20} text="Output" fontSize={11} fill="#6b7280" rotation={-90} />

          {/* True function (dashed) */}
          <Line
            points={generateCurvePoints(TRUE_FUNCTION)}
            stroke="#8b5cf6"
            strokeWidth={2}
            dash={[6, 4]}
            opacity={0.5}
          />

          {/* Model curve */}
          <Line
            points={generateCurvePoints(config.fn)}
            stroke={config.color}
            strokeWidth={3}
          />

          {/* Data points */}
          {DATA_POINTS.map((point, i) => (
            <Circle
              key={i}
              x={toX(point.x)}
              y={toY(point.y)}
              radius={6}
              fill="#3b82f6"
              stroke="#93c5fd"
              strokeWidth={2}
            />
          ))}

          {/* Legend */}
          <Line points={[width - 120, 30, width - 100, 30]} stroke="#8b5cf6" strokeWidth={2} dash={[6, 4]} />
          <Text x={width - 95} y={24} text="True pattern" fontSize={10} fill="#8b5cf6" />

          <Line points={[width - 120, 48, width - 100, 48]} stroke={config.color} strokeWidth={3} />
          <Text x={width - 95} y={42} text="Model" fontSize={10} fill={config.color} />
        </ZoomableCanvas>
      </div>

      {/* Description */}
      <div
        className="p-4 rounded-lg border text-center"
        style={{ borderColor: config.color + '40', backgroundColor: config.color + '10' }}
      >
        <p className="font-medium" style={{ color: config.color }}>
          {config.label}
        </p>
        <p className="text-sm text-muted-foreground mt-1">{config.description}</p>
      </div>

      {/* Comparison table */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="p-2 rounded bg-orange-500/10 border border-orange-500/20 text-center">
          <p className="font-medium text-orange-400">Underfitting</p>
          <p className="text-muted-foreground">High bias</p>
          <p className="text-muted-foreground">Bad on train & test</p>
        </div>
        <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-center">
          <p className="font-medium text-emerald-400">Good Fit</p>
          <p className="text-muted-foreground">Balanced</p>
          <p className="text-muted-foreground">Good on both</p>
        </div>
        <div className="p-2 rounded bg-rose-500/10 border border-rose-500/20 text-center">
          <p className="font-medium text-rose-400">Overfitting</p>
          <p className="text-muted-foreground">High variance</p>
          <p className="text-muted-foreground">Good train, bad test</p>
        </div>
      </div>
    </div>
  )
}
