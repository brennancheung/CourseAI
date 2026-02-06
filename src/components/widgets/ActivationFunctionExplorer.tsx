'use client'

import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { Line, Circle, Text, Arrow } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'

/**
 * ActivationFunctionExplorer - Interactive widget for exploring activation functions
 *
 * Single-function mode: Shows one function at a time, clear and focused.
 * Use the selector to switch between functions.
 */

type ActivationFunctionExplorerProps = {
  /** Which function to show initially */
  defaultFunction?: ActivationFunctionName
  /** Show derivatives toggle */
  showDerivatives?: boolean
  /** Width override (used by ExercisePanel in fullscreen) */
  width?: number
  /** Height override (used by ExercisePanel in fullscreen) */
  height?: number
  /** Initial input value */
  initialX?: number
}

export type ActivationFunctionName =
  | 'sigmoid'
  | 'tanh'
  | 'relu'
  | 'leaky-relu'
  | 'gelu'
  | 'swish'
  | 'linear'

type ActivationFunction = {
  name: ActivationFunctionName
  label: string
  color: string
  fn: (x: number) => number
  derivative: (x: number) => number
  formula: string
}

// Activation function definitions
const ACTIVATION_FUNCTIONS: Record<ActivationFunctionName, ActivationFunction> = {
  linear: {
    name: 'linear',
    label: 'Linear (no activation)',
    color: '#888888',
    fn: (x) => x,
    derivative: () => 1,
    formula: 'f(x) = x',
  },
  sigmoid: {
    name: 'sigmoid',
    label: 'Sigmoid',
    color: '#f97316',
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
      const s = 1 / (1 + Math.exp(-x))
      return s * (1 - s)
    },
    formula: 'σ(x) = 1 / (1 + e⁻ˣ)',
  },
  tanh: {
    name: 'tanh',
    label: 'Tanh',
    color: '#8b5cf6',
    fn: (x) => Math.tanh(x),
    derivative: (x) => 1 - Math.pow(Math.tanh(x), 2),
    formula: 'tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)',
  },
  relu: {
    name: 'relu',
    label: 'ReLU',
    color: '#22c55e',
    fn: (x) => Math.max(0, x),
    derivative: (x) => (x > 0 ? 1 : 0),
    formula: 'ReLU(x) = max(0, x)',
  },
  'leaky-relu': {
    name: 'leaky-relu',
    label: 'Leaky ReLU',
    color: '#06b6d4',
    fn: (x) => (x > 0 ? x : 0.1 * x),
    derivative: (x) => (x > 0 ? 1 : 0.1),
    formula: 'LeakyReLU(x) = max(0.1x, x)',
  },
  gelu: {
    name: 'gelu',
    label: 'GELU',
    color: '#ec4899',
    fn: (x) => {
      return x * (1 / (1 + Math.exp(-1.702 * x)))
    },
    derivative: (x) => {
      const sig = 1 / (1 + Math.exp(-1.702 * x))
      return sig + x * sig * (1 - sig) * 1.702
    },
    formula: 'GELU(x) ≈ x · σ(1.702x)',
  },
  swish: {
    name: 'swish',
    label: 'Swish',
    color: '#eab308',
    fn: (x) => x * (1 / (1 + Math.exp(-x))),
    derivative: (x) => {
      const sig = 1 / (1 + Math.exp(-x))
      return sig + x * sig * (1 - sig)
    },
    formula: 'Swish(x) = x · σ(x)',
  },
}

// Ordered list for selector
const FUNCTION_ORDER: ActivationFunctionName[] = [
  'linear',
  'sigmoid',
  'tanh',
  'relu',
  'leaky-relu',
  'gelu',
  'swish',
]

// Viewport sized to show full ReLU range (input goes -5 to 5, so output can be up to 5)
const VIEW = {
  xMin: -6,
  xMax: 6,
  yMin: -6,
  yMax: 6,
}

// Controls height for layout calculation
const CONTROLS_HEIGHT = 120

export function ActivationFunctionExplorer({
  defaultFunction = 'sigmoid',
  showDerivatives = false,
  width: widthOverride,
  height: heightOverride,
  initialX = 0,
}: ActivationFunctionExplorerProps) {
  const [selectedFunction, setSelectedFunction] = useState<ActivationFunctionName>(defaultFunction)
  const [inputX, setInputX] = useState(initialX)
  const [showDeriv, setShowDeriv] = useState(showDerivatives)

  // Measure container width
  const containerRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(500)

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

  // Use override width if provided (fullscreen), otherwise measured width
  const width = widthOverride ?? measuredWidth
  const canvasHeight = heightOverride
    ? Math.max(200, heightOverride - CONTROLS_HEIGHT)
    : 280

  // Current function
  const currentFn = ACTIVATION_FUNCTIONS[selectedFunction]

  // Coordinate transforms
  const toPixelX = useCallback(
    (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width,
    [width]
  )
  const toPixelY = useCallback(
    (y: number) => canvasHeight - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * canvasHeight,
    [canvasHeight]
  )

  // Generate curve points for a function
  const generateCurvePoints = useCallback(
    (fn: (x: number) => number) => {
      const points: number[] = []
      const step = (VIEW.xMax - VIEW.xMin) / 200

      for (let x = VIEW.xMin; x <= VIEW.xMax; x += step) {
        const y = fn(x)
        const clampedY = Math.max(VIEW.yMin, Math.min(VIEW.yMax, y))
        points.push(toPixelX(x), toPixelY(clampedY))
      }

      return points
    },
    [toPixelX, toPixelY]
  )

  // Grid lines
  const gridLines = useMemo(() => {
    const lines: { points: number[]; key: string }[] = []
    for (let x = -5; x <= 5; x++) {
      lines.push({ key: `v${x}`, points: [toPixelX(x), 0, toPixelX(x), canvasHeight] })
    }
    for (let y = -2; y <= 2; y++) {
      lines.push({ key: `h${y}`, points: [0, toPixelY(y), width, toPixelY(y)] })
    }
    return lines
  }, [toPixelX, toPixelY, width, canvasHeight])

  // Calculate output for current input
  const output = currentFn.fn(inputX)
  const derivative = currentFn.derivative(inputX)

  return (
    <div ref={containerRef} className="flex flex-col h-full">
      {/* Graph */}
      <div className="rounded-lg border bg-card overflow-hidden flex-shrink-0">
        <ZoomableCanvas width={width} height={canvasHeight} backgroundColor="#1a1a2e">
          {/* Grid */}
          {gridLines.map((line) => (
            <Line
              key={line.key}
              points={line.points}
              stroke="#333355"
              strokeWidth={1}
            />
          ))}

          {/* Axes - thicker */}
          <Arrow
            points={[0, toPixelY(0), width, toPixelY(0)]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />
          <Arrow
            points={[toPixelX(0), canvasHeight, toPixelX(0), 0]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />

          {/* Axis labels */}
          <Text x={width - 20} y={toPixelY(0) + 10} text="x" fontSize={14} fill="#888" />
          <Text x={toPixelX(0) + 10} y={10} text="y" fontSize={14} fill="#888" />

          {/* X axis numbers */}
          {[-4, -2, 2, 4].map((x) => (
            <Text
              key={`xl${x}`}
              x={toPixelX(x) - 6}
              y={toPixelY(0) + 8}
              text={x.toString()}
              fontSize={11}
              fill="#888"
            />
          ))}

          {/* Y axis numbers */}
          {[-1, 1].map((y) => (
            <Text
              key={`yl${y}`}
              x={toPixelX(0) + 8}
              y={toPixelY(y) - 5}
              text={y.toString()}
              fontSize={11}
              fill="#888"
            />
          ))}

          {/* Reference lines at y=0 and y=1 for sigmoid/tanh */}
          <Line
            points={[0, toPixelY(1), width, toPixelY(1)]}
            stroke="#444466"
            strokeWidth={1}
            dash={[4, 4]}
          />

          {/* Function curve */}
          <Line
            points={generateCurvePoints(currentFn.fn)}
            stroke={currentFn.color}
            strokeWidth={3}
            lineCap="round"
            lineJoin="round"
          />

          {/* Derivative curve (if enabled) */}
          {showDeriv && (
            <Line
              points={generateCurvePoints(currentFn.derivative)}
              stroke={currentFn.color}
              strokeWidth={2}
              dash={[8, 4]}
              opacity={0.6}
              lineCap="round"
              lineJoin="round"
            />
          )}

          {/* Input marker line */}
          <Line
            points={[toPixelX(inputX), 0, toPixelX(inputX), canvasHeight]}
            stroke="#ffffff"
            strokeWidth={1}
            dash={[4, 4]}
            opacity={0.5}
          />

          {/* Output point */}
          <Circle
            x={toPixelX(inputX)}
            y={toPixelY(Math.max(VIEW.yMin, Math.min(VIEW.yMax, output)))}
            radius={8}
            fill={currentFn.color}
            stroke="white"
            strokeWidth={2}
          />
        </ZoomableCanvas>
      </div>

      {/* Controls */}
      <div className="pt-3 space-y-3 flex-shrink-0">
        {/* Function selector */}
        <div className="flex flex-wrap gap-1.5 justify-center">
          {FUNCTION_ORDER.map((name) => {
            const fn = ACTIVATION_FUNCTIONS[name]
            const isSelected = selectedFunction === name

            return (
              <button
                key={name}
                onClick={() => setSelectedFunction(name)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
                  isSelected
                    ? 'ring-2 ring-offset-1 ring-offset-background'
                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                }`}
                style={
                  isSelected
                    ? {
                        backgroundColor: `${fn.color}20`,
                        color: fn.color,
                        // @ts-expect-error CSS custom property for ring color
                        '--tw-ring-color': fn.color,
                      }
                    : undefined
                }
              >
                {fn.label}
              </button>
            )
          })}
        </div>

        {/* Input slider with output display */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground w-12">x:</span>
          <input
            type="range"
            min={-5}
            max={5}
            step="0.1"
            value={inputX}
            onChange={(e) => setInputX(parseFloat(e.target.value))}
            className="flex-1"
          />
          <div className="text-right min-w-[140px]">
            <span className="font-mono text-sm">{inputX.toFixed(1)}</span>
            <span className="text-muted-foreground mx-1">→</span>
            <span className="font-mono text-sm font-bold" style={{ color: currentFn.color }}>
              {output.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Formula and derivative toggle */}
        <div className="flex items-center justify-between text-xs">
          <code className="px-2 py-1 rounded bg-muted/50 text-muted-foreground">
            {currentFn.formula}
          </code>
          <label className="flex items-center gap-2 text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={showDeriv}
              onChange={(e) => setShowDeriv(e.target.checked)}
              className="rounded h-3 w-3"
            />
            Show derivative
            {showDeriv && (
              <span className="font-mono" style={{ color: currentFn.color }}>
                f&apos;({inputX.toFixed(1)}) = {derivative.toFixed(3)}
              </span>
            )}
          </label>
        </div>
      </div>
    </div>
  )
}
