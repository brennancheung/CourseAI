'use client'

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, StepForward } from 'lucide-react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

/**
 * OptimizerExplorer - 2D contour loss landscape comparing optimizers
 *
 * Visualizes the difference between Vanilla SGD, Momentum, RMSProp, and Adam
 * on an elongated ravine-shaped loss landscape. Each optimizer produces a
 * visually distinct trajectory:
 * - Vanilla SGD: dramatic zigzag across the ravine
 * - Momentum: smooth sweeping path through the ravine
 * - RMSProp: adapts step sizes per direction
 * - Adam: combines both effects — smooth AND adaptive
 */

type OptimizerExplorerProps = {
  width?: number
  height?: number
}

type OptimizerType = 'sgd' | 'momentum' | 'rmsprop' | 'adam'

type OptimizerState = {
  x: number
  y: number
  // Momentum state
  vx: number
  vy: number
  // RMSProp / Adam state (second moment)
  sx: number
  sy: number
  // Adam bias correction step
  t: number
}

type PathPoint = { x: number; y: number }

// --- Elongated ravine loss landscape ---
// Rotated 30 degrees, 60:1 curvature ratio. Parameters tuned so that:
// - SGD at lr=0.30 produces dramatic zigzag (steep eigenvalue multiplier = -0.8)
// - Momentum at lr=0.03 produces smooth sweeping arcs
// - RMSProp at lr=0.03 produces direct path with adaptive step sizes
// - Adam at lr=0.30 produces smooth approach with mild spiral
const ANGLE = Math.PI / 6
const COS_A = Math.cos(ANGLE)
const SIN_A = Math.sin(ANGLE)
const GENTLE_CURV = 0.05 // along ravine — slow convergence
const STEEP_CURV = 3.0 // across ravine — causes zigzag for SGD

// Minimum at origin
const MIN_X = 0
const MIN_Y = 0

function lossAt(x: number, y: number): number {
  const u = COS_A * x + SIN_A * y // along ravine (gentle)
  const v = -SIN_A * x + COS_A * y // across ravine (steep)
  return GENTLE_CURV * u * u + STEEP_CURV * v * v + 0.5
}

function lossGradient(x: number, y: number): [number, number] {
  const u = COS_A * x + SIN_A * y
  const v = -SIN_A * x + COS_A * y

  const du = 2 * GENTLE_CURV * u
  const dv = 2 * STEEP_CURV * v

  const gx = COS_A * du - SIN_A * dv
  const gy = SIN_A * du + COS_A * dv

  return [gx, gy]
}

// --- Contour rendering ---
const VIEW = { xMin: -2, xMax: 4, yMin: -2, yMax: 4 }

const OPTIMIZER_OPTIONS: { id: OptimizerType; label: string; color: string }[] = [
  { id: 'sgd', label: 'Vanilla SGD', color: '#ef4444' },
  { id: 'momentum', label: 'Momentum', color: '#22c55e' },
  { id: 'rmsprop', label: 'RMSProp', color: '#3b82f6' },
  { id: 'adam', label: 'Adam', color: '#a855f7' },
]

function optimizerColor(opt: OptimizerType): string {
  const found = OPTIMIZER_OPTIONS.find((o) => o.id === opt)
  return found?.color ?? '#ffffff'
}

// Color mapping for contour heatmap (same palette as SGDExplorer)
function lossToRGB(t: number): [number, number, number] {
  const s = Math.sqrt(Math.max(0, Math.min(1, t)))

  const stops = [
    { pos: 0.0, r: 15, g: 20, b: 80 },
    { pos: 0.25, r: 20, g: 70, b: 140 },
    { pos: 0.5, r: 15, g: 130, b: 130 },
    { pos: 0.75, r: 50, g: 170, b: 80 },
    { pos: 1.0, r: 220, g: 200, b: 50 },
  ]

  let lower = stops[0]
  let upper = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (s >= stops[i].pos && s <= stops[i + 1].pos) {
      lower = stops[i]
      upper = stops[i + 1]
      break
    }
  }

  const range = upper.pos - lower.pos
  const f = range === 0 ? 0 : (s - lower.pos) / range

  return [
    Math.round(lower.r + (upper.r - lower.r) * f),
    Math.round(lower.g + (upper.g - lower.g) * f),
    Math.round(lower.b + (upper.b - lower.b) * f),
  ]
}

function computeLossGrid(resolution: number) {
  const grid = new Float64Array(resolution * resolution)
  let minLoss = Infinity
  let maxLoss = -Infinity

  for (let iy = 0; iy < resolution; iy++) {
    for (let ix = 0; ix < resolution; ix++) {
      const x = VIEW.xMin + (VIEW.xMax - VIEW.xMin) * (ix / (resolution - 1))
      const y = VIEW.yMax - (VIEW.yMax - VIEW.yMin) * (iy / (resolution - 1))
      const l = lossAt(x, y)
      grid[iy * resolution + ix] = l
      minLoss = Math.min(minLoss, l)
      maxLoss = Math.max(maxLoss, l)
    }
  }

  return { grid, minLoss, maxLoss }
}

// Starting position: along ravine axis with perpendicular offset for visible zigzag
// u=3.0 along ravine, v=1.0 perpendicular offset
const START_X = 3 * COS_A + 1.0 * (-SIN_A) // ≈ 2.10
const START_Y = 3 * SIN_A + 1.0 * COS_A // ≈ 2.37
const BETA1 = 0.9
const BETA2 = 0.999
const EPSILON = 1e-8

// Per-optimizer default LRs — tuned via eigenvalue analysis to showcase each algorithm
// SGD: lr=0.30 puts steep eigenvalue at -0.8 → persistent zigzag with decay
// Momentum: lr=0.03 → sweeping arcs (classical momentum amplifies effective LR by ~10x)
// RMSProp: lr=0.03 → direct path (per-parameter normalization eliminates zigzag)
// Adam: lr=0.30 → smooth approach with mild spiral (momentum + normalization)
function defaultLrForOptimizer(opt: OptimizerType): number {
  const lrMap: Record<OptimizerType, number> = {
    sgd: 0.30,
    momentum: 0.03,
    rmsprop: 0.03,
    adam: 0.30,
  }
  return lrMap[opt]
}

function initialState(): OptimizerState {
  return {
    x: START_X,
    y: START_Y,
    vx: 0,
    vy: 0,
    sx: 0,
    sy: 0,
    t: 0,
  }
}

function optimizerStep(
  state: OptimizerState,
  optimizer: OptimizerType,
  lr: number
): OptimizerState {
  const [gx, gy] = lossGradient(state.x, state.y)
  const t = state.t + 1

  if (optimizer === 'sgd') {
    return {
      ...state,
      x: state.x - lr * gx,
      y: state.y - lr * gy,
      t,
    }
  }

  if (optimizer === 'momentum') {
    const vx = BETA1 * state.vx + gx
    const vy = BETA1 * state.vy + gy
    return {
      ...state,
      x: state.x - lr * vx,
      y: state.y - lr * vy,
      vx,
      vy,
      t,
    }
  }

  if (optimizer === 'rmsprop') {
    const sx = BETA2 * state.sx + (1 - BETA2) * gx * gx
    const sy = BETA2 * state.sy + (1 - BETA2) * gy * gy
    return {
      ...state,
      x: state.x - (lr * gx) / (Math.sqrt(sx) + EPSILON),
      y: state.y - (lr * gy) / (Math.sqrt(sy) + EPSILON),
      sx,
      sy,
      t,
    }
  }

  // Adam
  const vx = BETA1 * state.vx + (1 - BETA1) * gx
  const vy = BETA1 * state.vy + (1 - BETA1) * gy
  const sx = BETA2 * state.sx + (1 - BETA2) * gx * gx
  const sy = BETA2 * state.sy + (1 - BETA2) * gy * gy

  // Bias correction
  const vxHat = vx / (1 - Math.pow(BETA1, t))
  const vyHat = vy / (1 - Math.pow(BETA1, t))
  const sxHat = sx / (1 - Math.pow(BETA2, t))
  const syHat = sy / (1 - Math.pow(BETA2, t))

  return {
    ...state,
    x: state.x - (lr * vxHat) / (Math.sqrt(sxHat) + EPSILON),
    y: state.y - (lr * vyHat) / (Math.sqrt(syHat) + EPSILON),
    vx,
    vy,
    sx,
    sy,
    t,
  }
}

export function OptimizerExplorer({ width: widthOverride, height: heightOverride }: OptimizerExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth
  const contourSize = Math.min(width, heightOverride ?? 400)

  const [optimizer, setOptimizer] = useState<OptimizerType>('sgd')
  const [lr, setLr] = useState(defaultLrForOptimizer('sgd'))
  const [isRunning, setIsRunning] = useState(false)
  const [speed, setSpeed] = useState(150)
  const [path, setPath] = useState<PathPoint[]>([{ x: START_X, y: START_Y }])
  const [lossHistory, setLossHistory] = useState<number[]>([lossAt(START_X, START_Y)])
  const [stepCount, setStepCount] = useState(0)
  const optStateRef = useRef<OptimizerState>(initialState())
  const [displayState, setDisplayState] = useState<OptimizerState>(initialState())

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Contour data
  const resolution = 150
  const { grid, minLoss, maxLoss } = useMemo(() => computeLossGrid(resolution), [])

  const contourLevels = useMemo(() => {
    const levels: number[] = []
    const numContours = 16
    for (let i = 1; i < numContours; i++) {
      const t = i / numContours
      levels.push(minLoss + t * t * (maxLoss - minLoss))
    }
    return levels
  }, [minLoss, maxLoss])

  // Render heatmap to canvas
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = resolution
    canvas.height = resolution

    const imageData = ctx.createImageData(resolution, resolution)
    const range = maxLoss - minLoss

    for (let iy = 0; iy < resolution; iy++) {
      for (let ix = 0; ix < resolution; ix++) {
        const loss = grid[iy * resolution + ix]
        const t = range === 0 ? 0 : (loss - minLoss) / range

        let isContour = false
        for (const level of contourLevels) {
          const checkNeighbor = (ddx: number, ddy: number): boolean => {
            const nx = ix + ddx
            const ny = iy + ddy
            if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution) return false
            const neighborLoss = grid[ny * resolution + nx]
            return (loss - level) * (neighborLoss - level) < 0
          }

          if (checkNeighbor(1, 0) || checkNeighbor(0, 1)) {
            isContour = true
            break
          }
        }

        const [r, g, b] = lossToRGB(t)
        const idx = (iy * resolution + ix) * 4
        const boost = isContour ? 50 : 0

        imageData.data[idx] = Math.min(255, r + boost)
        imageData.data[idx + 1] = Math.min(255, g + boost)
        imageData.data[idx + 2] = Math.min(255, b + boost)
        imageData.data[idx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [grid, minLoss, maxLoss, contourLevels])

  // Coordinate transforms
  const toPixelX = useCallback(
    (px: number) => ((px - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * contourSize,
    [contourSize]
  )
  const toPixelY = useCallback(
    (py: number) => ((VIEW.yMax - py) / (VIEW.yMax - VIEW.yMin)) * contourSize,
    [contourSize]
  )

  // One optimization step
  const doStep = useCallback(() => {
    const newState = optimizerStep(optStateRef.current, optimizer, lr)

    // Clamp within view (with small margin)
    const cx = Math.max(VIEW.xMin + 0.1, Math.min(VIEW.xMax - 0.1, newState.x))
    const cy = Math.max(VIEW.yMin + 0.1, Math.min(VIEW.yMax - 0.1, newState.y))
    const clamped = { ...newState, x: cx, y: cy }

    optStateRef.current = clamped
    setDisplayState(clamped)
    setPath((prev) => [...prev, { x: cx, y: cy }])
    setLossHistory((prev) => [...prev, lossAt(cx, cy)])
    setStepCount((c) => c + 1)
  }, [optimizer, lr])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > speed) {
        doStep()
        lastStepTime.current = time
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, doStep, speed])

  // Stop after 300 steps
  useEffect(() => {
    if (stepCount >= 300) {
      setIsRunning(false)
    }
  }, [stepCount])

  const reset = useCallback(() => {
    setIsRunning(false)
    setPath([{ x: START_X, y: START_Y }])
    setLossHistory([lossAt(START_X, START_Y)])
    setStepCount(0)
    optStateRef.current = initialState()
    setDisplayState(initialState())
  }, [])

  const handleOptimizerChange = useCallback(
    (newOpt: OptimizerType) => {
      setOptimizer(newOpt)
      setLr(defaultLrForOptimizer(newOpt))
      // Reset when changing optimizer
      setIsRunning(false)
      setPath([{ x: START_X, y: START_Y }])
      setLossHistory([lossAt(START_X, START_Y)])
      setStepCount(0)
      optStateRef.current = initialState()
      setDisplayState(initialState())
    },
    []
  )

  const currentPos = path[path.length - 1]
  const currentLoss = lossAt(currentPos.x, currentPos.y)
  const color = optimizerColor(optimizer)

  // Loss curve dimensions
  const lossCurveWidth = width
  const lossCurveHeight = 100
  const lossMin = Math.min(...lossHistory)
  const lossMax = Math.max(...lossHistory)
  const lossRange = lossMax - lossMin || 1

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Contour plot */}
      <div className="flex justify-center">
        <div
          className="relative rounded-lg overflow-hidden border border-border/50"
          style={{ width: contourSize, height: contourSize }}
        >
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            style={{ imageRendering: 'auto' }}
          />

          {/* SVG overlay */}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox={`0 0 ${contourSize} ${contourSize}`}
          >
            {/* Optimization path */}
            {path.length > 1 && (
              <polyline
                points={path.map((p) => `${toPixelX(p.x)},${toPixelY(p.y)}`).join(' ')}
                fill="none"
                stroke={color}
                strokeWidth="2"
                opacity="0.8"
                strokeLinejoin="round"
              />
            )}

            {/* Path dots */}
            {path.map((p, i) => {
              const opacity = 0.15 + (0.85 * i) / path.length
              return (
                <circle
                  key={i}
                  cx={toPixelX(p.x)}
                  cy={toPixelY(p.y)}
                  r={i === path.length - 1 ? 5 : 1.5}
                  fill={i === path.length - 1 ? '#f97316' : color}
                  opacity={opacity}
                />
              )
            })}

            {/* Start marker */}
            <circle
              cx={toPixelX(START_X)}
              cy={toPixelY(START_Y)}
              r={4}
              fill="none"
              stroke="#f97316"
              strokeWidth="2"
            />
            <text
              x={toPixelX(START_X) + 10}
              y={toPixelY(START_Y) - 8}
              fill="#f97316"
              fontSize="11"
              fontWeight="500"
            >
              start
            </text>

            {/* Minimum label */}
            <circle
              cx={toPixelX(MIN_X)}
              cy={toPixelY(MIN_Y)}
              r={4}
              fill="#22c55e"
              opacity="0.8"
            />
            <text
              x={toPixelX(MIN_X)}
              y={toPixelY(MIN_Y) - 10}
              textAnchor="middle"
              fill="#22c55e"
              fontSize="11"
              fontWeight="500"
            >
              minimum
            </text>
          </svg>
        </div>
      </div>

      {/* Optimizer selector */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-sm text-muted-foreground">Optimizer:</span>
        {OPTIMIZER_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => handleOptimizerChange(opt.id)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              optimizer === opt.id
                ? 'text-white'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
            style={
              optimizer === opt.id
                ? { backgroundColor: opt.color }
                : undefined
            }
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Learning rate slider */}
      <div className="flex items-center justify-center gap-3">
        <span className="text-sm text-muted-foreground">Learning rate:</span>
        <input
          type="range"
          min="0.005"
          max="0.50"
          step="0.005"
          value={lr}
          onChange={(e) => {
            setLr(parseFloat(e.target.value))
            reset()
          }}
          className="w-32 cursor-pointer"
        />
        <span className="text-sm font-mono w-12">{lr.toFixed(3)}</span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
            disabled={stepCount >= 300}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
            {isRunning ? 'Pause' : 'Run'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setIsRunning(false)
              doStep()
            }}
            disabled={isRunning || stepCount >= 300}
          >
            <StepForward className="w-4 h-4 mr-1" />
            Step
          </Button>
          <Button variant="outline" size="sm" onClick={reset}>
            <RotateCcw className="w-4 h-4 mr-1" />
            Reset
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Speed:</span>
          <input
            type="range"
            min="30"
            max="400"
            step="30"
            value={400 - speed + 30}
            onChange={(e) => setSpeed(400 - parseFloat(e.target.value) + 30)}
            className="w-20 cursor-pointer"
          />
        </div>
      </div>

      {/* Training loss curve */}
      {lossHistory.length > 1 && (
        <div className="rounded-lg border bg-card p-3">
          <div className="text-xs text-muted-foreground mb-1">Training Loss</div>
          <svg
            viewBox={`0 0 ${lossCurveWidth} ${lossCurveHeight}`}
            className="w-full"
            style={{ height: lossCurveHeight }}
          >
            <rect
              x="0"
              y="0"
              width={lossCurveWidth}
              height={lossCurveHeight}
              fill="#1a1a2e"
              rx="4"
            />
            <polyline
              points={lossHistory
                .map((l, i) => {
                  const px = (i / (lossHistory.length - 1)) * (lossCurveWidth - 20) + 10
                  const py =
                    lossCurveHeight -
                    10 -
                    ((l - lossMin) / lossRange) * (lossCurveHeight - 20)
                  return `${px},${py}`
                })
                .join(' ')}
              fill="none"
              stroke={color}
              strokeWidth="1.5"
            />
          </svg>
        </div>
      )}

      {/* Stats */}
      <div className="flex flex-wrap gap-3 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Step: </span>
          <span className="font-mono">{stepCount}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Loss: </span>
          <span className="font-mono">{currentLoss.toFixed(4)}</span>
        </div>
        <div className="px-3 py-2 rounded-md" style={{ backgroundColor: `${color}20` }}>
          <span style={{ color }} className="font-medium">
            {OPTIMIZER_OPTIONS.find((o) => o.id === optimizer)?.label}
          </span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Position: </span>
          <span className="font-mono">
            ({currentPos.x.toFixed(2)}, {currentPos.y.toFixed(2)})
          </span>
        </div>
      </div>

      {/* Internal optimizer state */}
      {optimizer !== 'sgd' && stepCount > 0 && (
        <div className="rounded-lg border bg-card p-3">
          <div className="text-xs text-muted-foreground mb-2 font-medium">Internal State</div>
          <div className="flex flex-wrap gap-3 text-xs justify-center">
            {(optimizer === 'momentum' || optimizer === 'adam') && (
              <>
                <div className="px-2 py-1.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                  <span className="text-emerald-400">v<sub>x</sub>: </span>
                  <span className="font-mono">{displayState.vx.toFixed(4)}</span>
                </div>
                <div className="px-2 py-1.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                  <span className="text-emerald-400">v<sub>y</sub>: </span>
                  <span className="font-mono">{displayState.vy.toFixed(4)}</span>
                </div>
              </>
            )}
            {(optimizer === 'rmsprop' || optimizer === 'adam') && (
              <>
                <div className="px-2 py-1.5 rounded bg-blue-500/10 border border-blue-500/20">
                  <span className="text-blue-400">s<sub>x</sub>: </span>
                  <span className="font-mono">{displayState.sx.toFixed(4)}</span>
                </div>
                <div className="px-2 py-1.5 rounded bg-blue-500/10 border border-blue-500/20">
                  <span className="text-blue-400">s<sub>y</sub>: </span>
                  <span className="font-mono">{displayState.sy.toFixed(4)}</span>
                </div>
              </>
            )}
          </div>
          <p className="text-xs text-muted-foreground text-center mt-2">
            {optimizer === 'momentum' && 'Velocity (v) shows the accumulated gradient direction. Watch v_x and v_y to see which direction momentum builds.'}
            {optimizer === 'rmsprop' && 'Second moment (s) tracks gradient magnitude per direction. Larger s means smaller effective learning rate.'}
            {optimizer === 'adam' && 'Velocity (v) smooths direction, second moment (s) normalizes magnitude. Adam combines both.'}
          </p>
        </div>
      )}

      <p className="text-xs text-muted-foreground text-center">
        Compare how different optimizers navigate the elongated ravine.{' '}
        <span style={{ color: '#ef4444' }}>Vanilla SGD</span> zigzags,{' '}
        <span style={{ color: '#22c55e' }}>Momentum</span> smooths the path,{' '}
        <span style={{ color: '#3b82f6' }}>RMSProp</span> adapts step sizes, and{' '}
        <span style={{ color: '#a855f7' }}>Adam</span> combines both.
      </p>
    </div>
  )
}
