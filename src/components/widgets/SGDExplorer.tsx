'use client'

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, StepForward } from 'lucide-react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

/**
 * SGDExplorer - 2D contour loss landscape with batch size control
 *
 * Visualizes the difference between full-batch gradient descent and
 * mini-batch SGD on a loss landscape with two minima (one sharp, one wide).
 *
 * Key behaviors:
 * - Full-batch: smooth, deterministic path to nearest minimum
 * - Small batch: noisy, jittery path that sometimes escapes the sharp minimum
 * - Batch size slider: 1 / 8 / 32 / 128 / ALL
 * - Training loss curve shown below
 */

type SGDExplorerProps = {
  width?: number
  height?: number
}

// --- Loss landscape with two minima ---

// Wide minimum centered around (-1.5, -1), sharp minimum around (1.5, 1.2)
function lossAt(x: number, y: number): number {
  // Wide minimum: gentle Gaussian well
  const dx1 = x + 1.5
  const dy1 = y + 1.0
  const wide = 2.0 * Math.exp(-(dx1 * dx1 * 0.3 + dy1 * dy1 * 0.3))

  // Sharp minimum: tight Gaussian well
  const dx2 = x - 1.5
  const dy2 = y - 1.2
  const sharp = 2.5 * Math.exp(-(dx2 * dx2 * 1.5 + dy2 * dy2 * 1.5))

  // Base quadratic bowl to keep things bounded
  const base = 0.08 * (x * x + y * y) + 3.0

  return base - wide - sharp
}

// Analytical gradient of the loss
function lossGradient(x: number, y: number): [number, number] {
  const dx1 = x + 1.5
  const dy1 = y + 1.0
  const wide = 2.0 * Math.exp(-(dx1 * dx1 * 0.3 + dy1 * dy1 * 0.3))

  const dx2 = x - 1.5
  const dy2 = y - 1.2
  const sharp = 2.5 * Math.exp(-(dx2 * dx2 * 1.5 + dy2 * dy2 * 1.5))

  const gx = 0.16 * x - wide * (-0.6 * dx1) - sharp * (-3.0 * dx2)
  const gy = 0.16 * y - wide * (-0.6 * dy1) - sharp * (-3.0 * dy2)

  return [gx, gy]
}

// --- Contour rendering ---

const VIEW = {
  xMin: -4,
  xMax: 4,
  yMin: -4,
  yMax: 4,
}

const BATCH_SIZE_OPTIONS = [1, 8, 32, 128, -1] as const // -1 = ALL

function batchSizeLabel(bs: number): string {
  if (bs === -1) return 'ALL'
  return String(bs)
}

// Color mapping for contour heatmap
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

// Precompute the loss grid
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

// Starting position: near the sharp minimum (so noise can escape it)
const START_X = 1.0
const START_Y = 0.5
const LEARNING_RATE = 0.25
// Simulated "dataset" size for batch noise scaling
const DATASET_SIZE = 256

export function SGDExplorer({ width: widthOverride, height: heightOverride }: SGDExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth
  const contourSize = Math.min(width, heightOverride ?? 400)

  const [batchSize, setBatchSize] = useState<number>(32)
  const [isRunning, setIsRunning] = useState(false)
  const [speed, setSpeed] = useState(200) // ms between steps
  const [path, setPath] = useState<Array<{ x: number; y: number }>>([{ x: START_X, y: START_Y }])
  const [lossHistory, setLossHistory] = useState<number[]>([lossAt(START_X, START_Y)])
  const [stepCount, setStepCount] = useState(0)

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Contour data (memoized — expensive)
  const resolution = 150
  const { grid, minLoss, maxLoss } = useMemo(() => computeLossGrid(resolution), [])

  // Contour levels
  const contourLevels = useMemo(() => {
    const levels: number[] = []
    const numContours = 14
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
          const checkNeighbor = (dx: number, dy: number): boolean => {
            const nx = ix + dx
            const ny = iy + dy
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

        if (isContour) {
          imageData.data[idx] = Math.min(255, r + 50)
          imageData.data[idx + 1] = Math.min(255, g + 50)
          imageData.data[idx + 2] = Math.min(255, b + 50)
          imageData.data[idx + 3] = 255
        } else {
          imageData.data[idx] = r
          imageData.data[idx + 1] = g
          imageData.data[idx + 2] = b
          imageData.data[idx + 3] = 255
        }
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [grid, minLoss, maxLoss, contourLevels])

  // Coordinate transforms
  const toPixelX = useCallback(
    (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * contourSize,
    [contourSize]
  )
  const toPixelY = useCallback(
    (y: number) => ((VIEW.yMax - y) / (VIEW.yMax - VIEW.yMin)) * contourSize,
    [contourSize]
  )

  // One optimization step
  const step = useCallback(() => {
    setPath((prevPath) => {
      const current = prevPath[prevPath.length - 1]
      const [gx, gy] = lossGradient(current.x, current.y)

      // Add noise inversely proportional to batch size
      // noise_std ~ 1/sqrt(batchSize). Full batch = no noise.
      let noiseX = 0
      let noiseY = 0
      const effectiveBatch = batchSize === -1 ? DATASET_SIZE : batchSize
      if (effectiveBatch < DATASET_SIZE) {
        const noiseScale = 1.2 / Math.sqrt(effectiveBatch)
        // Box-Muller for Gaussian noise
        const u1 = Math.random()
        const u2 = Math.random()
        const mag = Math.sqrt(-2 * Math.log(u1 + 1e-10))
        noiseX = mag * Math.cos(2 * Math.PI * u2) * noiseScale
        noiseY = mag * Math.sin(2 * Math.PI * u2) * noiseScale
      }

      const nx = current.x - LEARNING_RATE * (gx + noiseX)
      const ny = current.y - LEARNING_RATE * (gy + noiseY)

      // Clamp within view
      const cx = Math.max(VIEW.xMin + 0.3, Math.min(VIEW.xMax - 0.3, nx))
      const cy = Math.max(VIEW.yMin + 0.3, Math.min(VIEW.yMax - 0.3, ny))

      const newPath = [...prevPath, { x: cx, y: cy }]
      setLossHistory((prev) => [...prev, lossAt(cx, cy)])
      setStepCount((c) => c + 1)
      return newPath
    })
  }, [batchSize])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > speed) {
        step()
        lastStepTime.current = time
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, step, speed])

  // Stop after 200 steps to prevent runaway
  useEffect(() => {
    if (stepCount >= 200) {
      setIsRunning(false)
    }
  }, [stepCount])

  const reset = useCallback(() => {
    setIsRunning(false)
    setPath([{ x: START_X, y: START_Y }])
    setLossHistory([lossAt(START_X, START_Y)])
    setStepCount(0)
  }, [])

  const currentPos = path[path.length - 1]
  const currentLoss = lossAt(currentPos.x, currentPos.y)

  // Loss curve dimensions
  const lossCurveWidth = width
  const lossCurveHeight = 100

  // Min/max for loss curve scaling
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

          {/* SVG overlay for path and markers */}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox={`0 0 ${contourSize} ${contourSize}`}
          >
            {/* Optimization path */}
            {path.length > 1 && (
              <polyline
                points={path.map((p) => `${toPixelX(p.x)},${toPixelY(p.y)}`).join(' ')}
                fill="none"
                stroke="white"
                strokeWidth="1.5"
                opacity="0.7"
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
                  r={i === path.length - 1 ? 5 : 2}
                  fill={i === path.length - 1 ? '#f97316' : '#ffffff'}
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

            {/* Wide minimum label */}
            <circle
              cx={toPixelX(-1.5)}
              cy={toPixelY(-1.0)}
              r={4}
              fill="#22c55e"
              opacity="0.8"
            />
            <text
              x={toPixelX(-1.5)}
              y={toPixelY(-1.0) - 10}
              textAnchor="middle"
              fill="#22c55e"
              fontSize="11"
              fontWeight="500"
            >
              wide min
            </text>

            {/* Sharp minimum label */}
            <circle
              cx={toPixelX(1.5)}
              cy={toPixelY(1.2)}
              r={4}
              fill="#ef4444"
              opacity="0.8"
            />
            <text
              x={toPixelX(1.5)}
              y={toPixelY(1.2) - 10}
              textAnchor="middle"
              fill="#ef4444"
              fontSize="11"
              fontWeight="500"
            >
              sharp min
            </text>
          </svg>
        </div>
      </div>

      {/* Batch size selector */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-sm text-muted-foreground">Batch size:</span>
        {BATCH_SIZE_OPTIONS.map((bs) => (
          <button
            key={bs}
            onClick={() => { reset(); setBatchSize(bs) }}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              batchSize === bs
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            {batchSizeLabel(bs)}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
            disabled={stepCount >= 200}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
            {isRunning ? 'Pause' : 'Run'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => { setIsRunning(false); step() }}
            disabled={isRunning || stepCount >= 200}
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
            min="50"
            max="500"
            step="50"
            value={500 - speed + 50}
            onChange={(e) => setSpeed(500 - parseFloat(e.target.value) + 50)}
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
            {/* Background */}
            <rect
              x="0"
              y="0"
              width={lossCurveWidth}
              height={lossCurveHeight}
              fill="#1a1a2e"
              rx="4"
            />
            {/* Loss line */}
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
              stroke="#6366f1"
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
          <span className="font-mono">{currentLoss.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Batch: </span>
          <span className="font-mono">{batchSizeLabel(batchSize)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Position: </span>
          <span className="font-mono">
            ({currentPos.x.toFixed(2)}, {currentPos.y.toFixed(2)})
          </span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Watch how batch size affects the optimization path. Small batches create noisy paths that
        can escape the{' '}
        <span className="text-red-400">sharp minimum</span> and find the{' '}
        <span className="text-green-400">wide minimum</span>. Full-batch follows a smooth path to
        the nearest minimum.
      </p>
    </div>
  )
}
