'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Line, Circle, Text, Arrow, Group } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * GradientDescentExplorer - Animated gradient descent on 1D loss curve
 *
 * Built with react-konva for reliable canvas rendering.
 * Shows a ball rolling downhill on a loss function, demonstrating:
 * - Gradient as the slope at current position
 * - Update step moving opposite to gradient
 * - Convergence to minimum
 */

type GradientDescentExplorerProps = {
  /** Allow adjusting learning rate */
  showLearningRateSlider?: boolean
  /** Initial learning rate */
  initialLearningRate?: number
  /** Initial parameter value (x position) */
  initialPosition?: number
  /** Show the gradient arrow */
  showGradientArrow?: boolean
  /** Width of canvas */
  width?: number
  /** Height of canvas */
  height?: number
}

// Loss function: L(x) = (x - 1)^2 + 0.5, minimum at x=1
const loss = (x: number) => Math.pow(x - 1, 2) + 0.5
const lossDerivative = (x: number) => 2 * (x - 1)

// Viewport config
const VIEW = {
  xMin: -4,
  xMax: 6,
  yMin: -0.5,
  yMax: 10,
}

export function GradientDescentExplorer({
  showLearningRateSlider = true,
  initialLearningRate = 0.15,
  initialPosition = -2.5,
  showGradientArrow = true,
  width: widthOverride,
  height = 350,
}: GradientDescentExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [position, setPosition] = useState(initialPosition)
  const [learningRate, setLearningRate] = useState(initialLearningRate)
  const [isRunning, setIsRunning] = useState(false)
  const [stepCount, setStepCount] = useState(0)
  const [history, setHistory] = useState<number[]>([initialPosition])

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Coordinate transforms
  const toPixelX = (x: number) => ((x - VIEW.xMin) / (VIEW.xMax - VIEW.xMin)) * width
  const toPixelY = (y: number) => height - ((y - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * height

  // Current values
  const currentLoss = loss(position)
  const currentGradient = lossDerivative(position)

  // Single step
  const step = useCallback(() => {
    setPosition((pos) => {
      const grad = lossDerivative(pos)
      const newPos = pos - learningRate * grad
      // Clamp to viewport
      const clamped = Math.max(VIEW.xMin + 0.5, Math.min(VIEW.xMax - 0.5, newPos))
      setHistory((h) => [...h.slice(-30), clamped])
      setStepCount((c) => c + 1)
      return clamped
    })
  }, [learningRate])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > 400) {
        step()
        lastStepTime.current = time
      }
      // Stop if converged
      if (Math.abs(lossDerivative(position)) < 0.05) {
        setIsRunning(false)
        return
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, step, position])

  // Reset
  const reset = useCallback(() => {
    setIsRunning(false)
    setPosition(initialPosition)
    setHistory([initialPosition])
    setStepCount(0)
  }, [initialPosition])

  // Generate curve points
  const curvePoints: number[] = []
  for (let i = 0; i <= 100; i++) {
    const x = VIEW.xMin + (i / 100) * (VIEW.xMax - VIEW.xMin)
    const y = loss(x)
    if (y <= VIEW.yMax) {
      curvePoints.push(toPixelX(x), toPixelY(y))
    }
  }

  // Grid lines
  const gridLines: { points: number[]; key: string }[] = []
  for (let x = Math.ceil(VIEW.xMin); x <= VIEW.xMax; x++) {
    gridLines.push({
      key: `v${x}`,
      points: [toPixelX(x), 0, toPixelX(x), height],
    })
  }
  for (let y = Math.ceil(VIEW.yMin); y <= VIEW.yMax; y += 2) {
    gridLines.push({
      key: `h${y}`,
      points: [0, toPixelY(y), width, toPixelY(y)],
    })
  }

  const ballX = toPixelX(position)
  const ballY = toPixelY(currentLoss)

  return (
    <div ref={containerRef} className="space-y-4">
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
            {/* Grid */}
            {gridLines.map((line) => (
              <Line
                key={line.key}
                points={line.points}
                stroke="#333355"
                strokeWidth={1}
              />
            ))}

            {/* Axes */}
            <Arrow
              points={[0, toPixelY(0), width, toPixelY(0)]}
              stroke="#666688"
              strokeWidth={2}
              fill="#666688"
              pointerLength={8}
              pointerWidth={6}
            />
            <Arrow
              points={[toPixelX(0), height, toPixelX(0), 0]}
              stroke="#666688"
              strokeWidth={2}
              fill="#666688"
              pointerLength={8}
              pointerWidth={6}
            />

            {/* Axis labels */}
            <Text x={width - 25} y={toPixelY(0) + 10} text="θ" fontSize={14} fill="#888" />
            <Text x={toPixelX(0) + 10} y={10} text="L(θ)" fontSize={14} fill="#888" />

            {/* X axis numbers */}
            {[-2, 0, 2, 4].map((x) => (
              <Text
                key={`xl${x}`}
                x={toPixelX(x) - 8}
                y={toPixelY(0) + 8}
                text={x.toString()}
                fontSize={12}
                fill="#888"
              />
            ))}

            {/* Y axis numbers */}
            {[2, 4, 6, 8].map((y) => (
              <Text
                key={`yl${y}`}
                x={toPixelX(0) + 8}
                y={toPixelY(y) - 6}
                text={y.toString()}
                fontSize={12}
                fill="#888"
              />
            ))}

            {/* Loss curve */}
            <Line
              points={curvePoints}
              stroke="#6366f1"
              strokeWidth={3}
              lineCap="round"
              lineJoin="round"
            />

            {/* History trail */}
            {history.map((x, i) => {
              const y = loss(x)
              const opacity = 0.2 + (0.6 * i) / history.length
              return (
                <Circle
                  key={i}
                  x={toPixelX(x)}
                  y={toPixelY(y)}
                  radius={4}
                  fill="#22c55e"
                  opacity={opacity}
                />
              )
            })}

            {/* Gradient arrow */}
            {showGradientArrow && Math.abs(currentGradient) > 0.1 && (
              <Group>
                {/* Tangent line */}
                <Line
                  points={[
                    toPixelX(position - 1),
                    toPixelY(currentLoss - currentGradient),
                    toPixelX(position + 1),
                    toPixelY(currentLoss + currentGradient),
                  ]}
                  stroke="#ef4444"
                  strokeWidth={1}
                  dash={[6, 4]}
                  opacity={0.6}
                />

                {/* Gradient direction arrow */}
                <Arrow
                  points={[
                    ballX,
                    ballY - 25,
                    ballX + Math.sign(currentGradient) * 40,
                    ballY - 25,
                  ]}
                  stroke="#ef4444"
                  strokeWidth={2}
                  fill="#ef4444"
                  pointerLength={8}
                  pointerWidth={6}
                />
                <Text
                  x={ballX + Math.sign(currentGradient) * 20 - 25}
                  y={ballY - 45}
                  text="gradient"
                  fontSize={11}
                  fill="#ef4444"
                />

                {/* Update direction arrow */}
                <Arrow
                  points={[
                    ballX,
                    ballY + 25,
                    ballX - Math.sign(currentGradient) * 40,
                    ballY + 25,
                  ]}
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="#22c55e"
                  pointerLength={8}
                  pointerWidth={6}
                />
                <Text
                  x={ballX - Math.sign(currentGradient) * 20 - 20}
                  y={ballY + 30}
                  text="update"
                  fontSize={11}
                  fill="#22c55e"
                />
              </Group>
            )}

            {/* Minimum marker */}
            <Circle x={toPixelX(1)} y={toPixelY(0.5)} radius={6} fill="#22c55e" />
            <Text
              x={toPixelX(1) - 25}
              y={toPixelY(0.5) + 12}
              text="minimum"
              fontSize={11}
              fill="#22c55e"
            />

            {/* Current position (the ball) */}
            <Circle
              x={ballX}
              y={ballY}
              radius={12}
              fill="#f97316"
              shadowColor="black"
              shadowBlur={8}
              shadowOpacity={0.4}
            />
        </ZoomableCanvas>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
            {isRunning ? 'Pause' : 'Run'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => { setIsRunning(false); step() }}
            disabled={isRunning}
          >
            Step
          </Button>
          <Button variant="outline" size="sm" onClick={reset}>
            <RotateCcw className="w-4 h-4 mr-1" />
            Reset
          </Button>
        </div>

        {showLearningRateSlider && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Learning rate (α):</span>
            <input
              type="range"
              min="0.01"
              max="1.0"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-24"
            />
            <span className="font-mono text-sm w-10">{learningRate.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-4 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Step: </span>
          <span className="font-mono">{stepCount}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">θ = </span>
          <span className="font-mono">{position.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Loss = </span>
          <span className="font-mono">{currentLoss.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Gradient = </span>
          <span className="font-mono">{currentGradient.toFixed(3)}</span>
        </div>
      </div>

      {/* Update rule */}
      <div className="p-3 rounded-md bg-muted/50 font-mono text-sm text-center">
        θ<sub>new</sub> = {position.toFixed(2)} - {learningRate.toFixed(2)} × ({currentGradient.toFixed(2)})
        = {(position - learningRate * currentGradient).toFixed(3)}
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Watch the ball roll down the loss curve. The <span className="text-red-400">red arrow</span> shows
        the gradient (slope) direction. The <span className="text-green-400">green arrow</span> shows
        the update direction (opposite to gradient). Click &quot;Step&quot; to move one step at a time.
      </p>
    </div>
  )
}
