'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Mafs, Coordinates, Plot, Point, Vector, Text, Theme } from 'mafs'
import 'mafs/core.css'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw } from 'lucide-react'

/**
 * GradientDescentExplorer - Animated gradient descent on 1D loss curve
 *
 * Shows a ball rolling downhill on a loss function, demonstrating:
 * - Gradient as the slope at current position
 * - Update step moving opposite to gradient
 * - Convergence to minimum
 *
 * Used in:
 * - Lesson 1.1.4: Gradient Descent
 * - Lesson 1.1.5: Learning Rate (with adjustable LR)
 */

interface GradientDescentExplorerProps {
  /** Allow adjusting learning rate */
  showLearningRateSlider?: boolean
  /** Initial learning rate */
  initialLearningRate?: number
  /** Initial parameter value (x position) */
  initialPosition?: number
  /** Show the gradient arrow */
  showGradientArrow?: boolean
  /** Custom loss function (default: quadratic bowl) */
  lossFunction?: (x: number) => number
  /** Derivative of loss function */
  lossFunctionDerivative?: (x: number) => number
}

// Default: simple quadratic loss L(x) = (x - 1)^2 + 0.5
const defaultLoss = (x: number) => Math.pow(x - 1, 2) + 0.5
const defaultLossDerivative = (x: number) => 2 * (x - 1)

export function GradientDescentExplorer({
  showLearningRateSlider = true,
  initialLearningRate = 0.3,
  initialPosition = -2,
  showGradientArrow = true,
  lossFunction = defaultLoss,
  lossFunctionDerivative = defaultLossDerivative,
}: GradientDescentExplorerProps) {
  const [position, setPosition] = useState(initialPosition)
  const [learningRate, setLearningRate] = useState(initialLearningRate)
  const [isRunning, setIsRunning] = useState(false)
  const [history, setHistory] = useState<number[]>([initialPosition])
  const [stepCount, setStepCount] = useState(0)

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Calculate current gradient and loss
  const currentLoss = lossFunction(position)
  const currentGradient = lossFunctionDerivative(position)

  // Single step of gradient descent
  const step = useCallback(() => {
    setPosition((pos) => {
      const grad = lossFunctionDerivative(pos)
      const newPos = pos - learningRate * grad

      // Clamp to viewport
      const clampedPos = Math.max(-4, Math.min(4, newPos))

      setHistory((h) => [...h.slice(-50), clampedPos]) // Keep last 50 positions
      setStepCount((c) => c + 1)

      return clampedPos
    })
  }, [learningRate, lossFunctionDerivative])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      return
    }

    const animate = (time: number) => {
      // Step every 500ms
      if (time - lastStepTime.current > 500) {
        step()
        lastStepTime.current = time
      }

      // Stop if converged (gradient very small)
      if (Math.abs(lossFunctionDerivative(position)) < 0.01) {
        setIsRunning(false)
        return
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, step, position, lossFunctionDerivative])

  // Reset
  const reset = useCallback(() => {
    setIsRunning(false)
    setPosition(initialPosition)
    setHistory([initialPosition])
    setStepCount(0)
  }, [initialPosition])

  // Manual step
  const manualStep = useCallback(() => {
    setIsRunning(false)
    step()
  }, [step])

  return (
    <div className="space-y-4">
      <div className="rounded-lg border bg-card overflow-hidden">
        <Mafs
          height={350}
          viewBox={{ x: [-4.5, 4.5], y: [-0.5, 6] }}
          preserveAspectRatio={false}
        >
          <Coordinates.Cartesian
            xAxis={{ labels: (x) => (x % 2 === 0 ? x.toString() : '') }}
            yAxis={{ labels: (y) => (y % 2 === 0 ? y.toString() : '') }}
          />

          {/* Loss curve */}
          <Plot.OfX y={lossFunction} color={Theme.blue} />

          {/* History trail */}
          {history.map((x, i) => {
            if (i === history.length - 1) return null
            const y = lossFunction(x)
            const opacity = 0.2 + (0.6 * i) / history.length
            return (
              <Point
                key={i}
                x={x}
                y={y}
                color={Theme.green}
                opacity={opacity}
              />
            )
          })}

          {/* Current position (the "ball") */}
          <Point x={position} y={currentLoss} color={Theme.orange} />

          {/* Gradient arrow (tangent direction) */}
          {showGradientArrow && Math.abs(currentGradient) > 0.1 && (
            <>
              {/* Tangent line segment */}
              <Plot.OfX
                y={(x) => currentLoss + currentGradient * (x - position)}
                color={Theme.red}
                opacity={0.5}
                style="dashed"
              />

              {/* Arrow showing gradient direction */}
              <Vector
                tip={[position + Math.sign(currentGradient) * 0.8, currentLoss + 0.5]}
                tail={[position, currentLoss + 0.5]}
                color={Theme.red}
              />
              <Text
                x={position + Math.sign(currentGradient) * 0.5}
                y={currentLoss + 0.9}
                size={14}
                color={Theme.red}
              >
                gradient
              </Text>

              {/* Arrow showing update direction (opposite to gradient) */}
              <Vector
                tip={[position - Math.sign(currentGradient) * 0.8, currentLoss - 0.3]}
                tail={[position, currentLoss - 0.3]}
                color={Theme.green}
              />
              <Text
                x={position - Math.sign(currentGradient) * 0.5}
                y={currentLoss - 0.7}
                size={14}
                color={Theme.green}
              >
                update
              </Text>
            </>
          )}

          {/* Minimum marker */}
          <Point x={1} y={0.5} color={Theme.green} />
          <Text x={1} y={0.1} size={12} color={Theme.green}>
            minimum
          </Text>

          {/* Labels */}
          <Text x={-4} y={5.5} size={14}>
            Loss L(θ)
          </Text>
          <Text x={4} y={0.2} size={14}>
            θ
          </Text>
        </Mafs>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
            {isRunning ? 'Pause' : 'Run'}
          </Button>
          <Button variant="outline" size="sm" onClick={manualStep} disabled={isRunning}>
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
              min="0.05"
              max="1"
              step="0.05"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-24"
            />
            <span className="font-mono text-sm w-10">{learningRate.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-4 text-sm">
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

      {/* Update rule display */}
      <div className="p-3 rounded-md bg-muted/50 font-mono text-sm">
        θ_new = {position.toFixed(2)} - {learningRate.toFixed(2)} × ({currentGradient.toFixed(2)})
        = {(position - learningRate * currentGradient).toFixed(3)}
      </div>

      <p className="text-xs text-muted-foreground">
        Watch the ball roll down the loss curve. The <span className="text-red-400">red arrow</span> shows
        the gradient (slope) direction. The <span className="text-green-400">green arrow</span> shows
        the update direction (opposite to gradient). Click &quot;Step&quot; to move one step at a time.
      </p>
    </div>
  )
}
