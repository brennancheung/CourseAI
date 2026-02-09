'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Mafs, Coordinates, Plot, Point, Text, Theme } from 'mafs'
import 'mafs/core.css'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw } from 'lucide-react'

/**
 * LearningRateExplorer - Shows pathological learning rate behaviors
 *
 * Demonstrates:
 * - Good learning rate: smooth convergence
 * - Too small: very slow convergence
 * - Too large: oscillation and overshooting
 * - Way too large: divergence
 *
 * Used in:
 * - Lesson 1.1.5: Learning Rate Deep Dive
 */

// Simple quadratic loss: L(x) = (x - 0)^2 = x^2
const lossFunction = (x: number) => x * x
const lossDerivative = (x: number) => 2 * x

interface LearningRateExplorerProps {
  /** Preset scenarios */
  mode?: 'interactive' | 'comparison'
}

interface TrajectoryPoint {
  x: number
  loss: number
  step: number
}

function SimulationPanel({
  learningRate,
  label,
  color,
  showSlider = false,
  onLearningRateChange,
}: {
  learningRate: number
  label: string
  color: string
  showSlider?: boolean
  onLearningRateChange?: (lr: number) => void
}) {
  const [position, setPosition] = useState(-3)
  const [isRunning, setIsRunning] = useState(false)
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([
    { x: -3, loss: lossFunction(-3), step: 0 },
  ])
  const [currentLR, setCurrentLR] = useState(learningRate)

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)
  const stepCount = useRef(0)

  // Update LR when prop changes
  useEffect(() => {
    setCurrentLR(learningRate)
  }, [learningRate])

  const step = useCallback(() => {
    setPosition((pos) => {
      const grad = lossDerivative(pos)
      const newPos = pos - currentLR * grad
      stepCount.current++

      setTrajectory((t) => [
        ...t,
        { x: newPos, loss: lossFunction(newPos), step: stepCount.current },
      ])

      return newPos
    })
  }, [currentLR])

  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > 400) {
        step()
        lastStepTime.current = time
      }

      // Stop conditions
      if (
        Math.abs(position) < 0.001 || // Converged
        Math.abs(position) > 100 || // Diverged
        stepCount.current > 50 // Too many steps
      ) {
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
  }, [isRunning, step, position])

  const reset = useCallback(() => {
    setIsRunning(false)
    setPosition(-3)
    setTrajectory([{ x: -3, loss: lossFunction(-3), step: 0 }])
    stepCount.current = 0
  }, [])

  const currentLoss = lossFunction(position)
  const isDiverging = Math.abs(position) > 10

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="font-medium text-sm">{label}</span>
        </div>
        {!showSlider && (
          <span className="font-mono text-xs text-muted-foreground">
            α = {currentLR}
          </span>
        )}
      </div>

      <div className="rounded-lg border bg-card overflow-hidden">
        <Mafs
          height={300}
          viewBox={{ x: [-6, 6], y: [-1, 15] }}
          preserveAspectRatio={false}
        >
          <Coordinates.Cartesian
            xAxis={{ labels: false }}
            yAxis={{ labels: false }}
          />

          {/* Loss curve */}
          <Plot.OfX y={lossFunction} color={Theme.blue} />

          {/* Trajectory */}
          {trajectory.slice(0, -1).map((point, i) => (
            <Point
              key={i}
              x={point.x}
              y={Math.min(point.loss, 14)}
              color={color}
              opacity={0.3 + (0.5 * i) / trajectory.length}
            />
          ))}

          {/* Current position */}
          {!isDiverging && (
            <Point
              x={position}
              y={Math.min(currentLoss, 14)}
              color={color}
            />
          )}

          {/* Minimum marker */}
          <Point x={0} y={0} color={Theme.green} />

          {/* Divergence indicator */}
          {isDiverging && (
            <Text x={0} y={7} size={16} color={Theme.red}>
              DIVERGED!
            </Text>
          )}
        </Mafs>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsRunning(!isRunning)}
          disabled={isDiverging || Math.abs(position) < 0.001}
          className="flex-1"
        >
          {isRunning ? <Pause className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />}
          {isRunning ? 'Pause' : 'Run'}
        </Button>
        <Button variant="outline" size="sm" onClick={reset}>
          <RotateCcw className="w-3 h-3" />
        </Button>
      </div>

      {showSlider && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm text-orange-400 w-16">α = {currentLR.toFixed(2)}</span>
            <input
              type="range"
              min="0.01"
              max="1.2"
              step="0.01"
              value={currentLR}
              onChange={(e) => {
                const newLR = parseFloat(e.target.value)
                setCurrentLR(newLR)
                onLearningRateChange?.(newLR)
                reset()
              }}
              className="flex-1"
            />
          </div>
          <div className="flex gap-2 text-xs">
            <span className="text-muted-foreground">Presets:</span>
            {[
              { value: 0.1, label: '0.1', desc: 'slow' },
              { value: 0.5, label: '0.5', desc: 'good' },
              { value: 0.9, label: '0.9', desc: 'oscillating' },
              { value: 1.05, label: '1.05', desc: 'diverges' },
            ].map((preset) => (
              <button
                key={preset.value}
                onClick={() => {
                  setCurrentLR(preset.value)
                  onLearningRateChange?.(preset.value)
                  reset()
                }}
                className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                  Math.abs(currentLR - preset.value) < 0.01
                    ? 'border-orange-500 bg-orange-500/20 text-orange-400'
                    : 'border-muted hover:border-muted-foreground hover:bg-muted'
                }`}
                title={preset.desc}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Steps: {stepCount.current}</span>
        <span>Loss: {isDiverging ? '∞' : currentLoss.toFixed(4)}</span>
      </div>
    </div>
  )
}

export function LearningRateExplorer({ mode = 'comparison' }: LearningRateExplorerProps) {
  const [customLR, setCustomLR] = useState(0.3)

  if (mode === 'interactive') {
    return (
      <div className="space-y-4">
        <SimulationPanel
          learningRate={customLR}
          label="Interactive"
          color="#f97316"
          showSlider={true}
          onLearningRateChange={setCustomLR}
        />
        <p className="text-xs text-muted-foreground">
          Adjust the learning rate slider and observe how convergence behavior changes.
          Try values around 0.1 (slow), 0.5 (good), 0.9 (oscillating), and 1.1 (diverging).
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <SimulationPanel
          learningRate={0.05}
          label="Too Small"
          color="#fbbf24"
        />
        <SimulationPanel
          learningRate={0.5}
          label="Just Right"
          color="#22c55e"
        />
        <SimulationPanel
          learningRate={0.9}
          label="Too Large"
          color="#f97316"
        />
        <SimulationPanel
          learningRate={1.1}
          label="Way Too Large"
          color="#ef4444"
        />
      </div>
      <p className="text-xs text-muted-foreground text-center">
        Click &quot;Run&quot; on each panel to see how different learning rates affect convergence.
        The optimal learning rate for this problem is around α = 0.5.
      </p>
    </div>
  )
}
