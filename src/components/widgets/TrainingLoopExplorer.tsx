'use client'

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { Line, Circle, Text, Arrow, Rect } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, StepForward } from 'lucide-react'

/**
 * TrainingLoopExplorer - Interactive linear regression training visualization
 *
 * Shows the complete training loop:
 * - Data points and fitted line
 * - Parameter updates (w, b) in real-time
 * - Loss history chart
 * - Step-by-step or animated training
 *
 * Replaces the need for a Colab notebook by visualizing training in-browser.
 */

type TrainingLoopExplorerProps = {
  /** Number of data points */
  numPoints?: number
  /** Initial learning rate */
  initialLearningRate?: number
  /** Width of canvas */
  width?: number
  /** Height of canvas */
  height?: number
}

// Generate synthetic linear data with noise
function generateData(n: number, trueW: number, trueB: number, noise: number) {
  const X: number[] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 10 - 5 // x in [-5, 5]
    const yVal = trueW * x + trueB + (Math.random() - 0.5) * noise * 2
    X.push(x)
    y.push(yVal)
  }
  return { X, y }
}

// MSE loss
function computeLoss(X: number[], y: number[], w: number, b: number) {
  let sum = 0
  for (let i = 0; i < X.length; i++) {
    const pred = w * X[i] + b
    const err = y[i] - pred
    sum += err * err
  }
  return sum / X.length
}

// Gradients
function computeGradients(X: number[], y: number[], w: number, b: number) {
  let dwSum = 0
  let dbSum = 0
  const n = X.length
  for (let i = 0; i < n; i++) {
    const pred = w * X[i] + b
    const err = y[i] - pred
    dwSum += -2 * X[i] * err
    dbSum += -2 * err
  }
  return { dw: dwSum / n, db: dbSum / n }
}

// Data viewport
const DATA_VIEW = {
  xMin: -6,
  xMax: 6,
  yMin: -8,
  yMax: 12,
}

// Loss chart viewport
const LOSS_VIEW = {
  xMax: 100, // epochs
  yMax: 50, // max loss
}

export function TrainingLoopExplorer({
  numPoints = 30,
  initialLearningRate = 0.02,
  width = 600,
  height = 350,
}: TrainingLoopExplorerProps) {
  // True parameters (what we're trying to learn)
  const trueW = 1.5
  const trueB = 2

  // Generate data once
  const data = useMemo(
    () => generateData(numPoints, trueW, trueB, 3),
    [numPoints]
  )

  // Model parameters
  const [w, setW] = useState(0)
  const [b, setB] = useState(0)
  const [learningRate, setLearningRate] = useState(initialLearningRate)
  const [epoch, setEpoch] = useState(0)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Current loss
  const currentLoss = computeLoss(data.X, data.y, w, b)

  // Layout: split canvas into data view (left) and loss chart (right)
  const dataWidth = Math.floor(width * 0.65)
  const chartWidth = width - dataWidth - 20
  const chartHeight = height - 60

  // Coordinate transforms for data view
  const toDataX = (x: number) =>
    ((x - DATA_VIEW.xMin) / (DATA_VIEW.xMax - DATA_VIEW.xMin)) * dataWidth
  const toDataY = (y: number) =>
    height - ((y - DATA_VIEW.yMin) / (DATA_VIEW.yMax - DATA_VIEW.yMin)) * height

  // Coordinate transforms for loss chart
  const chartX = dataWidth + 20
  const chartY = 30
  const toLossX = (e: number) =>
    chartX + (Math.min(e, LOSS_VIEW.xMax) / LOSS_VIEW.xMax) * chartWidth
  const toLossY = (l: number) =>
    chartY + chartHeight - (Math.min(l, LOSS_VIEW.yMax) / LOSS_VIEW.yMax) * chartHeight

  // Single training step
  const doStep = useCallback(() => {
    const { dw, db } = computeGradients(data.X, data.y, w, b)
    const newW = w - learningRate * dw
    const newB = b - learningRate * db
    const newLoss = computeLoss(data.X, data.y, newW, newB)

    setW(newW)
    setB(newB)
    setLossHistory((prev) => [...prev, newLoss])
    setEpoch((e) => e + 1)
  }, [data.X, data.y, w, b, learningRate])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > 50) {
        doStep()
        lastStepTime.current = time
      }

      // Stop after many epochs
      if (epoch >= 200) {
        setIsRunning(false)
        return
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, doStep, epoch])

  // Reset
  const reset = useCallback(() => {
    setIsRunning(false)
    setW(0)
    setB(0)
    setEpoch(0)
    setLossHistory([])
  }, [])

  // Generate line points for current model
  const linePoints = [
    toDataX(DATA_VIEW.xMin),
    toDataY(w * DATA_VIEW.xMin + b),
    toDataX(DATA_VIEW.xMax),
    toDataY(w * DATA_VIEW.xMax + b),
  ]

  // Generate true line points (faded)
  const trueLinePoints = [
    toDataX(DATA_VIEW.xMin),
    toDataY(trueW * DATA_VIEW.xMin + trueB),
    toDataX(DATA_VIEW.xMax),
    toDataY(trueW * DATA_VIEW.xMax + trueB),
  ]

  // Loss curve points
  const lossPoints: number[] = []
  lossHistory.forEach((loss, i) => {
    lossPoints.push(toLossX(i), toLossY(loss))
  })

  // Grid lines for data view
  const dataGrid: { points: number[]; key: string }[] = []
  for (let x = Math.ceil(DATA_VIEW.xMin); x <= DATA_VIEW.xMax; x += 2) {
    dataGrid.push({
      key: `vd${x}`,
      points: [toDataX(x), 0, toDataX(x), height],
    })
  }
  for (let y = Math.ceil(DATA_VIEW.yMin); y <= DATA_VIEW.yMax; y += 4) {
    dataGrid.push({
      key: `hd${y}`,
      points: [0, toDataY(y), dataWidth, toDataY(y)],
    })
  }

  const { dw, db } = computeGradients(data.X, data.y, w, b)

  return (
    <div className="space-y-4">
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Data View Grid */}
          {dataGrid.map((line) => (
            <Line
              key={line.key}
              points={line.points}
              stroke="#333355"
              strokeWidth={1}
            />
          ))}

          {/* Data View Axes */}
          <Arrow
            points={[0, toDataY(0), dataWidth, toDataY(0)]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />
          <Arrow
            points={[toDataX(0), height, toDataX(0), 0]}
            stroke="#666688"
            strokeWidth={2}
            fill="#666688"
            pointerLength={8}
            pointerWidth={6}
          />
          <Text x={dataWidth - 20} y={toDataY(0) + 10} text="x" fontSize={14} fill="#888" />
          <Text x={toDataX(0) + 10} y={10} text="y" fontSize={14} fill="#888" />

          {/* True line (target) */}
          <Line
            points={trueLinePoints}
            stroke="#22c55e"
            strokeWidth={2}
            dash={[8, 4]}
            opacity={0.4}
          />

          {/* Data points */}
          {data.X.map((x, i) => (
            <Circle
              key={i}
              x={toDataX(x)}
              y={toDataY(data.y[i])}
              radius={5}
              fill="#6366f1"
              opacity={0.8}
            />
          ))}

          {/* Current fitted line */}
          <Line
            points={linePoints}
            stroke="#f97316"
            strokeWidth={3}
          />

          {/* Labels */}
          <Text x={10} y={10} text="Data + Fitted Line" fontSize={12} fill="#888" />

          {/* Loss Chart Area */}
          <Rect
            x={chartX}
            y={chartY}
            width={chartWidth}
            height={chartHeight}
            fill="#252540"
            cornerRadius={4}
          />

          {/* Loss chart axes */}
          <Line
            points={[chartX, chartY + chartHeight, chartX + chartWidth, chartY + chartHeight]}
            stroke="#666688"
            strokeWidth={1}
          />
          <Line
            points={[chartX, chartY, chartX, chartY + chartHeight]}
            stroke="#666688"
            strokeWidth={1}
          />
          <Text x={chartX + chartWidth / 2 - 30} y={chartY + chartHeight + 8} text="Epoch" fontSize={11} fill="#888" />
          <Text x={chartX - 5} y={chartY - 15} text="Loss" fontSize={11} fill="#888" />

          {/* Loss curve */}
          {lossPoints.length >= 4 && (
            <Line
              points={lossPoints}
              stroke="#ef4444"
              strokeWidth={2}
              lineCap="round"
              lineJoin="round"
            />
          )}

          {/* Current loss dot */}
          {lossHistory.length > 0 && (
            <Circle
              x={toLossX(lossHistory.length - 1)}
              y={toLossY(lossHistory[lossHistory.length - 1])}
              radius={4}
              fill="#ef4444"
            />
          )}
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
            {isRunning ? 'Pause' : 'Train'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={doStep}
            disabled={isRunning}
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
          <span className="text-sm text-muted-foreground">Learning rate:</span>
          <input
            type="range"
            min="0.001"
            max="0.1"
            step="0.001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-mono text-sm w-12">{learningRate.toFixed(3)}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="flex flex-wrap gap-3 text-sm justify-center">
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground">Epoch: </span>
          <span className="font-mono">{epoch}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-orange-500/10 border border-orange-500/30">
          <span className="text-muted-foreground">w = </span>
          <span className="font-mono text-orange-400">{w.toFixed(3)}</span>
          <span className="text-muted-foreground/60 ml-1">(true: {trueW})</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-orange-500/10 border border-orange-500/30">
          <span className="text-muted-foreground">b = </span>
          <span className="font-mono text-orange-400">{b.toFixed(3)}</span>
          <span className="text-muted-foreground/60 ml-1">(true: {trueB})</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-red-500/10 border border-red-500/30">
          <span className="text-muted-foreground">Loss = </span>
          <span className="font-mono text-red-400">{currentLoss.toFixed(3)}</span>
        </div>
      </div>

      {/* Update formula */}
      <div className="p-3 rounded-md bg-muted/50 font-mono text-xs text-center space-y-1">
        <div>
          w<sub>new</sub> = {w.toFixed(3)} - {learningRate.toFixed(3)} × ({dw.toFixed(3)})
          = <span className="text-orange-400">{(w - learningRate * dw).toFixed(3)}</span>
        </div>
        <div>
          b<sub>new</sub> = {b.toFixed(3)} - {learningRate.toFixed(3)} × ({db.toFixed(3)})
          = <span className="text-orange-400">{(b - learningRate * db).toFixed(3)}</span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Watch the <span className="text-orange-400">orange line</span> fit the{' '}
        <span className="text-indigo-400">data points</span>.
        The <span className="text-green-400/60">dashed green line</span> shows the true relationship.
        Click &quot;Step&quot; to run one gradient descent update, or &quot;Train&quot; to animate.
      </p>
    </div>
  )
}
