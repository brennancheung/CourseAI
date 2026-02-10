'use client'

import { useState, useMemo } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

/**
 * RegularizationExplorer - Interactive overfitting diagnosis and regularization
 *
 * Two visualizations:
 * 1. Training curves: train loss (blue) vs validation loss (red) over epochs
 *    Shows the "scissors" pattern when overfitting occurs
 * 2. Function fit: data points + learned curve, echoing OverfittingWidget
 *    Shows how regularization smooths the learned function
 *
 * Controls: model capacity (3 presets), dropout, weight decay, early stopping, epochs
 *
 * Uses parameterized/synthetic data (not real backprop) for fast, deterministic rendering.
 */

type RegularizationExplorerProps = {
  width?: number
  height?: number
}

type CapacityPreset = 'low' | 'medium' | 'high'

// --- Parameterized model for training dynamics ---

// True underlying function (same spirit as OverfittingWidget)
function trueFunction(x: number): number {
  return 3.0 * (x - 0.5) ** 2 + 0.1
}

// Generate training data points with noise
function generateDataPoints(seed: number, count: number): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = []
  for (let i = 0; i < count; i++) {
    const x = (i + 0.5) / count
    // Deterministic pseudo-noise using seed
    const noise = 0.08 * Math.sin(seed * 7 + i * 13.7) + 0.04 * Math.cos(seed * 3 + i * 5.3)
    points.push({ x, y: trueFunction(x) + noise })
  }
  return points
}

// Training data (blue dots the model trains on)
const TRAIN_POINTS = generateDataPoints(42, 8)
// Validation data (red dots the model doesn't see during training)
const VAL_POINTS = generateDataPoints(17, 6)

// Capacity parameters: controls how many "wiggles" the model can learn
function capacityParams(preset: CapacityPreset): { numTerms: number; label: string; paramCount: string } {
  const params: Record<CapacityPreset, { numTerms: number; label: string; paramCount: string }> = {
    low: { numTerms: 2, label: 'Low', paramCount: '~10' },
    medium: { numTerms: 4, label: 'Medium', paramCount: '~100' },
    high: { numTerms: 8, label: 'High', paramCount: '~1000' },
  }
  return params[preset]
}

// Compute the "learned function" at a given epoch with regularization settings.
// This is a parameterized model that captures the key pedagogical behaviors:
// - High capacity + no reg: wiggly curve through training points (overfitting)
// - Low capacity: smooth but poor fit (underfitting)
// - Medium capacity or regularized: smooth curve close to true function
function learnedFunction(
  x: number,
  epoch: number,
  maxEpochs: number,
  capacity: CapacityPreset,
  dropoutRate: number,
  dropoutEnabled: boolean,
  weightDecayLambda: number,
  weightDecayEnabled: boolean,
): number {
  const { numTerms } = capacityParams(capacity)
  const progress = Math.min(epoch / maxEpochs, 1.0)

  // Base: start with the true function
  let result = trueFunction(x)

  // The "overfitting" component: high-frequency wiggles that memorize noise
  // These grow with training progress and model capacity
  const effectiveCapacity = numTerms

  // Regularization reduces effective overfitting
  let regularizationFactor = 1.0
  if (dropoutEnabled) {
    // Dropout reduces overfitting proportionally to rate
    regularizationFactor *= (1 - dropoutRate * 0.8)
  }
  if (weightDecayEnabled) {
    // Weight decay reduces overfitting (log scale effect)
    const decayEffect = Math.min(1.0, weightDecayLambda * 100)
    regularizationFactor *= (1 - decayEffect * 0.7)
  }

  // Overfitting wiggles: only appear with enough capacity and training
  // High capacity gets a dramatic amplitude so the wiggly-vs-smooth distinction is visually obvious
  const baseAmplitude = effectiveCapacity >= 8 ? 0.4 : 0.15
  const overfitAmplitude = effectiveCapacity / 8 * progress * progress * regularizationFactor * baseAmplitude

  // Add wiggly overfitting terms
  for (let k = 1; k <= effectiveCapacity; k++) {
    const freq = k * 3.5
    const phase = k * 1.7
    // Each term contributes progressively more wiggle at higher frequencies
    const termAmplitude = overfitAmplitude * (1 / k) * (k > 3 ? 1.5 : 1.0)
    result += termAmplitude * Math.sin(freq * x * Math.PI + phase)
  }

  // Low capacity: the model can't even fit the true function well
  if (capacity === 'low') {
    // Blend toward a linear approximation (underfitting)
    const linearApprox = 0.35 - 0.02 * x
    const fitProgress = Math.min(progress * 2, 1.0) // learns slowly
    result = linearApprox + fitProgress * (result - linearApprox) * 0.6
  }

  return result
}

// Compute training loss at a given epoch
function computeTrainLoss(
  epoch: number,
  maxEpochs: number,
  capacity: CapacityPreset,
  dropoutRate: number,
  dropoutEnabled: boolean,
  weightDecayLambda: number,
  weightDecayEnabled: boolean,
): number {
  const { numTerms } = capacityParams(capacity)
  const progress = Math.min(epoch / maxEpochs, 1.0)

  // Base convergence rate depends on capacity
  const convergenceSpeed = numTerms * 0.15

  // Starting loss
  const startLoss = 2.0

  // Final training loss: high capacity models memorize (near zero)
  // Low capacity models plateau high (can't fit the data)
  let finalTrainLoss = 0.5 / numTerms

  // Weight decay raises the training loss floor
  if (weightDecayEnabled) {
    finalTrainLoss += weightDecayLambda * 3
  }

  // Dropout adds noise to training loss and raises it slightly
  let noise = 0
  if (dropoutEnabled) {
    finalTrainLoss += dropoutRate * 0.15
    // Dropout makes training loss noisier
    noise = dropoutRate * 0.08 * Math.sin(epoch * 2.3 + 0.7)
  }

  // Exponential decay toward final loss
  const decay = Math.exp(-convergenceSpeed * progress * 4)
  const baseLoss = finalTrainLoss + (startLoss - finalTrainLoss) * decay

  // Small training noise
  const baseNoise = 0.02 * Math.sin(epoch * 1.7 + 0.3) * decay
  return Math.max(0.01, baseLoss + baseNoise + noise)
}

// Compute validation loss at a given epoch
function computeValLoss(
  epoch: number,
  maxEpochs: number,
  capacity: CapacityPreset,
  dropoutRate: number,
  dropoutEnabled: boolean,
  weightDecayLambda: number,
  weightDecayEnabled: boolean,
): number {
  const { numTerms } = capacityParams(capacity)
  const progress = Math.min(epoch / maxEpochs, 1.0)

  // Validation loss starts same as training loss
  const startLoss = 2.0

  // Best achievable validation loss (the "sweet spot")
  let bestValLoss = 0.12

  // Low capacity: high floor (underfitting)
  if (capacity === 'low') {
    bestValLoss = 0.6
  }

  // Overfitting divergence: validation loss rises after the sweet spot
  // The rise is proportional to capacity and inversely proportional to regularization
  let overfitRise = 0
  if (numTerms > 3) {
    // High capacity models overfit
    let regularizationDamping = 1.0
    if (dropoutEnabled) {
      regularizationDamping *= (1 - dropoutRate * 0.7)
    }
    if (weightDecayEnabled) {
      const decayEffect = Math.min(1.0, weightDecayLambda * 80)
      regularizationDamping *= (1 - decayEffect * 0.6)
    }

    // The overfitting kicks in after the sweet spot
    const sweetSpotEpoch = maxEpochs * 0.25
    if (epoch > sweetSpotEpoch) {
      const overTraining = (epoch - sweetSpotEpoch) / maxEpochs
      overfitRise = overTraining * overTraining * numTerms * 0.12 * regularizationDamping
    }
  }

  // Medium capacity with no regularization: mild overfitting
  if (capacity === 'medium') {
    let regularizationDamping = 1.0
    if (dropoutEnabled) {
      regularizationDamping *= (1 - dropoutRate * 0.7)
    }
    if (weightDecayEnabled) {
      const decayEffect = Math.min(1.0, weightDecayLambda * 80)
      regularizationDamping *= (1 - decayEffect * 0.6)
    }

    const sweetSpotEpoch = maxEpochs * 0.35
    if (epoch > sweetSpotEpoch) {
      const overTraining = (epoch - sweetSpotEpoch) / maxEpochs
      overfitRise = overTraining * overTraining * 0.3 * regularizationDamping
    }
  }

  // Weight decay slightly raises best val loss but prevents divergence
  if (weightDecayEnabled) {
    bestValLoss += weightDecayLambda * 0.5
  }

  // Dropout slightly raises best val loss but prevents divergence
  if (dropoutEnabled && dropoutRate > 0.6) {
    bestValLoss += (dropoutRate - 0.6) * 0.2
  }

  // Convergence toward best val loss, then rise
  const convergenceSpeed = (numTerms > 3 ? numTerms * 0.12 : numTerms * 0.08)
  const decay = Math.exp(-convergenceSpeed * progress * 4)
  const baseLoss = bestValLoss + (startLoss - bestValLoss) * decay + overfitRise

  // Validation noise
  const noise = 0.025 * Math.sin(epoch * 1.3 + 2.1) * (1 - progress * 0.5)
  return Math.max(0.01, baseLoss + noise)
}

// Find the epoch with minimum validation loss (for early stopping)
function findBestValEpoch(
  maxEpochs: number,
  capacity: CapacityPreset,
  dropoutRate: number,
  dropoutEnabled: boolean,
  weightDecayLambda: number,
  weightDecayEnabled: boolean,
  patience: number,
): number {
  let bestEpoch = 0
  let bestLoss = Infinity
  let waitCount = 0

  for (let epoch = 0; epoch <= maxEpochs; epoch++) {
    const valLoss = computeValLoss(
      epoch, maxEpochs, capacity,
      dropoutRate, dropoutEnabled,
      weightDecayLambda, weightDecayEnabled,
    )
    if (valLoss < bestLoss - 0.005) {
      bestLoss = valLoss
      bestEpoch = epoch
      waitCount = 0
      continue
    }
    waitCount++
    if (waitCount >= patience) {
      return bestEpoch
    }
  }
  return bestEpoch
}

// Capacity preset options
const CAPACITY_OPTIONS: { id: CapacityPreset; label: string; detail: string }[] = [
  { id: 'low', label: 'Low', detail: '~10 params' },
  { id: 'medium', label: 'Medium', detail: '~100 params' },
  { id: 'high', label: 'High', detail: '~1000 params' },
]

export function RegularizationExplorer({
  width: widthOverride,
}: RegularizationExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  // Controls state
  const [capacity, setCapacity] = useState<CapacityPreset>('high')
  const [dropoutEnabled, setDropoutEnabled] = useState(false)
  const [dropoutRate, setDropoutRate] = useState(0.5)
  const [weightDecayEnabled, setWeightDecayEnabled] = useState(false)
  const [weightDecayLambda, setWeightDecayLambda] = useState(0.01)
  const [earlyStoppingEnabled, setEarlyStoppingEnabled] = useState(false)
  const [patience, setPatience] = useState(10)
  const [epochs, setEpochs] = useState(150)

  // Compute all training/validation curves
  const { trainCurve, valCurve, earlyStopEpoch, bestValEpoch } = useMemo(() => {
    const train: number[] = []
    const val: number[] = []

    for (let e = 0; e <= epochs; e++) {
      train.push(computeTrainLoss(
        e, epochs, capacity,
        dropoutRate, dropoutEnabled,
        weightDecayLambda, weightDecayEnabled,
      ))
      val.push(computeValLoss(
        e, epochs, capacity,
        dropoutRate, dropoutEnabled,
        weightDecayLambda, weightDecayEnabled,
      ))
    }

    const bestValE = findBestValEpoch(
      epochs, capacity,
      dropoutRate, dropoutEnabled,
      weightDecayLambda, weightDecayEnabled,
      patience,
    )

    // Early stopping: epoch where we would stop
    let stopEpoch = epochs
    if (earlyStoppingEnabled) {
      let bestLoss = Infinity
      let wait = 0
      for (let e = 0; e <= epochs; e++) {
        if (val[e] < bestLoss - 0.005) {
          bestLoss = val[e]
          wait = 0
          continue
        }
        wait++
        if (wait >= patience) {
          stopEpoch = e
          break
        }
      }
    }

    return { trainCurve: train, valCurve: val, earlyStopEpoch: stopEpoch, bestValEpoch: bestValE }
  }, [capacity, dropoutEnabled, dropoutRate, weightDecayEnabled, weightDecayLambda, earlyStoppingEnabled, patience, epochs])

  // The epoch to use for the function fit visualization
  const displayEpoch = earlyStoppingEnabled ? Math.min(earlyStopEpoch, epochs) : epochs

  // Compute function fit curve at the display epoch
  const fitCurvePoints = useMemo(() => {
    const points: { x: number; y: number }[] = []
    const numPoints = 80
    for (let i = 0; i <= numPoints; i++) {
      const x = i / numPoints
      const y = learnedFunction(
        x, displayEpoch, epochs, capacity,
        dropoutRate, dropoutEnabled,
        weightDecayLambda, weightDecayEnabled,
      )
      points.push({ x, y })
    }
    return points
  }, [displayEpoch, epochs, capacity, dropoutRate, dropoutEnabled, weightDecayLambda, weightDecayEnabled])

  // --- Chart dimensions ---
  const chartPadding = { left: 45, right: 15, top: 15, bottom: 28 }
  const lossChartHeight = 180
  const fitChartHeight = 200

  const chartAreaWidth = width - chartPadding.left - chartPadding.right
  const lossAreaHeight = lossChartHeight - chartPadding.top - chartPadding.bottom
  const fitAreaWidth = width - chartPadding.left - chartPadding.right
  const fitAreaHeight = fitChartHeight - chartPadding.top - chartPadding.bottom

  // Loss curve scaling
  const allLossValues = [...trainCurve, ...valCurve]
  const lossMin = 0
  const lossMax = Math.max(...allLossValues) * 1.05
  const lossRange = lossMax - lossMin || 1

  // Function fit scaling
  const fitYMin = -0.1
  const fitYMax = 1.0

  // Helpers for coordinate transforms
  const lossToY = (loss: number) =>
    lossChartHeight - chartPadding.bottom - ((loss - lossMin) / lossRange) * lossAreaHeight
  const epochToX = (epoch: number) =>
    chartPadding.left + (epoch / epochs) * chartAreaWidth
  const fitDataToX = (x: number) => chartPadding.left + x * fitAreaWidth
  const fitDataToY = (y: number) =>
    fitChartHeight - chartPadding.bottom - ((y - fitYMin) / (fitYMax - fitYMin)) * fitAreaHeight

  // Build polyline strings
  const trainLinePoints = trainCurve
    .map((loss, i) => `${epochToX(i)},${lossToY(loss)}`)
    .join(' ')
  const valLinePoints = valCurve
    .map((loss, i) => `${epochToX(i)},${lossToY(loss)}`)
    .join(' ')

  const fitLinePoints = fitCurvePoints
    .map((p) => `${fitDataToX(p.x)},${fitDataToY(Math.max(fitYMin, Math.min(fitYMax, p.y)))}`)
    .join(' ')

  // True function line
  const trueFunctionPoints = Array.from({ length: 81 }, (_, i) => {
    const x = i / 80
    return `${fitDataToX(x)},${fitDataToY(trueFunction(x))}`
  }).join(' ')

  // Determine overfitting status
  const finalTrainLoss = trainCurve[displayEpoch] ?? trainCurve[trainCurve.length - 1]
  const finalValLoss = valCurve[displayEpoch] ?? valCurve[valCurve.length - 1]
  const gap = finalValLoss - finalTrainLoss

  function statusInfo(): { text: string; color: string } {
    if (capacity === 'low') return { text: 'Underfitting', color: '#f97316' }
    if (gap > 0.3) return { text: 'Severe overfitting', color: '#ef4444' }
    if (gap > 0.15) return { text: 'Overfitting', color: '#f97316' }
    if (gap > 0.08) return { text: 'Mild overfitting', color: '#eab308' }
    return { text: 'Good generalization', color: '#22c55e' }
  }

  const status = statusInfo()

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Controls: Capacity */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-sm text-muted-foreground">Model capacity:</span>
        {CAPACITY_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => setCapacity(opt.id)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              capacity === opt.id
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            {opt.label} <span className="opacity-60">({opt.detail})</span>
          </button>
        ))}
      </div>

      {/* Controls: Regularization toggles */}
      <div className="flex flex-wrap items-center justify-center gap-4">
        {/* Dropout */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDropoutEnabled(!dropoutEnabled)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              dropoutEnabled
                ? 'bg-sky-500 text-white'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            Dropout {dropoutEnabled ? 'ON' : 'OFF'}
          </button>
          {dropoutEnabled && (
            <div className="flex items-center gap-1">
              <input
                type="range"
                min="0.1"
                max="0.8"
                step="0.1"
                value={dropoutRate}
                onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
                className="w-16 cursor-pointer"
              />
              <span className="text-xs font-mono w-7">p={dropoutRate.toFixed(1)}</span>
            </div>
          )}
        </div>

        {/* Weight decay */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setWeightDecayEnabled(!weightDecayEnabled)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              weightDecayEnabled
                ? 'bg-violet-500 text-white'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            Weight Decay {weightDecayEnabled ? 'ON' : 'OFF'}
          </button>
          {weightDecayEnabled && (
            <div className="flex items-center gap-1">
              <input
                type="range"
                min="-4"
                max="-1"
                step="0.5"
                value={Math.log10(weightDecayLambda)}
                onChange={(e) => setWeightDecayLambda(Math.pow(10, parseFloat(e.target.value)))}
                className="w-16 cursor-pointer"
              />
              <span className="text-xs font-mono w-14">{'λ'}={weightDecayLambda.toFixed(4)}</span>
            </div>
          )}
        </div>

        {/* Early stopping */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setEarlyStoppingEnabled(!earlyStoppingEnabled)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              earlyStoppingEnabled
                ? 'bg-emerald-500 text-white'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            Early Stopping {earlyStoppingEnabled ? 'ON' : 'OFF'}
          </button>
          {earlyStoppingEnabled && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted-foreground">patience:</span>
              <input
                type="range"
                min="3"
                max="30"
                step="1"
                value={patience}
                onChange={(e) => setPatience(parseInt(e.target.value))}
                className="w-16 cursor-pointer"
              />
              <span className="text-xs font-mono w-5">{patience}</span>
            </div>
          )}
        </div>
      </div>

      {/* Epochs slider */}
      <div className="flex items-center justify-center gap-3">
        <span className="text-sm text-muted-foreground">Epochs:</span>
        <input
          type="range"
          min="10"
          max="200"
          step="10"
          value={epochs}
          onChange={(e) => setEpochs(parseInt(e.target.value))}
          className="w-40 cursor-pointer"
        />
        <span className="text-sm font-mono w-8 text-center">{epochs}</span>
      </div>

      {/* Viz 1: Training curves */}
      <div className="rounded-lg border bg-card p-3">
        <div className="text-xs text-muted-foreground mb-1 font-medium">
          Training Curves (Train vs Validation Loss)
        </div>
        <svg
          viewBox={`0 0 ${width} ${lossChartHeight}`}
          className="w-full"
          style={{ height: lossChartHeight }}
        >
          <rect x="0" y="0" width={width} height={lossChartHeight} fill="#1a1a2e" rx="4" />

          {/* Y-axis gridlines and labels */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const val = lossMin + frac * lossRange
            const y = lossToY(val)
            return (
              <g key={frac}>
                <text
                  x={chartPadding.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  fill="#888"
                  fontSize="8"
                >
                  {val.toFixed(1)}
                </text>
                <line
                  x1={chartPadding.left}
                  y1={y}
                  x2={chartPadding.left + chartAreaWidth}
                  y2={y}
                  stroke="#444"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
              </g>
            )
          })}

          {/* Train loss curve (blue) */}
          <polyline
            points={trainLinePoints}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="1.5"
          />

          {/* Validation loss curve (red) */}
          <polyline
            points={valLinePoints}
            fill="none"
            stroke="#ef4444"
            strokeWidth="1.5"
          />

          {/* Early stopping vertical line */}
          {earlyStoppingEnabled && earlyStopEpoch < epochs && (
            <>
              <line
                x1={epochToX(earlyStopEpoch)}
                y1={chartPadding.top}
                x2={epochToX(earlyStopEpoch)}
                y2={lossChartHeight - chartPadding.bottom}
                stroke="#22c55e"
                strokeWidth="1.5"
                strokeDasharray="4 3"
                opacity="0.7"
              />
              <text
                x={epochToX(earlyStopEpoch)}
                y={chartPadding.top + 10}
                textAnchor="middle"
                fill="#22c55e"
                fontSize="8"
                fontWeight="500"
              >
                stopped
              </text>
            </>
          )}

          {/* Best validation epoch marker — always visible for diagnostic skill */}
          <circle
            cx={epochToX(bestValEpoch)}
            cy={lossToY(valCurve[bestValEpoch] ?? 0)}
            r={4}
            fill={earlyStoppingEnabled ? '#22c55e' : 'none'}
            stroke="#22c55e"
            strokeWidth={earlyStoppingEnabled ? 1 : 2}
          />
          <text
            x={epochToX(bestValEpoch)}
            y={lossToY(valCurve[bestValEpoch] ?? 0) - 7}
            textAnchor="middle"
            fill="#22c55e"
            fontSize="7"
            opacity="0.8"
          >
            best val
          </text>

          {/* X-axis label */}
          <text
            x={width / 2}
            y={lossChartHeight - 5}
            textAnchor="middle"
            fill="#888"
            fontSize="9"
          >
            Epoch
          </text>

          {/* Legend */}
          <line x1={chartPadding.left + 5} y1={chartPadding.top + 5} x2={chartPadding.left + 20} y2={chartPadding.top + 5} stroke="#3b82f6" strokeWidth="2" />
          <text x={chartPadding.left + 24} y={chartPadding.top + 8} fill="#3b82f6" fontSize="8">Train</text>
          <line x1={chartPadding.left + 55} y1={chartPadding.top + 5} x2={chartPadding.left + 70} y2={chartPadding.top + 5} stroke="#ef4444" strokeWidth="2" />
          <text x={chartPadding.left + 74} y={chartPadding.top + 8} fill="#ef4444" fontSize="8">Validation</text>
        </svg>
      </div>

      {/* Viz 2: Function fit */}
      <div className="rounded-lg border bg-card p-3">
        <div className="text-xs text-muted-foreground mb-1 font-medium">
          Learned Function vs Data (at epoch {displayEpoch})
        </div>
        <svg
          viewBox={`0 0 ${width} ${fitChartHeight}`}
          className="w-full"
          style={{ height: fitChartHeight }}
        >
          <rect x="0" y="0" width={width} height={fitChartHeight} fill="#1a1a2e" rx="4" />

          {/* Axes */}
          <line
            x1={chartPadding.left}
            y1={fitChartHeight - chartPadding.bottom}
            x2={chartPadding.left + fitAreaWidth}
            y2={fitChartHeight - chartPadding.bottom}
            stroke="#4a4a6a"
            strokeWidth="1"
          />
          <line
            x1={chartPadding.left}
            y1={chartPadding.top}
            x2={chartPadding.left}
            y2={fitChartHeight - chartPadding.bottom}
            stroke="#4a4a6a"
            strokeWidth="1"
          />

          {/* True function (dashed purple) */}
          <polyline
            points={trueFunctionPoints}
            fill="none"
            stroke="#8b5cf6"
            strokeWidth="1.5"
            strokeDasharray="5 3"
            opacity="0.5"
          />

          {/* Learned function curve */}
          <polyline
            points={fitLinePoints}
            fill="none"
            stroke="#f97316"
            strokeWidth="2.5"
          />

          {/* Training data points (blue) */}
          {TRAIN_POINTS.map((point, i) => (
            <circle
              key={`train-${i}`}
              cx={fitDataToX(point.x)}
              cy={fitDataToY(point.y)}
              r={5}
              fill="#3b82f6"
              stroke="#93c5fd"
              strokeWidth="1.5"
            />
          ))}

          {/* Validation data points (red) */}
          {VAL_POINTS.map((point, i) => (
            <circle
              key={`val-${i}`}
              cx={fitDataToX(point.x)}
              cy={fitDataToY(point.y)}
              r={4}
              fill="none"
              stroke="#ef4444"
              strokeWidth="2"
            />
          ))}

          {/* Legend */}
          <circle cx={chartPadding.left + 10} cy={chartPadding.top + 8} r={4} fill="#3b82f6" stroke="#93c5fd" strokeWidth="1" />
          <text x={chartPadding.left + 18} y={chartPadding.top + 11} fill="#3b82f6" fontSize="8">Train</text>
          <circle cx={chartPadding.left + 52} cy={chartPadding.top + 8} r={3.5} fill="none" stroke="#ef4444" strokeWidth="1.5" />
          <text x={chartPadding.left + 60} y={chartPadding.top + 11} fill="#ef4444" fontSize="8">Val</text>
          <line x1={chartPadding.left + 85} y1={chartPadding.top + 8} x2={chartPadding.left + 100} y2={chartPadding.top + 8} stroke="#8b5cf6" strokeWidth="1.5" strokeDasharray="4 2" opacity="0.6" />
          <text x={chartPadding.left + 104} y={chartPadding.top + 11} fill="#8b5cf6" fontSize="8" opacity="0.6">True</text>
          <line x1={chartPadding.left + 130} y1={chartPadding.top + 8} x2={chartPadding.left + 145} y2={chartPadding.top + 8} stroke="#f97316" strokeWidth="2.5" />
          <text x={chartPadding.left + 149} y={chartPadding.top + 11} fill="#f97316" fontSize="8">Model</text>

          {/* Axis labels */}
          <text
            x={width / 2}
            y={fitChartHeight - 5}
            textAnchor="middle"
            fill="#888"
            fontSize="9"
          >
            Input (x)
          </text>
        </svg>
      </div>

      {/* Status badges */}
      <div className="flex flex-wrap gap-3 text-sm justify-center">
        <div
          className="px-3 py-2 rounded-md"
          style={{ backgroundColor: `${status.color}15`, border: `1px solid ${status.color}30` }}
        >
          <span style={{ color: status.color }} className="font-medium text-xs">
            {status.text}
          </span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">Train loss: </span>
          <span className="font-mono text-xs">{finalTrainLoss.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">Val loss: </span>
          <span className="font-mono text-xs">{finalValLoss.toFixed(3)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">Gap: </span>
          <span className="font-mono text-xs">{gap.toFixed(3)}</span>
        </div>
        {earlyStoppingEnabled && earlyStopEpoch < epochs && (
          <div className="px-3 py-2 rounded-md bg-emerald-500/10 border border-emerald-500/20">
            <span className="text-emerald-400 text-xs">Stopped at epoch {earlyStopEpoch}</span>
          </div>
        )}
      </div>

      <p className="text-xs text-muted-foreground text-center">
        <span className="text-blue-400">Blue line</span> = training loss,{' '}
        <span className="text-red-400">red line</span> = validation loss.
        When the gap between them grows, the model is overfitting. Toggle regularization techniques to see how they close the gap.
      </p>
    </div>
  )
}
