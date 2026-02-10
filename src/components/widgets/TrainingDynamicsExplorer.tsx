'use client'

import { useState, useMemo } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

/**
 * TrainingDynamicsExplorer - Visualize gradient flow and training dynamics
 *
 * Interactive widget showing how network depth, activation function,
 * initialization strategy, and batch normalization affect gradient
 * magnitudes and training behavior.
 *
 * Two visualizations:
 * 1. Gradient magnitude bar chart (log scale) — one bar per layer
 * 2. Simulated training loss curve showing how settings affect convergence
 *
 * This uses a parameterized model (not real backpropagation) that captures
 * the key behaviors analytically for fast, deterministic rendering.
 */

type TrainingDynamicsExplorerProps = {
  width?: number
  height?: number
}

type ActivationFn = 'sigmoid' | 'relu' | 'tanh'
type InitStrategy = 'naive-uniform' | 'naive-normal' | 'xavier' | 'he'

// --- Gradient magnitude model ---
// For a network with L layers, the gradient at layer k is approximately:
//   |grad_k| = product of local derivatives from layer k to L
// We model this as: |grad_k| = (localDerivative)^(L - k) * jitter

function activationLabel(act: ActivationFn): string {
  const labels: Record<ActivationFn, string> = {
    sigmoid: 'Sigmoid',
    relu: 'ReLU',
    tanh: 'Tanh',
  }
  return labels[act]
}

function initLabel(init: InitStrategy): string {
  const labels: Record<InitStrategy, string> = {
    'naive-uniform': 'Naive Uniform',
    'naive-normal': 'Naive Normal',
    xavier: 'Xavier',
    he: 'He',
  }
  return labels[init]
}

// Effective local derivative factor per layer based on activation + init
function localDerivativeFactor(activation: ActivationFn, init: InitStrategy, numInputs: number): number {
  // Sigmoid: max derivative is 0.25. With proper init (Xavier), the expected
  // derivative factor is ~0.25. With naive init, it's worse (~0.15-0.2).
  if (activation === 'sigmoid') {
    if (init === 'xavier') return 0.92
    if (init === 'he') return 0.85
    if (init === 'naive-normal') return 0.4
    // naive-uniform: all positive weights cause activation saturation
    return 0.25
  }

  // ReLU: derivative is 0 or 1. With He init, ~50% neurons active -> factor ~1.0.
  // With Xavier, slight signal decay. With naive, unstable.
  if (activation === 'relu') {
    if (init === 'he') return 1.0
    if (init === 'xavier') return 0.85
    if (init === 'naive-normal') return 1.3
    // naive-uniform: large positive weights -> exploding
    return 1.6 + 0.005 * numInputs
  }

  // Tanh: max derivative is 1.0 at origin. Xavier keeps it near 1.0.
  if (init === 'xavier') return 0.98
  if (init === 'he') return 0.95
  if (init === 'naive-normal') return 0.55
  return 0.35
}

// Compute gradient magnitudes at each layer
function computeGradientProfile(
  numLayers: number,
  activation: ActivationFn,
  init: InitStrategy,
  batchNorm: boolean,
): number[] {
  const numInputs = 64 // assume 64-wide hidden layers
  const baseFactor = localDerivativeFactor(activation, init, numInputs)

  const gradients: number[] = []

  for (let layer = 0; layer < numLayers; layer++) {
    const distanceFromOutput = numLayers - 1 - layer
    let gradMagnitude = Math.pow(baseFactor, distanceFromOutput)

    // ReLU dead neurons: some layers randomly die (gradient = 0)
    // With naive init, more dying; with He, minimal dying
    if (activation === 'relu') {
      const seed = (layer * 7 + numLayers * 13) % 17
      if (init === 'naive-uniform' || init === 'naive-normal') {
        // ~20% chance of a dead layer with naive init
        if (seed < 3) {
          gradMagnitude = 0.001
        }
      }
    }

    // Batch norm stabilizes gradients toward 1.0
    if (batchNorm && numLayers > 2) {
      // BN pulls gradients toward a healthy range
      // The further from 1.0, the stronger the pull
      const logGrad = Math.log10(Math.max(gradMagnitude, 1e-10))
      // Dampen the log by 70% toward 0 (i.e., magnitude toward 1.0)
      const dampened = logGrad * 0.3
      gradMagnitude = Math.pow(10, dampened)

      // Add small per-layer variation for realism
      const noise = 1 + 0.1 * Math.sin(layer * 2.7 + numLayers * 0.3)
      gradMagnitude *= noise
    }

    gradients.push(gradMagnitude)
  }

  return gradients
}

// Classify gradient health and return training dynamics parameters
type TrainingDynamics = {
  convergenceRate: number
  finalLoss: number
  diverges: boolean
}

function classifyGradientHealth(
  minGrad: number,
  maxGrad: number,
  avgGrad: number,
  numLayers: number,
  batchNorm: boolean,
): TrainingDynamics {
  // Exploding gradients: if max gradient > 100, training will diverge
  if (maxGrad > 100) {
    return { convergenceRate: 0, finalLoss: 10, diverges: true }
  }
  // Vanishing: very slow convergence, high final loss
  if (minGrad < 0.01 && !batchNorm) {
    return {
      convergenceRate: 0.005 * Math.sqrt(minGrad / 0.001),
      finalLoss: 2.0 + 0.5 * (1 - minGrad),
      diverges: false,
    }
  }
  // Healthy range
  if (avgGrad > 0.3 && avgGrad < 3.0) {
    return {
      convergenceRate: 0.06,
      finalLoss: 0.1 + 0.02 * numLayers / 10,
      diverges: false,
    }
  }
  // Suboptimal but trainable
  return {
    convergenceRate: 0.02 * Math.min(avgGrad, 1 / Math.max(avgGrad, 0.01)),
    finalLoss: 0.5 + 0.3 * Math.abs(Math.log10(avgGrad)),
    diverges: false,
  }
}

// Compute loss at a single epoch for a diverging training run
function divergingLossAtEpoch(epoch: number, startLoss: number): number {
  // Initially decreasing normally (epochs 0-14)
  if (epoch < 15) {
    const t = epoch / 15
    return startLoss * (1 - 0.3 * t) + 0.1 * Math.sin(epoch * 0.5)
  }
  // Spike up (epochs 15-24)
  if (epoch < 25) {
    const t = (epoch - 15) / 10
    return startLoss * (0.7 + 3 * t * t) + 0.3 * Math.sin(epoch * 0.7)
  }
  // Flatline at high loss (represents NaN/divergence)
  return 10 + 0.5 * Math.sin(epoch * 0.3)
}

// Simulate training loss curve
// Returns an array of loss values over "epochs"
function computeTrainingCurve(
  numLayers: number,
  activation: ActivationFn,
  init: InitStrategy,
  batchNorm: boolean,
): number[] {
  const numEpochs = 80
  const gradients = computeGradientProfile(numLayers, activation, init, batchNorm)

  const minGrad = Math.min(...gradients)
  const maxGrad = Math.max(...gradients)
  const avgGrad = gradients.reduce((a, b) => a + b, 0) / gradients.length

  const { convergenceRate, finalLoss, diverges } = classifyGradientHealth(
    minGrad, maxGrad, avgGrad, numLayers, batchNorm,
  )

  const startLoss = 2.5
  const curve: number[] = []

  for (let epoch = 0; epoch < numEpochs; epoch++) {
    if (diverges) {
      curve.push(divergingLossAtEpoch(epoch, startLoss))
      continue
    }
    // Exponential decay toward finalLoss
    const decay = Math.exp(-convergenceRate * epoch * 3)
    const noise = 0.03 * Math.sin(epoch * 1.3 + numLayers) * decay
    curve.push(finalLoss + (startLoss - finalLoss) * decay + noise)
  }

  return curve
}

// Color for gradient bar: green = healthy (near 1.0), red = vanishing or exploding
function gradientBarColor(magnitude: number): string {
  if (magnitude < 0.01) return '#ef4444' // vanishing — red
  if (magnitude < 0.1) return '#f97316' // warning — orange
  if (magnitude < 0.5) return '#eab308' // caution — yellow
  if (magnitude <= 2.0) return '#22c55e' // healthy — green
  if (magnitude <= 10) return '#eab308' // caution — yellow
  if (magnitude <= 100) return '#f97316' // warning — orange
  return '#ef4444' // exploding — red
}

const ACTIVATION_OPTIONS: { id: ActivationFn; label: string }[] = [
  { id: 'sigmoid', label: 'Sigmoid' },
  { id: 'relu', label: 'ReLU' },
  { id: 'tanh', label: 'Tanh' },
]

const INIT_OPTIONS: { id: InitStrategy; label: string }[] = [
  { id: 'naive-uniform', label: 'Naive Uniform' },
  { id: 'naive-normal', label: 'Naive Normal' },
  { id: 'xavier', label: 'Xavier' },
  { id: 'he', label: 'He' },
]

export function TrainingDynamicsExplorer({
  width: widthOverride,
}: TrainingDynamicsExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [numLayers, setNumLayers] = useState(10)
  const [activation, setActivation] = useState<ActivationFn>('sigmoid')
  const [init, setInit] = useState<InitStrategy>('naive-uniform')
  const [batchNorm, setBatchNorm] = useState(false)

  // Compute gradient profile and training curve
  const gradients = useMemo(
    () => computeGradientProfile(numLayers, activation, init, batchNorm),
    [numLayers, activation, init, batchNorm]
  )

  const trainingCurve = useMemo(
    () => computeTrainingCurve(numLayers, activation, init, batchNorm),
    [numLayers, activation, init, batchNorm]
  )

  // Gradient bar chart dimensions
  const chartPadding = { left: 50, right: 20, top: 20, bottom: 30 }
  const barChartWidth = width
  const barChartHeight = 200
  const barAreaWidth = barChartWidth - chartPadding.left - chartPadding.right
  const barAreaHeight = barChartHeight - chartPadding.top - chartPadding.bottom

  // Log scale: map gradient magnitude to bar height
  // Range: 1e-8 to 1e4
  const logMin = -8
  const logMax = 4
  const logRange = logMax - logMin

  // "Healthy" band: gradient magnitude between 0.5 and 2.0
  const healthyLowY =
    barChartHeight -
    chartPadding.bottom -
    ((Math.log10(0.5) - logMin) / logRange) * barAreaHeight
  const healthyHighY =
    barChartHeight -
    chartPadding.bottom -
    ((Math.log10(2.0) - logMin) / logRange) * barAreaHeight

  // Bar chart bars
  const barWidth = Math.min(30, (barAreaWidth - numLayers * 2) / numLayers)
  const barSpacing = (barAreaWidth - barWidth * numLayers) / (numLayers + 1)

  // Training curve dimensions
  const curveHeight = 120
  const curvePadding = { left: 50, right: 20, top: 10, bottom: 25 }
  const curveAreaWidth = width - curvePadding.left - curvePadding.right
  const curveAreaHeight = curveHeight - curvePadding.top - curvePadding.bottom

  const lossMin = Math.min(...trainingCurve)
  const lossMax = Math.max(...trainingCurve)
  const lossRange = lossMax - lossMin || 1

  // Summary stats
  const minGrad = Math.min(...gradients)
  const maxGrad = Math.max(...gradients)
  const isExploding = maxGrad > 100
  const isVanishing = minGrad < 0.01 && !isExploding

  function statusLabel(): { text: string; color: string } {
    if (isExploding) return { text: 'Exploding gradients', color: '#ef4444' }
    if (isVanishing) return { text: 'Vanishing gradients', color: '#f97316' }
    if (minGrad < 0.1) return { text: 'Partially vanishing', color: '#eab308' }
    if (maxGrad > 10) return { text: 'Partially exploding', color: '#eab308' }
    return { text: 'Healthy gradient flow', color: '#22c55e' }
  }

  const status = statusLabel()

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Controls row 1: Layers slider */}
      <div className="flex items-center justify-center gap-3">
        <span className="text-sm text-muted-foreground">Layers:</span>
        <input
          type="range"
          min="2"
          max="20"
          step="1"
          value={numLayers}
          onChange={(e) => setNumLayers(parseInt(e.target.value))}
          className="w-40 cursor-pointer"
        />
        <span className="text-sm font-mono w-6 text-center">{numLayers}</span>
      </div>

      {/* Controls row 2: Activation function */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-sm text-muted-foreground">Activation:</span>
        {ACTIVATION_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => setActivation(opt.id)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              activation === opt.id
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Controls row 3: Initialization */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-sm text-muted-foreground">Init:</span>
        {INIT_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => setInit(opt.id)}
            className={cn(
              'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
              init === opt.id
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Controls row 4: Batch norm toggle */}
      <div className="flex items-center justify-center gap-3">
        <span className="text-sm text-muted-foreground">Batch Normalization:</span>
        <button
          onClick={() => setBatchNorm(!batchNorm)}
          className={cn(
            'px-4 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
            batchNorm
              ? 'bg-emerald-500 text-white'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground'
          )}
        >
          {batchNorm ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Gradient magnitude bar chart */}
      <div className="rounded-lg border bg-card p-3">
        <div className="text-xs text-muted-foreground mb-1 font-medium">
          Gradient Magnitude by Layer (log scale)
        </div>
        <svg
          viewBox={`0 0 ${barChartWidth} ${barChartHeight}`}
          className="w-full"
          style={{ height: barChartHeight }}
        >
          {/* Background */}
          <rect
            x="0"
            y="0"
            width={barChartWidth}
            height={barChartHeight}
            fill="#1a1a2e"
            rx="4"
          />

          {/* Healthy band */}
          <rect
            x={chartPadding.left}
            y={healthyHighY}
            width={barAreaWidth}
            height={healthyLowY - healthyHighY}
            fill="#22c55e"
            opacity="0.08"
          />
          <line
            x1={chartPadding.left}
            y1={healthyHighY}
            x2={chartPadding.left + barAreaWidth}
            y2={healthyHighY}
            stroke="#22c55e"
            strokeWidth="0.5"
            opacity="0.3"
            strokeDasharray="4 4"
          />
          <line
            x1={chartPadding.left}
            y1={healthyLowY}
            x2={chartPadding.left + barAreaWidth}
            y2={healthyLowY}
            stroke="#22c55e"
            strokeWidth="0.5"
            opacity="0.3"
            strokeDasharray="4 4"
          />

          {/* Y-axis labels (log scale) */}
          {[-6, -4, -2, 0, 2, 4].map((logVal) => {
            const y =
              barChartHeight -
              chartPadding.bottom -
              ((logVal - logMin) / logRange) * barAreaHeight
            return (
              <g key={logVal}>
                <text
                  x={chartPadding.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  fill="#888"
                  fontSize="9"
                >
                  10{logVal < 0 ? '⁻' : ''}{Math.abs(logVal).toString().split('').map(d => String.fromCharCode(0x2070 + parseInt(d))).join('')}
                </text>
                <line
                  x1={chartPadding.left}
                  y1={y}
                  x2={chartPadding.left + barAreaWidth}
                  y2={y}
                  stroke="#444"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
              </g>
            )
          })}

          {/* Reference line at 1.0 (log10(1) = 0) */}
          {(() => {
            const y1line =
              barChartHeight -
              chartPadding.bottom -
              ((0 - logMin) / logRange) * barAreaHeight
            return (
              <line
                x1={chartPadding.left}
                y1={y1line}
                x2={chartPadding.left + barAreaWidth}
                y2={y1line}
                stroke="#ffffff"
                strokeWidth="0.5"
                opacity="0.4"
              />
            )
          })()}

          {/* Gradient bars */}
          {gradients.map((grad, i) => {
            const logGrad = Math.log10(Math.max(grad, 1e-8))
            const clampedLog = Math.max(logMin, Math.min(logMax, logGrad))
            const barHeight =
              ((clampedLog - logMin) / logRange) * barAreaHeight
            const x = chartPadding.left + barSpacing + i * (barWidth + barSpacing)
            const y = barChartHeight - chartPadding.bottom - barHeight

            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  fill={gradientBarColor(grad)}
                  opacity="0.85"
                  rx="1"
                />
                {/* Layer number label */}
                {(numLayers <= 12 || i % 2 === 0) && (
                  <text
                    x={x + barWidth / 2}
                    y={barChartHeight - chartPadding.bottom + 12}
                    textAnchor="middle"
                    fill="#888"
                    fontSize="8"
                  >
                    {i + 1}
                  </text>
                )}
              </g>
            )
          })}

          {/* X-axis label */}
          <text
            x={barChartWidth / 2}
            y={barChartHeight - 3}
            textAnchor="middle"
            fill="#888"
            fontSize="9"
          >
            Layer
          </text>

          {/* "Healthy" label */}
          <text
            x={chartPadding.left + barAreaWidth + 2}
            y={(healthyHighY + healthyLowY) / 2 + 3}
            fill="#22c55e"
            fontSize="7"
            opacity="0.6"
          >
            healthy
          </text>
        </svg>
      </div>

      {/* Training loss curve */}
      <div className="rounded-lg border bg-card p-3">
        <div className="text-xs text-muted-foreground mb-1 font-medium">
          Simulated Training Loss
        </div>
        <svg
          viewBox={`0 0 ${width} ${curveHeight}`}
          className="w-full"
          style={{ height: curveHeight }}
        >
          <rect x="0" y="0" width={width} height={curveHeight} fill="#1a1a2e" rx="4" />

          {/* Y-axis ticks */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const val = lossMin + frac * lossRange
            const y = curveHeight - curvePadding.bottom - frac * curveAreaHeight
            return (
              <g key={frac}>
                <text
                  x={curvePadding.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  fill="#888"
                  fontSize="8"
                >
                  {val.toFixed(1)}
                </text>
                <line
                  x1={curvePadding.left}
                  y1={y}
                  x2={curvePadding.left + curveAreaWidth}
                  y2={y}
                  stroke="#444"
                  strokeWidth="0.5"
                  opacity="0.2"
                />
              </g>
            )
          })}

          {/* Loss curve */}
          <polyline
            points={trainingCurve
              .map((loss, i) => {
                const px =
                  curvePadding.left +
                  (i / (trainingCurve.length - 1)) * curveAreaWidth
                const py =
                  curveHeight -
                  curvePadding.bottom -
                  ((loss - lossMin) / lossRange) * curveAreaHeight
                return `${px},${py}`
              })
              .join(' ')}
            fill="none"
            stroke={isExploding ? '#ef4444' : '#6366f1'}
            strokeWidth="1.5"
          />

          {/* X-axis label */}
          <text
            x={width / 2}
            y={curveHeight - 3}
            textAnchor="middle"
            fill="#888"
            fontSize="9"
          >
            Epoch
          </text>

          {/* Divergence label */}
          {isExploding && (
            <text
              x={curvePadding.left + curveAreaWidth * 0.65}
              y={curvePadding.top + 15}
              textAnchor="middle"
              fill="#ef4444"
              fontSize="11"
              fontWeight="600"
            >
              NaN / Diverged
            </text>
          )}
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
          <span className="text-muted-foreground text-xs">Layers: </span>
          <span className="font-mono text-xs">{numLayers}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">{activationLabel(activation)} + {initLabel(init)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">Min |grad|: </span>
          <span className="font-mono text-xs">{minGrad < 0.0001 ? minGrad.toExponential(1) : minGrad.toFixed(4)}</span>
        </div>
        <div className="px-3 py-2 rounded-md bg-muted">
          <span className="text-muted-foreground text-xs">Max |grad|: </span>
          <span className="font-mono text-xs">{maxGrad > 1000 ? maxGrad.toExponential(1) : maxGrad.toFixed(4)}</span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        The{' '}
        <span className="text-emerald-400">green band</span>{' '}
        shows the healthy gradient range (0.5-2.0).{' '}
        <span className="text-red-400">Red bars</span>{' '}
        indicate vanishing or exploding gradients. Try different activation + initialization combinations to see what keeps gradients stable.
      </p>
    </div>
  )
}
