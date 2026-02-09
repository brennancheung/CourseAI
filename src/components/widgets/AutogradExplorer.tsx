'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Play, RotateCcw, StepForward } from 'lucide-react'

/**
 * AutogradExplorer - Interactive widget comparing manual backprop vs autograd
 *
 * Shows a proper computational graph with nodes and edges:
 * - Forward values in blue, gradients in red
 * - Parameter nodes in violet, operation nodes in slate
 *
 * Two modes:
 * - Manual mode: Student clicks through backward steps one at a time,
 *   seeing "incoming x local = outgoing" at each node with real numbers
 * - Autograd mode: Student clicks one "backward()" button and all gradients
 *   appear simultaneously
 *
 * Uses the same 2-layer network from backprop-worked-example:
 * x -> Linear(w1, b1) -> ReLU -> Linear(w2, b2) -> MSE Loss
 */

type AutogradExplorerProps = {
  height?: number
}

type NetworkValues = {
  w1: number
  b1: number
  w2: number
  b2: number
  x: number
  yTrue: number
  // Forward pass
  z1: number
  a1: number
  yHat: number
  loss: number
  // Backward pass
  dLdyHat: number
  dLdw2: number
  dLdb2: number
  dLda1: number
  dLdz1: number
  dLdw1: number
  dLdb1: number
  // Local derivatives (for showing "incoming x local = outgoing")
  reluGrad: number
}

function computeNetwork(w1: number, b1: number, w2: number, b2: number, x: number, yTrue: number): NetworkValues {
  const z1 = w1 * x + b1
  const a1 = Math.max(0, z1)
  const yHat = w2 * a1 + b2
  const loss = (yTrue - yHat) ** 2

  const dLdyHat = -2 * (yTrue - yHat)
  const dLdw2 = dLdyHat * a1
  const dLdb2 = dLdyHat
  const dLda1 = dLdyHat * w2
  const reluGrad = z1 > 0 ? 1 : 0
  const dLdz1 = dLda1 * reluGrad
  const dLdw1 = dLdz1 * x
  const dLdb1 = dLdz1

  return { w1, b1, w2, b2, x, yTrue, z1, a1, yHat, loss, dLdyHat, dLdw2, dLdb2, dLda1, dLdz1, dLdw1, dLdb1, reluGrad }
}

function fmt(n: number): string {
  return n.toFixed(4)
}

function fmtShort(n: number): string {
  return n.toFixed(2)
}

type Mode = 'manual' | 'autograd'

// Manual mode has 5 backward steps:
// 0 = nothing shown (forward only)
// 1 = dL/dyHat (loss gradient)
// 2 = dL/dw2, dL/db2 (layer 2 param grads)
// 3 = dL/da1 (gradient flowing to layer 1)
// 4 = dL/dz1 (through ReLU)
// 5 = dL/dw1, dL/db1 (layer 1 param grads) -- all done
const MAX_MANUAL_STEP = 5

function getStepDescription(step: number, net: NetworkValues): string {
  const descriptions: Record<number, string> = {
    0: 'Forward pass complete. Click "Next Step" to begin the backward pass.',
    1: `Step 1: Loss gradient. dL/d\u0177 = -2(y - \u0177) = -2(${fmtShort(net.yTrue)} - (${fmtShort(net.yHat)})) = ${fmt(net.dLdyHat)}`,
    2: `Step 2: Layer 2 param gradients. dL/dw2 = dL/d\u0177 \u00d7 a1 = ${fmt(net.dLdyHat)} \u00d7 ${fmt(net.a1)} = ${fmt(net.dLdw2)}. dL/db2 = dL/d\u0177 \u00d7 1 = ${fmt(net.dLdb2)}`,
    3: `Step 3: Pass gradient to Layer 1. dL/da1 = dL/d\u0177 \u00d7 w2 = ${fmt(net.dLdyHat)} \u00d7 (${fmt(net.w2)}) = ${fmt(net.dLda1)}`,
    4: `Step 4: Through ReLU. dL/dz1 = dL/da1 \u00d7 relu'(z1) = ${fmt(net.dLda1)} \u00d7 ${net.reluGrad} = ${fmt(net.dLdz1)}${net.reluGrad === 0 ? ' (ReLU killed the gradient!)' : ''}`,
    5: `Step 5: Layer 1 param gradients. dL/dw1 = dL/dz1 \u00d7 x = ${fmt(net.dLdz1)} \u00d7 ${fmtShort(net.x)} = ${fmt(net.dLdw1)}. dL/db1 = dL/dz1 \u00d7 1 = ${fmt(net.dLdb1)}. Done!`,
  }
  return descriptions[step] ?? ''
}

// --- Graph Node Components ---

function GraphNode({
  label,
  forwardValue,
  gradientValue,
  gradientLabel,
  showGradient,
  type,
  highlighted,
}: {
  label: string
  forwardValue?: string
  gradientValue?: string
  gradientLabel?: string
  showGradient: boolean
  type: 'input' | 'param' | 'operation' | 'loss'
  highlighted: boolean
}) {
  const borderColor = (() => {
    if (highlighted) return 'border-yellow-400/60'
    if (type === 'param') return 'border-violet-500/40'
    if (type === 'loss') return 'border-rose-500/40'
    if (type === 'input') return 'border-blue-500/40'
    return 'border-slate-500/40'
  })()

  const bgColor = (() => {
    if (highlighted) return 'bg-yellow-500/10'
    if (type === 'param') return 'bg-violet-500/10'
    if (type === 'loss') return 'bg-rose-500/10'
    if (type === 'input') return 'bg-blue-500/10'
    return 'bg-slate-500/10'
  })()

  return (
    <div className={cn(
      'rounded-lg border-2 px-2 py-1.5 text-center min-w-[60px] transition-all duration-300',
      borderColor,
      bgColor,
    )}>
      <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider leading-tight">{label}</div>
      {forwardValue && (
        <div className="font-mono text-xs font-semibold text-blue-400 mt-0.5">{forwardValue}</div>
      )}
      {showGradient && gradientValue && (
        <div className="mt-1 border-t border-border/50 pt-1">
          {gradientLabel && (
            <div className="text-[9px] text-muted-foreground leading-tight">{gradientLabel}</div>
          )}
          <div className="font-mono text-xs font-semibold text-rose-400">{gradientValue}</div>
        </div>
      )}
    </div>
  )
}

function Arrow({ direction = 'right', className }: { direction?: 'right' | 'down'; className?: string }) {
  if (direction === 'down') {
    return (
      <div className={cn('flex justify-center', className)}>
        <svg width="12" height="16" viewBox="0 0 12 16" className="text-slate-500">
          <path d="M6 0 L6 12 M2 8 L6 14 L10 8" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </div>
    )
  }
  return (
    <div className={cn('flex items-center', className)}>
      <svg width="20" height="12" viewBox="0 0 20 12" className="text-slate-500 shrink-0">
        <path d="M0 6 L16 6 M12 2 L18 6 L12 10" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}

export function AutogradExplorer(_props: AutogradExplorerProps) {

  const [mode, setMode] = useState<Mode>('manual')
  const [manualStep, setManualStep] = useState(0)
  const [autogradDone, setAutogradDone] = useState(false)

  // Adjustable parameters
  const [w1, setW1] = useState(0.5)
  const [b1, setB1] = useState(0.1)
  const [w2, setW2] = useState(-0.3)
  const [b2, setB2] = useState(0.2)

  const net = computeNetwork(w1, b1, w2, b2, 2.0, 1.0)

  const handleManualNext = useCallback(() => {
    if (manualStep < MAX_MANUAL_STEP) {
      setManualStep(prev => prev + 1)
    }
  }, [manualStep])

  const handleAutogradBackward = useCallback(() => {
    setAutogradDone(true)
  }, [])

  const handleReset = useCallback(() => {
    setManualStep(0)
    setAutogradDone(false)
    setW1(0.5)
    setB1(0.1)
    setW2(-0.3)
    setB2(0.2)
  }, [])

  const handleModeSwitch = useCallback((newMode: Mode) => {
    setMode(newMode)
    setManualStep(0)
    setAutogradDone(false)
  }, [])

  // Determine which gradients are visible based on mode and step
  const allVisible = mode === 'autograd' && autogradDone
  const gradsVisible = {
    dLdyHat: allVisible || manualStep >= 1,
    dLdw2: allVisible || manualStep >= 2,
    dLdb2: allVisible || manualStep >= 2,
    dLda1: allVisible || manualStep >= 3,
    dLdz1: allVisible || manualStep >= 4,
    dLdw1: allVisible || manualStep >= 5,
    dLdb1: allVisible || manualStep >= 5,
  }

  // Which step is currently highlighted (for manual mode)
  const highlightedStep = mode === 'manual' ? manualStep : 0

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => handleModeSwitch('manual')}
          className={cn(
            'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
            mode === 'manual'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground',
          )}
        >
          Manual (step-by-step)
        </button>
        <button
          onClick={() => handleModeSwitch('autograd')}
          className={cn(
            'px-3 py-1.5 rounded-full text-xs font-medium transition-colors cursor-pointer',
            mode === 'autograd'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted hover:bg-muted/80 text-muted-foreground',
          )}
        >
          Autograd (one click)
        </button>
      </div>

      {/* Computational Graph */}
      <div className="rounded-lg border bg-muted/30 p-4 space-y-2 overflow-x-auto">
        <div className="flex items-center gap-1 mb-2">
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Computational Graph</span>
          <span className="text-[10px] text-muted-foreground ml-2">
            <span className="text-blue-400">blue</span> = forward values
            {(gradsVisible.dLdyHat) && (
              <>{' '}<span className="text-rose-400">red</span> = gradients</>
            )}
          </span>
        </div>

        {/* Main forward flow: x -> (*) -> (+) -> ReLU -> (*) -> (+) -> MSE */}
        <div className="flex items-start gap-1 min-w-[680px]">
          {/* x input */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label="x"
              forwardValue={fmtShort(net.x)}
              type="input"
              showGradient={false}
              highlighted={false}
            />
          </div>

          <Arrow />

          {/* Multiply: w1 * x */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label="w1*x"
              forwardValue={fmtShort(net.w1 * net.x)}
              type="operation"
              showGradient={false}
              highlighted={highlightedStep === 5}
            />
            <Arrow direction="down" />
            <GraphNode
              label="w1"
              forwardValue={fmtShort(net.w1)}
              gradientLabel="dL/dw1"
              gradientValue={fmt(net.dLdw1)}
              type="param"
              showGradient={gradsVisible.dLdw1}
              highlighted={highlightedStep === 5}
            />
          </div>

          <Arrow />

          {/* Add: + b1 */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label="z1 = +b1"
              forwardValue={fmtShort(net.z1)}
              gradientLabel="dL/dz1"
              gradientValue={fmt(net.dLdz1)}
              type="operation"
              showGradient={gradsVisible.dLdz1}
              highlighted={highlightedStep === 4 || highlightedStep === 5}
            />
            <Arrow direction="down" />
            <GraphNode
              label="b1"
              forwardValue={fmtShort(net.b1)}
              gradientLabel="dL/db1"
              gradientValue={fmt(net.dLdb1)}
              type="param"
              showGradient={gradsVisible.dLdb1}
              highlighted={highlightedStep === 5}
            />
          </div>

          <Arrow />

          {/* ReLU */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label={`ReLU${net.reluGrad === 0 ? ' (off)' : ''}`}
              forwardValue={fmtShort(net.a1)}
              gradientLabel="dL/da1"
              gradientValue={fmt(net.dLda1)}
              type="operation"
              showGradient={gradsVisible.dLda1}
              highlighted={highlightedStep === 3 || highlightedStep === 4}
            />
          </div>

          <Arrow />

          {/* Multiply: w2 * a1 */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label="w2*a1"
              forwardValue={fmtShort(net.w2 * net.a1)}
              type="operation"
              showGradient={false}
              highlighted={highlightedStep === 2}
            />
            <Arrow direction="down" />
            <GraphNode
              label="w2"
              forwardValue={fmtShort(net.w2)}
              gradientLabel="dL/dw2"
              gradientValue={fmt(net.dLdw2)}
              type="param"
              showGradient={gradsVisible.dLdw2}
              highlighted={highlightedStep === 2}
            />
          </div>

          <Arrow />

          {/* Add: + b2 */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label={'\u0177 = +b2'}
              forwardValue={fmtShort(net.yHat)}
              gradientLabel="dL/d\u0177"
              gradientValue={fmt(net.dLdyHat)}
              type="operation"
              showGradient={gradsVisible.dLdyHat}
              highlighted={highlightedStep === 1 || highlightedStep === 2}
            />
            <Arrow direction="down" />
            <GraphNode
              label="b2"
              forwardValue={fmtShort(net.b2)}
              gradientLabel="dL/db2"
              gradientValue={fmt(net.dLdb2)}
              type="param"
              showGradient={gradsVisible.dLdb2}
              highlighted={highlightedStep === 2}
            />
          </div>

          <Arrow />

          {/* MSE Loss */}
          <div className="flex flex-col items-center gap-1">
            <GraphNode
              label="MSE Loss"
              forwardValue={fmt(net.loss)}
              type="loss"
              showGradient={false}
              highlighted={highlightedStep === 1}
            />
          </div>
        </div>
      </div>

      {/* Step description (manual mode) */}
      {mode === 'manual' && (
        <div className="rounded-md border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground font-mono leading-relaxed">
          {getStepDescription(manualStep, net)}
        </div>
      )}

      {/* Autograd description */}
      {mode === 'autograd' && !autogradDone && (
        <div className="rounded-md border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
          Click <strong>loss.backward()</strong> to compute all gradients in one call. Same algorithm, same numbers&mdash;automated.
        </div>
      )}

      {mode === 'autograd' && autogradDone && (
        <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-sm text-muted-foreground">
          All gradients computed in one call. Compare these values to the manual mode&mdash;they are identical. <strong>backward()</strong> runs the exact same chain rule computation you stepped through manually.
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        {mode === 'manual' && (
          <Button
            variant="default"
            size="sm"
            onClick={handleManualNext}
            disabled={manualStep >= MAX_MANUAL_STEP}
            className="cursor-pointer"
          >
            <StepForward className="w-4 h-4 mr-1" />
            Next Step ({manualStep}/{MAX_MANUAL_STEP})
          </Button>
        )}

        {mode === 'autograd' && (
          <Button
            variant="default"
            size="sm"
            onClick={handleAutogradBackward}
            disabled={autogradDone}
            className="cursor-pointer"
          >
            <Play className="w-4 h-4 mr-1" />
            loss.backward()
          </Button>
        )}

        <Button variant="outline" size="sm" onClick={handleReset} className="cursor-pointer">
          <RotateCcw className="w-4 h-4 mr-1" />
          Reset
        </Button>
      </div>

      {/* Parameter adjustment */}
      <details className="group">
        <summary className="cursor-pointer text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
          Adjust initial weights
        </summary>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">w1: {w1.toFixed(2)}</label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={w1}
              onChange={(e) => {
                setW1(parseFloat(e.target.value))
                setManualStep(0)
                setAutogradDone(false)
              }}
              className="w-full cursor-pointer"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">b1: {b1.toFixed(2)}</label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={b1}
              onChange={(e) => {
                setB1(parseFloat(e.target.value))
                setManualStep(0)
                setAutogradDone(false)
              }}
              className="w-full cursor-pointer"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">w2: {w2.toFixed(2)}</label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={w2}
              onChange={(e) => {
                setW2(parseFloat(e.target.value))
                setManualStep(0)
                setAutogradDone(false)
              }}
              className="w-full cursor-pointer"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">b2: {b2.toFixed(2)}</label>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.1"
              value={b2}
              onChange={(e) => {
                setB2(parseFloat(e.target.value))
                setManualStep(0)
                setAutogradDone(false)
              }}
              className="w-full cursor-pointer"
            />
          </div>
        </div>
      </details>

      {/* Explanation text */}
      <p className="text-xs text-muted-foreground">
        Both modes compute the same gradients using the same algorithm. Manual mode shows each step of the chain rule
        with the actual &ldquo;incoming x local = outgoing&rdquo; computation.
        Autograd mode shows what <code className="bg-muted px-1 rounded">loss.backward()</code> does internally&mdash;all at once.
        Try adjusting the weights to verify the match holds for different values.
      </p>
    </div>
  )
}
