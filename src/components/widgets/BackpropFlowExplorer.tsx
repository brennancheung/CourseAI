'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Circle, Text, Arrow, Rect, Group, Line } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, StepForward } from 'lucide-react'
import { useContainerWidth } from '@/hooks/useContainerWidth'

/**
 * BackpropFlowExplorer - Animated step-through backpropagation visualization
 *
 * Shows a 2-layer network: x -> Linear(w1,b1) -> ReLU -> Linear(w2,b2) -> Loss
 *
 * 11 steps per epoch:
 *   Forward: 0=idle, 1=input, 2=linear1, 3=relu, 4=linear2, 5=loss
 *   Backward: 6=loss grad, 7=layer2 grads, 8=relu grad, 9=layer1 grads
 *   Update: 10=weight update
 *
 * After step 10, weights update and next epoch begins.
 */

type BackpropFlowExplorerProps = {
  width?: number
  height?: number
}

const INITIAL_WEIGHTS = { w1: 0.5, b1: 0.0, w2: 0.5, b2: 0.0 }

type Snapshot = {
  x: number
  target: number
  w1: number
  b1: number
  w2: number
  b2: number
  z1: number
  a1: number
  yPred: number
  loss: number
  dL_dyPred: number
  dL_dw2: number
  dL_db2: number
  dL_da1: number
  reluGrad: number
  dL_dz1: number
  dL_dw1: number
  dL_db1: number
}

function computeSnapshot(
  x: number,
  target: number,
  w1: number,
  b1: number,
  w2: number,
  b2: number,
): Snapshot {
  const z1 = w1 * x + b1
  const a1 = Math.max(0, z1)
  const yPred = w2 * a1 + b2
  const loss = Math.pow(yPred - target, 2)

  const dL_dyPred = 2 * (yPred - target)
  const dL_dw2 = dL_dyPred * a1
  const dL_db2 = dL_dyPred * 1
  const dL_da1 = dL_dyPred * w2
  const reluGrad = z1 > 0 ? 1 : 0
  const dL_dz1 = dL_da1 * reluGrad
  const dL_dw1 = dL_dz1 * x
  const dL_db1 = dL_dz1 * 1

  return {
    x, target, w1, b1, w2, b2,
    z1, a1, yPred, loss,
    dL_dyPred, dL_dw2, dL_db2, dL_da1,
    reluGrad, dL_dz1, dL_dw1, dL_db1,
  }
}

const fmt = (n: number, digits = 2) => {
  if (Math.abs(n) < 0.0001) return '0.00'
  return n.toFixed(digits)
}

const fmt3 = (n: number) => n.toFixed(3)

type Speed = 'slow' | 'normal'

function getStepInterval(speed: Speed): number {
  if (speed === 'slow') return 800
  return 400
}

function getFormulaForStep(step: number, snap: Snapshot): string {
  if (step <= 0) return 'Ready — press Step or Run'
  if (step === 1) return `x = ${fmt(snap.x)}`
  if (step === 2) return `z₁ = w₁·x + b₁ = ${fmt(snap.w1)}×${fmt(snap.x)} + ${fmt(snap.b1)} = ${fmt(snap.z1)}`
  if (step === 3) return `a₁ = ReLU(${fmt(snap.z1)}) = ${fmt(snap.a1)}`
  if (step === 4) return `ŷ = w₂·a₁ + b₂ = ${fmt(snap.w2)}×${fmt(snap.a1)} + ${fmt(snap.b2)} = ${fmt(snap.yPred)}`
  if (step === 5) return `L = (ŷ - target)² = (${fmt(snap.yPred)} - ${fmt(snap.target)})² = ${fmt(snap.loss, 4)}`
  if (step === 6) return `∂L/∂ŷ = 2(ŷ - target) = 2(${fmt(snap.yPred)} - ${fmt(snap.target)}) = ${fmt(snap.dL_dyPred)}`
  if (step === 7) return `∂L/∂w₂ = ∂L/∂ŷ · a₁ = ${fmt(snap.dL_dyPred)} × ${fmt(snap.a1)} = ${fmt(snap.dL_dw2)}   |   ∂L/∂b₂ = ${fmt(snap.dL_db2)}`
  if (step === 8) {
    if (snap.reluGrad === 0) {
      return `∂L/∂z₁ = ∂L/∂a₁ · ReLU'(z₁) = ${fmt(snap.dL_da1)} × 0 = 0.00  ⚠ ReLU blocked!`
    }
    return `∂L/∂z₁ = ∂L/∂a₁ · ReLU'(z₁) = ${fmt(snap.dL_da1)} × 1 = ${fmt(snap.dL_dz1)}`
  }
  if (step === 9) return `∂L/∂w₁ = ∂L/∂z₁ · x = ${fmt(snap.dL_dz1)} × ${fmt(snap.x)} = ${fmt(snap.dL_dw1)}   |   ∂L/∂b₁ = ${fmt(snap.dL_db1)}`
  // step 10 - update
  return [
    `w₁: ${fmt3(snap.w1)} → ${fmt3(snap.w1)} - α×${fmt(snap.dL_dw1)}`,
    `b₁: ${fmt3(snap.b1)} → ${fmt3(snap.b1)} - α×${fmt(snap.dL_db1)}`,
    `w₂: ${fmt3(snap.w2)} → ${fmt3(snap.w2)} - α×${fmt(snap.dL_dw2)}`,
    `b₂: ${fmt3(snap.b2)} → ${fmt3(snap.b2)} - α×${fmt(snap.dL_db2)}`,
  ].join('   ')
}

function getStepLabel(step: number): string {
  if (step <= 0) return 'Idle'
  if (step === 1) return 'Forward: Input'
  if (step === 2) return 'Forward: Linear₁'
  if (step === 3) return 'Forward: ReLU'
  if (step === 4) return 'Forward: Linear₂'
  if (step === 5) return 'Forward: Loss'
  if (step === 6) return 'Backward: Loss Gradient'
  if (step === 7) return 'Backward: Layer 2 Gradients'
  if (step === 8) return 'Backward: Through ReLU'
  if (step === 9) return 'Backward: Layer 1 Gradients'
  return 'Update Weights'
}

// Which nodes are "computed" (filled) at each step
// Nodes: x, z1, a1, yPred, loss
function getComputedNodes(step: number): Set<string> {
  if (step <= 0) return new Set()
  if (step === 1) return new Set(['x'])
  if (step === 2) return new Set(['x', 'z1'])
  if (step === 3) return new Set(['x', 'z1', 'a1'])
  if (step === 4) return new Set(['x', 'z1', 'a1', 'yPred'])
  return new Set(['x', 'z1', 'a1', 'yPred', 'loss'])
}

// Which node is currently highlighted (active)
function getActiveNode(step: number): string | null {
  if (step === 1) return 'x'
  if (step === 2) return 'z1'
  if (step === 3) return 'a1'
  if (step === 4) return 'yPred'
  if (step === 5) return 'loss'
  if (step === 6) return 'loss'
  if (step === 7) return 'yPred'
  if (step === 8) return 'a1'
  if (step === 9) return 'z1'
  return null
}

// Which forward edges are visible at each step
function getForwardEdges(step: number): Set<number> {
  if (step <= 1) return new Set()
  if (step === 2) return new Set([0])
  if (step === 3) return new Set([0, 1])
  if (step === 4) return new Set([0, 1, 2])
  if (step >= 5) return new Set([0, 1, 2, 3])
  return new Set()
}

// Which backward edges are visible at each step
function getBackwardEdges(step: number): Set<number> {
  if (step <= 5) return new Set()
  if (step === 6) return new Set([3])
  if (step === 7) return new Set([3, 2])
  if (step === 8) return new Set([3, 2, 1])
  if (step >= 9) return new Set([3, 2, 1, 0])
  return new Set()
}

// Which nodes have gradient labels below them
function getGradientNodes(step: number): Set<string> {
  if (step <= 5) return new Set()
  if (step === 6) return new Set(['yPred'])
  if (step === 7) return new Set(['yPred', 'a1'])
  if (step === 8) return new Set(['yPred', 'a1', 'z1'])
  if (step >= 9) return new Set(['yPred', 'a1', 'z1', 'x'])
  return new Set()
}

export function BackpropFlowExplorer({
  width: widthOverride,
  height: heightOverride,
}: BackpropFlowExplorerProps) {
  // Weights persist across epochs
  const [weights, setWeights] = useState(INITIAL_WEIGHTS)
  const [inputX, setInputX] = useState(1.0)
  const [target, setTarget] = useState(2.0)
  const [learningRate, setLearningRate] = useState(0.1)
  const [speed, setSpeed] = useState<Speed>('normal')

  // Animation state
  const [currentStep, setCurrentStep] = useState(-1)
  const [isRunning, setIsRunning] = useState(false)
  const [epoch, setEpoch] = useState(0)

  // Snapshot: frozen computation for current epoch
  const [snapshot, setSnapshot] = useState<Snapshot>(() =>
    computeSnapshot(1.0, 2.0, INITIAL_WEIGHTS.w1, INITIAL_WEIGHTS.b1, INITIAL_WEIGHTS.w2, INITIAL_WEIGHTS.b2),
  )

  // Container measurement
  const { containerRef, width: measuredWidth } = useContainerWidth(700)
  const width = widthOverride ?? measuredWidth
  const height = heightOverride ?? 360

  // Animation refs
  const animationRef = useRef<number | null>(null)
  const lastStepTime = useRef<number>(0)

  // Compute new snapshot when entering step 0
  useEffect(() => {
    if (currentStep === 0) {
      setSnapshot(computeSnapshot(inputX, target, weights.w1, weights.b1, weights.w2, weights.b2))
    }
  }, [currentStep, inputX, target, weights])

  // Apply weight updates when leaving step 10
  const applyWeightUpdate = useCallback(() => {
    setWeights((prev) => {
      const snap = computeSnapshot(inputX, target, prev.w1, prev.b1, prev.w2, prev.b2)
      return {
        w1: prev.w1 - learningRate * snap.dL_dw1,
        b1: prev.b1 - learningRate * snap.dL_db1,
        w2: prev.w2 - learningRate * snap.dL_dw2,
        b2: prev.b2 - learningRate * snap.dL_db2,
      }
    })
    setEpoch((e) => e + 1)
  }, [inputX, target, learningRate])

  // Step button handler
  const handleStep = useCallback(() => {
    if (currentStep === 10) {
      // Apply update and move to next epoch's step 0
      applyWeightUpdate()
      setCurrentStep(0)
      return
    }
    if (currentStep === -1) {
      // Starting fresh, go to step 0 then 1
      setSnapshot(computeSnapshot(inputX, target, weights.w1, weights.b1, weights.w2, weights.b2))
      setCurrentStep(1)
      return
    }
    setCurrentStep((prev) => prev + 1)
  }, [currentStep, applyWeightUpdate, inputX, target, weights])

  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      return
    }

    const animate = (time: number) => {
      if (time - lastStepTime.current > getStepInterval(speed)) {
        lastStepTime.current = time

        setCurrentStep((prev) => {
          if (prev === -1) {
            // Starting: compute snapshot and go to step 1
            return 1
          }
          if (prev === 10) {
            // Apply weight update and start next epoch
            applyWeightUpdate()
            return 0
          }
          return prev + 1
        })
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    // Ensure snapshot is computed if starting from idle
    if (currentStep === -1) {
      setSnapshot(computeSnapshot(inputX, target, weights.w1, weights.b1, weights.w2, weights.b2))
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isRunning, speed, applyWeightUpdate, currentStep, inputX, target, weights])

  // Auto-stop on convergence
  useEffect(() => {
    if (isRunning && snapshot.loss < 0.0001 && currentStep === 5) {
      setIsRunning(false)
    }
  }, [isRunning, snapshot.loss, currentStep])

  // Reset
  const reset = useCallback(() => {
    setIsRunning(false)
    setWeights(INITIAL_WEIGHTS)
    setCurrentStep(-1)
    setEpoch(0)
    setSnapshot(computeSnapshot(1.0, 2.0, INITIAL_WEIGHTS.w1, INITIAL_WEIGHTS.b1, INITIAL_WEIGHTS.w2, INITIAL_WEIGHTS.b2))
  }, [])

  const isIdle = currentStep === -1

  // ==================== LAYOUT ====================
  const padding = 50
  const nodeRadius = 28
  const nodeSpacing = (width - 2 * padding) / 4
  const centerY = height / 2 - 20

  const nodePositions = {
    x: { x: padding, y: centerY },
    z1: { x: padding + nodeSpacing, y: centerY },
    a1: { x: padding + nodeSpacing * 2, y: centerY },
    yPred: { x: padding + nodeSpacing * 3, y: centerY },
    loss: { x: padding + nodeSpacing * 4, y: centerY },
  }

  const nodeLabels: Record<string, string> = {
    x: 'x',
    z1: 'z₁',
    a1: 'a₁',
    yPred: 'ŷ',
    loss: 'L',
  }

  const nodeValues: Record<string, number> = {
    x: snapshot.x,
    z1: snapshot.z1,
    a1: snapshot.a1,
    yPred: snapshot.yPred,
    loss: snapshot.loss,
  }

  const gradientValues: Record<string, number> = {
    loss: 1,
    yPred: snapshot.dL_dyPred,
    a1: snapshot.dL_da1,
    z1: snapshot.dL_dz1,
    x: snapshot.dL_dz1 * snapshot.w1,
  }

  const edges = [
    { from: 'x', to: 'z1', forwardLabel: `×${fmt(snapshot.w1)}+${fmt(snapshot.b1)}`, backLabel: `∂z/∂w₁=${fmt(snapshot.x)}` },
    { from: 'z1', to: 'a1', forwardLabel: 'ReLU', backLabel: `ReLU'=${snapshot.reluGrad}` },
    { from: 'a1', to: 'yPred', forwardLabel: `×${fmt(snapshot.w2)}+${fmt(snapshot.b2)}`, backLabel: `∂ŷ/∂w₂=${fmt(snapshot.a1)}` },
    { from: 'yPred', to: 'loss', forwardLabel: `(·-${fmt(snapshot.target)})²`, backLabel: `2(ŷ-t)=${fmt(snapshot.dL_dyPred)}` },
  ]

  const computedNodes = getComputedNodes(currentStep)
  const activeNode = getActiveNode(currentStep)
  const forwardEdges = getForwardEdges(currentStep)
  const backwardEdges = getBackwardEdges(currentStep)
  const gradientNodes = getGradientNodes(currentStep)
  const reluBlocking = snapshot.z1 <= 0

  const colors = {
    forward: '#3b82f6',
    forwardValue: '#22c55e',
    backward: '#f97316',
    backwardBlocked: '#6b7280',
    nodeEmpty: '#333355',
    nodeEmptyBorder: '#555577',
    nodeFilled: '#1e1e3f',
    nodeBorder: '#6366f1',
    activeGlow: '#ffffff',
    text: '#e5e5e5',
    dimmed: '#666',
  }

  const isForwardPhase = currentStep >= 1 && currentStep <= 5
  const isBackwardPhase = currentStep >= 6 && currentStep <= 9
  const isUpdatePhase = currentStep === 10

  return (
    <div ref={containerRef} className="flex flex-col gap-3">
      {/* Canvas */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Phase label */}
          <Text
            x={width / 2 - 60}
            y={12}
            text={getStepLabel(currentStep)}
            fontSize={13}
            fill={isForwardPhase ? colors.forward : isBackwardPhase ? colors.backward : isUpdatePhase ? '#22c55e' : colors.dimmed}
            fontStyle="bold"
          />

          {/* Draw edges */}
          {edges.map((edge, i) => {
            const from = nodePositions[edge.from as keyof typeof nodePositions]
            const to = nodePositions[edge.to as keyof typeof nodePositions]
            const midX = (from.x + to.x) / 2
            const midY = from.y

            const showForward = forwardEdges.has(i)
            const showBackward = backwardEdges.has(i)
            const isBlockedEdge = reluBlocking && (edge.from === 'x' || edge.from === 'z1')

            return (
              <Group key={`edge-${i}`}>
                {/* Forward arrow (blue, above) */}
                {showForward && (
                  <>
                    <Arrow
                      points={[from.x + nodeRadius, midY - 8, to.x - nodeRadius, midY - 8]}
                      stroke={colors.forward}
                      strokeWidth={2}
                      fill={colors.forward}
                      pointerLength={8}
                      pointerWidth={6}
                    />
                    <Rect
                      x={midX - 45}
                      y={midY - 45}
                      width={90}
                      height={22}
                      fill="#1a1a2e"
                      cornerRadius={4}
                    />
                    <Text
                      x={midX - 40}
                      y={midY - 42}
                      text={edge.forwardLabel}
                      fontSize={11}
                      fill={colors.forward}
                      align="center"
                      width={80}
                    />
                  </>
                )}

                {/* Backward arrow (orange, below) */}
                {showBackward && (
                  <>
                    <Arrow
                      points={[to.x - nodeRadius, midY + 8, from.x + nodeRadius, midY + 8]}
                      stroke={isBlockedEdge ? colors.backwardBlocked : colors.backward}
                      strokeWidth={2}
                      fill={isBlockedEdge ? colors.backwardBlocked : colors.backward}
                      pointerLength={8}
                      pointerWidth={6}
                      dash={isBlockedEdge ? [4, 4] : undefined}
                    />
                    <Rect
                      x={midX - 50}
                      y={midY + 25}
                      width={100}
                      height={20}
                      fill="#1a1a2e"
                      cornerRadius={4}
                    />
                    <Text
                      x={midX - 45}
                      y={midY + 28}
                      text={edge.backLabel}
                      fontSize={10}
                      fill={isBlockedEdge ? colors.backwardBlocked : colors.backward}
                      align="center"
                      width={90}
                    />
                  </>
                )}

                {/* Dashed placeholder edges when idle or not yet reached */}
                {!showForward && !showBackward && (
                  <Line
                    points={[from.x + nodeRadius, midY, to.x - nodeRadius, midY]}
                    stroke="#333355"
                    strokeWidth={1}
                    dash={[4, 6]}
                  />
                )}
              </Group>
            )
          })}

          {/* Draw nodes */}
          {Object.entries(nodePositions).map(([key, pos]) => {
            const isComputed = computedNodes.has(key)
            const isActive = activeNode === key
            const hasGradient = gradientNodes.has(key)
            const isBlocked = reluBlocking && (key === 'x' || key === 'z1')

            return (
              <Group key={key}>
                {/* Active glow ring */}
                {isActive && (
                  <Circle
                    x={pos.x}
                    y={pos.y}
                    radius={nodeRadius + 4}
                    stroke={colors.activeGlow}
                    strokeWidth={2}
                    opacity={0.6}
                  />
                )}

                {/* Node circle */}
                <Circle
                  x={pos.x}
                  y={pos.y}
                  radius={nodeRadius}
                  fill={isComputed ? colors.nodeFilled : colors.nodeEmpty}
                  stroke={isActive ? colors.activeGlow : isComputed ? colors.nodeBorder : colors.nodeEmptyBorder}
                  strokeWidth={isActive ? 3 : 2}
                  dash={isComputed ? undefined : [4, 4]}
                />

                {/* Node label (always shown) */}
                <Text
                  x={pos.x - 12}
                  y={pos.y - (isComputed ? 20 : 8)}
                  text={nodeLabels[key]}
                  fontSize={13}
                  fill={isComputed ? colors.text : colors.nodeEmptyBorder}
                  fontStyle="bold"
                />

                {/* Value (shown when computed) */}
                {isComputed && (
                  <Text
                    x={pos.x - 18}
                    y={pos.y - 3}
                    text={key === 'loss' ? fmt(nodeValues[key], 4) : fmt(nodeValues[key])}
                    fontSize={12}
                    fill={colors.forwardValue}
                    fontFamily="monospace"
                  />
                )}

                {/* Gradient label below node */}
                {hasGradient && (
                  <>
                    <Rect
                      x={pos.x - 25}
                      y={pos.y + nodeRadius + 8}
                      width={50}
                      height={18}
                      fill={isBlocked ? colors.backwardBlocked + '30' : colors.backward + '30'}
                      cornerRadius={3}
                    />
                    <Text
                      x={pos.x - 22}
                      y={pos.y + nodeRadius + 11}
                      text={fmt(gradientValues[key])}
                      fontSize={11}
                      fill={isBlocked ? colors.backwardBlocked : colors.backward}
                      fontFamily="monospace"
                    />
                  </>
                )}
              </Group>
            )
          })}

          {/* ReLU blocking indicator */}
          {reluBlocking && currentStep >= 8 && (
            <Group>
              <Rect
                x={nodePositions.a1.x - 60}
                y={height - 55}
                width={120}
                height={24}
                fill="#dc262630"
                stroke="#dc2626"
                strokeWidth={1}
                cornerRadius={4}
              />
              <Text
                x={nodePositions.a1.x - 55}
                y={height - 50}
                text={'⚠ ReLU blocked!'}
                fontSize={12}
                fill="#dc2626"
              />
            </Group>
          )}

          {/* Legend */}
          <Group>
            <Circle x={20} y={height - 25} radius={5} fill={colors.forward} />
            <Text x={30} y={height - 30} text="Forward" fontSize={10} fill={colors.forward} />
            <Circle x={90} y={height - 25} radius={5} fill={colors.backward} />
            <Text x={100} y={height - 30} text="Backward" fontSize={10} fill={colors.backward} />
          </Group>

          {/* Epoch + loss in canvas corner */}
          <Text
            x={width - 120}
            y={height - 30}
            text={`Epoch ${epoch}  L=${fmt(snapshot.loss, 4)}`}
            fontSize={11}
            fill={colors.dimmed}
          />
        </ZoomableCanvas>
      </div>

      {/* Formula panel */}
      <div className="px-4 py-3 rounded-lg bg-muted/50 border border-border font-mono text-sm min-h-[3rem] flex items-center">
        <span className={
          isForwardPhase ? 'text-blue-400' :
          isBackwardPhase ? 'text-orange-400' :
          isUpdatePhase ? 'text-emerald-400' :
          'text-muted-foreground'
        }>
          {getFormulaForStep(currentStep, snapshot)}
        </span>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap gap-4 items-center justify-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (isRunning) {
                setIsRunning(false)
              } else {
                if (currentStep === -1) {
                  setSnapshot(computeSnapshot(inputX, target, weights.w1, weights.b1, weights.w2, weights.b2))
                }
                setIsRunning(true)
              }
            }}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
            {isRunning ? 'Pause' : 'Run'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleStep}
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

        {/* Learning rate */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {'Learning rate (α):'}
          </span>
          <input
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="w-24 cursor-pointer"
          />
          <span className="font-mono text-sm w-10">{learningRate.toFixed(2)}</span>
        </div>

        {/* Speed toggle */}
        <div className="flex gap-1">
          <Button
            variant={speed === 'slow' ? 'default' : 'outline'}
            size="sm"
            className="text-xs h-7 px-2"
            onClick={() => setSpeed('slow')}
          >
            Slow
          </Button>
          <Button
            variant={speed === 'normal' ? 'default' : 'outline'}
            size="sm"
            className="text-xs h-7 px-2"
            onClick={() => setSpeed('normal')}
          >
            Normal
          </Button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex flex-wrap gap-3 text-sm justify-center">
        <div className="px-3 py-1.5 rounded-md bg-muted">
          <span className="text-muted-foreground">Epoch </span>
          <span className="font-mono">{epoch}</span>
        </div>
        <div className="px-3 py-1.5 rounded-md bg-red-500/10 border border-red-500/30">
          <span className="text-muted-foreground">Loss </span>
          <span className="font-mono text-red-400">{fmt(snapshot.loss, 4)}</span>
        </div>
        <div className="px-3 py-1.5 rounded-md bg-violet-500/10 border border-violet-500/30">
          <span className="text-muted-foreground">w{'₁'}=</span>
          <span className="font-mono text-violet-400">{fmt3(weights.w1)}</span>
        </div>
        <div className="px-3 py-1.5 rounded-md bg-violet-500/10 border border-violet-500/30">
          <span className="text-muted-foreground">b{'₁'}=</span>
          <span className="font-mono text-violet-400">{fmt3(weights.b1)}</span>
        </div>
        <div className="px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/30">
          <span className="text-muted-foreground">w{'₂'}=</span>
          <span className="font-mono text-emerald-400">{fmt3(weights.w2)}</span>
        </div>
        <div className="px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/30">
          <span className="text-muted-foreground">b{'₂'}=</span>
          <span className="font-mono text-emerald-400">{fmt3(weights.b2)}</span>
        </div>
      </div>

      {/* Compact x/target inputs (only editable when idle) */}
      <div className="flex flex-wrap gap-4 items-center justify-center text-sm">
        <div className="flex items-center gap-2">
          <label className="text-muted-foreground">x:</label>
          <input
            type="number"
            min={-2}
            max={2}
            step={0.1}
            value={inputX}
            onChange={(e) => setInputX(parseFloat(e.target.value))}
            disabled={!isIdle}
            className="w-16 px-2 py-1 rounded border bg-background font-mono text-sm disabled:opacity-50"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-muted-foreground">target:</label>
          <input
            type="number"
            min={-2}
            max={2}
            step={0.1}
            value={target}
            onChange={(e) => setTarget(parseFloat(e.target.value))}
            disabled={!isIdle}
            className="w-16 px-2 py-1 rounded border bg-background font-mono text-sm disabled:opacity-50"
          />
        </div>
        {!isIdle && (
          <span className="text-xs text-muted-foreground italic">Reset to change x/target</span>
        )}
      </div>
    </div>
  )
}
