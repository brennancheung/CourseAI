'use client'

import { useState, useRef, useEffect } from 'react'
import { Circle, Text, Arrow, Rect, Group } from 'react-konva'
import { ZoomableCanvas } from '@/components/canvas/ZoomableCanvas'
import { Button } from '@/components/ui/button'
import { RotateCcw, ChevronRight, ChevronLeft } from 'lucide-react'

/**
 * BackpropFlowExplorer - Interactive visualization of forward and backward passes
 *
 * Demonstrates:
 * - Forward pass: values flow left to right
 * - Backward pass: gradients flow right to left
 * - Chain rule: local × incoming = outgoing
 * - ReLU blocking gradients when z < 0
 *
 * Network: x → Linear(w₁,b₁) → ReLU → Linear(w₂,b₂) → Loss
 */

type BackpropFlowExplorerProps = {
  width?: number
  height?: number
}

type Phase = 'forward' | 'backward' | 'both'

export function BackpropFlowExplorer({
  width: widthOverride,
  height: heightOverride,
}: BackpropFlowExplorerProps) {
  // Parameters
  const [x, setX] = useState(1.0)
  const [target, setTarget] = useState(0.5)
  const [w1, setW1] = useState(0.7)
  const [b1, setB1] = useState(0.1)
  const [w2, setW2] = useState(0.8)
  const [b2, setB2] = useState(-0.1)

  // Display mode
  const [phase, setPhase] = useState<Phase>('both')
  const [highlightedNode, setHighlightedNode] = useState<string | null>(null)

  // Container measurement
  const containerRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(700)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateWidth = () => {
      const rect = container.getBoundingClientRect()
      setMeasuredWidth(rect.width)
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  const width = widthOverride ?? measuredWidth
  const height = heightOverride ?? 360

  // ==================== FORWARD PASS ====================
  const z1 = w1 * x + b1
  const a1 = Math.max(0, z1) // ReLU
  const yPred = w2 * a1 + b2
  const loss = Math.pow(yPred - target, 2)

  // ==================== BACKWARD PASS ====================
  // Start from loss, flow backward
  const dL_dyPred = 2 * (yPred - target) // ∂L/∂ŷ = 2(ŷ - t)

  const dL_db2 = dL_dyPred * 1 // ∂L/∂b₂ = ∂L/∂ŷ × 1
  const dL_dw2 = dL_dyPred * a1 // ∂L/∂w₂ = ∂L/∂ŷ × a₁
  const dL_da1 = dL_dyPred * w2 // ∂L/∂a₁ = ∂L/∂ŷ × w₂

  const reluGrad = z1 > 0 ? 1 : 0 // ReLU derivative
  const dL_dz1 = dL_da1 * reluGrad // ∂L/∂z₁ = ∂L/∂a₁ × ReLU'(z₁)

  const dL_db1 = dL_dz1 * 1 // ∂L/∂b₁ = ∂L/∂z₁ × 1
  const dL_dw1 = dL_dz1 * x // ∂L/∂w₁ = ∂L/∂z₁ × x

  // ==================== LAYOUT ====================
  const padding = 50
  const nodeRadius = 28
  const nodeSpacing = (width - 2 * padding) / 4

  // Node positions (left to right)
  const nodes = {
    x: { x: padding, y: height / 2 - 20, label: 'x', value: x },
    z1: { x: padding + nodeSpacing, y: height / 2 - 20, label: 'z₁', value: z1 },
    a1: { x: padding + nodeSpacing * 2, y: height / 2 - 20, label: 'a₁', value: a1 },
    yPred: { x: padding + nodeSpacing * 3, y: height / 2 - 20, label: 'ŷ', value: yPred },
    loss: { x: padding + nodeSpacing * 4, y: height / 2 - 20, label: 'L', value: loss },
  }

  // Operation labels between nodes
  const operations = [
    { from: 'x', to: 'z1', label: `×${w1.toFixed(1)}+${b1.toFixed(1)}`, localGrad: `∂z/∂w₁=${x.toFixed(1)}` },
    { from: 'z1', to: 'a1', label: 'ReLU', localGrad: `ReLU'=${reluGrad}` },
    { from: 'a1', to: 'yPred', label: `×${w2.toFixed(1)}+${b2.toFixed(1)}`, localGrad: `∂ŷ/∂w₂=${a1.toFixed(2)}` },
    { from: 'yPred', to: 'loss', label: `(·-${target.toFixed(1)})²`, localGrad: `2(ŷ-t)=${dL_dyPred.toFixed(2)}` },
  ]

  // Gradient values at each node (for backward flow)
  const gradients = {
    loss: 1, // ∂L/∂L = 1
    yPred: dL_dyPred,
    a1: dL_da1,
    z1: dL_dz1,
    x: dL_dz1 * w1, // continues to input
  }

  // Color coding
  const colors = {
    forward: '#3b82f6', // blue
    forwardValue: '#22c55e', // green
    backward: '#f97316', // orange
    backwardBlocked: '#6b7280', // gray
    node: '#1e1e3f',
    nodeBorder: '#6366f1',
    text: '#e5e5e5',
    dimmed: '#666',
  }

  // Check if ReLU is blocking
  const reluBlocking = z1 <= 0

  const reset = () => {
    setX(1.0)
    setTarget(0.5)
    setW1(0.7)
    setB1(0.1)
    setW2(0.8)
    setB2(-0.1)
  }

  // Format number for display
  const fmt = (n: number) => {
    if (Math.abs(n) < 0.01) return '0.00'
    return n.toFixed(2)
  }

  return (
    <div ref={containerRef} className="flex flex-col gap-4">
      {/* Canvas */}
      <div className="rounded-lg border bg-card overflow-hidden">
        <ZoomableCanvas width={width} height={height} backgroundColor="#1a1a2e">
          {/* Title */}
          <Text
            x={width / 2 - 80}
            y={15}
            text={phase === 'forward' ? 'Forward Pass →' : phase === 'backward' ? '← Backward Pass' : 'Forward → | ← Backward'}
            fontSize={14}
            fill={colors.text}
            fontStyle="bold"
          />

          {/* Draw connections */}
          {operations.map((op, i) => {
            const fromNode = nodes[op.from as keyof typeof nodes]
            const toNode = nodes[op.to as keyof typeof nodes]
            const midX = (fromNode.x + toNode.x) / 2
            const midY = fromNode.y

            return (
              <Group key={`conn-${i}`}>
                {/* Forward arrow (blue, top) */}
                {(phase === 'forward' || phase === 'both') && (
                  <>
                    <Arrow
                      points={[fromNode.x + nodeRadius, midY - 8, toNode.x - nodeRadius, midY - 8]}
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
                      text={op.label}
                      fontSize={11}
                      fill={colors.forward}
                      align="center"
                      width={80}
                    />
                  </>
                )}

                {/* Backward arrow (orange, bottom) */}
                {(phase === 'backward' || phase === 'both') && (
                  <>
                    <Arrow
                      points={[toNode.x - nodeRadius, midY + 8, fromNode.x + nodeRadius, midY + 8]}
                      stroke={reluBlocking && (op.from === 'x' || op.from === 'z1') ? colors.backwardBlocked : colors.backward}
                      strokeWidth={2}
                      fill={reluBlocking && (op.from === 'x' || op.from === 'z1') ? colors.backwardBlocked : colors.backward}
                      pointerLength={8}
                      pointerWidth={6}
                      dash={reluBlocking && (op.from === 'x' || op.from === 'z1') ? [4, 4] : undefined}
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
                      text={op.localGrad}
                      fontSize={10}
                      fill={reluBlocking && (op.from === 'x' || op.from === 'z1') ? colors.backwardBlocked : colors.backward}
                      align="center"
                      width={90}
                    />
                  </>
                )}
              </Group>
            )
          })}

          {/* Draw nodes */}
          {Object.entries(nodes).map(([key, node]) => {
            const isHighlighted = highlightedNode === key
            const grad = gradients[key as keyof typeof gradients]
            const isBlocked = reluBlocking && (key === 'x' || key === 'z1')

            return (
              <Group
                key={key}
                onMouseEnter={() => setHighlightedNode(key)}
                onMouseLeave={() => setHighlightedNode(null)}
              >
                {/* Node circle */}
                <Circle
                  x={node.x}
                  y={node.y}
                  radius={nodeRadius}
                  fill={colors.node}
                  stroke={isHighlighted ? '#fff' : colors.nodeBorder}
                  strokeWidth={isHighlighted ? 3 : 2}
                />

                {/* Node label */}
                <Text
                  x={node.x - 12}
                  y={node.y - 22}
                  text={node.label}
                  fontSize={13}
                  fill={colors.text}
                  fontStyle="bold"
                />

                {/* Forward value (inside node) */}
                {(phase === 'forward' || phase === 'both') && (
                  <Text
                    x={node.x - 18}
                    y={node.y - 5}
                    text={fmt(node.value)}
                    fontSize={12}
                    fill={colors.forwardValue}
                    fontFamily="monospace"
                  />
                )}

                {/* Backward gradient (below node) */}
                {(phase === 'backward' || phase === 'both') && (
                  <>
                    <Rect
                      x={node.x - 25}
                      y={node.y + nodeRadius + 8}
                      width={50}
                      height={18}
                      fill={isBlocked ? colors.backwardBlocked + '30' : colors.backward + '30'}
                      cornerRadius={3}
                    />
                    <Text
                      x={node.x - 22}
                      y={node.y + nodeRadius + 11}
                      text={fmt(grad)}
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
          {reluBlocking && (phase === 'backward' || phase === 'both') && (
            <Group>
              <Rect
                x={nodes.a1.x - 60}
                y={height - 55}
                width={120}
                height={24}
                fill="#dc262630"
                stroke="#dc2626"
                strokeWidth={1}
                cornerRadius={4}
              />
              <Text
                x={nodes.a1.x - 55}
                y={height - 50}
                text="⚠ ReLU blocked!"
                fontSize={12}
                fill="#dc2626"
              />
            </Group>
          )}

          {/* Legend */}
          <Group>
            {(phase === 'forward' || phase === 'both') && (
              <>
                <Circle x={20} y={height - 30} radius={5} fill={colors.forward} />
                <Text x={30} y={height - 35} text="Forward" fontSize={10} fill={colors.forward} />
              </>
            )}
            {(phase === 'backward' || phase === 'both') && (
              <>
                <Circle x={90} y={height - 30} radius={5} fill={colors.backward} />
                <Text x={100} y={height - 35} text="Backward" fontSize={10} fill={colors.backward} />
              </>
            )}
          </Group>

          {/* Target label */}
          <Text
            x={width - 80}
            y={height - 35}
            text={`target=${target.toFixed(1)}`}
            fontSize={11}
            fill={colors.dimmed}
          />
        </ZoomableCanvas>
      </div>

      {/* Phase selector */}
      <div className="flex items-center justify-center gap-2">
        <Button variant="outline" size="sm" onClick={() => setPhase('forward')}>
          <ChevronRight className="w-4 h-4 mr-1" />
          Forward
        </Button>
        <Button variant="outline" size="sm" onClick={() => setPhase('backward')}>
          <ChevronLeft className="w-4 h-4 mr-1" />
          Backward
        </Button>
        <Button variant={phase === 'both' ? 'default' : 'outline'} size="sm" onClick={() => setPhase('both')}>
          Both
        </Button>
        <Button variant="outline" size="sm" onClick={reset}>
          <RotateCcw className="w-4 h-4 mr-1" />
          Reset
        </Button>
      </div>

      {/* Controls */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Input & Target */}
        <div className="space-y-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-sm font-medium text-blue-400">Input & Target</p>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-12">x:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={x}
                onChange={(e) => setX(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{x.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-12">target:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={target}
                onChange={(e) => setTarget(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{target.toFixed(1)}</span>
            </div>
          </div>
        </div>

        {/* Weights Layer 1 */}
        <div className="space-y-3 p-3 rounded-lg bg-violet-500/10 border border-violet-500/20">
          <p className="text-sm font-medium text-violet-400">Layer 1 (w₁, b₁)</p>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-12">w₁:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={w1}
                onChange={(e) => setW1(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{w1.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground w-12">b₁:</label>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.1}
                value={b1}
                onChange={(e) => setB1(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="font-mono text-sm w-12 text-right">{b1.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Layer 2 weights */}
      <div className="space-y-3 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
        <p className="text-sm font-medium text-emerald-400">Layer 2 (w₂, b₂)</p>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground w-12">w₂:</label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={w2}
              onChange={(e) => setW2(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="font-mono text-sm w-12 text-right">{w2.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-muted-foreground w-12">b₂:</label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={b2}
              onChange={(e) => setB2(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="font-mono text-sm w-12 text-right">{b2.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* Gradient results */}
      <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/20">
        <p className="text-sm font-medium text-orange-400 mb-3">Final Gradients (what we use to update weights)</p>
        <div className="grid gap-2 md:grid-cols-2 font-mono text-sm">
          <div className={`px-3 py-2 rounded ${reluBlocking ? 'bg-gray-500/20 text-gray-400' : 'bg-muted'}`}>
            ∂L/∂w₁ = {fmt(dL_dw1)} {reluBlocking && '(blocked)'}
          </div>
          <div className={`px-3 py-2 rounded ${reluBlocking ? 'bg-gray-500/20 text-gray-400' : 'bg-muted'}`}>
            ∂L/∂b₁ = {fmt(dL_db1)} {reluBlocking && '(blocked)'}
          </div>
          <div className="px-3 py-2 rounded bg-muted">
            ∂L/∂w₂ = {fmt(dL_dw2)}
          </div>
          <div className="px-3 py-2 rounded bg-muted">
            ∂L/∂b₂ = {fmt(dL_db2)}
          </div>
        </div>
      </div>

      {/* Insight text */}
      <div className="text-sm text-muted-foreground space-y-2">
        <p>
          <span className="text-blue-400">Forward:</span> Values flow left to right.{' '}
          <span className="text-orange-400">Backward:</span> Gradients flow right to left.
        </p>
        <p>
          At each step: <span className="font-mono">outgoing = local × incoming</span>.
          {reluBlocking && (
            <span className="text-red-400 ml-2">
              ⚠ When z₁ ≤ 0, ReLU gradient = 0, blocking all upstream gradients!
            </span>
          )}
        </p>
      </div>
    </div>
  )
}
