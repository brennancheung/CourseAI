'use client'

import { useState, useMemo, useCallback } from 'react'
import { useContainerWidth } from '@/hooks/useContainerWidth'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type GraphNodeKind = 'input' | 'param' | 'op' | 'output'

type GraphNode = {
  id: string
  label: string
  kind: GraphNodeKind
  /** Column index (0 = leftmost) for layout */
  col: number
  /** Row index within the column */
  row: number
}

type GraphEdge = {
  from: string
  to: string
}

type GraphDef = {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

type ComputedNode = {
  id: string
  forwardValue: number
  /** Local derivative of this node's output w.r.t. each input */
  localDerivatives: Record<string, number>
  /** Gradient flowing INTO this node from the right (dL/d(this node's output)) */
  gradient: number
}

type GraphMode = 'simple' | 'network' | 'fanout'

type ViewMode = 'graph' | 'steps'

// ---------------------------------------------------------------------------
// Graph definitions
// ---------------------------------------------------------------------------

const SIMPLE_GRAPH: GraphDef = {
  nodes: [
    { id: 'x', label: 'x', kind: 'input', col: 0, row: 0 },
    { id: 'add', label: '+ 1', kind: 'op', col: 1, row: 0 },
    { id: 'sq', label: 'x²', kind: 'op', col: 2, row: 0 },
    { id: 'f', label: 'f', kind: 'output', col: 3, row: 0 },
  ],
  edges: [
    { from: 'x', to: 'add' },
    { from: 'add', to: 'sq' },
    { from: 'sq', to: 'f' },
  ],
}

const NETWORK_GRAPH: GraphDef = {
  nodes: [
    { id: 'x', label: 'x', kind: 'input', col: 0, row: 1 },
    { id: 'w1', label: 'w₁', kind: 'param', col: 0, row: 0 },
    { id: 'b1', label: 'b₁', kind: 'param', col: 0, row: 2 },
    { id: 'mul1', label: '×', kind: 'op', col: 1, row: 0 },
    { id: 'add1', label: '+', kind: 'op', col: 2, row: 0 },
    { id: 'relu', label: 'ReLU', kind: 'op', col: 3, row: 0 },
    { id: 'w2', label: 'w₂', kind: 'param', col: 3, row: 2 },
    { id: 'mul2', label: '×', kind: 'op', col: 4, row: 0 },
    { id: 'b2', label: 'b₂', kind: 'param', col: 4, row: 2 },
    { id: 'add2', label: '+', kind: 'op', col: 5, row: 0 },
    { id: 'y', label: 'y', kind: 'param', col: 6, row: 2 },
    { id: 'mse', label: 'MSE', kind: 'op', col: 6, row: 0 },
    { id: 'L', label: 'L', kind: 'output', col: 7, row: 0 },
  ],
  edges: [
    { from: 'x', to: 'mul1' },
    { from: 'w1', to: 'mul1' },
    { from: 'mul1', to: 'add1' },
    { from: 'b1', to: 'add1' },
    { from: 'add1', to: 'relu' },
    { from: 'relu', to: 'mul2' },
    { from: 'w2', to: 'mul2' },
    { from: 'mul2', to: 'add2' },
    { from: 'b2', to: 'add2' },
    { from: 'add2', to: 'mse' },
    { from: 'y', to: 'mse' },
    { from: 'mse', to: 'L' },
  ],
}

const FANOUT_GRAPH: GraphDef = {
  nodes: [
    { id: 'x', label: 'x', kind: 'input', col: 0, row: 1 },
    { id: 'add', label: '+ 1', kind: 'op', col: 1, row: 2 },
    { id: 'mul', label: '×', kind: 'op', col: 2, row: 1 },
    { id: 'f', label: 'f', kind: 'output', col: 3, row: 1 },
  ],
  edges: [
    { from: 'x', to: 'mul' },
    { from: 'x', to: 'add' },
    { from: 'add', to: 'mul' },
    { from: 'mul', to: 'f' },
  ],
}

// ---------------------------------------------------------------------------
// Forward / backward computation for each graph mode
// ---------------------------------------------------------------------------

function computeSimple(xVal: number): { nodes: Record<string, ComputedNode>; steps: StepItem[] } {
  const addOut = xVal + 1
  const sqOut = addOut * addOut
  const fOut = sqOut

  // Backward: dL/df = 1 (output seed)
  const dFdSq = 1
  const dSqDadd = 2 * addOut // d(u^2)/du = 2u
  const dAddDx = 1

  const gradF = 1
  const gradSq = gradF * dFdSq
  const gradAdd = gradSq * dSqDadd
  const gradX = gradAdd * dAddDx

  const nodes: Record<string, ComputedNode> = {
    x: { id: 'x', forwardValue: xVal, localDerivatives: {}, gradient: gradX },
    add: { id: 'add', forwardValue: addOut, localDerivatives: { x: dAddDx }, gradient: gradAdd },
    sq: { id: 'sq', forwardValue: sqOut, localDerivatives: { add: dSqDadd }, gradient: gradSq },
    f: { id: 'f', forwardValue: fOut, localDerivatives: { sq: dFdSq }, gradient: gradF },
  }

  const steps: StepItem[] = [
    { label: 'Forward: add', formula: `x + 1 = ${fmt(xVal)} + 1`, result: fmt(addOut) },
    { label: 'Forward: square', formula: `(${fmt(addOut)})²`, result: fmt(sqOut) },
    { label: 'Backward: df/dsq', formula: '1 (output seed)', result: '1' },
    { label: 'Backward: df/dadd', formula: `1 × 2·(${fmt(addOut)})`, result: fmt(gradAdd) },
    { label: 'Backward: df/dx', formula: `${fmt(gradAdd)} × 1`, result: fmt(gradX) },
    { label: 'Verify', formula: `d/dx (x+1)² = 2(x+1) = 2·${fmt(addOut)}`, result: fmt(gradX) },
  ]

  return { nodes, steps }
}

function computeNetwork(
  xVal: number,
  w1: number,
  b1: number,
  w2: number,
  b2: number,
  yVal: number,
): { nodes: Record<string, ComputedNode>; steps: StepItem[] } {
  const mul1Out = w1 * xVal
  const add1Out = mul1Out + b1
  const reluOut = Math.max(0, add1Out)
  const mul2Out = w2 * reluOut
  const add2Out = mul2Out + b2
  const yHat = add2Out
  const loss = (yVal - yHat) ** 2

  // Backward
  const gradL = 1
  const dMse = -2 * (yVal - yHat) // dL/dyHat
  const gradMse = gradL * 1 // mse -> L is identity
  const gradAdd2 = gradMse * dMse
  const gradMul2 = gradAdd2 * 1 // d(add2)/d(mul2) = 1
  const gradB2 = gradAdd2 * 1
  const gradRelu = gradMul2 * w2 // d(mul2)/d(relu) = w2
  const gradW2 = gradMul2 * reluOut
  const reluDeriv = add1Out > 0 ? 1 : 0
  const gradAdd1 = gradRelu * reluDeriv
  const gradMul1 = gradAdd1 * 1
  const gradB1 = gradAdd1 * 1
  const gradX = gradMul1 * w1
  const gradW1 = gradMul1 * xVal

  const nodes: Record<string, ComputedNode> = {
    x: { id: 'x', forwardValue: xVal, localDerivatives: {}, gradient: gradX },
    w1: { id: 'w1', forwardValue: w1, localDerivatives: {}, gradient: gradW1 },
    b1: { id: 'b1', forwardValue: b1, localDerivatives: {}, gradient: gradB1 },
    mul1: { id: 'mul1', forwardValue: mul1Out, localDerivatives: { w1: xVal, x: w1 }, gradient: gradMul1 },
    add1: { id: 'add1', forwardValue: add1Out, localDerivatives: { mul1: 1, b1: 1 }, gradient: gradAdd1 },
    relu: { id: 'relu', forwardValue: reluOut, localDerivatives: { add1: reluDeriv }, gradient: gradRelu },
    w2: { id: 'w2', forwardValue: w2, localDerivatives: {}, gradient: gradW2 },
    mul2: { id: 'mul2', forwardValue: mul2Out, localDerivatives: { relu: w2, w2: reluOut }, gradient: gradMul2 },
    b2: { id: 'b2', forwardValue: b2, localDerivatives: {}, gradient: gradB2 },
    add2: { id: 'add2', forwardValue: add2Out, localDerivatives: { mul2: 1, b2: 1 }, gradient: gradAdd2 },
    y: { id: 'y', forwardValue: yVal, localDerivatives: {}, gradient: 0 },
    mse: { id: 'mse', forwardValue: loss, localDerivatives: { add2: dMse, y: 2 * (yVal - yHat) }, gradient: gradMse },
    L: { id: 'L', forwardValue: loss, localDerivatives: { mse: 1 }, gradient: gradL },
  }

  const steps: StepItem[] = [
    { label: 'Forward: w₁×x', formula: `${fmt(w1)} × ${fmt(xVal)}`, result: fmt(mul1Out) },
    { label: 'Forward: +b₁', formula: `${fmt(mul1Out)} + ${fmt(b1)}`, result: fmt(add1Out) },
    { label: 'Forward: ReLU', formula: `max(0, ${fmt(add1Out)})`, result: fmt(reluOut) },
    { label: 'Forward: w₂×a₁', formula: `${fmt(w2)} × ${fmt(reluOut)}`, result: fmt(mul2Out) },
    { label: 'Forward: +b₂', formula: `${fmt(mul2Out)} + ${fmt(b2)}`, result: fmt(add2Out) },
    { label: 'Forward: MSE', formula: `(${fmt(yVal)} - ${fmt(yHat)})²`, result: fmt(loss) },
    { label: 'Backward: dL/dŷ', formula: `-2(${fmt(yVal)} - ${fmt(yHat)})`, result: fmt(dMse) },
    { label: 'Backward: dL/dw₂', formula: `${fmt(dMse)} × ${fmt(reluOut)}`, result: fmt(gradW2) },
    { label: 'Backward: dL/db₂', formula: `${fmt(dMse)} × 1`, result: fmt(gradB2) },
    { label: 'Backward: dL/da₁', formula: `${fmt(dMse)} × ${fmt(w2)}`, result: fmt(gradRelu) },
    { label: 'Backward: through ReLU', formula: `${fmt(gradRelu)} × ${reluDeriv}`, result: fmt(gradAdd1) },
    { label: 'Backward: dL/dw₁', formula: `${fmt(gradAdd1)} × ${fmt(xVal)}`, result: fmt(gradW1) },
    { label: 'Backward: dL/db₁', formula: `${fmt(gradAdd1)} × 1`, result: fmt(gradB1) },
  ]

  return { nodes, steps }
}

function computeFanout(xVal: number): { nodes: Record<string, ComputedNode>; steps: StepItem[] } {
  const addOut = xVal + 1
  const mulOut = xVal * addOut
  const fOut = mulOut

  // Backward
  const gradF = 1
  const gradMul = gradF * 1
  // d(mul)/d(x_top) = addOut (the other input), d(mul)/d(add) = xVal
  const gradFromMulToX = gradMul * addOut
  const gradFromMulToAdd = gradMul * xVal
  const gradAdd = gradFromMulToAdd
  // d(add)/d(x_bottom) = 1
  const gradFromAddToX = gradAdd * 1
  // Fan-out: x gets gradient from BOTH paths
  const gradX = gradFromMulToX + gradFromAddToX

  const nodes: Record<string, ComputedNode> = {
    x: { id: 'x', forwardValue: xVal, localDerivatives: {}, gradient: gradX },
    add: { id: 'add', forwardValue: addOut, localDerivatives: { x: 1 }, gradient: gradAdd },
    mul: { id: 'mul', forwardValue: mulOut, localDerivatives: { x: addOut, add: xVal }, gradient: gradMul },
    f: { id: 'f', forwardValue: fOut, localDerivatives: { mul: 1 }, gradient: gradF },
  }

  const steps: StepItem[] = [
    { label: 'Forward: x + 1', formula: `${fmt(xVal)} + 1`, result: fmt(addOut) },
    { label: 'Forward: x × (x+1)', formula: `${fmt(xVal)} × ${fmt(addOut)}`, result: fmt(mulOut) },
    { label: 'Backward: df/dmul', formula: '1 (output seed)', result: '1' },
    { label: 'Backward: × node → top (x)', formula: `1 × ${fmt(addOut)} (value of other input)`, result: fmt(gradFromMulToX) },
    { label: 'Backward: × node → bottom (add)', formula: `1 × ${fmt(xVal)} (value of other input)`, result: fmt(gradFromMulToAdd) },
    { label: 'Backward: add → x', formula: `${fmt(gradFromMulToAdd)} × 1`, result: fmt(gradFromAddToX) },
    { label: 'Fan-out sum at x', formula: `${fmt(gradFromMulToX)} + ${fmt(gradFromAddToX)}`, result: fmt(gradX) },
    { label: 'Verify', formula: `d/dx(x² + x) = 2x + 1 = 2·${fmt(xVal)} + 1`, result: fmt(2 * xVal + 1) },
  ]

  return { nodes, steps }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(n: number, decimals = 2): string {
  if (Number.isInteger(n) && Math.abs(n) < 1000) return String(n)
  return n.toFixed(decimals)
}

type StepItem = {
  label: string
  formula: string
  result: string
}

function getGraphModeLabel(mode: GraphMode): string {
  const labels: Record<GraphMode, string> = {
    simple: 'f(x) = (x+1)²',
    network: '2-Layer Network',
    fanout: 'f(x) = x·(x+1)',
  }
  return labels[mode]
}

function getGraphDef(mode: GraphMode): GraphDef {
  const graphs: Record<GraphMode, GraphDef> = {
    simple: SIMPLE_GRAPH,
    network: NETWORK_GRAPH,
    fanout: FANOUT_GRAPH,
  }
  return graphs[mode]
}

// ---------------------------------------------------------------------------
// Node colors
// ---------------------------------------------------------------------------

function getNodeFill(kind: GraphNodeKind): string {
  const fills: Record<GraphNodeKind, string> = {
    input: '#3b82f620',
    param: '#a78bfa20',
    op: '#1e293b',
    output: '#10b98120',
  }
  return fills[kind]
}

function getNodeStroke(kind: GraphNodeKind): string {
  const strokes: Record<GraphNodeKind, string> = {
    input: '#3b82f6',
    param: '#a78bfa',
    op: '#475569',
    output: '#10b981',
  }
  return strokes[kind]
}

function getNodeTextColor(kind: GraphNodeKind): string {
  const colors: Record<GraphNodeKind, string> = {
    input: '#60a5fa',
    param: '#c4b5fd',
    op: '#94a3b8',
    output: '#6ee7b7',
  }
  return colors[kind]
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const NODE_W = 52
const NODE_H = 36
const ROW_GAP = 56

function layoutNode(node: GraphNode, totalCols: number, totalRowsPerCol: Record<number, number>, svgWidth: number): { cx: number; cy: number } {
  const usableWidth = svgWidth - NODE_W * 2
  const colSpacing = totalCols <= 1 ? 0 : usableWidth / (totalCols - 1)
  const cx = NODE_W + node.col * colSpacing
  const rowsInCol = totalRowsPerCol[node.col] ?? 1
  const totalHeight = (rowsInCol - 1) * ROW_GAP
  const startY = 160 - totalHeight / 2
  const cy = startY + node.row * ROW_GAP
  return { cx, cy }
}

// ---------------------------------------------------------------------------
// SVG sub-components
// ---------------------------------------------------------------------------

function GraphNodeSVG({
  node,
  cx,
  cy,
  computed,
  isHovered,
  onHover,
  onLeave,
  onClick,
}: {
  node: GraphNode
  cx: number
  cy: number
  computed: ComputedNode
  isHovered: boolean
  onHover: () => void
  onLeave: () => void
  onClick: () => void
}) {
  const w = NODE_W
  const h = NODE_H
  const rx = node.kind === 'op' ? 8 : 16

  return (
    <g
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
      onClick={onClick}
      className="cursor-pointer"
    >
      {/* Node shape */}
      <rect
        x={cx - w / 2}
        y={cy - h / 2}
        width={w}
        height={h}
        rx={rx}
        fill={getNodeFill(node.kind)}
        stroke={isHovered ? '#f59e0b' : getNodeStroke(node.kind)}
        strokeWidth={isHovered ? 2 : 1.5}
      />
      {/* Node label */}
      <text
        x={cx}
        y={cy + 1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={getNodeTextColor(node.kind)}
        fontSize={12}
        fontWeight={600}
        fontFamily="ui-monospace, monospace"
      >
        {node.label}
      </text>
      {/* Forward value above */}
      <text
        x={cx}
        y={cy - h / 2 - 6}
        textAnchor="middle"
        fill="#60a5fa"
        fontSize={10}
        fontFamily="ui-monospace, monospace"
      >
        {fmt(computed.forwardValue)}
      </text>
      {/* Gradient below */}
      <text
        x={cx}
        y={cy + h / 2 + 14}
        textAnchor="middle"
        fill="#f87171"
        fontSize={10}
        fontFamily="ui-monospace, monospace"
      >
        {"∇"}{fmt(computed.gradient)}
      </text>
    </g>
  )
}

function EdgeSVG({
  fromX,
  fromY,
  toX,
  toY,
}: {
  fromX: number
  fromY: number
  toX: number
  toY: number
}) {
  // Offset by half node width
  const startX = fromX + NODE_W / 2
  const endX = toX - NODE_W / 2

  return (
    <line
      x1={startX}
      y1={fromY}
      x2={endX}
      y2={toY}
      stroke="#334155"
      strokeWidth={1.5}
      markerEnd="url(#arrowhead)"
    />
  )
}

// ---------------------------------------------------------------------------
// Tooltip panel for hovered node
// ---------------------------------------------------------------------------

function NodeDetail({
  nodeId,
  graphDef,
  computedNodes,
}: {
  nodeId: string
  graphDef: GraphDef
  computedNodes: Record<string, ComputedNode>
}) {
  const computed = computedNodes[nodeId]
  if (!computed) return null

  const node = graphDef.nodes.find(n => n.id === nodeId)
  if (!node) return null

  // Find incoming edges
  const incomingEdges = graphDef.edges.filter(e => e.to === nodeId)
  // Find outgoing edges
  const outgoingEdges = graphDef.edges.filter(e => e.from === nodeId)

  if (node.kind === 'input' || node.kind === 'param') {
    return (
      <div className="rounded-lg border bg-muted/30 p-3 text-sm space-y-2">
        <div className="font-semibold text-foreground">{node.label} ({node.kind})</div>
        <div className="flex gap-4">
          <span className="text-blue-400">Value: {fmt(computed.forwardValue, 4)}</span>
          <span className="text-red-400">Gradient: {fmt(computed.gradient, 4)}</span>
        </div>
        {node.kind === 'param' && (
          <p className="text-xs text-muted-foreground">
            This gradient tells us how to update {node.label} to reduce the loss.
          </p>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-lg border bg-muted/30 p-3 text-sm space-y-2">
      <div className="font-semibold text-foreground">{node.label} operation</div>
      <div className="flex gap-4">
        <span className="text-blue-400">Output: {fmt(computed.forwardValue, 4)}</span>
        <span className="text-red-400">Incoming {"∇"}: {fmt(computed.gradient, 4)}</span>
      </div>
      {incomingEdges.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs text-muted-foreground font-semibold">Backward through this node:</div>
          {incomingEdges.map(edge => {
            const localDeriv = computed.localDerivatives[edge.from]
            if (localDeriv === undefined) return null
            const outgoing = computed.gradient * localDeriv
            return (
              <div key={edge.from} className="font-mono text-xs flex items-center gap-1 flex-wrap">
                <span className="text-red-400">{fmt(computed.gradient, 4)}</span>
                <span className="text-muted-foreground">{"×"}</span>
                <span className="text-amber-400">{fmt(localDeriv, 4)}</span>
                <span className="text-muted-foreground">=</span>
                <span className="text-red-400">{fmt(outgoing, 4)}</span>
                <span className="text-muted-foreground text-xs ml-1">({"→"} {edge.from})</span>
              </div>
            )
          })}
          <p className="text-xs text-muted-foreground mt-1">
            {"incoming ∇ × local derivative = outgoing ∇"}
          </p>
        </div>
      )}
      {outgoingEdges.length === 0 && node.kind === 'output' && (
        <p className="text-xs text-muted-foreground">
          Output node: gradient seed = 1 (start of backward pass).
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step list view
// ---------------------------------------------------------------------------

function StepListView({ steps }: { steps: StepItem[] }) {
  return (
    <div className="space-y-2">
      {steps.map((step, i) => {
        const isBackward = step.label.toLowerCase().includes('backward') || step.label.toLowerCase().includes('fan-out')
        const isVerify = step.label.toLowerCase().includes('verify')
        return (
          <div
            key={i}
            className={cn(
              'flex items-start gap-3 rounded-md px-3 py-2 text-sm',
              isVerify && 'bg-emerald-500/10 border border-emerald-500/30',
              isBackward && !isVerify && 'bg-rose-500/5',
              !isBackward && !isVerify && 'bg-blue-500/5',
            )}
          >
            <div
              className={cn(
                'flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-bold',
                isBackward ? 'bg-rose-500/20 text-rose-400' : 'bg-blue-500/20 text-blue-400',
                isVerify && 'bg-emerald-500/20 text-emerald-400',
              )}
            >
              {i + 1}
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="font-semibold text-xs text-muted-foreground">{step.label}</div>
              <div className="font-mono text-xs">
                <span className="text-muted-foreground">{step.formula}</span>
                <span className="text-foreground font-bold ml-2">= {step.result}</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type ComputationalGraphExplorerProps = {
  width?: number
  height?: number
}

export function ComputationalGraphExplorer({
  width: widthOverride,
}: ComputationalGraphExplorerProps) {
  const { containerRef, width: measuredWidth } = useContainerWidth(600)
  const width = widthOverride ?? measuredWidth

  const [graphMode, setGraphMode] = useState<GraphMode>('simple')
  const [viewMode, setViewMode] = useState<ViewMode>('graph')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  // Parameters
  const [xVal, setXVal] = useState(3)
  const [w1, setW1] = useState(0.5)
  const [b1, setB1] = useState(0.1)
  const [w2, setW2] = useState(-0.3)
  const [b2, setB2] = useState(0.2)

  const graphDef = getGraphDef(graphMode)

  const { nodes: computedNodes, steps } = useMemo(() => {
    if (graphMode === 'simple') return computeSimple(xVal)
    if (graphMode === 'fanout') return computeFanout(xVal)
    return computeNetwork(xVal, w1, b1, w2, b2, 1)
  }, [graphMode, xVal, w1, b1, w2, b2])

  // Layout: figure out columns and rows
  const totalCols = useMemo(() => {
    return Math.max(...graphDef.nodes.map(n => n.col)) + 1
  }, [graphDef])

  const totalRowsPerCol = useMemo(() => {
    const map: Record<number, number> = {}
    for (const node of graphDef.nodes) {
      map[node.col] = Math.max(map[node.col] ?? 0, node.row + 1)
    }
    return map
  }, [graphDef])

  const svgHeight = 320

  const nodePositions = useMemo(() => {
    const pos: Record<string, { cx: number; cy: number }> = {}
    for (const node of graphDef.nodes) {
      pos[node.id] = layoutNode(node, totalCols, totalRowsPerCol, width)
    }
    return pos
  }, [graphDef, totalCols, totalRowsPerCol, width])

  const handleHover = useCallback((id: string) => setHoveredNode(id), [])
  const handleLeave = useCallback(() => setHoveredNode(null), [])
  const handleClick = useCallback((id: string) => {
    setHoveredNode(prev => prev === id ? null : id)
  }, [])

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Graph mode selector */}
      <div className="flex flex-wrap gap-2">
        {(['simple', 'network', 'fanout'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => {
              setGraphMode(mode)
              setHoveredNode(null)
              if (mode === 'simple' || mode === 'fanout') setXVal(3)
              if (mode === 'network') setXVal(2)
            }}
            className={cn(
              'cursor-pointer px-3 py-1.5 rounded-full text-xs font-medium transition-colors',
              graphMode === mode
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted hover:bg-muted/80 text-muted-foreground',
            )}
          >
            {getGraphModeLabel(mode)}
          </button>
        ))}
      </div>

      {/* View mode toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode('graph')}
          className={cn(
            'cursor-pointer px-3 py-1 rounded text-xs font-medium transition-colors',
            viewMode === 'graph'
              ? 'bg-blue-500/20 text-blue-400'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          Graph View
        </button>
        <button
          onClick={() => setViewMode('steps')}
          className={cn(
            'cursor-pointer px-3 py-1 rounded text-xs font-medium transition-colors',
            viewMode === 'steps'
              ? 'bg-rose-500/20 text-rose-400'
              : 'bg-muted text-muted-foreground hover:bg-muted/80',
          )}
        >
          Step-by-Step
        </button>
      </div>

      {/* Graph view */}
      {viewMode === 'graph' && (
        <div className="rounded-lg border bg-[#0f1219] overflow-hidden">
          <svg width={width} height={svgHeight} viewBox={`0 0 ${width} ${svgHeight}`}>
            <defs>
              <marker
                id="arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="8"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#475569" />
              </marker>
            </defs>

            {/* Legend */}
            <text x={12} y={16} fill="#60a5fa" fontSize={9} fontFamily="ui-monospace, monospace">
              blue = forward values
            </text>
            <text x={12} y={28} fill="#f87171" fontSize={9} fontFamily="ui-monospace, monospace">
              {"red = gradients (∇)"}
            </text>

            {/* Edges */}
            {graphDef.edges.map(edge => {
              const from = nodePositions[edge.from]
              const to = nodePositions[edge.to]
              if (!from || !to) return null
              return (
                <EdgeSVG
                  key={`${edge.from}-${edge.to}`}
                  fromX={from.cx}
                  fromY={from.cy}
                  toX={to.cx}
                  toY={to.cy}
                />
              )
            })}

            {/* Nodes */}
            {graphDef.nodes.map(node => {
              const pos = nodePositions[node.id]
              const computed = computedNodes[node.id]
              if (!pos || !computed) return null
              return (
                <GraphNodeSVG
                  key={node.id}
                  node={node}
                  cx={pos.cx}
                  cy={pos.cy}
                  computed={computed}
                  isHovered={hoveredNode === node.id}
                  onHover={() => handleHover(node.id)}
                  onLeave={handleLeave}
                  onClick={() => handleClick(node.id)}
                />
              )
            })}
          </svg>
        </div>
      )}

      {/* Step list view */}
      {viewMode === 'steps' && <StepListView steps={steps} />}

      {/* Node detail panel (shown when hovering in graph view) */}
      {viewMode === 'graph' && hoveredNode && (
        <NodeDetail
          nodeId={hoveredNode}
          graphDef={graphDef}
          computedNodes={computedNodes}
        />
      )}

      {/* Parameter controls */}
      <div className="space-y-3">
        <div className="space-y-1">
          <label className="flex justify-between text-xs text-muted-foreground">
            <span>x</span>
            <span className="font-mono">{fmt(xVal)}</span>
          </label>
          <input
            type="range"
            min={-3}
            max={5}
            step={0.1}
            value={xVal}
            onChange={e => setXVal(parseFloat(e.target.value))}
            className="w-full cursor-pointer"
          />
        </div>

        {graphMode === 'network' && (
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
              Adjust network parameters
            </summary>
            <div className="mt-3 space-y-3">
              <div className="text-xs text-muted-foreground">
                Target: <span className="font-mono">y = 1</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'w₁', value: w1, setter: setW1 },
                { label: 'b₁', value: b1, setter: setB1 },
                { label: 'w₂', value: w2, setter: setW2 },
                { label: 'b₂', value: b2, setter: setB2 },
              ].map(({ label, value, setter }) => (
                <div key={label} className="space-y-1">
                  <label className="flex justify-between text-xs text-muted-foreground">
                    <span>{label}</span>
                    <span className="font-mono">{fmt(value, 3)}</span>
                  </label>
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.01}
                    value={value}
                    onChange={e => setter(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                </div>
              ))}
              </div>
            </div>
          </details>
        )}
      </div>

      {/* Hint text */}
      {viewMode === 'graph' && (
        <p className="text-xs text-muted-foreground">
          {"Hover or tap any node to see the \"incoming gradient × local derivative = outgoing gradient\" breakdown."}
          {graphMode === 'fanout' && ' Notice how x receives gradients from both paths and they sum.'}
        </p>
      )}
    </div>
  )
}
