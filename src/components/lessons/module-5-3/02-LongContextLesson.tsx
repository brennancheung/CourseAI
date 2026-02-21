'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  ConstraintBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  SummaryBlock,
  GradientCard,
  ComparisonRow,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Long Context & Efficient Attention
 *
 * Second lesson in Module 5.3 (Scaling Architecture).
 * BUILD lesson—extends existing attention knowledge with three targeted modifications.
 *
 * Cognitive load: 3 new concepts:
 *   1. RoPE (rotary position embeddings)—encoding relative position via rotation
 *   2. Sparse attention patterns (sliding window, dilated)—subquadratic compute
 *   3. Grouped Query Attention (GQA)—KV cache compression
 *
 * Core concepts at DEVELOPED:
 * - RoPE mechanism (rotation in 2D subspaces, relative position in dot product)
 * - Quadratic attention bottleneck (concrete cost calculations)
 * - Sparse attention patterns (sliding window, dilated, stacked-layer recovery)
 * - GQA (sharing K/V across Q head groups, MHA-GQA-MQA spectrum)
 *
 * Concepts at INTRODUCED:
 * - Context extension via RoPE (principle of relative pattern transfer)
 * - Linear attention (kernel trick reformulation, O(n) compute concept)
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing RoPE, sparse attention, or GQA in production code
 * - NTK-aware scaling or YaRN extension formulas in detail
 * - ALiBi in depth (named as alternative only)
 * - Ring attention or sequence parallelism (deferred to Lesson 3)
 * - Flash attention implementation details (already INTRODUCED in 4.3.3)
 * - State space models (SSMs) or Mamba
 * - Multi-Head Latent Attention (MLA) in detail
 * - Benchmarking or specific model performance comparisons
 *
 * Previous: Mixture of Experts (Module 5.3, Lesson 1)
 * Next: Training & Serving at Scale (Module 5.3, Lesson 3)
 */

// ---------------------------------------------------------------------------
// Inline SVG: Three-Barrier Overview Diagram
// Shows the three bottlenecks side by side with solutions labeled.
// ---------------------------------------------------------------------------

function ThreeBarrierDiagram() {
  const svgW = 640
  const svgH = 340

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const posColor = '#f59e0b' // amber for position barrier
  const computeColor = '#f87171' // rose for compute barrier
  const memoryColor = '#60a5fa' // blue for memory barrier
  const solutionColor = '#34d399' // emerald for solutions
  const arrowColor = '#64748b'

  const barrierW = 170
  const barrierH = 150
  const gap = 20
  const startX = (svgW - 3 * barrierW - 2 * gap) / 2

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={svgW / 2}
          y={24}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Three Barriers to Long Context
        </text>
        <text
          x={svgW / 2}
          y={42}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Each barrier requires a different architectural solution
        </text>

        {/* Barrier 1: Position */}
        <rect
          x={startX}
          y={60}
          width={barrierW}
          height={barrierH}
          rx={8}
          fill={posColor}
          fillOpacity={0.08}
          stroke={posColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + barrierW / 2}
          y={82}
          textAnchor="middle"
          fill={posColor}
          fontSize="11"
          fontWeight="600"
        >
          Wall 1: Position
        </text>
        <text
          x={startX + barrierW / 2}
          y={100}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          Learned PE stops at
        </text>
        <text
          x={startX + barrierW / 2}
          y={114}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          max_seq_len
        </text>
        <text
          x={startX + barrierW / 2}
          y={132}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          Sinusoidal PE degrades
        </text>
        <text
          x={startX + barrierW / 2}
          y={146}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          on unseen lengths
        </text>
        {/* Cost label */}
        <text
          x={startX + barrierW / 2}
          y={170}
          textAnchor="middle"
          fill={posColor}
          fontSize="8"
          fontWeight="500"
        >
          Can&apos;t represent new positions
        </text>

        {/* Barrier 2: Compute */}
        <rect
          x={startX + barrierW + gap}
          y={60}
          width={barrierW}
          height={barrierH}
          rx={8}
          fill={computeColor}
          fillOpacity={0.08}
          stroke={computeColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={82}
          textAnchor="middle"
          fill={computeColor}
          fontSize="11"
          fontWeight="600"
        >
          Wall 2: Compute
        </text>
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={100}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          Attention FLOPs grow
        </text>
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={114}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          as O(n²)
        </text>
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={132}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          Flash attention fixes memory
        </text>
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={146}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          but not compute
        </text>
        {/* Cost label */}
        <text
          x={startX + barrierW + gap + barrierW / 2}
          y={170}
          textAnchor="middle"
          fill={computeColor}
          fontSize="8"
          fontWeight="500"
        >
          16,384× cost: 1K → 128K tokens
        </text>

        {/* Barrier 3: Memory */}
        <rect
          x={startX + 2 * (barrierW + gap)}
          y={60}
          width={barrierW}
          height={barrierH}
          rx={8}
          fill={memoryColor}
          fillOpacity={0.08}
          stroke={memoryColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={82}
          textAnchor="middle"
          fill={memoryColor}
          fontSize="11"
          fontWeight="600"
        >
          Wall 3: KV Memory
        </text>
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={100}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          KV cache grows linearly
        </text>
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={114}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          per head, per token
        </text>
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={132}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          64 heads × 128K tokens
        </text>
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={146}
          textAnchor="middle"
          fill={dimText}
          fontSize="8.5"
        >
          = hundreds of GB
        </text>
        {/* Cost label */}
        <text
          x={startX + 2 * (barrierW + gap) + barrierW / 2}
          y={170}
          textAnchor="middle"
          fill={memoryColor}
          fontSize="8"
          fontWeight="500"
        >
          KV cache larger than model
        </text>

        {/* Arrows down to solutions */}
        {[0, 1, 2].map((i) => {
          const cx = startX + i * (barrierW + gap) + barrierW / 2
          return (
            <g key={i}>
              <line
                x1={cx}
                y1={182}
                x2={cx}
                y2={210}
                stroke={arrowColor}
                strokeWidth={1.5}
              />
              <polygon
                points={`${cx},214 ${cx - 5},206 ${cx + 5},206`}
                fill={arrowColor}
              />
            </g>
          )
        })}

        {/* Solution boxes */}
        {[
          { label: 'RoPE', sub: 'Relative position in the dot product', x: startX },
          { label: 'Sparse Attention', sub: 'Compute only nearby scores', x: startX + barrierW + gap },
          { label: 'GQA', sub: 'Share K/V across Q head groups', x: startX + 2 * (barrierW + gap) },
        ].map((sol, i) => (
          <g key={i}>
            <rect
              x={sol.x}
              y={218}
              width={barrierW}
              height={50}
              rx={8}
              fill={solutionColor}
              fillOpacity={0.1}
              stroke={solutionColor}
              strokeWidth={1.5}
            />
            <text
              x={sol.x + barrierW / 2}
              y={238}
              textAnchor="middle"
              fill={solutionColor}
              fontSize="11"
              fontWeight="600"
            >
              {sol.label}
            </text>
            <text
              x={sol.x + barrierW / 2}
              y={256}
              textAnchor="middle"
              fill={dimText}
              fontSize="8"
            >
              {sol.sub}
            </text>
          </g>
        ))}

        {/* Bottom caption */}
        <text
          x={svgW / 2}
          y={300}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Three barriers. Three independent problems. Three targeted solutions.
        </text>
        <text
          x={svgW / 2}
          y={316}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          LLaMA and Mistral use all three together.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: RoPE Rotation Diagram
// Shows 2D vectors being rotated by position-proportional angles, with the
// relative distance producing the same angular difference regardless of
// absolute position.
// ---------------------------------------------------------------------------

function RoPERotationDiagram() {
  const svgW = 600
  const svgH = 320

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const qColor = '#a78bfa' // violet for query
  const kColor = '#f59e0b' // amber for key
  const angleColor = '#34d399' // emerald for angular difference
  const bgCircle = '#334155'

  // Helper: vector endpoint from angle and length
  const vecEnd = (cx: number, cy: number, angle: number, len: number) => ({
    x: cx + len * Math.cos(angle),
    y: cy - len * Math.sin(angle),
  })

  // Two examples side by side
  // Left: positions 3 and 7 (relative distance = 4)
  // Right: positions 100 and 104 (relative distance = 4)
  const theta = 0.15 // base rotation angle per position
  const vecLen = 60

  // Left circle
  const lcx = 155
  const lcy = 170
  const qAngleL = 3 * theta // position 3
  const kAngleL = 7 * theta // position 7
  const qEndL = vecEnd(lcx, lcy, qAngleL, vecLen)
  const kEndL = vecEnd(lcx, lcy, kAngleL, vecLen)

  // Right circle
  const rcx = 445
  const rcy = 170
  const qAngleR = 100 * theta // position 100
  const kAngleR = 104 * theta // position 104
  const qEndR = vecEnd(rcx, rcy, qAngleR, vecLen)
  const kEndR = vecEnd(rcx, rcy, kAngleR, vecLen)

  // Arc path helper for showing angle between two directions
  const arcPath = (cx: number, cy: number, startAngle: number, endAngle: number, r: number) => {
    const s = {
      x: cx + r * Math.cos(-startAngle),
      y: cy + r * Math.sin(-startAngle),
    }
    const e = {
      x: cx + r * Math.cos(-endAngle),
      y: cy + r * Math.sin(-endAngle),
    }
    const largeArc = Math.abs(endAngle - startAngle) > Math.PI ? 1 : 0
    const sweep = endAngle > startAngle ? 0 : 1
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} ${sweep} ${e.x} ${e.y}`
  }

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          RoPE: Same Relative Distance → Same Angular Difference
        </text>
        <text
          x={svgW / 2}
          y={38}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          The dot product depends on the angle between vectors, which depends on relative position
        </text>

        {/* Left example label */}
        <text
          x={lcx}
          y={68}
          textAnchor="middle"
          fill={brightText}
          fontSize="11"
          fontWeight="500"
        >
          Positions 3 and 7
        </text>
        <text
          x={lcx}
          y={82}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          relative distance = 4
        </text>

        {/* Left circle background */}
        <circle cx={lcx} cy={lcy} r={70} fill="none" stroke={bgCircle} strokeWidth={1} strokeDasharray="3,3" />

        {/* Left: q vector (position 3) */}
        <line x1={lcx} y1={lcy} x2={qEndL.x} y2={qEndL.y} stroke={qColor} strokeWidth={2} />
        <circle cx={qEndL.x} cy={qEndL.y} r={3} fill={qColor} />
        <text
          x={qEndL.x + 8}
          y={qEndL.y - 4}
          fill={qColor}
          fontSize="9"
          fontWeight="500"
        >
          q₃
        </text>

        {/* Left: k vector (position 7) */}
        <line x1={lcx} y1={lcy} x2={kEndL.x} y2={kEndL.y} stroke={kColor} strokeWidth={2} />
        <circle cx={kEndL.x} cy={kEndL.y} r={3} fill={kColor} />
        <text
          x={kEndL.x + 8}
          y={kEndL.y - 4}
          fill={kColor}
          fontSize="9"
          fontWeight="500"
        >
          k₇
        </text>

        {/* Left: angular difference arc */}
        <path
          d={arcPath(lcx, lcy, qAngleL, kAngleL, 40)}
          fill="none"
          stroke={angleColor}
          strokeWidth={1.5}
          strokeDasharray="4,2"
        />
        <text
          x={lcx + 48}
          y={lcy - 28}
          fill={angleColor}
          fontSize="8"
          fontWeight="500"
        >
          4θ
        </text>

        {/* Right example label */}
        <text
          x={rcx}
          y={68}
          textAnchor="middle"
          fill={brightText}
          fontSize="11"
          fontWeight="500"
        >
          Positions 100 and 104
        </text>
        <text
          x={rcx}
          y={82}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          relative distance = 4
        </text>

        {/* Right circle background */}
        <circle cx={rcx} cy={rcy} r={70} fill="none" stroke={bgCircle} strokeWidth={1} strokeDasharray="3,3" />

        {/* Right: q vector (position 100) */}
        <line x1={rcx} y1={rcy} x2={qEndR.x} y2={qEndR.y} stroke={qColor} strokeWidth={2} />
        <circle cx={qEndR.x} cy={qEndR.y} r={3} fill={qColor} />
        <text
          x={qEndR.x + 8}
          y={qEndR.y - 4}
          fill={qColor}
          fontSize="9"
          fontWeight="500"
        >
          q₁₀₀
        </text>

        {/* Right: k vector (position 104) */}
        <line x1={rcx} y1={rcy} x2={kEndR.x} y2={kEndR.y} stroke={kColor} strokeWidth={2} />
        <circle cx={kEndR.x} cy={kEndR.y} r={3} fill={kColor} />
        <text
          x={kEndR.x + 8}
          y={kEndR.y - 4}
          fill={kColor}
          fontSize="9"
          fontWeight="500"
        >
          k₁₀₄
        </text>

        {/* Right: angular difference arc */}
        <path
          d={arcPath(rcx, rcy, qAngleR, kAngleR, 40)}
          fill="none"
          stroke={angleColor}
          strokeWidth={1.5}
          strokeDasharray="4,2"
        />
        <text
          x={rcx + 48}
          y={rcy - 28}
          fill={angleColor}
          fontSize="8"
          fontWeight="500"
        >
          4θ
        </text>

        {/* Equals sign between */}
        <text
          x={svgW / 2}
          y={lcy + 5}
          textAnchor="middle"
          fill={angleColor}
          fontSize="18"
          fontWeight="700"
        >
          =
        </text>

        {/* Bottom explanation */}
        <rect
          x={60}
          y={260}
          width={svgW - 120}
          height={48}
          rx={6}
          fill={angleColor}
          fillOpacity={0.06}
          stroke={angleColor}
          strokeWidth={1}
          strokeOpacity={0.3}
        />
        <text
          x={svgW / 2}
          y={280}
          textAnchor="middle"
          fill={angleColor}
          fontSize="10"
          fontWeight="500"
        >
          Same angular difference (4θ) → same dot product contribution from position
        </text>
        <text
          x={svgW / 2}
          y={296}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          The dot product q^T k depends on relative distance (i − j), not absolute positions i and j
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Attention Pattern Comparison
// Shows full causal, sliding window, and dilated attention patterns side
// by side as heatmap-style grids.
// ---------------------------------------------------------------------------

function AttentionPatternDiagram() {
  const svgW = 620
  const svgH = 340

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const fullColor = '#f87171' // rose for full (expensive)
  const windowColor = '#34d399' // emerald for window (efficient)
  const dilatedColor = '#60a5fa' // blue for dilated

  const gridSize = 10
  const cellSize = 12
  const patternY = 85

  // Grid helper: render a grid of cells with a function determining fill
  const renderGrid = (
    originX: number,
    originY: number,
    fillFn: (row: number, col: number) => string | null,
    n: number,
  ) => {
    const cells = []
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const fill = fillFn(r, c)
        if (fill) {
          cells.push(
            <rect
              key={`${r}-${c}`}
              x={originX + c * cellSize}
              y={originY + r * cellSize}
              width={cellSize - 1}
              height={cellSize - 1}
              rx={1}
              fill={fill}
              fillOpacity={0.6}
            />,
          )
        }
      }
    }
    return cells
  }

  // Full causal: lower triangular
  const fullOriginX = 40
  const fullFill = (r: number, c: number) => (c <= r ? fullColor : null)

  // Sliding window: band of width 3 within causal
  const windowW = 3
  const windowOriginX = 230
  const windowFill = (r: number, c: number) => {
    if (c > r) return null // causal constraint
    if (r - c < windowW) return windowColor
    return null
  }

  // Dilated: every 2nd position within causal
  const dilatedOriginX = 420
  const dilatedFill = (r: number, c: number) => {
    if (c > r) return null // causal constraint
    // Attend to: self, every 2nd previous token
    if (c === r) return dilatedColor
    if ((r - c) % 2 === 0 && r - c <= 8) return dilatedColor
    return null
  }

  // Count filled cells
  const countFilled = (fillFn: (r: number, c: number) => string | null, n: number) => {
    let count = 0
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (fillFn(r, c)) count++
      }
    }
    return count
  }

  const fullCount = countFilled(fullFill, gridSize)
  const windowCount = countFilled(windowFill, gridSize)
  const dilatedCount = countFilled(dilatedFill, gridSize)

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Attention Patterns: Full vs Sparse
        </text>
        <text
          x={svgW / 2}
          y={38}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Each colored cell = one computed attention score. Fewer cells = less compute.
        </text>

        {/* Full Causal */}
        <text
          x={fullOriginX + (gridSize * cellSize) / 2}
          y={68}
          textAnchor="middle"
          fill={fullColor}
          fontSize="11"
          fontWeight="600"
        >
          Full Causal
        </text>
        {renderGrid(fullOriginX, patternY, fullFill, gridSize)}
        <text
          x={fullOriginX + (gridSize * cellSize) / 2}
          y={patternY + gridSize * cellSize + 18}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          {fullCount} scores—O(n²)
        </text>

        {/* Sliding Window */}
        <text
          x={windowOriginX + (gridSize * cellSize) / 2}
          y={68}
          textAnchor="middle"
          fill={windowColor}
          fontSize="11"
          fontWeight="600"
        >
          Sliding Window
        </text>
        {renderGrid(windowOriginX, patternY, windowFill, gridSize)}
        <text
          x={windowOriginX + (gridSize * cellSize) / 2}
          y={patternY + gridSize * cellSize + 18}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          {windowCount} scores—O(n·w)
        </text>

        {/* Dilated */}
        <text
          x={dilatedOriginX + (gridSize * cellSize) / 2}
          y={68}
          textAnchor="middle"
          fill={dilatedColor}
          fontSize="11"
          fontWeight="600"
        >
          Dilated
        </text>
        {renderGrid(dilatedOriginX, patternY, dilatedFill, gridSize)}
        <text
          x={dilatedOriginX + (gridSize * cellSize) / 2}
          y={patternY + gridSize * cellSize + 18}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          {dilatedCount} scores—O(n·w)
        </text>

        {/* Axis labels */}
        <text
          x={fullOriginX - 8}
          y={patternY + (gridSize * cellSize) / 2 + 4}
          textAnchor="end"
          fill={dimText}
          fontSize="8"
          transform={`rotate(-90, ${fullOriginX - 8}, ${patternY + (gridSize * cellSize) / 2})`}
        >
          query position →
        </text>
        <text
          x={fullOriginX + (gridSize * cellSize) / 2}
          y={patternY + gridSize * cellSize + 36}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          key position →
        </text>

        {/* Summary box */}
        <rect
          x={60}
          y={280}
          width={svgW - 120}
          height={45}
          rx={6}
          fill={dimText}
          fillOpacity={0.06}
          stroke={dimText}
          strokeWidth={1}
          strokeOpacity={0.2}
        />
        <text
          x={svgW / 2}
          y={300}
          textAnchor="middle"
          fill={brightText}
          fontSize="9.5"
          fontWeight="500"
        >
          At 128K tokens with window w=4096: full causal computes ~8.2 billion scores.
        </text>
        <text
          x={svgW / 2}
          y={316}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Sliding window computes ~524 million—a 16× reduction. Dilated captures longer range with similar savings.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function LongContextLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Long Context & Efficient Attention"
            description="The attention mechanism you built across Module 4.2 has a property you noticed early: every token computes a score with every other token. At 1,024 tokens, this is manageable. At 128K tokens, it is 16,384&times; more expensive&mdash;and the positional encoding cannot even represent positions that far. Three targeted architectural innovations address three distinct bottlenecks: RoPE for position, sparse attention for compute, and GQA for memory."
            category="Scaling Architecture"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Objective + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            By the end of this lesson, you will be able to explain how RoPE
            encodes relative position into the Q/K dot product, why sparse
            attention patterns reduce O(n&sup2;) compute to O(n&middot;w), and
            how GQA compresses the KV cache by sharing K/V across groups of Q
            heads. Three barriers, three targeted solutions.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Three barriers to long context: position encoding limits, quadratic compute, KV cache memory',
              'RoPE: encoding relative position in the Q/K dot product via rotation in 2D subspaces',
              'Sparse attention: sliding window and dilated patterns for subquadratic compute',
              'Linear attention: brief conceptual introduction (kernel trick, O(n) compute)',
              'GQA: sharing K/V heads across Q head groups, the MHA-GQA-MQA spectrum',
              'NOT: implementing RoPE, sparse attention, or GQA in production code (notebook demos concepts)',
              'NOT: NTK-aware scaling or YaRN formulas in detail (mentioned only)',
              'NOT: ring attention or sequence parallelism (Lesson 3)',
              'NOT: flash attention implementation details (already introduced in Scaling & Efficiency)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            After the STRETCH of Mixture of Experts, this lesson extends your
            existing attention knowledge with targeted modifications. No new
            paradigm&mdash;every concept here modifies or optimizes the
            attention mechanism you already know deeply.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap: Three Facts from Prior Lessons
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where You Left Off"
            subtitle="Three facts about to combine into a problem"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              From the attention lessons in Module 4.2 and the efficiency
              lesson in Module 4.3, you established three facts:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>The attention formula:</strong>{' '}
                <InlineMath math="\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" />{' '}
                &mdash; &ldquo;every token computes a score with every other
                token.&rdquo; No locality window, no distance preference. All
                pairs.
              </li>
              <li>
                <strong>Positional encoding:</strong> Sinusoidal PE adds
                position to embeddings and can generate any position&mdash;but
                was never tested beyond training lengths. Learned PE cannot
                generalize at all: there is no embedding for position 2049 in
                a model trained with max_seq_len=2048. Remember the DNA
                transfer question? Learned PE has no answer for unseen
                positions.
              </li>
              <li>
                <strong>Flash attention and KV caching:</strong> Flash
                attention reduces attention memory from O(n&sup2;) to O(n),
                but compute stays O(n&sup2;)&mdash;it computes the same dot
                products, just without materializing the full matrix. KV
                caching eliminates redundant computation during generation,
                but costs memory per head per token.
              </li>
            </ol>

            <p className="text-muted-foreground">
              Flash attention and KV caching are optimizations within the
              existing framework. They do not change the algorithmic cost.
              What happens when we push the attention mechanism to 128K
              tokens? Three things break.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Necessary but Insufficient">
            Flash attention and KV caching are essential. But flash attention
            fixes memory, not compute. KV caching fixes generation, not
            prefill. Neither addresses the position encoding limit. All three
            barriers remain.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: The Three Walls
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Three Walls"
            subtitle="What breaks when attention meets long context"
          />
          <div className="space-y-4">
            <ThreeBarrierDiagram />

            <GradientCard title="Wall 1: Position" color="amber">
              <div className="space-y-2 text-sm">
                <p>
                  A model trained with learned PE at max_seq_len=4096. Token
                  at position 4097 has <strong>no positional
                  encoding</strong>&mdash;the embedding table simply has no
                  row for it. The model cannot represent this position. Even
                  sinusoidal PE, which can generate any position, was never
                  trained on the attention patterns that emerge at 100K. The
                  Q/K projections learned to handle distances up to 4096, not
                  50,000.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Wall 2: Compute" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Attention FLOPs scale as ~2 &middot; n&sup2; &middot;
                  d_model per layer:
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-muted">
                        <th className="text-left py-1 px-2 font-medium">Sequence Length</th>
                        <th className="text-right py-1 px-2 font-medium">Attention FLOPs/Layer</th>
                        <th className="text-right py-1 px-2 font-medium">Factor vs 1K</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-muted/50">
                        <td className="py-1 px-2">1K tokens</td>
                        <td className="text-right py-1 px-2">~1.5 billion</td>
                        <td className="text-right py-1 px-2">1&times;</td>
                      </tr>
                      <tr className="border-b border-muted/50">
                        <td className="py-1 px-2">4K tokens</td>
                        <td className="text-right py-1 px-2">~24 billion</td>
                        <td className="text-right py-1 px-2">16&times;</td>
                      </tr>
                      <tr className="border-b border-muted/50">
                        <td className="py-1 px-2">32K tokens</td>
                        <td className="text-right py-1 px-2">~1.5 trillion</td>
                        <td className="text-right py-1 px-2">1,024&times;</td>
                      </tr>
                      <tr>
                        <td className="py-1 px-2">128K tokens</td>
                        <td className="text-right py-1 px-2">~24 trillion</td>
                        <td className="text-right py-1 px-2">16,384&times;</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="text-muted-foreground">
                  Flash attention makes this fit in memory. It does not make
                  this computation go away.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Wall 3: KV Cache Memory" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  KV cache for a 70B model with 64 heads,{' '}
                  <InlineMath math="d_k = 128" />, at 128K context:
                </p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>
                    Per-head KV cache: 2 &times; 128K &times; 128 &times; 2
                    bytes (bf16) &asymp; <strong>65 MB</strong>
                  </li>
                  <li>
                    Total: 64 heads &times; 80 layers &times; 65 MB &asymp;{' '}
                    <strong>~332 GB</strong>
                  </li>
                  <li>
                    The model weights themselves: ~140 GB
                  </li>
                </ul>
                <p className="text-muted-foreground">
                  The KV cache is now <strong>larger than the model
                  itself</strong>.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="Three Walls, Three Solutions" color="emerald">
              <p className="text-sm">
                Position encoding that generalizes beyond training length.
                Attention that scales subquadratically. KV cache that
                compresses. Three independent problems, three targeted
                architectural innovations. The rest of this lesson addresses
                each one.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Common Assumption">
            &ldquo;Longer context is just a matter of training with longer
            sequences.&rdquo; This ignores all three barriers: learned PE
            cannot represent unseen positions, quadratic compute makes naive
            long-context prohibitively expensive, and KV cache memory
            explodes. Context extension requires architectural changes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1: RoPE (Rotary Position Embeddings)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="RoPE: Rotary Position Embeddings"
            subtitle="Position encoded in the handshake, not the nametag"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The positional encoding methods you have seen&mdash;sinusoidal
              and learned&mdash;both add position information to the token
              embedding <em>before</em> Q/K projection. Position gets mixed
              with semantic content during the linear transformations{' '}
              <InlineMath math="W_Q" /> and <InlineMath math="W_K" />.
            </p>

            <p className="text-muted-foreground">
              What if position could be injected <strong>directly into the
              Q/K dot product</strong>? Instead of modifying the input to
              attention, modify the attention score itself.
            </p>

            <p className="text-muted-foreground">
              The key insight: <strong>rotate Q and K vectors by an angle
              proportional to their position</strong>. When computing{' '}
              <InlineMath math="q_i^T k_j" /> (the attention score between
              positions i and j), the rotation causes the result to depend on
              the <em>relative</em> position (i &minus; j), not the absolute
              positions.
            </p>

            {/* Analogy */}
            <GradientCard title="The Nametag vs Handshake Analogy" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Sinusoidal and learned PE put position on the{' '}
                  <strong>nametag</strong>&mdash;the embedding that the token
                  carries everywhere. The position information gets baked into
                  the representation before attention even happens.
                </p>
                <p>
                  RoPE encodes position in the <strong>handshake</strong>
                  &mdash;the Q/K dot product, which is the interaction between
                  two tokens. Two people 3 seats apart feel the same handshake
                  regardless of which seats they are in. The handshake depends
                  on <em>relative distance</em>, not absolute position.
                </p>
              </div>
            </GradientCard>

            <RoPERotationDiagram />

            <p className="text-muted-foreground">
              <strong>How it works:</strong> pair consecutive dimensions of Q
              and K vectors into 2D subspaces. Apply a rotation to each pair,
              where the rotation angle depends on the token&rsquo;s position
              and a frequency that varies by dimension pair. Here is what that
              looks like in code:
            </p>

            <CodeBlock
              code={`# RoPE: rotate Q and K vectors in 2D subspaces
# Each consecutive pair of dimensions is one 2D subspace

def apply_rope(x, positions, theta_base=10000.0):
    """Apply rotary position embeddings to Q or K vectors.
    x: (seq_len, d_model)—Q or K vectors after projection
    positions: (seq_len,)—position indices [0, 1, 2, ...]
    """
    d = x.shape[-1]
    # Frequencies: high for early dims, low for later dims
    # Same multi-frequency idea as sinusoidal PE
    freqs = 1.0 / (theta_base ** (torch.arange(0, d, 2) / d))

    # Angles: position × frequency for each 2D pair
    angles = positions[:, None] * freqs[None, :]  # (seq_len, d/2)

    # Split into pairs and rotate
    x_pairs = x.view(-1, d // 2, 2)          # (seq_len, d/2, 2)
    cos_a = torch.cos(angles).unsqueeze(-1)   # rotation components
    sin_a = torch.sin(angles).unsqueeze(-1)

    # 2D rotation: [cos, -sin; sin, cos] applied to each pair
    x_rot = torch.cat([
        x_pairs[..., 0:1] * cos_a - x_pairs[..., 1:2] * sin_a,
        x_pairs[..., 0:1] * sin_a + x_pairs[..., 1:2] * cos_a,
    ], dim=-1)

    return x_rot.view_as(x)

# The key property: dot(apply_rope(q, pos_i), apply_rope(k, pos_j))
# depends on (i - j), not on i and j individually`}
              language="python"
              filename="rope.py"
            />

            <p className="text-muted-foreground">
              The rotation matrix that each dimension pair applies is:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath
                math={`R(m, \\theta_d) = \\begin{bmatrix} \\cos(m \\cdot \\theta_d) & -\\sin(m \\cdot \\theta_d) \\\\ \\sin(m \\cdot \\theta_d) & \\cos(m \\cdot \\theta_d) \\end{bmatrix}`}
              />
            </div>

            <p className="text-muted-foreground">
              Where <InlineMath math="m" /> is the position index and{' '}
              <InlineMath math="\theta_d" /> varies by dimension pair (high
              frequency for early dimensions, low frequency for later ones).
              Connect this to the &ldquo;clock with many hands&rdquo; from
              Embeddings &amp; Position: same multi-frequency structure as
              sinusoidal PE. Different dimension pairs capture position at
              different scales&mdash;some rotate fast (local position), others
              rotate slow (global position). But instead of adding to the
              embedding, the rotation is applied to Q and K after projection.
            </p>

            <p className="text-muted-foreground">
              The mathematical property that makes this work: when you compute
              the dot product of two rotated vectors, the rotation from
              position <InlineMath math="i" /> and the rotation from position{' '}
              <InlineMath math="j" /> combine into a single rotation by{' '}
              <InlineMath math="(i - j) \cdot \theta" />. The dot product
              depends on the <em>difference</em> in positions:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath
                math={`q_{\\text{rot}}^T \\, k_{\\text{rot}} = q^T \\, R\\bigl((i - j) \\cdot \\theta\\bigr) \\, k`}
              />
            </div>

            {/* Misconception: RoPE is not just another PE */}
            <GradientCard title="Misconception: &ldquo;RoPE is just another positional encoding scheme&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  RoPE is not interchangeable with sinusoidal or learned PE.
                  Sinusoidal and learned PE modify the <strong>input</strong>{' '}
                  to attention&mdash;they add position to the embedding before
                  Q/K projection. RoPE modifies the <strong>dot product
                  itself</strong>&mdash;it rotates Q and K after projection.
                </p>
                <p className="text-muted-foreground">
                  This is the difference between encoding position in the
                  nametag (additive PE) and encoding it in the handshake
                  (RoPE). The handshake between tokens at positions 1000 and
                  1005 uses the same rotation as between positions 50000 and
                  50005. Additive PE cannot do this because position
                  information is mixed into the embedding before projection.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              <strong>Why RoPE enables context extension:</strong> The rotation
              for relative distance <InlineMath math="d" /> is always{' '}
              <InlineMath math="R(d \cdot \theta)" />. A model trained at 4K
              context learns what the rotation for distance 5, distance 100,
              distance 2000 looks like. At inference with 32K context,
              distances 5, 100, and 2000 still use the same rotations.
              Distances beyond 4K use rotations the model has not seen, but
              interpolation and extension techniques (NTK-aware scaling, YaRN)
              adjust the frequency base to smooth the transition. The key
              point: RoPE gives something to extend. Learned PE gives nothing.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Additive vs Multiplicative">
            Sinusoidal/learned PE: position is <strong>added</strong> to the
            embedding before attention. RoPE: position is{' '}
            <strong>multiplied</strong> (via rotation) into Q and K inside
            attention. This changes where position information enters the
            computation&mdash;and what it can express.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: Predict-and-Verify (RoPE vs Learned PE)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Two Models, One Long Document" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                Two models are trained on 4K-token sequences. Model A uses
                learned PE. Model B uses RoPE. Both are asked to process an
                8K-token document.
              </p>

              <p><strong>Predict before revealing:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>What happens to Model A at position 4097?</li>
                <li>What happens to Model B at position 4097?</li>
                <li>Which model has a better chance of handling the 8K document? Why?</li>
                <li>Does Model B work perfectly at 8K with no further modification?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Make your predictions, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                    <li>
                      <strong>Model A (learned PE): fails.</strong> The
                      embedding lookup table has no row for position 4097. The
                      model cannot represent this position at all&mdash;it is
                      outside the vocabulary.
                    </li>
                    <li>
                      <strong>Model B (RoPE): computes normally.</strong> The
                      rotation angle <InlineMath math="\theta \times 4097" />{' '}
                      is perfectly computable. The model has never seen
                      absolute position 4097, but the relative position
                      patterns for nearby tokens (distance 1, 5, 100) are
                      familiar from training.
                    </li>
                    <li>
                      <strong>Model B, because RoPE provides a
                      foundation.</strong> Relative position patterns learned
                      at 4K transfer to longer sequences. The rotation for
                      &ldquo;3 tokens apart&rdquo; is the same at position 100
                      as at position 7000. Model A has no foundation to
                      build on.
                    </li>
                    <li>
                      <strong>Not necessarily.</strong> RoPE makes context
                      extension <em>possible</em>, not automatic. Relative
                      distances beyond 4K use rotations the model has not
                      trained on. Performance may degrade for very long-range
                      attention patterns. Extension techniques (NTK-aware
                      scaling, YaRN) adjust the frequency base to improve
                      extrapolation&mdash;but the critical point is that RoPE
                      gives these techniques something to work with.
                    </li>
                  </ol>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain Part 2: The Quadratic Bottleneck and Sparse Attention
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sparse Attention"
            subtitle="Compute where attention concentrates, not everywhere"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Flash attention and KV caching are essential optimizations. But
              they do not change the fundamental algorithm. Flash attention
              computes the same O(n&sup2;) dot products&mdash;it just does so
              without materializing the full attention matrix. To reduce the
              actual computation, we need to <strong>compute fewer dot
              products</strong>.
            </p>

            <p className="text-muted-foreground">
              From Mixture of Experts: &ldquo;Not every FFN parameter needs
              to activate for every token.&rdquo; Sparse attention applies the
              same principle: <strong>not every token pair needs to compute an
              attention score</strong>.
            </p>

            <AttentionPatternDiagram />

            <GradientCard title="Sliding Window Attention" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  Each token attends to at most <InlineMath math="w" />{' '}
                  preceding tokens (plus itself). Cost:{' '}
                  <InlineMath math="O(n \cdot w)" /> instead of{' '}
                  <InlineMath math="O(n^2)" />. For{' '}
                  <InlineMath math="w = 4096" /> and{' '}
                  <InlineMath math="n = 128K" />, this is <strong>32&times;
                  cheaper</strong>. Used by Mistral-7B.
                </p>
                <p className="text-muted-foreground">
                  This is the same masking mechanism you learned for causal
                  attention&mdash;set entries to{' '}
                  <InlineMath math="-\infty" /> before softmax. The difference
                  is the mask pattern: causal masking uses a lower-triangular
                  pattern (attend to all previous tokens). Sliding window uses
                  a diagonal band (attend to only the nearest{' '}
                  <InlineMath math="w" /> previous tokens).
                </p>
              </div>
            </GradientCard>

            {/* "Of course" beat */}
            <GradientCard title="Of Course" color="violet">
              <p className="text-sm italic">
                You already knew from the attention lessons that trained
                attention heads develop messy specialization patterns. Some
                track the previous token. Some attend to nearby syntactic
                structure. Very few heads attend strongly to tokens thousands
                of positions away. If most attention weight is local, of
                course you should skip the distant computations that receive
                near-zero weight. Of course the model should focus its compute
                budget where attention actually concentrates.
              </p>
            </GradientCard>

            {/* Misconception: sparse attention loses information */}
            <GradientCard title="Misconception: &ldquo;Sparse attention loses too much information&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  You spent six lessons building up full attention as the core
                  mechanism. You might resist removing token pairs from the
                  attention computation. But consider what happens with
                  <strong> stacked layers</strong>.
                </p>
                <p className="text-muted-foreground">
                  A token at position 50,000 cannot directly attend to
                  position 1 through a sliding window of 4,096. But
                  information flows through intermediate tokens across layers.
                  Layer 1 propagates information from position 1 to ~4,096
                  (via the residual stream). Layer 2 propagates from ~4,096 to
                  ~8,192. By layer 13 (13 &times; 4,096 = ~53K), information
                  from position 1 has reached position 50,000.
                </p>
                <p className="text-muted-foreground">
                  Stacked local attention approximates global attention. The
                  same residual stream that carried information through your
                  GPT-2 carries it across layers of sparse attention.
                  Mistral-7B proves this works: sliding window attention
                  achieves performance competitive with full attention models.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              <strong>Dilated attention</strong> captures longer-range patterns
              without full compute. Instead of attending to the{' '}
              <InlineMath math="w" /> nearest consecutive tokens, attend to
              every 2nd (or 4th) token within a wider range. This gives the
              same compute budget as a window of size{' '}
              <InlineMath math="w" /> but covers a{' '}
              <InlineMath math="2w" /> (or <InlineMath math="4w" />) span.
              The attention pattern diagram above shows the effect: fewer
              cells than full causal, but spread over a wider region than the
              sliding window. Often combined with sliding window in different
              heads or layers: some heads handle local patterns, others handle
              longer-range structure.
            </p>

            <p className="text-muted-foreground">
              <strong>Linear attention</strong> takes a different approach
              entirely. Standard attention works in two steps: first compute
              all pairwise scores between tokens (the{' '}
              <InlineMath math="n \times n" /> matrix), then use those scores
              to take a weighted average of V. The bottleneck is that first
              step&mdash;forming the giant score matrix. What if you could
              skip it? Instead of asking &ldquo;how relevant is every token
              to every other token?&rdquo; and then averaging V, compute a{' '}
              <strong>summary of K and V first</strong>, then let each Q
              token query that summary directly.
            </p>

            <p className="text-muted-foreground">
              The trick: replace softmax with a feature map{' '}
              <InlineMath math="\phi" /> and change the order of operations.
              Instead of computing pairwise scores first (left to right:{' '}
              <InlineMath math="\phi(Q) \cdot \phi(K)^T \cdot V" />, which
              still forms the n&times;n matrix), compute the K/V summary
              first (right to left:{' '}
              <InlineMath math="\phi(Q) \cdot (\phi(K)^T V)" />). The
              summary <InlineMath math="\phi(K)^T V" /> is a small{' '}
              <InlineMath math="d_k \times d_v" /> matrix&mdash;independent of
              sequence length. Each query token multiplies against this
              fixed-size summary instead of against every other token. Cost
              drops from <InlineMath math="O(n^2 \cdot d_k)" /> to{' '}
              <InlineMath math="O(n \cdot d_k^2)" />. The tradeoff: the
              feature map <InlineMath math="\phi" /> approximates the softmax
              kernel, introducing some approximation error. Full softmax
              attention remains more expressive for tasks requiring sharp,
              position-dependent patterns.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Conditional Computation Again">
            MoE: &ldquo;Activate only the relevant experts.&rdquo; Sparse
            attention: &ldquo;Compute scores only for relevant token pairs.&rdquo;
            The same principle&mdash;not all computation is necessary&mdash;
            applied to a different bottleneck.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain Part 3: Grouped Query Attention (GQA)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Grouped Query Attention (GQA)"
            subtitle="Cache what's needed, not everything"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Sparse attention addresses the compute wall. What about the
              memory wall&mdash;the KV cache?
            </p>

            <p className="text-muted-foreground">
              Recall multi-head attention from Module 4.2: each head has its
              own <InlineMath math="W_Q^i" />,{' '}
              <InlineMath math="W_K^i" />,{' '}
              <InlineMath math="W_V^i" />. During generation, each head
              stores its own K and V cache. At 64 heads, 128K context,{' '}
              <InlineMath math="d_k = 128" />&mdash;the KV cache consumes
              hundreds of gigabytes.
            </p>

            <p className="text-muted-foreground">
              The GQA insight: <strong>multiple Q heads can share a single K/V
              pair</strong>. Queries still need diversity&mdash;different{' '}
              <InlineMath math="W_Q^i" /> per head captures different
              relevance patterns. But keys and values show significant
              redundancy across heads. Share K/V across groups of Q heads.
            </p>

            {/* Misconception: GQA is not just fewer heads */}
            <GradientCard title="Misconception: &ldquo;GQA is just using fewer attention heads&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Reducing heads means reducing Q heads, increasing{' '}
                  <InlineMath math="d_k" />, and changing the
                  quality-diversity tradeoff. GQA is different: it reduces K/V
                  heads while <strong>keeping all Q heads</strong>. The query
                  diversity is preserved. Only the KV cache memory is reduced.
                </p>
                <p className="text-muted-foreground">
                  With 32 Q heads and GQA groups of 8: there are 32 Q heads
                  (each with its own <InlineMath math="W_Q^i" />) but only 4
                  K/V heads (each shared across 8 Q heads). If you simply
                  reduced to 4 heads, you would have 4 Q heads&mdash;far fewer
                  queries, less diversity, different quality tradeoffs.
                </p>
              </div>
            </GradientCard>

            {/* MHA vs GQA vs MQA comparison */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Architecture</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">Q Heads</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">KV Heads</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">KV Cache</th>
                    <th className="text-right py-2 px-3 text-muted-foreground font-medium">Query Diversity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">MHA (Multi-Head)</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">32</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">32</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">Baseline</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">Full</td>
                  </tr>
                  <tr className="border-b border-muted/50 bg-emerald-500/5">
                    <td className="py-2 px-3 font-medium text-emerald-400">GQA (Grouped)</td>
                    <td className="text-right py-2 px-3 text-emerald-400">32</td>
                    <td className="text-right py-2 px-3 text-emerald-400">8</td>
                    <td className="text-right py-2 px-3 text-emerald-400">4&times; smaller</td>
                    <td className="text-right py-2 px-3 text-emerald-400">Full</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">MQA (Multi-Query)</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">32</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">1</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">32&times; smaller</td>
                    <td className="text-right py-2 px-3 text-muted-foreground">Full</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-amber-400">Fewer heads (4)</td>
                    <td className="text-right py-2 px-3 text-amber-400">4</td>
                    <td className="text-right py-2 px-3 text-amber-400">4</td>
                    <td className="text-right py-2 px-3 text-amber-400">8&times; smaller</td>
                    <td className="text-right py-2 px-3 text-amber-400">Reduced</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              <strong>Concrete numbers for LLaMA 2 70B</strong> (GQA with 8 KV
              groups, 64 Q heads): at 128K context with{' '}
              <InlineMath math="d_k = 128" />, compare the KV cache:
            </p>

            <ComparisonRow
              left={{
                title: 'MHA (64 KV heads)',
                color: 'amber',
                items: [
                  'Per-head: 2 × 128K × 128 × 2B ≈ 65 MB',
                  'Total: 64 heads × 80 layers × 65 MB',
                  '≈ 332 GB of KV cache',
                  'More than the model weights',
                ],
              }}
              right={{
                title: 'GQA (8 KV groups)',
                color: 'emerald',
                items: [
                  'Per-group: same 65 MB per KV head',
                  'Total: 8 groups × 80 layers × 65 MB',
                  '≈ 42 GB of KV cache',
                  '8× reduction—fits on fewer GPUs',
                ],
              }}
            />

            <p className="text-muted-foreground">
              In Module 4.2, you learned that each head has completely
              independent weights: &ldquo;Split, not multiplied.&rdquo; GQA
              takes this further: keep the query diversity (each of 64 Q heads
              has its own <InlineMath math="W_Q^i" />) but acknowledge that
              K/V across heads are often similar enough to share. The
              &ldquo;split, not multiplied&rdquo; principle still applies to
              queries. For keys and values, the practical reality is that full
              independence is not worth the memory cost at scale.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The MHA → GQA → MQA Spectrum">
            All three architectures keep the same number of Q heads with the
            same per-head dimension. The only change is how many independent
            K/V heads exist. MHA: full independence. GQA: grouped sharing.
            MQA: all queries share one K/V. A smooth tradeoff between KV cache
            memory and K/V diversity.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2: Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Design the Architecture Changes" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A team is building a document analysis system that processes
                64K-token legal documents. They have a 7B model with:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Learned positional encoding (max 4096)</li>
                <li>Full multi-head attention (32 heads)</li>
                <li>No special efficiency techniques beyond flash attention</li>
              </ul>

              <p>
                They want to handle 64K tokens. <strong>What needs to
                change?</strong>
              </p>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reason through the three barriers, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <p className="text-muted-foreground">
                    <strong>Wall 1—Position:</strong> Replace learned PE with
                    RoPE. The model cannot represent any position beyond 4096
                    with learned PE&mdash;it literally has no embedding for
                    position 4097. RoPE encodes relative position in the Q/K
                    dot product, making context extension possible. Apply an
                    extension technique (NTK-aware scaling) to smooth the
                    transition from 4K to 64K.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Wall 2—Compute:</strong> Add sliding window
                    attention. At 64K, quadratic attention is 256&times; more
                    expensive than at 4K. Flash attention handles memory but
                    not this compute increase. Sliding window with{' '}
                    <InlineMath math="w = 4096" /> makes per-token compute
                    independent of total context length. Information flows
                    across the full document through stacked layers.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Wall 3—Memory:</strong> Add GQA. At 64K with 32
                    heads, the KV cache is 16&times; larger than at 4K. Group
                    the 32 Q heads into 4&ndash;8 KV groups, reducing the KV
                    cache proportionally while preserving query diversity.
                  </p>
                  <p className="text-muted-foreground">
                    <strong>Key insight:</strong> These three modifications
                    address three <em>independent</em> bottlenecks. They
                    combine, not replace each other. This is exactly what
                    Mistral does: RoPE + sliding window + GQA.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Practice: Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Explore long-context techniques hands-on"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises that explore the three long-context techniques
              with toy models. Each builds on the previous conceptually but
              can be done independently.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: RoPE Rotation on 2D Vectors (Guided)" color="blue">
                <p className="text-sm">
                  Implement the 2D rotation matrix. Rotate pairs of vectors at
                  different absolute positions but the same relative distance.
                  Compute dot products and verify the relative position
                  property: the dot product depends only on (i &minus; j).
                  Predict before running: &ldquo;Will the dot product change
                  if we shift both positions by 1000?&rdquo;
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Sparse Attention Mask Patterns (Supported)" color="blue">
                <p className="text-sm">
                  Implement full causal, sliding window, and dilated attention
                  masks for a 64-token sequence. Visualize all three as
                  heatmaps. Compare the number of computed entries for each.
                  Extend to 256 tokens and plot compute scaling.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: GQA Forward Pass (Supported)" color="blue">
                <p className="text-sm">
                  Implement a GQA layer with h_Q=8, h_KV=2 (4 Q heads per KV
                  group). Run a forward pass on a batch of 16 tokens with
                  d_model=64. Compare output shape to standard MHA. Count KV
                  cache parameters: MHA (8 KV heads) vs GQA (2 KV groups).
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Attention Cost Calculator (Independent)" color="blue">
                <p className="text-sm">
                  Build a function that computes total attention FLOPs and KV
                  cache memory for a given model configuration. Compute costs
                  for GPT-2 at 1K, LLaMA 2 70B with MHA at 4K, with GQA at
                  4K, with GQA at 128K, and with sliding window at 128K.
                  Present results in a table showing how the three
                  optimizations compound.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                  Progression: RoPE mechanics &rarr; sparse attention patterns
                  &rarr; GQA architecture &rarr; cost analysis integrating all
                  three.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-3-2-long-context-and-efficient-attention.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M6 3H3a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3M9 2h5v5M8 8l5-5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            Exercise 1 demystifies RoPE rotation. Exercise 2 visualizes sparse
            patterns. Exercise 3 builds a GQA layer. Exercise 4 integrates
            all three into a single cost analysis. Each exercise targets a
            different barrier.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Three barriers prevent standard attention from scaling to long contexts.',
                description:
                  'Position encoding limits (learned PE stops at max_seq_len), quadratic compute cost (O(n²) FLOPs), and KV cache memory growth (linear per head, multiplied by num_heads). Flash attention and KV caching are necessary but insufficient\u2014they do not address all three.',
              },
              {
                headline:
                  'RoPE encodes position in the Q/K dot product via rotation, making attention scores depend on relative position.',
                description:
                  'Rotate Q and K in 2D subspaces by position-proportional angles. The dot product depends on the angular difference\u2014the relative distance (i \u2212 j), not the absolute positions. This enables context extension: patterns learned at training length transfer to longer sequences.',
              },
              {
                headline:
                  'Sparse attention computes scores for only a subset of token pairs, reducing O(n²) to O(n\u00b7w).',
                description:
                  'Sliding window attention attends to the nearest w tokens. Information propagates across the full context through stacked layers and the residual stream. Dilated attention captures longer-range patterns. Both achieve subquadratic compute.',
              },
              {
                headline:
                  'GQA shares K/V heads across groups of Q heads, reducing KV cache memory while preserving query diversity.',
                description:
                  'The MHA\u2192GQA\u2192MQA spectrum trades KV cache size for K/V independence. LLaMA 2 70B uses GQA with 8 KV groups (8\u00d7 cache reduction) while keeping all 64 Q heads for query diversity.',
              },
              {
                headline:
                  'These three innovations address three independent bottlenecks and combine in practice.',
                description:
                  'LLaMA uses RoPE + GQA. Mistral uses RoPE + GQA + sliding window. Each targets a different wall. Together, they enable 100K+ token context windows.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <GradientCard title="Mental Model" color="violet">
            <p className="text-sm italic">
              Position in the handshake, not the nametag. Compute where
              attention concentrates, not everywhere. Cache what&rsquo;s
              needed, not everything. Three barriers, three targeted solutions.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          12. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
                authors: 'Su et al., 2021',
                url: 'https://arxiv.org/abs/2104.09864',
                note: 'The original RoPE paper. Section 3.4 derives the relative position property from the rotation mechanism. The key result: the dot product of rotated Q and K depends only on relative distance.',
              },
              {
                title: 'Generating Long Sequences with Sparse Transformers',
                authors: 'Child et al., 2019',
                url: 'https://arxiv.org/abs/1904.10509',
                note: 'Introduces sparse attention patterns (strided and fixed) for scaling transformers to long sequences. Section 3 covers the sparse factorizations discussed in this lesson.',
              },
              {
                title: 'GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints',
                authors: 'Ainslie et al., 2023',
                url: 'https://arxiv.org/abs/2305.13245',
                note: 'The GQA paper. Shows how to convert existing MHA models to GQA via uptraining. Section 2 describes the MHA\u2192GQA\u2192MQA spectrum.',
              },
              {
                title: 'Mistral 7B',
                authors: 'Jiang et al., 2023',
                url: 'https://arxiv.org/abs/2310.06825',
                note: 'Combines sliding window attention, GQA, and RoPE in a 7B model. Section 2.1 covers the sliding window mechanism. A concrete example of all three techniques working together.',
              },
              {
                title: 'LLaMA 2: Open Foundation and Fine-Tuned Chat Models',
                authors: 'Touvron et al., 2023',
                url: 'https://arxiv.org/abs/2307.09288',
                note: 'The LLaMA 2 family uses RoPE and GQA (8 KV groups for the 70B model). Section 2 covers the architecture choices referenced in the KV cache calculations.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up next: Training & Serving at Scale"
            description="We have now addressed two scaling bottlenecks: MoE decouples parameters from per-token compute, and long-context techniques decouple context length from quadratic cost. But building and serving a model with trillions of tokens of training data, hundreds of billions of parameters distributed across hundreds of GPUs&mdash;that is an engineering challenge beyond any single machine. Next: how parallelism strategies distribute training, and how speculative decoding and continuous batching make serving practical."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
