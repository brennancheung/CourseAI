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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Training & Serving at Scale
 *
 * Third and final lesson in Module 5.3 (Scaling Architecture).
 * CONSOLIDATE lesson—connects existing knowledge to multi-GPU engineering.
 *
 * Cognitive load: 2 genuinely new concepts:
 *   1. Parallelism strategies (data, tensor, pipeline) as a family
 *   2. ZeRO optimizer sharding
 * Plus 2 upgrades from MENTIONED:
 *   3. Speculative decoding → DEVELOPED
 *   4. Continuous batching → DEVELOPED
 *
 * Core concepts at DEVELOPED:
 * - Data parallelism (gradient all-reduce)
 * - Tensor parallelism (split within layers)
 * - Pipeline parallelism (split across layers)
 * - Communication overhead as the central constraint
 * - Speculative decoding (draft-verify loop)
 * - Continuous batching (slot management)
 *
 * Concepts at INTRODUCED:
 * - ZeRO optimizer state sharding
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing any parallelism strategy in code (no PyTorch distributed, FSDP, DeepSpeed)
 * - NCCL, MPI, or communication primitives
 * - Specific hardware interconnect details beyond essential insight
 * - Ring attention or sequence parallelism (MENTIONED only)
 * - Expert parallelism for MoE in detail (MENTIONED only)
 * - Model architecture search or optimal parallelism configuration
 * - Quantized inference serving (already INTRODUCED in lora-and-quantization)
 * - vLLM, TGI, or specific serving frameworks (MENTIONED only)
 * - Gradient accumulation in depth (MENTIONED as related to microbatching)
 *
 * Previous: Long Context & Efficient Attention (Module 5.3, Lesson 2)
 * Next: Series 6 or next module
 */

// ---------------------------------------------------------------------------
// Inline SVG: Parallelism Comparison Diagram
// Three-panel showing data parallelism, tensor parallelism, pipeline
// parallelism with GPU blocks and communication arrows.
// ---------------------------------------------------------------------------

function ParallelismComparisonDiagram() {
  const svgW = 640
  const svgH = 380

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const dataColor = '#34d399' // emerald for data parallelism
  const tensorColor = '#a78bfa' // violet for tensor parallelism
  const pipelineColor = '#f59e0b' // amber for pipeline parallelism
  const gpuBg = '#1e293b'
  const commColor = '#f87171' // rose for communication

  const panelW = 180
  const gap = 20
  const startX = (svgW - 3 * panelW - 2 * gap) / 2
  const panelY = 60

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
          Three Parallelism Strategies
        </text>
        <text
          x={svgW / 2}
          y={38}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
          fontStyle="italic"
        >
          Same goal (use multiple GPUs), different splitting strategies
        </text>

        {/* Panel 1: Data Parallelism */}
        <rect
          x={startX}
          y={panelY}
          width={panelW}
          height={260}
          rx={8}
          fill={dataColor}
          fillOpacity={0.05}
          stroke={dataColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + panelW / 2}
          y={panelY + 18}
          textAnchor="middle"
          fill={dataColor}
          fontSize="11"
          fontWeight="600"
        >
          Data Parallelism
        </text>
        {/* 4 GPUs each with full model */}
        {[0, 1, 2, 3].map((i) => {
          const gx = startX + 20 + (i % 2) * 75
          const gy = panelY + 32 + Math.floor(i / 2) * 55
          return (
            <g key={i}>
              <rect
                x={gx}
                y={gy}
                width={65}
                height={42}
                rx={4}
                fill={gpuBg}
                stroke={dataColor}
                strokeWidth={1}
              />
              <text
                x={gx + 32}
                y={gy + 14}
                textAnchor="middle"
                fill={brightText}
                fontSize="8"
                fontWeight="500"
              >
                GPU {i}
              </text>
              <text
                x={gx + 32}
                y={gy + 26}
                textAnchor="middle"
                fill={dataColor}
                fontSize="7"
              >
                Full Model
              </text>
              <text
                x={gx + 32}
                y={gy + 36}
                textAnchor="middle"
                fill={dimText}
                fontSize="7"
              >
                Batch {i}
              </text>
            </g>
          )
        })}
        {/* All-reduce arrow */}
        <text
          x={startX + panelW / 2}
          y={panelY + 160}
          textAnchor="middle"
          fill={commColor}
          fontSize="8"
          fontWeight="500"
        >
          All-reduce gradients
        </text>
        <line
          x1={startX + 30}
          y1={panelY + 168}
          x2={startX + panelW - 30}
          y2={panelY + 168}
          stroke={commColor}
          strokeWidth={1.5}
          markerEnd="url(#arrowRose)"
        />
        <text
          x={startX + panelW / 2}
          y={panelY + 190}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Split: data (batches)
        </text>
        <text
          x={startX + panelW / 2}
          y={panelY + 202}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Replicated: full model
        </text>
        <text
          x={startX + panelW / 2}
          y={panelY + 216}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Comm: once per step
        </text>
        <text
          x={startX + panelW / 2}
          y={panelY + 240}
          textAnchor="middle"
          fill={dataColor}
          fontSize="8"
          fontWeight="500"
        >
          When: model fits on 1 GPU
        </text>

        {/* Panel 2: Tensor Parallelism */}
        <rect
          x={startX + panelW + gap}
          y={panelY}
          width={panelW}
          height={260}
          rx={8}
          fill={tensorColor}
          fillOpacity={0.05}
          stroke={tensorColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 18}
          textAnchor="middle"
          fill={tensorColor}
          fontSize="11"
          fontWeight="600"
        >
          Tensor Parallelism
        </text>
        {/* 2 GPUs splitting a single layer */}
        {[0, 1].map((i) => {
          const gx = startX + panelW + gap + 20 + i * 75
          const gy = panelY + 40
          return (
            <g key={i}>
              <rect
                x={gx}
                y={gy}
                width={65}
                height={50}
                rx={4}
                fill={gpuBg}
                stroke={tensorColor}
                strokeWidth={1}
              />
              <text
                x={gx + 32}
                y={gy + 14}
                textAnchor="middle"
                fill={brightText}
                fontSize="8"
                fontWeight="500"
              >
                GPU {i}
              </text>
              <text
                x={gx + 32}
                y={gy + 28}
                textAnchor="middle"
                fill={tensorColor}
                fontSize="7"
              >
                Half of W
              </text>
              <text
                x={gx + 32}
                y={gy + 40}
                textAnchor="middle"
                fill={dimText}
                fontSize="7"
              >
                cols {i === 0 ? '0-N/2' : 'N/2-N'}
              </text>
            </g>
          )
        })}
        {/* Bidirectional comm */}
        <line
          x1={startX + panelW + gap + 85}
          y1={panelY + 100}
          x2={startX + panelW + gap + 95}
          y2={panelY + 100}
          stroke={commColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 118}
          textAnchor="middle"
          fill={commColor}
          fontSize="7.5"
          fontWeight="500"
        >
          All-reduce every layer
        </text>
        {/* Label block */}
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 145}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Both GPUs process same input
        </text>
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 190}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Split: layers (within)
        </text>
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 202}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Replicated: nothing
        </text>
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 216}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Comm: within every layer
        </text>
        <text
          x={startX + panelW + gap + panelW / 2}
          y={panelY + 240}
          textAnchor="middle"
          fill={tensorColor}
          fontSize="8"
          fontWeight="500"
        >
          When: layers too wide
        </text>

        {/* Panel 3: Pipeline Parallelism */}
        <rect
          x={startX + 2 * (panelW + gap)}
          y={panelY}
          width={panelW}
          height={260}
          rx={8}
          fill={pipelineColor}
          fillOpacity={0.05}
          stroke={pipelineColor}
          strokeWidth={1.5}
        />
        <text
          x={startX + 2 * (panelW + gap) + panelW / 2}
          y={panelY + 18}
          textAnchor="middle"
          fill={pipelineColor}
          fontSize="11"
          fontWeight="600"
        >
          Pipeline Parallelism
        </text>
        {/* 4 GPUs in a vertical pipeline */}
        {[0, 1, 2, 3].map((i) => {
          const gx = startX + 2 * (panelW + gap) + 45
          const gy = panelY + 32 + i * 38
          return (
            <g key={i}>
              <rect
                x={gx}
                y={gy}
                width={90}
                height={28}
                rx={4}
                fill={gpuBg}
                stroke={pipelineColor}
                strokeWidth={1}
              />
              <text
                x={gx + 45}
                y={gy + 12}
                textAnchor="middle"
                fill={brightText}
                fontSize="8"
                fontWeight="500"
              >
                GPU {i}
              </text>
              <text
                x={gx + 45}
                y={gy + 23}
                textAnchor="middle"
                fill={pipelineColor}
                fontSize="7"
              >
                Blocks {i * 12}–{i * 12 + 11}
              </text>
              {i < 3 && (
                <line
                  x1={gx + 45}
                  y1={gy + 28}
                  x2={gx + 45}
                  y2={gy + 38}
                  stroke={commColor}
                  strokeWidth={1}
                  markerEnd="url(#arrowRose)"
                />
              )}
            </g>
          )
        })}
        <text
          x={startX + 2 * (panelW + gap) + panelW / 2}
          y={panelY + 190}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Split: layers (across)
        </text>
        <text
          x={startX + 2 * (panelW + gap) + panelW / 2}
          y={panelY + 202}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Replicated: nothing
        </text>
        <text
          x={startX + 2 * (panelW + gap) + panelW / 2}
          y={panelY + 216}
          textAnchor="middle"
          fill={dimText}
          fontSize="7.5"
        >
          Comm: between stages only
        </text>
        <text
          x={startX + 2 * (panelW + gap) + panelW / 2}
          y={panelY + 240}
          textAnchor="middle"
          fill={pipelineColor}
          fontSize="8"
          fontWeight="500"
        >
          When: model too deep
        </text>

        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowRose"
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill={commColor} />
          </marker>
        </defs>

        {/* Bottom caption */}
        <text
          x={svgW / 2}
          y={svgH - 30}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Each strategy addresses a different bottleneck. Frontier models combine all three.
        </text>
        <text
          x={svgW / 2}
          y={svgH - 14}
          textAnchor="middle"
          fill={commColor}
          fontSize="9"
        >
          Red = communication overhead (the constraint that shapes every choice)
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Pipeline Bubble Diagram
// Timeline showing GPU idle time with 1 microbatch vs 4 microbatches.
// ---------------------------------------------------------------------------

function PipelineBubbleDiagram() {
  const svgW = 620
  const svgH = 320

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const activeColor = '#34d399' // emerald for active
  const idleColor = '#334155' // dark for idle
  const borderColor = '#475569'

  const gpuH = 22
  const stepW = 40
  const leftLabelW = 50

  // Section 1: 1 microbatch, 4 GPUs — only 1 active at a time
  const s1Y = 55
  const s1Steps = 7 // time steps

  // Section 2: 4 microbatches — pipeline fills
  const s2Y = 185
  const s2Steps = 7

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Pipeline Bubble: Why Microbatching Matters
        </text>

        {/* Section 1: 1 Microbatch */}
        <text
          x={leftLabelW}
          y={s1Y - 8}
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          1 Microbatch—75% idle
        </text>
        {[0, 1, 2, 3].map((gpu) => {
          const gy = s1Y + gpu * (gpuH + 4)
          return (
            <g key={`s1-gpu-${gpu}`}>
              <text
                x={leftLabelW - 5}
                y={gy + gpuH / 2 + 4}
                textAnchor="end"
                fill={dimText}
                fontSize="8"
              >
                GPU {gpu}
              </text>
              {Array.from({ length: s1Steps }).map((_, step) => {
                const sx = leftLabelW + 10 + step * (stepW + 2)
                const isActive = step === gpu
                return (
                  <rect
                    key={step}
                    x={sx}
                    y={gy}
                    width={stepW}
                    height={gpuH}
                    rx={3}
                    fill={isActive ? activeColor : idleColor}
                    fillOpacity={isActive ? 0.7 : 0.3}
                    stroke={isActive ? activeColor : borderColor}
                    strokeWidth={1}
                  />
                )
              })}
            </g>
          )
        })}
        {/* Label: idle time */}
        <text
          x={svgW - 50}
          y={s1Y + 2 * (gpuH + 4) + gpuH / 2 + 4}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          Bubble
        </text>

        {/* Section 2: 4 Microbatches */}
        <text
          x={leftLabelW}
          y={s2Y - 8}
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          4 Microbatches—pipeline fills up
        </text>
        {[0, 1, 2, 3].map((gpu) => {
          const gy = s2Y + gpu * (gpuH + 4)
          return (
            <g key={`s2-gpu-${gpu}`}>
              <text
                x={leftLabelW - 5}
                y={gy + gpuH / 2 + 4}
                textAnchor="end"
                fill={dimText}
                fontSize="8"
              >
                GPU {gpu}
              </text>
              {Array.from({ length: s2Steps }).map((_, step) => {
                const sx = leftLabelW + 10 + step * (stepW + 2)
                // Each microbatch enters the pipeline one step apart
                // GPU g processes microbatch m at step (g + m)
                const microbatch = step - gpu
                const isActive = microbatch >= 0 && microbatch < 4
                // Color each microbatch slightly differently
                const colors = ['#34d399', '#60a5fa', '#f59e0b', '#a78bfa']
                const color = isActive ? colors[microbatch] : idleColor
                return (
                  <g key={step}>
                    <rect
                      x={sx}
                      y={gy}
                      width={stepW}
                      height={gpuH}
                      rx={3}
                      fill={color}
                      fillOpacity={isActive ? 0.6 : 0.3}
                      stroke={isActive ? color : borderColor}
                      strokeWidth={1}
                    />
                    {isActive && (
                      <text
                        x={sx + stepW / 2}
                        y={gy + gpuH / 2 + 3}
                        textAnchor="middle"
                        fill={brightText}
                        fontSize="7"
                      >
                        μB{microbatch}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          )
        })}

        {/* Bottom caption */}
        <text
          x={svgW / 2}
          y={svgH - 10}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          More microbatches = less idle time. The pipeline bubble shrinks as the pipeline fills.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Speculative Decoding Diagram
// Draft model produces 5 tokens, large model verifies in one pass.
// ---------------------------------------------------------------------------

function SpeculativeDecodingDiagram() {
  const svgW = 600
  const svgH = 300

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const draftColor = '#60a5fa' // blue for draft model
  const verifyColor = '#a78bfa' // violet for large model
  const acceptColor = '#34d399' // emerald for accepted
  const rejectColor = '#f87171' // rose for rejected
  const bgBox = '#1e293b'

  const tokenW = 80
  const tokenH = 28
  const tokStartX = 60
  const tokGap = 6

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Speculative Decoding: Draft and Verify
        </text>

        {/* Step 1: Draft model produces 5 tokens */}
        <text
          x={tokStartX}
          y={52}
          fill={draftColor}
          fontSize="10"
          fontWeight="500"
        >
          Step 1: Draft model (7B) generates 5 tokens quickly
        </text>
        {['Paris', ',', 'and', 'the', 'city'].map((tok, i) => {
          const tx = tokStartX + i * (tokenW + tokGap)
          return (
            <g key={`draft-${i}`}>
              <rect
                x={tx}
                y={62}
                width={tokenW}
                height={tokenH}
                rx={4}
                fill={bgBox}
                stroke={draftColor}
                strokeWidth={1}
              />
              <text
                x={tx + tokenW / 2}
                y={62 + tokenH / 2 + 4}
                textAnchor="middle"
                fill={draftColor}
                fontSize="9"
              >
                &ldquo;{tok}&rdquo;
              </text>
            </g>
          )
        })}

        {/* Step 2: Large model verifies in ONE forward pass */}
        <text
          x={tokStartX}
          y={120}
          fill={verifyColor}
          fontSize="10"
          fontWeight="500"
        >
          Step 2: Large model (70B) verifies all 5 in one forward pass
        </text>

        {/* Arrow down */}
        <line
          x1={svgW / 2}
          y1={95}
          x2={svgW / 2}
          y2={108}
          stroke={dimText}
          strokeWidth={1}
        />
        <polygon
          points={`${svgW / 2},112 ${svgW / 2 - 4},106 ${svgW / 2 + 4},106`}
          fill={dimText}
        />

        {/* Step 3: Accept/reject results */}
        <text
          x={tokStartX}
          y={158}
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          Step 3: Accept matches, reject from first disagreement
        </text>
        {[
          { tok: 'Paris', accepted: true },
          { tok: ',', accepted: true },
          { tok: 'and', accepted: true },
          { tok: 'the', accepted: false },
          { tok: 'city', accepted: false },
        ].map((item, i) => {
          const tx = tokStartX + i * (tokenW + tokGap)
          const color = item.accepted ? acceptColor : rejectColor
          return (
            <g key={`verify-${i}`}>
              <rect
                x={tx}
                y={168}
                width={tokenW}
                height={tokenH}
                rx={4}
                fill={bgBox}
                stroke={color}
                strokeWidth={1.5}
              />
              <text
                x={tx + tokenW / 2}
                y={168 + tokenH / 2 + 4}
                textAnchor="middle"
                fill={color}
                fontSize="9"
              >
                &ldquo;{item.tok}&rdquo; {item.accepted ? '✓' : '✗'}
              </text>
            </g>
          )
        })}

        {/* Result box */}
        <rect
          x={60}
          y={218}
          width={svgW - 120}
          height={65}
          rx={6}
          fill={acceptColor}
          fillOpacity={0.06}
          stroke={acceptColor}
          strokeWidth={1}
          strokeOpacity={0.3}
        />
        <text
          x={svgW / 2}
          y={240}
          textAnchor="middle"
          fill={acceptColor}
          fontSize="10"
          fontWeight="500"
        >
          Result: 3 tokens from 1 large forward pass (instead of 3 separate passes)
        </text>
        <text
          x={svgW / 2}
          y={258}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Large model resamples at first rejection. Draft tokens after rejection are discarded.
        </text>
        <text
          x={svgW / 2}
          y={272}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Output quality is identical to the large model alone—verification guarantees this.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Continuous Batching Diagram
// Timeline comparing static batching (wasted slots) vs continuous batching.
// ---------------------------------------------------------------------------

function ContinuousBatchingDiagram() {
  const svgW = 620
  const svgH = 340

  const dimText = '#94a3b8'
  const brightText = '#e2e8f0'
  const activeColors = [
    '#34d399', '#60a5fa', '#f59e0b', '#a78bfa',
    '#f87171', '#22d3ee', '#fb923c', '#e879f9',
  ]
  const wastedColor = '#334155'
  const newReqColor = '#6ee7b7' // lighter emerald for new requests

  const slotH = 18
  const slotGap = 3
  const timeStepW = 22
  const leftLabel = 60

  // Request completion lengths (8 requests)
  const completions = [3, 7, 12, 2, 6, 10, 4, 12]
  const maxLen = 12
  const numSlots = 8

  const s1Y = 55
  const s2Y = 210

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        <text
          x={svgW / 2}
          y={20}
          textAnchor="middle"
          fill={brightText}
          fontSize="13"
          fontWeight="600"
        >
          Static vs Continuous Batching
        </text>

        {/* Section 1: Static Batching */}
        <text
          x={leftLabel}
          y={s1Y - 8}
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          Static Batching—wasted slots after completion
        </text>
        {Array.from({ length: numSlots }).map((_, slot) => {
          const sy = s1Y + slot * (slotH + slotGap)
          return (
            <g key={`static-${slot}`}>
              <text
                x={leftLabel - 5}
                y={sy + slotH / 2 + 3}
                textAnchor="end"
                fill={dimText}
                fontSize="7"
              >
                Req {slot}
              </text>
              {Array.from({ length: maxLen }).map((_, step) => {
                const sx = leftLabel + 5 + step * (timeStepW + 1)
                const isActive = step < completions[slot]
                return (
                  <rect
                    key={step}
                    x={sx}
                    y={sy}
                    width={timeStepW}
                    height={slotH}
                    rx={2}
                    fill={isActive ? activeColors[slot] : wastedColor}
                    fillOpacity={isActive ? 0.5 : 0.2}
                    stroke={isActive ? activeColors[slot] : wastedColor}
                    strokeWidth={0.5}
                    strokeOpacity={0.5}
                  />
                )
              })}
            </g>
          )
        })}
        {/* Wasted label */}
        <text
          x={svgW - 40}
          y={s1Y + 4 * (slotH + slotGap)}
          textAnchor="middle"
          fill={dimText}
          fontSize="8"
        >
          Dark = wasted
        </text>

        {/* Section 2: Continuous Batching */}
        <text
          x={leftLabel}
          y={s2Y - 8}
          fill={brightText}
          fontSize="10"
          fontWeight="500"
        >
          Continuous Batching—completed slots filled immediately
        </text>
        {Array.from({ length: numSlots }).map((_, slot) => {
          const sy = s2Y + slot * (slotH + slotGap)
          return (
            <g key={`cont-${slot}`}>
              <text
                x={leftLabel - 5}
                y={sy + slotH / 2 + 3}
                textAnchor="end"
                fill={dimText}
                fontSize="7"
              >
                Slot {slot}
              </text>
              {Array.from({ length: maxLen }).map((_, step) => {
                const sx = leftLabel + 5 + step * (timeStepW + 1)
                // Original request is active until completion
                const originalActive = step < completions[slot]
                // After completion, slot gets a new request (shown in green)
                const color = originalActive ? activeColors[slot] : newReqColor
                return (
                  <rect
                    key={step}
                    x={sx}
                    y={sy}
                    width={timeStepW}
                    height={slotH}
                    rx={2}
                    fill={color}
                    fillOpacity={originalActive ? 0.5 : 0.3}
                    stroke={color}
                    strokeWidth={0.5}
                    strokeOpacity={0.5}
                  />
                )
              })}
            </g>
          )
        })}
        <text
          x={svgW - 40}
          y={s2Y + 4 * (slotH + slotGap)}
          textAnchor="middle"
          fill={newReqColor}
          fontSize="8"
        >
          Green = new req
        </text>

        {/* Bottom caption */}
        <text
          x={svgW / 2}
          y={svgH - 10}
          textAnchor="middle"
          fill={dimText}
          fontSize="9"
        >
          Continuous batching keeps GPU utilization near 100% by filling completed slots from a queue.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function TrainingAndServingAtScaleLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Training & Serving at Scale"
            description="You have spent two lessons learning how to make the transformer more powerful&mdash;MoE decoupled parameters from compute, and long-context techniques broke through the sequence length wall. But there is a gap between &ldquo;here is a better architecture&rdquo; and &ldquo;here is a model serving millions of users.&rdquo; A Mixtral 8&times;7B model has 47 billion parameters. Even before training starts, those parameters do not fit on a single GPU. And once trained, generating text one token at a time means a 70B model sits idle between forward passes. This lesson closes the gap."
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
            By the end of this lesson, you will be able to explain how training
            and inference workloads are distributed across multiple GPUs,
            identify which parallelism strategy addresses which bottleneck, and
            describe how speculative decoding and continuous batching transform
            autoregressive inference from a sequential bottleneck into a
            practical serving system.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Three parallelism strategies: data, tensor, pipeline—when to use each and what each trades off',
              'Communication overhead as the central constraint shaping all parallelism design',
              'ZeRO optimizer sharding: concept and motivation (not implementation details)',
              'Speculative decoding: the draft-verify mechanism that parallelizes autoregressive generation',
              'Continuous batching: slot management for high-throughput serving',
              'NOT: implementing parallelism in code (no PyTorch distributed, FSDP, or DeepSpeed)',
              'NOT: NCCL, MPI, or communication primitive details',
              'NOT: ring attention or sequence parallelism (mentioned only)',
              'NOT: quantized inference serving (already introduced in LoRA & Quantization)',
              'NOT: specific serving frameworks like vLLM or TGI (mentioned only)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Building on What You Know">
            After the conceptual work of Long Context &amp; Efficient
            Attention, this lesson connects concepts you already
            know&mdash;training memory breakdown, the generate() loop,
            compute-bound vs memory-bound&mdash;to multi-GPU engineering. No
            new paradigm shift. The same bottleneck reasoning, applied at a
            larger scale.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Recap: Three Facts
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Facts About to Combine"
            subtitle="What you already know that makes parallelism inevitable"
          />
          <div className="space-y-4">
            <ol className="list-decimal list-inside text-muted-foreground space-y-3 ml-4">
              <li>
                <strong>Training memory breakdown</strong> from LoRA &amp;
                Quantization: a 7B model needs ~84 GB for training. Weights in
                bf16 (14 GB) + gradients (14 GB) + optimizer states in fp32 (56
                GB). Optimizer states are two-thirds of the total.
              </li>
              <li>
                <strong>The generate() method</strong> from Building nanoGPT:
                each token requires one forward pass. Run forward, take the last
                logits, sample, append, repeat. Sequential by nature.
              </li>
              <li>
                <strong>Compute-bound vs memory-bound</strong> from Scaling &amp;
                Efficiency: moving data is slower than computing on it. The
                bottleneck is the delivery truck, not the chefs.
              </li>
            </ol>

            <p className="text-muted-foreground">
              Now connect these: what happens when the model has 70 billion
              parameters?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Foundation Is Solid">
            You have concrete arithmetic for training memory, a working
            generate() loop you built yourself, and the delivery truck analogy
            for data movement. Every concept in this lesson builds on these
            three.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Hook: The Scale Wall
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Scale Wall"
            subtitle="When the model does not fit"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A 70B model. Training memory: 70 billion parameters &times; 12
              bytes per parameter (bf16 weights + bf16 gradients + fp32 optimizer
              states) = <strong>840 GB</strong>. The best single GPU available,
              an A100, has <strong>80 GB</strong> of memory.
            </p>

            <p className="text-muted-foreground">
              This is not a performance problem. It is not &ldquo;slow.&rdquo;
              It is <strong>physically impossible to begin</strong>. The model
              does not fit. You cannot run a single forward pass, compute a
              single gradient, or take a single optimizer step.
            </p>

            <p className="text-muted-foreground">
              Second punch: even if you somehow train it, serving it means
              generating one token at a time in a sequential loop. A 70B model
              generates roughly 30&ndash;50 tokens per second. 1,000 concurrent
              users each wanting 100 tokens = 100,000 sequential forward passes.
              That is not slow&mdash;it is impossible to serve without
              engineering around the sequential bottleneck.
            </p>

            <GradientCard title="Two Walls" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Training wall:</strong> the model does not fit on one
                  GPU. 840 GB vs 80 GB. Not a speed issue&mdash;a physics issue.
                </p>
                <p>
                  <strong>Inference wall:</strong> the model generates one token
                  at a time. Sequential generation cannot serve thousands of
                  concurrent users.
                </p>
                <p className="text-muted-foreground">
                  Different problems, different solutions. The rest of this
                  lesson addresses each.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not &ldquo;Faster&rdquo;&mdash;Possible">
            The engineering here is not about making training faster. For a 70B
            model, single-GPU training is not slow&mdash;it is impossible.
            Parallelism makes it possible at all. Inference optimization makes
            serving possible at all. The bar is feasibility, not performance.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain Part 1a: Data Parallelism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Data Parallelism"
            subtitle="The simple case—when the model fits"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Start with what works. Your GPT-2 at 124M parameters fits easily
              on one GPU. Data parallelism: replicate the full model on each
              GPU, split the training data. Each GPU computes gradients on its
              own batch. Average the gradients across all GPUs. Each GPU applies
              the same averaged gradient update. The weights stay synchronized.
            </p>

            <p className="text-muted-foreground">
              4 GPUs training GPT-2. Each sees 1/4 of the data per step. Each
              computes gradients independently. An <strong>all-reduce</strong>
              {' '}operation averages gradients across GPUs. Each GPU takes the
              same optimizer step. Result: N GPUs ={' '}
              <InlineMath math="N\times" /> throughput.
            </p>

            <CodeBlock
              code={`# Data parallelism: each GPU runs the same model on different data
# After backward pass, average gradients across GPUs

for batch in data_loader:
    local_batch = batch[gpu_rank]        # Each GPU gets a different slice
    loss = model(local_batch)            # Forward pass (same model weights)
    loss.backward()                      # Backward pass (local gradients)

    all_reduce(model.gradients)          # Average gradients across all GPUs
    optimizer.step()                     # Same update on every GPU
    optimizer.zero_grad()                # Weights stay synchronized`}
              language="python"
              filename="data_parallelism.py"
            />

            <GradientCard title="Data Parallelism Works When..." color="emerald">
              <p className="text-sm">
                The model fits on one GPU. Each GPU holds a full copy of the
                model, its gradients, and its optimizer states. For GPT-2 at
                124M parameters, this is easy. For a 70B model needing 840 GB?
                Data parallelism alone cannot help&mdash;the full model does not
                fit on any single GPU.
              </p>
            </GradientCard>

            <GradientCard title="Misconception: &ldquo;Parallelism is just data parallelism&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  The most natural mental model of parallelism is data
                  parallelism&mdash;split the data, replicate the model. But
                  data parallelism requires <strong>each GPU to hold the full
                  model</strong>. A 70B model needs 840 GB for training. An
                  A100 has 80 GB. The model does not fit on a single GPU, so
                  there is nothing to replicate. Data parallelism is not an
                  option here&mdash;you need strategies that split the model
                  itself.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Gradients, Same Update">
            Data parallelism does not change the training algorithm. The model
            sees the same data, computes the same averaged gradients, and takes
            the same optimizer step. The only difference: the work is split
            across GPUs. The mathematics is identical to single-GPU training
            with a larger batch.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5b. Communication is the Constraint (pivot)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Communication Is the Constraint"
            subtitle="The delivery truck analogy, one level up"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before presenting solutions to &ldquo;model too large,&rdquo;
              establish the central constraint. Any time GPUs share computation,
              they must communicate. Concrete numbers:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                A100 compute: ~312 TFLOPS (bf16)
              </li>
              <li>
                NVLink bandwidth: ~600 GB/s
              </li>
              <li>
                Transferring 1 GB: ~1.7 ms
              </li>
              <li>
                In 1.7 ms, the GPU could have performed{' '}
                <strong>~530 billion floating-point operations</strong>
              </li>
            </ul>

            <p className="text-muted-foreground">
              You already knew this. In Scaling &amp; Efficiency, you learned
              that moving data is slower than computing on it&mdash;the delivery
              truck analogy. Parallelism puts this on a bigger stage: now the
              delivery truck drives between buildings, not between floors of the
              same building. The gap between compute speed and communication
              speed only grows.
            </p>

            <GradientCard title="Misconception: &ldquo;Communication between GPUs is fast because they are in the same machine&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  As a software engineer, inter-process communication feels
                  fast. And NVLink at 600 GB/s sounds impressive. But compare
                  it to what the GPU can <em>compute</em> in the same time.
                  Transferring 1 GB takes ~1.7 ms. In that 1.7 ms, the GPU
                  could have performed 530 billion floating-point operations.
                </p>
                <p className="text-muted-foreground">
                  At scale, this adds up. Training a 175B model with naive
                  tensor parallelism across 8 GPUs can spend{' '}
                  <strong>40&ndash;60% of time waiting for
                  communication</strong>&mdash;not computing, just waiting for
                  data to arrive from other GPUs. Communication is not a minor
                  overhead. It is the dominant constraint.
                </p>
              </div>
            </GradientCard>

            <GradientCard title="The Central Constraint" color="violet">
              <p className="text-sm">
                Communication overhead shapes every parallelism decision. Every
                strategy you will see in this lesson is a tradeoff between
                distributing computation (good&mdash;uses more GPUs) and
                communication cost (bad&mdash;GPUs wait for data). The art of
                distributed training is minimizing communication while
                maximizing parallelism.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Bottleneck, Bigger Scale">
            Compute-bound vs memory-bound was about data movement on one chip.
            Multi-GPU parallelism is the same bottleneck between chips. NVLink
            at 600 GB/s sounds fast. Compared to 312 TFLOPS of compute, it is
            the slow delivery truck all over again.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5c. Tensor Parallelism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tensor Parallelism"
            subtitle="Split within layers—both GPUs compute simultaneously"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A single linear layer in a 70B model might have a weight matrix of
              shape (8192, 32768). That is ~1 GB for one layer in bf16. Hundreds
              of these layers.
            </p>

            <p className="text-muted-foreground">
              <strong>Mechanism:</strong> split the weight matrix column-wise
              across GPUs. GPU 0 stores columns 0 through 16,383. GPU 1 stores
              columns 16,384 through 32,767. Each GPU computes its half of the
              output. Then GPUs exchange partial results (all-reduce) to
              reconstruct the full output.
            </p>

            <p className="text-muted-foreground">
              This is the same <InlineMath math="\texttt{nn.Linear}" /> you
              built in Building nanoGPT, split across two chips. Both GPUs
              compute simultaneously on the same input. Communication happens{' '}
              <strong>within each layer</strong>&mdash;after each split
              matmul. Fine-grained communication, high frequency.
            </p>

            <p className="text-muted-foreground">
              Where it is used: within a single transformer block&rsquo;s
              attention and FFN layers. For MoE models, placing different
              experts on different GPUs is a specialized form of tensor
              parallelism where the router determines which GPU does the work
              per token.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Layer, Two Chips">
            The student mental model: &ldquo;take my nn.Linear with shape
            (8192, 32768), slice it down the columns, put each half on a
            different GPU.&rdquo; That is tensor parallelism. Simple concept,
            but the communication cost of synchronizing after every layer is
            the price you pay.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5d. Pipeline Parallelism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Pipeline Parallelism"
            subtitle="Split across layers—the assembly line"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Even with tensor parallelism, extremely large models need another
              dimension of splitting. Pipeline parallelism assigns different
              layers to different GPUs. GPU 0 runs blocks 0&ndash;11, GPU 1
              runs blocks 12&ndash;23, GPU 2 runs blocks 24&ndash;35, GPU 3
              runs blocks 36&ndash;47. A microbatch flows through the pipeline:
              GPU 0 &rarr; GPU 1 &rarr; GPU 2 &rarr; GPU 3.
            </p>

            <GradientCard title="The Assembly Line Analogy" color="cyan">
              <p className="text-sm">
                An assembly line with 4 stations. Each station does one step.
                For a single item, only one station is active at a time&mdash;3
                out of 4 stations are idle. But when the line is full (many
                items flowing through), all stations work simultaneously. The
                pipeline is inefficient for one item but fast for many.
              </p>
            </GradientCard>

            <PipelineBubbleDiagram />

            <p className="text-muted-foreground">
              <strong>The pipeline bubble</strong> is pipeline parallelism&rsquo;s key
              inefficiency. With 4 stages and 1 microbatch, only one GPU is
              active at a time&mdash;75% of GPUs are idle. The solution:{' '}
              <strong>microbatching</strong>. Split the batch into microbatches.
              While GPU 1 processes microbatch 0, GPU 0 starts microbatch 1.
              The pipeline fills. With enough microbatches, utilization
              approaches 100%. Gradient accumulation across microbatches handles
              the optimizer step.
            </p>

            <p className="text-muted-foreground">
              Communication happens <strong>between stages only</strong>&mdash;
              when one GPU passes activations to the next. Lower frequency than
              tensor parallelism, but each transfer can be large.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Pipeline Bubble Is Real">
            Pipeline parallelism is not free. The startup and drain phases where
            the pipeline is partially filled represent wasted compute. More
            microbatches reduce the bubble but increase memory. There is always
            a tradeoff.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5e. Side-by-side comparison
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Comparing the Three Strategies"
            subtitle="Different splits, different communication, different use cases"
          />
          <div className="space-y-4">
            <ParallelismComparisonDiagram />

            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Property</th>
                    <th className="text-center py-2 px-3 text-emerald-400 font-medium">Data</th>
                    <th className="text-center py-2 px-3 text-violet-400 font-medium">Tensor</th>
                    <th className="text-center py-2 px-3 text-amber-400 font-medium">Pipeline</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">What is replicated</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Full model</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Nothing</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Nothing</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">What is split</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Data (batches)</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Layers (within)</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Layers (across)</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">Comm pattern</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">After backward</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Within every layer</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Between stages</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">Comm frequency</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Low</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">High</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Medium</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-muted-foreground">When to use</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Model fits on 1 GPU</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Layers too wide</td>
                    <td className="text-center py-2 px-3 text-muted-foreground">Model too deep</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Address Misconception 2: tensor vs pipeline are not the same */}
            <GradientCard title="Misconception: &ldquo;Tensor and pipeline parallelism are the same thing&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Both split the model across GPUs. But the granularity is
                  completely different. <strong>Tensor parallelism</strong>
                  {' '}splits a single matrix multiply&mdash;both GPUs compute
                  simultaneously on the same token, communicating{' '}
                  <em>within</em> each layer. <strong>Pipeline
                  parallelism</strong> assigns entire layers to different
                  GPUs&mdash;communication happens only{' '}
                  <em>between</em> stages. Different communication patterns,
                  different bottlenecks, different use cases.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Naming Tells You">
            &ldquo;Data&rdquo; parallelism splits data. &ldquo;Tensor&rdquo;
            parallelism splits tensors (weight matrices within a layer).
            &ldquo;Pipeline&rdquo; parallelism arranges layers into a
            pipeline. The name describes what is split.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check 1: Predict-and-Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Pick the Parallelism Strategy" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                You have a 30B dense model and 4 A100 GPUs (80 GB each).
                Training memory for 30B is ~360 GB (30B &times; 12 bytes).
              </p>

              <p><strong>Predict before revealing:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Can you use data parallelism alone?</li>
                <li>Can you use pipeline parallelism with 4 stages?</li>
                <li>What combination would work?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Work through the arithmetic, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                    <li>
                      <strong>Data parallelism alone: no.</strong> Each GPU
                      needs a full copy of the model, gradients, and optimizer
                      states. 360 GB &gt; 80 GB. The model does not fit on any
                      single GPU.
                    </li>
                    <li>
                      <strong>Pipeline parallelism with 4 stages: it depends.</strong>
                      {' '}Each stage has ~7.5B params. Weights alone: ~15 GB
                      (bf16). But with gradients and optimizer states for that
                      stage: 7.5B &times; 12 bytes = ~90 GB. Still does not fit
                      on one 80 GB GPU.
                    </li>
                    <li>
                      <strong>Combination needed.</strong> Pipeline parallelism
                      to split layers into stages, combined with tensor
                      parallelism within stages to further reduce per-GPU
                      memory. Or use ZeRO (coming next) to shard optimizer
                      states across GPUs, dramatically reducing per-GPU memory
                      requirements.
                    </li>
                  </ol>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain Part 2: ZeRO Optimizer Sharding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ZeRO: Optimizer State Sharding"
            subtitle="Data parallelism without the redundancy"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Data parallelism requires each GPU to hold the full model. We just
              showed that a 70B model does not fit. Is there a middle ground?
            </p>

            <p className="text-muted-foreground">
              ZeRO (Zero Redundancy Optimizer) keeps forward and backward passes
              working as normal&mdash;each GPU can still compute gradients. But
              instead of every GPU storing all optimizer states, each GPU stores
              only a fraction. Recall from LoRA &amp; Quantization: optimizer
              states are <strong>two-thirds of training memory</strong>. Sharding
              just the optimizer states across GPUs cuts the dominant cost.
            </p>

            <p className="text-muted-foreground">
              Concrete arithmetic for a 70B model on 8 GPUs:
            </p>

            <ComparisonRow
              left={{
                title: 'Without ZeRO',
                color: 'amber',
                items: [
                  'Each GPU: weights 140 GB (bf16)',
                  '+ gradients 140 GB',
                  '+ optimizer states 560 GB',
                  '= ~840 GB per GPU',
                  'Impossible on 80 GB A100s',
                ],
              }}
              right={{
                title: 'ZeRO Stage 1',
                color: 'emerald',
                items: [
                  'Each GPU: weights 140 GB',
                  '+ gradients 140 GB',
                  '+ 1/8 optimizer states: 70 GB',
                  '= ~350 GB per GPU',
                  'Still needs tensor/pipeline help',
                ],
              }}
            />

            <p className="text-muted-foreground">
              350 GB is still too much for an 80 GB GPU. But combined with
              tensor parallelism to split the weights and gradients, the
              per-GPU requirements become manageable. ZeRO does not replace
              other parallelism strategies&mdash;it reduces the memory overhead
              that makes them necessary.
            </p>

            <p className="text-muted-foreground">
              ZeRO has three stages: Stage 1 shards optimizer states, Stage 2
              also shards gradients, Stage 3 shards everything including model
              parameters. DeepSpeed and PyTorch FSDP (Fully Sharded Data
              Parallel) are frameworks that implement these stages.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Motivation Was in Your Arithmetic">
            In LoRA &amp; Quantization, you computed that 56 GB of 84 GB for a
            7B model is optimizer states. ZeRO&rsquo;s insight: &ldquo;If
            optimizer states are 2/3 of training memory, and data parallelism
            duplicates everything, then the biggest win is not duplicating
            optimizer states.&rdquo; Your arithmetic from that lesson is the
            motivation for this technique.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain Part 3a: Inference — The Sequential Bottleneck
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Inference Bottleneck"
            subtitle="Training solved. Now serving the model."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training is handled. The model works. Now it needs to serve users.
              New problem.
            </p>

            <p className="text-muted-foreground">
              Autoregressive generation is sequential. Each token requires a
              full forward pass. A 70B model generates ~30&ndash;50 tokens per
              second. 1,000 concurrent users each wanting 100 tokens is not a
              throughput problem you can solve by adding more requests to a
              batch&mdash;the sequential nature of generation means each user
              waits for their tokens one at a time.
            </p>

            <p className="text-muted-foreground">
              Two independent solutions for two independent bottlenecks:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Latency Problem" color="blue">
                <p className="text-sm">
                  Each user&rsquo;s generation is slow because each token
                  requires a separate large-model forward pass.{' '}
                  <strong>Speculative decoding</strong> addresses this.
                </p>
              </GradientCard>
              <GradientCard title="Throughput Problem" color="purple">
                <p className="text-sm">
                  Serving many users simultaneously wastes GPU compute when
                  requests complete at different times.{' '}
                  <strong>Continuous batching</strong> addresses this.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          8a. Speculative Decoding
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Speculative Decoding"
            subtitle="Draft fast, verify in parallel"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember your generate() loop from Building nanoGPT: each
              iteration runs one forward pass, takes the last logits, samples
              one token, appends it. One token per forward pass. The sequential
              bottleneck is not the compute per token&mdash;it is the number of
              serial forward passes.
            </p>

            <p className="text-muted-foreground">
              Key insight: the large model&rsquo;s forward pass costs roughly
              the same whether processing 1 token or 5 tokens. Remember from
              Scaling &amp; Efficiency: matrix multiplications are
              compute-bound&mdash;the time depends on the weight matrix
              dimensions (<InlineMath math="d_\text{model}" />,{' '}
              <InlineMath math="d_\text{vocab}" />), not the batch dimension.
              Processing 5 tokens instead of 1 increases the batch dimension
              from 1 to 5, but the weight matrices are the same size. The
              compute is dominated by the weights, not the inputs. So if you
              could give the large model 5 tokens to verify at once instead of
              generating them one at a time...
            </p>

            <GradientCard title="The Rough Draft and Editor Analogy" color="cyan">
              <p className="text-sm">
                A fast but imprecise writer (7B draft model) produces a rough
                draft of 5 sentences. A careful editor (70B large model) reads
                all 5 at once and marks which ones are good. The editor does not
                write faster&mdash;but reviewing 5 sentences at once is almost
                as fast as reviewing 1. The speed comes from the parallelism of
                verification, not from the speed of the draft writer.
              </p>
            </GradientCard>

            <SpeculativeDecodingDiagram />

            <p className="text-muted-foreground">
              <strong>Worked example:</strong> &ldquo;The capital of France
              is&rdquo;&mdash;the draft model (7B) produces [&ldquo;Paris&rdquo;,
              &ldquo;,&rdquo;, &ldquo;and&rdquo;, &ldquo;the&rdquo;,
              &ldquo;city&rdquo;]. The large model (70B) verifies in one
              forward pass: agrees on [&ldquo;Paris&rdquo;, &ldquo;,&rdquo;,
              &ldquo;and&rdquo;], disagrees on &ldquo;the&rdquo; (prefers
              &ldquo;it&rdquo;). Accept the first 3 tokens. One large forward
              pass produced 3 tokens instead of 3 separate passes. The large
              model resamples at the first rejection point.
            </p>

            {/* Misconception 3 */}
            <GradientCard title="Misconception: &ldquo;The speed comes from using a smaller model&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  If the small model were sufficient, you would just use the
                  small model. The speed does not come from the draft model
                  being fast. It comes from the <strong>large model verifying
                  multiple tokens in parallel</strong> in a single forward pass.
                  A single forward pass through the 70B model costs roughly the
                  same whether verifying 1 token or 5 tokens. The large model
                  does the same total work&mdash;it just does it in parallel.
                </p>
                <p className="text-muted-foreground">
                  Output quality is identical to using the large model alone.
                  The verification step guarantees this&mdash;any token the
                  large model disagrees with is rejected and resampled.
                </p>
              </div>
            </GradientCard>

            <CodeBlock
              code={`# Speculative decoding: modified generate() loop
def speculative_generate(draft_model, large_model, prompt, K=5):
    tokens = prompt

    while not done:
        # Step 1: Draft model generates K candidates quickly
        draft_tokens = []
        for _ in range(K):
            next_token = draft_model.generate_one(tokens + draft_tokens)
            draft_tokens.append(next_token)

        # Step 2: Large model verifies ALL K tokens in ONE forward pass
        # (processes the full sequence including drafts)
        large_logits = large_model.forward(tokens + draft_tokens)

        # Step 3: Accept tokens where draft and large model agree
        accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            large_choice = sample(large_logits[len(tokens) + i])
            if large_choice == draft_tok:
                accepted += 1
            else:
                # Reject from here — use large model's token instead
                tokens.append(large_choice)
                break
        else:
            tokens.extend(draft_tokens)  # All accepted

        # Result: up to K tokens from 1 large forward pass`}
              language="python"
              filename="speculative_decoding.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Modifying Your generate() Loop">
            The speculative decoding loop is a direct modification of the
            generate() method you built: instead of &ldquo;forward, sample,
            append, repeat,&rdquo; it is &ldquo;draft K tokens, verify all K
            in one forward pass, accept matches.&rdquo; Same loop, batched
            verification.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8b. Continuous Batching
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Continuous Batching"
            subtitle="Fill completed slots, not wasted compute"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Static batching: 8 requests start together. Request lengths vary:
              some finish at 20 tokens, others at 200. The batch runs until the
              longest finishes. Completed requests sit idle, consuming GPU
              memory and producing nothing.
            </p>

            {/* Negative example: static batching waste */}
            <GradientCard title="Static Batching Waste" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  8 requests with completion lengths [20, 45, 200, 15, 80,
                  150, 30, 200]. After 20 tokens, 2 requests are done. After
                  45 tokens, 3 are done. But the batch runs until token 200.
                  Those completed slots generate padding tokens&mdash;GPU
                  cycles spent producing nothing useful. Over 40% of GPU
                  compute is wasted.
                </p>
              </div>
            </GradientCard>

            <ContinuousBatchingDiagram />

            <GradientCard title="The Restaurant Waitlist Analogy" color="cyan">
              <p className="text-sm">
                A restaurant with 8 tables. Static batching: seat 8 parties at
                once. When a party finishes, the table sits empty until all 8
                parties finish. Only then seat the next 8. Continuous batching:
                as each table opens, seat the next party from the waitlist
                immediately. The restaurant stays full.
              </p>
            </GradientCard>

            <p className="text-muted-foreground">
              <strong>Mechanism:</strong> the server maintains a fixed number
              of batch slots (the tables). Each slot tracks one active
              request. At every step, all active slots generate one token
              simultaneously (one forward pass for the whole batch). When a
              slot&rsquo;s request completes, the server pulls the next
              request from a queue and assigns it to that slot. The batch
              size stays constant; the contents change dynamically. GPU
              utilization stays near 100% instead of declining as short
              requests complete.
            </p>

            {/* Misconception 5 */}
            <GradientCard title="Misconception: &ldquo;Continuous batching is just using a bigger batch size&rdquo;" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  The improvement is not in batch size&mdash;it is in{' '}
                  <strong>slot utilization</strong>. A static batch of 8 and a
                  continuous batch of 8 have the same maximum parallelism. The
                  difference: in static batching, completed slots sit idle. In
                  continuous batching, completed slots are immediately
                  reassigned. The batch size does not change&mdash;the waste
                  does.
                </p>
              </div>
            </GradientCard>

            <CodeBlock
              code={`# Continuous batching: slot management with a queue
class InferenceServer:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.slots = [None] * batch_size  # Active requests
        self.queue = []                    # Waiting requests

    def step(self):
        # Fill any empty slots from the queue
        for i, slot in enumerate(self.slots):
            if slot is None and self.queue:
                self.slots[i] = self.queue.pop(0)

        # Run one forward pass for all active slots simultaneously
        active = [s for s in self.slots if s is not None]
        if active:
            next_tokens = self.model.forward_batch(active)

            # Check for completed requests, free their slots
            for i, slot in enumerate(self.slots):
                if slot is not None and slot.is_complete():
                    slot.return_response()
                    self.slots[i] = None  # Slot ready for next request`}
              language="python"
              filename="continuous_batching.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Problems, Different Solutions">
            Speculative decoding reduces <strong>latency</strong>&mdash;each
            user gets tokens faster. Continuous batching increases{' '}
            <strong>throughput</strong>&mdash;the server handles more users
            simultaneously. They are independent and combine.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2: Transfer Question (MoE deployment)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Deploying Mixtral" color="emerald">
            <div className="space-y-4 text-sm">
              <p>
                A company deploys a Mixtral 8&times;7B model (47B total
                parameters, 13B active per token) for a customer service
                chatbot. They have 8 A100 GPUs.
              </p>

              <p><strong>Reason through these questions:</strong></p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Why does MoE make parallelism especially important?</li>
                <li>Would speculative decoding work well here?</li>
                <li>What role does continuous batching play?</li>
              </ol>

              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think it through, then reveal
                </summary>
                <div className="mt-3 space-y-3">
                  <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                    <li>
                      <strong>MoE&rsquo;s memory problem:</strong> all 47B
                      parameters must be loaded even though only 13B activate
                      per token. The memory requirement is determined by total
                      parameters, not active parameters. 47B in bf16 alone is
                      ~94 GB&mdash;already exceeding a single A100. Expert
                      parallelism (placing different experts on different GPUs)
                      is natural and necessary.
                    </li>
                    <li>
                      <strong>Speculative decoding: yes, with caveats.</strong>
                      {' '}The 13B active compute per token is still expensive for
                      generation. A small dense draft model could produce
                      candidates quickly. But the draft model must approximate
                      the MoE model&rsquo;s output distribution, which is
                      harder when different experts activate for different token
                      types. The acceptance rate may be lower than for a dense
                      model.
                    </li>
                    <li>
                      <strong>Continuous batching is critical.</strong> Customer
                      service = many concurrent short requests. Throughput
                      matters more than single-request latency. Continuous
                      batching keeps all 8 GPUs utilized as short customer
                      queries complete and are replaced by new ones from the
                      queue.
                    </li>
                  </ol>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Elaborate: Frontier Models Combine Everything
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="How Frontier Models Combine Everything"
            subtitle="It is never just one technique"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Modern frontier models use combinations of all these techniques.
              LLaMA 2 70B training: tensor parallelism across 8 GPUs per node,
              pipeline parallelism across nodes, data parallelism across node
              groups, ZeRO for optimizer sharding. Mixtral serving: expert
              parallelism (different experts on different GPUs) + continuous
              batching + speculative decoding.
            </p>

            <p className="text-muted-foreground">
              Each technique addresses a specific bottleneck. The art of
              distributed systems engineering is identifying which bottleneck
              dominates and applying the right combination. This echoes the
              module&rsquo;s connecting thread&mdash;&ldquo;three barriers,
              three targeted solutions&rdquo;&mdash;extended from architecture
              to engineering.
            </p>

            <p className="text-muted-foreground">
              Serving frameworks like vLLM and Text Generation Inference (TGI)
              package many of these techniques together. Ring attention and
              sequence parallelism extend parallelism to the sequence dimension
              for extremely long contexts. Gradient accumulation across
              microbatches connects to pipeline parallelism. These are
              extensions of the same principles.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Same Question, Every Time">
            &ldquo;What is the bottleneck?&rdquo; Memory &rarr; ZeRO or tensor
            parallelism. Depth &rarr; pipeline parallelism. Data throughput
            &rarr; data parallelism. Latency &rarr; speculative decoding. GPU
            utilization &rarr; continuous batching. The answer determines the
            tool.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Practice: Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Calculation and analysis—no multi-GPU hardware needed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Four exercises built around calculation and analysis. Distributed
              training requires multi-GPU hardware, but the reasoning behind
              every decision can be explored with arithmetic and simulation on a
              single machine.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Training Memory Calculator (Guided)" color="blue">
                <p className="text-sm">
                  Build a function that computes training memory for any model
                  config (num_params, precision, optimizer). Apply to GPT-2
                  (124M), LLaMA 7B, LLaMA 70B. Compare single-GPU requirement
                  to A100 capacity. Calculate ZeRO Stage 1 savings across 1, 2,
                  4, 8 GPUs. Predict: &ldquo;Will ZeRO Stage 1 alone make 70B
                  fit on 8 A100s?&rdquo;
                </p>
              </GradientCard>
              <GradientCard title="Exercise 2: Speculative Decoding Simulation (Supported)" color="blue">
                <p className="text-sm">
                  Simulate the draft-and-verify mechanism with configurable
                  match probability. Model the draft model&rsquo;s per-token
                  agreement rate, measure how acceptance rate changes with draft
                  length K=1 through 8, and find the optimal draft length that
                  balances draft overhead against expected accepted tokens. See
                  why the speedup comes from parallel verification, not from
                  the draft model being fast.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 3: Continuous Batching Simulation (Supported)" color="blue">
                <p className="text-sm">
                  Simulate an inference server: 50 requests with varying target
                  lengths (mean 80, std 60). Implement static batching
                  (batch_size=8) and continuous batching. Measure total time,
                  average latency, GPU utilization. Plot utilization over time
                  for both strategies.
                </p>
              </GradientCard>
              <GradientCard title="Exercise 4: Parallelism Strategy Advisor (Independent)" color="blue">
                <p className="text-sm">
                  Given model size, GPU count, and GPU memory, determine which
                  parallelism combination is needed. Build a function that
                  outputs: data parallelism alone? Tensor parallelism degree?
                  Pipeline stages? ZeRO stage recommendation? Test on 5
                  configurations from GPT-2 to a hypothetical 175B model.
                </p>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises. Progression:
                  memory arithmetic &rarr; speculative decoding simulation
                  &rarr; continuous batching simulation &rarr; parallelism
                  strategy design.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/5-3-3-training-and-serving-at-scale.ipynb"
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
          <TipBlock title="No Multi-GPU Required">
            All four exercises run on a single Colab GPU (or CPU for the
            simulation exercises). The memory calculator is pure arithmetic.
            The speculative decoding exercise simulates draft-and-verify
            with configurable match probabilities. The batching simulation
            is a Python simulation. The strategy advisor
            is logic and math.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline:
                  'Three parallelism strategies distribute training across multiple GPUs.',
                description:
                  'Data parallelism splits data and replicates the model. Tensor parallelism splits weight matrices within layers. Pipeline parallelism assigns different layers to different GPUs. Each addresses a different bottleneck; frontier models combine all three.',
              },
              {
                headline:
                  'Communication overhead is the constraint that shapes every parallelism decision.',
                description:
                  'GPU compute vastly exceeds inter-GPU bandwidth. Every parallelism strategy trades off computation distribution against communication cost. The delivery truck analogy from Scaling & Efficiency applies between GPUs, not just within one.',
              },
              {
                headline:
                  'ZeRO reduces memory redundancy by sharding optimizer states across GPUs.',
                description:
                  'Optimizer states are two-thirds of training memory. ZeRO Stage 1 shards them across GPUs, cutting per-GPU memory without changing the training algorithm. Combined with tensor and pipeline parallelism, it makes 70B+ model training feasible.',
              },
              {
                headline:
                  'Speculative decoding turns sequential generation into parallel verification.',
                description:
                  'A small draft model generates K candidate tokens. The large model verifies all K in one forward pass. Accepted tokens are produced at the cost of one large forward pass instead of K separate passes. Output quality is identical to the large model alone.',
              },
              {
                headline:
                  'Continuous batching eliminates wasted compute from static batching.',
                description:
                  'When a request completes, its slot is immediately filled from the queue. GPU utilization stays near 100% instead of declining as short requests finish. The improvement is in slot utilization, not batch size.',
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
              The math is elegant; the engineering makes it work. Every
              technique in this lesson is an answer to the same question: how
              do you distribute work across devices when communication is
              expensive? Data parallelism distributes data. Tensor parallelism
              distributes computation. Pipeline parallelism distributes layers.
              ZeRO distributes memory. Speculative decoding distributes
              generation steps. Continuous batching distributes serving
              capacity. The bottleneck determines the solution.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Module Completion
          ================================================================ */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="5.3"
            title="Scaling Architecture"
            achievements={[
              'MoE: decoupling parameters from per-token computation via conditional activation',
              'Long context: RoPE for position, sparse attention for compute, GQA for KV cache memory',
              'Distributed training: data, tensor, and pipeline parallelism + ZeRO optimizer sharding',
              'Inference serving: speculative decoding for latency, continuous batching for throughput',
              'The connecting thread: identify the bottleneck, apply the targeted solution',
            ]}
            nextModule="6.1"
            nextTitle="Classical Diffusion"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          14. References
          ================================================================ */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
                authors: 'Shoeybi et al., 2019',
                url: 'https://arxiv.org/abs/1909.08053',
                note: 'Introduces tensor parallelism for transformer models. Section 3 describes the column-wise and row-wise splitting of attention and FFN layers across GPUs.',
              },
              {
                title: 'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism',
                authors: 'Huang et al., 2019',
                url: 'https://arxiv.org/abs/1811.06965',
                note: 'Pipeline parallelism with micro-batching. Section 3 covers the pipeline schedule and bubble analysis discussed in this lesson.',
              },
              {
                title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
                authors: 'Rajbhandari et al., 2020',
                url: 'https://arxiv.org/abs/1910.02054',
                note: 'The ZeRO paper. Section 3 describes the three stages of optimizer state, gradient, and parameter sharding. The concrete memory savings match our lesson arithmetic.',
              },
              {
                title: 'Fast Inference from Transformers via Speculative Decoding',
                authors: 'Leviathan et al., 2023',
                url: 'https://arxiv.org/abs/2211.17192',
                note: 'Formalizes speculative decoding. Section 2 describes the draft-verify mechanism and proves that output quality is identical to the large model alone.',
              },
              {
                title: 'Orca: A Distributed Serving System for Transformer-Based Generative Models',
                authors: 'Yu et al., 2022',
                url: 'https://www.usenix.org/conference/osdi22/presentation/yu',
                note: 'Introduces iteration-level scheduling (continuous batching) for LLM serving. Section 3 covers the slot management mechanism.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          15. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Module 5.3 Complete: Scaling Architecture"
            description="You have traced the full arc of scaling: from the dense transformer's bottlenecks through architectural innovations (MoE, efficient attention) to the engineering that makes frontier models possible. Every technique addresses a specific bottleneck. The bottleneck determines the solution. Next: a new domain—how diffusion models generate images from noise."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
