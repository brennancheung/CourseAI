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
} from '@/components/lessons'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Scaling & Efficiency
 *
 * Third lesson in Module 4.3 (Building & Training GPT).
 * Twelfth lesson in Series 4 (LLMs & Transformers).
 *
 * Identifies the key computational bottlenecks in transformer
 * training and inference, and the engineering solutions that make
 * modern LLMs practical. This is a BUILD lesson: three genuinely
 * new concepts plus two reinforced/extended concepts.
 *
 * Core concepts at DEVELOPED:
 * - Mixed precision / bfloat16 (extends INTRODUCED from Series 2)
 * - KV caching for autoregressive inference
 *
 * Core concepts at INTRODUCED:
 * - Compute-bound vs memory-bound operations (arithmetic intensity)
 * - Flash attention (the insight, not the implementation)
 * - Scaling laws (Chinchilla-style compute-optimal training)
 *
 * Core concepts at MENTIONED:
 * - torch.compile, continuous batching, speculative decoding, MoE
 *
 * EXPLICITLY NOT COVERED:
 * - Implementing any of these optimizations (no notebook)
 * - Multi-GPU or distributed training
 * - Mixture of experts (MoE) -- MENTIONED only
 * - Specific GPU hardware details beyond the basic insight
 * - Quantization (deferred to Module 4.4)
 * - Inference serving optimizations in depth
 *
 * Previous: Pretraining on Real Text (Module 4.3, Lesson 2)
 * Next: Loading Pretrained Weights (Module 4.3, Lesson 4)
 */

// ---------------------------------------------------------------------------
// KV Cache Growth Diagram (inline SVG)
// Shows how the cache grows at each generation step
// ---------------------------------------------------------------------------

function KvCacheDiagram() {
  const steps = [
    { label: 'Step 1', prompt: 3, generated: 0, newToken: 1 },
    { label: 'Step 2', prompt: 3, generated: 1, newToken: 1 },
    { label: 'Step 3', prompt: 3, generated: 2, newToken: 1 },
    { label: 'Step 4', prompt: 3, generated: 3, newToken: 1 },
  ]
  const cellW = 36
  const cellH = 28
  const labelW = 54
  const rowH = 44
  const startY = 50

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={460}
        height={startY + steps.length * rowH + 40}
        viewBox={`0 0 460 ${startY + steps.length * rowH + 40}`}
        className="overflow-visible"
      >
        {/* Title */}
        <text
          x={230}
          y={18}
          textAnchor="middle"
          fill="#e2e8f0"
          fontSize="13"
          fontWeight="600"
        >
          KV Cache Growth During Generation
        </text>

        {/* Column headers */}
        <text x={labelW + cellW * 1.5} y={40} textAnchor="middle" fill="#9ca3af" fontSize="9">
          Prompt (cached)
        </text>
        <text x={labelW + cellW * 4.5} y={40} textAnchor="middle" fill="#9ca3af" fontSize="9">
          Previously generated (cached)
        </text>
        <text x={labelW + cellW * 8} y={40} textAnchor="middle" fill="#34d399" fontSize="9" fontWeight="600">
          New
        </text>

        {steps.map((step, row) => {
          const y = startY + row * rowH
          const elements: React.ReactElement[] = []

          // Row label
          elements.push(
            <text
              key={`label-${row}`}
              x={labelW - 6}
              y={y + cellH / 2 + 4}
              textAnchor="end"
              fill="#9ca3af"
              fontSize="10"
              fontWeight="500"
            >
              {step.label}
            </text>,
          )

          // Prompt tokens (cached, solid blue)
          for (let i = 0; i < step.prompt; i++) {
            elements.push(
              <rect
                key={`prompt-${row}-${i}`}
                x={labelW + i * cellW}
                y={y}
                width={cellW - 2}
                height={cellH}
                rx={3}
                fill="#6366f1"
                opacity={0.25}
                stroke="#6366f1"
                strokeWidth={1}
              />,
            )
            elements.push(
              <text
                key={`prompt-text-${row}-${i}`}
                x={labelW + i * cellW + (cellW - 2) / 2}
                y={y + cellH / 2 + 4}
                textAnchor="middle"
                fill="#6366f1"
                fontSize="8"
              >
                K,V
              </text>,
            )
          }

          // Previously generated tokens (cached, amber)
          for (let i = 0; i < step.generated; i++) {
            const x = labelW + (step.prompt + i) * cellW
            elements.push(
              <rect
                key={`gen-${row}-${i}`}
                x={x}
                y={y}
                width={cellW - 2}
                height={cellH}
                rx={3}
                fill="#f59e0b"
                opacity={0.2}
                stroke="#f59e0b"
                strokeWidth={1}
              />,
            )
            elements.push(
              <text
                key={`gen-text-${row}-${i}`}
                x={x + (cellW - 2) / 2}
                y={y + cellH / 2 + 4}
                textAnchor="middle"
                fill="#f59e0b"
                fontSize="8"
              >
                K,V
              </text>,
            )
          }

          // New token (highlighted green)
          const newX = labelW + (step.prompt + step.generated) * cellW
          elements.push(
            <rect
              key={`new-${row}`}
              x={newX}
              y={y}
              width={cellW - 2}
              height={cellH}
              rx={3}
              fill="#34d399"
              opacity={0.3}
              stroke="#34d399"
              strokeWidth={2}
            />,
          )
          elements.push(
            <text
              key={`new-text-${row}`}
              x={newX + (cellW - 2) / 2}
              y={y + cellH / 2 + 4}
              textAnchor="middle"
              fill="#34d399"
              fontSize="7"
              fontWeight="600"
            >
              Q,K,V
            </text>,
          )

          // Arrow pointing to new token
          elements.push(
            <text
              key={`arrow-${row}`}
              x={newX + cellW + 8}
              y={y + cellH / 2 + 4}
              fill="#34d399"
              fontSize="9"
            >
              {'\u2190'} compute only this
            </text>,
          )

          return elements
        })}

        {/* Summary label */}
        <text
          x={230}
          y={startY + steps.length * rowH + 25}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="10"
        >
          New token: compute Q, K, V. Cache the new K, V. Use Q for attention over cached K, V.
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Flash Attention Memory Comparison Diagram (inline SVG)
// ---------------------------------------------------------------------------

function FlashAttentionDiagram() {
  const blockSize = 28
  const gridSize = 5
  const matrixW = blockSize * gridSize
  const gap = 80

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={matrixW * 2 + gap + 120}
        height={matrixW + 80}
        viewBox={`0 0 ${matrixW * 2 + gap + 120} ${matrixW + 80}`}
        className="overflow-visible"
      >
        {/* Standard Attention */}
        <text
          x={60 + matrixW / 2}
          y={16}
          textAnchor="middle"
          fill="#e2e8f0"
          fontSize="11"
          fontWeight="600"
        >
          Standard Attention
        </text>
        <text
          x={60 + matrixW / 2}
          y={30}
          textAnchor="middle"
          fill="#ef4444"
          fontSize="9"
        >
          Full n{'\u00d7'}n matrix in GPU memory
        </text>

        {/* Full matrix (all cells filled = red/hot) */}
        {Array.from({ length: gridSize }).map((_, i) =>
          Array.from({ length: gridSize }).map((_, j) => (
            <rect
              key={`std-${i}-${j}`}
              x={60 + j * blockSize}
              y={40 + i * blockSize}
              width={blockSize - 2}
              height={blockSize - 2}
              rx={2}
              fill="#ef4444"
              opacity={0.25}
              stroke="#ef4444"
              strokeWidth={0.8}
            />
          )),
        )}
        <text
          x={60 + matrixW / 2}
          y={40 + matrixW + 18}
          textAnchor="middle"
          fill="#ef4444"
          fontSize="9"
        >
          O(n{'\u00b2'}) memory
        </text>

        {/* Arrow between */}
        <text
          x={60 + matrixW + gap / 2}
          y={40 + matrixW / 2 + 4}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="16"
        >
          {'\u2192'}
        </text>

        {/* Flash Attention */}
        <text
          x={60 + matrixW + gap + matrixW / 2}
          y={16}
          textAnchor="middle"
          fill="#e2e8f0"
          fontSize="11"
          fontWeight="600"
        >
          Flash Attention
        </text>
        <text
          x={60 + matrixW + gap + matrixW / 2}
          y={30}
          textAnchor="middle"
          fill="#34d399"
          fontSize="9"
        >
          Tiled blocks, never store full matrix
        </text>

        {/* Tiled matrix (only a few cells highlighted = green) */}
        {Array.from({ length: gridSize }).map((_, i) =>
          Array.from({ length: gridSize }).map((_, j) => {
            const isActive = (i === 1 && j === 2) || (i === 2 && j === 2)
            return (
              <rect
                key={`flash-${i}-${j}`}
                x={60 + matrixW + gap + j * blockSize}
                y={40 + i * blockSize}
                width={blockSize - 2}
                height={blockSize - 2}
                rx={2}
                fill={isActive ? '#34d399' : '#374151'}
                opacity={isActive ? 0.35 : 0.15}
                stroke={isActive ? '#34d399' : '#374151'}
                strokeWidth={isActive ? 1.5 : 0.5}
              />
            )
          }),
        )}

        {/* Highlight annotation */}
        <text
          x={60 + matrixW + gap + matrixW + 8}
          y={40 + 2 * blockSize + 4}
          fill="#34d399"
          fontSize="8"
        >
          {'\u2190'} current tile
        </text>

        <text
          x={60 + matrixW + gap + matrixW / 2}
          y={40 + matrixW + 18}
          textAnchor="middle"
          fill="#34d399"
          fontSize="9"
        >
          O(n) memory
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Scaling Laws Sketch (inline SVG)
// Shows loss vs compute with iso-parameter curves
// ---------------------------------------------------------------------------

function ScalingLawsDiagram() {
  const plotX = 60
  const plotY = 30
  const plotW = 360
  const plotH = 180

  // Generate a power-law-ish curve (loss = a * compute^(-b))
  // Y-axis: high loss at top, low loss at bottom (standard convention)
  // Curves go from upper-left (high loss, low compute) to lower-right (low loss, high compute)
  function lossCurve(
    points: number,
    floorFraction: number,
    exponent: number,
  ): string {
    const coords: string[] = []
    for (let i = 1; i <= points; i++) {
      const t = i / points
      const x = plotX + t * plotW
      // Loss decreases with compute: starts high (top), decays toward a floor
      // floorFraction controls where the curve plateaus (0 = bottom, 1 = top)
      const lossNormalized = floorFraction + (1 - floorFraction) * Math.pow(t, -exponent) * 0.3
      // Map to plot coordinates: high loss = top (plotY), low loss = bottom (plotY + plotH)
      const y = plotY + (1 - Math.min(lossNormalized, 1)) * plotH
      const clampedY = Math.min(Math.max(y, plotY), plotY + plotH)
      coords.push(`${x},${clampedY}`)
    }
    return `M ${coords.join(' L ')}`
  }

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={plotX + plotW + 100}
        height={plotY + plotH + 50}
        viewBox={`0 0 ${plotX + plotW + 100} ${plotY + plotH + 50}`}
        className="overflow-visible"
      >
        {/* Axes */}
        <line
          x1={plotX}
          y1={plotY}
          x2={plotX}
          y2={plotY + plotH}
          stroke="#374151"
          strokeWidth={1}
        />
        <line
          x1={plotX}
          y1={plotY + plotH}
          x2={plotX + plotW}
          y2={plotY + plotH}
          stroke="#374151"
          strokeWidth={1}
        />

        {/* Axis labels */}
        <text
          x={plotX + plotW / 2}
          y={plotY + plotH + 35}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="10"
        >
          Compute (FLOPs) {'\u2192'}
        </text>
        <text
          x={15}
          y={plotY + plotH / 2}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize="10"
          transform={`rotate(-90, 15, ${plotY + plotH / 2})`}
        >
          {'\u2190'} Loss
        </text>

        {/* Y-axis tick labels for orientation */}
        <text x={plotX - 6} y={plotY + 8} textAnchor="end" fill="#6b7280" fontSize="8">
          High
        </text>
        <text x={plotX - 6} y={plotY + plotH} textAnchor="end" fill="#6b7280" fontSize="8">
          Low
        </text>

        {/* Iso-parameter curves (small, medium, large model) */}
        {/* Small model: plateaus at higher loss (higher floor) */}
        <path
          d={lossCurve(80, 0.55, 0.15)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={1.5}
          opacity={0.6}
        />
        <text
          x={plotX + plotW + 4}
          y={plotY + plotH - 50}
          fill="#f59e0b"
          fontSize="8"
          opacity={0.7}
        >
          Small model
        </text>

        {/* Medium model: plateaus at middle loss */}
        <path
          d={lossCurve(80, 0.35, 0.2)}
          fill="none"
          stroke="#a78bfa"
          strokeWidth={1.5}
          opacity={0.6}
        />
        <text
          x={plotX + plotW + 4}
          y={plotY + plotH - 85}
          fill="#a78bfa"
          fontSize="8"
          opacity={0.7}
        >
          Medium model
        </text>

        {/* Large model: plateaus at lowest loss (lowest floor) */}
        <path
          d={lossCurve(80, 0.15, 0.25)}
          fill="none"
          stroke="#38bdf8"
          strokeWidth={1.5}
          opacity={0.6}
        />
        <text
          x={plotX + plotW + 4}
          y={plotY + plotH - 120}
          fill="#38bdf8"
          fontSize="8"
          opacity={0.7}
        >
          Large model
        </text>

        {/* Chinchilla frontier (dashed, upper-left to lower-right) */}
        <path
          d={`M ${plotX + 30},${plotY + 20} L ${plotX + 120},${plotY + 60} L ${plotX + 250},${plotY + 100} L ${plotX + plotW - 20},${plotY + 130}`}
          fill="none"
          stroke="#34d399"
          strokeWidth={2}
          strokeDasharray="6,4"
        />
        <text
          x={plotX + plotW + 4}
          y={plotY + plotH - 150}
          fill="#34d399"
          fontSize="9"
          fontWeight="600"
        >
          Compute-optimal
        </text>
        <text
          x={plotX + plotW + 4}
          y={plotY + plotH - 140}
          fill="#34d399"
          fontSize="9"
          fontWeight="600"
        >
          frontier
        </text>

        {/* Annotation: each curve plateaus */}
        <text
          x={plotX + plotW - 40}
          y={plotY + plotH - 10}
          textAnchor="middle"
          fill="#6b7280"
          fontSize="8"
        >
          Each model size eventually plateaus
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Lesson Component
// ---------------------------------------------------------------------------

export function ScalingAndEfficiencyLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Scaling & Efficiency"
            description="Your GPT works&mdash;but it&rsquo;s slow. Understand the engineering layer between elegant math and real systems."
            category="Training"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Identify the key computational bottlenecks in transformer training
            and inference, and understand the engineering solutions&mdash;mixed
            precision, KV caching, flash attention&mdash;and the empirical
            principles (scaling laws) that make modern LLMs practical.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="No Code Today">
            This is a conceptual lesson. No notebook, no implementation. You
            are building the mental framework for <em>why</em> these
            techniques exist, not practicing the <em>how</em>. Every
            technique here solves a problem you can now feel because you
            trained your own model.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why transformer training and inference are slow (concrete bottlenecks)',
              'Compute-bound vs memory-bound operations (the fundamental framework)',
              'Mixed precision with bfloat16 (extending float16/GradScaler from Series 2)',
              'KV caching for autoregressive inference (mechanism and why it\u2019s essential)',
              'Flash attention (the insight\u2014tiled computation\u2014not the implementation)',
              'Scaling laws (Chinchilla: how to allocate compute between model size and data)',
              'NOT: implementing any of these optimizations (no notebook)',
              'NOT: multi-GPU or distributed training',
              'NOT: quantization\u2014deferred to Module 4.4',
              'NOT: specific GPU hardware details beyond the basic insight',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          3. Hook: "Where Does the Time Go?"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where Does the Time Go?"
            subtitle="From your training run to GPT-3"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your training run on TinyShakespeare took minutes. Each step
              processed a batch and updated weights. You watched the loss
              drop and the generated text improve. It worked.
            </p>

            <p className="text-muted-foreground">
              Now consider the scale difference. GPT-3 trained on{' '}
              <strong>300 billion tokens</strong>. Your dataset had ~300,000.
              That is a million-to-one ratio. At your training speed, GPT-3
              would take{' '}
              <strong>centuries</strong>.
            </p>

            <p className="text-muted-foreground">
              And remember your <code className="text-xs">generate()</code>{' '}
              method from Building nanoGPT? It calls the full forward pass for
              every single token. At step 50 of generation, it recomputes K
              and V for all 50 positions even though only position 50 is new.
              At sequence length 1024, that is 1024x the necessary work.
            </p>

            <p className="text-muted-foreground">
              This lesson is about <strong>why it was slow</strong> and{' '}
              <strong>what the real engineering looks like</strong>. Not
              premature optimization on a toy model&mdash;the problems that
              emerge at real scale, and the solutions that make the difference
              between a model that trains in a day and one that never finishes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Problem Before Solution">
            Every technique in this lesson starts with a specific
            bottleneck you can now feel from your own training experience.
            The pattern is always the same: name the problem, understand
            why it is hard, then see the solution.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain: GPU Utilization -- Compute-Bound vs Memory-Bound
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Compute-Bound vs Memory-Bound"
            subtitle="The GPU is faster than its own memory"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In GPU Training, you learned that GPUs have thousands of simple
              cores for parallel computation. But having fast chefs does not
              help if the delivery truck is slow.
            </p>

            <p className="text-muted-foreground">
              A modern GPU (like an A100) can perform{' '}
              <strong>312 trillion float16 operations per second</strong>{' '}
              (312 TFLOPS). But it can only transfer data from its main
              memory (HBM) at <strong>2 TB/s</strong>. The compute engine
              is overwhelmingly faster than the memory bus.
            </p>

            <p className="text-muted-foreground">
              This creates two regimes for every operation:
            </p>

            <ComparisonRow
              left={{
                title: 'Memory-Bound',
                color: 'amber',
                items: [
                  'Simple operations: element-wise add, layer norm, softmax',
                  'Very few FLOPs per element',
                  'GPU finishes the math before data arrives from memory',
                  'Bottleneck: memory bandwidth, not compute',
                  'Making the GPU faster does not help',
                ],
              }}
              right={{
                title: 'Compute-Bound',
                color: 'blue',
                items: [
                  'Matrix multiplication: the core of attention and FFN',
                  'Many FLOPs per element',
                  'Enough work to keep the GPU busy while data streams in',
                  'Bottleneck: compute throughput',
                  'Faster GPU = faster operation',
                ],
              }}
            />

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Concrete example: two operations on vectors of length N
              </p>
              <div className="grid gap-4 md:grid-cols-2 text-sm text-muted-foreground">
                <div className="space-y-1">
                  <p className="font-medium text-amber-400">
                    Vector addition (memory-bound)
                  </p>
                  <p>
                    Compute: N additions = N FLOPs
                  </p>
                  <p>
                    Memory: read 2 vectors, write 1 = 3N transfers
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    Arithmetic intensity: N / 3N = 0.33 FLOPs/byte
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    The GPU finishes N additions instantly but waits for 3N
                    memory reads/writes.
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="font-medium text-sky-400">
                    Matrix multiplication (compute-bound)
                  </p>
                  <p>
                    Compute:{' '}
                    <InlineMath math="O(n^3)" /> FLOPs for n{'\u00d7'}n
                    matrices
                  </p>
                  <p>
                    Memory: read 2 matrices, write 1 ={' '}
                    <InlineMath math="O(n^2)" /> transfers
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    Arithmetic intensity:{' '}
                    <InlineMath math="O(n^3) / O(n^2) = O(n)" /> FLOPs/byte
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    Grows with matrix size. Large matmuls keep the GPU
                    saturated.
                  </p>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              <strong>Arithmetic intensity</strong> = FLOPs per byte
              transferred. Low arithmetic intensity means memory-bound. High
              means compute-bound. The training loop is a mix of both:
              attention and FFN are compute-bound (matrix multiplications),
              while layer norm, dropout, and activation functions are
              memory-bound (simple element-wise operations).
            </p>

            <p className="text-muted-foreground">
              Here is the misconception to let go of: your mental model
              from Series 2&mdash;&ldquo;GPUs are fast because they have
              thousands of cores&rdquo;&mdash;implies that{' '}
              <strong>more cores always means faster</strong>. It does not.
              For memory-bound operations like layer norm and softmax,
              making the GPU faster does <em>nothing</em>. The GPU is
              already done computing; it is waiting for data to arrive from
              memory. And most of the operations in a transformer (everything
              except matrix multiplication) are memory-bound. The bottleneck
              is the delivery truck, not the chefs.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Kitchen Analogy">
            Imagine a kitchen with incredibly fast chefs but a slow delivery
            truck. For complex dishes (matrix multiplication), the chefs stay
            busy while ingredients trickle in. For simple tasks like chopping
            (element-wise ops), the chefs stand idle waiting for the next
            delivery. The truck is the GPU&rsquo;s memory bandwidth.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Explain: Mixed Precision -- bfloat16
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Mixed Precision and bfloat16"
            subtitle="The precision format transformers actually use"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In GPU Training, you saw mixed precision: float16 forward
              passes are faster and use less memory, but gradients can
              underflow in float16 (the &ldquo;micrometer to ruler&rdquo;
              analogy). You added{' '}
              <code className="text-xs">autocast</code> and{' '}
              <code className="text-xs">GradScaler</code>&mdash;four lines
              that handled the complexity. That foundation is correct, but
              transformer training at scale uses a different format.
            </p>

            <p className="text-muted-foreground">
              <strong>bfloat16</strong> (brain floating point 16) keeps the
              same <strong>exponent range</strong> as float32 but reduces the
              <strong> mantissa</strong> (precision bits) to fit in 16 bits.
              Compare:
            </p>

            <div className="px-4 py-3 bg-muted/50 rounded-lg">
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-muted-foreground">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Format</th>
                      <th className="text-right py-2 px-3">Bits</th>
                      <th className="text-right py-2 px-3">Exponent</th>
                      <th className="text-right py-2 px-3">Mantissa</th>
                      <th className="text-right py-2 px-3">Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 text-sky-400">float32</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">32</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">8</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">23</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">{'\u00b1'}3.4e38</td>
                    </tr>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 text-amber-400">float16</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">16</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">5</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">10</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">{'\u00b1'}6.5e4</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-4 text-emerald-400 font-medium">bfloat16</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">16</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">8</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">7</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">{'\u00b1'}3.4e38</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <p className="text-muted-foreground">
              bfloat16 has the <strong>same range as float32</strong> (8-bit
              exponent) but <strong>less precision</strong> (7-bit mantissa
              vs 23-bit). This means gradients with very small magnitudes
              like 1e-8 do not underflow to zero&mdash;they can be
              represented, just with less precision. No GradScaler needed.
            </p>

            <p className="text-muted-foreground">
              Extending the precision analogy from GPU Training:
              float64 is a micrometer, float32 is a ruler, float16 is a tape
              measure that cannot read very small or very large values.
              bfloat16 is a <strong>different kind of ruler</strong>&mdash;it
              can measure the same range as float32 but with coarser
              markings.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Range, Less Precision">
            bfloat16 was designed by Google Brain specifically for deep
            learning. The insight: for neural network training, the{' '}
            <strong>range</strong> of representable numbers matters more than
            fine precision. Gradients span many orders of magnitude, but each
            individual gradient does not need 23 bits of precision.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: everything in float16 */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              There is a subtle but important distinction here. bfloat16
              can <strong>represent</strong> very small numbers like 1e-8
              (it has the range). But it cannot <strong>add</strong> 1e-8
              to 1.0 and keep the difference&mdash;it lacks the precision.
              The problem is not underflow of the gradient itself, but
              precision loss when a small gradient is added to a much
              larger parameter value:
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Why &ldquo;mixed&rdquo; is essential: the weight update
                example
              </p>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  Parameter ={' '}
                  <span className="font-mono">1.0000</span>, gradient ={' '}
                  <span className="font-mono">0.0001</span>
                </p>
                <p>
                  <strong className="text-sky-400">float32:</strong>{' '}
                  <span className="font-mono">
                    1.0000 + 0.0001 = 1.0001
                  </span>{' '}
                  {'\u2714'} correct
                </p>
                <p>
                  <strong className="text-amber-400">float16:</strong>{' '}
                  <span className="font-mono">1.0 + 0.0001 = 1.0</span>{' '}
                  {'\u2718'} gradient lost
                </p>
                <p>
                  <strong className="text-emerald-400">bfloat16:</strong>{' '}
                  <span className="font-mono">
                    1.0 + 0.0001 = 1.0
                  </span>{' '}
                  {'\u2718'} also lost (same 16-bit storage)
                </p>
              </div>
              <p className="text-xs text-muted-foreground/80">
                The parameter is 1.0 and the gradient is 0.0001. In
                float32, the addition preserves the small change. In
                both 16-bit formats, the gradient is too small relative to the
                parameter value to survive the addition. The weight
                never changes. Training stalls.
              </p>
            </div>

            <p className="text-muted-foreground">
              This is why the &ldquo;mixed&rdquo; in mixed precision is
              essential. The <strong>master weights</strong> pattern:
              keep a float32 copy of the weights for the accumulation step
              (where tiny gradients must survive addition to large
              parameters). Cast to bfloat16 for the forward and backward
              passes (where the speed and memory savings matter, and the
              reduced precision is tolerable). The forward pass computes
              approximate gradients in bfloat16. The update step accumulates
              those gradients precisely in float32.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not &ldquo;Just Use Less Precision&rdquo;">
            Mixed precision does not mean doing everything in 16-bit.
            The forward/backward passes run in bfloat16 for speed. The
            weight updates happen in float32 for correctness. The{' '}
            &ldquo;mix&rdquo; is the entire point&mdash;each format is used
            where its strengths matter.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Check: Predict What Breaks
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Predict What Breaks" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>1.</strong> Your colleague proposes doing the entire
                training loop in float16&mdash;forward pass, backward pass,{' '}
                <em>and weight updates</em>. What goes wrong?
              </p>
              <p>
                <strong>2.</strong> What about bfloat16 for everything,
                including weight accumulation?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>1.</strong> In float16, small gradients
                    (anything below ~1e-4 relative to the parameter value)
                    are lost during weight updates. The parameter never
                    changes. Training stalls, especially for parameters that
                    receive consistently small gradients. Additionally,
                    float16&rsquo;s limited range means values above 65,504
                    overflow to infinity.
                  </p>
                  <p>
                    <strong>2.</strong> bfloat16 avoids the overflow
                    problem (same range as float32) but still has only 7
                    mantissa bits. Accumulating thousands of tiny gradient
                    updates in bfloat16 loses precision cumulatively. After
                    many steps, the parameter drifts from what the float32
                    accumulation would produce. Master weights in float32
                    are still needed for precise accumulation.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          7. Explain: KV Caching
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="KV Caching"
            subtitle="Stop recomputing what hasn&rsquo;t changed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the{' '}
              <code className="text-xs">generate()</code> method from
              Building nanoGPT? It appends each new token and calls the
              full forward pass on the growing sequence. Look at what
              happens step by step:
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-2">
              <div className="text-sm text-muted-foreground space-y-1">
                <p>
                  <strong>Step 1:</strong> Forward pass over 10-token
                  prompt. Compute K, V for all 10 positions.
                </p>
                <p>
                  <strong>Step 2:</strong> Forward pass over 11 tokens. K
                  and V for positions 1&ndash;10 are{' '}
                  <strong>identical to Step 1</strong>. Only position 11 is
                  new. But <code className="text-xs">generate()</code>{' '}
                  recomputes all 11.
                </p>
                <p>
                  <strong>Step 3:</strong> Forward pass over 12 tokens. K
                  and V for 1&ndash;11 are identical. Only position 12 is
                  new. Recomputes all 12.
                </p>
                <p className="text-muted-foreground/70 italic">
                  At step t, we recompute t&minus;1 positions of K and V that
                  have not changed.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              <strong>The fix:</strong> cache the K and V tensors. At each
              generation step:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Compute Q, K, V <strong>only for the new token</strong>
              </li>
              <li>
                Concatenate new K and V with the cached K and V from
                previous steps
              </li>
              <li>
                Compute attention: new Q against all cached K
              </li>
              <li>
                Output: weighted sum of all cached V
              </li>
            </ul>

            <KvCacheDiagram />

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Cost comparison: total token positions processed during generation of T tokens from a P-token prompt
              </p>
              <div className="grid gap-4 md:grid-cols-2 text-sm text-muted-foreground">
                <div className="space-y-1">
                  <p className="font-medium text-rose-400">Without KV cache</p>
                  <p>
                    <InlineMath math="\sum_{t=1}^{T} (P + t)" />
                  </p>
                  <p>
                    = <InlineMath math="T \cdot P + \frac{T(T+1)}{2}" />
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    P=10, T=100: ~5,550 token positions processed
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="font-medium text-emerald-400">With KV cache</p>
                  <p>
                    <InlineMath math="T" /> new-token forward passes
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    P=10, T=100: 100 forward passes (one per token)
                  </p>
                  <p className="text-xs text-emerald-400/70 font-medium">
                    ~55x fewer operations
                  </p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground/80">
                At sequence length 1000: ~500,000 vs 1000. A{' '}
                <strong>500x difference</strong>. This is not an optional
                optimization&mdash;it is the standard approach for any
                production LLM.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course It Recomputes">
            Look at your <code className="text-xs">generate()</code>{' '}
            code from Building nanoGPT. It calls{' '}
            <code className="text-xs">self.forward(x)</code> with the
            full sequence each time. The K and V for all previous tokens
            have not changed. Why recompute them?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check: KV Cache Memory Tradeoff
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: KV Cache Memory" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                KV caching trades compute for memory. For each layer, you
                store K and V tensors of shape{' '}
                <InlineMath math="(\text{batch}, \text{heads}, \text{seq\_len}, d_k)" />.
              </p>
              <p>
                For GPT-2 (12 layers, 12 heads,{' '}
                <InlineMath math="d_k = 64" />): how much memory does the KV
                cache use at sequence length 1024 in float16?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    12 layers {'\u00d7'} 2 tensors (K and V) {'\u00d7'} 12
                    heads {'\u00d7'} 1024 positions {'\u00d7'} 64{' '}
                    {'\u00d7'} 2 bytes (float16) ={' '}
                    <strong>~37.7 MB per sequence</strong>.
                  </p>
                  <p>
                    For a batch of 32: ~1.2 GB. Not trivial. This is why
                    long-context models need more GPU memory even during
                    inference&mdash;the KV cache for a 128K context window
                    is substantial.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Compute vs Memory">
            KV caching eliminates redundant computation but uses memory that
            grows with sequence length. This tradeoff is fundamental:
            longer sequences need more cache memory, which is why
            long-context models are expensive to serve.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Explain: Flash Attention
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Flash Attention"
            subtitle="Same math, different memory access pattern"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              First, the promise: <strong>flash attention computes the
              exact same result as standard attention</strong>. Not
              approximately. Exactly.{' '}
              <code className="text-xs">torch.allclose</code> returns True.
              It is not a different algorithm&mdash;it is a different
              implementation of the same algorithm.
            </p>

            <p className="text-muted-foreground">
              <strong>The problem:</strong> standard attention materializes
              the full <InlineMath math="n \times n" /> attention matrix. For
              sequence length 4096, that is{' '}
              <InlineMath math="4096^2 = 16.7" /> million entries per head
              per layer. This matrix is:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Computed and stored in GPU memory (HBM)</li>
              <li>Read back for softmax</li>
              <li>Stored again after softmax</li>
              <li>Read back for the V multiplication</li>
            </ul>

            <p className="text-muted-foreground">
              Four trips between GPU compute units and GPU memory for one
              matrix. This is a <strong>memory-bound</strong>{' '}
              operation&mdash;the GPU is waiting on memory access, not on
              computation. And now you have the vocabulary to name that
              bottleneck.
            </p>

            <p className="text-muted-foreground">
              <strong>The insight:</strong> you do not need the full{' '}
              <InlineMath math="n \times n" /> matrix at once. You can{' '}
              <strong>tile</strong> the computation&mdash;process blocks of Q
              against blocks of K and V, accumulating the result without ever
              storing the full matrix. The softmax can be computed
              incrementally (the &ldquo;online softmax&rdquo; trick).
            </p>

            <FlashAttentionDiagram />

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                Concrete memory comparison: sequence length 4096, 12 heads, float16
              </p>
              <div className="grid gap-4 md:grid-cols-2 text-sm text-muted-foreground">
                <div className="space-y-1">
                  <p className="font-medium text-rose-400">Standard attention</p>
                  <p>
                    12 heads {'\u00d7'} 4096{'\u00b2'} {'\u00d7'} 2 bytes
                  </p>
                  <p>
                    = <strong>~384 MB</strong> of attention matrices
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    All stored in GPU memory simultaneously
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="font-medium text-emerald-400">Flash attention (tile size 128)</p>
                  <p>
                    12 heads {'\u00d7'} 128{'\u00b2'} {'\u00d7'} 2 bytes
                  </p>
                  <p>
                    = <strong>~384 KB</strong> working memory
                  </p>
                  <p className="text-xs text-emerald-400/70 font-medium">
                    ~1000x reduction in peak memory
                  </p>
                </div>
              </div>
            </div>

            <div className="px-4 py-3 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium">Result</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>
                  {'\u2022'} 2&ndash;4x faster than standard attention
                </li>
                <li>
                  {'\u2022'}{' '}
                  <InlineMath math="O(n)" /> memory instead of{' '}
                  <InlineMath math="O(n^2)" /> for the attention matrix
                </li>
                <li>
                  {'\u2022'} Fuses the causal mask into the tiled
                  computation (remember the concern from Decoder-Only
                  Transformers about computing the full{' '}
                  <InlineMath math="QK^T" /> and zeroing the upper triangle?
                  Flash attention skips those tiles entirely)
                </li>
                <li>
                  {'\u2022'} Built into PyTorch:{' '}
                  <code className="text-xs">
                    torch.nn.functional.scaled_dot_product_attention
                  </code>{' '}
                  uses flash attention automatically when available
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Different Algorithm">
            Flash attention produces <strong>numerically identical</strong>{' '}
            results to standard attention. The innovation is entirely in
            memory access patterns&mdash;how the GPU reads and writes data,
            not what it computes. If someone tells you flash attention is an
            approximation, they are wrong.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Elaborate: Broader Efficiency Landscape
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Broader Efficiency Landscape"
            subtitle="Names you should recognize"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Mixed precision, KV caching, and flash attention are the core
              techniques. But the efficiency landscape is wider. These are
              real techniques used in production systems&mdash;mentioned here
              so you recognize the names when you encounter them.
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="torch.compile" color="blue">
                <p className="text-sm">
                  Fuses multiple operations into a single GPU kernel,
                  eliminating intermediate memory reads/writes. Think of it
                  as combining multiple delivery trips into one.
                </p>
              </GradientCard>
              <GradientCard title="Continuous Batching" color="cyan">
                <p className="text-sm">
                  Instead of waiting for an entire batch to finish before
                  starting new requests, slot new requests into completed
                  positions. Maximizes GPU utilization during serving.
                </p>
              </GradientCard>
              <GradientCard title="Speculative Decoding" color="violet">
                <p className="text-sm">
                  Use a small, fast model to draft several tokens, then
                  verify them all at once with the large model. Correct
                  drafts are accepted; incorrect ones are regenerated.
                  Amortizes the cost of the large model.
                </p>
              </GradientCard>
              <GradientCard title="Mixture of Experts (MoE)" color="purple">
                <p className="text-sm">
                  Only activate a subset of the model&rsquo;s parameters for
                  each token. A router network decides which &ldquo;expert&rdquo;
                  sub-networks process each input. More total parameters, but
                  the same compute per token.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Recognition, Not Mastery">
            These four techniques are MENTIONED only&mdash;you do not need to
            understand their mechanisms. The goal is name recognition so you
            can follow discussions about LLM efficiency without being lost.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Explain: Scaling Laws
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Scaling Laws"
            subtitle="How to spend your compute budget"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a fixed compute budget&mdash;say, 1000 GPU-hours.
              Should you train a large model for fewer steps, or a small
              model for more steps? In Decoder-Only Transformers, you saw
              that &ldquo;scale, not architecture&rdquo; drove the biggest
              improvements. But <em>how</em> should you scale?
            </p>

            <p className="text-muted-foreground">
              The naive answer: bigger model is always better. Just make
              everything larger. Your own experience suggests otherwise.
            </p>

            <div className="px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg space-y-2">
              <p className="text-sm font-medium text-amber-400">
                The negative example you already experienced
              </p>
              <p className="text-sm text-muted-foreground">
                Your TinyShakespeare training eventually showed validation
                loss diverging from training loss&mdash;the scissors pattern
                from Series 1. A 124M-parameter model on 300K tokens is
                massively overtrained on too little data. More training steps
                would not help. More data would.
              </p>
            </div>

            <p className="text-muted-foreground">
              The <strong>Chinchilla result</strong> (2022) made this
              precise: compute-optimal scaling requires matching model size
              to data quantity. Roughly:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <BlockMath math="N_{\text{opt}} \propto \sqrt{C} \qquad D_{\text{opt}} \propto \sqrt{C}" />
              <p className="text-xs text-muted-foreground text-center">
                <InlineMath math="N" /> = model parameters,{' '}
                <InlineMath math="D" /> = training tokens,{' '}
                <InlineMath math="C" /> = compute budget
              </p>
            </div>

            <p className="text-muted-foreground">
              Both model size and data should scale together with the square
              root of compute. Doubling your compute budget? Increase model
              size by ~{'\u221a'}2 and data by ~{'\u221a'}2.
            </p>

            <div className="px-4 py-4 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium">
                The result that changed LLM training
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-muted-foreground">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Model</th>
                      <th className="text-right py-2 px-3">Parameters</th>
                      <th className="text-right py-2 px-3">Training tokens</th>
                      <th className="text-right py-2 px-3">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border/30">
                      <td className="py-1.5 pr-4 text-rose-400">Gopher</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">280B</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">300B</td>
                      <td className="text-right py-1.5 px-3 text-xs">Undertrained</td>
                    </tr>
                    <tr>
                      <td className="py-1.5 pr-4 text-emerald-400 font-medium">Chinchilla</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">70B</td>
                      <td className="text-right py-1.5 px-3 font-mono text-xs">1.4T</td>
                      <td className="text-right py-1.5 px-3 text-xs text-emerald-400">Outperforms Gopher</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground/80">
                A 4x <em>smaller</em> model with 4.7x more data outperformed
                the larger one. Gopher was undertrained&mdash;too many
                parameters, not enough data to fill them.
              </p>
            </div>

            <p className="text-muted-foreground">
              What this means in practice: most early LLMs were undertrained.
              GPT-3 (175B parameters, 300B tokens) is roughly 5x undertrained
              by Chinchilla standards. LLaMA (65B parameters, 1.4T tokens)
              explicitly followed the Chinchilla recipe and matched GPT-3 at
              one-third the size.
            </p>

            <ScalingLawsDiagram />

            <p className="text-muted-foreground">
              The diagram shows loss vs compute for different model sizes.
              Each curve eventually plateaus&mdash;adding more compute to a
              fixed-size model gives diminishing returns. The{' '}
              <strong>compute-optimal frontier</strong> (dashed green) traces
              the points where scaling both model and data together gives the
              best loss for a given compute budget.
            </p>

            <p className="text-muted-foreground">
              The deeper insight: loss scales as a <strong>power law</strong>{' '}
              with compute, roughly{' '}
              <InlineMath math="L \propto C^{-0.05}" />. What does{' '}
              <InlineMath math="C^{-0.05}" /> mean in practice?{' '}
              <strong>Doubling your compute budget reduces loss by about
              3.4%</strong>. To halve the loss, you need roughly{' '}
              <InlineMath math="2^{1/0.05} \approx" /> 1 million times more
              compute. This is why frontier models cost hundreds of millions
              of dollars&mdash;and why the relationship being predictable is
              so valuable. Labs can estimate the final loss before committing
              the compute.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Scale Both, Not Just One">
            Like building a house: you can hire more workers OR buy better
            materials, but the optimal strategy depends on the ratio. A
            10,000-square-foot foundation with cardboard walls is as
            wasteful as a tiny foundation with marble walls. Scale both
            together.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Check: Scaling Laws Transfer
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Scaling Laws" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Your team has 10x the compute budget of your last training
                run. A colleague suggests training a 10x larger model on the
                same dataset. What does Chinchilla suggest instead?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Roughly a <strong>3x larger model</strong> on{' '}
                    <strong>3x more data</strong> (since{' '}
                    <InlineMath math="\sqrt{10} \approx 3.16" />). Scale
                    both, do not just scale one. A 10x larger model on the
                    same data would be undertrained&mdash;the same mistake
                    as Gopher.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          13. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="The Engineering Layer"
            items={[
              {
                headline:
                  '"The GPU is waiting for data" \u2192 compute-bound vs memory-bound.',
                description:
                  'Not all operations benefit from faster GPUs. Memory-bound operations (layer norm, softmax) are limited by memory bandwidth, not compute. Arithmetic intensity tells you which regime you\u2019re in.',
              },
              {
                headline:
                  '"Half the bits, but gradients vanish" \u2192 mixed precision with bfloat16.',
                description:
                  'bfloat16 preserves float32\u2019s range while using 16 bits. The master weights pattern: float32 for accumulation, bfloat16 for forward/backward. The "mixed" is the essential part.',
              },
              {
                headline:
                  '"Generation recomputes everything" \u2192 KV caching.',
                description:
                  'Cache K and V from previous generation steps. Only compute Q for the new token. Reduces generation cost from O(n\u00b2) to O(n). Not optional at production scale.',
              },
              {
                headline:
                  '"The attention matrix doesn\u2019t fit in fast memory" \u2192 flash attention.',
                description:
                  'Tile the attention computation so the full n\u00d7n matrix is never materialized. Same result, O(n) memory instead of O(n\u00b2). Built into PyTorch.',
              },
              {
                headline:
                  '"How big should the model be?" \u2192 scaling laws.',
                description:
                  'Chinchilla: scale model size and data together. N_opt and D_opt both grow with \u221AC. Most early LLMs were undertrained\u2014too many parameters, not enough data.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Engineering layer mental model */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-violet-500/10 border border-violet-500/20 rounded-lg space-y-2">
            <p className="text-sm font-medium text-violet-400">
              The math is elegant. The engineering makes it work.
            </p>
            <p className="text-sm text-muted-foreground">
              These are not afterthoughts. Mixed precision, KV caching,
              flash attention, and compute-optimal scaling are what
              separates a research prototype from a real system. Every
              technique you learned today is standard practice in any
              production LLM. The architecture from Module 4.2 is the
              blueprint. This lesson is the construction manual.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">
              What comes next
            </p>
            <p className="text-sm text-muted-foreground">
              You understand the architecture, the training, and the
              engineering. One thing remains: validation. In the next
              lesson, you will load OpenAI&rsquo;s actual GPT-2 weights
              into the model you built. If the shapes match and the
              outputs are coherent, your implementation is correct. That is
              the &ldquo;I built GPT&rdquo; moment.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Review the five engineering techniques and how each solves a specific bottleneck. This is conceptual&mdash;no notebook to complete. Reflect on which insights surprised you most."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
