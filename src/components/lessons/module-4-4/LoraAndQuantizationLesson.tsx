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
  PhaseCard,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * LoRA, Quantization & Inference
 *
 * Fourth lesson in Module 4.4 (Beyond Pretraining).
 * STRETCH lesson — two new mathematical ideas with practical payoff.
 *
 * Core concepts at DEVELOPED:
 * - The training memory problem (weights + gradients + optimizer states)
 * - LoRA: low-rank adaptation (architecture, forward pass, where to apply, parameter savings)
 * - Quantization (absmax, zero-point): tracing the process with concrete numbers
 *
 * Core concepts at INTRODUCED:
 * - Low-rank decomposition (factor large matrix into two smaller matrices)
 * - QLoRA (quantized base + LoRA adapters)
 * - KV caching with quantization
 *
 * Core concepts at MENTIONED:
 * - GPTQ, AWQ (post-training quantization methods)
 *
 * EXPLICITLY NOT COVERED:
 * - SVD, eigenvalues, or formal linear algebra beyond rank/decomposition intuition
 * - Implementing LoRA from scratch (notebook uses PEFT library)
 * - Quantization-aware training (QAT)
 * - Mixture of experts, pruning, distillation
 * - Flash attention (covered in 4.3.3)
 * - Production deployment / serving infrastructure
 *
 * Previous: RLHF & Alignment (Module 4.4, Lesson 3)
 * Next: Putting It All Together (Module 4.4, Lesson 5)
 */

// ---------------------------------------------------------------------------
// Inline SVG: LoRA Bypass Diagram
// Shows input x flowing through two paths: frozen W (highway) and B*A (detour).
// Outputs are summed. The detour starts at zero.
// ---------------------------------------------------------------------------

function LoraBypassDiagram() {
  const svgW = 440
  const svgH = 260

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Input x */}
        <rect x={20} y={100} width={70} height={40} rx={5} fill="#312e81" stroke="#a5b4fc" strokeWidth={1} />
        <text x={55} y={124} textAnchor="middle" fill="#a5b4fc" fontSize="12" fontWeight="600">x</text>
        <text x={55} y={155} textAnchor="middle" fill="#9ca3af" fontSize="8">input</text>

        {/* Branch point */}
        <line x1={90} y1={120} x2={130} y2={120} stroke="#475569" strokeWidth={1} />

        {/* --- Top path: Frozen W (highway) --- */}
        <line x1={130} y1={120} x2={130} y2={50} stroke="#475569" strokeWidth={1} />
        <line x1={130} y1={50} x2={200} y2={50} stroke="#475569" strokeWidth={1} markerEnd="url(#loraArrow)" />

        <rect x={202} y={25} width={90} height={50} rx={6} fill="#44403c" stroke="#78716c" strokeWidth={1} />
        <text x={247} y={47} textAnchor="middle" fill="#a8a29e" fontSize="11" fontWeight="600">W</text>
        <text x={247} y={62} textAnchor="middle" fill="#78716c" fontSize="8">(frozen)</text>

        <line x1={292} y1={50} x2={340} y2={50} stroke="#475569" strokeWidth={1} />

        {/* --- Bottom path: LoRA detour (A then B) --- */}
        <line x1={130} y1={120} x2={130} y2={190} stroke="#475569" strokeWidth={1} />
        <line x1={130} y1={190} x2={160} y2={190} stroke="#475569" strokeWidth={1} markerEnd="url(#loraArrow)" />

        {/* A matrix (down-project) */}
        <rect x={162} y={170} width={60} height={40} rx={5} fill="#064e3b" stroke="#6ee7b7" strokeWidth={1} />
        <text x={192} y={188} textAnchor="middle" fill="#6ee7b7" fontSize="10" fontWeight="600">A</text>
        <text x={192} y={201} textAnchor="middle" fill="#6ee7b7" fontSize="7">(r × d_in)</text>

        <line x1={222} y1={190} x2={248} y2={190} stroke="#475569" strokeWidth={1} markerEnd="url(#loraArrow)" />

        {/* B matrix (up-project) */}
        <rect x={250} y={170} width={60} height={40} rx={5} fill="#4c1d95" stroke="#c4b5fd" strokeWidth={1} />
        <text x={280} y={188} textAnchor="middle" fill="#c4b5fd" fontSize="10" fontWeight="600">B</text>
        <text x={280} y={201} textAnchor="middle" fill="#c4b5fd" fontSize="7">(d_out × r)</text>

        <line x1={310} y1={190} x2={340} y2={190} stroke="#475569" strokeWidth={1} />

        {/* Sum circle */}
        <circle cx={350} cy={120} r={16} fill="#1e1b4b" stroke="#818cf8" strokeWidth={1} />
        <text x={350} y={125} textAnchor="middle" fill="#818cf8" fontSize="16" fontWeight="700">+</text>

        {/* Lines into sum */}
        <line x1={340} y1={50} x2={350} y2={50} stroke="#475569" strokeWidth={1} />
        <line x1={350} y1={50} x2={350} y2={104} stroke="#475569" strokeWidth={1} />
        <line x1={340} y1={190} x2={350} y2={190} stroke="#475569" strokeWidth={1} />
        <line x1={350} y1={190} x2={350} y2={136} stroke="#475569" strokeWidth={1} />

        {/* Output */}
        <line x1={366} y1={120} x2={395} y2={120} stroke="#475569" strokeWidth={1} markerEnd="url(#loraArrow)" />
        <rect x={397} y={100} width={35} height={40} rx={5} fill="#312e81" stroke="#a5b4fc" strokeWidth={1} />
        <text x={414} y={124} textAnchor="middle" fill="#a5b4fc" fontSize="12" fontWeight="600">h</text>

        {/* Labels */}
        <text x={247} y={14} textAnchor="middle" fill="#78716c" fontSize="8" fontStyle="italic">frozen highway</text>
        <text x={220} y={230} textAnchor="middle" fill="#6ee7b7" fontSize="8" fontStyle="italic">trainable detour (starts at zero)</text>

        {/* Scaling label */}
        <text x={350} y={245} textAnchor="middle" fill="#9ca3af" fontSize="8">
          h = Wx + BAx · (α/r)
        </text>

        <defs>
          <marker
            id="loraArrow"
            markerWidth="6"
            markerHeight="4"
            refX="5"
            refY="2"
            orient="auto"
          >
            <polygon points="0 0, 6 2, 0 4" fill="#475569" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Quantization Number Line
// Shows float32 values mapped to int8 grid points. Continuous line above,
// discrete grid below, with arrows showing the mapping.
// ---------------------------------------------------------------------------

function QuantizationNumberLineDiagram() {
  const svgW = 460
  const svgH = 180

  const lineY = 40
  const gridY = 120
  const lineLeft = 40
  const lineRight = 420

  // Example weights: [-0.8, 0.3, 1.2, -0.5]
  // Max abs = 1.2, scale = 1.2 / 127
  // Quantized: [-85, 32, 127, -53]

  const mapToX = (val: number, min: number, max: number) =>
    lineLeft + ((val - min) / (max - min)) * (lineRight - lineLeft)

  const floatMin = -1.2
  const floatMax = 1.2
  const weights = [
    { val: -0.8, q: -85, color: '#f87171' },
    { val: -0.5, q: -53, color: '#fbbf24' },
    { val: 0.3, q: 32, color: '#6ee7b7' },
    { val: 1.2, q: 127, color: '#a5b4fc' },
  ]

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {/* Float32 number line (continuous) */}
        <text x={svgW / 2} y={16} textAnchor="middle" fill="#9ca3af" fontSize="9" fontWeight="500">
          float32 (continuous)
        </text>
        <line x1={lineLeft} y1={lineY} x2={lineRight} y2={lineY} stroke="#475569" strokeWidth={1.5} />

        {/* Float ticks at -1.2, -0.6, 0, 0.6, 1.2 */}
        {[-1.2, -0.6, 0, 0.6, 1.2].map((v) => {
          const x = mapToX(v, floatMin, floatMax)
          return (
            <g key={`ftick-${v}`}>
              <line x1={x} y1={lineY - 4} x2={x} y2={lineY + 4} stroke="#475569" strokeWidth={1} />
              <text x={x} y={lineY + 16} textAnchor="middle" fill="#6b7280" fontSize="8">{v}</text>
            </g>
          )
        })}

        {/* Weight points on float line */}
        {weights.map((w) => {
          const x = mapToX(w.val, floatMin, floatMax)
          return (
            <g key={`fw-${w.val}`}>
              <circle cx={x} cy={lineY} r={4} fill={w.color} />
              <text x={x} y={lineY - 10} textAnchor="middle" fill={w.color} fontSize="8" fontWeight="600">
                {w.val}
              </text>
            </g>
          )
        })}

        {/* Int8 grid line (discrete) */}
        <text x={svgW / 2} y={gridY - 18} textAnchor="middle" fill="#9ca3af" fontSize="9" fontWeight="500">
          int8 (discrete grid: -127 to 127)
        </text>
        <line x1={lineLeft} y1={gridY} x2={lineRight} y2={gridY} stroke="#475569" strokeWidth={1.5} />

        {/* Int8 ticks */}
        {[-127, -85, -53, 0, 32, 85, 127].map((v) => {
          const x = mapToX(v, -127, 127)
          return (
            <g key={`itick-${v}`}>
              <line x1={x} y1={gridY - 4} x2={x} y2={gridY + 4} stroke="#475569" strokeWidth={1} />
              <text x={x} y={gridY + 16} textAnchor="middle" fill="#6b7280" fontSize="8">{v}</text>
            </g>
          )
        })}

        {/* Quantized weight points on int8 line */}
        {weights.map((w) => {
          const x = mapToX(w.q, -127, 127)
          return (
            <circle key={`iw-${w.q}`} cx={x} cy={gridY} r={4} fill={w.color} />
          )
        })}

        {/* Arrows connecting float to int */}
        {weights.map((w) => {
          const x1 = mapToX(w.val, floatMin, floatMax)
          const x2 = mapToX(w.q, -127, 127)
          return (
            <line
              key={`arrow-${w.val}`}
              x1={x1}
              y1={lineY + 5}
              x2={x2}
              y2={gridY - 5}
              stroke={w.color}
              strokeWidth={0.8}
              strokeDasharray="3,3"
              opacity={0.6}
            />
          )
        })}

        {/* Label: scale factor */}
        <text x={svgW / 2} y={gridY + 38} textAnchor="middle" fill="#9ca3af" fontSize="8">
          scale = max(|w|) / 127 = 1.2 / 127 ≈ 0.0094
        </text>
        <text x={svgW / 2} y={gridY + 52} textAnchor="middle" fill="#9ca3af" fontSize="8">
          Store: int8 values + one float32 scale factor per group
        </text>
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline SVG: Memory Comparison Bar Chart
// Shows memory for full finetuning vs LoRA vs quantized inference.
// ---------------------------------------------------------------------------

function MemoryComparisonDiagram() {
  const svgW = 420
  const svgH = 180

  const barX = 140
  const barMaxW = 250
  const barH = 28
  const gap = 14

  const entries = [
    { label: 'Full finetune (7B)', gb: 84, color: '#f87171', textColor: '#fca5a5' },
    { label: 'LoRA finetune (7B)', gb: 16, color: '#a78bfa', textColor: '#c4b5fd' },
    { label: 'QLoRA finetune (7B)', gb: 4, color: '#6ee7b7', textColor: '#6ee7b7' },
    { label: 'Quantized inf. (7B, 4-bit)', gb: 3.5, color: '#38bdf8', textColor: '#7dd3fc' },
  ]

  const maxGb = 84

  return (
    <div className="flex justify-center py-4 overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="overflow-visible"
      >
        {entries.map((entry, i) => {
          const y = 16 + i * (barH + gap)
          const w = (entry.gb / maxGb) * barMaxW

          return (
            <g key={entry.label}>
              <text
                x={barX - 8}
                y={y + barH / 2 + 4}
                textAnchor="end"
                fill="#9ca3af"
                fontSize="9"
              >
                {entry.label}
              </text>
              <rect
                x={barX}
                y={y}
                width={w}
                height={barH}
                rx={4}
                fill={entry.color}
                opacity={0.3}
              />
              <rect
                x={barX}
                y={y}
                width={w}
                height={barH}
                rx={4}
                fill="none"
                stroke={entry.color}
                strokeWidth={1}
              />
              <text
                x={barX + w + 8}
                y={y + barH / 2 + 4}
                fill={entry.textColor}
                fontSize="10"
                fontWeight="600"
              >
                {entry.gb} GB
              </text>
            </g>
          )
        })}

        {/* Consumer GPU reference line */}
        {(() => {
          const gpuX = barX + (24 / maxGb) * barMaxW
          return (
            <g>
              <line x1={gpuX} y1={8} x2={gpuX} y2={svgH - 10} stroke="#fbbf24" strokeWidth={1} strokeDasharray="4,3" />
              <text x={gpuX + 4} y={svgH - 2} fill="#fbbf24" fontSize="8">RTX 4090 (24 GB)</text>
            </g>
          )
        })()}
      </svg>
    </div>
  )
}

export function LoraAndQuantizationLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="LoRA, Quantization & Inference"
            description="The two techniques that take LLMs from 'requires a cluster' to 'runs on your laptop'—efficient finetuning with LoRA and efficient inference with quantization."
            category="Fine-tuning"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          2. Context + Constraints (Section 1 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            You understand the full pipeline: pretraining gives knowledge, SFT
            gives format, RLHF/DPO gives judgment. But you have been working
            with GPT-2 (124M parameters). Real models&mdash;Llama 2 7B, Mistral
            7B, Llama 3 70B&mdash;are 50 to 500 times larger. This lesson
            explains why full finetuning does not scale, how LoRA makes
            finetuning practical by adding tiny trainable matrices, and how
            quantization shrinks models for inference. By the end, you will
            understand the techniques that make LLM work accessible on real
            hardware.
          </ObjectiveBlock>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The memory wall: where training memory goes (weights, gradients, optimizer states)',
              'Low-rank decomposition: the mathematical idea behind LoRA (no SVD or eigenvalues)',
              'LoRA: architecture, forward pass, where to apply, parameter savings, merge at inference',
              'Quantization: absmax, zero-point, why neural networks tolerate precision loss',
              'GPTQ/AWQ at name-drop level, QLoRA as the combination of both techniques',
              'NOT: implementing LoRA from scratch (notebook uses PEFT library)',
              'NOT: quantization-aware training (post-training quantization only)',
              'NOT: SVD, eigenvalues, or formal linear algebra beyond rank intuition',
              'NOT: pruning, distillation, mixture of experts, or production deployment',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Notebook Ahead">
            This is a STRETCH lesson with a notebook. The exercises build
            incrementally: memory calculation, quantization by hand, LoRA from
            scratch, then practical finetuning and inference with libraries.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Hook — The Memory Wall (Section 2 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Memory Wall"
            subtitle="Full finetuning does not scale"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have been finetuning GPT-2 for the last three lessons:
              classification heads, instruction tuning, and alignment. GPT-2 has
              124 million parameters, and it fits comfortably in a Colab
              notebook. But the models people actually use are 50 to 500 times
              larger. What happens when you try to full-finetune a 7B model?
            </p>

            <p className="text-muted-foreground">
              Let&rsquo;s do the math. Training with AdamW requires storing four
              things per parameter:
            </p>

            <div className="space-y-3">
              <PhaseCard number={1} title="Weights" subtitle="The model itself" color="blue">
                <p className="text-sm">
                  7B parameters in bfloat16 = <strong>14 GB</strong> (2 bytes
                  per parameter).
                </p>
              </PhaseCard>

              <PhaseCard number={2} title="Gradients" subtitle="One per parameter" color="cyan">
                <p className="text-sm">
                  7B gradients in bfloat16 = <strong>14 GB</strong>. Every
                  parameter needs its gradient for the backward pass.
                </p>
              </PhaseCard>

              <PhaseCard number={3} title="Adam Optimizer States" subtitle="Momentum + variance" color="purple">
                <p className="text-sm">
                  Adam stores two running averages per parameter (momentum and
                  variance), both in float32 = <strong>28 GB each</strong>,{' '}
                  <strong>56 GB total</strong>. This is where the memory really
                  goes.
                </p>
              </PhaseCard>

              <PhaseCard number={4} title="Total" subtitle="Before activations" color="rose">
                <p className="text-sm">
                  Weights (14 GB) + gradients (14 GB) + Adam states (56 GB) ={' '}
                  <strong>~84 GB minimum</strong>. And this does not include
                  activations stored for the backward pass.
                </p>
              </PhaseCard>
            </div>

            <ComparisonRow
              left={{
                title: 'GPT-2 (124M)',
                color: 'emerald',
                items: [
                  'Weights: ~0.5 GB',
                  'Full training: ~1.5 GB',
                  'Fits on any GPU',
                  'What you have been using',
                ],
              }}
              right={{
                title: 'Llama 2 7B',
                color: 'rose',
                items: [
                  'Weights: 14 GB (bfloat16)',
                  'Full training: ~84 GB minimum',
                  'A100 80GB does not fit',
                  'Consumer RTX 4090 (24 GB): impossible',
                ],
              }}
            />

            <GradientCard title="Two Problems, Two Solutions" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Problem 1: Finetuning is too expensive.</strong> Too
                  many trainable parameters means too many gradients and optimizer
                  states. <strong>Solution: LoRA.</strong>
                </p>
                <p>
                  <strong>Problem 2: Inference is too expensive.</strong> The
                  model is too large to fit in memory for serving.{' '}
                  <strong>Solution: Quantization.</strong>
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Optimizer Dominates">
            The model weights are only one-sixth of training memory. Adam&rsquo;s
            momentum and variance tensors each require 28 GB in float32&mdash;56 GB
            combined, two-thirds of the total. The optimizer states, not the
            model, are the real memory bottleneck during training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain — Why Full Finetuning Is Low-Rank (Section 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Full Finetuning Is Low-Rank"
            subtitle="The key insight that makes LoRA possible"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If we cannot afford to update all 7 billion parameters, what if we
              did not need to?
            </p>

            <p className="text-muted-foreground">
              When you finetune a pretrained model for a specific task, you are
              not rewriting everything the model knows. You are making a{' '}
              <strong>targeted adjustment</strong>. The weight matrix changes
              from <InlineMath math="W" /> to{' '}
              <InlineMath math="W + \Delta W" />, and{' '}
              <InlineMath math="\Delta W" /> is typically{' '}
              <strong>low-rank</strong>.
            </p>

            <p className="text-muted-foreground">
              Think of it this way: you learned English over 20 years. Learning
              to write formal business emails does not rewrite your knowledge of
              English&mdash;it adds a small adjustment to how you use it. The
              adjustment is much simpler than the full knowledge.
            </p>

            <SectionHeader
              title="What 'Low-Rank' Means"
              subtitle="Building from matrix multiplication you already know"
            />

            <p className="text-muted-foreground">
              A <InlineMath math="768 \times 768" /> weight matrix has 589,824
              entries. But what if its &ldquo;effective information&rdquo; can be
              captured by far fewer numbers?
            </p>

            <p className="text-muted-foreground">
              Start with a small example. Consider this{' '}
              <InlineMath math="4 \times 4" /> matrix where every row is a
              scaled version of the same pattern:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math={`\\Delta W = \\begin{bmatrix} 2 & 4 & 6 & 8 \\\\ 1 & 2 & 3 & 4 \\\\ 3 & 6 & 9 & 12 \\\\ 0.5 & 1 & 1.5 & 2 \\end{bmatrix}`} />
            </div>

            <p className="text-muted-foreground">
              This matrix has 16 entries, but every row is a scalar multiple of{' '}
              <InlineMath math="[1, 2, 3, 4]" />. It can be written as:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math={`\\Delta W = \\underbrace{\\begin{bmatrix} 2 \\\\ 1 \\\\ 3 \\\\ 0.5 \\end{bmatrix}}_{\\mathbf{b}\\,(4 \\times 1)} \\times \\underbrace{\\begin{bmatrix} 1 & 2 & 3 & 4 \\end{bmatrix}}_{\\mathbf{a}\\,(1 \\times 4)}`} />
            </div>

            <p className="text-muted-foreground">
              That is a <strong>rank-1</strong> matrix: 16 entries captured by
              just 8 numbers (4 + 4). Now scale this up to real model dimensions.
            </p>

            <p className="text-muted-foreground">
              More generally: a rank-<InlineMath math="r" /> matrix of size{' '}
              <InlineMath math="(m \times n)" /> can be written as{' '}
              <InlineMath math="B \cdot A" /> where{' '}
              <InlineMath math="B" /> is <InlineMath math="(m \times r)" /> and{' '}
              <InlineMath math="A" /> is <InlineMath math="(r \times n)" />.
              The storage drops from <InlineMath math="m \cdot n" /> to{' '}
              <InlineMath math="r \cdot (m + n)" />.
            </p>

            <GradientCard title="The Parameter Savings" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  For GPT-2&rsquo;s <InlineMath math="W_Q" /> (768 × 768):
                </p>
                <ul className="space-y-1 ml-2">
                  <li>{'•'} Full matrix: 768 × 768 = <strong>589,824</strong> parameters</li>
                  <li>{'•'} Rank-8 decomposition: 768 × 8 + 8 × 768 = <strong>12,288</strong> parameters</li>
                  <li>{'•'} That is a <strong>48× reduction</strong>&mdash;about 2% of the full matrix</li>
                </ul>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Hu et al. (2021) showed that for GPT-3 finetuning,{' '}
              <InlineMath math="\Delta W" /> during adaptation is indeed
              low-rank. The singular values drop off sharply after the first
              few&mdash;most of the &ldquo;change&rdquo; during finetuning lives
              in a small subspace.
            </p>

            <p className="text-muted-foreground">
              Of course the update is low-rank. The pretrained model already
              understands language. Finetuning for sentiment analysis is a small
              adjustment, not a fundamental change.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Low-Rank Intuition">
            A low-rank matrix is one where the rows are not all independent.
            They lie in a small subspace. You do not need formal linear algebra
            to see this: if a 768-dimensional weight change can be described by
            just 8 directions, only 8 &ldquo;knobs&rdquo; are doing the work.
          </InsightBlock>
          <TipBlock title="No SVD Required">
            You do not need SVD or eigenvalues to understand LoRA. The only idea
            is: a big matrix can sometimes be written as the product of two
            smaller matrices. That is it.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Check 1 — Predict (Section 4 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Parameter Count" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You have a frozen weight matrix <InlineMath math="W" /> (768 ×
                768) and you want to add a trainable low-rank update with rank{' '}
                <InlineMath math="r = 8" />.
              </p>
              <p>
                <strong>
                  How many trainable parameters does this add? How does this
                  compare to training the full <InlineMath math="W" />?
                </strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>12,288 trainable parameters</strong> (768 × 8 + 8 ×
                    768) vs 589,824 for the full matrix. That is about{' '}
                    <strong>2%</strong> of the full parameter count.
                  </p>
                  <p>
                    And you can apply this to every attention projection in every
                    layer. Even with LoRA on all attention projections, the total
                    trainable parameters are a tiny fraction of the full model.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          6. Explain — LoRA: Low-Rank Adaptation (Section 5 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LoRA: Low-Rank Adaptation"
            subtitle="Freeze everything, add tiny trainable detours"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now you have the mathematical foundation. LoRA applies this idea
              directly: instead of updating the full weight matrix{' '}
              <InlineMath math="W" />, freeze it and add a trainable low-rank
              term.
            </p>

            <p className="text-muted-foreground">
              The original forward pass:
            </p>

            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="h = Wx \quad \text{(W is frozen, requires\_grad=False)}" />
            </div>

            <p className="text-muted-foreground">
              The LoRA forward pass:
            </p>

            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="h = Wx + (BA)x \cdot \frac{\alpha}{r}" />
            </div>

            <p className="text-muted-foreground">
              <InlineMath math="B" /> is{' '}
              <InlineMath math="(d_{\text{out}} \times r)" />, initialized to{' '}
              <strong>zeros</strong>. <InlineMath math="A" /> is{' '}
              <InlineMath math="(r \times d_{\text{in}})" />, initialized from
              a random normal distribution. Because <InlineMath math="B" />{' '}
              starts at zero, the LoRA output starts at zero&mdash;the model
              begins <strong>identical</strong> to the pretrained model. Training
              gradually learns the adaptation.
            </p>

            <p className="text-muted-foreground">
              <InlineMath math="\alpha / r" /> is a scaling factor.{' '}
              <InlineMath math="\alpha" /> is a hyperparameter (often set to{' '}
              <InlineMath math="2r" /> or fixed at 16). It controls the
              magnitude of the LoRA update relative to the original weights.
            </p>

            <LoraBypassDiagram />

            <p className="text-muted-foreground">
              Input <InlineMath math="x" /> flows through two paths: the frozen{' '}
              <InlineMath math="W" /> (the highway) and the trainable{' '}
              <InlineMath math="B \cdot A" /> (the detour). The outputs are
              summed. The detour starts at zero and learns the task-specific
              adjustment.
            </p>

            <SectionHeader
              title="Where to Apply LoRA"
              subtitle="Not every layer—just the attention projections"
            />

            <p className="text-muted-foreground">
              Standard practice: apply LoRA to the{' '}
              <InlineMath math="W_Q" /> and <InlineMath math="W_V" /> attention
              projections. Empirically, these capture the most task-relevant
              adaptation. You <em>can</em> also apply to{' '}
              <InlineMath math="W_K" />, <InlineMath math="W_O" />, and FFN
              layers, but with diminishing returns on the additional parameters.
            </p>

            <p className="text-muted-foreground">
              <strong>Rank as hyperparameter:</strong>{' '}
              <InlineMath math="r = 4, 8, 16" /> are common values. Higher rank
              means more expressiveness but more parameters.{' '}
              <InlineMath math="r = 8" /> is a common default for most tasks.
            </p>

            <p className="text-muted-foreground">
              A minimal LoRA layer in PyTorch:
            </p>

            <CodeBlock
              code={`class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False  # freeze base
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base(x)           # frozen highway
        lora_out = (x @ self.A.T @ self.B.T) * self.scale  # trainable detour
        return base_out + lora_out`}
              language="python"
              filename="lora.py"
            />

            <SectionHeader
              title="Merge at Inference"
              subtitle="No additional cost after training"
            />

            <p className="text-muted-foreground">
              After finetuning, you can fold the LoRA weights back into the
              base:
            </p>

            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="W_{\text{merged}} = W + BA \cdot \frac{\alpha}{r}" />
            </div>

            <p className="text-muted-foreground">
              The merged model is a standard weight matrix with no architectural
              change. <strong>Zero additional inference cost.</strong> The
              LoRA adapters disappear into the base weights.
            </p>

            <GradientCard title="LoRA Is Surgical Frozen Backbone" color="violet">
              <p className="text-sm">
                Remember frozen backbone finetuning from Finetuning for
                Classification? You froze the backbone and added a tiny
                classification head. LoRA extends this idea: instead of freezing
                the backbone and adding a head at the end, you freeze{' '}
                <strong>everything</strong> and add tiny detours{' '}
                <strong>inside</strong> the backbone. Same philosophy&mdash;preserve
                pretrained knowledge, adapt minimally&mdash;but more surgical.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Head, a Detour">
            LoRA adapters are <strong>not</strong> like the classification head
            from Lesson 1. The classification head was added at the output. LoRA
            adapters are distributed <strong>inside</strong> the existing
            layers&mdash;they modify the transformation, not the output
            interface.
          </WarningBlock>
          <TipBlock title="LoRA Is Not Everywhere">
            A common misconception: LoRA adds adapters to every layer. In
            practice, you typically apply LoRA only to{' '}
            <InlineMath math="W_Q" /> and <InlineMath math="W_V" />.
            Applying to all layers increases parameter count without
            proportional quality gain.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain — Why LoRA Works (Section 6 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why LoRA Works (Not Just How)"
            subtitle="It is not a lossy approximation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might expect LoRA to sacrifice quality for efficiency. After
              all, you are training 2% of the parameters instead of 100%. Surely
              that comes at a cost?
            </p>

            <p className="text-muted-foreground">
              Hu et al. (2021) showed that LoRA <strong>matches or exceeds</strong>{' '}
              full finetuning on multiple benchmarks. Concretely: on GPT-3 175B
              with GLUE SST-2, LoRA (rank 8) achieved <strong>95.1%</strong>{' '}
              accuracy vs <strong>95.2%</strong> for full finetuning&mdash;essentially
              identical, while training less than 0.01% of the parameters. On
              some tasks, LoRA slightly outperforms full finetuning.
            </p>

            <p className="text-muted-foreground">
              Why? LoRA&rsquo;s low-rank constraint acts as{' '}
              <strong>implicit regularization</strong>. Fewer trainable
              parameters means less overfitting on small datasets. This is the
              same argument from Finetuning for Classification: a
              1,536-parameter classification head does not overfit because it
              does not have the capacity to memorize the training data. LoRA
              adapters have limited capacity by design, and this is a{' '}
              <strong>feature</strong> for small-dataset finetuning.
            </p>

            <ComparisonRow
              left={{
                title: 'When LoRA Excels',
                color: 'emerald',
                items: [
                  'Classification, sentiment analysis',
                  'Instruction following (SFT)',
                  'Domain adaptation (medical, legal)',
                  'Most practical finetuning tasks',
                  'Small to medium datasets',
                ],
              }}
              right={{
                title: 'When LoRA May Underperform',
                color: 'amber',
                items: [
                  'Learning a completely new language',
                  'Radical domain shifts (text model → code)',
                  'Tasks requiring high-rank weight changes',
                  'Fundamentally reorganizing representations',
                  'Rare in practice',
                ],
              }}
            />

            <p className="text-muted-foreground">
              The weight change during finetuning is low-rank because finetuning
              is a <strong>refinement</strong>, not a revolution. For the vast
              majority of practical tasks, LoRA captures the essential
              adaptation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Constraint as Feature">
            LoRA&rsquo;s low-rank constraint is not a compromise. It is
            regularization. On small datasets, limiting the model&rsquo;s
            capacity to adapt <em>improves</em> generalization. The same
            principle you saw with frozen backbone training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Check 2 — Transfer Question (Section 7 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Checkpoint: Rank and Overfitting" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A researcher LoRA-finetunes a model with rank 4 and rank 64.
                The rank-4 version performs well on a simple classification task.
                The rank-64 version performs <strong>slightly worse</strong>.
              </p>
              <p>
                <strong>Why?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Think about it, then reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>Overfitting.</strong> Rank-64 has more trainable
                    parameters (more capacity) but the task is simple. The
                    additional capacity leads to memorizing the small
                    classification dataset rather than learning the essential
                    pattern.
                  </p>
                  <p>
                    Rank-4&rsquo;s tighter constraint acts as
                    regularization&mdash;it forces the model to learn only the
                    essential adaptation. This is the same principle as the
                    overfitting argument from Finetuning for Classification: 124M
                    parameters on a small dataset vs a 1,536-parameter head.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          9. Explain — Quantization: From Float to Integer (Section 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quantization: From Float to Integer"
            subtitle="Shrinking models for inference"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              LoRA solves the training problem. But inference also has a memory
              problem: the model weights themselves take 14 GB in float16 for a
              7B model. How do we shrink them?
            </p>

            <p className="text-muted-foreground">
              In Scaling &amp; Efficiency, you traded float32 for bfloat16 and
              lost almost nothing. The key insight: neural network weights are
              redundant. They tolerate precision loss. Let&rsquo;s push this
              further.
            </p>

            <p className="text-muted-foreground">
              <strong>Quantization</strong> maps continuous floating-point values
              to a discrete grid of integers. Instead of 16 or 32 bits per
              weight, use 8 or 4 bits.
            </p>

            <SectionHeader
              title="Absmax Quantization"
              subtitle="The simplest method—trace it with real numbers"
            />

            <p className="text-muted-foreground">
              Start with a vector of weights:{' '}
              <InlineMath math="w = [-0.8,\; 0.3,\; 1.2,\; -0.5]" />
            </p>

            <div className="space-y-3">
              <PhaseCard number={1} title="Find the Scale" subtitle="Normalize by max absolute value" color="blue">
                <div className="text-sm space-y-1">
                  <p>
                    Max absolute value:{' '}
                    <InlineMath math="|1.2| = 1.2" />
                  </p>
                  <p>
                    Scale factor:{' '}
                    <InlineMath math="s = 1.2 / 127 \approx 0.0094" /> (for
                    int8, range is [&minus;127, 127])
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard number={2} title="Quantize" subtitle="Divide by scale and round" color="cyan">
                <div className="text-sm space-y-1">
                  <p>
                    <InlineMath math="q = \text{round}(w / s)" />
                  </p>
                  <p>
                    <InlineMath math="= \text{round}([-85.1,\; 31.9,\; 127.0,\; -53.2])" />
                  </p>
                  <p>
                    <InlineMath math="= [-85,\; 32,\; 127,\; -53]" />
                  </p>
                </div>
              </PhaseCard>

              <PhaseCard number={3} title="Store" subtitle="int8 values + one float32 scale" color="purple">
                <p className="text-sm">
                  Store <InlineMath math="q" /> as int8 (1 byte each) and{' '}
                  <InlineMath math="s" /> as one float32 (4 bytes for the whole
                  group). Total: 8 bytes vs 16 bytes in float32&mdash;
                  <strong>2× savings</strong> for int8. For int4:{' '}
                  <strong>4× savings</strong>.
                </p>
              </PhaseCard>

              <PhaseCard number={4} title="Dequantize" subtitle="Reconstruct when needed" color="orange">
                <div className="text-sm space-y-1">
                  <p>
                    <InlineMath math="w_{\text{approx}} = q \times s" />
                  </p>
                  <p>
                    <InlineMath math="= [-0.80,\; 0.30,\; 1.20,\; -0.50]" />
                  </p>
                  <p>Very close to the original values. The rounding error is tiny.</p>
                </div>
              </PhaseCard>
            </div>

            <QuantizationNumberLineDiagram />

            <SectionHeader
              title="The Outlier Problem"
              subtitle="When absmax breaks down"
            />

            <p className="text-muted-foreground">
              Now consider a different weight vector:{' '}
              <InlineMath math="w = [-0.1,\; 0.05,\; 0.02,\; -0.03,\; 8.5]" />.
              One extreme outlier.
            </p>

            <p className="text-muted-foreground">
              Absmax maps the range [&minus;8.5, 8.5] to [&minus;127, 127]. The
              four small values near zero all map to{' '}
              <InlineMath math="q \approx [-1,\; 1,\; 0,\; 0]" />&mdash;they
              lose almost all their information. Most of the int8 range is
              wasted on values that do not exist. The outlier hijacks the scale.
            </p>

            <GradientCard title="Negative Example: Outliers Break Absmax" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Small values:</strong> &minus;0.1, 0.05, 0.02, &minus;0.03
                  all map to int8 values near zero. They are
                  indistinguishable after quantization.
                </p>
                <p>
                  <strong>The outlier (8.5):</strong> claims the entire positive
                  range. The quantization grid is stretched thin across a range
                  where almost no weights exist.
                </p>
                <p>
                  This motivates more sophisticated approaches.
                </p>
              </div>
            </GradientCard>

            <SectionHeader
              title="Zero-Point Quantization"
              subtitle="Better range utilization"
            />

            <p className="text-muted-foreground">
              <strong>Zero-point quantization</strong> shifts the range so the
              minimum maps to &minus;128 and the maximum maps to 127. Two
              parameters: scale and zero_point.
            </p>

            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="q = \text{round}(w / \text{scale}) + \text{zero\_point}" />
              <BlockMath math="w_{\text{approx}} = (q - \text{zero\_point}) \times \text{scale}" />
            </div>

            <p className="text-muted-foreground">
              This better utilizes the integer range, especially for asymmetric
              distributions where the mean is not at zero. The additional
              cost is one extra integer (the zero point) per group.
            </p>

            <p className="text-muted-foreground">
              <strong>Why quantization works for neural networks:</strong> weight
              distributions are approximately Gaussian&mdash;most values
              clustered near zero. The quantization grid captures the dense
              region well. Outliers are relatively rare and can be handled
              specially.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Precision Spectrum">
            You already stepped down the precision spectrum once: float32 to
            bfloat16 in Scaling &amp; Efficiency, and nothing broke.
            Quantization continues this: bfloat16 → int8 → int4. Each step
            trades precision for memory, and neural networks tolerate it.
          </InsightBlock>
          <WarningBlock title="Quantization ≠ Training">
            Quantization is for <strong>inference</strong>. You quantize the
            weights after training to make the model smaller for serving. The
            training itself still happens in higher precision (float16 or
            float32). Post-training quantization is what we cover here.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          10. Explain — Practical Quantization (Section 9 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practical Quantization: GPTQ, AWQ, and the 4-bit Frontier"
            subtitle="Going beyond simple absmax"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Real quantization methods go beyond simple absmax and zero-point:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="GPTQ" color="purple">
                <p className="text-sm">
                  Post-training quantization that uses a small calibration dataset
                  to find the optimal quantization minimizing reconstruction
                  error. Compensates for each layer&rsquo;s quantization error
                  when quantizing the next layer.
                </p>
              </GradientCard>
              <GradientCard title="AWQ" color="blue">
                <p className="text-sm">
                  Activation-Aware Weight Quantization. Identifies which weights
                  matter most by looking at activation magnitudes, and keeps those
                  at higher precision. Smarter about which information to
                  preserve.
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              The key result: these methods achieve <strong>INT4
              quantization</strong> with less than 1% perplexity degradation. A
              7B model goes from 14 GB (float16) to{' '}
              <strong>~3.5 GB</strong> (int4). That fits on a consumer GPU.
            </p>

            <p className="text-muted-foreground">
              <strong>NF4 (NormalFloat4):</strong> a 4-bit data type designed
              specifically for normally-distributed neural network weights.
              Instead of uniformly spaced quantization levels, NF4 puts more
              precision near zero where most weights cluster. Used in QLoRA.
            </p>

            <GradientCard title="4 Bits Does Not Destroy Quality" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  Intuition says 4 bits (16 possible values) must lose most
                  information compared to 16 bits (65,536 values). But GPTQ-quantized
                  models show less than 1% perplexity degradation.
                </p>
                <p>
                  <strong>Why?</strong> Neural network weights are highly
                  redundant. Most cluster in a narrow range. The information
                  density of a trained weight matrix is much lower than its
                  bit-width suggests. The weights are compressible because the
                  model&rsquo;s learned representations have structure.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Name-Drop Level">
            You do not need to understand the GPTQ or AWQ algorithms. Just know
            they exist, they are post-training quantization methods, and they
            achieve impressive INT4 results. When you see
            &ldquo;GPTQ&rdquo; or &ldquo;AWQ&rdquo; in a model name on
            HuggingFace, you know what it means.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Explain — QLoRA: Putting It Together (Section 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="QLoRA: Putting It Together"
            subtitle="Quantized base + LoRA adapters"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              LoRA and quantization are <strong>not</strong> alternatives&mdash;they
              solve different problems. LoRA makes <em>training</em> efficient
              (fewer trainable parameters, fewer gradients and optimizer states).
              Quantization makes the <em>base model</em> smaller (fits in
              memory for storage and inference).
            </p>

            <p className="text-muted-foreground">
              <strong>QLoRA</strong> combines both: quantize the base model to
              4-bit (saves memory for storage), add LoRA adapters in bfloat16
              (trainable, higher precision).
            </p>

            <div className="space-y-3">
              <PhaseCard number={1} title="Base Model (4-bit)" subtitle="Quantized with NF4" color="blue">
                <p className="text-sm">~3.5 GB for a 7B model</p>
              </PhaseCard>
              <PhaseCard number={2} title="LoRA Adapters (bfloat16)" subtitle="Trainable low-rank matrices" color="purple">
                <p className="text-sm">~10&ndash;50 MB depending on rank and which layers</p>
              </PhaseCard>
              <PhaseCard number={3} title="Gradients (LoRA only)" subtitle="Only for the adapters" color="cyan">
                <p className="text-sm">~10&ndash;50 MB&mdash;gradients only for the tiny adapter parameters</p>
              </PhaseCard>
              <PhaseCard number={4} title="Optimizer States (LoRA only)" subtitle="Adam momentum + variance" color="orange">
                <p className="text-sm">~20&ndash;100 MB&mdash;Adam states only for the adapters</p>
              </PhaseCard>
            </div>

            <GradientCard title="Total: ~4 GB" color="emerald">
              <div className="space-y-2 text-sm">
                <p>
                  Compare to full finetuning: <strong>~84 GB</strong>. QLoRA is{' '}
                  <strong>~21× more memory-efficient</strong>. A 7B model fits
                  on a consumer GPU for both finetuning and inference.
                </p>
              </div>
            </GradientCard>

            <MemoryComparisonDiagram />

            <p className="text-muted-foreground">
              <strong>KV caching revisited:</strong> remember from Scaling &amp;
              Efficiency that the KV cache stores key and value tensors from
              previous tokens. For long sequences, the KV cache can become the
              dominant memory cost. It can also be quantized&mdash;from float16
              to int8&mdash;halving its memory with minimal quality impact.
            </p>

            <GradientCard title="The Empowerment Moment" color="violet">
              <div className="space-y-2 text-sm">
                <p>
                  You do not need a cluster. You do not need an A100. A single
                  consumer GPU (24 GB) can finetune a 7B model with QLoRA.
                  A 16 GB GPU can run inference on a 13B model quantized to
                  4-bit.
                </p>
                <p>
                  The &ldquo;massive GPU&rdquo; barrier is largely a myth for
                  parameter-efficient finetuning and quantized inference. The
                  techniques in this lesson make real LLM work accessible on
                  hardware you can actually buy.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Problems, Combined Solution">
            LoRA solves the training problem (too many trainable parameters).
            Quantization solves the inference problem (model too large). QLoRA
            uses both together: quantized storage + efficient training. They
            are complementary, not competing.
          </InsightBlock>
          <TipBlock title="Spectrum of Approaches">
            Think of it as a spectrum. Frozen backbone (Lesson 1) → KL penalty
            (Lesson 3) → LoRA (this lesson). Three different approaches to
            the same challenge: adapt without forgetting, at increasing levels
            of surgical precision.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Notebook Exercises (Section 11 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Notebook Exercises"
            subtitle="Memory calculation, LoRA, quantization, and practical finetuning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The notebook builds incrementally. Each exercise adds context for
              the next.
            </p>

            <div className="space-y-3">
              <GradientCard title="Exercise 1: Memory Calculator (Guided)" color="blue">
                <div className="space-y-1 text-sm">
                  <p>
                    Calculate memory requirements for inference and training at
                    different precisions. Compute for GPT-2 (124M) and Llama 2
                    7B. Verify that optimizer states dominate training memory.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Exercise 2: Quantization by Hand (Guided)" color="cyan">
                <div className="space-y-1 text-sm">
                  <p>
                    Apply absmax quantization step by step. Compute
                    reconstruction error. Try with outlier-heavy distributions
                    and see the error increase.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Exercise 3: LoRA from Scratch (Supported)" color="purple">
                <div className="space-y-1 text-sm">
                  <p>
                    Implement a LoRALinear layer, verify base weights are frozen,
                    count trainable vs total parameters. See the parameter
                    savings firsthand.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Exercise 4: LoRA Finetuning with PEFT (Supported)" color="emerald">
                <div className="space-y-1 text-sm">
                  <p>
                    Use the HuggingFace PEFT library to LoRA-finetune a model.
                    Compare trainable parameters, training time, and memory
                    usage. The conceptual understanding from Exercise 3 makes
                    the library feel transparent, not magical.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Exercise 5: Quantized Inference (Minimal Scaffolding)" color="orange">
                <div className="space-y-1 text-sm">
                  <p>
                    Load a quantized model (4-bit or 8-bit). Compare memory
                    usage, generation speed, and output quality vs full
                    precision. See that the quality difference is barely
                    noticeable while the memory difference is dramatic.
                  </p>
                </div>
              </GradientCard>
            </div>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the notebook to work through the exercises hands-on.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/4-4-4-lora-and-quantization.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  Exercises build incrementally: memory math → quantization by
                  hand → LoRA from scratch → PEFT library → quantized inference.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Build Before Using Libraries">
            Exercises 2 and 3 have you implement quantization and LoRA from
            scratch before using libraries in Exercises 4 and 5. This way, the
            libraries are transparent tools, not black boxes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Summary (Section 12 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="What You Learned"
            items={[
              {
                headline: 'Full finetuning hits a memory wall.',
                description:
                  'Training a 7B model requires ~84 GB: weights + gradients + Adam optimizer states. The optimizer states (56 GB alone) dominate the cost.',
              },
              {
                headline: 'LoRA: freeze base weights, add trainable low-rank matrices.',
                description:
                  'Train 0.1–1% of parameters by adding small B·A detours alongside frozen weight matrices. Matches full finetuning quality—the low-rank constraint acts as regularization.',
              },
              {
                headline: 'Quantization: map float weights to int8/int4.',
                description:
                  'Neural network weights are compressible. INT4 quantization achieves 4× memory savings with less than 1% quality loss. Extends the precision spectrum from float32 → bfloat16 → int8 → int4.',
              },
              {
                headline: 'QLoRA combines both: quantized base + LoRA adapters.',
                description:
                  'Finetune a 7B model in ~4 GB instead of ~84 GB. A consumer GPU can finetune and serve real LLMs. The "massive GPU" barrier is largely a myth.',
              },
              {
                headline: 'Finetuning is a refinement, not a revolution.',
                description:
                  'Weight changes during finetuning are low-rank because you are adjusting, not rewriting. LoRA captures this insight directly: the adaptation lives in a small subspace.',
              },
              {
                headline: 'LoRA is the surgical version of "freeze the backbone."',
                description:
                  'Same philosophy as frozen backbone training—preserve pretrained knowledge, adapt minimally—but applied inside the model rather than only at the output.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          14. Next Step (Section 13 from outline)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="px-4 py-3 bg-primary/10 border border-primary/30 rounded-lg space-y-2">
            <p className="text-sm font-medium text-primary">What comes next</p>
            <p className="text-sm text-muted-foreground">
              You now have the complete toolkit: pretrain (Building &amp;
              Training GPT), finetune with SFT (Instruction Tuning), align
              with RLHF/DPO (RLHF &amp; Alignment), make it practical with
              LoRA and quantization (this lesson). Next: we put the entire
              pipeline together. No new concepts&mdash;just synthesis. How does a
              model go from random weights to a deployed assistant? Where does
              each technique fit? The final lesson of the module and the
              capstone of Series 4.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'LoRA: Low-Rank Adaptation of Large Language Models',
                authors: 'Hu et al., 2021',
                url: 'https://arxiv.org/abs/2106.09685',
                note: 'The original LoRA paper. Section 4 shows the key result: LoRA matches full finetuning on GPT-3 and RoBERTa benchmarks.',
              },
              {
                title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
                authors: 'Dettmers et al., 2023',
                url: 'https://arxiv.org/abs/2305.14314',
                note: 'Introduces 4-bit NormalFloat and shows QLoRA matches 16-bit finetuning quality. Section 2 covers the NF4 data type.',
              },
              {
                title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers',
                authors: 'Frantar et al., 2022',
                url: 'https://arxiv.org/abs/2210.17323',
                note: 'The paper behind GPTQ quantization. Shows INT4 quantization with minimal perplexity loss on large language models.',
              },
              {
                title: 'AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration',
                authors: 'Lin et al., 2023',
                url: 'https://arxiv.org/abs/2306.00978',
                note: 'Activation-aware approach to quantization. Section 3 explains why protecting salient weights matters.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="When you're done"
            description="Make sure you can explain: why full finetuning is expensive (where the memory goes), how LoRA works (the bypass architecture and low-rank insight), and how quantization maps floats to integers (the absmax walkthrough). The notebook exercises ground all three concepts."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
