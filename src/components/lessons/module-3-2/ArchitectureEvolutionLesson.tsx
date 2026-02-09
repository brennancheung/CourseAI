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
  TryThisBlock,
  ConceptBlock,
  WarningBlock,
  SummaryBlock,
  NextStepBlock,
  GradientCard,
  ComparisonRow,
} from '@/components/lessons'
import { ArchitectureComparisonExplorer } from '@/components/widgets/ArchitectureComparisonExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { cn } from '@/lib/utils'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Architecture Evolution
 *
 * First lesson in Module 3.2 (Modern Architectures).
 * Traces how CNNs evolved from LeNet to AlexNet to VGG,
 * teaching why depth helps and the 3x3 stacking insight.
 *
 * Core concepts at DEVELOPED:
 * - Effective receptive field of stacked small filters (quantitative)
 * - Parameter efficiency of small vs large filters (3x3+3x3 vs 5x5)
 * - VGG architecture and 3x3 philosophy
 *
 * Concepts at INTRODUCED:
 * - LeNet architecture (map to student's MNIST CNN)
 * - AlexNet innovations (ReLU, dropout, GPU scaling)
 * - Architecture evolution as problem-driven innovation
 *
 * Previous: MNIST CNN Project (module 3.1)
 * Next: ResNets (module 3.2, lesson 2)
 */

// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

function StageRow({
  label,
  shape,
  color,
  annotation,
}: {
  label: string
  shape: string
  color: string
  annotation?: string
}) {
  const dotColor = stageRowDotColor(color)

  return (
    <div className="flex items-center gap-3 text-sm">
      <div className={cn('w-2 h-2 rounded-full flex-shrink-0', dotColor)} />
      <span className="text-muted-foreground w-48 truncate">{label}</span>
      <span className="font-semibold text-foreground w-28">{shape}</span>
      {annotation && (
        <span className="text-xs text-muted-foreground/70">{annotation}</span>
      )}
    </div>
  )
}

function stageRowDotColor(color: string): string {
  if (color === 'violet') return 'bg-violet-400'
  if (color === 'sky') return 'bg-sky-400'
  if (color === 'amber') return 'bg-amber-400'
  if (color === 'emerald') return 'bg-emerald-400'
  if (color === 'rose') return 'bg-rose-400'
  return 'bg-zinc-400'
}

/**
 * Inline SVG diagram showing how two stacked 3x3 convolutions
 * produce a 5x5 receptive field on the original input.
 *
 * Left: Layer 2's single 3x3 window reads 3 positions from Layer 1.
 * Right: Those 3 positions in Layer 1 each see 3 input positions,
 * but they overlap — covering a 5x5 region total.
 */
function ReceptiveFieldDiagram() {
  const cellSize = 28
  const gap = 2
  const step = cellSize + gap

  // Grid dimensions
  const inputCols = 5
  const inputRows = 5
  const midCols = 3
  const midRows = 3

  // Offsets for the three grids
  const inputX = 0
  const inputY = 40
  const midX = (inputCols * step - midCols * step) / 2
  const midY = inputY + inputRows * step + 50
  const outX = (inputCols * step - cellSize) / 2
  const outY = midY + midRows * step + 50

  const totalWidth = inputCols * step
  const totalHeight = outY + cellSize + 24

  function cell(
    x: number,
    y: number,
    fill: string,
    stroke: string,
    label?: string,
  ) {
    return (
      <g key={`${x}-${y}-${fill}`}>
        <rect
          x={x}
          y={y}
          width={cellSize}
          height={cellSize}
          rx={3}
          fill={fill}
          stroke={stroke}
          strokeWidth={1.5}
        />
        {label && (
          <text
            x={x + cellSize / 2}
            y={y + cellSize / 2 + 4}
            textAnchor="middle"
            fontSize={9}
            fill="currentColor"
            className="text-muted-foreground"
          >
            {label}
          </text>
        )}
      </g>
    )
  }

  const inputCells = []
  const midCells = []

  // Input grid: 5x5, highlight the full 5x5 region
  for (let r = 0; r < inputRows; r++) {
    for (let c = 0; c < inputCols; c++) {
      const x = inputX + c * step
      const y = inputY + r * step
      // All cells are in the receptive field (the entire 5x5 grid)
      inputCells.push(
        cell(x, y, 'rgba(139, 92, 246, 0.25)', 'rgba(139, 92, 246, 0.6)'),
      )
    }
  }

  // Middle grid: 3x3, highlight all
  for (let r = 0; r < midRows; r++) {
    for (let c = 0; c < midCols; c++) {
      const x = midX + c * step
      const y = midY + r * step
      midCells.push(
        cell(x, y, 'rgba(56, 189, 248, 0.25)', 'rgba(56, 189, 248, 0.6)'),
      )
    }
  }

  // Output: single cell
  const outputCell = cell(
    outX,
    outY,
    'rgba(251, 191, 36, 0.3)',
    'rgba(251, 191, 36, 0.7)',
  )

  // Bracket/connection lines from input to mid
  const inputBottom = inputY + inputRows * step
  const midTop = midY
  const midBottom = midY + midRows * step
  const outTop = outY

  return (
    <div className="flex justify-center">
      <svg
        width={totalWidth}
        height={totalHeight}
        viewBox={`-10 0 ${totalWidth + 20} ${totalHeight}`}
        className="text-muted-foreground"
      >
        {/* Labels */}
        <text x={totalWidth / 2} y={14} textAnchor="middle" fontSize={11} fill="currentColor" fontWeight={600}>
          Input (5x5 receptive field)
        </text>
        <text x={totalWidth / 2} y={midY - 8} textAnchor="middle" fontSize={11} fill="currentColor" fontWeight={600}>
          After 1st 3x3 conv
        </text>
        <text x={totalWidth / 2} y={outY - 8} textAnchor="middle" fontSize={11} fill="currentColor" fontWeight={600}>
          After 2nd 3x3 conv
        </text>

        {/* Input grid */}
        {inputCells}

        {/* Connection lines: input -> mid */}
        <line
          x1={totalWidth / 2}
          y1={inputBottom + 2}
          x2={totalWidth / 2}
          y2={midTop - 20}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.4}
        />
        <text x={totalWidth / 2 + 8} y={(inputBottom + midTop) / 2 + 2} fontSize={9} fill="currentColor" opacity={0.6}>
          3x3 conv
        </text>

        {/* Mid grid */}
        {midCells}

        {/* Connection lines: mid -> output */}
        <line
          x1={totalWidth / 2}
          y1={midBottom + 2}
          x2={totalWidth / 2}
          y2={outTop - 20}
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.4}
        />
        <text x={totalWidth / 2 + 8} y={(midBottom + outTop) / 2 + 2} fontSize={9} fill="currentColor" opacity={0.6}>
          3x3 conv
        </text>

        {/* Output cell */}
        {outputCell}
      </svg>
    </div>
  )
}

/**
 * Proportional block diagram for VGG-16.
 * Block width represents spatial dimensions (proportional to 224).
 * Block color intensity represents channel depth.
 */
function Vgg16BlockDiagram() {
  const blocks = [
    { label: 'Input', spatial: 224, channels: 3, color: 'bg-zinc-500' },
    { label: 'B1', spatial: 112, channels: 64, color: 'bg-violet-400' },
    { label: 'B2', spatial: 56, channels: 128, color: 'bg-violet-500' },
    { label: 'B3', spatial: 28, channels: 256, color: 'bg-violet-600' },
    { label: 'B4', spatial: 14, channels: 512, color: 'bg-violet-700' },
    { label: 'B5', spatial: 7, channels: 512, color: 'bg-violet-800' },
    { label: 'FC', spatial: 4, channels: 512, color: 'bg-emerald-500' },
  ]

  const maxSpatial = 224

  return (
    <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
      <p className="text-xs text-muted-foreground font-medium">
        VGG-16 proportional block diagram (width = spatial size)
      </p>
      <div className="flex items-end gap-1.5 h-28 pt-2">
        {blocks.map((block) => {
          const widthPct = Math.max((block.spatial / maxSpatial) * 100, 6)
          return (
            <div
              key={block.label}
              className="flex flex-col items-center gap-1 flex-shrink-0"
              style={{ width: `${widthPct}%` }}
            >
              <span className="text-[9px] text-muted-foreground/60 font-mono">
                {block.label === 'FC' ? 'flat' : `${block.spatial}x${block.spatial}`}
              </span>
              <div
                className={cn('w-full rounded-sm', block.color)}
                style={{
                  height: `${Math.max((block.channels / 512) * 72, 8)}px`,
                  opacity: block.label === 'FC' ? 0.8 : 1,
                }}
              />
              <span className="text-[9px] text-muted-foreground font-medium">{block.label}</span>
              <span className="text-[8px] text-muted-foreground/50 font-mono">
                {block.label === 'FC' ? '1000 classes' : `${block.channels}ch`}
              </span>
            </div>
          )
        })}
      </div>
      <div className="flex justify-between text-[9px] text-muted-foreground/50 pt-1">
        <span>{'\u2190'} Spatial wide, few channels</span>
        <span>Spatial narrow, many channels {'\u2192'}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ArchitectureEvolutionLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Architecture Evolution"
            description="How CNNs evolved from LeNet to AlexNet to VGG&mdash;and why going deeper works."
            category="Modern Architectures"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 1: Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain why stacking more convolutional layers improves performance
            (deeper feature hierarchies, larger receptive fields, more nonlinearity)
            and trace how the field discovered this through LeNet, AlexNet, and VGG.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In the MNIST CNN Project, you built a 2-layer CNN that crushed MNIST
            at 99% accuracy. You understand the conv-pool-fc pattern, pooling,
            stride, padding, and dimension tracking. This lesson extends that
            understanding to real-world scale.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The architectural progression from LeNet to AlexNet to VGG',
              "Why depth helps: hierarchical features, receptive field growth, more nonlinearity",
              "VGG's 3x3 stacking insight: parameter efficiency and the math behind it",
              'NOT: ResNets or skip connections (next lesson)',
              'NOT: implementing or training these architectures (conceptual understanding)',
              'NOT: GoogLeNet/Inception, modern architectures, or Vision Transformers',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Module Arc">
            This lesson establishes &ldquo;deeper = better&rdquo; and its limits.
            Next: ResNets solve the degradation problem. Then: transfer learning
            puts these architectures to practical use.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 2: Hook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From MNIST to ImageNet"
            subtitle="Your CNN works. Can it scale?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your MNIST CNN has 2 convolutional layers and handles 10 digit classes
              at 99% accuracy. Impressive&mdash;but MNIST is a toy problem: 28x28
              grayscale images of centered, isolated digits.
            </p>
            <ComparisonRow
              left={{
                title: 'Your MNIST CNN',
                color: 'blue',
                items: [
                  '28x28 grayscale input',
                  '10 classes (digits)',
                  '2 conv layers',
                  '~62K parameters',
                ],
              }}
              right={{
                title: 'ImageNet Challenge',
                color: 'orange',
                items: [
                  '224x224 RGB input',
                  '1,000 classes (real objects)',
                  'Cluttered, multi-object scenes',
                  'Winner in 2014: 138M parameters',
                ],
              }}
            />
            <p className="text-muted-foreground">
              The winning network in 2012 had <strong>8 layers</strong>. By 2014,
              the winners had <strong>19</strong>. How did the field figure out that
              going deeper was the answer&mdash;and what does &ldquo;deeper&rdquo;
              actually buy you?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="You Already Know This">
            Your MNIST CNN follows the same conv-pool-fc pattern as LeNet-5 from
            1998. You basically built a first-generation architecture. The journey
            from there to modern networks is a series of comprehensible steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 3: RGB / Multi-channel Recap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: RGB Images"
            subtitle="From 1 channel to 3"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every architecture from here on processes RGB images&mdash;3 channels
              instead of 1. The change is smaller than you might think.
            </p>
            <p className="text-muted-foreground">
              Your MNIST CNN used <code className="text-xs bg-muted px-1 rounded">Conv2d(1, 32, 3)</code>&mdash;one
              input channel. For RGB images, that
              becomes <code className="text-xs bg-muted px-1 rounded">Conv2d(3, 64, 3)</code>.
              Each filter is now <strong>3x3x3 = 27 weights</strong> instead of
              3x3x1 = 9. Everything else&mdash;the sliding, the output size formula,
              the feature map concept&mdash;works identically.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground font-medium">
                RGB convolution dimension example:
              </p>
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="Input" shape="224x224x3" color="zinc" annotation="RGB image" />
                <StageRow label="Conv2d(3, 64, 3, pad=1)" shape="224x224x64" color="violet" annotation="64 filters, each 3x3x3 = 27 weights" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Parameters: 64 filters x (3x3x3 weights + 1 bias) = 64 x 28 = 1,792
              </p>
            </div>
            <p className="text-muted-foreground">
              The <code className="text-xs bg-muted px-1 rounded">in_channels</code> parameter
              in <InlineMath math="\texttt{Conv2d}" /> simply sets how deep each
              filter is. After the first layer, the &ldquo;channels&rdquo; are
              feature maps from the previous layer, not color channels.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The General Pattern">
            A conv filter always has shape <InlineMath math="K \times K \times C_{in}" /> where{' '}
            <InlineMath math="C_{in}" /> is the number of input channels. For the
            first layer that is 3 (RGB). For subsequent layers it is whatever the
            previous layer&apos;s output channels are.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: LeNet (1998)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LeNet (1998): The Proof of Concept"
            subtitle="Where it all started"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              LeNet-5 was one of the first successful CNNs, designed by Yann LeCun
              for reading handwritten zip codes and checks. Its architecture will
              look familiar:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                LeNet-5 Architecture (simplified)
              </p>
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="Input" shape="32x32x1" color="zinc" annotation="Grayscale" />
                <StageRow label="Conv(6, 5x5)" shape="28x28x6" color="violet" annotation="Sigmoid activation" />
                <StageRow label="AvgPool(2x2)" shape="14x14x6" color="sky" annotation="Average, not max" />
                <StageRow label="Conv(16, 5x5)" shape="10x10x16" color="violet" annotation="Sigmoid activation" />
                <StageRow label="AvgPool(2x2)" shape="5x5x16" color="sky" />
                <StageRow label="Flatten" shape="400" color="amber" />
                <StageRow label="FC(120) + FC(84) + FC(10)" shape="10" color="emerald" annotation="~62K total params" />
              </div>
            </div>
            <p className="text-muted-foreground">
              Compare this to your MNIST CNN: conv-pool-conv-pool-flatten-fc.
              <strong> Same pattern.</strong> The differences are details: LeNet used
              sigmoid activations, average pooling, and 5x5 filters. Those choices
              reflect what the field knew in 1998.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="You Built This">
            Your MNIST CNN is essentially a modernized LeNet. Same architecture
            pattern, similar scale, similar accuracy. You already understand
            first-generation CNNs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Comprehension Check" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                Knowing what you know now, what would you change about LeNet?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>ReLU instead of sigmoid</strong>&mdash;sigmoid causes
                    vanishing gradients (remember the telephone game from Training
                    Dynamics? Each layer slightly shrinks the gradient signal, and
                    after many layers almost nothing gets through). ReLU lets
                    gradients flow through deeper networks.
                  </p>
                  <p>
                    <strong>Max pooling instead of average</strong>&mdash;max pooling
                    preserves the strongest feature response, giving &ldquo;is the
                    feature present?&rdquo; rather than a blurred average.
                  </p>
                  <p>
                    If you thought of either of these, you already understand the
                    innovations that separated LeNet from the next generation.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Era Constraints">
            LeNet was designed when compute was scarce and backpropagation through
            deep networks was poorly understood. Sigmoid was the standard activation.
            The architecture worked for zip codes but could not scale to real-world
            images.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: AlexNet (2012)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="AlexNet (2012): The Breakthrough"
            subtitle="14 years later, CNNs come back"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Between 1998 and 2012, most image recognition research moved away
              from neural networks. Then AlexNet won the ImageNet competition by a
              massive margin, reigniting interest in deep learning overnight.
            </p>
            <p className="text-muted-foreground">
              AlexNet was not just &ldquo;a bigger LeNet.&rdquo; It introduced
              several innovations, each solving a specific problem:
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="ReLU Activation" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Problem:</strong> Sigmoid squashes gradients toward zero,
                    making it impossible to train networks deeper than a few layers.
                  </p>
                  <p>
                    <strong>Solution:</strong> <InlineMath math="\text{ReLU}(x) = \max(0, x)" />.
                    For positive inputs, the gradient is exactly 1&mdash;no
                    shrinking, no compounding decay through layers. Unlike
                    sigmoid&apos;s 0.25 multiplier at each layer, ReLU passes
                    the gradient through unchanged. Remember the telephone
                    game? With ReLU, the message arrives intact.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Dropout" color="blue">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Problem:</strong> A network with 60 million parameters
                    overfits catastrophically without regularization.
                  </p>
                  <p>
                    <strong>Solution:</strong> Randomly silence 50% of neurons during
                    training. You already know this from Overfitting and
                    Regularization&mdash;creates an implicit ensemble.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="GPU Training" color="cyan">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Problem:</strong> Training on ImageNet (1.2M images,
                    224x224 RGB) is computationally brutal on CPUs.
                  </p>
                  <p>
                    <strong>Solution:</strong> Parallel training across 2 GPUs. This
                    was necessary but <em>not sufficient</em>&mdash;a big sigmoid
                    network on GPUs would still fail.
                  </p>
                </div>
              </GradientCard>

              <GradientCard title="Scale" color="orange">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>LeNet:</strong> ~62K parameters, 5 weight layers,
                    32x32 grayscale input.
                  </p>
                  <p>
                    <strong>AlexNet:</strong> ~62M parameters, 8 weight layers,
                    227x227 RGB input. A 1,000x scale-up in parameters.
                  </p>
                </div>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Misconception: GPUs Made It Work">
            GPUs were necessary for training at ImageNet scale, but they were not
            sufficient. A large network with sigmoid activations and no dropout
            would still fail&mdash;the architectural innovations (ReLU, dropout)
            were essential. GPU was the engine; the architecture was the design.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
            <p className="text-sm text-muted-foreground font-medium">
              AlexNet Architecture (simplified)
            </p>
            <div className="space-y-2 font-mono text-sm">
              <StageRow label="Input" shape="227x227x3" color="zinc" annotation="RGB, much larger than MNIST" />
              <StageRow label="Conv(96, 11x11, stride=4)" shape="55x55x96" color="violet" annotation="Large filters for initial features" />
              <StageRow label="MaxPool(3x3, stride=2)" shape="27x27x96" color="sky" />
              <StageRow label="Conv(256, 5x5, pad=2)" shape="27x27x256" color="violet" />
              <StageRow label="MaxPool(3x3, stride=2)" shape="13x13x256" color="sky" />
              <StageRow label="Conv(384, 3x3, pad=1)" shape="13x13x384" color="violet" annotation="3x3 filters from here" />
              <StageRow label="Conv(384, 3x3, pad=1)" shape="13x13x384" color="violet" />
              <StageRow label="Conv(256, 3x3, pad=1)" shape="13x13x256" color="violet" />
              <StageRow label="MaxPool(3x3, stride=2)" shape="6x6x256" color="sky" />
              <StageRow label="Flatten + Dropout" shape="9216" color="amber" />
              <StageRow label="FC(4096) + FC(4096) + FC(1000)" shape="1000" color="emerald" annotation="~62M total params" />
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Mixed Filter Sizes">
            AlexNet uses 11x11 in the first layer, then 5x5, then 3x3. This mix
            was ad-hoc&mdash;there was no principled reason for these specific
            sizes. VGG would later show that all-3x3 works better.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check 1 - Predict and Verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Why Mixed Filter Sizes?"
            subtitle="Predict before you peek"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                AlexNet uses 11x11 filters in the first layer and 3x3 filters in
                later layers. <strong>Why might larger filters be useful early
                and smaller filters later?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal reasoning
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The first layer processes 227x227 raw pixels. An 11x11 filter
                    with stride 4 covers enough input to detect basic features (edges,
                    color gradients) while aggressively reducing spatial dimensions
                    from 227 to 55.
                  </p>
                  <p>
                    Later layers operate on <em>already-processed</em> feature
                    maps&mdash;a 3x3 filter on a feature map at 13x13 already has a
                    large effective receptive field back to the original image. Small
                    filters suffice because each position already encodes rich
                    information.
                  </p>
                  <p>
                    But was the 11x11 really necessary? VGG would prove it was not.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: VGG (2014) - The Core Concept
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="VGG (2014): The 3x3 Insight"
            subtitle="What if we ONLY use 3x3 filters and go deeper?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The VGG team from Oxford asked a radical question: instead of mixing
              large and small filters like AlexNet, what if the entire network used
              <strong> only 3x3 convolutions</strong>&mdash;the smallest filter that
              still captures left/right, up/down, and center?
            </p>
            <p className="text-muted-foreground">
              The result was VGG-16: a network of 16 weight layers built from a
              repeating block pattern. Two or three 3x3 conv layers, then a max
              pool. Repeat 5 times. Flatten and classify.
            </p>
            <p className="text-muted-foreground">
              But wait&mdash;if you only use 3x3 filters, each one only sees a 3x3
              patch. How does the network detect large features? The answer is the
              key insight of this lesson: <strong>stacking small filters
              builds large receptive fields</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="VGG&apos;s Philosophy">
            VGG proved that a disciplined, simple architecture (all 3x3) stacked
            deep beats ad-hoc complexity (mixed filter sizes). Simplicity + depth
            won.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Receptive field equivalence */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Receptive Field Equivalence
            </p>
            <p className="text-muted-foreground">
              In Building a CNN, you learned that stacking conv layers with pooling
              expands the receptive field. Now let&apos;s quantify it for stacked
              convolutions <em>without</em> pooling:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-4">
              <div className="space-y-3">
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>
                    <strong>One 3x3 conv:</strong> Each output position sees a 3x3
                    region of the input. <strong>RF = 3</strong>.
                  </p>
                  <p>
                    <strong>Two stacked 3x3 convs:</strong> The second conv&apos;s 3x3
                    window covers 3 positions, each of which sees 3 inputs. But they
                    overlap by 2 positions, so the total
                    is: <strong>RF = 5</strong>. Same as a single 5x5 filter.
                  </p>
                  <p>
                    <strong>Three stacked 3x3 convs:</strong> Extends the pattern.{' '}
                    <strong>RF = 7</strong>. Same as a single 7x7 filter.
                  </p>
                </div>
                <div className="pt-2 pb-1">
                  <p className="text-xs text-muted-foreground/70 font-medium mb-2 text-center">
                    Two stacked 3x3 convs: one output position sees a 5x5 input region
                  </p>
                  <ReceptiveFieldDiagram />
                  <p className="text-xs text-muted-foreground/60 text-center mt-2">
                    The 2nd conv reads a 3x3 region of the 1st conv&apos;s output.
                    Each of those 9 positions sees a 3x3 input region.
                    With overlap, the total input coverage is 5x5.
                  </p>
                </div>
              </div>
              <div className="text-sm">
                <p className="text-muted-foreground mb-2">
                  The general formula for <InlineMath math="n" /> stacked{' '}
                  <InlineMath math="k \times k" /> convs (stride 1):
                </p>
                <div className="py-3 px-4 bg-muted/50 rounded">
                  <BlockMath math="\text{RF} = n \times (k - 1) + 1" />
                </div>
                <div className="mt-2 font-mono text-xs space-y-0.5 text-muted-foreground">
                  <p>n=1, k=3: 1(2) + 1 = 3</p>
                  <p>n=2, k=3: 2(2) + 1 = <strong className="text-foreground">5</strong> (same as one 5x5)</p>
                  <p>n=3, k=3: 3(2) + 1 = <strong className="text-foreground">7</strong> (same as one 7x7)</p>
                </div>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Extending the Zoom-Out Analogy">
            In Building a CNN, each conv-pool stage was a &ldquo;zoom out.&rdquo;
            Here, stacking convs <em>without</em> pooling also increases the
            receptive field&mdash;a gentler zoom that captures more context
            while maintaining spatial resolution.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Parameter comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Parameter Efficiency
            </p>
            <p className="text-muted-foreground">
              Two stacked 3x3 convs and one 5x5 conv cover the same receptive field.
              But they are not equivalent computationally:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-medium">
                Comparing 3x3+3x3 vs 5x5 (same input/output channels{' '}
                <InlineMath math="C" />)
              </p>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground/70 font-medium">Metric</p>
                  <p className="text-sm text-muted-foreground">Parameters</p>
                  <p className="text-sm text-muted-foreground">Nonlinearities</p>
                  <p className="text-sm text-muted-foreground">Receptive Field</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-violet-400 font-medium">Two 3x3 convs</p>
                  <p className="text-sm font-mono text-foreground">
                    2 x (3x3xCxC) = <InlineMath math="18C^2" />
                  </p>
                  <p className="text-sm text-foreground">2 (ReLU after each)</p>
                  <p className="text-sm text-foreground">5x5</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-amber-400 font-medium">One 5x5 conv</p>
                  <p className="text-sm font-mono text-foreground">
                    5x5xCxC = <InlineMath math="25C^2" />
                  </p>
                  <p className="text-sm text-foreground">1 (ReLU after)</p>
                  <p className="text-sm text-foreground">5x5</p>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                Two 3x3 convs use <strong>28% fewer parameters</strong> and have{' '}
                <strong>twice the nonlinearity</strong>&mdash;for the same receptive
                field. More nonlinearity means more expressive power: the network
                can learn more complex patterns.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Same RF Does Not Mean Same Computation">
            A common misconception: &ldquo;two 3x3 filters and one 5x5 filter do
            the same thing.&rdquo; They cover the same region, but two 3x3 convs
            have a ReLU between them&mdash;that nonlinearity lets them compute more
            complex functions than a single 5x5 ever could.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* VGG block structure */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              VGG-16: The Repeating Block Pattern
            </p>
            <p className="text-muted-foreground">
              Sixteen weight layers and 138 million parameters sounds intimidating.
              But VGG-16 only has two building blocks: a 3x3 conv and a 2x2 max
              pool. Everything else is repetition. Despite its depth, VGG-16 is
              one of the simplest architectures to understand because it follows
              a rigid template:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <div className="space-y-2 font-mono text-sm">
                <StageRow label="Input" shape="224x224x3" color="zinc" />
                <StageRow label="Block 1: 2x Conv(64, 3x3)" shape="224x224x64" color="violet" annotation="RF=5 after convs" />
                <StageRow label="MaxPool(2x2)" shape="112x112x64" color="sky" />
                <StageRow label="Block 2: 2x Conv(128, 3x3)" shape="112x112x128" color="violet" annotation="Channels double" />
                <StageRow label="MaxPool(2x2)" shape="56x56x128" color="sky" />
                <StageRow label="Block 3: 3x Conv(256, 3x3)" shape="56x56x256" color="violet" annotation="RF=40 after convs" />
                <StageRow label="MaxPool(2x2)" shape="28x28x256" color="sky" />
                <StageRow label="Block 4: 3x Conv(512, 3x3)" shape="28x28x512" color="violet" />
                <StageRow label="MaxPool(2x2)" shape="14x14x512" color="sky" />
                <StageRow label="Block 5: 3x Conv(512, 3x3)" shape="14x14x512" color="violet" annotation="RF=196 after convs" />
                <StageRow label="MaxPool(2x2)" shape="7x7x512" color="sky" />
                <StageRow label="Flatten" shape="25088" color="amber" />
                <StageRow label="FC(4096) + FC(4096) + FC(1000)" shape="1000" color="emerald" annotation="138M total params" />
              </div>
            </div>
            <Vgg16BlockDiagram />
            <p className="text-muted-foreground">
              The familiar pattern is still there: <strong>spatial shrinks, channels
              grow, then flatten.</strong> VGG just applies it more systematically
              and more deeply. The diagram above makes the pattern visceral: blocks
              get narrower (less spatial area) and taller (more channels) as you go
              deeper.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Where the Parameters Live">
            Despite having 13 conv layers, most of VGG-16&apos;s 138M parameters are
            in the three FC layers at the end. The first FC layer alone
            (25088 x 4096) has ~103M parameters&mdash;75% of the total. The conv
            layers are surprisingly efficient.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Check 2 - Transfer Question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: The 7x7 Question"
            subtitle="Apply the 3x3 stacking principle"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                A colleague proposes using a <strong>single 7x7 filter</strong>.
              </p>
              <ol className="space-y-1 list-decimal list-inside">
                <li>How many stacked 3x3 filters give the same receptive field?</li>
                <li>Compare the parameter counts (assume <InlineMath math="C" /> input and output channels).</li>
                <li>How many nonlinearities does each option have?</li>
              </ol>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    1. Three 3x3 convs give RF = 3(2) + 1 = <strong>7</strong>.
                  </p>
                  <p>
                    2. Three 3x3: <InlineMath math="3 \times 9C^2 = 27C^2" />.
                    One 7x7: <InlineMath math="49C^2" />.
                    That is <strong>45% fewer parameters</strong> for the 3x3 stack.
                  </p>
                  <p>
                    3. Three 3x3: <strong>3 nonlinearities</strong> (ReLU after each).
                    One 7x7: <strong>1 nonlinearity</strong>. The 3x3 stack is
                    significantly more expressive.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="VGG&apos;s Core Insight">
            Given a receptive field budget, spend it on many small filters rather
            than fewer large ones. Fewer parameters, more nonlinearity, same spatial
            coverage.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 9: Interactive Widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Architecture Comparison"
            subtitle="Compare LeNet, AlexNet, and VGG-16 layer by layer"
          />
          <p className="text-muted-foreground mb-4">
            Select an architecture to see its full layer pipeline with output shapes,
            parameter counts, and receptive field growth. Use the comparison table
            at the bottom to see the three architectures side by side.
          </p>
          <ExercisePanel
            title="Architecture Comparison Explorer"
            subtitle="Select an architecture and trace its dimensions"
          >
            <ArchitectureComparisonExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>
                Compare VGG-16&apos;s total parameter count to AlexNet&apos;s. Which is
                bigger? Where do most of the parameters live?
              </li>
              <li>
                In VGG-16, find where the receptive field first exceeds 7x7 (three
                stacked 3x3 convs without pooling in between).
              </li>
              <li>
                Notice that most of VGG&apos;s parameters are in the FC layers,
                not the conv layers. The 13 conv layers contribute less than 15%
                of total parameters.
              </li>
              <li>
                Look at how AlexNet&apos;s filter sizes decrease through layers (11x11,
                5x5, 3x3). VGG just starts at 3x3.
              </li>
              <li>
                Use the &ldquo;What If VGG Used Larger Filters?&rdquo; toggle to
                replace VGG blocks with single large filters. How much do the
                parameters increase? How many nonlinearities are lost?
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: The Pattern of Innovation
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Pattern of Innovation"
            subtitle="Each generation solved the previous one's limitation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Step back and look at the meta-pattern. This is not random progress.
              Each architecture generation was a response to the previous one&apos;s
              limitations:
            </p>
            <div className="space-y-3">
              <GradientCard title="LeNet (1998)" color="blue">
                <p className="text-sm">
                  <strong>Proved CNNs work</strong> on simple tasks. But sigmoid
                  activations limited depth, average pooling blurred features, and
                  compute constraints kept networks small. <em>Limitation: could
                  not scale.</em>
                </p>
              </GradientCard>
              <GradientCard title="AlexNet (2012)" color="violet">
                <p className="text-sm">
                  <strong>Solved vanishing gradients</strong> (ReLU), <strong>overfitting</strong>{' '}
                  (dropout), and <strong>compute</strong> (GPU). Scaled to ImageNet
                  with 8 layers and 62M parameters. But used ad-hoc filter sizes
                  (11x11, 5x5, 3x3) with no principled justification.{' '}
                  <em>Limitation: no design principle for filter size or depth.</em>
                </p>
              </GradientCard>
              <GradientCard title="VGG (2014)" color="orange">
                <p className="text-sm">
                  <strong>Found the design principle:</strong> all 3x3 filters,
                  stacked deep. Proved that disciplined simplicity + depth beats
                  ad-hoc complexity. 16-19 layers, 138M parameters.{' '}
                  <em>Limitation: deeper networks eventually degrade (next lesson).</em>
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Notice that each generation did not just add more layers. AlexNet
              changed the activation function, the pooling type, and added
              regularization. VGG changed the filter size philosophy entirely.
              The innovations were <strong>qualitative</strong>, not just
              quantitative&mdash;each generation rethought what to stack, not
              just how much.
            </p>
            <p className="text-muted-foreground">
              This pattern&mdash;identify the bottleneck, design the
              solution&mdash;is how all architecture innovation works. When you read
              about a new architecture, ask: &ldquo;what problem was the previous
              one struggling with?&rdquo; The answer usually explains the design.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Architecture Encodes Assumptions">
            In the MNIST CNN Project, you learned that architecture encodes
            assumptions about data. Here, that insight extends: architecture
            <strong> design</strong> encodes assumptions about the{' '}
            <strong>problem</strong>&mdash;what features matter, how complex they
            are, and how to find them efficiently.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 11: But Deeper Has Limits
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="But Deeper Has Limits"
            subtitle="The cliffhanger"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If deeper is better, why stop at 19 layers? VGG researchers and others
              tried going to 30, 50, 100 layers. Something unexpected happened:
              <strong> deeper networks performed worse</strong>.
            </p>
            <p className="text-muted-foreground">
              Not just on test data&mdash;on <strong>training data too</strong>.
            </p>
            <div className="py-4 px-6 bg-rose-500/5 border border-rose-500/20 rounded-lg space-y-3">
              <p className="text-sm font-medium text-rose-300">
                The Degradation Problem
              </p>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  A 56-layer network has <em>higher training error</em> than a
                  20-layer network. Not higher test error (that would be
                  overfitting)&mdash;higher <strong>training</strong> error.
                </p>
                <p>
                  This rules out overfitting as the cause. If the deeper network
                  were memorizing, its training accuracy would be high and its test
                  accuracy would be low. Instead, it cannot even fit the training
                  data well.
                </p>
                <p>
                  Something is preventing the deeper network from learning what the
                  shallower network already can.
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              The next lesson explains what causes this and the elegant solution
              that enabled 152-layer networks.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Overfitting">
            The degradation problem is often confused with overfitting. They look
            different: overfitting shows high training accuracy + low test accuracy.
            Degradation shows low training accuracy in deeper networks&mdash;the
            network literally cannot learn as well.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Deeper networks learn richer hierarchical features.',
                description:
                  'Each layer detects patterns at the current scale. Deeper networks detect edges, then shapes, then parts, then whole objects. The same "spatial shrinks, channels grow" pattern from your MNIST CNN, applied more deeply.',
              },
              {
                headline: 'VGG\'s insight: stacking small 3x3 filters beats using large filters.',
                description:
                  'Two 3x3 convs cover the same receptive field as one 5x5, but use 28% fewer parameters and have twice the nonlinearity. Three 3x3 convs replace a 7x7 with 45% fewer parameters and triple the nonlinearity.',
              },
              {
                headline: 'Architecture evolution is problem-driven.',
                description:
                  'LeNet proved CNNs work. AlexNet solved vanishing gradients (ReLU) and overfitting (dropout). VGG found the design principle: disciplined depth with small filters. Each generation solved the previous one\'s limitation.',
              },
              {
                headline: 'Depth has a limit: the degradation problem.',
                description:
                  'Networks deeper than ~20 layers (pre-ResNet) performed worse on both training and test data. This is NOT overfitting. The next lesson explains why and introduces the solution.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            Depth buys hierarchy and receptive field, but each step must be earned
            with the right innovations. LeNet earned 5 layers with the
            conv-pool-fc pattern. AlexNet earned 8 with ReLU and dropout. VGG
            earned 19 with disciplined 3x3 stacking. Going beyond that requires
            something new.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/resnets"
            title="ResNets"
            description="We know deeper is better but deeper eventually breaks. The next lesson tackles the degradation problem head-on and introduces the residual connection&mdash;the innovation that took networks from 19 layers to 152."
            buttonText="Continue to ResNets"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
