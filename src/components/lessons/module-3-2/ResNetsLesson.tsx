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
import { CodeBlock } from '@/components/common/CodeBlock'
import { ResNetBlockExplorer } from '@/components/widgets/ResNetBlockExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * ResNets and Skip Connections
 *
 * Second lesson in Module 3.2 (Modern Architectures).
 * Resolves the degradation cliffhanger from Architecture Evolution,
 * explains residual connections, and implements a ResNet block in PyTorch.
 *
 * Core concepts at DEVELOPED:
 * - Degradation problem (cause + explanation)
 * - Residual connection / skip connection (F(x) + x)
 * - Batch normalization in CNN practice (nn.BatchNorm2d, Conv-BN-ReLU)
 * - Identity shortcut
 *
 * Concepts at INTRODUCED:
 * - Projection shortcut (1x1 conv for dimension matching)
 * - Global average pooling
 * - Full ResNet architecture overview
 *
 * Previous: Architecture Evolution (module 3.2, lesson 1)
 * Next: Transfer Learning (module 3.2, lesson 3)
 */

// ---------------------------------------------------------------------------
// Inline SVG: ResNet block diagram comparing plain vs residual
// ---------------------------------------------------------------------------

function ResNetBlockDiagram() {
  const blockW = 220
  const blockH = 260
  const gap = 40
  const totalW = blockW * 2 + gap
  const totalH = blockH + 40

  function plainBlock(offsetX: number) {
    const cx = offsetX + blockW / 2
    const startY = 40
    const layerH = 28
    const layerGap = 12

    const layers = [
      { label: 'Conv 3x3', y: startY, color: 'rgba(251, 146, 60, 0.3)', border: 'rgba(251, 146, 60, 0.6)' },
      { label: 'BN', y: startY + layerH + layerGap, color: 'rgba(168, 85, 247, 0.2)', border: 'rgba(168, 85, 247, 0.5)' },
      { label: 'ReLU', y: startY + 2 * (layerH + layerGap), color: 'rgba(56, 189, 248, 0.2)', border: 'rgba(56, 189, 248, 0.5)' },
      { label: 'Conv 3x3', y: startY + 3 * (layerH + layerGap), color: 'rgba(251, 146, 60, 0.3)', border: 'rgba(251, 146, 60, 0.6)' },
      { label: 'BN', y: startY + 4 * (layerH + layerGap), color: 'rgba(168, 85, 247, 0.2)', border: 'rgba(168, 85, 247, 0.5)' },
    ]

    const lastY = layers[layers.length - 1].y

    return (
      <g>
        <text x={cx} y={16} textAnchor="middle" fontSize={12} fill="currentColor" fontWeight={600}>
          Plain Block
        </text>
        {/* Input arrow */}
        <line x1={cx} y1={24} x2={cx} y2={startY - 2} stroke="currentColor" strokeWidth={1.5} opacity={0.4} />
        {layers.map((layer) => (
          <g key={`plain-${layer.label}-${layer.y}`}>
            <rect x={cx - 60} y={layer.y} width={120} height={layerH} rx={4} fill={layer.color} stroke={layer.border} strokeWidth={1.2} />
            <text x={cx} y={layer.y + layerH / 2 + 4} textAnchor="middle" fontSize={10} fill="currentColor">{layer.label}</text>
            {layer.y < lastY && (
              <line x1={cx} y1={layer.y + layerH} x2={cx} y2={layer.y + layerH + layerGap} stroke="currentColor" strokeWidth={1} opacity={0.3} />
            )}
          </g>
        ))}
        {/* Output label */}
        <text x={cx} y={lastY + layerH + 18} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
          Output = F(x)
        </text>
      </g>
    )
  }

  function residualBlock(offsetX: number) {
    const cx = offsetX + blockW / 2
    const startY = 40
    const layerH = 28
    const layerGap = 12

    const layers = [
      { label: 'Conv 3x3', y: startY, color: 'rgba(251, 146, 60, 0.3)', border: 'rgba(251, 146, 60, 0.6)' },
      { label: 'BN', y: startY + layerH + layerGap, color: 'rgba(168, 85, 247, 0.2)', border: 'rgba(168, 85, 247, 0.5)' },
      { label: 'ReLU', y: startY + 2 * (layerH + layerGap), color: 'rgba(56, 189, 248, 0.2)', border: 'rgba(56, 189, 248, 0.5)' },
      { label: 'Conv 3x3', y: startY + 3 * (layerH + layerGap), color: 'rgba(251, 146, 60, 0.3)', border: 'rgba(251, 146, 60, 0.6)' },
      { label: 'BN', y: startY + 4 * (layerH + layerGap), color: 'rgba(168, 85, 247, 0.2)', border: 'rgba(168, 85, 247, 0.5)' },
    ]

    const lastY = layers[layers.length - 1].y
    const addY = lastY + layerH + 12

    return (
      <g>
        <text x={cx} y={16} textAnchor="middle" fontSize={12} fill="currentColor" fontWeight={600}>
          Residual Block
        </text>
        {/* Input arrow */}
        <line x1={cx} y1={24} x2={cx} y2={startY - 2} stroke="currentColor" strokeWidth={1.5} opacity={0.4} />
        {layers.map((layer) => (
          <g key={`res-${layer.label}-${layer.y}`}>
            <rect x={cx - 60} y={layer.y} width={120} height={layerH} rx={4} fill={layer.color} stroke={layer.border} strokeWidth={1.2} />
            <text x={cx} y={layer.y + layerH / 2 + 4} textAnchor="middle" fontSize={10} fill="currentColor">{layer.label}</text>
            {layer.y < lastY && (
              <line x1={cx} y1={layer.y + layerH} x2={cx} y2={layer.y + layerH + layerGap} stroke="currentColor" strokeWidth={1} opacity={0.3} />
            )}
          </g>
        ))}
        {/* Arrow from last BN to + node */}
        <line x1={cx} y1={lastY + layerH} x2={cx} y2={addY - 2} stroke="currentColor" strokeWidth={1} opacity={0.3} />
        {/* + node */}
        <circle cx={cx} cy={addY + 10} r={12} fill="rgba(34, 197, 94, 0.15)" stroke="rgba(34, 197, 94, 0.5)" strokeWidth={2} />
        <text x={cx} y={addY + 14} textAnchor="middle" fontSize={14} fill="rgba(34, 197, 94, 0.9)" fontWeight={700}>+</text>
        {/* Skip connection */}
        <line x1={cx + 60} y1={startY + 14} x2={cx + 80} y2={startY + 14} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={cx + 80} y1={startY + 14} x2={cx + 80} y2={addY + 10} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={cx + 80} y1={addY + 10} x2={cx + 12} y2={addY + 10} stroke="rgba(34, 197, 94, 0.7)" strokeWidth={2} strokeDasharray="4 3" />
        <text x={cx + 90} y={(startY + 14 + addY + 10) / 2 + 4} fontSize={9} fill="rgba(34, 197, 94, 0.7)" fontWeight={500}>x (identity)</text>
        {/* Output label */}
        <text x={cx} y={addY + 32} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
          Output = F(x) + x
        </text>
      </g>
    )
  }

  return (
    <div className="flex justify-center py-2">
      <svg
        width={totalW}
        height={totalH}
        viewBox={`0 0 ${totalW} ${totalH}`}
        className="text-muted-foreground"
      >
        {plainBlock(0)}
        {residualBlock(blockW + gap)}
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ResNetsLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          1. Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="ResNets and Skip Connections"
            description="Why deeper networks fail to train&mdash;and the elegant solution that made 152-layer networks possible."
            category="Modern Architectures"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          1. Context + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain why residual (skip) connections solve the degradation problem and
            implement a ResNet block in PyTorch. By the end, you will understand the
            architectural idea behind every modern deep network.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In Architecture Evolution, you saw the degradation problem: a 56-layer
            network trains worse than a 20-layer one. You also wrote skip connection
            code in the nn.Module lesson:{' '}
            <code className="text-xs bg-muted px-1 rounded">self.linear(relu(x)) + x</code>.
            This lesson connects those threads.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The degradation problem: why plain deep networks fail even with BN + He init',
              'Residual connections: the identity mapping insight and F(x) + x formulation',
              'Batch normalization in practice: nn.BatchNorm2d, Conv-BN-ReLU, train/eval mode',
              'Implementing a ResNet block and small ResNet (Colab notebook)',
              'NOT: Bottleneck blocks (1x1-3x3-1x1 pattern used in ResNet-50+)',
              'NOT: ResNet variants (ResNeXt, DenseNet, WideResNet)',
              'NOT: Training tricks (LR scheduling, data augmentation, warmup)',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Module Arc">
            Architecture Evolution established &ldquo;deeper = better&rdquo; and its
            limits. This lesson solves the limit. Next: transfer learning puts
            pretrained ResNets to practical use.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          2. Hook (mystery resolution)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Unsolved Mystery"
            subtitle="A 56-layer network with every tool in our recipe still fails"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Architecture Evolution, you saw something surprising: a 56-layer
              network with batch normalization and He initialization&mdash;every training
              tool we know&mdash;has <strong>higher training error</strong> than a
              20-layer network.
            </p>
            <div className="py-4 px-6 bg-rose-500/5 border border-rose-500/20 rounded-lg space-y-3">
              <p className="text-sm font-medium text-rose-300">
                Ruling Out the Usual Suspects
              </p>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  <strong>Is it overfitting?</strong> No. Overfitting means low training
                  error but high test error&mdash;the network memorizes. Here, training
                  error itself is higher. The deeper network cannot even fit the training
                  data.
                </p>
                <p>
                  <strong>Is it vanishing gradients?</strong> Partly, but we applied
                  the fix: batch normalization stabilizes gradient flow, He initialization
                  sets the right starting scale. These tools let us train 20 layers just
                  fine. Yet 56 layers still degrades. Something deeper is wrong.
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              This is the <strong>degradation problem</strong>&mdash;and it blocked the
              field from going deeper than about 20 layers. What follows is the insight
              that broke through.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just Gradient Flow">
            If you are thinking &ldquo;vanishing gradients&rdquo; fully explains the
            degradation problem&mdash;not quite. BN + He init address gradient flow,
            yet degradation persists. The problem is about what the optimizer can
            find, not just what gradients it receives.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          3. Explain: The Identity Mapping Argument
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Identity Mapping Argument"
            subtitle="Why degradation is surprising&mdash;and what it reveals"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a thought experiment that reveals the real problem.
            </p>
            <p className="text-muted-foreground">
              Take a 20-layer network that trains well. Now bolt on 36 extra
              layers&mdash;but configure them as <strong>identity functions</strong> that
              simply pass their input through unchanged. This 56-layer network should be
              <strong> at least as good</strong> as the 20-layer one. The extra layers do
              nothing, so they cannot hurt.
            </p>
            <p className="text-muted-foreground">
              But the optimizer <strong>cannot find</strong> those identity mappings in
              plain layers. Why not?
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Concrete example: input x = 5.0, desired output H(x) = 5.1
              </p>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  <strong>Plain layer:</strong> Must learn H(x) = 5.1 directly. Weights
                  initialized near zero produce output near 0.0. The target (5.1) is far
                  from the starting point (0.0).
                </p>
                <p>
                  <strong>Residual formulation:</strong> Learn F(x) = H(x) - x = 5.1 -
                  5.0 = 0.1. Weights initialized near zero produce F(x) near 0.0, and
                  the output is F(x) + x = 0.0 + 5.0 = 5.0. The target (5.1) is only
                  0.1 away from the starting point (5.0).
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              The identity function is not the &ldquo;easy&rdquo; solution for a plain
              conv layer&mdash;it is a specific, non-trivial weight configuration. Plain
              layers initialized near zero output near zero, not the input. The optimizer
              must travel a long distance in weight space to find the identity. This is
              the degradation problem: <strong>the optimizer is stuck trying to
              approximate a complex function when a simpler solution exists but is not
              accessible</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Argument">
            If a 56-layer network were at least as powerful as a 20-layer one (it has
            more capacity), it should be at least as good. The fact that it is worse
            means the extra layers cannot learn even the identity function&mdash;the
            simplest possible mapping.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          4. Explain: The Residual Connection
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Residual Connection"
            subtitle="Learn the correction, not the whole function"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The solution is to restructure the block so the network learns a
              <strong> residual</strong> (a correction) instead of the full mapping.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                Instead of learning the desired mapping{' '}
                <InlineMath math="H(x)" /> directly, define:
              </p>
              <BlockMath math="H(x) = F(x) + x" />
              <p className="text-sm text-muted-foreground">
                Rearranging: <InlineMath math="F(x) = H(x) - x" />. The conv layers
                learn only the <strong>residual</strong>&mdash;the difference from the
                input.
              </p>
            </div>
            <p className="text-muted-foreground">
              If the optimal <InlineMath math="F(x)" /> is zero (the block should do
              nothing), the weights just stay near their initialization&mdash;easy. If
              the optimal <InlineMath math="F(x)" /> is non-trivial, the block learns
              that correction. Either way, the starting point is the identity function,
              not zero.
            </p>

            {/* Analogy */}
            <GradientCard title="Analogy: Editing vs Writing" color="blue">
              <div className="text-sm space-y-2">
                <p>
                  A <strong>plain block</strong> is like writing a document from scratch.
                  You start with a blank page and must reconstruct everything.
                </p>
                <p>
                  A <strong>residual block</strong> is like editing a draft. The input
                  is the draft (x), and the conv layers propose changes (F(x)). The
                  output is the edited version (F(x) + x). Editing is easier because you
                  start from something good and only change what needs changing.
                </p>
              </div>
            </GradientCard>

            {/* Block diagram */}
            <ResNetBlockDiagram />
            <p className="text-xs text-muted-foreground text-center">
              Left: plain block outputs F(x) only. Right: residual block adds the
              input back via a skip connection, outputting F(x) + x.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Key Reframe">
            The residual connection changes <em>what</em> the network learns. Instead
            of learning the full mapping from scratch, each block learns only the
            <strong> difference from identity</strong>. Making &ldquo;do nothing&rdquo;
            the default behavior, not the hardest.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          5. Check 1: Predict-and-verify
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: What Happens at Zero?"
            subtitle="Predict before you peek"
          />
          <GradientCard title="Predict and Verify" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> In a ResNet block, what happens if the conv
                layers learn weights that are all exactly zero?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-1">
                  <p>
                    F(x) = 0, so output = 0 + x = <strong>x</strong>. The block becomes
                    an identity function. This is the default safe behavior&mdash;the block
                    does nothing harmful.
                  </p>
                </div>
              </details>
              <p className="pt-2">
                <strong>Question 2:</strong> What happens in a <em>plain</em> block if
                weights are near zero?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-1">
                  <p>
                    Output is near zero, <strong>not</strong> near x. The input is lost.
                    A plain block cannot safely &ldquo;do nothing&rdquo;&mdash;its default
                    behavior destroys the signal.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The &ldquo;At Least as Good&rdquo; Guarantee">
            A residual block that learns nothing (F(x) = 0) is not harmful&mdash;it
            just passes the input through. A plain block that learns nothing outputs
            garbage. This is why ResNets can be deeper without degrading: the worst
            case for each block is identity, not zero.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          6. Explain: Batch Normalization in Practice
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Batch Normalization in Practice"
            subtitle="From concept to Conv-BN-ReLU"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now understand the conceptual insight behind residual connections.
              To actually build a ResNet block, you need one more piece: batch
              normalization in its CNN form. You already know what BN does
              conceptually from Training Dynamics&mdash;normalize activations between
              layers to stabilize gradient flow. Here is how to use it in practice:
            </p>
            <CodeBlock
              code={`# The Conv-BN-ReLU pattern (the ResNet building block)
import torch.nn as nn
import torch.nn.functional as F

# One conv layer with batch norm:
conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
bn = nn.BatchNorm2d(64)   # one value per channel

# In forward():
out = F.relu(bn(conv(x)))   # Conv -> BN -> ReLU`}
              language="python"
              filename="conv_bn_relu.py"
            />
            <p className="text-muted-foreground">
              <code className="text-xs bg-muted px-1 rounded">nn.BatchNorm2d(channels)</code>{' '}
              takes the number of channels (not spatial size). It normalizes each
              channel independently across the spatial dimensions and the batch, then
              applies learned <InlineMath math="\gamma" /> (scale) and{' '}
              <InlineMath math="\beta" /> (shift) parameters.
            </p>
            <p className="text-muted-foreground">
              Those learned parameters are what make BN a <strong>trainable layer</strong>,
              not just preprocessing. If the network wants to undo the normalization for a
              particular channel, <InlineMath math="\gamma" /> and{' '}
              <InlineMath math="\beta" /> can learn to do that.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="BN is Not Input Normalization">
            A common misconception: &ldquo;batch norm is just normalizing inputs
            applied at every layer.&rdquo; Input normalization is fixed (zero-mean,
            unit-variance). BN has <em>learned</em> parameters that can undo the
            normalization if useful. It is a trainable layer, not a preprocessing step.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Train vs Eval Mode
            </p>
            <p className="text-muted-foreground">
              Batch norm behaves differently during training and inference:
            </p>
            <ComparisonRow
              left={{
                title: 'model.train()',
                color: 'blue',
                items: [
                  'Uses per-batch statistics (mean & variance)',
                  'Updates running averages for later',
                  'Adds slight noise (regularization effect)',
                ],
              }}
              right={{
                title: 'model.eval()',
                color: 'amber',
                items: [
                  'Uses stored running averages',
                  'Deterministic output',
                  'Must switch before inference!',
                ],
              }}
            />
            <p className="text-muted-foreground">
              This is why <code className="text-xs bg-muted px-1 rounded">model.train()</code>{' '}
              and <code className="text-xs bg-muted px-1 rounded">model.eval()</code> matter.
              Without switching to eval mode before inference, batch norm uses the
              current batch&apos;s statistics instead of the learned running averages,
              producing inconsistent predictions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Practical Rule">
            Before training: <code className="text-xs bg-muted px-1 rounded">model.train()</code>.
            Before evaluation/inference:{' '}
            <code className="text-xs bg-muted px-1 rounded">model.eval()</code>.
            Always. This also matters for dropout, which you already know.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          7. Explain: The Full ResNet Block
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full ResNet Block"
            subtitle="Putting Conv-BN-ReLU into a residual structure"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The standard ResNet basic block combines everything you have seen:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground font-medium">
                Basic ResNet Block (used in ResNet-18 and ResNet-34):
              </p>
              <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-1 ml-2">
                <li>Conv(3x3) &rarr; BN &rarr; ReLU</li>
                <li>Conv(3x3) &rarr; BN</li>
                <li>Add the shortcut (input x)</li>
                <li>ReLU</li>
              </ol>
              <p className="text-xs text-muted-foreground mt-2 opacity-70">
                Note: ReLU comes after the addition, not before. The addition can produce
                any value; ReLU ensures positive outputs for the next block.
              </p>
            </div>
            <p className="text-muted-foreground">
              Compare this to a VGG block (two 3x3 convs with BN and ReLU): it is
              the <strong>same computation</strong>, just with one added line&mdash;the
              shortcut connection. Take exactly what you know from VGG, add the shortcut.
            </p>
            <WarningBlock title="Why Two Convs, Not One?">
              The skip connection wraps <em>two</em> conv layers, not one. A skip around a
              single conv would collapse to a near-identity function&mdash;the lone conv
              only needs to learn a tiny residual, giving it almost no incentive to learn
              anything useful. Grouping 2+ layers into a block gives the conv path enough
              capacity to learn meaningful corrections while the shortcut still provides
              the identity baseline.
            </WarningBlock>
            <CodeBlock
              code={`class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                         # save the input
        out = F.relu(self.bn1(self.conv1(x))) # Conv-BN-ReLU
        out = self.bn2(self.conv2(out))       # Conv-BN (no ReLU yet)
        out = out + identity                  # add the shortcut
        out = F.relu(out)                     # ReLU after addition
        return out`}
              language="python"
              filename="resnet_block.py"
            />
            <p className="text-muted-foreground">
              This is the code you saw in the nn.Module lesson&mdash;
              <code className="text-xs bg-muted px-1 rounded">self.conv(x) + x</code>&mdash;extended
              to two conv layers with batch norm. The LEGO brick pattern holds:
              <code className="text-xs bg-muted px-1 rounded">__init__</code> defines
              the layers, <code className="text-xs bg-muted px-1 rounded">forward</code>{' '}
              wires them together.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why ReLU After Addition?">
            If ReLU came before the addition, the shortcut path would still carry raw
            (potentially negative) values while the conv path would be non-negative.
            Applying ReLU after the addition ensures both contributions are combined
            first, then activated together.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          8. Explain: Dimension Mismatch
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dimension Mismatch"
            subtitle="When the shortcut needs help"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The identity shortcut (x + F(x)) only works when x and F(x) have the
              same shape. But what happens when a block changes spatial size (stride=2)
              or channel count (e.g., 64 to 128)?
            </p>
            <ComparisonRow
              left={{
                title: 'Identity Shortcut',
                color: 'emerald',
                items: [
                  'Input & output have same shape',
                  'Just add: out + x',
                  'No extra parameters',
                  'Used in most blocks',
                ],
              }}
              right={{
                title: 'Projection Shortcut',
                color: 'violet',
                items: [
                  'Input & output shapes differ',
                  '1x1 conv adjusts dimensions',
                  'Matches channels + spatial size',
                  'Used at stage transitions',
                ],
              }}
            />
            <p className="text-muted-foreground">
              A <strong>1x1 convolution</strong> is a per-pixel linear transformation
              that changes channel count without changing spatial size. With an
              appropriate stride, it can also downsample spatially:
            </p>
            <CodeBlock
              code={`# Projection shortcut for dimension mismatch
# Input: 64 channels, 32x32 spatial
# Output: 128 channels, 16x16 spatial

self.shortcut = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=1, stride=2),  # 1x1 conv
    nn.BatchNorm2d(128),
)

# In forward:
out = F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))`}
              language="python"
              filename="projection_shortcut.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="1x1 Convolutions">
            A 1x1 convolution sounds odd&mdash;a filter that covers one pixel? It does
            not look at spatial neighbors. Instead, it linearly combines channels at each
            position, like a per-pixel dense layer. Here, it projects 64 channels to 128
            with stride=2 to match the output dimensions.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          9. Check 2: Transfer question
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Projection Shortcut"
            subtitle="Apply what you just learned"
          />
          <GradientCard title="Transfer Question" color="emerald">
            <div className="space-y-3 text-sm">
              <p>
                You are building a ResNet and your block takes <strong>64-channel
                input at 32x32</strong> but produces <strong>128-channel output at
                16x16</strong> (stride=2 in the first conv). Can you use a simple
                identity shortcut? What do you need instead?
              </p>
              <details>
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal answer
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    No&mdash;dimensions mismatch. The input x is 64-channel, 32x32.
                    The output F(x) is 128-channel, 16x16. You cannot add them.
                  </p>
                  <p>
                    You need a <strong>projection shortcut</strong>: a 1x1 convolution
                    from 64 to 128 channels with stride=2, followed by BN. This
                    transforms the shortcut to match:{' '}
                    <code className="text-xs bg-muted px-1 rounded">Conv2d(64, 128, 1, stride=2)</code> +{' '}
                    <code className="text-xs bg-muted px-1 rounded">BatchNorm2d(128)</code>.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          10. Explain: Global Average Pooling
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Global Average Pooling"
            subtitle="Replacing VGG's expensive FC layers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now have the two core ingredients of a ResNet: residual connections
              and batch normalization. One more architectural decision completes the
              picture&mdash;how to go from the final feature maps to a class prediction.
            </p>
            <p className="text-muted-foreground">
              Remember how VGG-16 ends: flatten the 7x7x512 feature maps into a
              25,088-element vector, then pass through FC layers with 103 million
              parameters. Most of VGG&apos;s parameters live in those FC layers.
            </p>
            <p className="text-muted-foreground">
              ResNet replaces this with <strong>global average pooling</strong>: average
              each channel&apos;s entire spatial grid into a single number. If you have
              512 channels at 7x7, you get a 512-element vector. No learnable parameters.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground font-medium">
                VGG ending vs ResNet ending:
              </p>
              <div className="space-y-1 text-sm text-muted-foreground font-mono">
                <p>VGG:    7x7x512 &rarr; flatten(25,088) &rarr; FC(4096) &rarr; FC(4096) &rarr; FC(1000)</p>
                <p>ResNet: 7x7x512 &rarr; AvgPool(7x7) &rarr; 512 &rarr; FC(1000)</p>
              </div>
              <p className="text-xs text-muted-foreground mt-2 opacity-70">
                VGG FC params: ~103M. ResNet FC params: ~512K. A 200x reduction.
              </p>
            </div>
            <p className="text-muted-foreground">
              In PyTorch:{' '}
              <code className="text-xs bg-muted px-1 rounded">nn.AdaptiveAvgPool2d(1)</code>{' '}
              averages any spatial size down to 1x1, giving you one value per channel.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why It Works">
            By the final layer, each channel is a feature detector (e.g., &ldquo;dog
            ear detector&rdquo;). Averaging that 7x7 grid asks: &ldquo;how much of this
            feature is present overall?&rdquo; The spatial location no longer matters for
            classification&mdash;just the presence.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          11. Explore: ResNet Block Explorer widget
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: Plain vs Residual Block"
            subtitle="Toggle the skip connection and observe what changes"
          />
          <p className="text-muted-foreground mb-4">
            This widget simulates a simplified block with a single weight parameter.
            Toggle between plain and residual modes, adjust the conv weight, and observe
            the output value and gradient flow.
          </p>
          <ExercisePanel
            title="ResNet Block Explorer"
            subtitle="Compare plain and residual block behavior"
          >
            <ResNetBlockExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>
                Set w = 0 in both modes. In residual mode, output = x (identity).
                In plain mode, output = 0 (signal lost).
              </li>
              <li>
                Watch the gradient bars. The skip path always contributes
                gradient = 1.0. Even when the conv path gradient is tiny, the total
                gradient stays healthy.
              </li>
              <li>
                Set w = 0.5 in residual mode. The output is F(x) + x = 0.5x + x =
                7.5 (a correction on top of identity).
              </li>
              <li>
                Try negative w. In plain mode, the output flips sign. In residual
                mode, the shortcut anchors the output near x.
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          12. Elaborate: Why It Works (Deeper)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why It Works: Two Perspectives"
            subtitle="Residual learning + gradient flow"
          />
          <div className="space-y-4">
            <GradientCard title="Perspective 1: Easier Optimization" color="violet">
              <div className="text-sm space-y-2">
                <p>
                  Skip connections create smoother loss landscapes with fewer sharp
                  minima. The optimizer has an easier time navigating toward good
                  solutions because the &ldquo;do nothing&rdquo; path is always
                  available as a fallback.
                </p>
                <p>
                  This is the <strong>fundamental insight</strong>: the residual
                  formulation changes what the network must learn, making optimization
                  easier.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Perspective 2: Gradient Highway" color="blue">
              <div className="text-sm space-y-2">
                <p>
                  The skip path has derivative 1.0 with respect to its input. Even if
                  the conv path&apos;s gradients vanish (because its local derivatives
                  are small), the skip path provides a <strong>direct gradient
                  highway</strong> back to earlier layers.
                </p>
                <p>
                  Remember the telephone game from Training Dynamics? The skip
                  connection is a direct phone line that bypasses the chain of whispered
                  messages. Even if the chain garbles the signal, the direct line carries
                  it intact.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Gradient Highway is Incomplete">
            The &ldquo;gradient highway&rdquo; explanation is widely repeated and
            partially true. But if gradient flow were the whole story, batch
            normalization alone would suffice. The deeper insight is the <strong>residual
            learning formulation</strong>&mdash;making identity the default. Gradient
            flow is a consequence of the architecture, not the full motivation.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          13. Elaborate: The ResNet Family
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The ResNet Family"
            subtitle="From 18 to 152 layers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              With the residual connection in hand, He et al. built a family of networks
              at unprecedented depths:
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-muted-foreground">
                <thead>
                  <tr className="border-b border-muted/30">
                    <th className="text-left py-2 pr-4 font-medium">Model</th>
                    <th className="text-left py-2 pr-4 font-medium">Layers</th>
                    <th className="text-left py-2 pr-4 font-medium">Block Type</th>
                    <th className="text-left py-2 pr-4 font-medium">Params</th>
                    <th className="text-left py-2 font-medium">Top-5 Acc</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">ResNet-18</td>
                    <td className="py-2 pr-4">18</td>
                    <td className="py-2 pr-4">Basic (2 convs)</td>
                    <td className="py-2 pr-4">11.7M</td>
                    <td className="py-2">89.1%</td>
                  </tr>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">ResNet-34</td>
                    <td className="py-2 pr-4">34</td>
                    <td className="py-2 pr-4">Basic (2 convs)</td>
                    <td className="py-2 pr-4">21.8M</td>
                    <td className="py-2">91.4%</td>
                  </tr>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">ResNet-50</td>
                    <td className="py-2 pr-4">50</td>
                    <td className="py-2 pr-4">Bottleneck (3 convs)</td>
                    <td className="py-2 pr-4">25.6M</td>
                    <td className="py-2">92.2%</td>
                  </tr>
                  <tr className="border-b border-muted/10">
                    <td className="py-2 pr-4">ResNet-101</td>
                    <td className="py-2 pr-4">101</td>
                    <td className="py-2 pr-4">Bottleneck (3 convs)</td>
                    <td className="py-2 pr-4">44.5M</td>
                    <td className="py-2">93.0%</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4">ResNet-152</td>
                    <td className="py-2 pr-4">152</td>
                    <td className="py-2 pr-4">Bottleneck (3 convs)</td>
                    <td className="py-2 pr-4">60.2M</td>
                    <td className="py-2">93.3%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground opacity-60">
              Approximate single-crop top-5 accuracy on ImageNet validation set.
            </p>
            <p className="text-muted-foreground">
              ResNet-50 and above use <strong>bottleneck blocks</strong>: a 1x1 conv
              reduces channels, a 3x3 conv processes them, and another 1x1 conv expands
              channels back. This is more parameter-efficient for very deep networks.
              We will not develop bottleneck blocks here&mdash;the basic block you just
              learned is the same principle, and bottleneck blocks are a straightforward
              extension.
            </p>
            <p className="text-muted-foreground">
              Notice: ResNet-152 has <strong>fewer parameters</strong> than VGG-16 (60M
              vs 138M) despite being 8x deeper. Global average pooling eliminates the
              massive FC layers, and residual connections make each layer more efficient.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Architecture Evolution Continues">
            The pattern from the previous lesson holds: each generation solves the
            previous one&apos;s limitation. VGG proved depth matters. ResNet solved
            the degradation problem that limited depth. Every major model you will
            encounter from here&mdash;U-Net, Transformers, GPT&mdash;uses some form
            of residual connection.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          14. Practice: Colab Notebook
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Implement a ResNet"
            subtitle="Build and train a ResNet on CIFAR-10"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Time to write the code. The notebook provides data loading, training
              loops, and a plain deep CNN baseline. You implement:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>A <code className="text-xs bg-muted px-1 rounded">ResidualBlock</code> class with Conv-BN-ReLU-Conv-BN + shortcut</li>
              <li>A small ResNet model that stacks these blocks</li>
              <li>Compare the ResNet to a plain network of similar depth</li>
            </ul>
            <p className="text-muted-foreground">
              The key moment: watch the ResNet train successfully at depths where the
              plain network degrades.
            </p>

            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Open the scaffolded notebook in Google Colab. The training loop and
                  data loading are provided&mdash;your job is the architecture.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/3-2-2-resnets.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook includes a plain CNN baseline, TODO sections for your
                  ResNet implementation, a training loop, and accuracy comparison.
                  Expected training time: ~5 minutes on a Colab GPU.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What&apos;s Provided">
            <ul className="space-y-1 text-sm">
              <li>CIFAR-10 data loading with DataLoader</li>
              <li>Plain deep CNN (no skip connections)</li>
              <li>Training + evaluation loop</li>
              <li>Accuracy comparison code</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          15. Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'The degradation problem is an optimization failure, not overfitting.',
                description:
                  'Plain networks deeper than ~20 layers have higher TRAINING error. Even with BN and He init, the optimizer cannot find identity mappings in plain layers. The extra capacity goes to waste.',
              },
              {
                headline: 'Residual connections make identity the default behavior.',
                description:
                  'H(x) = F(x) + x. The conv layers learn the residual (correction from identity). If F(x) = 0, the block passes input through unchanged. "Editing a document, not writing from scratch."',
              },
              {
                headline: 'Batch normalization in practice: Conv-BN-ReLU.',
                description:
                  'nn.BatchNorm2d(channels) normalizes per channel with learned scale and shift. Use model.train() for training (batch statistics) and model.eval() for inference (running averages).',
              },
              {
                headline: 'Global average pooling eliminates expensive FC layers.',
                description:
                  'Average each channel\'s spatial grid into one number. ResNet-152 has fewer parameters than VGG-16 despite being 8x deeper.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            A residual block starts from identity and learns to deviate&mdash;making
            &ldquo;do nothing&rdquo; the easiest path, not the hardest. This single
            insight enabled networks of 152 layers and appears in every major
            architecture you will study from here forward.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          16. Next Step
          ================================================================ */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/transfer-learning"
            title="Transfer Learning"
            description="You now understand the architecture that made modern deep learning possible. The next lesson puts it to practical use: take a pretrained ResNet and adapt it to new tasks in minutes instead of training from scratch."
            buttonText="Continue to Transfer Learning"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
