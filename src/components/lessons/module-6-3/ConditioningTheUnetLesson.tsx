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
  ConceptBlock,
  GradientCard,
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { NormalizationComparisonWidget } from '@/components/widgets/NormalizationComparisonWidget'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ArchitectureDiagram } from '@/components/widgets/ArchitectureDiagram'
import { UNET_CONDITIONING_DATA } from '@/components/widgets/ArchitectureDiagram/unet-conditioning-data'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Conditioning the U-Net
 *
 * Lesson 2 in Module 6.3 (Architecture & Conditioning). Lesson 11 overall in Series 6.
 * Cognitive load: BUILD (re-entering theoretical mode with heavily leveraged prior knowledge).
 *
 * Teaches how the U-Net knows what noise level it is denoising:
 * - Sinusoidal timestep embeddings (direct transfer from positional encoding)
 * - Adaptive group normalization (timestep-dependent scale and shift)
 * - Why conditioning must happen at every residual block, not just the input
 * - Completing the pseudocode from the previous lesson
 *
 * Core concepts:
 * - Sinusoidal timestep embedding: DEVELOPED
 * - Adaptive group normalization: DEVELOPED
 * - Group normalization: INTRODUCED (elevated from MENTIONED)
 * - Global conditioning pattern: INTRODUCED
 * - FiLM conditioning: MENTIONED
 *
 * Previous: The U-Net Architecture (module 6.3, lesson 1)
 * Next: CLIP (module 6.3, lesson 3)
 */

export function ConditioningTheUnetLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Conditioning the U-Net"
            description="How sinusoidal embeddings and adaptive normalization let a single U-Net handle every noise level&mdash;from pure static to nearly clean."
            category="Architecture & Conditioning"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how the U-Net receives the timestep t and uses it to modulate
            its behavior at every layer&mdash;via sinusoidal timestep embeddings and
            adaptive group normalization. By the end, the pseudocode from{' '}
            <strong>The U-Net Architecture</strong> will be complete.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <strong>Embeddings and Position</strong>, you implemented sinusoidal
            positional encoding from the formula&mdash;the &ldquo;clock with many
            hands.&rdquo; In <strong>Build a Diffusion Model</strong>, you used
            a simple linear timestep projection. In <strong>The U-Net
            Architecture</strong>, the pseudocode had a comment: &ldquo;timestep
            parameter omitted&mdash;next lesson.&rdquo; This is that lesson.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How the timestep integer t becomes a rich embedding vector (sinusoidal encoding + MLP)',
              'How that embedding modulates the U-Net at every layer (adaptive group normalization)',
              'Completing the pseudocode from the previous lesson with the timestep parameter',
              'NOT: text conditioning or cross-attention\u2014that comes in a later lesson',
              'NOT: CLIP or text embeddings\u2014the next lesson',
              'NOT: classifier-free guidance',
              'NOT: implementation\u2014the conceptual BUILD; full implementation is in the Module 6.4 capstone',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — Group Normalization (gap fill) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap: Group Normalization"
            subtitle="The normalization variant diffusion models actually use"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>The U-Net Architecture</strong>, group normalization was
              mentioned by name: &ldquo;diffusion U-Nets use group norm instead of
              batch norm.&rdquo; Before we build on it, you need to understand what
              it actually does.
            </p>
            <p className="text-muted-foreground">
              You already know two normalization variants. <strong>Batch
              normalization</strong> computes mean and variance{' '}
              <em>across the batch</em> for each channel&mdash;it asks &ldquo;how
              does this feature behave across all images in the batch?&rdquo;{' '}
              <strong>Layer normalization</strong> computes mean and variance{' '}
              <em>across all channels</em> for each example&mdash;it asks &ldquo;how
              do all features relate within this one example?&rdquo;
            </p>
            <p className="text-muted-foreground">
              <strong>Group normalization</strong> is the middle ground. It divides
              the channels into groups (typically 32 groups) and normalizes{' '}
              <em>within each group</em>. Like layer norm, it computes statistics
              per-example, so it does not depend on batch size. Like batch norm, it
              only normalizes subsets of channels, so different groups can capture
              different feature distributions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why Group Norm for Diffusion?">
            Diffusion training uses small batch sizes because images are
            large (64&times;64 or 256&times;256). Batch norm needs a large batch for
            reliable statistics. Group norm computes statistics per-example, so it
            works with any batch size&mdash;even batch size 1.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Which elements are normalized together? Each row is one example
              in a batch, each column is one channel. Same-color cells share a
              mean/variance computation:
            </p>
            <NormalizationComparisonWidget />
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Batch Norm" color="amber">
                <ul className="space-y-1">
                  <li>&bull; Normalize across <strong>batch</strong></li>
                  <li>&bull; Per channel</li>
                  <li>&bull; Needs large batches</li>
                  <li>&bull; Good for classification CNNs</li>
                </ul>
              </GradientCard>
              <GradientCard title="Layer Norm" color="blue">
                <ul className="space-y-1">
                  <li>&bull; Normalize across <strong>all channels</strong></li>
                  <li>&bull; Per example</li>
                  <li>&bull; Any batch size</li>
                  <li>&bull; Good for transformers</li>
                </ul>
              </GradientCard>
              <GradientCard title="Group Norm" color="emerald">
                <ul className="space-y-1">
                  <li>&bull; Normalize within <strong>channel groups</strong></li>
                  <li>&bull; Per example</li>
                  <li>&bull; Any batch size</li>
                  <li>&bull; Good for diffusion U-Nets</li>
                </ul>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The formula is the same as batch norm: normalize the features to zero
              mean and unit variance, then apply a learned scale{' '}
              <InlineMath math="\gamma" /> and shift{' '}
              <InlineMath math="\beta" />. The only difference is{' '}
              <em>which set of values</em> you compute the mean and variance over.
              Remember those learned <InlineMath math="\gamma" /> and{' '}
              <InlineMath math="\beta" /> parameters&mdash;they are about to become
              very important.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 4: Hook — The Pseudocode Is Incomplete */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Pseudocode Is Incomplete"
            subtitle="One network, a thousand different tasks"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Recall the pseudocode from <strong>The U-Net Architecture</strong>.
              The function signature had a conspicuous gap:
            </p>
            <CodeBlock
              code={`def forward(self, x):  # timestep t omitted — next lesson`}
              language="python"
              filename="unet_pseudocode.py"
            />
            <p className="text-muted-foreground">
              This is that next lesson. The U-Net has <strong>one set of
              weights</strong>. It must denoise at{' '}
              <strong>t&nbsp;=&nbsp;900</strong> (hallucinating structure from pure
              static) <em>and</em> at <strong>t&nbsp;=&nbsp;50</strong> (polishing
              fine details in a nearly clean image). These are radically different
              tasks. How does the same network do both?
            </p>
            <p className="text-muted-foreground">
              In <strong>Build a Diffusion Model</strong>, the capstone&rsquo;s
              answer was: normalize t to [0, 1], pass through a 2-layer MLP, and add
              the result to the bottleneck features. That &ldquo;worked&rdquo; for
              MNIST. But it is like describing your GPS position with a single number
              instead of latitude, longitude, and altitude. And it only injects
              conditioning at <em>one</em> point (the bottleneck), not at every
              processing stage.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Promise">
            You already know the formula for the solution. You saw it in{' '}
            <strong>Embeddings and Position</strong>&mdash;the sinusoidal positional
            encoding. Same formula, different input, different question.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — Sinusoidal Timestep Embedding */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sinusoidal Timestep Embedding"
            subtitle="Same clock, different question"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In transformers, sinusoidal positional encoding tells the model{' '}
              <em>where</em> a token sits in the sequence. In diffusion, the same
              formula tells the model <em>how noisy</em> the input is. Side by side:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Building Blocks, Different Question">
            This is the recurring pattern. Sinusoidal encoding is a{' '}
            <strong>building block</strong>. In transformers, it answers
            &ldquo;what position?&rdquo; In diffusion, it answers &ldquo;what
            noise level?&rdquo; Same mechanism, different purpose.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-foreground">
                Positional Encoding (Transformers):
              </p>
              <BlockMath math="\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)" />
              <p className="text-sm font-medium text-foreground">
                Timestep Embedding (Diffusion):
              </p>
              <BlockMath math="\text{TE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right)" />
            </div>
            <p className="text-muted-foreground">
              <strong>Same formula. Different input. Different question.</strong>{' '}
              Replace &ldquo;pos&rdquo; with &ldquo;t&rdquo; and you have the
              timestep embedding. The four requirements you learned in{' '}
              <strong>Embeddings and Position</strong> still hold:{' '}
              <em>unique</em> (each timestep gets a distinct pattern),{' '}
              <em>smooth</em> (nearby timesteps produce similar embeddings),{' '}
              <em>any range</em> (works for 1000 timesteps or 10,000), and{' '}
              <em>deterministic</em> (no learned parameters needed for the
              encoding itself).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Clock Analogy">
            Remember the &ldquo;clock with many hands&rdquo;? The second hand
            (high-frequency dimension) distinguishes t=500 from t=501. The hour
            hand (low-frequency dimension) distinguishes &ldquo;early in
            denoising&rdquo; from &ldquo;late in denoising.&rdquo; Same clock,
            now telling noise-level time.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Concrete example: specific timestep values */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Concrete example. Compute the sinusoidal embedding for t=500 and t=50
              at four dimensions (using d=8 for illustration&mdash;small enough to
              see the frequency progression clearly):
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  t = 500 (heavy noise)
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>&bull; dim 0 (i=0): sin(500/1) = <strong>&minus;0.47</strong> &mdash; oscillates rapidly</li>
                  <li>&bull; dim 2 (i=1): sin(500/10) = <strong>&minus;0.26</strong></li>
                  <li>&bull; dim 4 (i=2): sin(500/100) = <strong>&minus;0.96</strong></li>
                  <li>&bull; dim 6 (i=3): sin(500/1000) = <strong>0.48</strong> &mdash; changes slowly</li>
                </ul>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  t = 50 (light noise)
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>&bull; dim 0 (i=0): sin(50/1) = <strong>&minus;0.26</strong></li>
                  <li>&bull; dim 2 (i=1): sin(50/10) = <strong>&minus;0.96</strong></li>
                  <li>&bull; dim 4 (i=2): sin(50/100) = <strong>0.48</strong></li>
                  <li>&bull; dim 6 (i=3): sin(50/1000) = <strong>0.05</strong></li>
                </ul>
              </div>
            </div>
            <p className="text-muted-foreground">
              The denominators are 10000<sup>0/8</sup>=1, 10000<sup>2/8</sup>=10,
              10000<sup>4/8</sup>=100, 10000<sup>6/8</sup>=1000&mdash;exponentially
              increasing, just like the formula predicts. The high-frequency
              dimensions (dim 0) oscillate rapidly, capturing fine differences
              between adjacent timesteps. The low-frequency dimensions (dim 6)
              change slowly, capturing the broad distinction between &ldquo;lots
              of noise&rdquo; and &ldquo;almost clean.&rdquo; This is the clock
              analogy in action: the second hand and the hour hand encode different
              granularities. (Real models use d=128 or d=256, spreading the
              frequency range across many more dimensions.)
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Why not the simple projection? */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Simple Projection (Capstone)',
              color: 'rose',
              items: [
                't -> normalize -> MLP -> 128-dim vector',
                'All info encoded along learned directions',
                'No guaranteed smoothness between t and t+1',
                'Like describing position with a single number',
                'Worked for MNIST; fails at scale',
              ],
            }}
            right={{
              title: 'Sinusoidal + MLP (Real DDPMs)',
              color: 'emerald',
              items: [
                't -> sinusoidal encoding -> MLP -> 512-dim vector',
                'Multi-frequency representation from the start',
                'Adjacent timesteps are inherently similar',
                'Like GPS coordinates: latitude + longitude + altitude',
                'Rich input gives the MLP much more to work with',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* Negative example: random embeddings */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>What if the embeddings were random?</strong> If each timestep
              mapped to a random vector (like an untrained lookup table), t=500 and
              t=501 would have completely unrelated embeddings. The network would
              treat adjacent timesteps as independent tasks, unable to leverage the
              fact that denoising at t=500 is nearly identical to denoising at t=501.
              Smooth embeddings let the network generalize across nearby
              timesteps&mdash;a prediction for t=500 transfers to t=501 for free.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Lookup Table">
            Token embeddings in language models use{' '}
            <code className="text-xs">nn.Embedding</code>&mdash;a learned lookup
            table. Timestep embeddings do NOT. There are up to 1,000 timesteps,
            and smoothness between adjacent timesteps is critical. The sinusoidal
            encoding provides that smoothness by design.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The MLP refinement step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The sinusoidal encoding is not the final embedding. It is the{' '}
              <em>input</em> to a 2-layer MLP:
            </p>
            <CodeBlock
              code={`# Timestep embedding pipeline
t_emb = sinusoidal_encoding(t)       # t -> 256-dim (structured frequencies)
t_emb = Linear(256, 512)(t_emb)      # Project to hidden dimension
t_emb = GELU(t_emb)                  # Nonlinearity
t_emb = Linear(512, 512)(t_emb)      # Final timestep embedding (512-dim)`}
              language="python"
              filename="timestep_mlp.py"
            />
            <p className="text-muted-foreground">
              The MLP learns to combine and transform the raw frequency components
              into features that are useful for denoising. This is a standard
              pattern: provide a structured input (the sinusoidal encoding) and let
              the network refine it (the MLP). The sinusoidal part has no learned
              parameters&mdash;the learning happens entirely in the MLP.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Practical Dimensions">
            In practice, 128 or 256 sinusoidal dimensions are common, processed
            by an MLP into a 512-dim embedding. More sinusoidal dimensions means
            finer frequency discrimination, but with diminishing returns.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check — Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> The sinusoidal encoding for t=500 and
                t=501&mdash;would you expect them to be nearly identical, completely
                different, or somewhere in between?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Nearly identical.</strong> Sine and cosine are smooth
                    functions&mdash;a change of 1 in the input barely changes the
                    output. This is the smoothness requirement from positional
                    encoding, now applied to timesteps. The network can generalize
                    across adjacent timesteps.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> The sinusoidal encoding for t=500 and
                t=50&mdash;same question.
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <strong>Quite different.</strong> The &ldquo;hour hand&rdquo;
                    dimensions have changed significantly over 450 steps. This
                    captures the fact that denoising at t=500 (heavy noise,
                    structural decisions) and t=50 (light noise, detail
                    polishing) are fundamentally different tasks.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — Adaptive Group Normalization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Adaptive Group Normalization"
            subtitle="Making gamma and beta depend on the timestep"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The timestep embedding is a 512-dimensional vector. The U-Net
              processes 2D feature maps. How does the vector influence the feature
              maps?
            </p>
            <p className="text-muted-foreground">
              Start from what you know. In standard batch norm or group norm, after
              normalizing the features, you apply a learned scale{' '}
              <InlineMath math="\gamma" /> and shift{' '}
              <InlineMath math="\beta" />. These are{' '}
              <strong>fixed parameters</strong>&mdash;the same{' '}
              <InlineMath math="\gamma" /> and <InlineMath math="\beta" /> for
              every input, every timestep. They are learned during training and then
              frozen.
            </p>
            <p className="text-muted-foreground">
              The conceptual move is small:{' '}
              <strong>
                what if <InlineMath math="\gamma" /> and{' '}
                <InlineMath math="\beta" /> depended on the timestep?
              </strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Minimal Delta">
            Standard normalization has fixed tone controls set during training.
            Adaptive normalization has tone controls that <em>respond to the
            timestep signal</em>&mdash;like a graphic equalizer that adjusts based
            on what genre is playing.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* AdaGN formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm font-medium text-foreground">
                Standard Group Norm:
              </p>
              <BlockMath math="\text{GN}(x) = \gamma \cdot \text{Normalize}(x) + \beta" />
              <p className="text-sm font-medium text-foreground">
                Adaptive Group Norm:
              </p>
              <BlockMath math="\text{AdaGN}(x, t) = \gamma(t) \cdot \text{Normalize}(x) + \beta(t)" />
              <p className="text-sm text-muted-foreground">
                where{' '}
                <InlineMath math="[\gamma(t), \beta(t)] = \text{Linear}(\text{emb}_t)" />
              </p>
            </div>
            <p className="text-muted-foreground">
              <strong>One conceptual change.</strong> The normalization step
              (computing mean and variance) is identical. The only difference is
              where <InlineMath math="\gamma" /> and{' '}
              <InlineMath math="\beta" /> come from: not from a fixed parameter
              table, but from a linear projection of the timestep embedding.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Insight">
            &ldquo;Oh, you just make <InlineMath math="\gamma" /> and{' '}
            <InlineMath math="\beta" /> depend on t.&rdquo; That is the entire
            idea. The normalization is standard. The adaptation is in the scale
            and shift.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Concrete example with numbers */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Concrete example. A normalized feature map passes through adaptive
              group norm at two different timesteps:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  t = 500 (heavy noise)
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>&bull; <InlineMath math="\gamma(500)" /> = [1.3, 0.7, 1.1, ...]</li>
                  <li>&bull; <InlineMath math="\beta(500)" /> = [&minus;0.2, 0.4, 0.0, ...]</li>
                  <li>&bull; Amplifies some channels, suppresses others</li>
                  <li>&bull; Shifts feature distributions for structural tasks</li>
                </ul>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  t = 50 (light noise)
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>&bull; <InlineMath math="\gamma(50)" /> = [0.8, 1.2, 0.9, ...]</li>
                  <li>&bull; <InlineMath math="\beta(50)" /> = [0.5, &minus;0.1, 0.3, ...]</li>
                  <li>&bull; <em>Different</em> channels amplified and suppressed</li>
                  <li>&bull; Shifts features for fine-detail refinement</li>
                </ul>
              </div>
            </div>
            <p className="text-muted-foreground">
              Same feature map after normalization. Different{' '}
              <InlineMath math="\gamma" /> and <InlineMath math="\beta" />{' '}
              &rarr; different output features. <strong>Same orchestra,
              different dynamics.</strong> The conductor (timestep embedding) tells
              the musicians (features) how loud to play (scale) and what key to
              play in (shift). Different measures (timesteps) produce different
              performances from the same instruments.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Warning: adaptive does NOT mean weights change */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Where does this happen inside the residual block? The timestep
              embedding enters at the normalization layers:
            </p>
            <CodeBlock
              code={`# Residual block WITH adaptive group norm
def forward(self, x, t_emb):
    # First half of the residual block
    h = self.conv1(x)
    h = self.adagn1(h, t_emb)   # <-- timestep modulates here
    h = self.activation(h)

    # Second half
    h = self.conv2(h)
    h = self.adagn2(h, t_emb)   # <-- and here
    h = self.activation(h)

    return h + x                 # residual skip connection`}
              language="python"
              filename="residual_block.py"
            />
            <MermaidDiagram chart={`
graph LR
    X["Input x"] --> CONV1["Conv1"]
    CONV1 --> ADAGN1["AdaGN"]
    ADAGN1 --> ACT1["Activation"]
    ACT1 --> CONV2["Conv2"]
    CONV2 --> ADAGN2["AdaGN"]
    ADAGN2 --> ACT2["Activation"]
    ACT2 --> ADD("+")
    X --> ADD
    ADD --> OUT["Output"]

    TEMB["t_emb"] -.->|"γ₁(t), β₁(t)"| ADAGN1
    TEMB -.->|"γ₂(t), β₂(t)"| ADAGN2

    style ADAGN1 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style ADAGN2 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style TEMB fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style ADD fill:#1e293b,stroke:#a855f7,color:#e2e8f0
`} />
            <p className="text-sm text-muted-foreground italic">
              A single residual block. The timestep embedding (amber) enters at
              both AdaGN nodes, providing timestep-dependent scale and shift. The
              residual skip connection (purple +) is unchanged from standard
              ResNet blocks.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Weights Do NOT Change">
            &ldquo;Adaptive&rdquo; does <strong>not</strong> mean the conv weights
            change based on t. The conv weights are fixed. The normalization
            computation (mean, variance) is standard. What changes are{' '}
            <strong>only</strong> the scale and shift parameters after
            normalization. Same weights, different{' '}
            <InlineMath math="\gamma" /> and <InlineMath math="\beta" />{' '}
            &rarr; different effective behavior.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Per-block projection */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The timestep embedding is <strong>one vector</strong> (512-dim). But
              different residual blocks have different numbers of channels: 64 at
              the highest resolution, 128 at the next level, 256, 512. How does one
              embedding serve all blocks?
            </p>
            <p className="text-muted-foreground">
              Each residual block has its own small linear layer:
            </p>
            <CodeBlock
              code={`# Per-block projection (same pattern as Q/K/V)
# One timestep embedding, different "lens" per block
gamma, beta = Linear(512, 2 * channels)(t_emb).chunk(2)

# At level 0 (64 channels):  Linear(512, 128) -> split into gamma(64), beta(64)
# At level 1 (128 channels): Linear(512, 256) -> split into gamma(128), beta(128)
# At level 2 (256 channels): Linear(512, 512) -> split into gamma(256), beta(256)
# At bottleneck (512 channels): Linear(512, 1024) -> split into gamma(512), beta(512)`}
              language="python"
              filename="per_block_projection.py"
            />
            <p className="text-muted-foreground">
              Same embedding, different lens per block. You know this pattern
              from <strong>Queries and Keys</strong>&mdash;one embedding, multiple
              learned projections. Each block&rsquo;s linear layer learns to extract
              the <em>aspect</em> of the timestep that matters at its resolution
              level.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Learned Lens Pattern">
            In attention, one token embedding is projected through different
            learned matrices to produce Q, K, and V. Here, one timestep embedding
            is projected through different learned matrices to produce{' '}
            <InlineMath math="\gamma" /> and <InlineMath math="\beta" /> for
            each block. Same idea: one source, multiple views.
          </TipBlock>
          <WarningBlock title="One Embedding, Not Many">
            You might expect each resolution level to need its own conditioning
            signal. It does not. All blocks share the <strong>same</strong>{' '}
            timestep embedding. Each block only has its own small projection
            layer to extract the aspect of the timestep relevant at its
            resolution. One source, many views&mdash;not many sources.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Elaborate — Why at Every Layer? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why at Every Layer?"
            subtitle="Global conditioning means injecting everywhere"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Negative example: conditioning at the input only.</strong>{' '}
              Suppose you concatenate t as an extra channel to the input image. The
              conditioning signal passes through 4 downsampling blocks before
              reaching the bottleneck. Each conv layer blends the conditioning
              channel with spatial features. By the bottleneck, the conditioning
              signal is thoroughly mixed and diluted&mdash;a single extra channel
              has negligible influence on 512 feature channels.
            </p>
            <p className="text-muted-foreground">
              The capstone&rsquo;s approach (add at the bottleneck only) is better,
              but still limited. The decoder must recover conditioning information
              from the bottleneck features, and the skip connections carry{' '}
              <strong>no conditioning information at all</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Input-Only Conditioning Fails">
            A single number t concatenated to the input has negligible influence
            on a 64&times;64&times;64 feature map. The network can trivially learn
            to ignore one extra channel. By the time it passes through several
            conv layers, the signal is gone.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Adaptive norm at every block means the timestep modulates the
              network&rsquo;s behavior at <strong>every resolution level</strong>,
              on <strong>both</strong> the encoder and decoder paths:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>High-resolution encoder blocks</strong> get timestep
                awareness: they need to know whether to emphasize fine details or
                ignore them as noise.
              </li>
              <li>
                <strong>The bottleneck</strong> gets timestep awareness: it makes
                global structural decisions that depend heavily on the noise level.
              </li>
              <li>
                <strong>Decoder blocks</strong> get timestep awareness: they decide
                how aggressively to reconstruct details.
              </li>
              <li>
                <strong>Skip connections</strong> carry spatial information that
                is then <em>modulated</em> by the decoder&rsquo;s
                timestep-aware processing.
              </li>
            </ul>
          </div>
        </Row.Content>
      </Row>

      {/* Global conditioning pattern */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This pattern has a name: <strong>global conditioning</strong>.
              The timestep applies equally to every pixel&mdash;whether you are
              denoising the top-left corner or the bottom-right corner, the noise
              level is the same. Global conditioning information should influence
              every spatial location and every processing stage.
            </p>
            <p className="text-muted-foreground">
              Later in this module, you will see a contrasting pattern: text
              conditioning via cross-attention is{' '}
              <strong>spatially-varying conditioning</strong>, where different
              spatial locations attend to different parts of the text. The timestep
              tells the network <em>when</em> it is in the denoising process. The
              text will tell it <em>what</em> to generate.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="FiLM Conditioning">
            This pattern&mdash;predicting scale and shift parameters from a
            conditioning signal&mdash;is called <strong>Feature-wise Linear
            Modulation (FiLM)</strong>. Adaptive group norm is a specific
            instance of the FiLM pattern. You do not need the details, but the
            name is useful for reading papers.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check — Transfer Questions */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> The previous lesson&rsquo;s mental
                model was &ldquo;bottleneck decides WHAT, skip connections decide
                WHERE.&rdquo; With adaptive normalization, how does the timestep
                influence the what/where balance?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    At high t (heavy noise), the adaptive norm at bottleneck blocks
                    could amplify structural features (the WHAT). At low t (fine
                    detail refinement), the adaptive norm at high-resolution blocks
                    could amplify fine spatial features. The timestep does not change
                    which <em>path</em> information flows through&mdash;it changes
                    what each path <em>emphasizes</em> via scale and shift.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Your colleague says: &ldquo;Conditioning
                at every layer is wasteful. Most of the information is the same
                timestep repeated. You only need it at the bottleneck.&rdquo; What is
                wrong with this reasoning?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The timestep is the same, but the <strong>projection is
                    different</strong> at each block. Each block&rsquo;s linear
                    layer learns to extract different aspects of the timestep for
                    its resolution level. The bottleneck block might emphasize
                    &ldquo;how much structure to hallucinate,&rdquo; while a
                    high-resolution encoder block might emphasize &ldquo;how much
                    to trust the fine details.&rdquo; Same source, different
                    learned lenses.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: Explain — Complete Forward Pass */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Forward Pass"
            subtitle="Fulfilling the promise from last lesson"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The pseudocode from <strong>The U-Net Architecture</strong> is now
              complete. The spatial architecture is unchanged&mdash;only the
              conditioning is new:
            </p>
            <CodeBlock
              code={`def forward(self, x, t):
    # --- NEW: Timestep embedding ---
    t_emb = sinusoidal_encoding(t)           # scalar -> 256-dim
    t_emb = self.mlp(t_emb)                  # 256-dim -> 512-dim

    # Encoder: compress, saving features at each level
    e0 = self.enc_block_0(x, t_emb)          # 64x64x64   (AdaGN inside)
    e1 = self.enc_block_1(down(e0), t_emb)   # 32x32x128  (AdaGN inside)
    e2 = self.enc_block_2(down(e1), t_emb)   # 16x16x256  (AdaGN inside)

    # Bottleneck: maximum compression, global context
    b = self.bottleneck(down(e2), t_emb)     # 8x8x512    (AdaGN inside)

    # Decoder: expand, concatenating encoder features
    d2 = self.dec_block_2(cat(up(b), e2), t_emb)   # 16x16x256
    d1 = self.dec_block_1(cat(up(d2), e1), t_emb)  # 32x32x128
    d0 = self.dec_block_0(cat(up(d1), e0), t_emb)  # 64x64x64

    return self.output(d0)  # 64x64x3 (predicted noise)`}
              language="python"
              filename="unet_complete.py"
            />
            <p className="text-muted-foreground">
              Every block now receives <code className="text-xs">t_emb</code>.
              Inside each block, the adaptive group norm uses{' '}
              <code className="text-xs">t_emb</code> to compute{' '}
              <InlineMath math="\gamma(t)" /> and{' '}
              <InlineMath math="\beta(t)" /> via a per-block linear projection.
              The conv weights, the skip connections, the downsampling and
              upsampling&mdash;all unchanged. The <em>only</em> addition is the
              timestep signal flowing into each normalization layer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Skeleton + Nervous System">
            The U-Net from the previous lesson was the skeleton&mdash;the spatial
            architecture. This lesson added the nervous system&mdash;the timestep
            signal that tells each bone how to move. Same skeleton, now with the
            ability to respond to signals.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Architecture diagram with conditioning */}
      <Row>
        <Row.Content>
          <ExercisePanel title="U-Net Architecture with Timestep Conditioning">
            <ArchitectureDiagram data={UNET_CONDITIONING_DATA} />
          </ExercisePanel>
          <p className="text-sm text-muted-foreground italic mt-4">
            Dashed arrows from the timestep embedding reach every block. Each
            block has its own learned projection to convert the embedding into
            block-specific <InlineMath math="\gamma(t)" /> and{' '}
            <InlineMath math="\beta(t)" /> parameters.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Reading the Diagram">
            The top box shows the timestep pipeline: integer &rarr; sinusoidal
            encoding &rarr; MLP &rarr; embedding vector. The dashed arrows
            from the embedding to each block show the conditioning signal
            reaching every level. Compare to the previous lesson&rsquo;s
            diagram&mdash;the spatial architecture is identical.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Notebook exercises */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On Exercises"
            subtitle="Implement the conditioning pipeline in a notebook"
          />
          <div className="space-y-4">
            <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Implement both conditioning mechanisms yourself in a Jupyter
                  notebook.
                </p>
                <a
                  href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-3-2-conditioning-the-unet.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  Open in Google Colab
                </a>
              </div>
            </div>
            <p className="text-muted-foreground">
              The notebook covers four exercises:
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 1: Sinusoidal Timestep Embedding (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Implement the sinusoidal encoding function from the formula.
                  Compute embeddings for t=0, 50, 500, 950, 999. Visualize as a
                  heatmap (like the positional encoding heatmap from{' '}
                  <strong>Embeddings and Position</strong>). Verify that
                  adjacent timesteps produce similar embeddings and distant
                  timesteps produce different ones.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 2: Timestep MLP (Guided)
                </p>
                <p className="text-sm text-muted-foreground">
                  Build the 2-layer MLP that processes the sinusoidal encoding.
                  Visualize a cosine similarity matrix: t=500 vs t=501 should be
                  very similar; t=500 vs t=50 should be quite different.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 3: Adaptive Group Normalization (Supported)
                </p>
                <p className="text-sm text-muted-foreground">
                  Implement AdaGN as a module: takes feature maps and timestep
                  embedding, applies GroupNorm, then scales and shifts with
                  timestep-dependent <InlineMath math="\gamma" /> and{' '}
                  <InlineMath math="\beta" />. Verify that the output changes
                  when the timestep changes, even though the feature map is the
                  same.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-4 space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Exercise 4: Compare Simple vs Sinusoidal (Independent)
                </p>
                <p className="text-sm text-muted-foreground">
                  Replace the capstone&rsquo;s simple timestep embedding with
                  sinusoidal + MLP. Train both on MNIST for 10 epochs and compare
                  loss curves. The sinusoidal version should converge faster.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What to Focus On">
            Exercises 1&ndash;3 are the core. Exercise 4 is a bonus that
            connects back to the capstone. If time is short, skip Exercise 4.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Sinusoidal timestep embedding is the same formula as positional encoding.',
                description:
                  'Replace "position" with "timestep" and you have the diffusion timestep embedding. Multi-frequency encoding provides a rich, smooth representation that the MLP refines into the final embedding.',
              },
              {
                headline:
                  'Adaptive group normalization makes gamma and beta depend on the timestep.',
                description:
                  'The normalization is standard. The scale and shift are timestep-dependent, computed by a per-block linear projection. Same architecture, same weights\u2014different behavior at different noise levels.',
              },
              {
                headline:
                  'Conditioning happens at every residual block, not just the input.',
                description:
                  'Global conditioning requires injection at every processing stage. Each block\u2019s linear projection extracts different aspects of the timestep for its resolution level.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>
              The U-Net is the orchestra. The timestep embedding is the
              conductor&rsquo;s score. The adaptive norm is how the conductor
              communicates dynamics to each section.
            </strong>{' '}
            Same musicians, same instruments&mdash;different performance depending
            on the measure number.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM paper. Appendix B describes the U-Net architecture including sinusoidal timestep embeddings.',
              },
              {
                title: 'Diffusion Models Beat GANs on Image Synthesis',
                authors: 'Dhariwal & Nichol, 2021',
                url: 'https://arxiv.org/abs/2105.05233',
                note: 'Introduces the improved U-Net with adaptive group normalization. Section 4 details the conditioning mechanism.',
              },
              {
                title: 'Attention Is All You Need',
                authors: 'Vaswani et al., 2017',
                url: 'https://arxiv.org/abs/1706.03762',
                note: 'The original sinusoidal positional encoding (Section 3.5). Same formula used for timestep embeddings.',
              },
              {
                title: 'FiLM: Visual Reasoning with a General Conditioning Layer',
                authors: 'Perez et al., 2018',
                url: 'https://arxiv.org/abs/1709.07871',
                note: 'Introduces Feature-wise Linear Modulation. Adaptive group norm is an instance of this general pattern.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The U-Net now knows the noise level, completing the unconditional
              DDPM architecture. But Stable Diffusion generates images from{' '}
              <strong>text descriptions</strong>. For that, the network needs a
              second conditioning signal: text. The next lesson introduces{' '}
              <strong>CLIP</strong>&mdash;a model that creates a shared embedding
              space for text and images.
            </p>
            <p className="text-muted-foreground">
              The timestep tells the network <em>when</em> it is in the denoising
              process. The text will tell it <em>what</em> to generate.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: CLIP"
            description="How a shared text-image embedding space gives the U-Net the ability to understand what to generate&mdash;not just when."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
