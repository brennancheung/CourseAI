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
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-1-1-controlnet.ipynb'

/**
 * ControlNet
 *
 * Lesson 1 in Module 7.1 (Controllable Generation). Lesson 1 overall in Series 7.
 * Cognitive load: STRETCH (2 new concepts).
 *
 * Teaches how ControlNet adds spatial conditioning to a frozen Stable Diffusion
 * model by cloning the encoder, training only the clone on spatial maps, and
 * connecting it to the frozen decoder via zero convolutions that guarantee the
 * original model starts unchanged.
 *
 * The two genuinely new concepts are:
 * 1. Trainable encoder copy architecture
 * 2. Zero convolution (1x1 conv initialized to all zeros)
 *
 * Everything else is assembly of existing knowledge: U-Net encoder-decoder,
 * skip connections, frozen-model pattern, residual/additive connections.
 *
 * Previous: Textual Inversion (Module 6.5, Lesson 3 / STRETCH)
 * Next: ControlNet in Practice (Module 7.1, Lesson 2)
 */

export function ControlNetLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="ControlNet"
            description="You understand the machine. Now see what the field built on top of it—a trainable encoder copy that adds spatial conditioning to a frozen model, connected by zero convolutions that guarantee nothing breaks."
            category="Controllable Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how ControlNet adds spatial control (edges, depth, pose)
            to a frozen Stable Diffusion model by cloning the U-Net encoder,
            training only the clone on spatial maps, and connecting it to the
            frozen decoder via zero convolutions that guarantee the original
            model starts completely unchanged.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Capstone Series">
            This is Series 7: Post-SD Advances. You built Stable Diffusion from
            scratch. Now we explore what the field built on top of it. The
            tone shifts from &ldquo;learn the foundations&rdquo; to
            &ldquo;let&rsquo;s read the frontier together.&rdquo;
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The ControlNet architecture: trainable encoder copy + zero convolution + additive connection to frozen decoder',
              'Why this architecture preserves the original model (zero-initialization safety)',
              'How spatial maps (edges, depth, pose) serve as the conditioning input',
              'Where ControlNet fits in the SD pipeline and how it coexists with text conditioning',
              'NOT: how to preprocess images into spatial maps (Canny, depth estimation, OpenPose)—that is next lesson',
              'NOT: conditioning scale parameter or control-creativity tradeoff—next lesson',
              'NOT: IP-Adapter or image-based conditioning—lesson 3',
              'NOT: training a ControlNet from scratch',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Quick Recap — Cross-attention reactivation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Reactivating cross-attention before we add a new conditioning dimension"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              From <strong>Text Conditioning &amp; Guidance</strong>: cross-attention
              uses <InlineMath math="Q" /> from spatial features and{' '}
              <InlineMath math="K" />, <InlineMath math="V" /> from text
              embeddings. Each spatial location independently attends to all text
              tokens, producing spatially-varying text conditioning. Alongside
              that, timestep conditioning operates via adaptive group
              normalization—a global signal that tells every layer{' '}
              <em>when</em> in the denoising process it is.
            </p>
            <p className="text-muted-foreground">
              Two conditioning dimensions so far: <strong>timestep</strong> tells
              the network WHEN (adaptive norm, global).{' '}
              <strong>Text</strong> tells it WHAT (cross-attention, spatially
              varying). But what about <strong>WHERE</strong>? What if you want
              to specify spatial structure directly—exact edges, precise depth,
              specific pose?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Conditioning Signals">
            Timestep: adaptive group norm. Global, same everywhere.
            Text: cross-attention. Spatially varying, per-location.
            Both are unchanged by ControlNet—it adds a third signal
            without modifying these two.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — The Spatial Control Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Spatial Control Problem"
            subtitle="Text is the wrong tool for spatial precision"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Try this prompt: &ldquo;a house on a hill, watercolor
              painting.&rdquo; You get a house on a hill. But you cannot
              control <em>where</em> the house sits in the frame, what angle
              the hill has, where the horizon falls, or which edges the
              composition follows. Text describes <em>what</em> to
              generate—not the spatial structure.
            </p>
            <p className="text-muted-foreground">
              What you <em>want</em> is to hand the model a spatial map—an edge
              image, a depth map, a skeleton—and say &ldquo;follow this
              structure, but use the text prompt for everything else.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Gap">
            Text can describe &ldquo;a person with their right arm
            raised.&rdquo; It cannot describe which pixel the elbow occupies,
            what angle the forearm makes, or how the body contour curves.
            Spatial precision needs a spatial input.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Design challenge — think before reveal */}
      <Row>
        <Row.Content>
          <GradientCard title="Design Challenge: Think Before You Read" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                You have all the pieces to design this yourself. Consider:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>The U-Net encoder extracts spatial features at multiple resolutions.</li>
                <li>You know how to freeze models and train new parameters alongside them (LoRA, textual inversion).</li>
                <li>You know that additive connections (residual/bypass) can be initialized safely (LoRA&rsquo;s B=0).</li>
              </ul>
              <p className="font-medium">
                How would you add spatial control without breaking anything?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  See the constraint that guides the design
                </summary>
                <p className="mt-2 text-muted-foreground">
                  You cannot modify the frozen SD model&rsquo;s weights. Any new
                  conditioning must be <strong>additive</strong> and must start
                  with <strong>zero contribution</strong> so the model is
                  unchanged at initialization. This rules out retraining the
                  encoder or adding new inputs to existing layers. You need a
                  parallel pathway that starts silent and gradually learns to
                  speak.
                </p>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 5: Explain — ControlNet Architecture */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ControlNet Architecture"
            subtitle="The &ldquo;of course&rdquo; chain"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The architecture follows from the constraints, one step at a time:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                Spatial features live in the encoder. You know this from{' '}
                <strong>The U-Net Architecture</strong>—the encoder extracts
                multi-resolution features that carry spatial detail.
              </li>
              <li>
                Training new spatial features requires an encoder that takes
                spatial maps as input—not noisy latents.
              </li>
              <li>
                You cannot retrain the existing encoder. It is frozen and it
                works.
              </li>
              <li>
                So: <strong>copy the encoder</strong>. Initialize the copy
                with the original&rsquo;s weights (a good starting point for
                extracting spatial features). Train the copy on spatial maps.
              </li>
              <li>
                <strong>Add</strong> the copy&rsquo;s outputs to the original
                encoder&rsquo;s outputs at each resolution level—enriching
                the skip connections.
              </li>
              <li>
                Initialize the connections at <strong>zero</strong> so the
                model starts completely unchanged.
              </li>
            </ol>
            <p className="text-muted-foreground">
              Every step follows from the previous one. This is not a clever
              trick—it is the obvious solution given the constraints. You could
              have designed this yourself.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="&ldquo;Of Course&rdquo;">
            The encoder extracts spatial features. You want new spatial
            features. You cannot change the existing encoder. So copy it.
            Initialize the connection at zero so you start safely. Train the
            copy. Each step is the only reasonable choice given the previous
            one.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Architecture diagram */}
      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph TB
              subgraph FROZEN["Frozen SD Model"]
                direction TB
                E1["Encoder Block 1<br/>64×64 ❄️"]:::frozen
                E2["Encoder Block 2<br/>32×32 ❄️"]:::frozen
                E3["Encoder Block 3<br/>16×16 ❄️"]:::frozen
                E4["Encoder Block 4<br/>8×8 ❄️"]:::frozen
                BN["Bottleneck ❄️"]:::frozen
                D4["Decoder Block 4<br/>8×8 ❄️"]:::frozen
                D3["Decoder Block 3<br/>16×16 ❄️"]:::frozen
                D2["Decoder Block 2<br/>32×32 ❄️"]:::frozen
                D1["Decoder Block 1<br/>64×64 ❄️"]:::frozen

                ADD3("+"):::merge
                ADD2("+"):::merge
                ADD1("+"):::merge

                E1 --> E2 --> E3 --> E4 --> BN --> D4

                E3 -. "skip" .-> ADD3 --> D3
                E2 -. "skip" .-> ADD2 --> D2
                E1 -. "skip" .-> ADD1 --> D1
                D4 --> D3 --> D2 --> D1
              end

              subgraph CONTROLNET["Trainable ControlNet Copy"]
                direction TB
                C1["Copy Block 1<br/>64×64 🔥"]:::trainable
                C2["Copy Block 2<br/>32×32 🔥"]:::trainable
                C3["Copy Block 3<br/>16×16 🔥"]:::trainable
                C4["Copy Block 4<br/>8×8 🔥"]:::trainable

                C1 --> C2 --> C3 --> C4
              end

              SM["Spatial Map<br/>(edges, depth, pose)"]:::input --> C1

              C1 --> ZC1["zero conv ⚡"]:::zeroconv --> ADD1
              C2 --> ZC2["zero conv ⚡"]:::zeroconv --> ADD2
              C3 --> ZC3["zero conv ⚡"]:::zeroconv --> ADD3
              C4 --> ZC4["zero conv ⚡"]:::zeroconv --> D4

              NL["Noisy Latent z_t"]:::input --> E1

              classDef frozen fill:#374151,stroke:#6b7280,color:#d1d5db
              classDef trainable fill:#5b21b6,stroke:#8b5cf6,color:#f5f3ff
              classDef zeroconv fill:#92400e,stroke:#f59e0b,color:#fef3c7
              classDef input fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
              classDef merge fill:#065f46,stroke:#10b981,color:#d1fae5
          `} />
          <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded bg-gray-600" /> Frozen (unchanged)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded bg-violet-800" /> Trainable (encoder copy)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded bg-amber-800" /> Zero convolution (connection)
            </span>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="What Is the Copy?">
            The trainable copy has the <strong>same architecture</strong> as the
            frozen encoder—same conv blocks, same downsampling, same number of
            layers. It is initialized with the frozen encoder&rsquo;s weights
            so it already knows how to extract spatial features. Training adapts
            it from &ldquo;features from noisy latents&rdquo; to &ldquo;features
            from spatial maps.&rdquo;
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Pseudocode forward pass */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Forward Pass"
            subtitle="One addition per resolution level"
          />
          <CodeBlock
            code={`# Frozen encoder (unchanged, same forward pass you know)
e1 = frozen_encoder_block_1(z_t)          # 64×64
e2 = frozen_encoder_block_2(e1)           # 32×32
e3 = frozen_encoder_block_3(e2)           # 16×16
e4 = frozen_encoder_block_4(e3)           # 8×8

# Trainable copy (processes spatial map instead of noisy latent)
c1 = copy_encoder_block_1(spatial_map)    # 64×64
c2 = copy_encoder_block_2(c1)            # 32×32
c3 = copy_encoder_block_3(c2)            # 16×16
c4 = copy_encoder_block_4(c3)            # 8×8

# Zero convolution connections (initialized to all zeros)
z1 = zero_conv_1(c1)   # starts at 0.0
z2 = zero_conv_2(c2)   # starts at 0.0
z3 = zero_conv_3(c3)   # starts at 0.0
z4 = zero_conv_4(c4)   # starts at 0.0

# Decoder receives enriched skip connections
d4 = decoder_block_4(bottleneck + z4)     # z4 added at bottleneck
d3 = decoder_block_3(cat(d4, e3 + z3))   # original + control
d2 = decoder_block_2(cat(d3, e2 + z2))
d1 = decoder_block_1(cat(d2, e1 + z1))`}
            language="python"
            filename="controlnet_forward.py"
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Compare this to the U-Net forward pass from{' '}
              <strong>The U-Net Architecture</strong>. The only change is{' '}
              <InlineMath math="e_i + z_i" /> instead of{' '}
              <InlineMath math="e_i" /> at the skip connections. One addition
              per resolution level. Everything else is identical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Connection to Prior Knowledge">
            In <strong>The U-Net Architecture</strong>, the decoder received{' '}
            <code className="text-xs">cat(up(b), e2)</code> at each skip
            connection. ControlNet changes this to{' '}
            <code className="text-xs">cat(up(b), e2 + zero_conv(c2))</code>.
            That is the entire modification.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Parameter breakdown — misconception: doubles the model */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Frozen SD Model',
              color: 'blue',
              items: [
                'Full U-Net: ~860M parameters',
                'Encoder + decoder + bottleneck',
                'Cross-attention, self-attention, conv blocks',
                'All parameters frozen during ControlNet training',
                'Unchanged—bit-for-bit identical with or without ControlNet',
              ],
            }}
            right={{
              title: 'ControlNet Addition',
              color: 'violet',
              items: [
                'Encoder copy: ~300M parameters (~35% of U-Net)',
                'Encoder half only—no decoder copy',
                'Same architecture as frozen encoder',
                'All parameters trainable',
                'Plus zero convolution layers (negligible extra params)',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Full Duplicate">
            The trainable copy is the encoder half only. The frozen decoder
            still runs once—it receives enriched skip connections from both the
            frozen encoder and the ControlNet copy. Total added parameters are
            ~35% of the original U-Net, not 100%.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check — Predict-and-Verify */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict, then verify"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1: Initial Contribution" color="cyan">
              <p className="text-sm">
                Before any training, what is the ControlNet&rsquo;s contribution
                to the decoder?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Exactly zero. The zero convolutions produce all zeros.
                  The decoder receives{' '}
                  <InlineMath math="e_i + 0 = e_i" />—the frozen
                  encoder&rsquo;s features unchanged. The model&rsquo;s output
                  is identical to vanilla SD.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Question 2: Disconnection Test" color="cyan">
              <p className="text-sm">
                If you disconnect the ControlNet entirely—remove the{' '}
                <InlineMath math="z_i" /> additions—does the output change?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. The frozen model was never modified. Its weights are
                  bit-for-bit identical. Disconnecting ControlNet restores the
                  original skip connections. This is the safety guarantee.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Question 3: Weight Initialization" color="cyan">
              <p className="text-sm">
                Why initialize the trainable copy from the original
                encoder&rsquo;s weights instead of random weights?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The original encoder already knows how to extract spatial
                  features from images. Training only needs to adapt it from
                  &ldquo;extract features from noisy latents&rdquo; to
                  &ldquo;extract features from spatial maps.&rdquo; Starting
                  from random weights would require learning spatial feature
                  extraction from scratch—much harder and slower.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — Zero Convolution */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Zero Convolution"
            subtitle="The simplest possible connection mechanism"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The name sounds technical. The reality is almost anticlimactic: a
              zero convolution is a <strong>1×1 convolution with weights
              initialized to 0.0 and bias initialized to 0.0</strong>. That is
              the complete definition. The &ldquo;cleverness&rdquo; is not in
              the operation—it is in the <em>initialization</em>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Just a Name">
            &ldquo;Zero convolution&rdquo; describes the initialization, not
            the operation. It is a standard 1×1 conv layer. After training,
            the weights are no longer zero—they have learned to scale and mix
            the ControlNet&rsquo;s features.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Concrete numerical example */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground font-medium">
              Concrete example at initialization:
            </p>
            <CodeBlock
              code={`# Feature map from trainable copy (any values)
feature = [[0.5, -0.3],
           [1.2,  0.8]]

# Zero convolution: weight = 0.0, bias = 0.0
zero_conv_weight = 0.0
zero_conv_bias   = 0.0

# Output: all zeros regardless of input
output = [[0.0, 0.0],
          [0.0, 0.0]]

# Added to frozen encoder feature: unchanged
frozen_feature + output = frozen_feature  # e_i + 0 = e_i`}
              language="python"
              filename="zero_conv_init.py"
            />
            <p className="text-muted-foreground font-medium mt-4">
              After 100 training steps:
            </p>
            <CodeBlock
              code={`# Same feature map
feature = [[0.5, -0.3],
           [1.2,  0.8]]

# Zero conv weight has drifted from 0.0 to 0.03
# Zero conv bias has drifted to 0.001
output = [[0.016, -0.008],
          [0.037,  0.025]]

# A tiny signal: the control is fading in gradually
frozen_feature + output ≈ frozen_feature + small_nudge`}
              language="python"
              filename="zero_conv_after_training.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The control signal fades in gradually as training progresses. The
              model does not suddenly receive a large, untrained control
              signal—it receives nothing at first, then a whisper, then a clear
              voice. Training is stable because the ControlNet&rsquo;s
              influence grows smoothly from zero.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Nothing, Then a Whisper">
            At initialization, the zero convolution output is exactly zero—the
            frozen model hears nothing. As training progresses, the signal
            grows: first a whisper, then a clear voice. The volume knob starts
            at zero and training gradually turns it up.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: ControlNet does NOT modify frozen weights */}
      <Row>
        <Row.Content>
          <WarningBlock title="ControlNet Does NOT Modify Frozen Weights">
            Not a single parameter in the frozen U-Net changes. The zero
            convolution output is added to the encoder&rsquo;s{' '}
            <strong>features</strong>, not to its <strong>weights</strong>.
            This is different from LoRA, where the bypass output merges with
            the weight&rsquo;s computation (<InlineMath math="Wx + BAx" />).
            ControlNet&rsquo;s frozen model is bit-for-bit identical with or
            without ControlNet connected—it never knows ControlNet exists.
            Disconnect it and the original model produces exactly its original
            output.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* LoRA B=0 connection */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'LoRA B=0 Initialization',
              color: 'violet',
              items: [
                'Matrix B initialized to zero',
                'Bypass output: BA·x = 0·A·x = 0',
                'Frozen weight W unchanged at start',
                'Operates at the weight level',
                '~1M trainable parameters',
              ],
            }}
            right={{
              title: 'ControlNet Zero Convolution',
              color: 'amber',
              items: [
                '1×1 conv initialized to zero (weight + bias)',
                'Zero conv output: 0·c_i + 0 = 0',
                'Frozen encoder feature unchanged at start',
                'Operates at the feature level',
                '~300M trainable parameters',
              ],
            }}
          />
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Same principle—ensure the frozen model starts unchanged—applied
              at different scales. LoRA&rsquo;s B=0 initializes a bypass at
              each projection matrix. ControlNet&rsquo;s zero convolution
              initializes a bypass at each resolution level of the encoder.
              The &ldquo;highway and detour&rdquo; analogy extends directly:
              ControlNet is a detour that starts as a dead end and gradually
              connects.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="ControlNet Is Not LoRA">
            LoRA adds small bypass matrices (~1M params) at individual weight
            projections. ControlNet trains a full copy of the encoder (~300M
            params) connected at the feature level. Same safety principle,
            vastly different scale and mechanism. Do not conflate them.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Elaborate — Coexistence with Text Conditioning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Coexistence with Text Conditioning"
            subtitle="ControlNet adds WHERE without replacing WHAT"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A common assumption: ControlNet replaces text conditioning—you
              use either text <em>or</em> spatial maps. This is wrong. Both
              are active simultaneously, controlling different dimensions.
            </p>
            <p className="text-muted-foreground">
              Concrete example: take the same Canny edge map of a room with
              furniture. Prompt it with &ldquo;a cat in a living room, oil
              painting&rdquo; and then with &ldquo;a robot in a factory,
              cyberpunk style.&rdquo; Same spatial structure—same edges, same
              composition—but completely different content and style. The edge
              map controls WHERE things go. The text prompt controls WHAT those
              things are and what they look like.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Conditioning Dimensions">
            Timestep says WHEN (adaptive norm, global). Text says WHAT
            (cross-attention, spatially varying). ControlNet says WHERE
            (additive to encoder features, spatially precise). Three signals,
            three mechanisms, all coexisting.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Where each conditioning type operates */}
      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="Timestep → WHEN" color="blue">
              <ul className="space-y-1 text-sm">
                <li>Adaptive group normalization</li>
                <li>Global signal (same everywhere)</li>
                <li>Tells each layer the noise level</li>
                <li>Unchanged by ControlNet</li>
              </ul>
            </GradientCard>
            <GradientCard title="Text → WHAT" color="violet">
              <ul className="space-y-1 text-sm">
                <li>Cross-attention (Q spatial, K/V text)</li>
                <li>Spatially varying</li>
                <li>Controls content, style, semantics</li>
                <li>Unchanged by ControlNet</li>
              </ul>
            </GradientCard>
            <GradientCard title="ControlNet → WHERE" color="amber">
              <ul className="space-y-1 text-sm">
                <li>Additive to encoder skip connections</li>
                <li>Multi-resolution spatial features</li>
                <li>Controls edges, depth, pose, structure</li>
                <li>New—does not modify the other two</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Generalization: depth map as second spatial map type */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The architecture is spatial-map agnostic. A Canny edge map, a
              depth map, an OpenPose skeleton—they are all just spatial inputs
              to the trainable encoder copy. The ControlNet architecture is
              identical for each; only the preprocessing and training data
              differ. You train one ControlNet per spatial map type, but the
              architecture never changes.
            </p>
          </div>
          <div className="mt-4">
            <ComparisonRow
              left={{
                title: 'Canny Edge ControlNet',
                color: 'emerald',
                items: [
                  'Input: binary edge map',
                  'Controls: contour shapes, object boundaries',
                  'Architecture: identical trainable encoder copy',
                  'Training data: image + Canny edge pairs',
                ],
              }}
              right={{
                title: 'Depth Map ControlNet',
                color: 'orange',
                items: [
                  'Input: per-pixel depth values',
                  'Controls: 3D layout, spatial arrangement',
                  'Architecture: identical trainable encoder copy',
                  'Training data: image + depth estimation pairs',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Modularity Again">
            SD is three independently trained models connected by tensor
            handoffs. ControlNet adds a fourth: you can swap ControlNet
            checkpoints (edge, depth, pose) without retraining anything
            else. Same modularity principle, another application.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Brief note on compatibility */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Compatibility note:</strong> ControlNet checkpoints
              trained on SD v1.5 work with any SD v1.5-compatible model. The
              encoder copy learns to produce control signals in the same
              feature space as the original encoder, so it transfers across
              fine-tuned variants of the same base. A ControlNet trained for
              SD v1.5 does <em>not</em> work with SDXL—different architecture,
              different feature space.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Check — Transfer Questions */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Check"
            subtitle="Probing the boundaries"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1: Style vs Structure" color="cyan">
              <p className="text-sm">
                A friend says they want to use ControlNet to change the{' '}
                <em>style</em> of an image. Is ControlNet the right tool?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. ControlNet provides spatial/structural control—edges,
                  depth, pose. For style, use LoRA (modifies the text-to-image
                  mapping at cross-attention projections) or prompt engineering.
                  ControlNet controls structure, not style. Different tools for
                  different conditioning dimensions.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Question 2: Why This Architecture?" color="cyan">
              <p className="text-sm">
                Why does the trainable copy process the spatial map through
                the <em>same architecture</em> as the encoder (conv blocks,
                downsampling) rather than a simple MLP?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The spatial map needs to produce features at every resolution
                  level (64×64, 32×32, 16×16, 8×8) to match the
                  decoder&rsquo;s expectations at each skip connection. An MLP
                  would produce a single feature vector—not multi-resolution
                  spatial feature maps. The encoder architecture is designed for
                  exactly this kind of multi-resolution spatial feature
                  extraction.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Practice — Notebook link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Exploring the ControlNet Architecture"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook grounds the architectural understanding in real
                models—inspect parameters, verify the zero-initialization
                property, trace the forward pass with real tensors, and compare
                ControlNet to vanilla SD generation.
              </p>
              <a
                href={NOTEBOOK_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>
                  <strong>Exercise 1 (Guided):</strong> Inspect the ControlNet
                  architecture. Load a pre-trained ControlNet (Canny) and the
                  frozen SD model via diffusers. Print parameter counts, identify
                  trainable vs frozen parameters, verify the encoder copy matches
                  the U-Net encoder structure.
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> Verify the
                  zero-initialization property. Create a fresh zero convolution
                  layer, pass a random feature map through it, verify the output
                  is all zeros, confirm adding it to a frozen feature leaves it
                  unchanged.
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Trace the forward
                  pass. Load a pre-trained ControlNet and feed a Canny edge map.
                  Inspect feature map shapes at each resolution level from both
                  the frozen encoder and the ControlNet copy. Verify additive
                  connections produce matching shapes.
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> ControlNet vs
                  vanilla SD comparison. Generate with and without ControlNet
                  using the same seed and prompt. Vary the text prompt while
                  keeping the edge map fixed to verify structure is preserved
                  while content changes.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Inspect the architecture (which parameters, how many)</li>
              <li>Verify zero initialization (the safety guarantee)</li>
              <li>Trace the forward pass (feature shapes and additive connections)</li>
              <li>Compare outputs (see the spatial control in action)</li>
            </ol>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'ControlNet adds spatial conditioning via a trainable encoder copy.',
                description:
                  'Clone the frozen encoder, initialize it with the original weights, train it on spatial maps. The copy produces multi-resolution features that are added to the frozen encoder\'s skip connections. The decoder runs once, receiving enriched features from both sources.',
              },
              {
                headline:
                  'Zero convolutions guarantee the original model starts unchanged.',
                description:
                  'A 1×1 conv initialized to all-zero weights and bias. Output is exactly zero before training, so the frozen model\'s behavior is preserved. The control signal fades in gradually as training progresses. Same principle as LoRA\'s B=0 initialization, applied at the feature level.',
              },
              {
                headline:
                  'ControlNet adds a new conditioning dimension, not a new mechanism.',
                description:
                  'Timestep says WHEN. Text says WHAT. ControlNet says WHERE. Three conditioning signals, three mechanisms, all coexisting. The architecture is map-agnostic—same design for edges, depth, pose. Swap ControlNet checkpoints without retraining.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            The trainable copy extracts spatial features from your control
            image. Zero convolutions ensure it starts silent. Training
            gradually turns up the volume. The frozen model never changes.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Adding Conditional Control to Text-to-Image Diffusion Models',
                authors: 'Zhang, Rao & Agrawala, 2023',
                url: 'https://arxiv.org/abs/2302.05543',
                note: 'The original ControlNet paper. Sections 3.1–3.2 cover the architecture and zero convolution. Section 4 shows results across spatial map types.',
              },
              {
                title: 'T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models',
                authors: 'Mou et al., 2023',
                url: 'https://arxiv.org/abs/2302.08453',
                note: 'Alternative lightweight approach to spatial control. Uses smaller adapter modules instead of a full encoder copy. Good comparison point.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: ControlNet in Practice"
            description="The architecture you learned today is the engine. Next lesson you drive it—real preprocessors (Canny, depth, OpenPose), the conditioning scale parameter, and stacking multiple ControlNets for combined control."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
