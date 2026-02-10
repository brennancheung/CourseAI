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
  TryThisBlock,
} from '@/components/lessons'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { CodeBlock } from '@/components/common/CodeBlock'

/**
 * U-Net Architecture
 *
 * Lesson 1 in Module 6.3 (Architecture & Conditioning). Lesson 10 overall in Series 6.
 * Cognitive load: BUILD (re-entering theoretical mode after CONSOLIDATE capstone).
 *
 * Teaches the U-Net's encoder-decoder architecture with skip connections:
 * - Why a stack of same-resolution conv layers fails for denoising
 * - The encoder (downsampling) path as a CNN feature hierarchy
 * - The bottleneck as global context
 * - The decoder (upsampling) path with skip connections
 * - Why skip connections are essential, not optional
 * - Multi-resolution processing mapped to coarse-to-fine denoising
 * - Residual blocks within the U-Net (INTRODUCED)
 *
 * Core concepts:
 * - U-Net skip connections as essential for denoising: DEVELOPED
 * - Multi-resolution processing mapped to coarse-to-fine denoising: DEVELOPED
 *
 * Previous: Build a Diffusion Model (module 6.2, lesson 5 / capstone)
 * Next: Timestep Conditioning (module 6.3, lesson 2)
 */

export function UnetArchitectureLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The U-Net Architecture"
            description="Why the denoising network is shaped like a hourglass with side doors&mdash;and why every piece of that shape is essential."
            category="Architecture & Conditioning"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain why the U-Net&rsquo;s encoder-decoder architecture with skip
            connections is the natural choice for multi-scale denoising&mdash;understand
            how each design choice maps to the denoising task.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <strong>Autoencoders</strong>, you built an encoder-decoder that
            compressed images through a bottleneck. The reconstructions were
            blurry&mdash;fine details were lost through the bottleneck. Hold that
            thought. Then in <strong>Build a Diffusion Model</strong>, you used a
            minimal U-Net with 2 skip connections and a simple timestep embedding.
            It generated recognizable MNIST digits. Now: why was it shaped like
            that?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              `The U-Net’s spatial architecture: encoder, bottleneck, decoder, skip connections`,
              'WHY each design choice matters for denoising specifically',
              'Tensor dimensions at each resolution level',
              'Residual blocks within the U-Net (introduced, not developed)',
              'NOT: timestep conditioning—how the network receives t is the next lesson',
              'NOT: attention layers (self-attention, cross-attention)—mentioned only, developed later',
              'NOT: text conditioning or guided generation',
              'NOT: implementation—the capstone already covered that',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Hook — Why Not Just Conv Layers? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Not Just Conv Layers?"
            subtitle="The simplest denoising architecture fails at heavy noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>Build a Diffusion Model</strong>, the neural network was a
              black box. You fed it a noisy image and a timestep, and it predicted
              noise. It worked. But why was the network shaped the way it was?
            </p>
            <p className="text-muted-foreground">
              Start with the simplest possible approach: a stack of convolutional
              layers, all operating at the same resolution. No downsampling, no
              upsampling. Just conv after conv after conv. Each layer has a 3&times;3
              receptive field, so after 10 layers, each output pixel &ldquo;sees&rdquo;
              a roughly 21&times;21 neighborhood of the input.
            </p>
            <p className="text-muted-foreground">
              For light noise (<strong>t&nbsp;=&nbsp;50</strong>), this works
              passably. The image is mostly intact. The network only needs to clean
              up small local perturbations&mdash;a task that local neighborhoods
              can handle.
            </p>
            <p className="text-muted-foreground">
              For heavy noise (<strong>t&nbsp;=&nbsp;900</strong>), it fails
              catastrophically. The image is almost pure static. The network needs
              to decide: <em>is this a shoe or a shirt?</em> That is a
              global decision. A 21&times;21 pixel window cannot reason about the
              overall composition of a 64&times;64 image. It has no way to see
              the big picture.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Problem">
            At heavy noise, the network must make <strong>global structural
            decisions</strong> (what is this image of?) and <strong>local detail
            decisions</strong> (where exactly is this edge?). A stack of same-resolution
            conv layers can only do the local part.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You need two things simultaneously: a way to see the{' '}
              <strong>big picture</strong> (global context for structural decisions)
              and a way to preserve <strong>fine details</strong> (local precision
              for pixel-accurate output). The U-Net gives you both.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 4: The Encoder Path */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Encoder Path"
            subtitle="Zooming out to see the big picture"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The encoder is a familiar CNN: convolutional layers followed by
              downsampling. Each level doubles the number of channels and halves
              the spatial resolution. You have seen this pattern before&mdash;it is
              the same feature hierarchy from the convolution lessons.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Building Blocks">
            The encoder path IS a CNN feature hierarchy. Early layers capture
            edges and textures. Deeper layers capture shapes and structure. The
            bottleneck captures global composition. You already know this.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a concrete dimension walkthrough. These specific numbers are
              illustrative&mdash;real diffusion U-Nets vary in exact channel
              counts&mdash;but the pattern is universal:
            </p>
            <div className="space-y-2">
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Level 0&mdash;64&times;64&times;64
                </p>
                <p className="text-sm text-muted-foreground">
                  Input image (3 channels) projected to 64 feature channels. Full
                  resolution. Each pixel sees a small local neighborhood. Captures
                  edges, textures, fine detail.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Level 1&mdash;32&times;32&times;128
                </p>
                <p className="text-sm text-muted-foreground">
                  Half resolution, double channels. Each &ldquo;pixel&rdquo; in this
                  feature map summarizes a 2&times;2 region of level 0. Captures
                  larger patterns&mdash;shapes, parts of objects.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Level 2&mdash;16&times;16&times;256
                </p>
                <p className="text-sm text-muted-foreground">
                  Quarter resolution. Each position covers a 4&times;4 region of the
                  original. Captures object-level structure&mdash;the overall shape
                  of a shoe, the outline of a shirt.
                </p>
              </div>
              <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Bottleneck&mdash;8&times;8&times;512
                </p>
                <p className="text-sm text-muted-foreground">
                  Eighth resolution, maximum channels. Each position has a{' '}
                  <strong>global receptive field</strong>&mdash;it can &ldquo;see&rdquo;
                  the entire input image. This is where the network decides
                  &ldquo;this is a shoe&rdquo; rather than &ldquo;this is a shirt.&rdquo;
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Receptive Field Growth">
            As spatial resolution shrinks, each pixel in the feature map
            represents a larger region of the original image. At 8&times;8, each
            &ldquo;pixel&rdquo; covers the entire 64&times;64 input. This is how
            the network gains global context.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: The Bottleneck */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bottleneck"
            subtitle="Maximum compression, global context"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The bottleneck is the lowest resolution: maximum channels, minimum
              spatial size. It captures the global context of the image.
            </p>
            <p className="text-muted-foreground">
              At <strong>t&nbsp;=&nbsp;900</strong>, the image is almost pure
              static. But the bottleneck features capture whatever faint global
              structure remains&mdash;a vague sense of &ldquo;something bright in
              the center, dark around the edges.&rdquo; This is where the important
              structural decisions happen.
            </p>
            <p className="text-muted-foreground">
              This is the same principle as the autoencoder bottleneck from{' '}
              <strong>Autoencoders</strong>&mdash;information compression. But
              unlike the autoencoder, the U-Net does <em>not</em> rely solely on
              the bottleneck to reconstruct the output. That difference is
              everything.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Connection to Autoencoders">
            Remember &ldquo;describe a shoe in 32 words&rdquo;? The bottleneck
            captures the gist. But in the autoencoder, the gist was all you
            had for reconstruction. In the U-Net, you also get the details
            through a different path.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: The Decoder + Skip Connections */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Decoder and Skip Connections"
            subtitle="Reconstructing with both global context and local detail"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The decoder mirrors the encoder: each level upsamples the spatial
              resolution (via ConvTranspose2d or nearest-neighbor upsampling plus
              convolution) while halving the channel count. 8&times;8 becomes
              16&times;16 becomes 32&times;32 becomes 64&times;64.
            </p>
            <p className="text-muted-foreground">
              <strong>Without skip connections</strong>, the decoder is just an
              autoencoder decoder. It must reconstruct all spatial detail from the
              8&times;8&times;512 bottleneck alone. Remember the blurry
              reconstructions from <strong>Autoencoders</strong>? The fine
              details&mdash;exact edge positions, textures, the precise boundary
              between a shoe and its background&mdash;are lost through the
              bottleneck. The decoder has to <em>hallucinate</em> them.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Without Skip Connections">
            This is just an autoencoder. The decoder must reconstruct
            pixel-precise output from compressed features. For denoising, this
            means blurry results&mdash;exactly what you saw in{' '}
            <strong>Autoencoders</strong>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>With skip connections</strong>, each decoder level receives
              features from the corresponding encoder level via concatenation. The
              encoder&rsquo;s 32&times;32&times;128 features are concatenated with
              the decoder&rsquo;s 32&times;32&times;128 features to produce
              32&times;32&times;256 features. A subsequent convolution reduces this
              back to 128 channels.
            </p>
            <p className="text-muted-foreground">
              The decoder now has access to <strong>both</strong>: global context
              from the upsampled lower levels (processed through the bottleneck)
              and local detail from the skip connections (directly from the
              encoder, never compressed). The global context tells it{' '}
              <em>what</em> to reconstruct. The local detail tells it{' '}
              <em>where</em> things are.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Keyhole and Side Doors">
            The bottleneck is a keyhole&mdash;global structure passes through. The
            skip connections are side doors&mdash;fine details bypass the keyhole
            entirely. The decoder gets both the gist and the specifics.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Autoencoder (No Skips)',
              color: 'rose',
              items: [
                'All information through the bottleneck',
                'Fine details lost during compression',
                'Blurry reconstructions',
                'Works for learning representations',
                'Fails for pixel-precise denoising',
              ],
            }}
            right={{
              title: 'U-Net (With Skips)',
              color: 'emerald',
              items: [
                'Global context through the bottleneck',
                'Fine details via skip connections',
                'Sharp, pixel-precise output',
                'Essential for denoising tasks',
                'Same building blocks, crucial difference',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* Section 7: Architecture Diagram */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full Architecture"
            subtitle="Tracing information flow through the U-Net"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete U-Net architecture. The left side is the
              encoder (downsampling). The right side is the decoder (upsampling).
              The horizontal arrows are skip connections. Notice the symmetric
              structure&mdash;the encoder descends, the bottleneck sits at the
              bottom, and the decoder ascends, forming the U shape the
              architecture is named for.
            </p>
          </div>
          <MermaidDiagram chart={`
graph TD
    subgraph Encoder["Encoder (Downsampling)"]
        E0["64×64×64<br/>Edges, textures"]
        E1["32×32×128<br/>Shapes, parts"]
        E2["16×16×256<br/>Object structure"]
    end

    subgraph BN["Bottleneck"]
        B["8×8×512<br/>Global context"]
    end

    subgraph Decoder["Decoder (Upsampling)"]
        D2["16×16×256<br/>+ encoder features"]
        D1["32×32×128<br/>+ encoder features"]
        D0["64×64×64<br/>+ encoder features"]
    end

    IN["Input: 64×64×3<br/>Noisy image"] --> E0
    E0 -->|"downsample"| E1
    E1 -->|"downsample"| E2
    E2 -->|"downsample"| B
    B -->|"upsample"| D2
    D2 -->|"upsample"| D1
    D1 -->|"upsample"| D0
    D0 --> OUT["Output: 64×64×3<br/>Predicted noise"]

    E2 -.->|"skip: concat"| D2
    E1 -.->|"skip: concat"| D1
    E0 -.->|"skip: concat"| D0

    style IN fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    style OUT fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    style B fill:#1e293b,stroke:#a855f7,color:#e2e8f0
`} />
          <p className="text-sm text-muted-foreground italic mt-4">
            You might wonder: if skip connections bypass the bottleneck, why have
            a bottleneck at all? That is exactly the right question&mdash;the next
            section answers it.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Reading the Diagram">
            Solid arrows show the main data flow: down through the encoder,
            through the bottleneck, up through the decoder. Dashed arrows show
            skip connections: encoder features concatenated directly with decoder
            features at matching resolutions.
          </ConceptBlock>
          <TryThisBlock title="Trace the Path">
            Follow a single pixel through the U-Net: down through all three
            encoder levels, through the bottleneck, back up through all three
            decoder levels. At each decoder level, it picks up features from the
            matching encoder level via concatenation. Count: how many times do
            the encoder features influence the final output?
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Why Both — Two-Path Information Flow */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why You Need Both Paths"
            subtitle="The bottleneck decides WHAT. The skip connections decide WHERE."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              An obvious question: if skip connections bypass the bottleneck, why
              have a bottleneck at all? Why not skip the downsampling path entirely
              and just pass everything through at full resolution?
            </p>
            <p className="text-muted-foreground">
              Without the downsampling path, the network has no global context. At{' '}
              <strong>t&nbsp;=&nbsp;900</strong>, it would try to refine
              pixel-level details in what is essentially pure static. It needs to
              first decide &ldquo;this is probably a shoe&rdquo; (global, from the
              bottleneck) before it can refine &ldquo;the toe is here&rdquo;
              (local, from the skip connections).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Common Misconception">
            &ldquo;If skip connections carry the details, the bottleneck is
            unnecessary overhead.&rdquo; No. Without the bottleneck, the network
            cannot make global decisions. You need BOTH: global context from the
            bottleneck AND local details from the skip connections.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-2">
            <GradientCard title="The Bottleneck Provides" color="violet">
              <ul className="space-y-1">
                <li>&bull; Global composition (shoe vs shirt)</li>
                <li>&bull; Overall structure and layout</li>
                <li>&bull; Coarse spatial arrangement</li>
                <li>&bull; Dominates at high noise levels</li>
              </ul>
            </GradientCard>
            <GradientCard title="Skip Connections Provide" color="blue">
              <ul className="space-y-1">
                <li>&bull; Exact edge positions</li>
                <li>&bull; Fine textures and details</li>
                <li>&bull; Pixel-precise spatial information</li>
                <li>&bull; Dominate at low noise levels</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Multi-Resolution Maps to Coarse-to-Fine */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multi-Resolution Meets Coarse-to-Fine"
            subtitle="The architecture mirrors the denoising task"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the &ldquo;of course&rdquo; moment. Remember the denoising
              trajectory from <strong>Sampling and Generation</strong>? You watched
              structure emerge first and details appear later. The U-Net&rsquo;s
              multi-resolution processing is <em>why</em>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Connection">
            The coarse-to-fine denoising progression you observed in the
            DenoisingTrajectoryWidget is not a coincidence. The U-Net&rsquo;s
            architecture <em>is</em> the mechanism that makes it happen.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <div className="rounded-lg bg-muted/30 p-4 space-y-2">
              <p className="text-sm font-medium text-foreground">
                At t&nbsp;=&nbsp;900 (heavily noised)
              </p>
              <p className="text-sm text-muted-foreground">
                The image is almost pure static. The skip connections carry mostly
                noise&mdash;there is little useful fine detail to preserve. The
                bottleneck layers do the heavy lifting: they detect whatever faint
                global structure remains and make the big structural decisions.
                Low-resolution features: &ldquo;this is roughly a shoe
                shape.&rdquo; Mid-resolution features: &ldquo;the sole is here,
                the opening is there.&rdquo; High-resolution features: mostly noise,
                minimal contribution.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-4 space-y-2">
              <p className="text-sm font-medium text-foreground">
                At t&nbsp;=&nbsp;50 (lightly noised)
              </p>
              <p className="text-sm text-muted-foreground">
                The structure is already correct from earlier denoising steps. The
                bottleneck barely changes anything&mdash;the global composition is
                fine. Now the high-resolution skip connections dominate. They carry
                the fine detail corrections: exact edge positions, subtle textures,
                the precise boundary between objects. The decoder uses these to
                make pixel-precise refinements.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a Routing Mechanism">
            The U-Net does NOT route different noise levels to different resolution
            layers. <strong>Every timestep processes through all resolution
            levels.</strong> But the importance of each level shifts with the
            noise level. The architecture does not change&mdash;the data does.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Same architecture. Same weights. Same forward pass. The only
              difference is the input (how noisy is the image) and the timestep
              (which tells the network how much noise to expect). The
              multi-resolution structure naturally adapts: at high noise, the
              bottleneck features matter most; at low noise, the skip connection
              features matter most.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Pseudocode */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Forward Pass in Code"
            subtitle="Pseudocode for information flow"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You saw real U-Net code in the capstone. Here is the same idea at a
              higher level of abstraction, focusing on the information flow pattern
              rather than PyTorch specifics:
            </p>
            <CodeBlock
              code={`def forward(self, x):  # timestep t omitted — next lesson
    # Encoder: compress, saving features at each level
    e0 = self.enc_block_0(x)         # 64x64x64
    e1 = self.enc_block_1(down(e0))  # 32x32x128
    e2 = self.enc_block_2(down(e1))  # 16x16x256

    # Bottleneck: maximum compression, global context
    b = self.bottleneck(down(e2))     # 8x8x512

    # Decoder: expand, concatenating encoder features
    d2 = self.dec_block_2(cat(up(b), e2))   # 16x16x256
    d1 = self.dec_block_1(cat(up(d2), e1))  # 32x32x128
    d0 = self.dec_block_0(cat(up(d1), e0))  # 64x64x64

    return self.output(d0)  # 64x64x3 (predicted noise)`}
              language="python"
              filename="unet_pseudocode.py"
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Channel Doubling at cat()">
            When encoder features (128 channels) are concatenated with decoder
            features (128 channels), the result is 256 channels. The subsequent
            conv block reduces this back to 128. This channel doubling is how
            skip connections inject information.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Notice the symmetry. The encoder compresses in three steps, saving
              features at each level. The decoder expands in three steps,
              concatenating the saved encoder features at each matching resolution.
              The bottleneck sits at the bottom of the U. This is the pattern you
              saw in <strong>Build a Diffusion Model</strong>&mdash;now you
              understand <em>why</em> each piece exists.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Residual Blocks (brief, INTRODUCED) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Inside Each Level: Residual Blocks"
            subtitle="The building blocks you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each &ldquo;block&rdquo; in the diagram above is not a single
              convolution. It is a <strong>residual block</strong>: Conv &rarr;
              Norm &rarr; Activation &rarr; Conv &rarr; Norm &rarr; Activation,
              plus a skip connection around the whole thing.
            </p>
            <p className="text-muted-foreground">
              These are the same residual blocks from{' '}
              <strong>ResNet</strong>&mdash;the identity shortcut that lets the
              network learn residuals and helps gradients flow. The only
              difference: diffusion U-Nets typically use <strong>group
              normalization</strong> instead of batch normalization, because it
              works better with the small batch sizes common in diffusion training.
              You do not need the details of group norm right now.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Kinds of Skip Connections">
            The U-Net has <strong>short-range</strong> skip connections (residual
            blocks within each level) and <strong>long-range</strong> skip
            connections (encoder-to-decoder across the U). In ResNet, skip
            connections primarily help <em>gradients flow</em> during training. In
            the U-Net, the long-range skip connections primarily carry{' '}
            <em>high-resolution spatial information</em> from encoder to decoder.
            Same mechanism, fundamentally different purpose.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Check — Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Your colleague proposes removing all skip connections to reduce
                memory usage. <strong>What happens to the denoised output at
                t&nbsp;=&nbsp;50 vs t&nbsp;=&nbsp;900?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    <strong>At t&nbsp;=&nbsp;900:</strong> Barely affected. The image
                    is mostly noise anyway, and the bottleneck is doing most of the
                    work. The skip connections were carrying noise, not useful detail.
                  </p>
                  <p>
                    <strong>At t&nbsp;=&nbsp;50:</strong> Catastrophically worse.
                    The structure is correct (from earlier steps), but the fine
                    details&mdash;exact edge positions, textures, subtle
                    features&mdash;are lost. The decoder must hallucinate them from
                    the bottleneck. The output becomes blurry, just like autoencoder
                    reconstructions.
                  </p>
                  <p>
                    This asymmetry is the key insight: skip connections matter most
                    at low noise levels, where fine details need to be preserved.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 13: Check — Transfer Question */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Question" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                In the original 2015 paper, U-Net was designed for{' '}
                <strong>medical image segmentation</strong>&mdash;labeling each
                pixel of a brain scan as tumor, tissue, or background.{' '}
                <strong>Why would the same architecture be useful for a
                completely different task like denoising?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Both tasks require <strong>pixel-precise output</strong> that
                    combines global understanding with local detail. Segmentation
                    needs to know &ldquo;this region is a tumor&rdquo; (global
                    context) AND &ldquo;the boundary of the tumor is exactly
                    here&rdquo; (local precision). Denoising needs &ldquo;this is
                    a shoe&rdquo; (global context) AND &ldquo;the edge of the
                    shoe is exactly here&rdquo; (local precision).
                  </p>
                  <p>
                    Same architectural need, different task. The U-Net&rsquo;s
                    combination of global context (bottleneck) and local detail
                    (skip connections) is the solution to both.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 14: Brief mention of attention */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before we summarize, one preview of what is coming next. In real
              diffusion U-Nets,{' '}
              <strong>attention layers</strong> are interleaved at certain
              resolution levels&mdash;typically the middle resolutions (16&times;16
              and 32&times;32). Self-attention lets the network relate distant
              parts of the feature map (left side of the shoe to right side),
              and cross-attention connects the image features to text embeddings
              for text-to-image generation. You will see these in detail in
              upcoming lessons. For now, just know they exist as additional
              components within the U-Net.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Coming Soon">
            Attention layers are what make text-to-image possible. The spatial
            architecture you learned today is the foundation. Attention and
            conditioning are layered on top.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 15: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'The U-Net is an encoder-decoder with skip connections—not just an autoencoder.',
                description:
                  'The skip connections carry fine-grained spatial details that the bottleneck would destroy. This is what enables pixel-precise denoising.',
              },
              {
                headline:
                  'The multi-resolution structure maps to the coarse-to-fine denoising progression.',
                description:
                  'Low-resolution layers provide global context; high-resolution layers preserve local detail. Both are needed at every timestep, but their relative importance shifts with the noise level.',
              },
              {
                headline:
                  'Skip connections are essential, not optional.',
                description:
                  'Without them, the decoder produces blurry output—the same problem as autoencoder reconstructions. They are not just a training trick; they are an architectural necessity.',
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
              The bottleneck decides WHAT. The skip connections decide WHERE. The
              U-Net gives you both at every resolution.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Section 16: References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
                authors: 'Ronneberger, Fischer & Brox, 2015',
                url: 'https://arxiv.org/abs/1505.04597',
                note: 'The original U-Net paper. Section 2 describes the architecture. Designed for medical segmentation, later adopted by diffusion models.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM paper. Appendix B describes the U-Net architecture used for diffusion.',
              },
              {
                title: 'Diffusion Models Beat GANs on Image Synthesis',
                authors: 'Dhariwal & Nichol, 2021',
                url: 'https://arxiv.org/abs/2105.05233',
                note: 'Introduces the improved U-Net architecture with attention and adaptive group normalization. Section 4 details the architecture changes.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 17: Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now understand the spatial architecture. But the network needs to
              know one more thing: what noise level is it denoising? The same U-Net
              handles <strong>t&nbsp;=&nbsp;50</strong> (barely noisy) and{' '}
              <strong>t&nbsp;=&nbsp;900</strong> (pure static), but it needs to
              behave very differently for each. Next, you will see how timestep
              embeddings give the network that awareness.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Timestep Conditioning"
            description="How sinusoidal embeddings and adaptive normalization let a single U-Net handle every noise level&mdash;from pure static to nearly clean."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
