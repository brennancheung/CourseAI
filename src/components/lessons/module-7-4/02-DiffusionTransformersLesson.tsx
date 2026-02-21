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
  ReferencesBlock,
  NextStepBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-4-2-diffusion-transformers.ipynb'

/**
 * Diffusion Transformers (DiT)
 *
 * Lesson 2 in Module 7.4 (Next-Generation Architectures). Tenth lesson
 * in Series 7.
 * Cognitive load: STRETCH (2-3 genuinely new concepts: ViT on latent patches
 * elevated from MENTIONED to DEVELOPED, adaLN-Zero conditioning, scaling laws
 * for DiT).
 *
 * DiT replaces the U-Net with a standard vision transformer operating on
 * latent patches. Two deep knowledge threads—transformers from Series 4 and
 * latent diffusion from Series 6—converge. Every component of the DiT block
 * is familiar except the conditioning mechanism (adaLN-Zero). The lesson
 * shows how known pieces fit together in a new context.
 *
 * Builds on: transformer block (4.2.5), multi-head attention (4.2.4),
 * adaptive group norm (6.3.2), latent diffusion (6.3.5), U-Net (6.3.1),
 * cross-attention (6.3.4), SDXL (7.4.1), ControlNet zero conv (7.1.1),
 * flow matching (7.2.2)
 *
 * Previous: SDXL (Module 7.4, Lesson 1 / BUILD)
 * Next: SD3 & Flux (Module 7.4, Lesson 3)
 */

export function DiffusionTransformersLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Diffusion Transformers (DiT)"
            description="Replace the U-Net entirely—patchify the latent into tokens, process with standard transformer blocks, condition via adaptive layer norm with a zero-initialized gate. Two knowledge threads converge."
            category="Next-Generation Architectures"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain how the Diffusion Transformer (DiT) replaces the
            U-Net&rsquo;s convolutional encoder-decoder with a standard vision
            transformer operating on latent patches, using adaptive layer norm
            (adaLN-Zero) for timestep/class conditioning, and why this
            architecture scales more predictably than U-Nets.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="STRETCH Lesson">
            Two to three genuinely new concepts: patchifying latents into tokens
            (ViT on latent patches), adaLN-Zero conditioning, and transformer
            scaling laws for diffusion. The stretch is mitigated by deep
            transformer knowledge from Series 4—you already know ~80% of DiT.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The DiT architecture from Peebles & Xie (2023)—patchify, transformer blocks, adaLN-Zero, unpatchify',
              'Why transformers scale more predictably than U-Nets (the "two knobs" recipe)',
              'DiT is class-conditional on ImageNet—it does NOT use text prompts',
              'NOT: text conditioning via joint attention (next lesson, SD3/Flux)',
              'NOT: ViT pretraining on classification tasks (DiT is trained from scratch on diffusion)',
              'NOT: every DiT variant (U-ViT, MDT, etc.)—only the original DiT',
              'NOT: full DiT training from scratch (computational requirements are too large)',
              'NOT: the SD3/Flux/MMDiT architecture (next lesson)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — brief reactivation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Concepts to Reactivate"
            subtitle="Concepts you already know deeply"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This lesson connects three things you have built up over the
              course. A quick reactivation before they converge:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>The transformer block</strong> (from{' '}
                <strong>The Transformer Block</strong>): LayerNorm →
                Multi-Head Attention → residual → LayerNorm → FFN → residual.
                &ldquo;Attention reads, FFN writes.&rdquo; Shape-preserving:{' '}
                <InlineMath math="(n, d_{\text{model}})" /> in,{' '}
                <InlineMath math="(n, d_{\text{model}})" /> out. Stack N of them
                identically.
              </li>
              <li>
                <strong>Adaptive group norm</strong> (from{' '}
                <strong>Conditioning the U-Net</strong>): the timestep produces{' '}
                <InlineMath math="\gamma(t)" /> and{' '}
                <InlineMath math="\beta(t)" /> that scale and shift the
                normalized features. The normalization parameters become
                conditioning-dependent. Every residual block gets its own
                projection from the timestep embedding.
              </li>
              <li>
                <strong>SDXL&rsquo;s three limitations</strong> (from{' '}
                <strong>SDXL</strong>): (1) no clear scaling recipe—making a
                U-Net bigger requires many ad hoc engineering decisions, (2)
                text-image interaction only at cross-attention layers, not
                everywhere, (3) convolutions hard-code local spatial structure,
                which is both a useful prior and a limiting assumption.
              </li>
            </ul>
            <p className="text-muted-foreground">
              What if you replaced the U-Net with an architecture that scales
              predictably, has global interaction at every layer, and learns
              spatial relationships from data instead of hard-coding them? You
              already know such an architecture. You built it in Series 4.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Convergence">
            This lesson is where two deep knowledge threads meet: transformers
            from Series 4 and latent diffusion from Series 6. Everything
            converges into one architecture.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — "Of Course" chain + convergence reveal */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Threads Converge"
            subtitle="You already have the pieces"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have two deep knowledge threads running through this course:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Thread 1: Transformers (Series 4)" color="sky">
                <div className="space-y-2 text-sm">
                  <p>
                    You built GPT from scratch. Self-attention lets every token
                    attend to every other token. FFN transforms what attention
                    finds. The block stacks identically. GPT-2 (124M) scales to
                    GPT-3 (175B) by increasing d_model and N. Same architecture,
                    different scale.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Thread 2: Latent Diffusion (Series 6)" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    You built a diffusion model. The denoising network operates
                    on a fixed-size latent tensor [4, H, W]. Timestep
                    conditioning via adaptive normalization. The U-Net processes
                    this tensor with convolutions and attention.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              <strong>This lesson is where these two threads meet.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The &ldquo;Of Course&rdquo; Chain">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>The denoising network processes a fixed-size tensor.</li>
              <li>A fixed-size spatial tensor can be split into patches.</li>
              <li>Patches are just tokens.</li>
              <li>
                You already know the best architecture for processing token
                sequences.
              </li>
              <li>
                Of course the field replaced the U-Net with a transformer.
              </li>
            </ol>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Challenge: predict before learning */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <details className="rounded-lg border bg-muted/30 p-4">
              <summary className="font-medium cursor-pointer text-primary">
                Challenge: predict before you read on
              </summary>
              <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                <p>
                  If you were designing a transformer-based denoising network,
                  what would you need to figure out? Think about what changes
                  when the input is an image latent instead of text tokens.
                </p>
                <p className="mt-2 italic">
                  Expected answers: how to turn the image into tokens, how to
                  inject the timestep, how to get an image back out at the end.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* Section 5: Explain — Tokenize the Image (Patchify) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tokenize the Image"
            subtitle="From latent tensor to token sequence"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Series 4, text became a token sequence: words → tokenizer →
              token IDs → embedding lookup →{' '}
              <InlineMath math="[n, d_{\text{model}}]" />. The transformer
              processes this sequence.
            </p>
            <p className="text-muted-foreground">
              For images, the equivalent operation is{' '}
              <strong>patchify</strong>: take the latent tensor{' '}
              <InlineMath math="[C, H, W]" /> and split it into non-overlapping
              patches of size <InlineMath math="p \times p" />. The structural
              correspondence is exact:
            </p>

            <ComparisonRow
              left={{
                title: 'Text Pipeline (Series 4)',
                color: 'sky',
                items: [
                  'Input: raw text',
                  'Tokenize: words → BPE → token IDs',
                  'Embed: token ID → embedding table → [d_model]',
                  'Result: [n, d_model] token sequence',
                ],
              }}
              right={{
                title: 'Image Pipeline (DiT)',
                color: 'violet',
                items: [
                  'Input: latent tensor [C, H, W]',
                  'Patchify: spatial grid → non-overlapping p×p patches → flatten',
                  'Project: flattened patch → nn.Linear → [d_model]',
                  'Result: [L, d_model] patch token sequence',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Both pipelines end at the same place: a sequence of{' '}
              <InlineMath math="d_{\text{model}}" />-dimensional vectors. The
              transformer processes both identically. Here is the concrete shape
              trace for DiT:
            </p>

            <CodeBlock
              code={`Noisy latent: [4, 32, 32]  (e.g., 256×256 image with 8× VAE downsampling)
Patch size: p = 2

Step 1: Split into patches
  32/2 = 16 patches per row, 16 patches per column
  Total patches: 16 × 16 = 256 tokens

Step 2: Flatten each patch
  Each patch is [4, 2, 2] = 4 × 2 × 2 = 16 dimensions per patch
  Sequence: [256 tokens, 16 dims]

Step 3: Linear projection to model dimension
  Project [256, 16] → [256, d_model] via nn.Linear(16, d_model)
  For DiT-XL/2: d_model = 1152
  Sequence: [256, 1152]

Step 4: Add positional embeddings
  Same concept as positional encoding from Series 4
  Learned positional embeddings: [256, 1152]
  Final input: [256, 1152]`}
              language="text"
              filename="patchify_trace.txt"
            />

            <p className="text-muted-foreground">
              Step 4 adds learned positional embeddings—the same concept as
              GPT-2&rsquo;s positional encoding, extended to 2D patch positions.
              Without them, the transformer treats the patch sequence as an
              unordered set: it cannot distinguish patch (0,0) from patch
              (15,15). The positional embeddings are DiT&rsquo;s only spatial
              prior—everything else about spatial relationships is learned from
              data. This is what &ldquo;minimal spatial bias&rdquo; means in the
              architecture comparison below.
            </p>

            <p className="text-muted-foreground">
              The transformer now has a sequence of 256 tokens, each with 1152
              dimensions. It does not know or care that these tokens came from
              image patches rather than text. The architecture is the same.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Tokenize the Image">
            In Series 4, text goes from words to tokens to embeddings. Here,
            the latent goes from a spatial grid to patches to embeddings.{' '}
            <strong>Same pipeline, different input modality.</strong> The
            transformer processes patch tokens the same way it processes word
            tokens.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Sequence length and patch size */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sequence Length and Patch Size"
            subtitle="The resolution-compute tradeoff"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The number of tokens depends on the latent resolution and patch
              size:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="L = \frac{H}{p} \times \frac{W}{p}" />
            </div>
            <p className="text-muted-foreground">
              Three concrete examples with a{' '}
              <InlineMath math="[4, 64, 64]" /> latent (from a 512×512 image):
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="p = 8" />: <InlineMath math="L = 8 \times 8 = 64" />{' '}
                tokens
              </li>
              <li>
                <InlineMath math="p = 4" />: <InlineMath math="L = 16 \times 16 = 256" />{' '}
                tokens
              </li>
              <li>
                <InlineMath math="p = 2" />: <InlineMath math="L = 32 \times 32 = 1024" />{' '}
                tokens
              </li>
            </ul>
            <p className="text-muted-foreground">
              Smaller patch size = more tokens = finer spatial detail = quadratic
              attention cost. This is the same tradeoff you saw in{' '}
              <LessonLink slug="sdxl">SDXL</LessonLink>: more tokens means{' '}
              <InlineMath math="O(L^2)" /> attention compute.{' '}
              <strong>Patch size is the resolution knob for DiT.</strong>
            </p>
            <p className="text-muted-foreground">
              In <LessonLink slug="sdxl">SDXL</LessonLink>, you traced these exact token counts: 256
              at 16×16, 1024 at 32×32, 4096 at 64×64. DiT gives you explicit
              control over this tradeoff via the patch size parameter.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Positional Embeddings">
            Positional embeddings for patches work the same way as for text
            tokens. Each patch position gets a learned vector. Without them, the
            transformer cannot distinguish patch (0,0) from patch (15,15). DiT
            uses standard learnable positional embeddings—same as GPT-2.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Patchify predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A DiT model uses patch size <InlineMath math="p = 4" /> on a
                  latent of size [4, 64, 64]. How many tokens does the
                  transformer process? What is the dimension of each token
                  before projection?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <InlineMath math="(64/4) \times (64/4) = 16 \times 16 = 256" />{' '}
                    tokens. Each token has dimension{' '}
                    <InlineMath math="4 \times 4 \times 4 = 64" /> before
                    projection to d_model.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  If you halve the patch size from <InlineMath math="p = 4" />{' '}
                  to <InlineMath math="p = 2" /> on the same latent, what
                  happens to: (a) the number of tokens, (b) the attention
                  compute cost?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    (a) Tokens quadruple from 256 to 1024 because{' '}
                    <InlineMath math="(64/2)^2 = 1024" />. (b) Attention compute
                    increases by 16× because it is{' '}
                    <InlineMath math="O(L^2)" />:{' '}
                    <InlineMath math="(1024/256)^2 = 16" />. Same quadratic
                    tradeoff from SDXL, now controlled by patch size.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague claims: &ldquo;DiT&rsquo;s patchify is
                  fundamentally different from tokenization in LLMs.&rdquo; Is
                  this accurate?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The operations are analogous but not identical. Both convert
                    raw input into a sequence of embedding vectors that
                    transformers process. Text tokenization uses a learned
                    vocabulary with discrete token IDs; patchify uses a learned
                    linear projection on continuous pixel values. The key
                    similarity: both produce{' '}
                    <InlineMath math="[n, d_{\text{model}}]" /> sequences that
                    the transformer processes identically. The transformer does
                    not know what kind of tokens it received.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — The DiT Block */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The DiT Block"
            subtitle="The transformer block you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A DiT block is a standard transformer block: self-attention + FFN
              + residual connections + normalization. Here is what you recognize:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Multi-head self-attention with Q/K/V projections—same as{' '}
                <strong>Multi-Head Attention</strong>
              </li>
              <li>
                Feed-forward network with expansion (d_model → 4·d_model →
                d_model, GELU)—same as{' '}
                <strong>The Transformer Block</strong>
              </li>
              <li>
                Residual connections (<InlineMath math="x + f(x)" />) around
                both sub-layers—same as{' '}
                <strong>The Transformer Block</strong>
              </li>
              <li>
                Normalization before each sub-layer (pre-norm)—same as{' '}
                <strong>The Transformer Block</strong>
              </li>
            </ul>
            <p className="text-muted-foreground">
              <strong>
                Every component is the one you built. The difference: how the
                normalization parameters are computed.
              </strong>
            </p>
            <p className="text-muted-foreground">
              You already know every component of this block except the
              conditioning mechanism. DiT is not a new architecture—it is your
              transformer applied to patches with a specific conditioning
              technique.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not New—Familiar">
            Strip away the conditioning and you have a standard ViT. Strip away
            the patches and you have the same transformer block from GPT. The
            student can identify every component except the conditioning.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Architecture comparison: U-Net vs DiT */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Architecture Comparison"
            subtitle="U-Net vs DiT side by side"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'U-Net (SD v1.5, SDXL)',
                color: 'amber',
                items: [
                  'Basic unit: conv residual block + optional attention',
                  'Multi-resolution (64×64 → 32×32 → 16×16 → 8×8 → back up)',
                  'Encoder-to-decoder skip connections across resolution levels',
                  'Attention only at middle resolutions (16×16, 32×32)',
                  'Convolutions at every layer (primary spatial operation)',
                  'Strong spatial inductive bias (local connectivity, translational equivariance)',
                  'Ad hoc scaling (which channels? which resolutions? how deep?)',
                ],
              }}
              right={{
                title: 'DiT',
                color: 'emerald',
                items: [
                  'Basic unit: standard transformer block (MHA + FFN)',
                  'Single resolution (all patches at same level)',
                  'No skip connections (only within-block residual connections)',
                  'Attention at every layer, every patch token',
                  'No convolutions (only linear projection for patchify/unpatchify)',
                  'Minimal spatial bias (position embeddings only; relationships learned from data)',
                  'Systematic scaling (increase d_model and/or N—same as GPT)',
                ],
              }}
            />
            <p className="text-muted-foreground">
              U-Net skip connections exist because downsampling destroys spatial
              detail that the decoder needs to recover. DiT never downsamples.
              Every layer operates on the full set of patch tokens at the same
              resolution. There is no lost spatial information to skip across.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Simpler, Not More Complex">
            Notice what DiT <strong>removes</strong>: convolutions,
            encoder-decoder hierarchy, skip connections, resolution changes. And
            what it <strong>keeps</strong>: self-attention, FFN, residual
            connections, normalization. The transformer block is simpler than
            the U-Net block. Fewer hand-designed components, fewer engineering
            decisions.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Explain — adaLN-Zero Conditioning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="adaLN-Zero Conditioning"
            subtitle="The one genuinely new mechanism"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The DiT block needs to know the timestep (and optionally the class
              label). In the U-Net, this was handled by adaptive group
              normalization:{' '}
              <InlineMath math="\gamma(t)" /> and{' '}
              <InlineMath math="\beta(t)" /> from the timestep embedding
              modulate GroupNorm. What is the equivalent for a transformer block
              that uses LayerNorm?
            </p>
            <p className="text-muted-foreground">
              <strong>Adaptive layer norm (adaLN)</strong>: make LayerNorm
              parameters depend on the conditioning signal. Instead of learned{' '}
              <InlineMath math="\gamma" /> and <InlineMath math="\beta" />,
              predict <InlineMath math="\gamma(c)" /> and{' '}
              <InlineMath math="\beta(c)" /> from the conditioning vector{' '}
              <InlineMath math="c" /> (which encodes timestep + class label).
              Same idea as adaptive group norm: replace fixed normalization
              parameters with conditioning-dependent ones.
            </p>
            <p className="text-muted-foreground">
              The DiT paper tested several conditioning approaches. The winner
              adds one more parameter: a <strong>gate</strong>{' '}
              <InlineMath math="\alpha" />.
            </p>

            <CodeBlock
              code={`Conditioning vector c (timestep + class embedding)
  → MLP → six parameters: (γ₁, β₁, α₁, γ₂, β₂, α₂)

MHA sub-layer:
  h = γ₁ · LayerNorm(x) + β₁      (adaptive layer norm)
  h = MultiHeadAttention(h)         (standard MHA)
  x' = x + α₁ · h                  (gated residual)

FFN sub-layer:
  h = γ₂ · LayerNorm(x') + β₂     (adaptive layer norm)
  h = FFN(h)                        (standard FFN)
  output = x' + α₂ · h             (gated residual)`}
              language="text"
              filename="adaln_zero_block.txt"
            />

            <p className="text-muted-foreground">
              Six parameters per block, from one conditioning signal. Three per
              sub-layer: scale (<InlineMath math="\gamma" />), shift (
              <InlineMath math="\beta" />), gate (<InlineMath math="\alpha" />
              ).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Idea, New Context">
            AdaGN makes GroupNorm parameters depend on the timestep. adaLN makes
            LayerNorm parameters depend on the conditioning. Same principle,
            different normalization type, with one addition: the gate.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* adaLN-Zero: the zero initialization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Zero?"
            subtitle="The critical design choice"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At initialization, <strong>all alpha values are set to zero</strong>.
              This means:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <BlockMath math="x' = x + 0 \cdot \text{MHA}(\ldots) = x" />
              <BlockMath math="\text{output} = x' + 0 \cdot \text{FFN}(\ldots) = x' = x" />
            </div>

            <p className="text-muted-foreground">
              The entire block is an identity function. The input passes through
              unchanged.
            </p>
            <p className="text-muted-foreground">
              You have seen this pattern before. In{' '}
              <strong>ControlNet</strong>, zero convolution ensures the trainable
              encoder starts contributing nothing to the frozen decoder. In LoRA,
              the B matrix is initialized to zero so the bypass starts at zero.
              adaLN-Zero follows the same principle: the transformer block starts
              as a no-op, then gradually learns what to contribute.
            </p>

            <ComparisonRow
              left={{
                title: 'Adaptive Group Norm (U-Net)',
                color: 'amber',
                items: [
                  'Normalization: GroupNorm',
                  'Parameters from conditioning: γ, β (2 per layer)',
                  'Gate parameter: No',
                  'Initialization: γ=1, β=0 (standard norm behavior)',
                  'Zero-initialization: Not used',
                ],
              }}
              right={{
                title: 'adaLN-Zero (DiT)',
                color: 'emerald',
                items: [
                  'Normalization: LayerNorm',
                  'Parameters from conditioning: γ, β, α (3 per sub-layer, 6 per block)',
                  'Gate parameter: Yes (α)',
                  'Initialization: α=0 (block starts as identity)',
                  'Zero-initialization: Core design choice',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The &ldquo;Zero&rdquo; Matters">
            The &ldquo;Zero&rdquo; in adaLN-Zero is not just a name. It is the
            critical design choice. Every DiT block starts as an identity
            function: input in, same input out. The model gradually learns which
            blocks should contribute what. Same safety pattern as ControlNet and
            LoRA: start contributing nothing, learn what to add.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="adaLN-Zero predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A DiT model has N=28 blocks, each with 6 adaLN-Zero
                  parameters (γ₁, β₁, α₁, γ₂, β₂, α₂). At initialization,
                  what does the entire N-block model compute?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Every block is an identity function because all α values are
                    zero. The entire model outputs its input unchanged:
                    patchified latent in, same patchified latent out. The model
                    must <em>learn</em> to denoise through training. This is
                    different from the U-Net, where the architecture starts with
                    nonzero contributions from every layer.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  You know adaptive group norm from{' '}
                  <strong>Conditioning the U-Net</strong> uses γ(t) and β(t) but
                  no gate. What happens if you add a gate to AdaGN and
                  initialize it to zero?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    You would get the same identity-at-initialization property
                    for U-Net residual blocks. This was not done historically
                    because the U-Net architecture predates the
                    zero-initialization insight. The DiT paper compared multiple
                    conditioning variants and adaLN-Zero performed best. The
                    gate+zero-initialization combination outperformed standard
                    scale+shift.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A DiT block at initialization: α=0, so the block is identity.
                  After training, α has learned nonzero values. Some blocks
                  might have small α values (near-identity) and others might
                  have large α values (strong contribution). Does this remind
                  you of anything from Series 4?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    This is similar to how different transformer blocks in GPT
                    serve different roles—some early blocks primarily copy/route
                    information, some later blocks do more transformation. The α
                    values provide a learned measure of each block&rsquo;s
                    contribution magnitude. In practice, different blocks
                    specialize for different noise levels and spatial scales.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Explain — From Patches Back to Image (Unpatchify) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="From Patches Back to Image"
            subtitle="Reverse the patchify operation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After N transformer blocks, the model has a sequence of{' '}
              <InlineMath math="[L, d_{\text{model}}]" /> patch tokens. To get
              back to a spatial latent, reverse the patchify operation:
            </p>

            <CodeBlock
              code={`Transformer output: [256, 1152]  (L=256 tokens, d_model=1152)

Step 1: Linear projection to patch dimensions
  [256, 1152] → [256, 4·p·p] = [256, 16]  via nn.Linear(1152, 16)

Step 2: Reshape to spatial grid
  [256, 16] → [256, 4, 2, 2] → rearrange to [4, 32, 32]

Predicted noise (or velocity): [4, 32, 32]—same shape as the noisy latent input`}
              language="text"
              filename="unpatchify_trace.txt"
            />

            <p className="text-muted-foreground">
              The output is a denoised latent (or a noise prediction, or a
              velocity prediction—the output parameterization is independent of
              the architecture). It enters the same diffusion sampling loop and
              VAE decode that you know from Series 6.
            </p>
            <p className="text-muted-foreground">
              <strong>
                DiT replaces ONLY the denoising network.
              </strong>{' '}
              The VAE still encodes to [4, H, W]. The text encoder (when used)
              still produces embeddings. The noise schedule or flow matching
              objective still applies. DiT slots into the same latent diffusion
              pipeline: text encode → denoise in latent space → VAE decode. Only
              the middle box changes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Same Pipeline, Different Middle">
            Do not think of DiT as a complete replacement of the SD pipeline.
            The VAE, text encoder, noise schedule, and sampling algorithm are
            all unchanged. Only the denoising network—the box in the middle of
            the loop—is different.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Pipeline predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  After the DiT produces its [4, 32, 32] output, what happens
                  next in the generation pipeline?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The same thing as with a U-Net: the output is used in the
                    sampling step (DDPM reverse step, DDIM step, ODE solver
                    step, etc.) to compute the next latent. After all sampling
                    steps, the final z₀ is decoded by the frozen VAE to pixel
                    space [3, 256, 256]. The rest of the pipeline is identical.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Could you use a DiT denoising network with DPM-Solver++ or
                  LCM-LoRA?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    In principle, yes for any solver (DPM-Solver++ works on any
                    denoising network). For LCM-LoRA specifically, the LoRA
                    would need to target the DiT&rsquo;s attention projections
                    (W_Q, W_K, W_V) instead of U-Net cross-attention
                    projections, and the matrix dimensions would be different.
                    The acceleration approaches from{' '}
                    <strong>The Speed Landscape</strong> are solver/training
                    techniques that are architecture-independent in concept,
                    though practical implementations need to match the specific
                    architecture&rsquo;s weight shapes.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 12: Explain — Why Transformers Scale Better */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Transformers Scale Better"
            subtitle="The U-Net scaling problem, solved"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="sdxl">SDXL</LessonLink>, you identified a limitation: there is no
              clear recipe for making a U-Net bigger. To double the parameters,
              you must decide:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Wider channels at which resolution levels?</li>
              <li>More attention blocks at which resolutions?</li>
              <li>Deeper encoder or deeper decoder?</li>
              <li>Add attention at higher resolutions (expensive)?</li>
              <li>
                How do skip connections work with asymmetric encoder/decoder?
              </li>
            </ul>
            <p className="text-muted-foreground">
              Each decision is a manual engineering choice. There is no
              &ldquo;GPT-2 → GPT-3&rdquo; recipe for U-Nets.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Ad Hoc vs Systematic">
            Scaling a U-Net is like renovating a house with custom parts—each
            room is different, each wall has bespoke plumbing. Scaling a
            transformer is like stacking LEGO: increase the brick size (d_model)
            or add more layers (N).
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The transformer scaling recipe */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Transformer Scaling Recipe"
            subtitle="Two knobs, not twenty"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Transformers have two primary scaling knobs:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>d_model</strong>—the hidden dimension per token (wider)
              </li>
              <li>
                <strong>N</strong>—the number of transformer blocks (deeper)
              </li>
            </ol>
            <p className="text-muted-foreground">
              GPT-2 (d_model=768, N=12, 124M params) → GPT-3 (d_model=12288,
              N=96, 175B params). Same architecture, different scale. The
              scaling recipe is: increase d_model and N.
            </p>
            <p className="text-muted-foreground">
              The DiT paper tested this systematically:
            </p>

            <CodeBlock
              code={`DiT Model Family:
  DiT-S:   d_model=384,  N=12,  ~33M params
  DiT-B:   d_model=768,  N=12,  ~130M params
  DiT-L:   d_model=1024, N=24,  ~458M params
  DiT-XL:  d_model=1152, N=28,  ~675M params

Each model uses the same architecture with different scale parameters.
Result: loss decreases predictably as compute increases.
Bigger DiT = better images, smooth improvement curve.`}
              language="text"
              filename="dit_model_family.txt"
            />

            <p className="text-muted-foreground">
              Convolutions embed spatial priors that are genuinely useful: local
              connectivity and translational equivariance make CNNs
              data-efficient for small datasets. But at the scale of modern
              diffusion training (hundreds of millions of images), the
              transformer&rsquo;s ability to learn spatial relationships from
              data outperforms the CNN&rsquo;s hard-coded assumptions. The
              tradeoff: CNNs are more efficient at small scale, transformers are
              more capable at large scale. Modern diffusion training operates in
              the regime where transformers win.
            </p>
            <p className="text-muted-foreground">
              This mirrors the NLP story: RNNs had useful sequential inductive
              biases, but transformers won because they scale better with data
              and compute.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Knobs, Not Twenty">
            Scaling a U-Net requires deciding which channels to widen, which
            resolutions to add attention, how deep to make each stage. Scaling a
            DiT requires choosing d_model and N. The simplicity of the scaling
            recipe is the architectural advantage.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* DiT scaling results */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The DiT paper showed class-conditional ImageNet generation
              results: DiT-XL/2 (patch size 2, 675M params) achieved{' '}
              <strong>FID 2.27</strong> on ImageNet 256×256 class-conditional
              generation—state-of-the-art at the time, outperforming the best
              U-Net models (ADM, LDM).
            </p>
            <p className="text-muted-foreground">
              The FID number is less important than the trend: bigger DiT =
              better results, with a smooth relationship. No engineering
              plateaus, no diminishing returns from ad hoc architectural
              decisions. The same scaling law behavior that made GPT successful
              applies to image generation.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 13: Check #4 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Scaling transfer questions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Suppose you want to build a DiT twice as large as DiT-XL. What
                  would you change?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Increase d_model beyond 1152, increase N beyond 28, or both.
                    The same two knobs. No structural decisions required.
                    Compare: doubling a U-Net requires choosing WHERE to add the
                    parameters—wider channels at what resolution? More attention
                    blocks where? There is no clear answer.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  The original DiT paper uses class-conditional generation
                  (ImageNet classes, no text). Does the scaling argument depend
                  on this? Would you expect the same scaling behavior with text
                  conditioning?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The scaling argument is about the denoising backbone
                    architecture, not the conditioning type. Whether you
                    condition on class labels, text embeddings, or something
                    else, the transformer&rsquo;s scaling properties come from
                    the architecture itself. SD3 and Flux—which use text
                    conditioning with DiT-style backbones—confirm this: they
                    scale the transformer and get predictable improvements. This
                    is the subject of the next lesson.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 14: Elaborate — The Complete DiT Pipeline */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete DiT Pipeline"
            subtitle="What changes and what is preserved"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The full DiT pipeline for class-conditional ImageNet generation,
              placed within the latent diffusion framework you already know:
            </p>

            <CodeBlock
              code={`Full DiT Pipeline (class-conditional ImageNet):
 1. Sample class label y → class embedding [d_model]
 2. Timestep t → sinusoidal embedding → MLP → timestep embedding [d_model]
 3. Combine: c = class_emb + timestep_emb  (conditioning vector)
 4. Noisy latent z_t [4, 32, 32] → patchify → [256, d_model]
 5. Add positional embeddings → [256, d_model]
 6. N DiT blocks with adaLN-Zero conditioned on c → [256, d_model]
 7. Final adaLN-Zero + linear projection → [256, 4·p·p]
 8. Unpatchify → [4, 32, 32]  (predicted noise or velocity)
 9. Sampling step (DDPM, DDIM, etc.) → next z_{t-1}
10. Repeat steps 4-9 for all timesteps
11. VAE decode z_0 → [3, 256, 256] image`}
              language="text"
              filename="dit_pipeline.txt"
            />

            <p className="text-muted-foreground">
              Steps 1-2 are conditioning preparation. Steps 4-8 are the
              DiT-specific part (the denoising network). Steps 9-11 are the same
              as every diffusion model you have seen.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Pipeline Orientation">
            Steps 1-3: conditioning prep. Steps 4-8: DiT-specific (the new
            part). Steps 9-11: unchanged from Series 6. You already know 6 of
            the 11 steps.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* What DiT enables */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What DiT Enables"
            subtitle="Possibilities the architecture change opens up"
          />
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard
                title="Joint Text-Image Attention (MMDiT)"
                color="emerald"
              >
                <p className="text-sm">
                  If the denoising network processes a sequence of tokens, you
                  can concatenate text tokens and image tokens into one sequence
                  and let them attend to each other. No more separate
                  cross-attention—text and image interact through the same
                  self-attention mechanism.
                </p>
              </GradientCard>
              <GradientCard
                title="Architecture-Agnostic Training"
                color="sky"
              >
                <p className="text-sm">
                  DiT&rsquo;s architecture is independent of the training
                  objective. You can train with DDPM noise prediction, velocity
                  prediction, or flow matching. SD3 uses flow matching (from{' '}
                  <strong>Flow Matching</strong>) with a DiT backbone. Two
                  independent improvements that compose.
                </p>
              </GradientCard>
              <GradientCard title="Clear Scaling Path" color="amber">
                <p className="text-sm">
                  From DiT-S to DiT-XL, the same architecture scales smoothly.
                  This path continues: SD3 and Flux use larger DiT variants
                  with billions of parameters. The scaling recipe discovered in
                  DiT directly enables frontier models.
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              The next lesson shows how these possibilities converge in SD3 and
              Flux.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 15: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: DiT Architecture"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the lesson concrete—implement patchify from
                scratch, trace adaLN-Zero conditioning, inspect a pretrained DiT
                model, and generate class-conditional images.
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
              <ul className="text-sm text-muted-foreground space-y-3">
                <li>
                  <strong>Exercise 1 (Guided): Patchify and Unpatchify.</strong>{' '}
                  Implement the patchify operation: take a random tensor [4, 32,
                  32], split into patches of size p=2, flatten, project with
                  nn.Linear. Verify shapes at every step: [4, 32, 32] → [256,
                  16] → [256, d_model]. Implement unpatchify as the reverse.
                  Predict: how many patches will a [4, 64, 64] latent with p=4
                  produce?
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): adaLN-Zero Forward Pass.
                  </strong>{' '}
                  Implement one adaLN-Zero conditioning step: take a conditioning
                  vector, produce (γ, β, α) via MLP. Apply to a LayerNorm
                  output: γ · LayerNorm(x) + β. Apply the gate: x + α ·
                  attention_output. Verify: at α=0, the block output equals the
                  input. Gradually increase α and observe the block&rsquo;s
                  contribution grow.
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): DiT Architecture Inspection.
                  </strong>{' '}
                  Load a pretrained DiT model from HuggingFace. Inspect the
                  model: count parameters, list layer types, verify no
                  convolutions in the transformer blocks. Compare to U-Net:
                  print both models&rsquo; layer summaries side-by-side.
                  Identify the adaLN-Zero components. Vary the model size
                  (DiT-S, DiT-B, DiT-L, DiT-XL) and observe parameter count
                  scaling.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): DiT Generation and Scaling
                    Comparison.
                  </strong>{' '}
                  Use a pretrained DiT-XL/2 to generate class-conditional
                  ImageNet images (e.g., class 207 = golden retriever). Compare
                  generation quality across DiT model sizes if VRAM allows.
                  Vary the number of sampling steps and classifier-free guidance
                  scale. Observe: larger DiT = better image quality.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: patchify/unpatchify (shapes)</li>
              <li>Guided: adaLN-Zero (conditioning)</li>
              <li>Supported: architecture inspection (compare)</li>
              <li>Independent: generation + scaling (observe)</li>
            </ol>
            <p className="text-sm mt-2">
              DiT-XL/2 requires ~3 GB VRAM in float16. Smaller variants (DiT-S)
              require even less—Colab&rsquo;s free GPU is sufficient.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 16: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Tokenize the image.',
                description:
                  'The patchify operation splits the latent into patches and projects them to d_model dimensions—the image equivalent of text tokenization. The transformer processes patch tokens the same way it processes word tokens. Patch size controls the resolution-compute tradeoff.',
              },
              {
                headline: 'The transformer block you already know.',
                description:
                  'DiT blocks are standard transformer blocks: self-attention + FFN + residual + normalization. No convolutions, no encoder-decoder hierarchy, no U-Net-style skip connections. Every component except the conditioning mechanism is from Series 4.',
              },
              {
                headline: 'adaLN-Zero: condition via the normalization.',
                description:
                  'Conditioning-dependent scale (γ), shift (β), and gate (α) on LayerNorm—extending adaptive group norm with a zero-initialized gate. Each block starts as an identity function and learns what to contribute. Same zero-initialization safety pattern as ControlNet and LoRA.',
              },
              {
                headline: 'Two knobs, not twenty.',
                description:
                  'Scaling DiT means increasing d_model and N—the same recipe that scaled GPT-2 to GPT-3. No ad hoc architectural decisions. Predictable loss improvement with scale. This directly addresses SDXL\'s "no scaling recipe" limitation.',
              },
              {
                headline: 'Same pipeline, different denoising network.',
                description:
                  'DiT replaces only the U-Net. The VAE, text encoder, noise schedule, and sampling algorithm are unchanged. The latent diffusion pipeline you learned in Series 6 still applies—with a transformer in the middle instead of a U-Net.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Scalable Diffusion Models with Transformers',
                authors: 'Peebles & Xie, 2023',
                url: 'https://arxiv.org/abs/2212.09748',
                note: 'The original DiT paper. Section 3 covers the architecture (patchify, adaLN-Zero, model variants). Section 4 covers the scaling experiments. Table 1 compares conditioning approaches.',
              },
              {
                title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
                authors: 'Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit & Houlsby, 2021',
                url: 'https://arxiv.org/abs/2010.11929',
                note: 'The ViT paper. Section 3 covers the patchify operation and positional embeddings that DiT adapts for latent diffusion.',
              },
              {
                title: 'Attention Is All You Need',
                authors: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017',
                url: 'https://arxiv.org/abs/1706.03762',
                note: 'The transformer paper. DiT uses the same self-attention + FFN + residual architecture. Included for completeness—you studied this in Series 4.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 17: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: SD3 & Flux"
            description="DiT showed that transformers can replace U-Nets for diffusion, with better scaling. But the original DiT is class-conditional—it uses class labels, not text prompts. The next lesson introduces SD3 and Flux: T5-XXL text encoder, MMDiT where text and image tokens attend to each other jointly, and flow matching as the training objective. Every concept from this course converges."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
