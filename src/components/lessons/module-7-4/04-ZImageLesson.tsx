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
  ModuleCompleteBlock,
  ReferencesBlock,
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-4-4-z-image.ipynb'

/**
 * Z-Image & Z-Image Turbo
 *
 * Lesson 4 in Module 7.4 (Next-Generation Architectures). Twelfth and
 * final lesson in Series 7 (Post-SD Advances).
 * Cognitive load: BUILD (2-3 new concepts that are clever combinations
 * of familiar building blocks: single-stream S3-DiT unification,
 * Decoupled-DMD distillation, DMDR reinforcement learning post-training.
 * The student already has all constituent pieces.)
 *
 * Z-Image takes the same building blocks the student knows--transformers,
 * patchify, flow matching, joint attention, adaLN--and recombines them
 * more efficiently. S3-DiT simplifies MMDiT's dual-stream into a fully
 * single-stream design. Decoupled-DMD decomposes distillation into
 * "spear" (quality) and "shield" (stability). DMDR combines distillation
 * with RL to break the teacher ceiling.
 *
 * Builds on: sd3-and-flux (7.4.3), diffusion-transformers (7.4.2),
 * flow-matching (7.2.2), consistency-models (7.3.1),
 * latent-consistency-and-turbo (7.3.2), clip (6.3.3),
 * positional-encoding (4.2.3), Series 5 (RLHF/DPO/GRPO)
 *
 * Previous: SD3 & Flux (Module 7.4, Lesson 3 / BUILD)
 * Next: (Series and Module complete)
 */

export function ZImageLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Z-Image & Z-Image Turbo"
            description="What comes after the convergence—Z-Image&rsquo;s single-stream S3-DiT simplifies MMDiT&rsquo;s dual-stream design, an LLM replaces triple text encoders, and Decoupled-DMD plus RL post-training breaks the teacher ceiling. 6.15B parameters competing with 32B Flux."
            category="Next-Generation Architectures"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain how Z-Image&rsquo;s S3-DiT architecture simplifies
            MMDiT&rsquo;s dual-stream design into a fully single-stream
            transformer with shared projections, and how its Decoupled-DMD and
            DMDR post-training pipeline breaks the teacher ceiling through the
            synergy of distillation and reinforcement learning.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            Two to three new concepts, all of which extend or recombine things
            you already know: S3-DiT single-stream (simplifies MMDiT),
            Decoupled-DMD (a new distillation approach), and DMDR (DPO + GRPO
            from Series 5 applied to image generation).
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'S3-DiT single-stream architecture: why single-stream vs dual-stream, refiner layers, shared projections, SwiGLU FFN',
              'Qwen3-4B as text encoder: the text encoder evolution trajectory from CLIP to LLM',
              '3D Unified RoPE for multi-modal position encoding',
              'Decoupled-DMD: DMD background, spear/shield decomposition, separate noise schedules',
              'DMDR: combining DMD with DPO and GRPO to break the teacher ceiling',
              'NOT: implementing S3-DiT from scratch (architectural understanding, not coding)',
              'NOT: full training procedure details (dataset construction, hyperparameters)',
              'NOT: the Prompt Enhancer VLM\u2019s architecture or training in detail',
              'NOT: every ablation from the paper (only key design choices)',
              'NOT: comparing to every other architecture (SD3/Flux as primary comparison)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Concepts to Reactivate"
            subtitle="Quick reactivation before we read a new paper together"
          />
          <div className="space-y-4">
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>MMDiT joint attention</strong> (from{' '}
                <strong>SD3 & Flux</strong>): Text and image tokens are
                concatenated and attend to each other through standard
                self-attention. But each modality maintains separate Q/K/V
                projections and separate FFNs&mdash;&ldquo;shared listening,
                separate thinking.&rdquo; This dual-stream design doubles the
                projection and FFN parameters at every layer.
              </li>
              <li>
                <strong>Distillation approaches</strong> (from{' '}
                <strong>Consistency Models</strong> and{' '}
                <strong>Latent Consistency & Turbo</strong>): You have seen two
                distillation approaches: consistency distillation (map any
                trajectory point to the endpoint) and adversarial distillation
                (use a discriminator to ensure realism). Both produce students
                bounded by teacher quality. Z-Image introduces a third approach
                that breaks this bound.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Z-Image asks two questions: Can we make the architecture simpler?
              Can we make the student better than the teacher? The answers are
              S3-DiT and DMDR.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Open Question">
            The previous lesson ended with SD3/Flux as the convergence
            architecture and the claim &ldquo;you can read frontier diffusion
            papers and understand the design choices.&rdquo; This lesson tests
            that claim.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The 1/5 Question"
            subtitle="How few parameters do you actually need?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is a question: Flux.1 Dev has 32 billion parameters and is
              one of the top open-source image generators. How many parameters
              would you need to match or beat it?
            </p>
            <p className="text-muted-foreground">
              Z-Image does it with 6.15 billion&mdash;roughly 1/5 the
              parameters. And Z-Image Turbo generates images in 8 steps,
              sub-second on an H800, fitting in under 16GB of VRAM.
            </p>
            <p className="text-muted-foreground">
              The question is not &ldquo;what new architecture makes this
              possible?&rdquo; The question is: &ldquo;which of the building
              blocks you already know were unnecessary, and which combinations
              were missing?&rdquo; Let us trace the design choices.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Revolution, Refinement">
            Z-Image published in November 2025 by Alibaba&rsquo;s Tongyi Lab.
            It uses the same building blocks you know&mdash;transformers,
            patchify, flow matching, joint attention, adaLN&mdash;recombined
            more efficiently.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: S3-DiT Architecture — Part A: The dual-stream overhead */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="S3-DiT: The Single-Stream Architecture"
            subtitle="Where the parameters were hiding"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              MMDiT uses modality-specific projections and FFNs at every layer.
              For <InlineMath math="d_{model} = 3840" /> and 30 layers, this
              means:
            </p>

            <CodeBlock
              code={`Per MMDiT block (dual-stream):
  Text W_Q, W_K, W_V:  3 × d_model² = 3 × 3840² = ~44.2M params
  Image W_Q, W_K, W_V: 3 × d_model² = ~44.2M params
  Text FFN (SwiGLU):   ~8/3 × d_model² = ~39.3M params
  Image FFN (SwiGLU):  ~39.3M params
  Total modality-specific: ~167M params per block

Per S3-DiT block (single-stream):
  Shared W_Q, W_K, W_V: 3 × d_model² = ~44.2M params
  Shared FFN (SwiGLU):  ~39.3M params
  Total: ~83.5M params per block

Savings: ~50% of projection + FFN parameters per block
Over 30 layers: ~2.5B fewer parameters`}
              language="text"
              filename="parameter_comparison.txt"
            />

            <p className="text-muted-foreground">
              This is where Z-Image&rsquo;s parameter efficiency comes from.
              Not from a novel mechanism, but from eliminating duplication.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Simplicity, Not Novelty">
            Z-Image&rsquo;s competitive performance with 1/5 the parameters of
            Flux comes from being <strong>simpler</strong> than MMDiT, not more
            complex. Single-stream eliminates the parameter overhead of
            separate per-modality projections and FFNs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Part B: Why did MMDiT use separate projections? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="But Wait—Why Did MMDiT Use Separate Projections?"
            subtitle="The refiner layer solution"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Recall the negative example from <strong>SD3 & Flux</strong>:
              naive concatenation fails because text embeddings and image patch
              embeddings live in different representation spaces. Forcing them
              through the same projections is like asking a French speaker and a
              Japanese speaker to use the same dictionary.
            </p>
            <p className="text-muted-foreground">
              Z-Image&rsquo;s solution: 2 lightweight refiner layers per
              modality. These are small transformer layers (much smaller than
              the main 30-layer backbone) that pre-process each
              modality&rsquo;s raw embeddings into a shared representation
              space. After refinement, the text and image representations are
              compatible enough for shared projections.
            </p>

            <ComparisonRow
              left={{
                title: 'MMDiT (Dual-Stream)',
                color: 'amber',
                items: [
                  'Modality-specific processing at every layer',
                  'Separate Q/K/V projections per modality',
                  'Separate FFN per modality',
                  '~2× projection + FFN parameters per block',
                  '"Shared listening, separate thinking"',
                ],
              }}
              right={{
                title: 'S3-DiT (Single-Stream)',
                color: 'emerald',
                items: [
                  'Modality-specific processing in 2 refiner layers only',
                  'Shared Q/K/V projections for all token types',
                  'Shared SwiGLU FFN for all token types',
                  '~1× projection + FFN parameters per block',
                  '"Translate once, then speak together"',
                ],
              }}
            />

            <p className="text-muted-foreground">
              &ldquo;Translate once, then speak together.&rdquo; MMDiT
              translates at every exchange. S3-DiT translates once at the
              beginning, then everyone speaks the same language.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Misconception: No Modality Awareness">
            &ldquo;Single-stream means the model treats text and image
            identically.&rdquo; No. The modality awareness is not
            removed&mdash;it is concentrated upfront. The refiner layers serve
            the same function as MMDiT&rsquo;s separate projections: mapping
            each modality into a space where shared attention is meaningful.
            The difference is architectural allocation: 2 dedicated layers vs
            duplication across 30 layers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Part C: Token types */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Unified Token Sequence"
            subtitle="Text and image tokens in one stream"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For text-to-image generation, the unified sequence consists of
              two token types:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Text tokens</strong> from Qwen3-4B (the LLM text
                encoder)
              </li>
              <li>
                <strong>Image VAE tokens</strong> from patchify (same as
                DiT/MMDiT)
              </li>
            </ol>
            <p className="text-muted-foreground">
              Both are concatenated at the sequence level into one unified
              input. Every self-attention layer performs dense cross-modal
              interaction across both token types. This is the &ldquo;one room,
              one conversation&rdquo; analogy from{' '}
              <strong>SD3 & Flux</strong>, taken to its extreme: not just
              shared attention but shared everything.
            </p>
            <p className="text-muted-foreground text-sm italic">
              Z-Image&rsquo;s editing variants (Z-Image-Omni, Z-Image-Edit)
              add a third token type&mdash;visual semantic tokens from SigLIP 2,
              an image encoder that provides abstract visual concept
              embeddings as conditioning. These are primarily relevant for
              image editing, not text-to-image generation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Scope: Token Types">
            For text-to-image generation, the unified sequence has two token
            types: text and image. Z-Image&rsquo;s editing variants add visual
            semantic tokens from an image encoder, but those are outside this
            lesson&rsquo;s scope.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Part D: Block internals */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Block Internals"
            subtitle="Familiar components with modern refinements"
          />
          <div className="space-y-4">
            <CodeBlock
              code={`# Pseudocode for one S3-DiT block
# (see transformer.py: ZImageTransformerBlock)

class S3DiTBlock:
    # Normalization: RMSNorm + Sandwich-Norm pattern
    # Attention: standard multi-head self-attention,
    #   shared across all token types
    # FFN: SwiGLU (from LLaMA) -- gated linear unit
    #   with Swish activation
    # Conditioning: adaLN with shared low-rank
    #   down-projection + layer-specific up-projections
    # Position: 3D Unified RoPE
    #   (temporal for text, spatial for image)`}
              language="python"
              filename="s3dit_block_pseudocode.py"
            />

            <p className="text-muted-foreground">
              Every component here has a precedent:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>SwiGLU FFN:</strong> from LLaMA / modern LLM
                architectures (Series 5)
              </li>
              <li>
                <strong>RMSNorm:</strong> from LLaMA, a simplification of
                LayerNorm that removes the mean-centering
              </li>
              <li>
                <strong>QK-Norm:</strong> normalizes queries and keys before
                the dot product, stabilizing attention at scale
              </li>
              <li>
                <strong>Sandwich-Norm:</strong> applies normalization both
                before and after attention, additional stability
              </li>
            </ul>
            <p className="text-muted-foreground">
              The adaLN conditioning uses a parameter-efficient trick: instead
              of a full MLP per layer mapping the timestep embedding to adaLN
              parameters, Z-Image uses a shared low-rank down-projection
              (compressing the conditioning vector) followed by layer-specific
              up-projections. This is the LoRA pattern applied to timestep
              conditioning&mdash;factor the per-layer conditioning into a shared
              low-rank component and a per-layer residual.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The LoRA Pattern Again">
            Shared low-rank down-projection + layer-specific up-projections is
            the same factorization pattern as LoRA: reduce a large per-layer
            matrix into a shared low-rank component. Z-Image applies it to
            timestep conditioning instead of weight adaptation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="S3-DiT architecture predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Z-Image uses one set of Q/K/V projections shared across text
                  and image tokens. The{' '}
                  <strong>SD3 & Flux</strong> lesson showed why shared
                  projections fail for raw text and image embeddings. What makes
                  shared projections work in Z-Image but not in naive
                  concatenation?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The refiner layers. They pre-process each modality into a
                    shared representation space before the main transformer. By
                    the time tokens reach the shared projections, they have been
                    &ldquo;translated&rdquo; into compatible representations.
                    Naive concatenation feeds raw, incompatible embeddings
                    directly into shared projections.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Z-Image has 6.15B total parameters with 30 layers of{' '}
                  <InlineMath math="d_{model} = 3840" />. Flux.1 Dev has ~32B
                  parameters. If the main source of Z-Image&rsquo;s parameter
                  savings is single-stream (eliminating per-modality
                  duplication), estimate how many parameters Flux&rsquo;s
                  dual-stream overhead adds.
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Rough estimate: if dual-stream roughly doubles projection +
                    FFN parameters per block, and these account for the majority
                    of block parameters, dual-stream could add 50-100% overhead
                    on the block parameters. For a 6B single-stream model, the
                    equivalent dual-stream would be ~9-12B in blocks alone. The
                    remaining ~20B in Flux includes larger dimensions, more
                    heads, and different design choices. Single-stream is a
                    significant but not sole source of the difference.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Text Encoding — LLM as Encoder */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Text Encoding: LLM as Encoder"
            subtitle="The text encoder evolution completes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Trace the progression:
            </p>

            <CodeBlock
              code={`SD v1.5:   CLIP ViT-L (123M)     -- one encoder, trained on image-text pairs
SDXL:      CLIP ViT-L + OpenCLIP ViT-bigG (123M + 354M)
                                   -- two encoders, both image-text
SD3/Flux:  CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL (123M + 354M + 4.7B)
                                   -- three encoders, adding a language model
Z-Image:   Qwen3-4B (4B)          -- ONE encoder, a chat-capable LLM`}
              language="text"
              filename="text_encoder_evolution.txt"
            />

            <p className="text-muted-foreground">
              The trajectory: from vision-language models (CLIP) to language
              models (T5) to full LLMs (Qwen3). And from multiple specialized
              encoders back to one powerful encoder. Simpler pipeline, richer
              embeddings, and the logical endpoint of the trajectory traced
              across Series 6&ndash;7.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Convergence to Simplicity">
            SD v1.5: 1 encoder. SDXL: 2 encoders. SD3: 3 encoders. Z-Image:
            back to 1. The trajectory is not &ldquo;more encoders = better&rdquo;
            but &ldquo;one powerful enough encoder replaces all.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Why Qwen3-4B? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Qwen3-4B?"
            subtitle="An LLM provides stronger instruction understanding"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              An LLM trained on diverse language tasks provides stronger
              instruction understanding than CLIP or T5:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Better compositional reasoning (spatial relationships, counting,
                negation)
              </li>
              <li>
                Bilingual support (English and Chinese natively)
              </li>
              <li>
                Richer contextual embeddings from large-scale language
                pretraining
              </li>
              <li>
                Instruction-following capabilities baked into the embeddings
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Misconception: LLM = Chatbot">
            &ldquo;Using an LLM as text encoder means the model understands
            language like a chatbot.&rdquo; No. Qwen3-4B provides embeddings,
            not chat responses. It processes the prompt once to produce token
            embeddings that enter the transformer. The richer embeddings come
            from the LLM&rsquo;s training on diverse language tasks, not from
            inference-time reasoning.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Prompt Enhancer */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Prompt Enhancer (PE)"
            subtitle="Baked-in reasoning, zero inference cost"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Z-Image adds one more innovation in text processing: a Prompt
              Enhancer (PE) that expands short user prompts with descriptive
              detail. For example, a prompt &ldquo;West Lake at sunset&rdquo;
              might be enhanced to &ldquo;the famous West Lake in Hangzhou,
              China, with its iconic Broken Bridge and Su Causeway, warm golden
              light reflecting off the water.&rdquo; The PE compensates for the
              6B model&rsquo;s limited world knowledge by providing this
              context during training.
            </p>
            <p className="text-muted-foreground">
              The key engineering insight: the PE is integrated during
              supervised fine-tuning (SFT), so the model learns to generate
              as if given enhanced prompts. There is <strong>no extra cost at
              inference time</strong>. The PE teaches the model during training;
              the model generates independently during inference.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Training-Time Only">
            The Prompt Enhancer improves generation quality without adding
            inference latency. Its reasoning is distilled into the model
            during fine-tuning, not executed at generation time.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: 3D Unified RoPE */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="3D Unified RoPE"
            subtitle="Position encoding for multi-modal sequences"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DiT used learned positional embeddings for patch positions.
              Z-Image uses RoPE&mdash;the same rotary position encoding you
              learned in Series 4&mdash;extended to three dimensions:
            </p>

            <CodeBlock
              code={`3D RoPE dimension allocation:
  Temporal axis (d_t = 32 dim pairs):
    encodes sequential position for text tokens
  Height axis (d_h = 48 dim pairs):
    encodes vertical patch position for image tokens
  Width axis (d_w = 48 dim pairs):
    encodes horizontal patch position for image tokens

Text token at position 5:
  temporal = 5, height = 0, width = 0
  RoPE rotation applied to d_t pairs,
  d_h and d_w pairs get zero rotation

Image patch at row 3, column 7:
  temporal = 0, height = 3, width = 7
  RoPE rotation applied to d_h and d_w pairs,
  d_t pairs get zero rotation`}
              language="text"
              filename="3d_rope.txt"
            />

            <p className="text-muted-foreground">
              This is elegant: text tokens use the temporal axis (they have
              sequential order but no spatial position). Image tokens use the
              spatial axes (they have grid position but no sequential order).
              The axes are orthogonal, so text-to-text attention depends on
              sequential distance, image-to-image attention depends on spatial
              distance, and cross-modal attention has no positional
              bias&mdash;the non-applicable axes contribute zero rotation.
            </p>
            <p className="text-muted-foreground">
              Compare to DiT&rsquo;s learned positional embeddings: RoPE
              generalizes to arbitrary resolutions without retraining (the
              rotation frequencies are continuous), while learned embeddings
              are fixed to the training resolution.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Principle, More Axes">
            RoPE encodes position by rotating pairs of dimensions. 1D RoPE
            rotates by sequence position. Z-Image&rsquo;s 3D RoPE rotates by
            three independent axes: temporal, height, width. Same principle
            from Series 4, applied to a multi-modal sequence.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Text encoding and position predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In Z-Image&rsquo;s 3D RoPE, cross-modal attention (text
                  attending to image or image attending to text) has zero
                  rotation on the non-applicable axes. What does this mean for
                  how the model treats cross-modal relationships?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Cross-modal attention has no positional bias&mdash;a text
                    token attends to all image patches equally regardless of
                    their spatial position, and an image patch attends to all
                    text tokens equally regardless of their sequential position.
                    The model must learn cross-modal relationships entirely from
                    content, not position. This makes sense: there is no
                    inherent spatial correspondence between &ldquo;the word
                    &lsquo;cat&rsquo; at position 3&rdquo; and &ldquo;the patch
                    at row 5, column 7.&rdquo; The content determines the
                    relationship.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Z-Image replaces three text encoders (CLIP ViT-L + OpenCLIP
                  ViT-bigG + T5-XXL, total ~5.2B parameters) with one
                  (Qwen3-4B, ~4B parameters). Besides parameter savings, what
                  is the practical benefit?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Simpler pipeline. SD3 requires loading and running three
                    separate text encoders, each with their own tokenizer,
                    forward pass, and output projection. Z-Image runs one
                    encoder with one tokenizer. This reduces implementation
                    complexity, memory fragmentation, and inference latency. It
                    also eliminates the need to design how to combine embeddings
                    from three different sources.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Decoupled-DMD — Part A: DMD background */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Decoupled-DMD: Rethinking Distillation"
            subtitle="A third distillation paradigm"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know two distillation approaches from Module 7.3: consistency
              distillation (learn to map any trajectory point directly to the
              endpoint) and adversarial distillation (use a discriminator to
              ensure the student&rsquo;s outputs look realistic). DMD&mdash;
              Distribution Matching Distillation&mdash;is a third approach.
            </p>
            <p className="text-muted-foreground">
              DMD uses two training signals:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>CFG-augmented regression</strong> (the
                &ldquo;spear&rdquo;): Run the teacher model with
                classifier-free guidance to produce high-quality outputs. Train
                the student to match these outputs via regression loss. CFG
                amplifies the teacher&rsquo;s quality, pushing the student
                toward better generations.
              </li>
              <li>
                <strong>Distribution matching</strong> (the
                &ldquo;shield&rdquo;): A regularization loss ensuring the
                student&rsquo;s output distribution matches the real data
                distribution. Without this, the student might collapse to
                generating &ldquo;average&rdquo; images or mode-specific
                artifacts.
              </li>
            </ol>
            <p className="text-muted-foreground">
              The metaphor is Z-Image&rsquo;s own: CFG augmentation is the{' '}
              <strong>spear</strong> that pushes quality. Distribution matching
              is the <strong>shield</strong> that prevents degenerate solutions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Distillation Taxonomy">
            <ul className="space-y-1 text-sm">
              <li>
                <strong>Consistency:</strong> map trajectory point to endpoint
              </li>
              <li>
                <strong>Adversarial:</strong> discriminator ensures realism
              </li>
              <li>
                <strong>DMD:</strong> CFG regression + distribution matching
              </li>
            </ul>
            <p className="text-sm mt-2">
              Three paradigms, each with different tradeoffs. DMD is the
              foundation for Z-Image Turbo.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Part B: The decoupling insight */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Decoupling Insight"
            subtitle="Different components need different operating conditions"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The Z-Image team discovered that DMD&rsquo;s success comes
              primarily from the spear (CFG augmentation), while distribution
              matching is &ldquo;just&rdquo; a regularizer. More importantly,
              these two components operate best at different noise levels:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                The <strong>spear</strong> needs high noise levels: large-scale
                composition and structure guidance
              </li>
              <li>
                The <strong>shield</strong> needs low noise levels: fine-detail
                distribution matching
              </li>
            </ul>
            <p className="text-muted-foreground">
              In standard DMD, both losses are coupled at the same noise level,
              forcing a compromise. Decoupled-DMD separates them, applying
              appropriate noise schedules to each:
            </p>

            <CodeBlock
              code={`# Standard DMD (coupled):
t = sample_timestep()                        # same t for both losses
loss_spear = regression_loss(student, teacher_cfg, x, t)
loss_shield = distribution_matching_loss(student, real_data, t)
loss = loss_spear + lambda * loss_shield

# Decoupled-DMD:
t_spear = sample_timestep(bias='high_noise')  # large-scale composition
t_shield = sample_timestep(bias='low_noise')  # fine-detail preservation
loss_spear = regression_loss(student, teacher_cfg, x, t_spear)
loss_shield = distribution_matching_loss(student, real_data, t_shield)
loss = loss_spear + lambda * loss_shield`}
              language="python"
              filename="decoupled_dmd_pseudocode.py"
            />

            <p className="text-muted-foreground">
              The result: cleaner generations without the artifacts that coupled
              DMD produces. Each loss operates at the noise level where it is
              most informative.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Misconception: Minor Tweak">
            &ldquo;Decoupled-DMD is a minor tweak to DMD.&rdquo; No. The
            decoupling is a conceptual advance: understand which component of
            your method drives quality vs which provides stability, then
            optimize each independently. This &ldquo;understand WHY before you
            optimize&rdquo; pattern appears throughout ML research. The
            decoupling also enables DMDR (combining DMD with RL), which would
            not be possible with the coupled version.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Part C: Performance comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Z-Image Turbo Performance"
            subtitle="Where it fits in the speed landscape"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Z-Image Turbo (the Decoupled-DMD distilled variant) generates
              images in 8 steps, sub-second on an H800 GPU, and fits in under
              16GB of VRAM. Compare to Module 7.3&rsquo;s speed landscape:
            </p>

            <CodeBlock
              code={`Approach                  Steps   Quality tradeoff
DDPM (Series 6)            50+    Baseline
DDIM (Series 6)            20-50  Minor quality loss
LCM (Module 7.3)           4-8    Noticeable quality loss
SDXL Turbo (Module 7.3)    1-4    Good quality, limited diversity
Z-Image Turbo (this)       8      Competitive with many-step models`}
              language="text"
              filename="speed_landscape.txt"
            />

            <p className="text-muted-foreground">
              Z-Image Turbo at 8 steps is competitive with models that require
              50+ steps. The distillation preserves quality because the
              spear/shield decomposition lets each component operate at its
              optimal noise level.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Distillation predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In DMD, the &ldquo;spear&rdquo; (CFG-augmented regression)
                  pushes the student toward high-quality teacher outputs. What
                  would happen if you only used the spear without the shield
                  (distribution matching)?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The student would learn to reproduce the teacher&rsquo;s
                    CFG-augmented outputs, but without distribution matching, it
                    could collapse to a degenerate mapping&mdash;e.g., always
                    producing &ldquo;average&rdquo; high-quality images that
                    look good but lack diversity. The shield ensures the
                    student&rsquo;s outputs cover the full data distribution,
                    not just the modes that CFG emphasizes. This is analogous to
                    mode collapse in GANs.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Why does decoupling the noise schedules for spear and shield
                  help? What goes wrong when they are coupled?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    At high noise levels, the image is mostly noise&mdash;
                    distribution matching of fine details is meaningless because
                    there are no fine details to match. At low noise levels, the
                    image is nearly clean&mdash;large-scale composition guidance
                    from CFG is unnecessary because composition is already
                    determined. Coupling forces both losses to operate at the
                    same noise level, where one or the other is irrelevant but
                    still contributing gradient signal. This irrelevant gradient
                    creates artifacts. Decoupling lets each loss operate where
                    it is most informative.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 12: DMDR — Part A: The teacher ceiling problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="DMDR: Breaking the Teacher Ceiling"
            subtitle="Distillation + reinforcement learning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In every distillation approach from Module 7.3, the student model
              is bounded by the teacher&rsquo;s quality. The student learns to
              approximate the teacher in fewer steps&mdash;a compression. It
              cannot exceed what the teacher can produce.
            </p>
            <p className="text-muted-foreground">
              DMDR (DMD + Reinforcement Learning) breaks this bound. The
              insight: use DMD as the <strong>regularizer</strong> and RL as
              the <strong>quality driver</strong>. The teacher&rsquo;s role
              shifts from &ldquo;ceiling&rdquo; to &ldquo;guardrail.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Misconception: Cannot Beat Teacher">
            &ldquo;Distillation cannot produce models better than the
            teacher.&rdquo; Distillation alone cannot. But DMDR combines
            distillation with RL. The RL provides an external reward signal
            that is not limited by the teacher&rsquo;s quality. The
            teacher&rsquo;s role shifts from ceiling to guardrail&mdash;
            keeping the student from wandering too far while RL pushes quality
            beyond what the teacher could achieve.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Part B: Two-stage RL post-training */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two-Stage RL Post-Training"
            subtitle="DPO for precision, GRPO for aesthetics"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DMDR applies two stages of RL, using techniques you know from
              Series 5:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Stage 1: DPO" color="sky">
                <div className="space-y-2 text-sm">
                  <p>
                    Trains on preference pairs for specific capabilities: text
                    rendering accuracy, object counting, instruction following.
                  </p>
                  <p className="text-muted-foreground">
                    This is the same DPO from Series 5, applied to image
                    generation instead of text generation. The preference
                    signal: &ldquo;this image renders the text
                    correctly&rdquo; vs &ldquo;this image does not.&rdquo;
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Stage 2: GRPO" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    Group Relative Policy Optimization for subjective
                    qualities: photorealism, aesthetics, visual appeal.
                  </p>
                  <p className="text-muted-foreground">
                    This is the same GRPO from Series 5. The reward signal
                    comes from aesthetic scoring models instead of human
                    annotators.
                  </p>
                </div>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              The DMD component is critical in both stages: it prevents reward
              hacking. Without DMD&rsquo;s regularization, the RL optimization
              would exploit the reward model (e.g., generating images that
              score high on &ldquo;aesthetics&rdquo; but look nothing like
              real photographs). DMD keeps the outputs grounded in the real
              data distribution&mdash;the &ldquo;shield&rdquo; protecting
              against reward hacking.
            </p>
            <p className="text-muted-foreground">
              This is the same reward hacking problem you studied in Series 5,
              now appearing in image generation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Techniques, New Domain">
            DPO and GRPO are the same algorithms from Series 5. The only
            difference: they are applied to image generation instead of text
            generation. DMD plays the role of the KL divergence penalty from
            RLHF&mdash;keeping the model close to a reference distribution
            while RL pushes toward higher reward.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 13: Check #4 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="DMDR predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  DMDR uses DPO for text rendering and GRPO for aesthetics.
                  Why not use the same RL method for both?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    DPO works best with binary preference pairs that have clear
                    right/wrong answers&mdash;&ldquo;this image renders
                    &lsquo;HELLO&rsquo; correctly&rdquo; vs &ldquo;this image
                    does not.&rdquo; Text rendering and counting have
                    objectively verifiable correctness. GRPO works best with
                    scalar reward scores for subjective qualities&mdash;
                    aesthetics, photorealism. There is no single
                    &ldquo;correct&rdquo; answer for aesthetics, but a spectrum
                    of quality. Different reward structures match different RL
                    methods.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  You have seen reward hacking in LLMs (Series 5): the model
                  learns to exploit the reward model rather than genuinely
                  improving. How does DMD prevent this in DMDR?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    DMD provides a distribution matching loss that acts as an
                    anchor to the real data distribution. Even if the RL reward
                    signal pushes toward degenerate but high-scoring outputs,
                    the DMD loss pulls back toward realistic generations. The
                    combined gradient: RL pushes toward higher reward, DMD
                    prevents straying too far from reality. The balance prevents
                    reward hacking without limiting the student to teacher
                    quality. The teacher is the guardrail, not the ceiling.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 14: Performance and Positioning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Performance and Positioning"
            subtitle="The numbers behind the design"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Z-Image at 6.15B parameters achieves:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Ranked #1 open-source on Artificial Analysis leaderboard
              </li>
              <li>
                87.4% &ldquo;Good+Same&rdquo; rate against Flux Dev (32B
                params) in human evaluation&mdash;with 1/5 the parameters
              </li>
              <li>
                Z-Image Turbo: 8 steps, sub-second on H800, &lt;16GB VRAM
              </li>
            </ul>
            <p className="text-muted-foreground">
              The $630K total training cost (314K H800 GPU hours) is significant
              but modest by frontier model standards.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Simpler Beats Bigger">
            Z-Image&rsquo;s competitive performance with 1/5 the parameters
            comes from being <strong>simpler</strong>, not more complex.
            Architecture simplification (single-stream) plus training
            innovation (three-phase curriculum) plus post-training
            (Decoupled-DMD + DMDR) beats architecture complexity.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Training curriculum overview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Training Curriculum"
            subtitle="Three-phase progressive resolution"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Z-Image&rsquo;s training follows a three-phase curriculum:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Phase 1: Low-Res Pre-training" color="sky">
                <p className="text-sm">
                  256×256 resolution, 147.5K H800 hours. Learn basic image
                  generation at low compute cost.
                </p>
              </GradientCard>
              <GradientCard title="Phase 2: Omni Pre-training" color="violet">
                <p className="text-sm">
                  Arbitrary resolution, 142.5K H800 hours. Scale to multiple
                  resolutions with joint tasks.
                </p>
              </GradientCard>
              <GradientCard title="Phase 3: PE-aware SFT" color="emerald">
                <p className="text-sm">
                  High-quality curated data, 24K H800 hours. Fine-tune with
                  Prompt Enhancer integration.
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              The progressive resolution training mirrors SDXL&rsquo;s approach
              of training at different resolutions, but formalized into a
              curriculum.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 15: References and Further Reading */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title:
                  'An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer',
                authors: 'Tongyi Lab (Alibaba), 2025',
                url: 'https://arxiv.org/abs/2511.22699',
                note: 'The Z-Image paper. Sections 3-4 cover the S3-DiT architecture, text encoding, and 3D RoPE. Section 5 covers the training curriculum.',
              },
              {
                title:
                  'Decoupled-DMD: Distribution Matching Distillation for Fewer-Step Generation',
                authors: 'Tongyi Lab (Alibaba), 2025',
                url: 'https://arxiv.org/abs/2511.22677',
                note: 'The Decoupled-DMD paper. The spear/shield decomposition and the argument for separate noise schedules.',
              },
              {
                title:
                  'DMDR: Distribution Matching Distillation with Reinforcement Learning',
                authors: 'Tongyi Lab (Alibaba), 2025',
                url: 'https://arxiv.org/abs/2511.13649',
                note: 'The DMDR paper. How DPO and GRPO combine with DMD to break the teacher ceiling.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Source code references */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Source Code References"
            subtitle="Your knowledge, implemented"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Z-Image is open source. Each file maps to concepts from this
              course:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>
                  <code>src/zimage/transformer.py</code>
                </strong>{' '}
                &mdash; The S3-DiT implementation. Look for{' '}
                <code>ZImageTransformer2DModel</code> (top-level model),{' '}
                <code>ZImageTransformerBlock</code> (single-stream block with
                shared projections and SwiGLU FFN),{' '}
                <code>RopeEmbedder</code> (3D Unified RoPE),{' '}
                <code>FinalLayer</code>. This is where you can verify the
                single-stream design: one set of Q/K/V projections, one FFN,
                applied to the unified token sequence.
              </li>
              <li>
                <strong>
                  <code>src/zimage/pipeline.py</code>
                </strong>{' '}
                &mdash; The generation pipeline. Trace the flow: text encoding
                → patchify → denoising loop → unpatchify → VAE decode. The same
                pipeline structure you have traced in every architecture since
                Series 6.
              </li>
              <li>
                <strong>
                  <code>src/zimage/autoencoder.py</code>
                </strong>{' '}
                &mdash; The VAE. Note that Z-Image reuses the Flux
                VAE&mdash;a pragmatic choice leveraging proven reconstruction
                quality rather than training a new VAE.
              </li>
              <li>
                <strong>
                  <code>src/zimage/scheduler.py</code>
                </strong>{' '}
                &mdash; Noise scheduling for the flow matching training
                objective.
              </li>
              <li>
                <strong>
                  <code>src/config/model.py</code>
                </strong>{' '}
                &mdash; Full model hyperparameters. You can verify: 30 layers,
                3840 dimension, 30 heads.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Reading this source code is a capstone exercise in its own right.
              Every class name maps to a concept from this course: transformer
              blocks (Series 4), RoPE (from <strong>Positional Encoding</strong>
              ), timestep embedding (from{' '}
              <strong>Conditioning the U-Net</strong>), patchify (from{' '}
              <strong>Diffusion Transformers</strong>), flow matching (from{' '}
              <strong>Flow Matching</strong>). The code is your knowledge,
              implemented.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Capstone Reading">
            Reading Z-Image&rsquo;s source code is the ultimate test of your
            understanding. If you can trace each class to a concept you
            learned, you have achieved the course&rsquo;s goal: fluency in
            the building blocks of modern generative AI.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 16: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Z-Image Pipeline"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the lesson concrete&mdash;inspect the
                single-stream architecture, compare parameter counts, trace the
                pipeline, and explore the Turbo variant.
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
                  <strong>
                    Exercise 1 (Guided): Z-Image Pipeline Inspection.
                  </strong>{' '}
                  Load Z-Image via diffusers. Inspect the architecture: print
                  model class names, parameter counts. Verify the single-stream
                  design: one set of Q/K/V projections, shared FFN. Compare
                  parameter count to SD3/Flux.
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): Single-Stream vs Dual-Stream.
                  </strong>{' '}
                  Extract transformer block structure from Z-Image and compare
                  to MMDiT blocks. Count modality-specific parameters in each.
                  Verify the ~50% parameter savings from single-stream design.
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): Z-Image Generation and Turbo
                    Comparison.
                  </strong>{' '}
                  Generate images with Z-Image at varying step counts (8, 20,
                  30, 50). Generate with Z-Image Turbo at 8 steps. Compare
                  quality and generation time. Test with compositional prompts
                  that require strong text understanding.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): Architecture Capstone Trace.
                  </strong>{' '}
                  Generate an image with Z-Image, capturing intermediate
                  outputs. Trace the full pipeline: Qwen3-4B text encoding,
                  patchify, refiner layers, 30 single-stream blocks, unpatchify,
                  VAE decode. For each step, annotate which lesson covered the
                  relevant concept.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: pipeline inspection (architecture)</li>
              <li>Guided: parameter comparison (efficiency)</li>
              <li>Supported: generation with Turbo comparison (quality)</li>
              <li>Independent: full pipeline trace (capstone)</li>
            </ol>
            <p className="text-sm mt-2">
              Z-Image fits in &lt;16GB VRAM. Use Colab&rsquo;s T4 or A100 GPU.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 17: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: '"Translate once, then speak together."',
                description:
                  'S3-DiT concentrates modality-specific processing in lightweight refiner layers, then uses fully shared projections and FFN across all token types. This eliminates the ~50% parameter overhead of MMDiT\u2019s dual-stream design. Same attention mechanism, different allocation of where modality awareness lives.',
              },
              {
                headline: 'The text encoder trajectory.',
                description:
                  'CLIP \u2192 dual CLIP \u2192 CLIP+T5 \u2192 LLM. Z-Image replaces three specialized encoders with one powerful LLM (Qwen3-4B). Simpler pipeline, richer embeddings, and the logical endpoint of the trajectory traced across Series 6\u20137.',
              },
              {
                headline: '"Spear and shield."',
                description:
                  'Decoupled-DMD separates distillation into quality-driving CFG augmentation (spear) and distribution-preserving regularization (shield). Applying separate noise schedules to each eliminates artifacts from forced coupling. The lesson: understand WHY your method works, then optimize each component independently.',
              },
              {
                headline:
                  '"The teacher becomes the guardrail, not the ceiling."',
                description:
                  'DMDR combines distillation with RL (DPO + GRPO from Series 5). DMD regularizes RL (prevents reward hacking). RL guides DMD (provides quality signal beyond the teacher). The student model can exceed the teacher because RL provides external reward signal that is not limited by teacher quality.',
              },
              {
                headline: 'Simplicity beats complexity.',
                description:
                  'Z-Image matches 32B Flux with 6.15B parameters\u2014not through architectural novelty but through eliminating unnecessary duplication and investing in better training. The architecture is simpler than MMDiT. The innovations are in training strategy and post-training.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 18: Series and Module Conclusion */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Series 7 Complete"
            subtitle="From Stable Diffusion to the frontier and beyond"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You started Series 7 with Stable Diffusion v1.5 as your
              reference architecture: a U-Net trained with DDPM noise
              prediction, conditioned via a single CLIP encoder through
              cross-attention, generating 512×512 images in 50+ steps.
            </p>
            <p className="text-muted-foreground">
              Over twelve lessons, you traced the evolution:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Module 7.1:</strong> Added spatial and image
                conditioning without changing the model (ControlNet, IP-Adapter)
              </li>
              <li>
                <strong>Module 7.2:</strong> Reframed diffusion through the
                score function and discovered flow matching&mdash;straight
                paths instead of curved
              </li>
              <li>
                <strong>Module 7.3:</strong> Collapsed the multi-step process
                with consistency models and adversarial distillation
              </li>
              <li>
                <strong>Module 7.4:</strong> Replaced the architecture
                entirely&mdash;U-Net to DiT to MMDiT to S3-DiT&mdash;and
                broke the teacher ceiling with RL post-training
              </li>
            </ul>
            <p className="text-muted-foreground">
              Z-Image demonstrates the current direction of the field: simplify
              the architecture, invest in training and post-training. The next
              paper you read will use these same building blocks in yet another
              combination. You have the vocabulary, the mental models, and the
              technical depth to read it and understand why each choice was
              made.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Goal Achieved">
            You started overwhelmed by the pace of AI. Now you can read
            frontier research and trace every design choice to concepts you
            understand. The papers are not beyond you&mdash;they are
            combinations of ideas you have built.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Module + Series Completion */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="7.4"
            title="Next-Generation Architectures"
            achievements={[
              'SDXL as the U-Net ceiling: dual encoders, micro-conditioning, refiner',
              'DiT: patchify, adaLN-Zero, transformer scaling recipe',
              'MMDiT: joint text-image attention, modality-specific projections',
              'S3-DiT: single-stream architecture with refiner layers, shared projections',
              'LLM as text encoder: from CLIP to Qwen3-4B',
              'Decoupled-DMD: spear/shield decomposition for distillation',
              'DMDR: RL post-training that breaks the teacher ceiling',
              'The full SD3/Flux/Z-Image pipeline traced end-to-end',
            ]}
            nextModule="(Series Complete)"
            nextTitle="You have completed the Post-SD Advances series"
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Series Complete"
            description="You have traced the evolution of generative models from the foundations of deep learning through the current frontier and beyond. Every design choice in Z-Image traces back to a concept you built from scratch. The next paper you read will use these same building blocks in yet another combination. You have the vocabulary, the mental models, and the technical depth to read it."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
