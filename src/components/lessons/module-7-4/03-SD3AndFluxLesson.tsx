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
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-4-3-sd3-and-flux.ipynb'

/**
 * SD3 & Flux
 *
 * Lesson 3 in Module 7.4 (Next-Generation Architectures). Eleventh and
 * final lesson in Series 7 (Post-SD Advances).
 * Cognitive load: BUILD (2-3 new concepts that extend the DiT architecture:
 * MMDiT joint attention as a simplification of cross-attention, T5-XXL as
 * a stronger text encoder, rectified flow applied in practice).
 *
 * SD3 and Flux combine the DiT architecture with joint text-image attention
 * (MMDiT), stronger text encoding (T5-XXL alongside CLIP), and flow matching
 * to create the current frontier diffusion architecture. Every component
 * traces to a lesson the student has already completed.
 *
 * Builds on: diffusion-transformers (7.4.2), multi-head-attention (4.2.4),
 * text-conditioning-and-guidance (6.3.4), clip (6.3.3), sdxl (7.4.1),
 * flow-matching (7.2.2), ip-adapter (7.1.3)
 *
 * Previous: Diffusion Transformers (Module 7.4, Lesson 2 / STRETCH)
 * Next: (Series complete)
 */

export function SD3AndFluxLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="SD3 & Flux"
            description="The current frontier—combine DiT with joint text-image attention (MMDiT), T5-XXL text encoding, and flow matching. Every component traces to a lesson you have already completed."
            category="Next-Generation Architectures"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain how SD3 and Flux combine the DiT architecture with joint
            text-image attention (MMDiT), stronger text encoding (T5-XXL
            alongside CLIP), and flow matching to create the current frontier
            diffusion architecture, understanding why these three changes
            converge and how each addresses a specific limitation of prior
            approaches.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            Two to three new concepts, all of which extend or simplify things
            you already know: MMDiT joint attention (simpler than
            cross-attention), T5-XXL (a bigger text encoder), and flow matching
            applied in practice (same objective you trained with in{' '}
            <LessonLink slug="flow-matching">Flow Matching</LessonLink>).
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'MMDiT joint attention: concatenating text and image tokens, self-attention on the combined sequence, modality-specific projections',
              'T5-XXL as a text encoder: why a language model provides richer text understanding than CLIP, triple encoder setup',
              'Flow matching as SD3\u2019s training objective: connecting to your existing flow matching knowledge',
              'The full SD3/Flux pipeline traced end-to-end, annotated with lesson sources',
              'NOT: full SD3/Flux training procedures (dataset, hyperparameters, compute)',
              'NOT: every Flux variant (dev/schnell/pro/fill)—mentioned for vocabulary only',
              'NOT: implementing MMDiT from scratch (too much architecture code for one lesson)',
              'NOT: video extensions, multimodal extensions, or ControlNet/IP-Adapter for SD3/Flux',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — brief reactivation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Concepts to Reactivate"
            subtitle="Quick reactivation before they converge"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This lesson connects concepts from across the entire course. A
              brief reactivation of the three that matter most:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>DiT architecture</strong> (from{' '}
                <strong>Diffusion Transformers</strong>): Patchify the latent
                into tokens, process with standard transformer blocks
                (self-attention + FFN + residual + norm), condition via
                adaLN-Zero. Class-conditional on ImageNet. The architecture
                scales with two knobs: d_model and N.
              </li>
              <li>
                <strong>Flow matching</strong> (from{' '}
                <LessonLink slug="flow-matching">Flow Matching</LessonLink>): Straight-line interpolation{' '}
                <InlineMath math="x_t = (1-t) \cdot x_0 + t \cdot \epsilon" />.
                Velocity prediction{' '}
                <InlineMath math="v = \epsilon - x_0" />. Straight trajectories
                by construction, fewer ODE solver steps needed. You trained a
                flow matching model from scratch.
              </li>
              <li>
                <strong>Cross-attention for text conditioning</strong> (from{' '}
                <strong>Conditioning the U-Net</strong>): In the U-Net, text
                enters through cross-attention: Q from spatial features, K/V
                from text embeddings. Each spatial location attends to all text
                tokens. One-directional: the image reads the text, but the text
                embeddings are fixed—they never respond to what the image
                contains.
              </li>
            </ul>
            <p className="text-muted-foreground">
              DiT solved the architecture problem. Flow matching solved the
              training objective problem. But DiT is class-conditional—no text.
              And the U-Net&rsquo;s cross-attention is one-directional. SD3 and
              Flux solve both.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Open Question">
            The previous lesson ended with DiT generating golden retrievers and
            volcanos from class labels. The question left unanswered: how do
            you add text conditioning to a transformer-based denoising network?
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — Convergence map */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Five Threads Converge"
            subtitle="Every component traces to a lesson you completed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every component of the current frontier architecture traces back
              to a lesson you have already completed:
            </p>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <GradientCard title="Thread 1: Transformers" color="sky">
                <p className="text-sm">
                  Self-attention, FFN, residual connections, scaling. Built GPT
                  from scratch in Series 4.
                </p>
              </GradientCard>
              <GradientCard title="Thread 2: Latent Diffusion" color="violet">
                <p className="text-sm">
                  VAE, latent space, conditioning, denoising loop. Built a
                  diffusion model in Series 6.
                </p>
              </GradientCard>
              <GradientCard title="Thread 3: Flow Matching" color="emerald">
                <p className="text-sm">
                  Straight trajectories, velocity prediction, fewer steps.
                  Trained from scratch in Module 7.2.
                </p>
              </GradientCard>
              <GradientCard title="Thread 4: DiT" color="orange">
                <p className="text-sm">
                  Patchify, adaLN-Zero, transformer as denoising network.
                  Previous lesson.
                </p>
              </GradientCard>
              <GradientCard title="Thread 5: Text Encoding" color="cyan">
                <p className="text-sm">
                  CLIP from <LessonLink slug="clip">CLIP</LessonLink>, dual encoders from{' '}
                  <LessonLink slug="sdxl">SDXL</LessonLink>. How text conditions the model.
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              SD3 and Flux combine all five threads. Before showing you how,
              consider this: the original DiT used class labels for
              conditioning. If you wanted to add text conditioning to a DiT,
              how would you do it? You have text embeddings from CLIP. You have
              a transformer processing patch tokens. How do you make the
              transformer use the text?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Convergence, Not Revolution">
            This lesson should feel like arriving at a destination you have
            been traveling toward. Not a surprise, but an inevitable
            convergence. Every piece was established along the way.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Design challenge in reveal */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <details className="rounded-lg border bg-muted/30 p-4">
              <summary className="font-medium cursor-pointer text-primary">
                Challenge: predict the design choice
              </summary>
              <div className="mt-3 space-y-3 text-sm text-muted-foreground">
                <p>Three options for adding text to a DiT:</p>
                <ol className="list-decimal list-inside space-y-1 ml-2">
                  <li>
                    <strong>Cross-attention</strong> (the U-Net approach)—add
                    cross-attention layers where patch tokens attend to text
                    tokens
                  </li>
                  <li>
                    <strong>adaLN-Zero</strong> (the DiT approach)—project text
                    into the conditioning vector alongside the timestep
                  </li>
                  <li>
                    <strong>Concatenation</strong> (the simple approach)—add
                    text tokens to the patch token sequence and let
                    self-attention handle everything
                  </li>
                </ol>
                <p className="mt-2 italic">
                  Option 3 is what SD3 and Flux do. And it turns out to be both
                  simpler and better.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* Section 5: Explain — The Text Encoder Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Text Encoder Problem"
            subtitle="CLIP&rsquo;s limitations revisited"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              CLIP&rsquo;s text encoder was trained on image-text pairs via
              contrastive learning. It understands text through the lens of
              &ldquo;what image does this describe?&rdquo; This makes it good
              at visual concepts (colors, objects, scenes) but weak at
              compositional reasoning (spatial relationships, counting,
              negation), long descriptions, and abstract concepts that do not
              correspond to visual patterns.
            </p>
            <p className="text-muted-foreground">
              SDXL partially addressed this by adding a second, larger CLIP
              encoder (OpenCLIP ViT-bigG). But both encoders share the same
              training paradigm: learn text representations through image-text
              alignment. What if you used a text encoder trained purely on
              language understanding?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="CLIP&rsquo;s Blind Spot">
            CLIP understands text through images. It knows &ldquo;red
            car&rdquo; because it has seen red cars. But &ldquo;the car to the
            left of the tree&rdquo; requires compositional language
            understanding that contrastive training does not teach.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* T5-XXL — a language model as text encoder */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="T5-XXL: A Language Model as Text Encoder"
            subtitle="4.7 billion parameters of pure text understanding"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              T5-XXL is a 4.7 billion parameter encoder-decoder transformer
              trained on massive text corpora. You saw T5 named in the
              three-variant comparison in Series 4 as an encoder-decoder
              transformer (alongside encoder-only BERT and decoder-only GPT).
            </p>
            <p className="text-muted-foreground">
              Where CLIP&rsquo;s text encoder has 123M parameters and was
              trained to match images, T5-XXL has 4.7B parameters and was
              trained to understand language itself. The result: T5 produces
              text embeddings that capture compositional structure, complex
              relationships, and nuanced meaning that CLIP misses.
            </p>

            <CodeBlock
              code={`Text Encoder Comparison:
  CLIP ViT-L:       123M params   [77, 768]   trained on image-text pairs
  OpenCLIP ViT-bigG: 354M params  [77, 1280]  trained on image-text pairs
  T5-XXL:           4.7B params   [77, 4096]  trained on text alone`}
              language="text"
              filename="text_encoder_comparison.txt"
            />

            <p className="text-muted-foreground">
              T5&rsquo;s embeddings carry ~5x more information per token than
              CLIP&rsquo;s (4096 vs 768 dimensions). More importantly, they
              capture <em>different</em> information: linguistic structure
              rather than visual alignment.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Different Training, Different Understanding">
            CLIP: &ldquo;What image does this text describe?&rdquo; T5:
            &ldquo;What does this text mean?&rdquo; Different questions produce
            different embeddings. Both are valuable.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Why keep CLIP alongside T5? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Keep CLIP Alongside T5?"
            subtitle="Complementary, not redundant"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If T5 is so much better at understanding text, why not just use
              T5? Because CLIP and T5 provide complementary information:
            </p>

            <ComparisonRow
              left={{
                title: 'CLIP Encoders',
                color: 'sky',
                items: [
                  'Trained on image-text pairs',
                  'Embeddings aligned with visual features',
                  'Pooled CLS embedding for global adaLN-Zero conditioning',
                  'Good at: visual concepts, style, aesthetics',
                ],
              }}
              right={{
                title: 'T5-XXL',
                color: 'violet',
                items: [
                  'Trained on text alone',
                  'Embeddings capture linguistic meaning',
                  'No pooled embedding (encoder output only)',
                  'Good at: compositional descriptions, spatial reasoning, counting, abstract concepts',
                ],
              }}
            />

            <p className="text-muted-foreground">
              SD3 uses all three: CLIP ViT-L, OpenCLIP ViT-bigG, and T5-XXL.
              The CLIP pooled embeddings provide global conditioning via
              adaLN-Zero (same as SDXL). The per-token embeddings from all
              three are combined to form the text token sequence for joint
              attention.
            </p>
            <p className="text-muted-foreground">
              <strong>
                T5 does not replace CLIP. All three encoders contribute
                simultaneously.
              </strong>{' '}
              Removing any one degrades quality because each provides
              information the others lack.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Progression">
            <ul className="space-y-1 text-sm">
              <li>SD v1.5: one encoder [77, 768]</li>
              <li>SDXL: two encoders [77, 2048]</li>
              <li>SD3: three encoders, including T5&rsquo;s [77, 4096]</li>
            </ul>
            <p className="text-sm mt-2">
              Each generation adds richer text understanding through the same
              pipeline pattern.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Text encoder predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  SD3 uses T5-XXL with 4.7B parameters as one of its text
                  encoders. The denoising network (MMDiT) has ~2B parameters.
                  What does this imply about the total model size?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The text encoder alone is larger than the denoising network.
                    Total model size is ~8B+ across all components. This is a
                    significant practical consideration—you need substantial
                    VRAM just for the text encoders. It reflects the
                    field&rsquo;s recognition that text understanding is a
                    bottleneck: investing more parameters in understanding the
                    prompt is worth the cost.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague says: &ldquo;T5-XXL is a language model, so it
                  understands text better than CLIP. We should just use T5 and
                  drop CLIP entirely.&rdquo; What would be lost?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    CLIP&rsquo;s pooled embedding provides global conditioning
                    via adaLN-Zero—T5 does not produce this kind of summary
                    vector. More importantly, CLIP&rsquo;s embeddings are
                    aligned with visual features because of contrastive
                    training. T5&rsquo;s embeddings are purely linguistic. The
                    combination captures &ldquo;what does this text mean
                    linguistically&rdquo; AND &ldquo;what visual content does
                    this text map to.&rdquo; Dropping either loses information.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — MMDiT Joint Text-Image Attention */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="MMDiT: Joint Text-Image Attention"
            subtitle="The cross-attention limitation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the U-Net (and in a hypothetical text-conditioned DiT with
              cross-attention), text conditioning is one-directional:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Image tokens compute Q, text tokens provide K and V
              </li>
              <li>
                The image reads from the text: &ldquo;what does the prompt say
                about this spatial location?&rdquo;
              </li>
              <li>
                But the text embeddings are frozen in the attention
                computation—they do not change in response to what the image
                contains
              </li>
              <li>
                Each block has <strong>two</strong> attention operations:
                self-attention on image tokens (image reads image) AND
                cross-attention from image to text (image reads text)
              </li>
            </ul>
            <p className="text-muted-foreground">
              Joint attention asks: what if text and image were in the same
              sequence?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="One-Way Mirror">
            Cross-attention is like a one-way mirror: the image can see the
            text, but the text cannot see or respond to the image. The text
            embeddings are the same at every layer, regardless of what the
            image contains.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* The simple idea */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Simple Idea"
            subtitle="Concatenate and self-attend"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Concatenate the text tokens and image patch tokens into one
              sequence. Run standard self-attention on the combined sequence.
            </p>

            <CodeBlock
              code={`Text tokens:  [77, d_model]     (from text encoders, projected to d_model)
Image tokens: [256, d_model]    (from patchify, same as DiT)

Concatenated: [333, d_model]    (77 + 256 = 333 tokens)

Self-attention on [333, d_model]:
  Q: [333, d_head]  (every token produces a query)
  K: [333, d_head]  (every token produces a key)
  V: [333, d_head]  (every token produces a value)
  Attention weights: [333, 333]  (every token attends to every other token)
  Output: [333, d_model]

Split back: text [77, d_model], image [256, d_model]`}
              language="text"
              filename="joint_attention_trace.txt"
            />

            <p className="text-muted-foreground">
              One attention operation replaces two. And it provides four types
              of attention simultaneously:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Image-to-text</strong> (image tokens read text
                tokens)—equivalent to cross-attention in U-Net
              </li>
              <li>
                <strong>Text-to-image</strong> (text tokens read image
                tokens)—NEW: text representations update based on image content
              </li>
              <li>
                <strong>Image-to-image</strong> (image tokens read image
                tokens)—equivalent to self-attention in DiT
              </li>
              <li>
                <strong>Text-to-text</strong> (text tokens read text
                tokens)—NEW: text representations refine each other within the
                block
              </li>
            </ul>
            <p className="text-muted-foreground">
              Cross-attention provided only the first. Joint attention provides
              all four.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Simpler, Not More Complex">
            The &ldquo;M&rdquo; in MMDiT stands for &ldquo;Multimodal&rdquo;—
            but the mechanism is just standard self-attention on a multimodal
            sequence. One attention operation instead of two.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Cross-attention vs Joint attention comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'Cross-Attention (U-Net)',
                color: 'amber',
                items: [
                  'Two attention operations per block: self-attention on image + cross-attention to text',
                  'One-directional: image reads text',
                  'Text representation fixed across blocks (same embeddings at every layer)',
                  'Two separate attention computations with different Q sources',
                  'Attention matrices: self [256, 256] + cross [256, 77]',
                ],
              }}
              right={{
                title: 'Joint Attention (MMDiT)',
                color: 'emerald',
                items: [
                  'One attention operation: self-attention on concatenated text+image',
                  'Bidirectional: image reads text AND text reads image',
                  'Text representation updated by each block (text refines based on image context)',
                  'One standard self-attention',
                  'Attention matrix: joint [333, 333]',
                ],
              }}
            />
          </div>
        </Row.Content>
      </Row>

      {/* The "one room" analogy */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One Room, One Conversation"
            subtitle="Why bidirectional attention matters"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think of cross-attention as two rooms connected by a one-way
              mirror. The image room can see the text room, but the text room
              cannot see or respond to what is happening with the image. Joint
              attention puts everyone in the same room having one conversation.
              Text tokens hear what image tokens are saying and can respond.
              Image tokens hear what text tokens are saying and can respond.
              The interaction is symmetric.
            </p>
            <p className="text-muted-foreground">
              This matters because text conditioning in the U-Net is static—the
              same text embeddings are used at every denoising step, regardless
              of what the image looks like at that point. In MMDiT, the text
              representations evolve through the network&rsquo;s layers,
              refining their meaning based on the image context. &ldquo;A crane
              near a river&rdquo; can be disambiguated based on what the image
              actually contains.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Room, One Conversation">
            Cross-attention: two rooms, one-way mirror, the image watches the
            text. Joint attention: one room, everyone hears everyone. Simpler
            and richer.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Joint attention predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In cross-attention, the attention matrix is [256, 77] (256
                  image tokens attending to 77 text tokens). In joint
                  attention, it is [333, 333]. How does the attention compute
                  cost compare?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Cross-attention attention cost is proportional to 256 × 77
                    = 19,712. But the block also has self-attention at 256 ×
                    256 = 65,536. Total: ~85,248. Joint attention cost: 333 ×
                    333 = 110,889. Joint attention is ~30% more expensive per
                    block in attention alone. However, there is only ONE
                    attention operation instead of two, which reduces other
                    overhead. The practical cost is comparable.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  After the joint attention computation, MMDiT splits the
                  output back into text tokens [77, d_model] and image tokens
                  [256, d_model]. Why split them?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Text tokens and image tokens need different post-attention
                    processing—each modality has its own FFN. Text and image
                    representations live in different spaces, so the same FFN
                    would not serve both well. The split also allows different
                    adaLN-Zero modulation for each modality&rsquo;s FFN.
                    Splitting preserves modality-specific processing while the
                    attention itself is shared.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague claims: &ldquo;Joint attention means SD3 treats
                  text and image identically.&rdquo; Is this accurate?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The attention computation is shared, but text and image
                    tokens have separate Q/K/V projections and separate FFN
                    layers. Each modality maintains its own representational
                    identity. They attend to each other through a shared
                    attention mechanism, but they &ldquo;think&rdquo;
                    differently. This is the key nuance of MMDiT that
                    distinguishes it from naive concatenation.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Explain — The MMDiT Block in Detail */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The MMDiT Block in Detail"
            subtitle="Modality-specific projections"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The simple version—&ldquo;concatenate and self-attend&rdquo;—
              raises a question: text embeddings (from T5 or CLIP) and image
              patch embeddings (from patchify) live in very different
              representational spaces. Should they share the same W_Q, W_K,
              W_V projection matrices?
            </p>
            <p className="text-muted-foreground">
              SD3&rsquo;s answer: no. Each modality gets its own projection
              matrices.
            </p>

            <CodeBlock
              code={`MMDiT Block:
  Text tokens [77, d_model] -> text W_Q, text W_K, text W_V -> text Q/K/V
  Image tokens [256, d_model] -> image W_Q, image W_K, image W_V -> image Q/K/V

  Concatenate: Q = [text_Q; image_Q] -> [333, d_head]
               K = [text_K; image_K] -> [333, d_head]
               V = [text_V; image_V] -> [333, d_head]

  Standard self-attention on concatenated Q/K/V -> [333, d_model]

  Split: text output [77, d_model], image output [256, d_model]

  Text output -> text FFN -> text residual
  Image output -> image FFN -> image residual`}
              language="text"
              filename="mmdit_block.txt"
            />

            <p className="text-muted-foreground">
              This is not naive concatenation where everything shares the same
              projections. Each modality formulates its Q, K, V through its own
              learned matrices—they &ldquo;speak their own language.&rdquo; The
              shared attention lets them &ldquo;hear each other.&rdquo; The
              separate FFNs let them &ldquo;think independently.&rdquo;{' '}
              <strong>Shared listening, separate thinking.</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Naive Concatenation">
            If text and image shared all projections (same W_Q, W_K, W_V, same
            FFN), the model would struggle: text embeddings from T5 and patch
            embeddings from patchify have fundamentally different structures.
            Forcing them through the same projections is like asking a French
            speaker and a Japanese speaker to use the same dictionary.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Negative example + contrast with IP-Adapter */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Compare this to IP-Adapter&rsquo;s approach from{' '}
              <strong>IP-Adapter</strong>: IP-Adapter reads from two separate
              documents (text K/V and image K/V) and combines the readings
              with weighted addition. MMDiT puts everything into one document.
              Different design philosophies for multi-source attention:
              IP-Adapter keeps sources separate, MMDiT merges them.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Timestep conditioning in MMDiT */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Timestep Conditioning in MMDiT"
            subtitle="Text enters through two paths"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The timestep still enters through adaLN-Zero, just like in DiT.
              The conditioning vector c (timestep embedding + pooled CLIP
              embedding) produces gamma, beta, alpha for each sub-layer&rsquo;s
              LayerNorm. This is unchanged from{' '}
              <strong>Diffusion Transformers</strong>.
            </p>
            <p className="text-muted-foreground">
              So text enters through <strong>two paths</strong> in MMDiT:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Per-token path:</strong> T5 and CLIP per-token
                embeddings join the attention sequence as text tokens (spatially
                varying, context-dependent)
              </li>
              <li>
                <strong>Global path:</strong> Pooled CLIP embeddings join the
                timestep in adaLN-Zero (global, same at every spatial location)
              </li>
            </ol>
            <p className="text-muted-foreground">
              This dual-path text conditioning mirrors SDXL, where per-token
              CLIP embeddings entered cross-attention and the pooled OpenCLIP
              embedding entered adaptive norm. Same design principle, different
              mechanism for the per-token path.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Paths for Text">
            Per-token path: detailed, spatially varying text conditioning via
            joint attention. Global path: summary conditioning via adaLN-Zero.
            Both needed. Same dual-path pattern as SDXL, carried forward.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 10: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="MMDiT block predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In a standard DiT block, the adaLN-Zero MLP produces 6
                  parameters (gamma_1, beta_1, alpha_1, gamma_2, beta_2,
                  alpha_2). In an MMDiT block, which has separate FFNs for text
                  and image, how many adaLN-Zero parameters do you expect?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Each FFN sub-layer needs its own gamma, beta, alpha. With
                    separate text and image FFNs, that is 3 params for the text
                    FFN + 3 params for the image FFN + 3 params for the shared
                    attention sub-layer&rsquo;s pre-norm. The exact count
                    depends on implementation details, but the principle is:
                    more sub-layers means more adaLN-Zero parameters per block.
                    The conditioning MLP grows accordingly.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Could you use IP-Adapter with an MMDiT model? How would the
                  architecture differ from IP-Adapter with a U-Net?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    In principle, yes. IP-Adapter adds image conditioning via
                    decoupled cross-attention K/V. In an MMDiT, you could add
                    CLIP image tokens to the combined sequence (alongside text
                    and patch tokens), or add a separate decoupled attention
                    path. The concept is the same—add a new conditioning
                    source—but the implementation would differ. MMDiT&rsquo;s
                    joint attention provides a natural way to incorporate
                    additional token sequences, so adding image reference
                    tokens to the combined sequence is the more elegant
                    approach.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Explain — Flow Matching in Practice */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Flow Matching in Practice"
            subtitle="Applying what you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SD3&rsquo;s training objective is the conditional flow matching
              you already know from <LessonLink slug="flow-matching">Flow Matching</LessonLink>:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Straight-line interpolation:{' '}
                <InlineMath math="x_t = (1-t) \cdot x_0 + t \cdot \epsilon" />
              </li>
              <li>
                Velocity prediction:{' '}
                <InlineMath math="v_\theta(x_t, t)" /> predicts{' '}
                <InlineMath math="v = \epsilon - x_0" />
              </li>
              <li>
                Training loss:{' '}
                <InlineMath math="\text{MSE}(v_\theta(x_t, t), \, \epsilon - x_0)" />
              </li>
              <li>
                Straight trajectories by construction, fewer ODE solver steps
                needed
              </li>
            </ul>
            <p className="text-muted-foreground">
              <strong>
                You already know everything about flow matching that SD3 uses.
              </strong>{' '}
              The training objective is identical to what you implemented in
              your flow matching notebook. This is not new content—it is the
              same concept applied to a real model.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Objective, Bigger Model">
            SD3&rsquo;s flow matching training is what you trained in{' '}
            <LessonLink slug="flow-matching">Flow Matching</LessonLink>: same interpolation, same velocity
            target, same MSE loss. Only the scale changes—millions of images
            and billions of parameters instead of 2D toy data.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Rectified flow application */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Rectified Flow and Logit-Normal Sampling"
            subtitle="Two practical refinements"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SD3 uses rectified flow (from <LessonLink slug="flow-matching">Flow Matching</LessonLink>):
              after initial training, generate aligned (noise, data) pairs
              using the model, then retrain on these pairs. This makes the
              aggregate velocity field straighter, reducing the number of
              inference steps needed even further.
            </p>
            <p className="text-muted-foreground">
              In practice, SD3/Flux achieves good results in 20-30 steps,
              compared to 50+ for DDPM-based models. This is the practical
              payoff of the &ldquo;curved vs straight&rdquo; insight from{' '}
              <LessonLink slug="flow-matching">Flow Matching</LessonLink>.
            </p>
            <p className="text-muted-foreground">
              One new detail: instead of sampling timesteps uniformly during
              training (every <InlineMath math="t" /> equally likely), SD3 uses{' '}
              <strong>logit-normal sampling</strong>. This biases training
              toward intermediate timesteps (around{' '}
              <InlineMath math="t = 0.5" />) where the denoising task is
              hardest—not pure noise (easy: predict roughly toward data center)
              and not nearly clean (easy: predict small refinements).
            </p>
            <p className="text-muted-foreground">
              The intuition: intermediate noise levels are where the model must
              make the most important decisions about composition and
              structure. Spending more training compute at these timesteps
              improves overall quality.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="A Training Optimization">
            Logit-normal sampling is a training optimization, not a
            fundamental change. The training objective (flow matching velocity
            prediction) is the same. Only the distribution over which
            timesteps get trained more changes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Check #4 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Flow matching in practice"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  SD3 uses flow matching with velocity prediction. The model
                  you trained in <LessonLink slug="flow-matching">Flow Matching</LessonLink> used the same
                  objective on 2D data. What changes when you apply this to a
                  real image generation model?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Nothing fundamental changes. The interpolation is the same
                    (linear between noise and data). The training target is the
                    same (velocity). The loss is the same (MSE). What changes
                    is scale: the input is a high-dimensional latent [4, 64,
                    64] instead of 2D points, the model is a billion-parameter
                    MMDiT instead of a small MLP, and the training data is
                    millions of images. The concept is identical; the
                    engineering is different.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Compare SD3&rsquo;s training setup to DDPM training from
                  Series 6: what is the same and what is different?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Same: sample data, add noise, predict something, MSE loss,
                    backprop. Different: (1) interpolation formula (linear vs
                    sqrt(alpha_bar) weighting), (2) prediction target (velocity
                    vs noise), (3) timestep sampling (logit-normal vs uniform),
                    (4) trajectory shape (straight vs curved). The training
                    LOOP is identical in structure. The OBJECTIVE and SCHEDULE
                    differ.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 13: Elaborate — The Convergence */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Convergence"
            subtitle="The full SD3 pipeline, annotated"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete SD3 pipeline, with each component annotated
              by where you learned it. Note: the earlier joint attention example
              used 256 image tokens from a 256×256 latent (matching the DiT
              lesson&rsquo;s ImageNet example). At SD3&rsquo;s native
              1024×1024 resolution, the latent is [4, 64, 64] and patchify
              produces 1024 tokens—the same operation at higher resolution.
            </p>

            <CodeBlock
              code={`Full SD3 Pipeline:
 1. Prompt -> CLIP ViT-L text encoder [77, 768]           (CLIP)
 2. Prompt -> OpenCLIP ViT-bigG text encoder [77, 1280]    (SDXL)
 3. Prompt -> T5-XXL text encoder [77, 4096]               (this lesson)
 4. CLIP pooled embeddings + timestep -> c for adaLN-Zero   (Diffusion Transformers)
 5. Per-token embeddings projected -> text tokens [77, d]    (this lesson)
 6. Noisy latent z_t [4, 64, 64] -> patchify -> [1024, d]  (Diffusion Transformers)
 7. Concatenate: [77 + 1024, d] = [1101, d]                 (this lesson: MMDiT)
 8. N MMDiT blocks with joint attention + adaLN-Zero         (this lesson + Diffusion Transformers)
 9. Split output -> image tokens [1024, d]                   (this lesson)
10. Unpatchify -> [4, 64, 64]                               (Diffusion Transformers)
11. Flow matching sampling step                              (Flow Matching)
12. Repeat steps 6-11 for ~28 steps                         (Flow Matching: fewer steps needed)
13. VAE decode z_0 -> [3, 1024, 1024]                       (From Pixels to Latents)`}
              language="text"
              filename="sd3_pipeline.txt"
            />

            <p className="text-muted-foreground">
              Thirteen steps. Every one traces to a lesson you completed.
              Nothing in this pipeline is unexplained.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="You Already Knew All of This">
            SD3 is not a new paradigm. It is the convergence of concepts you
            built over the entire course: transformers (Series 4) + latent
            diffusion (Series 6) + flow matching (Module 7.2) + DiT
            (previous lesson) + better text encoding (this lesson). The
            frontier is the synthesis of your understanding.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* What makes SD3/Flux the current frontier */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Independent Advances"
            subtitle="Each addresses a specific limitation"
          />
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard
                title="Architecture: DiT → MMDiT"
                color="emerald"
              >
                <p className="text-sm">
                  Replace the U-Net with a transformer. Replace
                  cross-attention with joint attention. Result: simpler
                  architecture, better scaling, bidirectional text-image
                  interaction.
                </p>
              </GradientCard>
              <GradientCard
                title="Training: DDPM → Flow Matching"
                color="sky"
              >
                <p className="text-sm">
                  Replace noise prediction with velocity prediction on straight
                  trajectories. Result: fewer inference steps (20-30 vs 50+),
                  simpler training.
                </p>
              </GradientCard>
              <GradientCard
                title="Text Encoding: CLIP → CLIP + T5"
                color="violet"
              >
                <p className="text-sm">
                  Add a language model alongside the vision-language model.
                  Result: better compositional understanding, handling of
                  complex prompts, richer text conditioning.
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Each advance is independent. You could have a U-Net with flow
              matching (and some models do). You could have DiT with DDPM noise
              prediction (the original DiT paper does). You could have T5 with
              cross-attention (Imagen does). SD3 combines all three.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* SD3 vs Flux (brief positioning) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SD3 vs Flux"
            subtitle="Same family, different variants"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SD3 and Flux are both MMDiT architectures from the same research
              lineage (Stability AI / Black Forest Labs). The key differences:
            </p>
            <ComparisonRow
              left={{
                title: 'SD3',
                color: 'sky',
                items: [
                  'Full triple encoder setup (CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL)',
                  'Multiple size variants (SD3 Medium, SD3.5 Large)',
                  'Stability AI',
                ],
              }}
              right={{
                title: 'Flux',
                color: 'violet',
                items: [
                  'Single-stream blocks in later layers (text+image share projections after initial dual-stream processing)',
                  'Drops the second CLIP encoder—uses only CLIP ViT-L + T5-XXL',
                  'Variants: Dev (research), Schnell (distilled, ~4 steps), Pro (commercial API)',
                  'Black Forest Labs',
                ],
              }}
            />
            <p className="text-muted-foreground">
              The architectural principles are the same. The differences are in
              scale, training data, and distillation. For understanding how the
              architecture works, SD3 and Flux are the same family.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Vocabulary">
            When you encounter &ldquo;Flux.1 Schnell&rdquo; or &ldquo;SD3.5
            Large,&rdquo; you now know what they are: MMDiT variants with flow
            matching training, differing in scale and distillation approach.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 14: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: SD3 Pipeline"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the lesson concrete—inspect the triple
                encoder setup, visualize joint attention patterns, generate
                images with flow matching, and trace the full SD3 pipeline.
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
                    Exercise 1 (Guided): SD3 Pipeline Inspection.
                  </strong>{' '}
                  Load SD3 Medium via diffusers
                  (StableDiffusion3Pipeline). Inspect the triple text encoder
                  setup: print model class names, parameter counts. Verify
                  embedding shapes: CLIP ViT-L [77, 768], OpenCLIP ViT-bigG
                  [77, 1280], T5-XXL [77, 4096]. Inspect the transformer
                  (MMDiT): count parameters, identify modality-specific Q/K/V
                  projections.
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): Joint Attention Structure.
                  </strong>{' '}
                  Encode a prompt through all three text encoders. Verify the
                  concatenated sequence length (text tokens + image tokens).
                  Inspect the MMDiT block&rsquo;s modality-specific projections.
                  Visualize the four-quadrant structure of the joint attention
                  matrix: text-to-text, text-to-image, image-to-text,
                  image-to-image.
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): SD3 Generation and Flow Matching
                    Steps.
                  </strong>{' '}
                  Generate &ldquo;a cat sitting on a beach at sunset&rdquo;
                  with SD3. Vary the number of inference steps: 10, 20, 30, 50.
                  Compare quality across step counts: 20-30 steps should be
                  sufficient (the flow matching payoff).
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): The Convergence Pipeline Trace.
                  </strong>{' '}
                  Generate an image with SD3, capturing intermediate outputs.
                  Trace the full pipeline: text encoding (measure shapes),
                  patchify (verify token count), denoising steps (count), VAE
                  decode (verify output shape). For each step, annotate which
                  lesson covered the relevant concept.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: pipeline inspection (shapes)</li>
              <li>Guided: joint attention visualization (patterns)</li>
              <li>Supported: generation with step counts (flow matching)</li>
              <li>Independent: full pipeline trace (convergence)</li>
            </ol>
            <p className="text-sm mt-2">
              SD3 Medium requires ~12 GB VRAM in float16. Use Colab&rsquo;s T4
              or A100 GPU.
            </p>
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
                headline: 'One room, one conversation.',
                description:
                  'MMDiT replaces cross-attention with joint self-attention: concatenate text and image tokens, let everything attend to everything. Bidirectional interaction (text reads image, image reads text), one attention operation instead of two. Simpler and richer.',
              },
              {
                headline: 'Modality-specific processing, shared attention.',
                description:
                  'Text and image tokens have their own Q/K/V projections and their own FFN layers. They attend together but think differently. The projections map each modality into a shared attention space; the FFNs keep their representations distinct.',
              },
              {
                headline: 'Three encoders for three kinds of understanding.',
                description:
                  'CLIP ViT-L (visual alignment), OpenCLIP ViT-bigG (richer visual alignment), T5-XXL (deep linguistic understanding). Pooled CLIP for global conditioning via adaLN-Zero. Per-token embeddings for joint attention. Each contributes information the others cannot.',
              },
              {
                headline: 'Flow matching delivers.',
                description:
                  'The straight-line trajectories from Flow Matching produce the practical result: SD3/Flux generates good images in 20-30 steps. Rectified flow makes the aggregate trajectories even straighter. Same concept you trained, applied at scale.',
              },
              {
                headline: 'Convergence, not revolution.',
                description:
                  'SD3/Flux combines: transformer blocks (Series 4) + latent diffusion (Series 6) + flow matching (Module 7.2) + DiT (previous lesson) + better text encoding (this lesson). Every component traces to a lesson you completed. The frontier is not beyond your understanding—it IS your understanding, combined.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 16: Series Conclusion */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Series 7 Complete"
            subtitle="From Stable Diffusion to the frontier"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You started Series 7 with Stable Diffusion v1.5 as your
              reference architecture: a U-Net trained with DDPM noise
              prediction, conditioned via a single CLIP encoder through
              cross-attention, generating 512×512 images in 50+ steps.
            </p>
            <p className="text-muted-foreground">
              Over eleven lessons, you traced the evolution:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Module 7.1:</strong> Added spatial and image
                conditioning without changing the model (ControlNet, IP-Adapter)
              </li>
              <li>
                <strong>Module 7.2:</strong> Reframed diffusion through the
                score function and discovered flow matching—straight paths
                instead of curved
              </li>
              <li>
                <strong>Module 7.3:</strong> Collapsed the multi-step process
                with consistency models and adversarial distillation
              </li>
              <li>
                <strong>Module 7.4:</strong> Replaced the architecture
                entirely—U-Net to DiT to MMDiT
              </li>
            </ul>
            <p className="text-muted-foreground">
              The current frontier (SD3, Flux) is the synthesis: a transformer
              processing a joint text-image token sequence, trained with flow
              matching, conditioned by three text encoders including a large
              language model. Every design choice in this architecture has a
              reason you now understand.
            </p>
            <p className="text-muted-foreground">
              You can read a diffusion model paper published today and trace
              its design choices back to concepts you have built from scratch.
              That was the goal of this course.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Goal">
            You started overwhelmed by the pace of AI. Now you can read
            frontier research and trace every design choice to concepts you
            understand. The papers are not beyond you—they are combinations of
            ideas you have built.
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
              'T5-XXL as complementary text encoder alongside CLIP',
              'Flow matching applied in practice: fewer steps, better quality',
              'The full SD3/Flux pipeline traced end-to-end',
            ]}
            nextModule="(Series Complete)"
            nextTitle="You have completed the Post-SD Advances series"
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Scaling Rectified Flow Transformers for High-Resolution Image Synthesis',
                authors: 'Esser, Kulal, Blattmann, Entezari, Muller, Saini, Levi, Noroozi, Omer, Rombach & Poole, 2024',
                url: 'https://arxiv.org/abs/2403.03206',
                note: 'The SD3 paper. Section 3 covers the MMDiT architecture (joint attention, modality-specific projections). Section 4 covers rectified flow and logit-normal sampling. This is the primary reference for this lesson.',
              },
              {
                title: 'Scalable Diffusion Models with Transformers',
                authors: 'Peebles & Xie, 2023',
                url: 'https://arxiv.org/abs/2212.09748',
                note: 'The DiT paper. Foundation for MMDiT—covered in the previous lesson.',
              },
              {
                title: 'Flow Matching for Generative Modeling',
                authors: 'Lipman, Chen, Ben-Hamu & Nickel, 2023',
                url: 'https://arxiv.org/abs/2210.02747',
                note: 'The flow matching paper. Sections 2-3 cover the straight-line interpolation and velocity prediction that SD3 uses as its training objective.',
              },
              {
                title: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transfer Transformer',
                authors: 'Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li & Liu, 2020',
                url: 'https://arxiv.org/abs/1910.10683',
                note: 'The T5 paper. SD3 uses the XXL variant (4.7B params) as one of its text encoders. The architecture details are not needed—what matters is that T5 provides richer text understanding than CLIP.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Series Complete"
            description="You have traced the evolution of generative models from the foundations of deep learning through the current frontier. Every design choice in SD3 and Flux traces back to a concept you built from scratch. The papers are not beyond you—they are combinations of ideas you understand."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
