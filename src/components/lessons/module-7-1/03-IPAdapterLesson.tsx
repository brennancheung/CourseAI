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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-1-3-ip-adapter.ipynb'

/**
 * IP-Adapter
 *
 * Lesson 3 in Module 7.1 (Controllable Generation). Final lesson in the module.
 * Cognitive load: BUILD (2 new concepts).
 *
 * Core concepts:
 * 1. Decoupled cross-attention (parallel K/V projections for image embeddings)
 * 2. IP-Adapter as a lightweight, general-purpose image conditioning adapter
 *
 * Builds on deep cross-attention knowledge from 6.3.4, CLIP shared embedding
 * space from 6.3.3, ControlNet architecture from 7.1.1, and LoRA/textual
 * inversion from Series 4/6.
 *
 * Previous: ControlNet in Practice (Module 7.1, Lesson 2 / CONSOLIDATE)
 * Next: Module complete — next module in Series 7
 */

export function IPAdapterLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="IP-Adapter"
            description="Add image-based semantic conditioning to a frozen Stable Diffusion model—not edges or depth, but the visual identity of a reference photograph—via a parallel K/V pathway in cross-attention."
            category="Controllable Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how IP-Adapter adds a second set of K/V projections for
            CLIP image embeddings alongside the existing text K/V projections in
            cross-attention. Trace the data flow through decoupled
            cross-attention, reason about why the parallel path preserves text
            conditioning, and predict behavior at different scale values.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Module Finale">
            This is the final lesson in Module 7.1. You will complete the
            controllable generation story: ControlNet adds WHERE, text provides
            WHAT, and now IP-Adapter adds WHAT-IT-LOOKS-LIKE.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'How decoupled cross-attention adds a parallel K/V pathway for image embeddings',
              'Why CLIP image embeddings (not VAE latents) are the right conditioning signal',
              'How IP-Adapter preserves text conditioning—decoupling, not replacement',
              'The scale parameter for controlling image influence strength',
              'How IP-Adapter compares to LoRA, textual inversion, and ControlNet',
              'NOT: IP-Adapter training procedure in detail (briefly mentioned for understanding)',
              'NOT: implementing IP-Adapter from scratch',
              'NOT: IP-Adapter Plus, Face ID, or other variants (mentioned for vocabulary only)',
              'NOT: CLIP image encoder internals (ViT architecture—used as black box)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — Cross-Attention and CLIP Image Encoder */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Two concepts you will need in the next ten minutes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Cross-attention K/V mechanism.</strong> In{' '}
              <strong>Text Conditioning and Guidance</strong>, you learned that
              cross-attention works by projecting spatial features into queries
              (Q) and conditioning embeddings into keys (K) and values (V). Each
              spatial location generates its own query and attends independently
              to all text tokens. This is the WHAT channel in the
              WHEN/WHAT/WHERE framework—cross-attention is how the U-Net reads
              from the text prompt.
            </p>
            <p className="text-muted-foreground">
              <strong>CLIP image encoder output.</strong> In{' '}
              <strong>CLIP</strong>, you learned that CLIP has two separate
              encoders producing vectors in a shared embedding space. You worked
              extensively with the text encoder&rsquo;s output: a sequence of 77
              token embeddings, not just a single summary vector. The same is
              true for the image encoder—it produces a sequence of image patch
              embeddings (typically 257 tokens for ViT-H/14: 256 patches + 1
              CLS token). IP-Adapter uses this full sequence, not just the
              pooled CLS vector. A small trainable projection maps these image
              patch embeddings to the U-Net&rsquo;s expected cross-attention
              dimensions before they are fed into the K/V projections.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Sequence, Not Summary">
            Just as the text encoder gives cross-attention 77 token embeddings
            to attend over, the image encoder gives 257 patch embeddings.
            Each spatial location in the U-Net can attend to different image
            patches—the same per-spatial-location attention pattern you
            already know from text.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — The Description Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Description Problem"
            subtitle="What text cannot capture"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now have spatial control with ControlNet—edges, depth maps,
              and poses that tell the model WHERE to put things. But consider a
              reference photograph with a specific color palette, lighting mood,
              or material texture. Try to describe it in words:{' '}
              <em>
                &ldquo;warm amber lighting with soft bokeh and muted earth
                tones.&rdquo;
              </em>{' '}
              That description is lossy—it captures some of the feeling but
              misses the precise quality you want.
            </p>
            <p className="text-muted-foreground">
              And for a specific object—your cat, a particular ceramic vase, a
              company&rsquo;s product—text fails entirely. You cannot describe
              the exact visual identity of a specific object in 77 tokens.
              Spatial maps cannot help either: a Canny edge map captures
              contours, not color palette. A depth map captures layering, not
              material texture.
            </p>
            <p className="text-muted-foreground">
              What if you could just <strong>show</strong> the model the image?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Missing Dimension">
            Text provides WHAT. ControlNet provides WHERE. Neither captures
            WHAT-IT-LOOKS-LIKE—the precise visual character of a reference
            image. That is the gap IP-Adapter fills.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Hook part 2: Challenge */}
      <Row>
        <Row.Content>
          <GradientCard title="Design Challenge" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                You already have all the pieces to solve this. Think about what
                you know:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>
                  CLIP&rsquo;s image encoder produces embeddings in the same
                  geometric space as text embeddings
                </li>
                <li>
                  Cross-attention reads from K/V embeddings projected from
                  conditioning signals
                </li>
                <li>
                  Each spatial location in the U-Net attends independently to
                  conditioning tokens
                </li>
              </ul>
              <p className="font-medium">
                How would you feed image features into cross-attention WITHOUT
                disrupting the text path that is already there?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <p className="mt-2 text-muted-foreground">
                  Add a <strong>second</strong> set of K/V projections for image
                  embeddings, parallel to the existing text K/V projections. The
                  Q projection is shared—spatial features ask the same questions
                  of both text and image. The two attention outputs are combined.
                  The existing text path is untouched.
                </p>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Pattern">
            This parallels the ControlNet design challenge from{' '}
            <strong>ControlNet</strong>—you were challenged to sketch the
            solution before being told. The answer here is even simpler: one
            additional K/V pathway.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — Decoupled Cross-Attention */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Decoupled Cross-Attention"
            subtitle="The core architectural change"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The solution is called <strong>decoupled cross-attention</strong>.
              Follow each step—it should feel like an &ldquo;of course&rdquo;
              chain:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                CLIP&rsquo;s image encoder produces embeddings in the shared
                text-image space (you know this from <strong>CLIP</strong>)
              </li>
              <li>
                Cross-attention&rsquo;s K/V projections translate conditioning
                embeddings into the U-Net&rsquo;s internal language (you know
                this from <strong>Text Conditioning and Guidance</strong>)
              </li>
              <li>
                Adding a <strong>separate</strong> set of K/V projections for
                image embeddings feeds image semantics into the U-Net without
                touching the text path
              </li>
              <li>
                The two attention outputs are added:{' '}
                <code className="text-xs">
                  output = text_attn + scale × image_attn
                </code>
              </li>
              <li>
                The Q projection is shared—spatial features ask the same
                questions of both text and image references
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Reference Documents">
            Extend the &ldquo;reading from a reference document&rdquo; analogy.
            Standard cross-attention: the U-Net reads from one reference
            document (text embeddings). Decoupled cross-attention: it reads
            from <strong>two</strong> reference documents simultaneously (text
            and image), each with its own translation layer (K/V projections),
            and combines what it reads.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Architecture diagram */}
      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph TD
              SF["Spatial Features"]:::frozen --> WQ["W_Q (shared)"]:::frozen
              WQ --> Q["Q"]:::frozen

              TE["Text Embeddings (77 tokens)"]:::frozen --> WKt["W_K_text (frozen)"]:::frozen
              TE --> WVt["W_V_text (frozen)"]:::frozen
              WKt --> Kt["K_text"]:::frozen
              WVt --> Vt["V_text"]:::frozen

              IE["Image Embeddings (257 tokens)"]:::new --> WKi["W_K_image (trainable)"]:::new
              IE --> WVi["W_V_image (trainable)"]:::new
              WKi --> Ki["K_image"]:::new
              WVi --> Vi["V_image"]:::new

              Q --> ATTN_T["Attention(Q, K_text, V_text)"]:::frozen
              Kt --> ATTN_T
              Vt --> ATTN_T

              Q --> ATTN_I["Attention(Q, K_image, V_image)"]:::new
              Ki --> ATTN_I
              Vi --> ATTN_I

              ATTN_T --> ADD["text_out + scale × image_out"]:::output
              ATTN_I --> ADD

              classDef frozen fill:#374151,stroke:#6b7280,color:#d1d5db
              classDef new fill:#5b21b6,stroke:#8b5cf6,color:#f5f3ff
              classDef output fill:#065f46,stroke:#10b981,color:#d1fae5
          `} />
          <p className="mt-3 text-sm text-muted-foreground">
            Gray components are frozen (existing SD cross-attention). Violet
            components are new and trainable (IP-Adapter). The Q projection is
            shared—the same spatial queries attend to both text and image
            references. The two attention outputs are combined via weighted
            addition.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Color Code">
            <ul className="space-y-1 text-sm">
              <li><strong>Gray:</strong> Frozen (existing SD)</li>
              <li><strong>Violet:</strong> New trainable (IP-Adapter)</li>
              <li><strong>Green:</strong> Combined output</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Pseudocode */}
      <Row>
        <Row.Content>
          <CodeBlock
            code={`# Standard cross-attention (frozen—unchanged from vanilla SD)
Q = W_Q(spatial_features)       # shared query
K_text = W_K_text(text_emb)     # existing frozen K
V_text = W_V_text(text_emb)     # existing frozen V
text_out = attention(Q, K_text, V_text)

# IP-Adapter branch (trainable—the only new thing)
K_image = W_K_image(image_emb)  # NEW trainable K
V_image = W_V_image(image_emb)  # NEW trainable V
image_out = attention(Q, K_image, V_image)

# Decoupled output
output = text_out + scale * image_out`}
            language="python"
            filename="decoupled_cross_attention.py"
          />
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One-Line Change">
            The architectural delta is genuinely small. The entire IP-Adapter
            mechanism is: compute a second attention output from image
            embeddings, scale it, and add. The text path is bit-for-bit
            identical to vanilla SD.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Shape walkthrough */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Shape Walkthrough"
            subtitle="Concrete dimensions through the decoupled attention"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              At the 16×16 resolution level (256 spatial positions), here are
              the concrete tensor shapes flowing through decoupled
              cross-attention:
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="text-left py-2 pr-4 text-muted-foreground font-medium">
                      Tensor
                    </th>
                    <th className="text-left py-2 pr-4 text-muted-foreground font-medium">
                      Shape
                    </th>
                    <th className="text-left py-2 text-muted-foreground font-medium">
                      Source
                    </th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs">Q</td>
                    <td className="py-2 pr-4 font-mono text-xs">[256, d_k]</td>
                    <td className="py-2 text-xs">
                      Spatial features, SHARED between text and image paths
                    </td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs">K_text, V_text</td>
                    <td className="py-2 pr-4 font-mono text-xs">[77, d_k]</td>
                    <td className="py-2 text-xs">
                      CLIP text encoder, FROZEN projections
                    </td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs">K_image, V_image</td>
                    <td className="py-2 pr-4 font-mono text-xs">[257, d_k]</td>
                    <td className="py-2 text-xs">
                      CLIP image encoder + trainable projection, NEW
                    </td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs">
                      Text attention weights
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">[256, 77]</td>
                    <td className="py-2 text-xs">
                      Same as standard SD
                    </td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="py-2 pr-4 font-mono text-xs">
                      Image attention weights
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">[256, 257]</td>
                    <td className="py-2 text-xs">
                      New—rectangular, not square
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-mono text-xs">
                      text_out, image_out
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">[256, d_v]</td>
                    <td className="py-2 text-xs">
                      Both produce same-shaped output, combined by addition
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-muted-foreground">
              The key insight: the text attention path is{' '}
              <strong>completely untouched</strong>. No frozen weights are
              modified. The image path is purely additive, with its own learned
              projections. This is why text conditioning is preserved.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Different Token Counts">
            Notice that the text attention weights are [256, 77] and image
            attention weights are [256, 257]. The two paths operate over
            different numbers of tokens—they are not interchangeable or
            symmetric. But both produce the same output shape [256, d_v],
            which is what makes the addition possible.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Misconception #4: IP-Adapter is NOT img2img */}
      <Row>
        <Row.Content>
          <WarningBlock title="Not Img2Img">
            IP-Adapter does <strong>not</strong> encode the reference image with
            the VAE and feed it into the denoising process. That would be
            img2img—where the image enters as a starting latent tensor in the
            denoising loop. IP-Adapter encodes the reference image with{' '}
            <strong>CLIP</strong> (semantic representation, not
            pixel-level) and injects it via cross-attention as a semantic
            signal. The reference image never enters the denoising loop as a
            latent. IP-Adapter with pure random noise as the starting point
            still produces output influenced by the reference image.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Misconception #5: Addition, not averaging */}
      <Row>
        <Row.Content>
          <WarningBlock title="Addition, Not Averaging">
            The formula{' '}
            <code className="text-xs">text_out + scale × image_out</code>{' '}
            is <strong>not</strong> an average. The text path always contributes
            at full strength—there is no scale factor on{' '}
            <code className="text-xs">text_out</code>. Setting scale=0.5 does
            not mean &ldquo;half text, half image.&rdquo; It means{' '}
            <strong>full text, plus half-strength image</strong>. The two paths
            also produce different-shaped attention weight matrices ([256, 77]
            for text vs [256, 257] for image)—they are not symmetric or
            interchangeable. The text and image branches are independent
            computations with separate learned K/V projections, combined by
            weighted addition.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Section 6: Check #1 — Predict-and-Verify */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1: Scale = 0" color="cyan">
              <p className="text-sm">
                If you set the IP-Adapter scale to 0, what happens to the
                output?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The image branch contributes nothing:{' '}
                  <code className="text-xs">
                    output = text_out + 0 × image_out = text_out
                  </code>
                  . The output is identical to standard SD with text only. Same
                  principle as ControlNet at conditioning_scale=0.
                </p>
              </details>
            </GradientCard>

            <GradientCard title="Question 2: Remove IP-Adapter Entirely" color="cyan">
              <p className="text-sm">
                If you remove the IP-Adapter entirely (delete W_K_image and
                W_V_image), does the frozen model&rsquo;s text-to-image output
                change?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No—IP-Adapter is purely additive. The frozen model is
                  bit-for-bit identical without it. The text K/V projections
                  were never modified. Same principle as the ControlNet
                  disconnect test from <strong>ControlNet</strong>.
                </p>
              </details>
            </GradientCard>

            <GradientCard title="Question 3: Why Share Q?" color="cyan">
              <p className="text-sm">
                Why does IP-Adapter share the Q projection instead of having its
                own W_Q_image?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Spatial features determine what each location is
                  &ldquo;seeking.&rdquo; The same spatial location should ask
                  the same question of both text and image—&ldquo;what should I
                  look like here?&rdquo; The answer comes from different sources
                  (text tokens vs image patches), but the question is the same.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explore — IP-Adapter in Practice */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="IP-Adapter in Practice"
            subtitle="Reference images, text prompts, and the scale dial"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider a reference photograph of a golden retriever. You load
              IP-Adapter and generate with two different text prompts, using
              the same reference image:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Example 1: Golden retriever with two prompts */}
      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-2">
            <GradientCard title="Prompt A" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Reference:</strong> Golden retriever photograph
                </p>
                <p>
                  <strong>Text:</strong> &ldquo;a painting of a dog in a
                  garden&rdquo;
                </p>
                <p>
                  <strong>Expected output:</strong> A painterly image of a dog
                  in a garden setting. The dog has the golden retriever&rsquo;s
                  coat color, fur texture, and facial features from the
                  reference photo. The scene (garden, composition, style) comes
                  from the text prompt.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Prompt B" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Reference:</strong> Same golden retriever photograph
                </p>
                <p>
                  <strong>Text:</strong> &ldquo;a dog running on a beach at
                  sunset&rdquo;
                </p>
                <p>
                  <strong>Expected output:</strong> A beach sunset scene with a
                  dog running. The dog carries the golden retriever&rsquo;s
                  visual identity from the reference—the same coat, face, and
                  build. The scene (beach, sunset, action) comes entirely from
                  the text.
                </p>
              </div>
            </GradientCard>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            The reference image provides the visual identity—WHAT-IT-LOOKS-LIKE.
            The text prompt provides the content and composition—WHAT is
            happening and WHERE. Both conditioning channels are active
            simultaneously. This is decoupling in action.
          </p>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern">
            This directly parallels the ControlNet coexistence example from{' '}
            <strong>ControlNet</strong>: same spatial map, two different text
            prompts producing different content. Here it is the same reference
            image instead of the same edge map—but the principle is identical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Example 2: Scale parameter sweep */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Scale Parameter"
            subtitle="A volume knob for image influence"
          />
          <p className="text-sm text-muted-foreground mb-3">
            Imagine a reference image of an ornate ceramic vase, with the prompt
            &ldquo;a vase on a wooden table, photorealistic.&rdquo; Only the
            IP-Adapter scale changes:
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="Scale 0.0" color="blue">
              <p className="text-xs">
                Image has no effect. The model generates a generic vase based
                purely on the text prompt—whatever &ldquo;vase&rdquo; means to
                SD&rsquo;s training data. No influence from the reference
                photograph&rsquo;s specific design, color, or ornamentation.
              </p>
            </GradientCard>
            <GradientCard title="Scale 0.5" color="emerald">
              <p className="text-xs">
                The reference image&rsquo;s visual character begins to show.
                The vase takes on some of the ornate detailing, the color
                palette shifts toward the reference, but the text
                prompt&rsquo;s influence is still strong. A blended result—the
                overall shape and setting come from the text, the visual flavor
                comes from the image.
              </p>
            </GradientCard>
            <GradientCard title="Scale 1.0" color="violet">
              <p className="text-xs">
                Strong image influence. The vase closely resembles the
                reference photograph&rsquo;s specific design—its color palette,
                ornamentation style, and material quality are clearly carried
                over. The text prompt still controls the scene (wooden table,
                photorealistic rendering), but the vase&rsquo;s visual identity
                is dominated by the reference.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Knob, New Context">
            This is the same &ldquo;volume knob&rdquo; pattern from
            ControlNet&rsquo;s conditioning_scale. Scale=0 means the
            conditioning signal is silent. Scale=1.0 means full volume.
            Familiar control, new conditioning dimension.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: IP-Adapter vs LoRA (highest-priority comparison) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="IP-Adapter vs LoRA"
            subtitle="Both target cross-attention—but the mechanism is fundamentally different"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the most important comparison. You have deep LoRA
              knowledge—you know that LoRA adds a low-rank bypass to the
              existing W_Q, W_K, W_V weight matrices. IP-Adapter also targets
              cross-attention. It is natural to assume they work the same way.
              They do not.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'LoRA',
              color: 'amber',
              items: [
                'Modifies EXISTING W_K, W_V matrices (Wx + BAx)',
                'Changes how text is processed for ALL prompts',
                'Trained on a specific concept or style',
                'Baked into the model weights',
                'Remove LoRA → text-to-image output changes',
              ],
            }}
            right={{
              title: 'IP-Adapter',
              color: 'violet',
              items: [
                'Adds NEW W_K_image, W_V_image matrices',
                'Adds a new information source (image embeddings)',
                'Trained once, works with ANY reference image',
                'Separate adapter, original weights untouched',
                'Remove IP-Adapter → text-to-image output is identical',
              ],
            }}
          />
          <div className="mt-4">
            <p className="text-sm text-muted-foreground">
              <strong>The key test:</strong> Remove LoRA and the text-to-image
              output changes (because the text K/V projections have been
              modified via the bypass). Remove IP-Adapter and the text-to-image
              output is{' '}
              <strong>identical</strong> (because the text K/V projections were
              never touched). LoRA changes how existing information is
              processed. IP-Adapter adds a new information source.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Different Mechanism">
            LoRA: &ldquo;same highway, add a detour&rdquo; (modifies the
            existing weight computation). IP-Adapter: &ldquo;build a second
            highway&rdquo; (adds an entirely new K/V pathway). Same
            destination (cross-attention output), different approach.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Elaborate — More Comparisons and Misconceptions */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Comparisons and Boundaries"
            subtitle="Placing IP-Adapter in the customization spectrum"
          />
        </Row.Content>
      </Row>

      {/* IP-Adapter vs Textual Inversion */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Textual Inversion',
              color: 'amber',
              items: [
                '768 trainable parameters',
                'Operates at INPUT to CLIP (embedding lookup table)',
                'One embedding per concept—train per concept',
                'Limited to object identity',
                'Preserves frozen model (purely additive)',
              ],
            }}
            right={{
              title: 'IP-Adapter',
              color: 'violet',
              items: [
                '~22M trainable parameters across all cross-attention layers',
                'Operates INSIDE the U-Net at cross-attention',
                'General-purpose—works with ANY image at inference',
                'Captures style, color, texture, identity, and mood',
                'Preserves frozen model (purely additive)',
              ],
            }}
          />
          <p className="mt-4 text-sm text-muted-foreground">
            Both preserve the frozen model. Both are purely additive. The
            difference is intervention point and expressiveness: textual
            inversion adds one word to CLIP&rsquo;s vocabulary (768 params, one
            concept). IP-Adapter adds image understanding to the U-Net&rsquo;s
            cross-attention (~22M params, any image).
          </p>
        </Row.Content>
      </Row>

      {/* IP-Adapter vs ControlNet */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'ControlNet (WHERE)',
              color: 'emerald',
              items: [
                'Structural/spatial control',
                'Input: spatial maps (edges, depth, pose)',
                'Adds features via encoder copy + zero convolutions',
                'Controls composition and layout',
                '~300M trainable parameters',
              ],
            }}
            right={{
              title: 'IP-Adapter (WHAT-IT-LOOKS-LIKE)',
              color: 'violet',
              items: [
                'Semantic/visual identity control',
                'Input: CLIP image embeddings',
                'Adds K/V via decoupled cross-attention',
                'Controls style, color, texture, identity',
                '~22M trainable parameters',
              ],
            }}
          />
          <p className="mt-4 text-sm text-muted-foreground">
            Together: ControlNet provides structure (WHERE), IP-Adapter provides
            visual identity (WHAT-IT-LOOKS-LIKE), text provides semantic content
            (WHAT). All three are additive and composable. Each targets a
            different part of the U-Net and a different dimension of control.
          </p>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Four Conditioning Channels">
            The full spectrum is now: timestep (WHEN) via adaptive norm, text
            (WHAT) via cross-attention K/V, ControlNet (WHERE) via encoder
            feature addition, IP-Adapter (WHAT-IT-LOOKS-LIKE) via decoupled
            cross-attention K/V. Four additive channels, one frozen model.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: spatial control boundary */}
      <Row>
        <Row.Content>
          <GradientCard title="Boundary Example: IP-Adapter Cannot Control Layout" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Setup:</strong> Reference image of a specific room
                layout—a living room with a sofa on the left, a bookshelf on the
                right, and a coffee table in the center.
              </p>
              <p>
                <strong>What IP-Adapter captures:</strong> The color palette
                (warm wood tones, cream upholstery), lighting quality (soft
                afternoon light), furniture style (mid-century modern), and
                overall mood. The generated output feels like the same design
                aesthetic.
              </p>
              <p>
                <strong>What IP-Adapter does NOT capture:</strong> The exact
                positions of furniture. The sofa might end up on the right, the
                bookshelf might disappear, the coffee table might shift.
                IP-Adapter captures semantic visual qualities, not spatial
                arrangement.
              </p>
              <p>
                <strong>For spatial control:</strong> Use ControlNet with a
                depth map of the room. Depth preserves the 3D layout. Combine
                ControlNet (layout) + IP-Adapter (visual identity) for both
                WHERE and WHAT-IT-LOOKS-LIKE.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Semantic, Not Spatial">
            IP-Adapter captures &ldquo;what things look like,&rdquo; not
            &ldquo;where things are.&rdquo; For spatial control, you still need
            ControlNet. This is the critical distinction for the module.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Misconception #1: Image prompting is not image replacement */}
      <Row>
        <Row.Content>
          <WarningBlock title="Image Prompting Is Not Image Replacement">
            It is tempting to think IP-Adapter replaces the text prompt with an
            image. It does not. The word &ldquo;image prompting&rdquo; suggests
            substitution, but the mechanism is{' '}
            <strong>addition</strong>. The text K/V path is untouched. The image
            K/V path runs in parallel. The same reference image with two
            different text prompts produces clearly different outputs—the image
            provides semantic flavor while text still controls content and
            composition. You saw this with the golden retriever example above.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Misconception #3: General-purpose, not per-concept */}
      <Row>
        <Row.Content>
          <WarningBlock title="Trained Once, Works With Any Image">
            IP-Adapter is <strong>not</strong> trained per-concept like textual
            inversion. It is trained once on millions of image-text pairs (the
            same kind of data CLIP was trained on). The K/V projections learn to
            extract general visual features from any image. At inference, you
            feed any reference image—a photograph, a painting, a product
            shot—and the adapter extracts its visual character. Textual
            inversion: one training per concept. LoRA: one training per
            style/subject. IP-Adapter: one training, any image.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Style transfer example */}
      <Row>
        <Row.Content>
          <GradientCard title="Style Without Words" color="blue">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Reference image:</strong> A dramatic sunset photograph
                (no subject—just sky, clouds, warm colors).
              </p>
              <p>
                <strong>Text prompt:</strong> &ldquo;a cat sitting in a
                field&rdquo;
              </p>
              <p>
                <strong>Expected output:</strong> A cat in a field, but the
                entire scene is bathed in the warm amber and rose tones of the
                sunset reference. The atmospheric quality, color temperature, and
                lighting mood transfer from the reference image—without the cat
                needing to be in a sunset scene. The text says &ldquo;field&rdquo;
                and the image says &ldquo;this kind of light and color.&rdquo;
              </p>
              <p className="text-muted-foreground">
                This demonstrates the most powerful use case: style transfer
                without explicit style description in text. CLIP image features
                capture more than just &ldquo;what objects are present&rdquo;—they
                encode color palette, lighting quality, atmospheric mood, and
                visual texture.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Beyond Object Identity">
            CLIP embeddings encode much more than &ldquo;this is a dog&rdquo;
            or &ldquo;this is a sunset.&rdquo; They capture the visual
            character—colors, textures, lighting, mood. IP-Adapter leverages
            this richness to transfer style and atmosphere, not just subject
            identity.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Training overview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="How IP-Adapter Is Trained"
            subtitle="Brief overview for understanding—not the focus"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You do not need to train IP-Adapter yourself. But understanding
              the training setup clarifies why the adapter works the way it
              does:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Training data:</strong> Large dataset of image-text
                pairs (the same kind of data CLIP was trained on)
              </li>
              <li>
                <strong>Loss:</strong> Same DDPM noise prediction loss as
                standard diffusion training
              </li>
              <li>
                <strong>Frozen:</strong> Entire SD model (U-Net, CLIP text
                encoder, VAE)—nothing in the original model changes
              </li>
              <li>
                <strong>Trained:</strong> Only the new W_K_image and W_V_image
                projections at every cross-attention layer, plus a small image
                projection network that adapts CLIP image features to the
                U-Net&rsquo;s expected dimensions
              </li>
              <li>
                <strong>Initialization:</strong> New projections initialized to
                produce zero output—the same &ldquo;Nothing, Then a
                Whisper&rdquo; pattern from ControlNet&rsquo;s zero convolutions
                and LoRA&rsquo;s B=0 initialization
              </li>
              <li>
                <strong>Result:</strong> ~22M trainable parameters (much smaller
                than ControlNet&rsquo;s ~300M, larger than LoRA&rsquo;s ~1M or
                textual inversion&rsquo;s 768)
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Safety Pattern">
            Zero initialization at training start is a recurring theme: zero
            convolutions (ControlNet), B=0 (LoRA), zero K/V projections
            (IP-Adapter). The principle is always the same—new components start
            contributing nothing, so the frozen model is undisturbed at the
            beginning of training.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Variants — brief mention */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Variants</strong> (for vocabulary breadth—not covered in
              detail): <strong>IP-Adapter Plus</strong> uses more CLIP image
              features for higher fidelity.{' '}
              <strong>IP-Adapter Face ID</strong> specializes in face identity
              preservation. And IP-Adapter + ControlNet can be composed—IP-Adapter
              provides WHAT-IT-LOOKS-LIKE while ControlNet provides WHERE,
              targeting different parts of the U-Net simultaneously.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Check #2 — Transfer Questions */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Questions"
            subtitle="Apply what you know to new scenarios"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1: Style Matching" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague wants to generate images that match a specific
                  painting&rsquo;s color palette and brushstroke style. They are
                  deciding between LoRA and IP-Adapter. What would you
                  recommend and why?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <strong>IP-Adapter</strong>—no training needed. Just provide
                    the painting as the reference image. LoRA would require
                    collecting training images of that painting style and
                    running a training loop. If they need this specific style
                    permanently for thousands of generations, LoRA might be
                    worth the training investment. For one-off or exploratory
                    use, IP-Adapter is immediate.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2: Composability" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Could you use IP-Adapter with a photograph as the reference
                  image AND a ControlNet with an edge map from a completely
                  different image?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes—IP-Adapter provides semantic/visual identity from the
                    reference photo via cross-attention K/V. ControlNet provides
                    spatial structure from the edge map via encoder features.
                    They target different parts of the U-Net and are composable.
                    The edge map controls WHERE things go; the reference photo
                    controls WHAT they look like.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Practice — Notebook Exercises */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Hands-On IP-Adapter"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook is where you build practical intuition—loading
                IP-Adapter, varying the scale parameter, testing text-image
                coexistence, and composing with ControlNet.
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
                  <strong>Exercise 1 (Guided): Load and Generate.</strong>{' '}
                  Load IP-Adapter and generate with a reference image + text
                  prompt. Compare output with and without IP-Adapter (scale=0
                  vs scale=0.6). Observe that text still controls composition.
                </li>
                <li>
                  <strong>Exercise 2 (Guided): Scale Sweep.</strong>{' '}
                  Same reference image, same prompt, five scale values (0.0,
                  0.3, 0.5, 0.7, 1.0). Observe the transition from
                  text-dominant to image-dominant. Connect to the conditioning
                  scale pattern from{' '}
                  <strong>ControlNet in Practice</strong>.
                </li>
                <li>
                  <strong>Exercise 3 (Supported): Text-Image Coexistence.</strong>{' '}
                  Same reference image + three different text prompts. Observe
                  that text controls content and scene while the reference image
                  controls visual character. This directly tests whether you
                  understand that image prompting is addition, not replacement.
                </li>
                <li>
                  <strong>Exercise 4 (Independent): Compose with ControlNet.</strong>{' '}
                  Use a reference image for visual style (IP-Adapter) and an
                  edge map from a different image for spatial structure
                  (ControlNet). Design your own experiment and interpret the
                  results. WHERE + WHAT-IT-LOOKS-LIKE + WHAT, all in one
                  generation.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: basic IP-Adapter functionality</li>
              <li>Guided: scale parameter behavior</li>
              <li>Supported: text-image coexistence</li>
              <li>Independent: composability with ControlNet</li>
            </ol>
            <p className="text-sm mt-2">
              Exercises are cumulative: Exercise 1&rsquo;s reference image
              carries through all four.
            </p>
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
                  'IP-Adapter adds a parallel K/V path for CLIP image embeddings.',
                description:
                  'Alongside the existing text K/V projections, IP-Adapter adds new trainable W_K_image and W_V_image at every cross-attention layer. The Q projection is shared. The two attention outputs combine via weighted addition. "Two reference documents, one reader."',
              },
              {
                headline:
                  'The text path is completely untouched—decoupled, not replaced.',
                description:
                  'No frozen weights are modified. Remove IP-Adapter and the model is bit-for-bit identical to vanilla SD. The image path is purely additive, just like ControlNet\'s zero convolution features.',
              },
              {
                headline:
                  'Three complementary conditioning channels.',
                description:
                  'Text provides WHAT (semantic content). ControlNet provides WHERE (spatial structure). IP-Adapter provides WHAT-IT-LOOKS-LIKE (visual identity). All are additive, composable, and operate on different parts of the U-Net.',
              },
              {
                headline:
                  'Same frozen-model, additive-adapter pattern throughout.',
                description:
                  'Zero initialization, purely additive, disconnect and nothing changes. ControlNet, LoRA, textual inversion, and IP-Adapter all follow this principle—they differ in where they intervene and what they control, but the safety pattern is the same.',
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
                title: 'IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models',
                authors: 'Ye et al., 2023',
                url: 'https://arxiv.org/abs/2308.06721',
                note: 'The original IP-Adapter paper. Section 3 covers the decoupled cross-attention mechanism. Figure 2 shows the architecture diagram.',
              },
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)',
                authors: 'Radford et al., 2021',
                url: 'https://arxiv.org/abs/2103.00020',
                note: 'The foundation IP-Adapter builds on. IP-Adapter uses CLIP\'s image encoder to extract the reference image embeddings.',
              },
              {
                title: 'Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)',
                authors: 'Zhang, Rao & Agrawala, 2023',
                url: 'https://arxiv.org/abs/2302.05543',
                note: 'IP-Adapter and ControlNet are composable. Understanding both enables the full WHERE + WHAT-IT-LOOKS-LIKE control spectrum.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 13: Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="7.1"
            title="Controllable Generation"
            achievements={[
              'ControlNet architecture: trainable encoder copy, zero convolutions, additive feature merging for spatial conditioning',
              'Practical ControlNet: preprocessors (Canny, depth, OpenPose), conditioning scale tuning, multi-ControlNet stacking',
              'IP-Adapter: decoupled cross-attention with parallel K/V projections for image-based semantic conditioning',
              'The complete controllable generation spectrum: WHEN (timestep) + WHAT (text) + WHERE (ControlNet) + WHAT-IT-LOOKS-LIKE (IP-Adapter)',
            ]}
            nextModule="7.2"
            nextTitle="Efficient Architectures"
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Module 7.1 Complete"
            description="You have learned three controllable generation techniques that compose with a frozen SD model: ControlNet for spatial structure, IP-Adapter for visual identity, and text for semantic content. Each is additive, each targets a different dimension of control, and all three can work together."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
