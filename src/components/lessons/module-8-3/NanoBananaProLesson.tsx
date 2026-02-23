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
  PhaseCard,
  SummaryBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'

/**
 * Nano Banana Pro (Gemini 3 Pro Image) — Architecture Analysis
 *
 * Lesson 1 in Module 8.3 (Architecture Analysis). Series 8 Special Topics.
 * Cognitive load: STRETCH (2 new concepts + analytical reasoning framework).
 *
 * Teaches the student to analyze how Nano Banana Pro likely works by reasoning
 * from observable behavior, disclosed architecture fragments, and published
 * precedents. In doing so, introduces two new concepts:
 * - Discrete visual tokenization (VQ-VAE/ViT-VQGAN codebook approach): INTRODUCED
 * - Autoregressive image generation (generating image tokens sequentially like text): INTRODUCED
 *
 * Core analogy: "Writing a letter vs painting a mural"
 * Epistemic framing: hypothesis vs fact, disclosed vs inferred vs open
 *
 * Previous: Image Generation Safety (module 8.2, lesson 1)
 * Next: Standalone (no specific next lesson)
 */

export function NanoBananaProLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Nano Banana Pro"
            description="How Google&rsquo;s image generator likely works&mdash;and why autoregressive generation is a viable alternative to diffusion that excels at text rendering and compositional reasoning."
            category="Architecture Analysis"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how autoregressive image generation works as a paradigm
            alternative to diffusion, why it excels at text rendering, and how
            to construct an informed architectural hypothesis for a production
            system from observable behavior, disclosed fragments, and published
            precedents.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            From <LessonLink slug="building-nanogpt">Building NanoGPT</LessonLink>,
            you know the autoregressive generate() loop. From{' '}
            <LessonLink slug="the-diffusion-idea">The Diffusion Idea</LessonLink> through{' '}
            <LessonLink slug="sampling-and-generation">Sampling and Generation</LessonLink>,
            you know the full diffusion pipeline. From{' '}
            <LessonLink slug="variational-autoencoders">Variational Autoencoders</LessonLink>,
            you know continuous latent spaces. All of these carry over directly.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Understanding autoregressive image generation as a paradigm alternative to diffusion',
              'Discrete visual tokenization (VQ-VAE/codebook) at intuition level',
              'Constructing a plausible architectural hypothesis from evidence',
              'Why autoregressive excels at text rendering and compositional reasoning',
              'NOT: implementing VQ-VAE or any image tokenizer from scratch',
              'NOT: training an autoregressive image generator',
              'NOT: the mathematical details of VQ-VAE training (commitment loss, codebook collapse)',
              'NOT: a comprehensive survey of all autoregressive image generation methods',
            ]}
          />
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Epistemic Honesty">
            Google has not published a comprehensive paper on Nano Banana Pro.
            This lesson constructs <strong>informed hypotheses</strong>, not
            confirmed facts. We will be explicit about what is disclosed,
            what is inferred, and what remains open.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: Hook — The Text Rendering Mystery */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Text Rendering Mystery"
            subtitle="Something structurally different is happening"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Diffusion models have been bad at text rendering for years.
              DALL-E 3, Midjourney, Stable Diffusion&mdash;all struggle with
              spelling. Ask for &ldquo;HELLO WORLD&rdquo; in a sunset scene and
              you get garbled characters, missing letters, invented glyphs.
              This is not a training data problem&mdash;it is a{' '}
              <strong>structural</strong> limitation of how diffusion works.
            </p>
            <p className="text-muted-foreground">
              Then Google shipped Nano Banana Pro with 94%+ text rendering
              accuracy. Not &ldquo;pretty good for an AI&rdquo;&mdash;legible
              fonts, correct spelling, proper kerning. What changed?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Puzzle">
            If every diffusion model struggles with text, and this model does
            not, then this model is probably not using diffusion for
            generation&mdash;at least not in the way you have seen before.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Diffusion Models',
              color: 'amber',
              items: [
                'Text rendering: garbled, missing letters',
                'Refines all pixels simultaneously',
                'No explicit awareness of character sequence',
                'Text injected via cross-attention conditioning',
              ],
            }}
            right={{
              title: 'Nano Banana Pro',
              color: 'emerald',
              items: [
                'Text rendering: 94%+ accuracy',
                'Mandatory "thinking" step before generation',
                'Native multimodal (text + images in one model)',
                'Under 10 seconds for high-resolution output',
              ],
            }}
          />
          <div className="mt-4 space-y-4">
            <p className="text-muted-foreground">
              These behavioral differences are clues. A mandatory thinking step,
              native multimodality, and accurate text rendering all point to a
              fundamentally different generation paradigm. The rest of this
              lesson is the detective work: figuring out what that paradigm is.
            </p>
            <p className="text-muted-foreground">
              You might think that without a published paper, we are stuck
              guessing. We are not. Observable behavior, disclosed architecture
              fragments, and published precedents are not guesses&mdash;they are{' '}
              <strong>constraints that eliminate most possible architectures</strong>.
              By the end of this lesson, you will have a specific, falsifiable
              hypothesis constructed from evidence&mdash;not speculation.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 4: What Google Has Disclosed */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What We Actually Know"
            subtitle="Disclosed facts vs informed inference"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before speculating, let us separate what Google has actually
              disclosed from what we can infer. This distinction matters&mdash;it
              is the difference between a hypothesis and a fact.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Evidence Categories">
            <ul className="space-y-1 text-sm">
              <li>&bull; <strong>Disclosed:</strong> Google stated it publicly</li>
              <li>&bull; <strong>Observable:</strong> Anyone can verify by using the model</li>
              <li>&bull; <strong>Inferred:</strong> Follows logically from disclosed + precedent</li>
              <li>&bull; <strong>Open:</strong> Could go either way</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <GradientCard title="Disclosed by Google" color="blue">
              <ul className="space-y-1">
                <li>&bull; Built on the Gemini 3 Pro backbone (decoder-only transformer)</li>
                <li>&bull; Uses &ldquo;GemPix 2&rdquo; rendering engine</li>
                <li>&bull; &ldquo;Reasoning-guided synthesis&rdquo;&mdash;the model plans before rendering</li>
                <li>&bull; Autoregressive image token generation</li>
                <li>&bull; Token efficiency: ~1,120 tokens for 2K output</li>
                <li>&bull; Multi-stage internal pipeline with self-correction</li>
              </ul>
            </GradientCard>
            <GradientCard title="Observable Behavior" color="emerald">
              <ul className="space-y-1">
                <li>&bull; 94%+ text rendering accuracy</li>
                <li>&bull; Mandatory &ldquo;thinking&rdquo; step (visible in the UI)</li>
                <li>&bull; Under 10 seconds for high-resolution images</li>
                <li>&bull; Multi-image consistency (up to 5 people across 14 input images)</li>
                <li>&bull; Native image editing within the same conversation</li>
                <li>&bull; Physics-accurate lighting and reflections</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Key Disclosed Fact">
            &ldquo;Autoregressive image token generation&rdquo; is the most
            important disclosure. It tells us the generation paradigm is
            fundamentally different from the diffusion models you have studied.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Gap Resolution — Discrete Visual Tokenization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Missing Piece: Discrete Visual Tokenization"
            subtitle="From continuous latent vectors to discrete integer tokens"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Google says Nano Banana Pro generates &ldquo;image tokens
              autoregressively.&rdquo; You know autoregressive generation from{' '}
              <LessonLink slug="building-nanogpt">Building NanoGPT</LessonLink>:
              predict the next token, append it, repeat. But that loop generates{' '}
              <strong>discrete integer tokens</strong> from a vocabulary. How do
              you turn an image into discrete integers?
            </p>
            <p className="text-muted-foreground">
              You already know most of the answer. In{' '}
              <LessonLink slug="variational-autoencoders">Variational Autoencoders</LessonLink>,
              you built a VAE that encodes images to <strong>continuous
              vectors</strong> in a latent space. VQ-VAE (Vector Quantized VAE)
              takes one more step: it maps each encoder output to the{' '}
              <strong>nearest vector in a learned codebook</strong>, producing
              a discrete integer index.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Bridge">
            You know VAE: image &rarr; continuous vector. VQ-VAE extends
            this: image &rarr; continuous vector &rarr; nearest codebook
            entry &rarr; <strong>discrete integer</strong>. Same
            encoder-decoder structure, but the bottleneck is discrete.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The codebook analogy
            </p>
            <p className="text-muted-foreground">
              A text tokenizer has a vocabulary of ~50,000 words/subwords. Each
              input word gets mapped to an integer ID. A visual tokenizer has a{' '}
              <strong>codebook</strong> of ~8,000&ndash;100,000 learned visual
              patterns. Each spatial region of the image gets mapped to the
              nearest codebook entry&rsquo;s integer ID.
            </p>
            <ComparisonRow
              left={{
                title: 'Text Tokenization',
                color: 'blue',
                items: [
                  '"a cat on a mat"',
                  '→ [64, 2815, 319, 257, 2420]',
                  'Vocabulary: ~50,000 entries',
                  'Each token: a word or subword',
                ],
              }}
              right={{
                title: 'Image Tokenization (VQ-VAE)',
                color: 'violet',
                items: [
                  '256×256 cat photo',
                  '→ [4821, 1293, 7744, 502, ...]',
                  'Codebook: ~8,000–100,000 entries',
                  'Each token: a visual pattern',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Concrete example: a 256&times;256 image passes through the VQ-VAE
              encoder, producing a 32&times;32 grid of continuous vectors. Each
              vector is matched to its nearest codebook entry, yielding
              32&times;32 = 1,024 integer tokens. The image is now a sequence
              of discrete integers&mdash;structurally identical to a tokenized
              sentence.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Do NOT Need">
            You do NOT need to understand how VQ-VAE is trained (commitment
            loss, codebook collapse, straight-through estimator). The key
            idea is: <strong>images in, discrete integers out</strong>. Same
            as text tokenization.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph LR
              A["256×256 Image"] --> B["VQ-VAE Encoder"]
              B --> C["32×32 Continuous Vectors"]
              C --> D["Codebook Lookup"]
              D --> E["1,024 Integer Tokens"]
              E --> F["Same format as text tokens"]
          `} />
          <div className="mt-4">
            <p className="text-muted-foreground">
              <strong>ViT-VQGAN</strong> is Google&rsquo;s specific visual
              tokenizer: a Vision Transformer encoder paired with a VQGAN
              codebook. Google&rsquo;s Parti model (2022) used ViT-VQGAN. Nano
              Banana Pro likely uses an evolved version of this tokenizer,
              possibly the &ldquo;GemPix 2&rdquo; component.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="ViT-VQGAN vs DiT Patchify">
            In <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>,
            you learned patchify: &ldquo;tokenize the image&rdquo; into
            continuous embeddings for a diffusion transformer. VQ-VAE
            tokenization produces <strong>discrete integers</strong> for an
            autoregressive transformer. Same intuition (&ldquo;images as
            token sequences&rdquo;), different representation (continuous vs
            discrete).
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Published Precedents */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Published Precedents"
            subtitle="The research trail leading to Nano Banana Pro"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Nano Banana Pro did not emerge from nothing. There is a clear
              research lineage of autoregressive image generation that
              constrains our hypothesis.
            </p>
            <div className="space-y-3">
              <GradientCard title="Google Parti (2022)" color="blue">
                <p>
                  Proved autoregressive can match diffusion quality at scale.
                  Used ViT-VQGAN for visual tokenization (8&times;8 patches
                  producing 1,024 tokens for a 256&times;256 image) + a 20B
                  parameter encoder-decoder transformer. The direct ancestor of
                  Nano Banana Pro&rsquo;s approach.
                </p>
              </GradientCard>
              <GradientCard title="Chameleon (Meta, 2024)" color="violet">
                <p>
                  Unified autoregressive model for text + images in the same
                  token stream. Early-fusion multimodal&mdash;text and image
                  tokens are interleaved in one sequence. Proved the concept of
                  a single model generating both modalities.
                </p>
              </GradientCard>
              <GradientCard title="DALL-E 1 → DALL-E 2/3 Trajectory" color="amber">
                <p>
                  DALL-E 1 (2021) was autoregressive. OpenAI switched to
                  diffusion for DALL-E 2 and 3 because diffusion produced{' '}
                  <strong>visibly better quality</strong> at 2021-era scale
                  (~12B parameters)&mdash;sharper details, more coherent
                  compositions, better fine textures. DALL-E 1&rsquo;s images
                  looked noticeably worse than DALL-E 2&rsquo;s. Autoregressive
                  was not always viable. It returns now at 2025-era scale
                  (hundreds of billions of parameters), where the quality gap
                  closes&mdash;the same pattern as &ldquo;convolutions vs
                  attention&rdquo; from{' '}
                  <LessonLink slug="diffusion-transformers">Diffusion Transformers</LessonLink>.
                  This does not mean autoregressive is strictly better than
                  diffusion&mdash;diffusion still excels at fine-grained texture
                  details and high-resolution spatial coherence at smaller model
                  scales. The paradigm that wins depends on scale and task.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Scale Changes Winners">
            DALL-E 1&rsquo;s autoregressive approach lost to diffusion in 2021.
            At 2025 scale (Gemini 3 Pro backbone), autoregressive quality
            matches or exceeds diffusion. The same pattern you saw with
            convolutions vs attention: scale changes which approach dominates.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 7: Core Explanation — Two Paradigms */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Paradigms for Image Generation"
            subtitle="Writing a letter vs painting a mural"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now you have the pieces to understand the paradigm split.
              You have spent the course building two tracks of understanding:
              how transformers generate text token-by-token, and how diffusion
              models generate images by iteratively denoising. These tracks
              have never crossed&mdash;until now.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Diffusion: Painting a Mural',
              color: 'amber',
              items: [
                'Start with pure noise (blank canvas)',
                'Iteratively refine ALL pixels simultaneously',
                '20–50 denoising steps, each refines the whole image',
                'No concept of "earlier" or "later" pixels',
                'Text injected via cross-attention or joint attention',
              ],
            }}
            right={{
              title: 'Autoregressive: Writing a Letter',
              color: 'blue',
              items: [
                'Start with text prompt tokens',
                'Generate image tokens ONE AT A TIME',
                'Each token conditioned on all previous tokens',
                'Explicit ordering: earlier tokens are "decided"',
                'Text and image tokens share the same sequence',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Analogy">
            <strong>Autoregressive</strong> = writing a letter. Each word
            follows the last. You can reference what you already wrote.
            Quoting text is natural&mdash;you write one character at a time.
            <br /><br />
            <strong>Diffusion</strong> = painting a mural. You rough in the
            whole scene, then refine everywhere simultaneously. Text on the
            mural requires careful planning because you paint all letters at
            once.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              Side-by-side: the generation loops you already know
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <CodeBlock
                  code={`# Diffusion sampling loop
# (from Sampling and Generation)
x = torch.randn(1, C, H, W)  # pure noise
for t in reversed(range(T)):
    # Predict noise in ENTIRE image
    noise_pred = model(x, t, text_emb)
    # Refine ALL pixels one step
    x = denoise_step(x, noise_pred, t)
# x is the final image`}
                  language="python"
                  filename="diffusion_sampling.py"
                />
              </div>
              <div>
                <CodeBlock
                  code={`# Autoregressive image generation
# (same loop as GPT generate())
tokens = tokenize(prompt)  # text tokens
for i in range(num_image_tokens):
    # Forward pass on all tokens so far
    logits = model(tokens)
    # Sample NEXT image token
    next_token = sample(logits[-1])
    tokens = cat(tokens, next_token)
# Decode image tokens to pixels`}
                  language="python"
                  filename="autoregressive_image.py"
                />
              </div>
            </div>
            <p className="text-muted-foreground">
              Look at the autoregressive loop. You already wrote this code
              in <LessonLink slug="building-nanogpt">Building NanoGPT</LessonLink>.
              The only difference is the vocabulary&mdash;instead of sampling
              from ~50,000 text tokens, you sample from ~8,000&ndash;100,000
              visual tokens. The generation mechanics are identical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="You Already Wrote This Code">
            The autoregressive image generation loop is structurally identical
            to the GPT generate() method from{' '}
            <LessonLink slug="building-nanogpt">Building NanoGPT</LessonLink>.
            Forward pass, take last position logits, apply temperature, sample,
            append. The only change is what the tokens represent.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Partial generation comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The partial generation test: a concrete way to see the difference
            </p>
            <ComparisonRow
              left={{
                title: 'Stop Diffusion at 50% Steps',
                color: 'amber',
                items: [
                  'You see the ENTIRE image',
                  'But it is blurry and noisy everywhere',
                  'Global structure visible, details missing',
                  'Every pixel has been partially refined',
                ],
              }}
              right={{
                title: 'Stop Autoregressive at 50% Tokens',
                color: 'blue',
                items: [
                  'You see the top half of the image, sharp and complete',
                  'Raster scan order: left-to-right, top-to-bottom (like reading)',
                  'The bottom half is blank/missing',
                  'Tokens generated so far are final',
                ],
              }}
            />
            <p className="text-muted-foreground">
              This reveals the structural difference: diffusion builds the image
              like developing a photograph (everything emerges simultaneously),
              autoregressive builds it like a printer (each line is final as it
              is laid down).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Diffusion with a Different Name">
            Autoregressive image generation is NOT &ldquo;diffusion with a
            different name.&rdquo; Diffusion refines ALL pixels simultaneously
            from noise. Autoregressive generates tokens one at a time in
            sequence order. Fundamentally different generation mechanics.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The "of course" chain */}
      <Row>
        <Row.Content>
          <GradientCard title="The &ldquo;Of Course&rdquo; Chain" color="orange">
            <div className="space-y-2">
              <p>Follow the logic from what you already know:</p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Images can be tokenized into discrete integers (VQ-VAE codebook)</li>
                <li>You have a transformer that generates tokens autoregressively (GPT)</li>
                <li><em>Of course</em> you can generate images the same way you generate text</li>
                <li><em>Of course</em> text rendering works&mdash;each character is just another token in the sequence, generated with full awareness of what came before</li>
              </ol>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Not the Same as MMDiT">
            In <LessonLink slug="sd3-and-flux">SD3 and Flux</LessonLink>,
            text and image tokens share attention (&ldquo;one room, one
            conversation&rdquo;)&mdash;but generation is still diffusion,
            refining all tokens simultaneously. Here, text and image tokens
            share the same sequence too, but generation is{' '}
            <strong>autoregressive</strong>&mdash;one token at a time. Same
            attention mechanism, different generation paradigm.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Check #1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> If Nano Banana Pro generates image
                tokens autoregressively, what happens when you ask it to
                generate TWO images in one conversation?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    The second image&rsquo;s tokens are generated with the
                    first image&rsquo;s tokens still in the context window.
                    This is how it maintains multi-image consistency for up
                    to 5 people across 14 input images&mdash;everything stays
                    in context, just like a long conversation.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> Why does autoregressive generation
                naturally handle text rendering while diffusion struggles?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Think about it: when generating the visual tokens for the
                    letter &ldquo;E,&rdquo; what does the model &ldquo;see&rdquo;
                    in its context window? The text prompt AND the
                    already-rendered &ldquo;H.&rdquo; The next section develops
                    this structural argument in detail.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 9: The Inferred Architecture */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Architectural Hypothesis"
            subtitle="Putting the pieces together into a falsifiable picture"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Combining disclosed facts, observable behavior, and published
              precedents, here is the most plausible hypothesis for how Nano
              Banana Pro works:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Hypothesis, Not Fact">
            Everything in this section is an informed inference. It could be
            wrong in specific details. The value is in the reasoning process,
            not in being right about every architectural choice.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <PhaseCard number={1} title="User Prompt" subtitle="Text enters the Gemini 3 Pro backbone" color="cyan">
              <p>
                The user&rsquo;s text prompt enters the same decoder-only
                transformer that handles all Gemini tasks. No separate image
                generation model&mdash;the LLM itself is the image generator.
              </p>
            </PhaseCard>
            <PhaseCard number={2} title="Thinking Step" subtitle="Composition planning before committing to pixels" color="blue">
              <p>
                The model generates reasoning tokens (visible as
                &ldquo;thinking&rdquo; in the UI) that decompose the prompt:
                spatial layout, object placement, text positioning. Analogous
                to chain-of-thought from{' '}
                <LessonLink slug="chain-of-thought">Chain of Thought</LessonLink>.
              </p>
            </PhaseCard>
            <PhaseCard number={3} title="Image Token Generation" subtitle="Autoregressive generation from a visual codebook" color="violet">
              <p>
                The transformer generates ~1,120 discrete image tokens
                autoregressively, each conditioned on the text prompt, the
                thinking tokens, and all previously generated image tokens.
                Same generate() loop as text, different vocabulary.
              </p>
            </PhaseCard>
            <PhaseCard number={4} title="GemPix 2 Decoding" subtitle="Discrete tokens back to high-resolution pixels" color="emerald">
              <p>
                GemPix 2 (likely an evolved ViT-VQGAN decoder) converts the
                discrete token sequence back into a high-resolution image.
                &ldquo;GemPix&rdquo; = Gemini + Pixels&mdash;the bridge
                between the LLM&rsquo;s token space and pixel space.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <MermaidDiagram chart={`
            graph TD
              A["User Prompt"] --> B["Gemini 3 Pro Backbone"]
              B --> C["Thinking Tokens (composition planning)"]
              C --> D["Autoregressive Image Token Generation (~1,120 tokens)"]
              D --> E["GemPix 2 Decoder"]
              E --> F["High-Resolution Image"]
              style A fill:#1e293b,stroke:#6366f1,color:#e2e8f0
              style B fill:#1e293b,stroke:#6366f1,color:#e2e8f0
              style C fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
              style D fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
              style E fill:#1e293b,stroke:#22c55e,color:#e2e8f0
              style F fill:#1e293b,stroke:#22c55e,color:#e2e8f0
          `} />
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="What Is GemPix 2?">
            Google&rsquo;s name for the visual tokenizer + decoder component.
            Likely an evolved ViT-VQGAN: a visual tokenizer that maps between
            images and discrete tokens. The token efficiency (~1,120 tokens
            for 2K output) suggests a highly optimized tokenizer&mdash;each
            token represents a large spatial region.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 10: Why Text Rendering Works */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Text Rendering Works"
            subtitle="The architectural argument, not magic"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now you can explain <em>why</em> Nano Banana Pro renders text
              accurately, without invoking any specialized text-detection module.
            </p>
            <ComparisonRow
              left={{
                title: 'Diffusion: Text via Cross-Attention',
                color: 'amber',
                items: [
                  'Text embeddings condition denoising of ALL pixels',
                  'Model must learn spatial placement of characters',
                  'Refines all character regions simultaneously',
                  'No explicit character ordering during generation',
                  'Like painting "HELLO" on a mural—all letters at once',
                ],
              }}
              right={{
                title: 'Autoregressive: Text in Context',
                color: 'emerald',
                items: [
                  'Text instruction in the context window',
                  'Image tokens for "H" region generated first',
                  'Then "E" tokens with "H" already rendered in context',
                  'Each character aware of all previous characters',
                  'Like typing "HELLO"—one character at a time',
                ],
              }}
            />
            <p className="text-muted-foreground">
              This is the same reason GPT can spell words: it generates one
              token at a time, each conditioned on the previous ones. No
              specialized text module needed. The architecture{' '}
              <strong>inherently</strong> handles text because autoregressive
              generation IS sequential, conditional generation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="No Special Module">
            The misconception: text rendering must require a specialized
            text-detection component. The reality: autoregressive generation
            is inherently sequential and conditional. Each visual token is
            generated with awareness of everything before it&mdash;including
            the text instruction and already-rendered characters.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: The Thinking Step */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Thinking Step"
            subtitle="Why composition planning is architecturally motivated"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Nano Banana Pro has a mandatory &ldquo;thinking&rdquo; step before
              image generation. This is not marketing. It is architecturally
              motivated by a specific limitation of autoregressive generation.
            </p>
            <p className="text-muted-foreground">
              Consider the prompt: &ldquo;a red car in front of a blue house
              with the text &lsquo;SOLD&rsquo; on a sign.&rdquo; Without
              planning, the model must commit to early tokens (sky, background)
              before knowing where the car, house, and sign will go. With
              planning, it can decompose:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Objects: car, house, sign</li>
              <li>Properties: red car, blue house, &ldquo;SOLD&rdquo; text</li>
              <li>Spatial layout: car in foreground, house behind, sign on the house</li>
              <li>Generation order: sky/background first, house, car in front, sign last</li>
            </ul>
            <p className="text-muted-foreground">
              This is analogous to chain-of-thought reasoning from{' '}
              <LessonLink slug="chain-of-thought">Chain of Thought</LessonLink>:
              decompose the problem before committing to the answer. For
              autoregressive generation specifically, planning matters because
              early tokens are <strong>irrevocable</strong> (self-correction
              happens at the pipeline level&mdash;regenerate the whole sequence,
              not edit mid-stream)&mdash;you cannot go back and change the sky
              after you have rendered the foreground.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why Diffusion Needs This Less">
            Diffusion refines the whole image simultaneously in every step.
            It does not commit to any region first&mdash;the composition
            emerges from iterative refinement. Autoregressive generation
            commits to early tokens permanently, making upfront planning
            much more valuable.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Check #2 */}
      <Row>
        <Row.Content>
          <GradientCard title="Check Your Understanding" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>Question 1:</strong> Google says Nano Banana Pro uses
                ~1,120 tokens for 2K output. Recall how many tokens Parti used
                for 256&times;256 output. What does the comparison tell us about
                Nano Banana Pro&rsquo;s tokenizer?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Generating 2K resolution with only ~1,120 tokens means each
                    token represents a <em>much</em> larger spatial region than
                    Parti&rsquo;s. Likely a hierarchical or more efficient
                    tokenizer, or the ~1,120 tokens encode an intermediate
                    resolution that gets upsampled by GemPix 2.
                  </p>
                </div>
              </details>

              <p className="mt-4">
                <strong>Question 2:</strong> If Nano Banana Pro edits images
                within the same model, how might that work architecturally?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    Encode the input image to tokens, place those tokens in
                    the context alongside the edit instruction, then generate
                    new image tokens conditioned on both. Same autoregressive
                    mechanism, different input context.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 13: Speed, Efficiency, and the Hybrid Question */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Speed, Efficiency, and the Hybrid Question"
            subtitle="How autoregressive can be fast, and what we do not know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A natural objection: &ldquo;Sequential generation must be slower
              than diffusion because it generates tokens one at a time.&rdquo;
              But Nano Banana Pro generates high-resolution images in under 10
              seconds. How?
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="KV Caching" color="blue">
                <p>
                  From <LessonLink slug="scaling-and-efficiency">Scaling and Efficiency</LessonLink>,
                  you know KV caching: each new token only needs one forward
                  pass through the transformer, reusing cached keys and values
                  from previous tokens. 1,120 tokens = 1,120 forward passes,
                  not 1,120 full sequence recomputations.
                </p>
              </GradientCard>
              <GradientCard title="Token Efficiency" color="violet">
                <p>
                  Compare with diffusion: 20&ndash;50 full forward passes
                  through a DiT, each processing ~1,000&ndash;4,000 latent
                  tokens. With autoregressive: 1,120 forward passes (one per
                  token), but each only computes one new token&rsquo;s logits
                  thanks to KV caching. The total computation is in a similar
                  ballpark&mdash;the tradeoff is parallel processing of all
                  tokens per step (diffusion) vs sequential processing of one
                  token per step with caching (autoregressive).
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Sequential Does Not Mean Slow">
            KV caching + efficient tokenization + fewer total steps can make
            autoregressive competitive with or faster than diffusion. The
            bottleneck is not &ldquo;sequential vs parallel&rdquo;&mdash;it
            is total computation.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Hybrid landscape */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-sm font-medium text-foreground">
              The hybrid landscape: purely autoregressive or something more?
            </p>
            <p className="text-muted-foreground">
              Is Nano Banana Pro purely autoregressive, or does it use a
              hybrid approach? This is an open question.
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Evidence for Purely Autoregressive" color="emerald">
                <ul className="space-y-1">
                  <li>&bull; Google says &ldquo;autoregressive image token generation&rdquo;</li>
                  <li>&bull; Mandatory thinking step (not typical for diffusion)</li>
                  <li>&bull; Gemini backbone is a decoder-only transformer</li>
                  <li>&bull; Text rendering accuracy (autoregressive advantage)</li>
                </ul>
              </GradientCard>
              <GradientCard title="Evidence for Possible Hybrid" color="amber">
                <ul className="space-y-1">
                  <li>&bull; Under 10s for 4K (fast for pure autoregressive)</li>
                  <li>&bull; Physics-accurate lighting (diffusion excels at this)</li>
                  <li>&bull; HART and BLIP3o-NEXT show hybrid is viable</li>
                  <li>&bull; GemPix 2 could include diffusion refinement</li>
                </ul>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Recent work like <strong>HART</strong> (autoregressive for global
              structure, diffusion refinement for local detail) and{' '}
              <strong>BLIP3o-NEXT</strong> (autoregressive tokens conditioning a
              small diffusion model) suggests a spectrum between pure
              autoregressive and pure diffusion. Best hypothesis: primarily
              autoregressive with possible diffusion refinement in GemPix 2.
              But honestly&mdash;we do not know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Honest Answer">
            &ldquo;We do not know&rdquo; is a valid conclusion. The value of
            this analysis is not in being right about every detail&mdash;it is
            in knowing which questions to ask, what evidence constrains the
            answer, and where genuine uncertainty remains.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 14: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Two paradigms for image generation: diffusion refines all pixels from noise, autoregressive generates tokens sequentially.',
                description:
                  'Diffusion is like painting a mural\u2014rough in the whole scene, refine everywhere simultaneously. Autoregressive is like writing a letter\u2014each token follows the last, conditioned on everything before it.',
              },
              {
                headline:
                  'Autoregressive image generation requires discrete visual tokenization.',
                description:
                  'VQ-VAE/ViT-VQGAN maps images to discrete integer tokens\u2014the same format as text tokens. Once images are discrete tokens, the same GPT generate() loop produces images.',
              },
              {
                headline:
                  'Text rendering works because autoregressive generation is inherently sequential and conditional.',
                description:
                  'Each character\u2019s visual tokens are generated with awareness of the text prompt and already-rendered characters\u2014like typing. No specialized text module needed.',
              },
              {
                headline:
                  'Nano Banana Pro likely: Gemini 3 Pro backbone generates image tokens autoregressively, GemPix 2 decodes to pixels.',
                description:
                  'Mandatory thinking step provides composition planning. Possibly hybrid with diffusion refinement. This is an informed hypothesis, not a confirmed architecture.',
              },
              {
                headline:
                  'Architectural analysis is possible without a paper: observable behavior + disclosed fragments + published precedents = informed, falsifiable hypotheses.',
                description:
                  'The value is in the reasoning process\u2014knowing which questions to ask, what evidence constrains the answer, and where genuine uncertainty remains.',
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
              &ldquo;Convergence, not revolution.&rdquo; Autoregressive image
              generation combines transformers (Series 4) + visual tokenization
              (extending Series 6 VAE) + the scaling that makes it work. Nano
              Banana Pro is not alien technology&mdash;it is components you
              already understand, assembled in a different configuration.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Scaling Autoregressive Models for Content-Rich Text-to-Image Generation',
                authors: 'Yu et al., 2022 (Google Research)',
                url: 'https://arxiv.org/abs/2206.10789',
                note: 'The Parti paper. Proved autoregressive can match diffusion quality at scale. The direct ancestor of Nano Banana Pro\u2019s approach.',
              },
              {
                title: 'Vector-quantized Image Modeling with Improved VQGAN',
                authors: 'Yu et al., 2021 (Google Research)',
                url: 'https://arxiv.org/abs/2110.04627',
                note: 'The ViT-VQGAN paper. Google\u2019s visual tokenizer that maps images to discrete tokens. Section 3 explains the codebook mechanism.',
              },
              {
                title: 'Zero-Shot Text-to-Image Generation',
                authors: 'Ramesh et al., 2021 (OpenAI)',
                url: 'https://arxiv.org/abs/2102.12092',
                note: 'The DALL-E 1 paper. The original autoregressive text-to-image model. Interesting to compare with Nano Banana Pro\u2019s approach.',
              },
              {
                title: 'Chameleon: Mixed-Modal Early-Fusion Foundation Models',
                authors: 'Team et al., 2024 (Meta)',
                url: 'https://arxiv.org/abs/2405.09818',
                note: 'Unified autoregressive model for text + images in one token stream. Demonstrates the early-fusion multimodal approach.',
              },
              {
                title: 'Autoregressive Image Generation without Vector Quantization',
                authors: 'Tang et al., 2024',
                url: 'https://arxiv.org/abs/2406.11838',
                note: 'The HART paper. Hybrid autoregressive + diffusion refinement. The title refers to HART using continuous tokens rather than discrete VQ codebook tokens\u2014it generates continuous latent tokens autoregressively and uses a small diffusion model to refine residual detail, avoiding the quality ceiling of discrete quantization. Section 3 explains why hybrids capture advantages of both paradigms.',
              },
              {
                title: 'BLIP3-o: A Family of Fully Open Unified Multimodal Models',
                authors: 'Xie et al., 2025 (Salesforce)',
                url: 'https://arxiv.org/abs/2505.09568',
                note: 'BLIP3o-NEXT uses autoregressive tokens to condition a diffusion model. An example of the hybrid approach.',
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
              This lesson explored how autoregressive generation offers a
              fundamentally different approach to image synthesis&mdash;one
              where the same architecture that generates text can generate
              images. The paradigm shift from diffusion to autoregressive (or
              hybrid) image generation is one of the most significant trends in
              generative AI. As more details about Nano Banana Pro emerge, you
              now have the conceptual framework to evaluate them.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Back to Dashboard"
            description="Review your learning journey and explore other topics."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
