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
  LessonLink,
} from '@/components/lessons'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { CodeBlock } from '@/components/common/CodeBlock'

/**
 * Stable Diffusion Architecture
 *
 * Lesson 1 in Module 6.4 (Stable Diffusion). Lesson 15 overall in Series 6.
 * Cognitive load: CONSOLIDATE (zero new concepts—assembles all components
 * from Modules 6.1–6.3 into one pipeline trace).
 *
 * Teaches the complete Stable Diffusion pipeline from text prompt to pixel image:
 * - Full pipeline trace with real tensor shapes at every handoff
 * - How CLIP, the conditioned U-Net, and the VAE work together
 * - The denoising loop with CFG, cross-attention, and adaptive group norm
 * - Training vs inference pipeline comparison
 * - Component parameter counts and modularity
 * - Negative prompts as a direct application of CFG
 *
 * Core concepts (all previously taught, assembled here):
 * - Full SD pipeline data flow: DEVELOPED
 * - Training vs inference pipeline: INTRODUCED
 * - Component modularity: INTRODUCED
 * - Negative prompts as CFG application: INTRODUCED
 *
 * Previous: From Pixels to Latents (Module 6.3, Lesson 5 / CONSOLIDATE)
 * Next: Samplers and Efficiency (Module 6.4, Lesson 2 / STRETCH)
 */

export function StableDiffusionArchitectureLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Stable Diffusion Pipeline"
            description="Every component you built across 14 lessons, assembled into one system. Zero new concepts&mdash;just the complete picture."
            category="Stable Diffusion"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Trace the complete Stable Diffusion pipeline from a text prompt to a
            generated image, naming every component and the tensor shape at every
            handoff. By the end, you will see that nothing in this pipeline is
            new to you.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            Over the last 14 lessons you learned: how to compress images (VAE),
            how to denoise (diffusion), the architecture that denoises (U-Net),
            how it knows the noise level (timestep conditioning), how to connect
            language to vision (CLIP), how to inject text (cross-attention), how
            to make text matter (CFG), and how to make it fast (latent
            diffusion). You have never seen them all working together.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The complete inference pipeline traced with real tensor shapes',
              'How CLIP, U-Net, and VAE connect through tensor handoffs',
              'The denoising loop with all conditioning mechanisms firing simultaneously',
              'Training vs inference comparison',
              'Component parameter counts and modularity',
              'Negative prompts as a direct application of CFG',
              'NOT: implementing the pipeline from scratch (notebook uses diffusers)',
              'NOT: samplers beyond DDPM (DDIM, Euler, DPM-Solver are next lesson)',
              'NOT: any new mathematical formulas or derivations',
              'NOT: SD v1 vs v2 vs XL differences, LoRA, or fine-tuning',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Hook — "You know every instrument" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You Know Every Instrument. Time to Hear the Symphony."
            subtitle="14 lessons of building blocks, one assembled pipeline"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have spent 14 lessons learning every component of Stable
              Diffusion from scratch. You built a VAE and explored its latent
              space. You derived the diffusion algorithm and implemented a
              pixel-space model on MNIST. You studied the U-Net architecture,
              timestep conditioning, CLIP, cross-attention, classifier-free
              guidance, and latent diffusion.
            </p>
            <p className="text-muted-foreground">
              Each lesson focused on one piece at a time. It is like learning
              every instrument in an orchestra individually&mdash;the violin, the
              oboe, the timpani&mdash;and then finally hearing the symphony.
            </p>
            <p className="text-muted-foreground">
              This lesson is the symphony. You will trace one complete generation
              from a text prompt through every component, watching real tensor
              shapes transform at each stage. Nothing in this pipeline is new.
              Every piece is something you built or deeply studied.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Prompt">
            The traced example uses <strong>&ldquo;a cat sitting on a beach at
            sunset&rdquo;</strong>&mdash;the same prompt from the cross-attention
            lesson in <LessonLink slug="text-conditioning-and-guidance">Text Conditioning &amp; Guidance</LessonLink>, where
            you saw per-spatial-location attention weights. You already know what
            the attention pattern looks like for this prompt.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4a: CLIP Stage */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stage 1: Text to Embeddings"
            subtitle="The CLIP stage transforms your prompt into numbers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The pipeline starts with your text prompt. But the U-Net cannot
              read text&mdash;it operates on tensors. CLIP is the translator.
            </p>
            <p className="text-muted-foreground">
              <strong>Tokenizer:</strong> The prompt &ldquo;a cat sitting on a
              beach at sunset&rdquo; is split into subword tokens, just like the
              BPE tokenization you learned in <LessonLink slug="tokenization">Tokenization</LessonLink>.
              CLIP&rsquo;s tokenizer adds a start-of-text (SOT) token at the
              beginning, an end-of-text (EOT) token after the last word, and
              pads the remaining positions to a fixed length of 77 tokens.
            </p>
            <p className="text-muted-foreground">
              <strong>CLIP text encoder:</strong> The transformer processes the
              77 token IDs and outputs 77 contextual embedding vectors, each 768
              dimensions. These are not simple lookup embeddings&mdash;each
              token&rsquo;s representation includes context from all other tokens
              via self-attention inside CLIP&rsquo;s transformer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Shape Checkpoint">
            <div className="space-y-1 text-sm">
              <p><strong>Input:</strong> text string</p>
              <p><strong>After tokenizer:</strong> [77] integer token IDs</p>
              <p><strong>After CLIP:</strong> [77, 768] float embeddings</p>
            </div>
            <p className="text-sm mt-2">
              This [77, 768] tensor is the <strong>only thing</strong> the rest
              of the pipeline sees. The original text is gone.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <WarningBlock title="The U-Net Never Sees Text">
            The U-Net will never see the text &ldquo;a cat sitting on a beach at
            sunset.&rdquo; It will only see a 77&times;768 tensor of
            floating-point numbers. CLIP is the translator from human language to
            the geometric representation space the U-Net operates in. If you
            changed the text encoder to produce the same 77&times;768 tensor from
            a different language, the U-Net would generate the same image. It is
            a tensor-processing machine that knows nothing about language.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Section 4b: Starting Point — Random Noise */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stage 2: The Starting Point"
            subtitle="Random noise in latent space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For text-to-image generation, the starting point
              is pure random noise in latent space:
              z<sub>T</sub>&nbsp;&sim;&nbsp;N(0,&nbsp;I) with shape{' '}
              <strong>[4, 64, 64]</strong>. There is no input image during
              inference.
            </p>
            <p className="text-muted-foreground">
              The 4 channels correspond to the VAE&rsquo;s latent dimensions.
              The 64&times;64 spatial dimensions correspond to 512&times;512
              pixels&mdash;an 8&times; downsampling factor in each spatial
              direction, the same compression you computed in{' '}
              <strong>From Pixels to Latents</strong>.
            </p>
            <p className="text-muted-foreground">
              The <strong>seed</strong> (random number generator state)
              determines z<sub>T</sub> and therefore the generated image. Same
              prompt + same seed = same image. Different seed = different image
              of the same concept.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why 4 Channels?">
            In <LessonLink slug="exploring-latent-spaces">Exploring Latent Spaces</LessonLink>, you worked with VAE
            latent vectors. Stable Diffusion&rsquo;s VAE preserves spatial
            structure: each image becomes a 4-channel 64&times;64 &ldquo;latent
            image&rdquo; instead of collapsing to a flat vector. The 4 channels
            encode color and structure information that the decoder can
            reconstruct.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 4c: The Denoising Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stage 3: The Denoising Loop"
            subtitle="Where everything happens—50 steps, 100 U-Net forward passes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the heart of the pipeline. The denoising loop iterates
              from timestep T down to 1. At each step, every conditioning
              mechanism you learned fires simultaneously:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>Timestep embedding</strong> (from{' '}
                <strong>Conditioning the U-Net</strong>): t &rarr; sinusoidal
                encoding &rarr; MLP &rarr; 512-dim vector
              </li>
              <li>
                <strong>Cross-attention</strong> (from{' '}
                <strong>Text Conditioning &amp; Guidance</strong>): text
                embeddings provide K/V at each attention resolution
              </li>
              <li>
                <strong>Adaptive group normalization</strong> (from{' '}
                <strong>Conditioning the U-Net</strong>): timestep-dependent
                &gamma; and &beta; at every residual block
              </li>
              <li>
                <strong>Classifier-free guidance</strong> (from{' '}
                <strong>Text Conditioning &amp; Guidance</strong>): two forward
                passes per step, combined with the CFG formula
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Conditioning Signals">
            Remember the mental model from{' '}
            <strong>Text Conditioning &amp; Guidance</strong>: the timestep
            tells the network <strong>WHEN</strong> (via adaptive norm). The
            text tells it <strong>WHAT</strong> (via cross-attention). CFG turns
            up the volume on the WHAT. All three happen at every step of the
            loop.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* One step expanded */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Inside One Denoising Step"
            subtitle="Five sub-operations per step—expanded"
          />
          <div className="space-y-3">
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                1. Timestep embedding
              </p>
              <p className="text-sm text-muted-foreground">
                t &rarr; sinusoidal encoding &rarr; MLP &rarr; t<sub>emb</sub>{' '}
                (512-dim vector). This is the &ldquo;conductor&rsquo;s
                score&rdquo; from <LessonLink slug="conditioning-the-unet">Conditioning the U-Net</LessonLink>,
                telling the network how much noise to expect.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                2. Unconditional U-Net pass
              </p>
              <p className="text-sm text-muted-foreground">
                unet(z<sub>t</sub>, t<sub>emb</sub>, text<sub>uncond</sub>)
                &rarr; &epsilon;<sub>uncond</sub>. The text<sub>uncond</sub> is
                the CLIP encoding of an empty string. Inside the U-Net: adaptive
                group norm uses t<sub>emb</sub> at every residual block,
                cross-attention uses the empty-string embeddings at 16&times;16
                and 32&times;32 resolutions.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                3. Conditional U-Net pass
              </p>
              <p className="text-sm text-muted-foreground">
                unet(z<sub>t</sub>, t<sub>emb</sub>, text<sub>cond</sub>)
                &rarr; &epsilon;<sub>cond</sub>. Same U-Net, same weights, same
                z<sub>t</sub>, same t<sub>emb</sub>. The only difference:
                cross-attention uses the <strong>real text
                embeddings</strong> from CLIP. This is the &ldquo;contrast
                slider&rdquo; from <LessonLink slug="text-conditioning-and-guidance">Text Conditioning &amp;
                Guidance</LessonLink>.
              </p>
            </div>
            <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                4. CFG combine
              </p>
              <p className="text-sm text-muted-foreground">
                &epsilon;<sub>cfg</sub> = &epsilon;<sub>uncond</sub> +
                w&nbsp;&times;&nbsp;(&epsilon;<sub>cond</sub> &minus;
                &epsilon;<sub>uncond</sub>). Typical w&nbsp;=&nbsp;7.5. This
                amplifies the direction the text embeddings push the prediction.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                5. Scheduler step
              </p>
              <p className="text-sm text-muted-foreground">
                Uses &epsilon;<sub>cfg</sub> and the noise schedule to compute
                z<sub>t&minus;1</sub>. This is the reverse step formula from{' '}
                <strong>Sampling and Generation</strong>.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="CFG Is Per-Step, Not Post-Processing">
            The two forward passes happen at <strong>every step</strong> of the
            loop. With 50 steps, that is 100 U-Net forward passes, not 50. CFG
            is woven into every step of the generation&mdash;it is not something
            applied after the loop finishes.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Tensor shapes inside the U-Net */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Inside the U-Net at one step, the tensor shapes follow the same
              pattern from <LessonLink slug="unet-architecture">The U-Net Architecture</LessonLink>, but with
              Stable Diffusion&rsquo;s larger channel counts:
            </p>
            <div className="space-y-2">
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Encoder path
                </p>
                <p className="text-sm text-muted-foreground">
                  z<sub>t</sub> [4, 64, 64] &rarr; 64&times;64&times;320
                  &rarr; 32&times;32&times;640 &rarr; 16&times;16&times;1280
                  &rarr; 8&times;8&times;1280
                </p>
              </div>
              <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Bottleneck
                </p>
                <p className="text-sm text-muted-foreground">
                  8&times;8&times;1280&mdash;global context. The bottleneck
                  decides <strong>WHAT</strong>.
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  Decoder path (with skip connections)
                </p>
                <p className="text-sm text-muted-foreground">
                  8&times;8&times;1280 &rarr; 16&times;16&times;1280 &rarr;
                  32&times;32&times;640 &rarr; 64&times;64&times;320 &rarr;
                  &epsilon; [4, 64, 64]. The skip connections decide{' '}
                  <strong>WHERE</strong>.
                </p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground italic">
              At 16&times;16 resolution, cross-attention produces a
              256&times;77 attention matrix (16&times;16&nbsp;=&nbsp;256 spatial
              locations, each attending to 77 text tokens). At 32&times;32, it
              is 1024&times;77. These are the shapes you predicted in{' '}
              <LessonLink slug="text-conditioning-and-guidance">Text Conditioning &amp; Guidance</LessonLink>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Bigger Numbers, Same Pattern">
            In <LessonLink slug="unet-architecture">The U-Net Architecture</LessonLink>, you traced a toy U-Net
            with channels 64 &rarr; 128 &rarr; 256 &rarr; 512. Stable
            Diffusion&rsquo;s U-Net uses 320 &rarr; 640 &rarr; 1280 &rarr;
            1280. The numbers are larger, but the pattern is identical: double
            channels, halve spatial resolution, bottleneck at the bottom, skip
            connections at every matching level.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4d: VAE Decode */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stage 4: Latent to Pixels"
            subtitle="The VAE decoder translates back to image space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              After all denoising steps, z<sub>0</sub> has shape [4, 64, 64].
              It is the denoised latent representation. The VAE decoder
              translates this back to pixel space: z<sub>0</sub> [4, 64, 64]
              &rarr; VAE decoder &rarr; image [3, 512, 512].
            </p>
            <p className="text-muted-foreground">
              This is the &ldquo;translator from latent language to pixel
              language&rdquo; from <LessonLink slug="from-pixels-to-latents">From Pixels to Latents</LessonLink>. The
              decoder runs <strong>once</strong>, after the entire denoising
              loop. It is fast compared to the 50-step (100 forward pass)
              denoising loop.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Compute Cost Breakdown">
            The denoising loop dominates: 100 U-Net forward passes (50 steps
            &times; 2 for CFG). The CLIP encoding runs once. The VAE decode
            runs once. The U-Net is where nearly all the compute goes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 4e: Pipeline Diagram */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Pipeline"
            subtitle="One diagram, every component, every tensor shape"
          />
          <MermaidDiagram chart={`
graph LR
    subgraph CLIP["CLIP (frozen, ~123M params)"]
        T["Text Prompt"] --> TK["Tokenizer"]
        TK -->|"[77] int IDs"| TE["Text Encoder"]
    end

    subgraph LOOP["Denoising Loop — U-Net (~860M params) × 50 steps"]
        ZT["z_T ~ N(0,I)<br/>[4, 64, 64]"] --> STEP["Per step:<br/>1. Embed timestep<br/>2. U-Net pass (uncond)<br/>3. U-Net pass (cond)<br/>4. CFG combine<br/>5. Scheduler step"]
        STEP --> Z0["z_0<br/>[4, 64, 64]"]
    end

    subgraph VAE["VAE Decoder (frozen, ~84M params)"]
        DEC["Decoder"] --> IMG["Image<br/>[3, 512, 512]"]
    end

    TE -->|"[77, 768]<br/>text embeddings"| STEP
    Z0 -->|"[4, 64, 64]"| DEC

    style T fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    style IMG fill:#1e293b,stroke:#22c55e,color:#e2e8f0
    style ZT fill:#1e293b,stroke:#a855f7,color:#e2e8f0
    style Z0 fill:#1e293b,stroke:#a855f7,color:#e2e8f0
`} />
          <p className="text-sm text-muted-foreground italic mt-4">
            Color-coded by component: CLIP (left), denoising loop with U-Net
            (center), VAE decoder (right). Every arrow is annotated with the
            tensor shape at that handoff. The denoising loop is the
            computational bottleneck&mdash;100 U-Net forward passes.
          </p>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Three Translators">
            Extending the analogy from <LessonLink slug="from-pixels-to-latents">From Pixels to
            Latents</LessonLink>: CLIP translates human language to
            geometric-meaning space. The U-Net generates in latent space. The
            VAE translates latent language back to pixel language. Three
            translators, one pipeline.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4f: Inference Pseudocode */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Inference Procedure"
            subtitle="Annotated pseudocode—every line is something you know"
          />
          <CodeBlock
            code={`def generate(prompt, num_steps=50, guidance_scale=7.5):
    # Stage 1: Text → embeddings (CLIP)
    token_ids = tokenizer(prompt)            # [77] int — padded to 77
    text_emb = clip_text_encoder(token_ids)  # [77, 768] float
    uncond_emb = clip_text_encoder("")       # [77, 768] float — empty string

    # Stage 2: Random starting point
    z_t = torch.randn(1, 4, 64, 64)         # [1, 4, 64, 64] — pure noise

    # Stage 3: Denoising loop
    for t in scheduler.timesteps:            # T, T-1, ..., 1
        t_emb = timestep_embed(t)            # [512] — sinusoidal + MLP

        # Two U-Net passes for CFG
        eps_uncond = unet(z_t, t_emb, uncond_emb)  # [1, 4, 64, 64]
        eps_cond   = unet(z_t, t_emb, text_emb)    # [1, 4, 64, 64]

        # CFG combine
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Scheduler step (reverse process formula)
        z_t = scheduler.step(eps, t, z_t)    # [1, 4, 64, 64]

    # Stage 4: Decode to pixels (VAE)
    image = vae.decode(z_t)                  # [1, 3, 512, 512]
    return image`}
            language="python"
            filename="stable_diffusion_inference.py"
          />
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="You Know Every Line">
            <div className="space-y-1 text-sm">
              <p>Line 3: <strong>Tokenization</strong></p>
              <p>Line 4: <strong>CLIP</strong></p>
              <p>Line 8: <strong>From Pixels to Latents</strong></p>
              <p>Line 11: <strong>Conditioning the U-Net</strong></p>
              <p>Lines 14–15: <strong>The U-Net Architecture</strong></p>
              <p>Line 18: <strong>Text Conditioning &amp; Guidance</strong></p>
              <p>Line 21: <strong>Sampling and Generation</strong></p>
              <p>Line 24: <strong>From Pixels to Latents</strong></p>
            </div>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Predict and Verify #1 */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  How many U-Net forward passes happen in a 50-step generation
                  with CFG?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    <strong>100.</strong> Two per step (one unconditional + one
                    conditional), times 50 steps. This is why CFG doubles
                    inference compute.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  At what point in the pipeline does text actually influence the
                  image?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    Inside <strong>every denoising step</strong>, via
                    cross-attention. CLIP encodes the text once at the start;
                    cross-attention applies the text embeddings at every step.
                    The text influence accumulates across all 50 steps, not just
                    once.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  If you changed the seed but kept the same prompt and settings,
                  what would change and what would stay the same?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    z<sub>T</sub> would be different random noise. Everything
                    else&mdash;CLIP encoding, guidance scale, number of
                    steps&mdash;stays the same. The result is a different image
                    of the same concept.
                  </p>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 6: One Pipeline, Three Models */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One Pipeline, Three Models"
            subtitle="Independently trained, connected by tensor handoffs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now step back and look at the pipeline from an engineering
              perspective. Stable Diffusion is not one big model trained
              end-to-end. It is three independently trained models that
              communicate through standardized tensor shapes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not End-to-End">
            Unlike most deep learning you have seen (CNNs, transformers, GPT),
            Stable Diffusion was <strong>never trained end-to-end</strong> with
            one loss function. Each component was trained separately, with a
            different loss, on different data. They were connected after
            training.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="CLIP Text Encoder" color="blue">
              <ul className="space-y-1">
                <li>&bull; ~123M parameters</li>
                <li>&bull; Trained by OpenAI</li>
                <li>&bull; 400M text-image pairs</li>
                <li>&bull; Contrastive loss</li>
                <li>&bull; <strong>Frozen</strong> in SD</li>
              </ul>
            </GradientCard>
            <GradientCard title="U-Net" color="violet">
              <ul className="space-y-1">
                <li>&bull; ~860M parameters</li>
                <li>&bull; Trained on diffusion</li>
                <li>&bull; Latent noise prediction</li>
                <li>&bull; MSE loss</li>
                <li>&bull; <strong>The only</strong> diffusion-trained component</li>
              </ul>
            </GradientCard>
            <GradientCard title="VAE" color="emerald">
              <ul className="space-y-1">
                <li>&bull; ~84M parameters</li>
                <li>&bull; Trained on reconstruction</li>
                <li>&bull; Perceptual + adversarial loss</li>
                <li>&bull; 8&times; spatial compression</li>
                <li>&bull; <strong>Frozen</strong> in SD</li>
              </ul>
            </GradientCard>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            Total: ~1.07B parameters. But they were never trained together.
          </p>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This modularity is not accidental. Because the components
              communicate through standardized tensor shapes (not shared
              weights), you can swap any component independently:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                Swap the text encoder (OpenAI CLIP vs OpenCLIP vs a different
                language model)
              </li>
              <li>
                Swap the VAE (higher-quality VAE = sharper decoded images)
              </li>
              <li>
                Swap the scheduler/sampler (DDPM, DDIM, Euler&mdash;next lesson)
              </li>
            </ul>
            <p className="text-muted-foreground">
              Each swap is possible <strong>because</strong> the components
              communicate through tensor shapes, not through shared weights or
              end-to-end gradients.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Parameter Count ≠ Importance">
            The U-Net has ~860M parameters, but it is useless without the other
            two. Without CLIP, you get unconditional generation (no text
            control). Without the VAE, you need pixel-space diffusion (48&times;
            slower). Each component is essential. The parameter count reflects
            the difficulty of the task each component performs (denoising at
            multiple scales is hard), not its importance to the pipeline.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Negative Examples */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Without CLIP',
              color: 'rose',
              items: [
                'U-Net still denoises—it does not need text',
                'Every step uses epsilon_uncond only',
                'Result: coherent images, but no control over content',
                'This is unconditional latent diffusion',
                'Exactly what you built in Build a Diffusion Model, just faster',
              ],
            }}
            right={{
              title: 'Without the VAE',
              color: 'rose',
              items: [
                'CLIP and U-Net still work together',
                'Text-guided denoising still happens',
                'But it must run in pixel space (512×512×3)',
                '48× more computation per step',
                'DDPM in pixel space—you felt this cost in the capstone',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* Negative example: raw text to U-Net */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if you tried to feed the raw text &ldquo;a cat&rdquo;
              directly to the U-Net? The U-Net expects a [77, 768] float tensor
              as the cross-attention K/V input. A text string is not a tensor.
              Even raw token IDs (integers) would not work&mdash;the U-Net needs
              dense 768-dimensional embedding vectors that encode
              visual-semantic meaning. CLIP is not optional preprocessing. It is
              the essential translation from language to the geometric
              representation space that cross-attention operates in.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Languages, One Translator">
            Remember from <LessonLink slug="clip">CLIP</LessonLink>: &ldquo;two encoders, one
            shared space&mdash;the loss creates the alignment, not the
            architecture.&rdquo; CLIP&rsquo;s contrastive training is what makes
            the 768-dimensional embeddings meaningful to the U-Net&rsquo;s
            cross-attention. Without that training, the numbers would be
            meaningless.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 7: Negative Prompts */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Negative Prompts: CFG in Action"
            subtitle="Not a new mechanism—a direct application of the formula you know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the CFG formula, &epsilon;<sub>uncond</sub> is the U-Net&rsquo;s
              prediction when conditioned on &ldquo;no text.&rdquo; By default,
              this means an empty string encoded by CLIP.
            </p>
            <p className="text-muted-foreground">
              A <strong>negative prompt</strong> replaces the empty string.
              Instead of &epsilon;<sub>uncond</sub> = unet(z<sub>t</sub>, t,
              CLIP(&ldquo;&rdquo;)), you use &epsilon;<sub>uncond</sub> =
              unet(z<sub>t</sub>, t, CLIP(&ldquo;blurry, low quality&rdquo;)).
            </p>
            <p className="text-muted-foreground">
              The CFG formula is unchanged: &epsilon;<sub>cfg</sub> =
              &epsilon;<sub>neg</sub> + w&nbsp;&times;&nbsp;(&epsilon;<sub>cond</sub>
              &minus; &epsilon;<sub>neg</sub>). The result: the model steers{' '}
              <strong>toward</strong> the positive prompt and <strong>away
              from</strong> the negative prompt.
            </p>
            <p className="text-muted-foreground">
              Negative prompts are not a new mechanism. They are a direct
              application of the CFG formula you already know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Formula, Different Input">
            The CFG formula does not change. Only what you feed as the
            &ldquo;unconditional&rdquo; direction changes. This is the
            modularity pattern again: one formula, swappable inputs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Training vs Inference */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Training vs Inference"
            subtitle="Same components, different data flow"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The same components are arranged differently for training and
              inference. The U-Net is the same. The VAE is the same. But the
              data flow direction and each component&rsquo;s role differ.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Training Pipeline',
              color: 'amber',
              items: [
                'Starts with a real image from the dataset',
                'VAE encoder compresses image → z₀ [4, 64, 64]',
                'Sample random timestep t and noise ε',
                'Forward process: z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε',
                'U-Net predicts noise: ε̂ = unet(z_t, t, text)',
                'MSE loss: ||ε - ε̂||²',
                'Update U-Net weights only (CLIP and VAE are frozen)',
              ],
            }}
            right={{
              title: 'Inference Pipeline',
              color: 'blue',
              items: [
                'Starts from random noise z_T ~ N(0, I)',
                'VAE encoder is NOT used (no input image)',
                'Loop through ALL timesteps T → 1',
                'Two U-Net passes per step (CFG)',
                'Scheduler step: z_{t-1} from ε_cfg',
                'No loss, no gradient updates',
                'VAE decoder runs once at the end: z₀ → image',
              ],
            }}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Key differences to notice:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Training uses the VAE encoder. Inference does not.</strong>{' '}
                During text-to-image generation, there is no input image to
                encode. The starting point is random noise.
              </li>
              <li>
                <strong>Training processes one random timestep per
                image.</strong>{' '}
                Inference loops through all timesteps sequentially.
              </li>
              <li>
                <strong>Training updates the U-Net weights.</strong>{' '}
                Inference does not update anything&mdash;all components are
                frozen.
              </li>
              <li>
                <strong>Training needs real images.</strong>{' '}
                Inference starts from noise and creates images that never
                existed.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="VAE Encoder at Inference?">
            During text-to-image generation, the VAE encoder is{' '}
            <strong>never used</strong>. There is no input image to encode. The
            starting point is random noise in latent space. Only the VAE
            decoder runs&mdash;once, at the very end. (Img2img DOES use the
            encoder, but that is a later topic.)
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Transfer check */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Question" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  Your colleague wants to add a new conditioning signal&mdash;a
                  depth map that controls the 3D structure of the generated
                  image. Based on your understanding of the pipeline, where would
                  this signal enter? What form would it need to take?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2 space-y-2">
                    <p>
                      It would need to enter the <strong>U-Net</strong>,
                      probably via cross-attention or by concatenation with the
                      latent input. It would need to be encoded into a tensor
                      representation the U-Net can process&mdash;similar to how
                      text becomes a 77&times;768 tensor via CLIP.
                    </p>
                    <p>
                      The rest of the pipeline (VAE, scheduler) would be
                      unchanged. This is essentially what ControlNet does.
                    </p>
                  </div>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  If you doubled the VAE&rsquo;s compression factor
                  (32&times;32&times;4 instead of 64&times;64&times;4 latents),
                  what would change in the pipeline and what would stay the same?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2 space-y-2">
                    <p>
                      The U-Net would process 32&times;32&times;4
                      tensors&mdash;4&times; fewer spatial positions, much
                      faster. Cross-attention at the lowest resolution would
                      have fewer spatial tokens.
                    </p>
                    <p>
                      The denoising algorithm, CFG, and timestep conditioning
                      would all stay the same. The risk: the VAE might lose too
                      much detail at higher compression, producing
                      lower-quality decoded images. The tradeoff is compression
                      vs reconstruction quality.
                    </p>
                  </div>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: "You know every piece" enumeration */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Everything You Know, Nothing You Don't"
            subtitle="Every component in this pipeline is something you learned"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Look at this pipeline one more time. Every single component is
              something you built or deeply studied:
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  The VAE
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>Autoencoders</strong> through{' '}
                  <strong>Exploring Latent Spaces</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  The diffusion algorithm
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>The Diffusion Idea</strong> through{' '}
                  <strong>Build a Diffusion Model</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  The U-Net architecture
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>The U-Net Architecture</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  Timestep conditioning
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>Conditioning the U-Net</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  CLIP text encoder
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>CLIP</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  Cross-attention + CFG
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>Text Conditioning &amp; Guidance</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  Latent diffusion
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>From Pixels to Latents</strong>
                </p>
              </div>
              <div className="rounded-lg bg-muted/30 p-3">
                <p className="text-sm font-medium text-foreground">
                  Tokenization
                </p>
                <p className="text-xs text-muted-foreground">
                  <strong>Tokenization</strong> (Series 4)
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              There is nothing in this pipeline you have not already studied.
              You understand a state-of-the-art text-to-image system, component
              by component.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Payoff">
            This is what 14 lessons of deliberate practice built. Not a surface
            understanding of &ldquo;Stable Diffusion generates images from
            text,&rdquo; but a deep understanding of <strong>every
            component</strong> and <strong>every tensor handoff</strong> in the
            pipeline.
          </InsightBlock>
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
                  'Stable Diffusion is three independently trained models connected by tensor handoffs.',
                description:
                  'CLIP translates text to a 77×768 embedding tensor. The U-Net denoises a 4×64×64 latent tensor over 50 steps, using cross-attention (text), adaptive group norm (timestep), and CFG (two passes per step). The VAE decoder translates the final latent to a 3×512×512 pixel image.',
              },
              {
                headline:
                  'CFG happens at every step, not as post-processing.',
                description:
                  'Each denoising step requires two U-Net forward passes—one unconditional, one conditional. The CFG formula combines them. With 50 steps, that is 100 U-Net forward passes total.',
              },
              {
                headline:
                  'Training and inference use the same components differently.',
                description:
                  'Training uses the VAE encoder to compress real images into latents. Inference starts from random noise—the VAE encoder is never used. Only the VAE decoder runs, once, at the very end.',
              },
              {
                headline:
                  'Nothing in this pipeline is new to you.',
                description:
                  'Every component—the VAE, the diffusion algorithm, the U-Net, timestep conditioning, CLIP, cross-attention, CFG, latent diffusion—is something you built or deeply studied across 14 lessons.',
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
              CLIP translates language. The U-Net generates in latent space. The
              VAE translates back to pixels. Three translators, one pipeline.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Section 12: References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach, Blattmann, Lorenz, Esser & Ommer, 2022',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The Stable Diffusion paper. Section 3 describes the latent diffusion architecture. Figure 3 shows the pipeline diagram.',
              },
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision',
                authors: 'Radford et al., 2021',
                url: 'https://arxiv.org/abs/2103.00020',
                note: 'The CLIP paper. Sections 2.1–2.3 describe the dual-encoder architecture and contrastive training.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM paper. The training and sampling algorithms used in the denoising loop.',
              },
              {
                title: 'Classifier-Free Diffusion Guidance',
                authors: 'Ho & Salimans, 2022',
                url: 'https://arxiv.org/abs/2207.12598',
                note: 'The CFG paper. Describes the two-pass inference strategy and guidance scale.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 13: Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The pipeline works, but DDPM&rsquo;s 1000 steps (or even 50) is
              still slow. The next lesson explains why advanced samplers let you
              generate in 20 steps without retraining the model.
            </p>
            <p className="text-muted-foreground">
              Preview: the U-Net predicts noise at each step. The{' '}
              <strong>sampler</strong> decides how to <em>use</em> that
              prediction to take a step. Swapping the sampler is like changing
              the route, not the vehicle.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Samplers and Efficiency"
            description="Why advanced samplers let you generate in 20 steps instead of 1000&mdash;without retraining the model."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
