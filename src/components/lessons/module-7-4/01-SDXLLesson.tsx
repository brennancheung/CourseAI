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
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-4-1-sdxl.ipynb'

/**
 * SDXL
 *
 * Lesson 1 in Module 7.4 (Next-Generation Architectures). First lesson of
 * Module 7.4 and ninth lesson in Series 7.
 * Cognitive load: BUILD (2-3 new concepts extending familiar patterns:
 * dual text encoders, micro-conditioning, refiner model).
 *
 * SDXL is the U-Net pushed to its practical ceiling—larger backbone, higher
 * training resolution, dual text encoders for richer conditioning, a refiner
 * model for fine detail, and micro-conditioning to handle multi-resolution
 * training data. Every innovation is about what goes IN, what goes AROUND,
 * or what goes ALONGSIDE the U-Net, not a new architecture. Sets up the
 * implicit question the next lesson (DiT) answers: what if you replaced
 * the U-Net entirely?
 *
 * Builds on: SD v1.5 pipeline (6.4.1), CLIP (6.3.3), cross-attention (6.3.4),
 * img2img (6.5.2), U-Net architecture (6.3.1), conditioning mechanisms (6.3.2),
 * IP-Adapter (7.1.3)
 *
 * Previous: The Speed Landscape (Module 7.3, Lesson 3 / CONSOLIDATE)
 * Next: DiT (Module 7.4, Lesson 2)
 */

export function SDXLLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="SDXL"
            description="The U-Net pushed to its limit—dual text encoders, micro-conditioning, and a refiner model. Every improvement is about what goes IN to the U-Net, not a new architecture."
            category="Next-Generation Architectures"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain the key architectural and conditioning innovations in
            SDXL—dual text encoders, micro-conditioning, and the refiner
            model—and why these represent the limits of what U-Net-based scaling
            can achieve before the paradigm shift to transformers.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            Two to three new concepts, all extensions of familiar patterns. Dual
            text encoders extend CLIP + cross-attention. Micro-conditioning
            extends adaptive norm. The refiner extends img2img. Nothing here is
            a conceptual leap—just familiar mechanisms applied in new ways.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'WHAT changed between SD v1.5 and SDXL, and WHY each change was made',
              'Tensor shape trace through the SDXL pipeline (mirroring the trace from Stable Diffusion Architecture)',
              'The refiner as img2img with a specialized model',
              'Micro-conditioning as a solution to multi-resolution training artifacts',
              'NOT: training SDXL from scratch (Series 6 had the training experience)',
              'NOT: SDXL Turbo or LCM-LoRA for SDXL (already covered in Latent Consistency & Turbo)',
              'NOT: every internal U-Net detail (exact channel counts, attention placement)',
              'NOT: ControlNet or IP-Adapter for SDXL (same concepts from Module 7.1)',
              'NOT: the DiT architecture (next lesson)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap — brief reactivation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Foundation"
            subtitle="Three concepts you already know deeply"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SDXL starts from the exact same foundation you traced in{' '}
              <strong>Stable Diffusion Architecture</strong>. A quick
              reactivation of the three pieces that SDXL modifies:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>The SD v1.5 pipeline:</strong> text → CLIP [77, 768] →
                U-Net denoising loop with CFG → VAE decode [3, 512, 512]. The
                U-Net has channels 320/640/1280/1280, cross-attention at 16×16
                and 32×32, and generates in [4, 64, 64] latent space.
              </li>
              <li>
                <strong>Cross-attention:</strong> Q from spatial features, K/V
                from text embeddings. Each spatial location independently
                attends to all 77 text tokens. The quality of the text embedding
                directly determines how well the U-Net follows the prompt.
              </li>
              <li>
                <strong>Img2img:</strong> take an image, encode with VAE, noise
                to an intermediate level, denoise from there. Strength parameter
                controls the starting timestep. You implemented this from
                scratch.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Everything you know about SD v1.5 still applies. The question is:{' '}
              <strong>what did Stability AI change, and why?</strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Backbone">
            SDXL is still a U-Net model. The same diagram from{' '}
            <strong>U-Net Architecture</strong> still applies. We are tracing
            what changed around the U-Net, not inside it.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — Before/After + Challenge */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Quality Jump"
            subtitle="Same prompt, dramatically different results"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the prompt you traced through SD v1.5 in{' '}
              <strong>Stable Diffusion Architecture</strong>:{' '}
              <em>&ldquo;a cat sitting on a beach at sunset.&rdquo;</em>
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="SD v1.5 at 512×512" color="amber">
                <div className="space-y-2 text-sm">
                  <p>
                    Recognizable scene, decent composition, but soft details.
                    The cat&rsquo;s fur lacks individual strands. Sand has a
                    slight plastic texture. Sunset colors are pleasant but lack
                    the complexity of real light scattering. Text following is
                    approximate—the cat might be standing rather than sitting.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="SDXL at 1024×1024" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    Dramatically sharper. Individual fur strands visible. Sand
                    has realistic granularity. Sunset has complex color
                    gradients and atmospheric haze. Text following is noticeably
                    better—the cat is sitting, the beach has waves, the sunset
                    is prominent.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              This improvement comes from <strong>five changes</strong>. Before
              reading what they are, predict: what would{' '}
              <strong>you</strong> change about SD v1.5 to get better results?
              Think about the components you know—what are the weaknesses?
            </p>
            <details className="mt-2 rounded-lg border bg-muted/30 p-4">
              <summary className="font-medium cursor-pointer text-primary">
                What you might have predicted
              </summary>
              <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                <p>
                  From your existing knowledge, you should be able to identify
                  at least 2-3 of these:
                </p>
                <ol className="list-decimal list-inside space-y-1 ml-2">
                  <li>
                    <strong>Text understanding is limited</strong>—CLIP ViT-L
                    has only 123M params, 77 tokens, and known limitations
                    (typographic attacks, spatial reasoning, counting).
                  </li>
                  <li>
                    <strong>Resolution is stuck at 512×512</strong>—the model
                    was trained at this resolution and generates poorly at
                    higher sizes.
                  </li>
                  <li>
                    <strong>Details are soft</strong>—a single denoising pass
                    must handle both global composition and fine-grained detail.
                  </li>
                </ol>
                <p>
                  SDXL addresses all three—plus two more you might not have
                  predicted.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Five Changes, Zero New Ideas">
            Every SDXL improvement uses a mechanism you already know. The
            innovation is in combining and scaling them—not in new
            architectures.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: The Five Changes (Overview) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Five Changes"
            subtitle="What SDXL actually changed"
          />
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <GradientCard title="1. Dual Text Encoders" color="blue">
                <p className="text-sm">
                  Two CLIP models instead of one, for richer text understanding.
                </p>
              </GradientCard>
              <GradientCard title="2. Larger U-Net" color="violet">
                <p className="text-sm">
                  Wider channels, more attention at higher resolutions.
                </p>
              </GradientCard>
              <GradientCard title="3. Higher Resolution" color="cyan">
                <p className="text-sm">
                  1024×1024 base resolution instead of 512×512.
                </p>
              </GradientCard>
              <GradientCard title="4. Refiner Model" color="emerald">
                <p className="text-sm">
                  A second U-Net for fine detail polish.
                </p>
              </GradientCard>
              <GradientCard title="5. Micro-Conditioning" color="orange">
                <p className="text-sm">
                  Resolution and crop metadata as conditioning inputs.
                </p>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Species">
            Notice what is <strong>not</strong> on this list: a new
            architecture. The U-Net is the same species. Every change is about
            what goes IN to the U-Net, what comes OUT, or what goes AROUND it.
            The U-Net backbone itself is still a convolutional encoder-decoder
            with skip connections and cross-attention.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5b: Pipeline Comparison — visual modality */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              How do these five changes fit into the pipeline you already know?
              Compare the two pipelines side-by-side. What stayed the same is
              as important as what changed:
            </p>
            <ComparisonRow
              left={{
                title: 'SD v1.5 Pipeline',
                color: 'amber',
                items: [
                  'Text → CLIP ViT-L [77, 768]',
                  'Latent: [4, 64, 64] at 512×512',
                  'U-Net: 320/640/1280/1280 channels',
                  'Cross-attention K/V from [77, 768]',
                  'CFG: two forward passes, guidance scale',
                  'VAE decode → [3, 512, 512]',
                  'Single model, single pass',
                ],
              }}
              right={{
                title: 'SDXL Pipeline',
                color: 'emerald',
                items: [
                  'Text → CLIP ViT-L [77, 768] + OpenCLIP ViT-bigG [77, 1280] → concat [77, 2048]',
                  'Latent: [4, 128, 128] at 1024×1024',
                  'U-Net: wider channels, more attention layers',
                  'Cross-attention K/V from [77, 2048] + pooled [1280] via adaptive norm',
                  'CFG: same mechanism, richer embeddings',
                  'VAE decode → [3, 1024, 1024]',
                  'Optional refiner (second U-Net, img2img at t_switch)',
                  'Micro-conditioning: original_size, crop_top_left, target_size via adaptive norm',
                ],
              }}
            />
            <p className="text-sm text-muted-foreground italic">
              The core loop is identical: text encoding → cross-attention
              conditioning → U-Net denoising → VAE decode. Every SDXL change
              is about richer inputs, larger scale, or an additional pipeline
              stage—not a new architecture.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Reading This Comparison">
            Items that look familiar (CFG, VAE decode, cross-attention) are
            unchanged mechanisms. Items that look different (dual encoders,
            wider K/V, refiner, micro-conditioning) are SDXL&rsquo;s
            innovations. The U-Net box is the same shape in both
            pipelines—only what feeds into it changed.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Dual Text Encoders */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Dual Text Encoders"
            subtitle="Two translators, one microphone"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SD v1.5 uses CLIP ViT-L/14, producing [77, 768] embeddings. This
              encoder has 123M parameters and was trained on 400M image-text
              pairs. It is good at understanding common concepts and
              compositions but struggles with nuance: specific art styles,
              precise spatial descriptions, fine-grained attributes.
            </p>
            <p className="text-muted-foreground">
              SDXL adds a second encoder: OpenCLIP ViT-bigG/14. This is a much
              larger CLIP model (354M parameters in the text encoder alone)
              trained on 2B image-text pairs (LAION-2B). It produces [77, 1280]
              embeddings.
            </p>
            <p className="text-muted-foreground">
              The two encoders&rsquo; outputs are{' '}
              <strong>concatenated along the embedding dimension</strong>:
            </p>

            <CodeBlock
              code={`Prompt: "a cat sitting on a beach at sunset"

CLIP ViT-L/14:     [77 tokens, 768 dims]   ← SD v1.5's original encoder
OpenCLIP ViT-bigG:  [77 tokens, 1280 dims]  ← SDXL's addition
Concatenated:       [77 tokens, 2048 dims]  ← One embedding per token, richer

Cross-attention K/V source: [77, 2048]  (was [77, 768] in SD v1.5)`}
              language="text"
              filename="sdxl_text_embeddings.txt"
            />

            <p className="text-muted-foreground">
              The cross-attention mechanism is unchanged. Q still comes from
              spatial features. K and V come from this wider embedding. Each
              spatial location still independently attends to all 77 tokens.
              The only difference: the embedding at each token position carries
              2048 dimensions of meaning instead of 768.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why Not Replace?">
            Why keep the smaller CLIP ViT-L and add a second encoder instead of
            just using the bigger one? Both encoders were trained on different
            data with different objectives. Their representations are
            complementary—the concatenation captures information that neither
            encoder alone provides.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Dual Encoders: SDXL vs IP-Adapter comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might be thinking: &ldquo;two encoders feeding
              cross-attention—isn&rsquo;t that what IP-Adapter does?&rdquo; The
              mechanism is fundamentally different:
            </p>
            <ComparisonRow
              left={{
                title: 'SDXL: Concatenation',
                color: 'blue',
                items: [
                  'Inputs combined before cross-attention',
                  'One cross-attention path (wider K/V source)',
                  'Fixed balance (no scale parameter)',
                  'Both inputs are text (same prompt, different encoders)',
                ],
              }}
              right={{
                title: 'IP-Adapter: Decoupled Attention',
                color: 'violet',
                items: [
                  'Outputs combined after cross-attention',
                  'Two cross-attention paths (separate K/V projections)',
                  'Adjustable balance (IP-Adapter scale)',
                  'Different modalities (text vs image)',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Goal, Different Approach">
            Both SDXL and IP-Adapter want richer conditioning. SDXL combines
            two text encoders <strong>before</strong> the bottleneck. IP-Adapter
            combines text and image <strong>after</strong> the bottleneck.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Pooled embeddings */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <SectionHeader
              title="Pooled Embeddings"
              subtitle="Global conditioning alongside per-token conditioning"
            />
            <p className="text-muted-foreground">
              Beyond the per-token sequence embeddings, SDXL also uses the
              pooled (CLS) embedding from OpenCLIP ViT-bigG. This is a single
              [1280]-dim vector summarizing the entire prompt. It is
              concatenated with the timestep embedding and fed through the
              adaptive norm pathway—the same global conditioning mechanism you
              know from <strong>Conditioning the U-Net</strong>.
            </p>
            <p className="text-muted-foreground">
              So SDXL uses text in <strong>two ways</strong>:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Per-Token (Spatially Varying)" color="blue">
                <div className="space-y-1 text-sm">
                  <p>
                    <strong>Shape:</strong> [77, 2048]
                  </p>
                  <p>
                    <strong>Injection:</strong> Cross-attention K/V
                  </p>
                  <p>
                    <strong>Effect:</strong> Different spatial locations attend
                    to different tokens
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Pooled (Global)" color="violet">
                <div className="space-y-1 text-sm">
                  <p>
                    <strong>Shape:</strong> [1280]
                  </p>
                  <p>
                    <strong>Injection:</strong> Adaptive norm (with timestep)
                  </p>
                  <p>
                    <strong>Effect:</strong> Same influence at every spatial
                    location
                  </p>
                </div>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Two Injection Points">
            Per-token embeddings through cross-attention tell the U-Net{' '}
            <em>what goes where</em>. The pooled embedding through adaptive
            norm tells the U-Net the <em>overall vibe</em>. Two channels, two
            purposes—same text prompt.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 7: Check #1 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In SDXL, the cross-attention K/V source is [77, 2048]. In SD
                  v1.5, it was [77, 768]. Does this change the cross-attention
                  output shape?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The attention output shape is determined by Q and V
                    dimensions. Q comes from spatial features, same as before.
                    The attention weights are still [n_spatial, 77]. The output
                    dimension per spatial location changes from 768 to 2048, but
                    this is handled by the projection layers within the
                    attention block.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  If you loaded an SD v1.5 LoRA into SDXL, would it work?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The LoRA targets cross-attention projection matrices. In
                    SD v1.5, W_K and W_V project from 768 dimensions. In SDXL,
                    they project from 2048 dimensions. The matrix shapes are
                    incompatible. This is why &ldquo;SD v1.5 LoRAs&rdquo; and
                    &ldquo;SDXL LoRAs&rdquo; are separate categories—same
                    mechanism, different dimensions.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague says: &ldquo;SDXL uses two encoders, so it takes
                  twice as long to encode the prompt.&rdquo; Is this accurate?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Roughly yes for the text encoding step—two forward passes
                    through two models. But text encoding is a negligible
                    fraction of total inference time (one pass each, vs 50+ U-Net
                    passes for denoising). The actual compute cost increase
                    comes from the larger U-Net, not the dual encoders.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 8: Micro-Conditioning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Micro-Conditioning"
            subtitle="Teaching the model what it is looking at"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training SDXL required massive datasets—hundreds of millions of
              images. These images come in every resolution: 256×256 thumbnails,
              512×512 web images, 1024×1024 photographs, 2000×3000 DSLR shots.
              How do you train on all of them?
            </p>

            <div className="space-y-3">
              <GradientCard title="Option 1: Only High-Resolution Images" color="rose">
                <p className="text-sm">
                  Only use images at or above 1024×1024. This throws away most
                  of the dataset. You need the data—discarding it is wasteful
                  and limits diversity.
                </p>
              </GradientCard>
              <GradientCard title="Option 2: Resize Everything" color="rose">
                <p className="text-sm">
                  Resize everything to 1024×1024. A 256×256 thumbnail upscaled
                  to 1024×1024 is blurry. The model learns &ldquo;sometimes
                  images are blurry&rdquo; and occasionally generates blurry
                  output.
                </p>
              </GradientCard>
              <GradientCard title="Option 3: Crop from Larger Images" color="rose">
                <p className="text-sm">
                  Crop 1024×1024 regions from larger images. A 2000×3000 photo
                  cropped to 1024×1024 might cut off heads, miss the main
                  subject, or show off-center compositions. The model learns
                  &ldquo;sometimes the subject is partially visible.&rdquo;
                </p>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              SD v1.5 suffered from these exact issues at 512×512. Images with
              cut-off heads, off-center subjects, and inconsistent quality. The
              larger and more diverse the dataset, the worse these artifacts
              become.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Training Data Dilemma">
            Every option has a cost: lose data (Option 1), learn artifacts
            (Option 2), or learn bad compositions (Option 3). SD v1.5 accepted
            all three costs. SDXL found a way to avoid them.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Micro-conditioning: the solution */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <SectionHeader
              title="The Solution"
              subtitle="Tell the model what it is looking at"
            />
            <p className="text-muted-foreground">
              Micro-conditioning tells the model about each training
              image&rsquo;s context. Three additional numbers fed alongside the
              timestep:
            </p>

            <GradientCard title="Micro-Conditioning Inputs" color="orange">
              <ul className="space-y-3 text-sm">
                <li>
                  <strong>original_size</strong>—The resolution of the
                  training image before any resizing. &ldquo;This image was
                  originally 256×256&rdquo; vs &ldquo;This image was originally
                  2048×1536.&rdquo;
                </li>
                <li>
                  <strong>crop_top_left</strong>—Where the 1024×1024 crop was
                  taken from. (0, 0) means top-left corner, (256, 128) means
                  cropped from the middle. The model learns that (0, 0) crops
                  often have cut-off bottoms and centered crops are
                  well-composed.
                </li>
                <li>
                  <strong>target_size</strong>—The resolution the model should
                  generate. Always set to (1024, 1024) at inference. During
                  training, it matches the training resolution.
                </li>
              </ul>
            </GradientCard>

            <p className="text-muted-foreground">
              At inference, you set{' '}
              <strong>
                original_size = (1024, 1024), crop_top_left = (0, 0),
                target_size = (1024, 1024)
              </strong>
              . This tells the model: &ldquo;generate as if the original image
              was high-resolution and well-centered.&rdquo; The model has
              learned to separate image quality from image content.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Metadata as Conditioning">
            The model learns: &ldquo;when the original was 256×256, details
            look like this. When it was 1024×1024, details look like this.&rdquo;
            At inference you ask for 1024×1024 quality regardless of what the
            training image resolution was.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Micro-conditioning: injection mechanism */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <SectionHeader
              title="How Micro-Conditioning Is Injected"
              subtitle="Same plumbing, more information"
            />
            <p className="text-muted-foreground">
              These numbers are encoded and added to the timestep embedding.
              Same injection mechanism as the timestep: through the adaptive
              norm pathway at every processing stage. The U-Net already has the
              plumbing for global conditioning inputs—micro-conditioning just
              sends more information through the same pipes.
            </p>

            <ComparisonRow
              left={{
                title: 'Timestep Conditioning (SD v1.5)',
                color: 'blue',
                items: [
                  'Tells the U-Net WHEN in the denoising process',
                  'Injected via adaptive group normalization',
                  'Same at every spatial location (global)',
                ],
              }}
              right={{
                title: 'Micro-Conditioning (SDXL addition)',
                color: 'orange',
                items: [
                  'Tells the U-Net AT-WHAT-QUALITY the training data was',
                  'Injected via the same adaptive norm pathway',
                  'Same at every spatial location (global)',
                ],
              }}
            />

            <p className="text-muted-foreground">
              This is not a minor implementation detail. Without
              micro-conditioning, SDXL would either waste most of its training
              data (only using 1024×1024 images) or produce
              resolution-dependent artifacts (blurry outputs from upscaled
              training data, cropped compositions from random crops).
              Micro-conditioning is what makes it possible to train on a
              massive, diverse dataset <strong>and</strong> generate
              high-quality output.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Extending the Framework">
            From <strong>IP-Adapter</strong>, you know four conditioning
            channels: WHEN (timestep), WHAT (text), WHERE (ControlNet),
            WHAT-IT-LOOKS-LIKE (IP-Adapter). Micro-conditioning adds a fifth:
            AT-WHAT-QUALITY (resolution awareness). Same philosophy—one more
            input.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Micro-conditioning predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  At inference, you set original_size = (1024, 1024) and
                  crop_top_left = (0, 0). What would happen if you set
                  original_size = (256, 256) and crop_top_left = (0, 0)
                  instead?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The model would generate as if the original image was a
                    256×256 thumbnail. The output would likely be softer, less
                    detailed—the model learned that 256×256 originals tend to be
                    lower quality. This is sometimes used intentionally to
                    produce a specific aesthetic, but generally you want
                    original_size = target_size for best quality.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Micro-conditioning is injected through the same adaptive norm
                  pathway as the timestep. Could you add other metadata the same
                  way—for example, the camera model or the aesthetic quality
                  rating of the training image?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes, in principle. Any global metadata about the training
                    image can be encoded and added to the timestep embedding.
                    Some models do exactly this—for example, conditioning on
                    aesthetic scores to steer toward higher-quality outputs at
                    inference. The key insight is that the adaptive norm pathway
                    is a general-purpose global conditioning channel, not
                    specialized for timesteps.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: The Refiner Model */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Refiner Model"
            subtitle="Same trail, different hiker for the final stretch"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SDXL&rsquo;s base model generates at 1024×1024 with dramatically
              better composition and text following than SD v1.5. But a single
              U-Net must handle everything: global composition, mid-level
              structure, and fine-grained details (skin pores, fabric weave,
              individual leaves). The denoising process allocates these
              responsibilities across timesteps—high noise for composition, low
              noise for details—but asking one model to be expert at both is a
              lot.
            </p>
            <p className="text-muted-foreground">
              The SDXL refiner is a second U-Net, fine-tuned specifically on
              high-quality, high-resolution images. It specializes in the
              low-noise timesteps where fine detail matters.
            </p>

            <GradientCard title="The Two-Model Pipeline" color="emerald">
              <ol className="space-y-2 text-sm list-decimal list-inside">
                <li>
                  <strong>Base model</strong> denoises from t=T (pure noise) to
                  t=t_switch (e.g., ~200)—handles composition, structure, color
                </li>
                <li>
                  <strong>Refiner model</strong> denoises from t=t_switch to
                  t=0—handles fine detail, texture, sharpness
                </li>
              </ol>
            </GradientCard>

            <p className="text-muted-foreground">
              This is the <strong>img2img mechanism you already know</strong>.
              The base model&rsquo;s output at t_switch is a partially denoised
              latent. The refiner takes this latent, treats it as an img2img
              input at strength corresponding to t_switch, and completes the
              denoising. Same mechanism you implemented in the{' '}
              <strong>Img2img & Inpainting</strong> notebook, applied with a
              different model.
            </p>
            <p className="text-muted-foreground">
              In img2img, you started from a real image noised to an
              intermediate level. Here, you start from a model-generated
              intermediate result. The pipeline is identical: latent at
              intermediate noise level → denoise from there to t=0. The only
              difference: the model doing the denoising was specifically
              trained for this finishing role.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Optional Polish">
            The refiner is optional. SDXL&rsquo;s base model generates good
            results without it. The refiner adds polish—sharper details, more
            realistic textures—at the cost of a second denoising pass. Whether
            it is worth the compute depends on the use case.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 11: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Refiner predictions"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  The refiner model receives a partially denoised latent at
                  timestep t_switch. Does it also receive the text prompt?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes. The refiner uses the same dual text encoder embeddings
                    for cross-attention guidance during its denoising pass. The
                    text prompt continues to guide the generation at the
                    fine-detail level. Without text conditioning, the refiner
                    would polish details without semantic guidance—it might
                    sharpen fur on a dog when the prompt said &ldquo;cat.&rdquo;
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague suggests setting t_switch = T (the refiner starts
                  from pure noise, doing all the denoising). What would happen?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The refiner was fine-tuned on the low-noise timestep range.
                    It was not trained to handle high-noise denoising (global
                    composition, structure). Starting from pure noise would
                    likely produce poor results because the refiner&rsquo;s
                    specialization is in fine detail, not composition. This is
                    like asking a detail painter to also sketch the entire scene
                    from scratch—different skills.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 12: Resolution and Architecture Scale */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Resolution and Architecture Scale"
            subtitle="Bigger, not different"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SDXL generates at 1024×1024 base resolution. With the same VAE
              (8× downsampling), the latent is [4, 128, 128]—four times the
              spatial area of SD v1.5&rsquo;s [4, 64, 64]. The U-Net processes
              this larger latent with a correspondingly larger architecture:
              deeper attention at higher resolutions, wider channels.
            </p>
            <p className="text-muted-foreground">
              The compute cost is concrete. In SD v1.5&rsquo;s [4, 64, 64]
              latent, self-attention at 16×16 resolution operates on 256
              tokens. In SDXL&rsquo;s [4, 128, 128] latent, self-attention at
              32×32 operates on 1,024 tokens—4× more. At 64×64, that is 4,096
              tokens. Attention is O(n²) in token count, so 4× more tokens
              means roughly 16× more compute per attention layer. This is why
              SDXL is significantly slower than SD v1.5, not just because of a
              wider U-Net.
            </p>
            <p className="text-muted-foreground">
              And scaling the architecture alone is not enough—the model must
              be <strong>trained on 1024×1024 images</strong> to generate them
              well. SD v1.5 was trained primarily on 512×512 and produces poor
              results when asked to generate at higher resolutions. Resolution
              improvement required both architectural scaling (more compute)
              and training strategy changes (multi-resolution training with
              micro-conditioning).
            </p>
            <p className="text-muted-foreground">
              This points to the U-Net&rsquo;s fundamental constraint:{' '}
              <strong>scaling a convolutional architecture is ad hoc</strong>.
              You can widen channels, add more attention blocks, increase
              resolution—but there is no systematic recipe. Each scaling
              decision requires manual engineering. Transformers, as you will
              see in the next lesson, scale more predictably.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The U-Net's Last Stand">
            Every SDXL innovation is about what goes <strong>in</strong> (dual
            encoders), what goes <strong>around</strong> (refiner), or what
            goes <strong>alongside</strong> (micro-conditioning). The U-Net
            backbone itself is the same species, just larger. The question the
            next lesson answers: what if you replaced it entirely?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 13: SDXL in the Ecosystem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="SDXL in the Ecosystem"
            subtitle="What it changed for practitioners"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              SDXL became the standard base model for the community, replacing
              SD v1.5 as the default starting point. Everything you learned in
              Series 7 works with SDXL:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>ControlNet</strong> works with SDXL (same mechanism,
                different dimensions—requires SDXL-trained ControlNets)
              </li>
              <li>
                <strong>IP-Adapter</strong> works with SDXL (same decoupled
                cross-attention, SDXL-specific adapters)
              </li>
              <li>
                <strong>LoRA</strong> works with SDXL (same bypass mechanism,
                SDXL-specific weight dimensions)
              </li>
              <li>
                <strong>SDXL Turbo</strong> (from{' '}
                <strong>Latent Consistency & Turbo</strong>) applies adversarial
                distillation to the SDXL base model
              </li>
              <li>
                <strong>LCM-LoRA for SDXL</strong> enables 4-step generation
                from SDXL checkpoints
              </li>
              <li>
                The <strong>speed landscape</strong> from{' '}
                <strong>The Speed Landscape</strong> applies directly—all
                acceleration approaches work with SDXL
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Mechanisms, Different Dimensions">
            Every adapter type (ControlNet, IP-Adapter, LoRA) works with
            SDXL—but you need SDXL-specific versions. The matrix shapes
            changed, so SD v1.5 adapters are incompatible. Same concept,
            different weight files.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* What SDXL could NOT solve */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What SDXL Could Not Solve"
            subtitle="The limits of the U-Net approach"
          />
          <div className="space-y-4">
            <div className="space-y-3">
              <GradientCard title="No Scaling Recipe" color="amber">
                <p className="text-sm">
                  There is no clear &ldquo;double compute, halve the
                  loss&rdquo; relationship. Each U-Net scaling decision—wider
                  channels here, more attention there—is manual engineering, not
                  a predictable formula.
                </p>
              </GradientCard>
              <GradientCard title="Limited Text-Image Interaction" color="amber">
                <p className="text-sm">
                  Cross-attention is the only text-image interaction point. Text
                  tokens and image features interact only at cross-attention
                  layers (16×16, 32×32). The rest of the U-Net processes
                  spatial features without text input. Is this the best way?
                </p>
              </GradientCard>
              <GradientCard title="Convolutional Inductive Biases" color="amber">
                <p className="text-sm">
                  Convolutions assume local spatial structure. This is
                  useful—translational equivariance—but also limiting. Global
                  relationships require many layers, large receptive fields, or
                  attention.
                </p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              These limitations are not bugs in SDXL. They are properties of
              the U-Net architecture itself. To address them, you need a
              different architecture. That is the subject of the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Architecture, Not Engineering">
            Every SDXL improvement is an engineering achievement within the
            U-Net paradigm. The limitations above are architectural—they
            require a paradigm shift, not more engineering.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 14: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: SDXL Pipeline"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes the lesson concrete—inspect the dual
                encoders, compare base vs SDXL quality, explore
                micro-conditioning, and set up the refiner pipeline.
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
                  <strong>Exercise 1 (Guided): SDXL Pipeline Inspection.</strong>{' '}
                  Load SDXL base model via diffusers. Inspect the dual text
                  encoders: print model class names, parameter counts, output
                  shapes. Verify: CLIP ViT-L produces [77, 768], OpenCLIP
                  ViT-bigG produces [77, 1280]. Predict the combined text
                  embedding shape before running.
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): SDXL Base Generation and Comparison.
                  </strong>{' '}
                  Generate &ldquo;a cat sitting on a beach at sunset&rdquo; with
                  SDXL base at 1024×1024. Compare to SD v1.5 at 512×512. Vary
                  guidance_scale (5, 7.5, 10) and observe quality differences.
                  Predict whether the optimal guidance_scale for SDXL will be
                  higher or lower than SD v1.5.
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): Micro-Conditioning Exploration.
                  </strong>{' '}
                  Generate the same prompt with different micro-conditioning
                  values: original_size = (1024, 1024) vs (256, 256); crop
                  offset = (0, 0) vs (512, 512). Compare outputs to see how
                  micro-conditioning affects quality and composition.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): Base + Refiner Pipeline.
                  </strong>{' '}
                  Set up the two-stage pipeline: base model denoises for most
                  steps, refiner finishes the last ~20%. Compare base-only vs
                  base+refiner output. Vary t_switch (10%, 20%, 40% of steps)
                  and compare sharpness. Bonus: time the pipeline to quantify
                  the compute tradeoff.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: pipeline inspection (shapes)</li>
              <li>Guided: generation comparison (quality)</li>
              <li>Supported: micro-conditioning (explore)</li>
              <li>Independent: refiner pipeline (build)</li>
            </ol>
            <p className="text-sm mt-2">
              SDXL requires significant VRAM (~7 GB for the base model). If
              running locally, use float16 or consider Colab&rsquo;s GPU
              runtime.
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
                headline: 'Same architecture, better everything around it.',
                description:
                  'SDXL is still a U-Net. The improvements are about conditioning (dual text encoders, micro-conditioning), scale (larger U-Net, higher resolution), and pipeline (refiner model). The denoising backbone is the same species as SD v1.5.',
              },
              {
                headline: 'Dual encoders = wider text embedding.',
                description:
                  'CLIP ViT-L [77, 768] + OpenCLIP ViT-bigG [77, 1280] = [77, 2048]. One cross-attention path, richer K/V source. Not decoupled attention—concatenated before the bottleneck.',
              },
              {
                headline: 'Micro-conditioning solves multi-resolution training.',
                description:
                  'Tell the model what it is looking at (original resolution, crop position, target resolution), and it learns to separate content from quality. At inference, ask for high-quality output regardless of training data diversity.',
              },
              {
                headline: 'The refiner is img2img with a specialist.',
                description:
                  'Same mechanism you implemented in Img2img & Inpainting. The base model handles composition; the refiner polishes details. Optional but effective.',
              },
              {
                headline: "The U-Net's last stand.",
                description:
                  'Every SDXL improvement pushes the U-Net architecture further without replacing it. The implicit question: is there a ceiling? The next lesson answers that question.',
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
                title: 'SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis',
                authors: 'Podell, English, Lacey, Blattmann, Dockhorn, Müller, Penna & Rombach, 2023',
                url: 'https://arxiv.org/abs/2307.01952',
                note: 'The SDXL paper. Section 2 covers the architecture changes (dual encoders, micro-conditioning). Section 3 covers the refiner model. Table 1 compares to SD v1.5.',
              },
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)',
                authors: 'Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger & Sutskever, 2021',
                url: 'https://arxiv.org/abs/2103.00020',
                note: 'The CLIP paper. Relevant for understanding the baseline text encoder architecture that SDXL extends with a second encoder.',
              },
              {
                title: 'Reproducible scaling laws for contrastive language-image learning (OpenCLIP)',
                authors: 'Cherti, Beaumont, Wightman, Wortsman, Ilharco, Gordon, Schuhmann, Schmidt & Jitsev, 2023',
                url: 'https://arxiv.org/abs/2212.07143',
                note: 'The OpenCLIP paper. Section 4 covers the ViT-bigG/14 model that SDXL uses as its second text encoder.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 16: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Diffusion Transformer (DiT)"
            description="SDXL showed what happens when you push the U-Net to its limit. The next lesson replaces the U-Net entirely with a vision transformer: patchify the latent into tokens, process with standard transformer blocks. The architecture you learned in Series 4 meets the diffusion framework from Series 6. Everything converges."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
