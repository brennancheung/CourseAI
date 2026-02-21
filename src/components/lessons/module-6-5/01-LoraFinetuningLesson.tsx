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
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-5-1-lora-finetuning.ipynb'

/**
 * LoRA Fine-Tuning for Diffusion Models
 *
 * Lesson 1 in Module 6.5 (Customization). Lesson 18 overall in Series 6.
 * Cognitive load: BUILD (2 new concepts).
 *
 * Transfer lesson: applies LoRA (DEVELOPED from Module 4.4) to the
 * Stable Diffusion U-Net (DEVELOPED from Module 6.4). The two genuinely
 * new concepts are (1) where LoRA goes in the diffusion U-Net and why,
 * and (2) the diffusion LoRA training loop.
 *
 * Previous: Generate with Stable Diffusion (Module 6.4, Lesson 3 / CONSOLIDATE)
 * Next: Img2Img and Inpainting (Module 6.5, Lesson 2)
 */

export function LoraFinetuningLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="LoRA Fine-Tuning for Diffusion Models"
            description="Same detour, different highway. Apply LoRA to the Stable Diffusion U-Net for style and subject customization—without retraining 860M parameters."
            category="Customization"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how LoRA fine-tuning adapts the Stable Diffusion U-Net
            for new styles and subjects by targeting cross-attention projections,
            using the same noise-prediction training objective you already
            know—but with LoRA adapters as the only trainable parameters.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You built LoRA from scratch in{' '}
            <LessonLink slug="lora-and-quantization">LoRA and Quantization</LessonLink>—the highway-and-detour
            bypass, B=0 initialization, merge at inference. You traced the
            full SD pipeline in <LessonLink slug="stable-diffusion-architecture">The Stable Diffusion Pipeline</LessonLink>.
            This lesson connects those two systems.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Where LoRA adapters go in the diffusion U-Net and why cross-attention is the primary target',
              'How the diffusion LoRA training loop differs from LLM LoRA training',
              'Practical data requirements for style vs subject LoRA',
              'Multiple LoRA composition (applying two LoRAs simultaneously)',
              'NOT: reimplementing LoRA from scratch (done in LoRA and Quantization)',
              'NOT: DreamBooth, textual inversion, or other fine-tuning techniques',
              'NOT: ControlNet or structural conditioning',
              'NOT: SD v1 vs v2 vs XL LoRA differences',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Quick Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Reactivating two systems you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>LoRA</strong> (from{' '}
              <strong>LoRA and Quantization</strong>): the highway-and-detour
              bypass. Freeze the original weight matrix <InlineMath math="W" />{' '}
              (the highway). Attach a small trainable bypass{' '}
              <InlineMath math="BA" /> (the detour). The forward pass becomes{' '}
              <InlineMath math="h = Wx + BAx \cdot \frac{\alpha}{r}" />.{' '}
              <InlineMath math="B" /> starts at zero so the detour has no effect
              at initialization. At inference, merge the detour into the highway
              ({' '}
              <InlineMath math="W_{\text{merged}} = W + BA \cdot \frac{\alpha}{r}" />
              ) for zero overhead.
            </p>
            <p className="text-muted-foreground">
              <strong>Stable Diffusion pipeline</strong> (from{' '}
              <strong>The Stable Diffusion Pipeline</strong>): text → CLIP
              encode → U-Net denoising loop (cross-attention injects text, CFG
              amplifies it) → VAE decode. The U-Net&rsquo;s cross-attention
              layers are where text meaning meets spatial features—
              <InlineMath math="W_Q" /> projects spatial features (the seeking
              lens), <InlineMath math="W_K" /> and <InlineMath math="W_V" />{' '}
              project text embeddings (the advertising and contributing lenses).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Two Systems, One Connection">
            You have LoRA at DEVELOPED depth and the SD pipeline at DEVELOPED
            depth. This lesson connects them—applying a known technique to a
            known system. The genuinely new content is the specific
            intersection: which layers, what training data, how the training
            loop adapts.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — Prediction Checkpoint */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Prediction Checkpoint"
            subtitle="Test your intuition before we connect the pieces"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know LoRA. You know Stable Diffusion. Before we connect them,
              predict:
            </p>
          </div>
          <div className="space-y-4 mt-4">
            <GradientCard title="Prediction 1: Target Layers" color="cyan">
              <p className="text-sm">
                Which layers of the U-Net would you target with LoRA for a style
                adaptation task? Think about what each layer type does.
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Cross-attention projection matrices ({' '}
                  <InlineMath math="W_Q, W_K, W_V, W_{\text{out}}" />
                  ). Style and subject are about how text concepts map to visual
                  features—and cross-attention is where text meets image.
                  Conv layers handle low-level spatial features (edges,
                  textures) and do not interact with text conditioning.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 2: Loss Function" color="cyan">
              <p className="text-sm">
                What is the loss function for diffusion LoRA training? Be
                specific—what does the model predict, and what is the target?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  MSE on noise prediction:{' '}
                  <InlineMath math="L = \text{MSE}(\varepsilon, \hat{\varepsilon})" />.
                  The model predicts the noise that was added to the latent, and
                  the loss is the same MSE from DDPM training. If you guessed
                  cross-entropy or next-token prediction, you were thinking of
                  LLM LoRA—the mechanism transfers, but the training context
                  does not.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 3: Data Requirements" color="cyan">
              <p className="text-sm">
                How many training images do you need for a style LoRA?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  50–200 images for a style LoRA, 5–20 for a subject LoRA. The
                  model already knows how to generate diverse images—the LoRA
                  only encodes the delta. Overfitting (reproducing training
                  images) is a bigger risk than underfitting with too few.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Calibrating Confidence">
            Most people get #1 partially right (attention layers), #2 wrong
            (they guess cross-entropy), and #3 wrong (they guess too many).
            The LoRA mechanism transfers directly—but the training context
            changes completely.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Where LoRA Goes in the U-Net */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where LoRA Goes in the U-Net"
            subtitle="Of course it's the cross-attention projections"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Think about what you want to change. A style LoRA teaches the
              model a new visual style—&ldquo;watercolor,&rdquo;
              &ldquo;pixel art,&rdquo; &ldquo;oil painting.&rdquo; A subject
              LoRA teaches it a specific face, pet, or object. In both cases,
              you are changing <strong>how text concepts map to visual
              features</strong>. Where in the U-Net does that mapping happen?
            </p>
            <p className="text-muted-foreground">
              Cross-attention. Specifically, the projection matrices{' '}
              <InlineMath math="W_Q" />, <InlineMath math="W_K" />,{' '}
              <InlineMath math="W_V" />, and <InlineMath math="W_{\text{out}}" />{' '}
              at every cross-attention block. You learned in{' '}
              <strong>Text Conditioning &amp; Guidance</strong> that{' '}
              <InlineMath math="W_Q" /> transforms spatial features into
              queries (what each pixel is looking for),{' '}
              <InlineMath math="W_K" /> and <InlineMath math="W_V" /> transform
              text embeddings into keys and values (what each word advertises
              and contributes). LoRA on these projections modifies the
              text-to-image translation—exactly what style and subject
              adaptation require.
            </p>
          </div>
          <div className="mt-4">
            <MermaidDiagram chart={`
              graph TD
                subgraph U-Net Residual Block
                  CONV1[Conv Block ❄️]:::frozen
                  CONV2[Conv Block ❄️]:::frozen
                  SELF[Self-Attention ❄️]:::frozen
                  subgraph Cross-Attention
                    WQ["W_Q + LoRA 🔥"]:::lora
                    WK["W_K + LoRA 🔥"]:::lora
                    WV["W_V + LoRA 🔥"]:::lora
                    WOUT["W_out + LoRA 🔥"]:::lora
                  end
                  NORM[Adaptive GroupNorm ❄️]:::frozen
                end

                CONV1 --> CONV2
                CONV2 --> SELF
                SELF --> WQ & WK & WV
                WQ & WK & WV --> WOUT
                WOUT --> NORM

                classDef frozen fill:#374151,stroke:#6b7280,color:#d1d5db
                classDef lora fill:#7c3aed,stroke:#a78bfa,color:#f5f3ff
            `} />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The conv blocks, self-attention layers, and adaptive group norm
              layers all stay frozen. They handle spatial processing—edges,
              textures, structural layout—that does not need to change for a new
              style. The cross-attention projections are where text meaning
              meets spatial features, and that is where style and subject live.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="&ldquo;Of Course&rdquo;">
            If you want to change what &ldquo;watercolor&rdquo; means visually,
            you change the projections that translate between text meaning and
            spatial features. The conv layers handle edges and textures at a
            low level—they do not know about &ldquo;watercolor.&rdquo;
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* LoRA-enabled vs frozen layers */}
      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-2">
            <GradientCard title="LoRA-Enabled (Trainable)" color="violet">
              <ul className="space-y-1 text-sm">
                <li>• <InlineMath math="W_Q" /> in cross-attention blocks</li>
                <li>• <InlineMath math="W_K" /> in cross-attention blocks</li>
                <li>• <InlineMath math="W_V" /> in cross-attention blocks</li>
                <li>• <InlineMath math="W_{\text{out}}" /> in cross-attention blocks</li>
                <li>• At attention resolutions (16×16 and 32×32)</li>
              </ul>
            </GradientCard>
            <GradientCard title="Frozen (Unchanged)" color="blue">
              <ul className="space-y-1 text-sm">
                <li>• All conv blocks (~70% of U-Net params)</li>
                <li>• Self-attention layers</li>
                <li>• Adaptive group norm layers</li>
                <li>• VAE encoder and decoder</li>
                <li>• CLIP text encoder</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Negative example: conv-only LoRA */}
      <Row>
        <Row.Content>
          <GradientCard title="Negative Example: LoRA on Conv Layers Only" color="rose">
            <div className="space-y-2 text-sm">
              <p>
                What if you applied LoRA to the conv layers instead of
                cross-attention? The conv layers process spatial features—edges,
                textures, gradients—but they never interact with text
                conditioning. They do not know what &ldquo;watercolor&rdquo;
                means because that concept only enters the network through
                cross-attention.
              </p>
              <p>
                Result: minimal style effect. The LoRA adapters would learn
                low-level spatial adjustments (slightly different edge
                processing) but could not steer the text-to-image mapping
                because they do not sit on that pathway. The mechanism alone is
                not enough—<strong>where</strong> you apply it is critical.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Location Matters">
            LoRA on the wrong layers is like installing a better steering wheel
            in the trunk. The mechanism is fine, but it is not connected to the
            system it needs to influence.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: The Diffusion LoRA Training Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Diffusion LoRA Training Loop"
            subtitle="Same DDPM training. Same LoRA adapters. Combined."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know two training loops: DDPM training (from{' '}
              <strong>Learning to Denoise</strong>) and LLM LoRA training (from{' '}
              <strong>LoRA and Quantization</strong>). Diffusion LoRA training
              is their intersection. Compare them side by side—notice that
              roughly 80% of the loop is identical:
            </p>
          </div>
          <div className="mt-4">
            <ComparisonRow
              left={{
                title: 'LLM LoRA Training Step',
                color: 'blue',
                items: [
                  '1. Load text sequence from dataset',
                  '2. Tokenize text',
                  '3. Forward pass: predict next token',
                  '4. Loss: cross-entropy on token prediction',
                  '5. Backprop into LoRA params only',
                  '6. Optimizer step (LoRA params only)',
                ],
              }}
              right={{
                title: 'Diffusion LoRA Training Step',
                color: 'violet',
                items: [
                  '1. Load (image, caption) pair from dataset',
                  '2. Encode image with frozen VAE, tokenize caption',
                  '3. Sample random t, add noise to latent',
                  '4. Forward pass: U-Net predicts noise',
                  '5. Loss: MSE(ε, ε̂) on noise prediction',
                  '6. Backprop into LoRA params only',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Steps 5–6 on the left and 6 on the right are identical: freeze
              the base model, only LoRA adapter parameters receive gradients,
              only they are updated. The LoRA bypass mechanism does not change.
              What changes is the <strong>context</strong>: the data format
              (images+captions vs text), the preprocessing (VAE encoding +
              noise addition vs tokenization), what the model predicts (noise
              vs next token), and the loss function (MSE vs cross-entropy).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Detour, Different Highway">
            The LoRA bypass is the same small trainable detour from{' '}
            <strong>LoRA and Quantization</strong>. The highway it attaches to
            is different (diffusion U-Net cross-attention instead of LLM
            self-attention). The traffic on the highway is different (spatial
            features and text embeddings instead of token sequences). But the
            detour mechanism is identical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Worked Example: One Training Step */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: One Training Step"
            subtitle="Training a watercolor style LoRA, end to end"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&rsquo;s trace one complete training step for a watercolor
              style LoRA. You have seen this pattern before—a full training step
              trace—in <LessonLink slug="learning-to-denoise">Learning to Denoise</LessonLink>. The only
              differences are the VAE encoding and the LoRA adapter setup.
            </p>
          </div>
          <div className="mt-4">
            <CodeBlock
              code={`# 1. Load training image + caption
image = load("watercolor_village.png")     # [3, 512, 512]
caption = "a watercolor painting of a village"

# 2. Encode image with frozen VAE
z_0 = vae.encode(image)                    # [4, 64, 64]

# 3. Sample random timestep
t = 500

# 4. Add noise using forward process (you know this formula)
epsilon = torch.randn_like(z_0)            # [4, 64, 64]
z_t = sqrt(alpha_bar_500) * z_0 + sqrt(1 - alpha_bar_500) * epsilon

# 5. Encode caption with frozen CLIP
text_emb = clip.encode(caption)            # [1, 77, 768]

# 6. U-Net predicts noise (only LoRA params have gradients)
epsilon_hat = unet(z_t, t, text_emb)       # [4, 64, 64]

# 7. Compute loss—same MSE from DDPM training
loss = MSE(epsilon, epsilon_hat)

# 8. Backprop—gradients flow only through LoRA adapters
loss.backward()   # base W: no grad. LoRA B, A: grad ✓
optimizer.step()  # updates only LoRA params`}
              language="python"
              filename="one_training_step.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Every line maps to something you already know. Steps 2–4 are the
              forward process from <LessonLink slug="the-forward-process">The Forward Process</LessonLink>, now
              applied in latent space (from{' '}
              <LessonLink slug="from-pixels-to-latents">From Pixels to Latents</LessonLink>). Step 6 is the same
              U-Net forward pass from{' '}
              <LessonLink slug="stable-diffusion-architecture">The Stable Diffusion Pipeline</LessonLink>. Steps 7–8 are
              DDPM training loss from <LessonLink slug="learning-to-denoise">Learning to Denoise</LessonLink> with
              LoRA&rsquo;s frozen-base-weights pattern from{' '}
              <LessonLink slug="lora-and-quantization">LoRA and Quantization</LessonLink>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Tensor Shape Check">
            <ul className="space-y-1 text-sm">
              <li>• Image: [3, 512, 512]</li>
              <li>• Latent z₀: [4, 64, 64]</li>
              <li>• Noise ε: [4, 64, 64]</li>
              <li>• Noised latent z_t: [4, 64, 64]</li>
              <li>• Text embedding: [1, 77, 768]</li>
              <li>• Predicted noise ε̂: [4, 64, 64]</li>
            </ul>
            <p className="text-sm mt-2">
              Every shape is familiar from prior lessons.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 7: Check — Predict-and-Verify */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check Your Understanding"
            subtitle="Predict, then verify"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1: Style Leakage" color="cyan">
              <p className="text-sm">
                You trained a style LoRA on watercolor images with captions
                mentioning &ldquo;watercolor.&rdquo; You generate with the
                prompt &ldquo;a photograph of a mountain.&rdquo; What would
                you expect?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The watercolor style would likely still apply. The LoRA
                  modified the cross-attention projection matrices that process{' '}
                  <strong>all</strong> text-to-image mappings, not just ones
                  containing &ldquo;watercolor.&rdquo; Every prompt passes
                  through the same modified projections. The style is baked
                  into the projections themselves, not gated by specific words.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Question 2: Merge and Speed" color="cyan">
              <p className="text-sm">
                You merge the LoRA adapters into the base weights ({' '}
                <InlineMath math="W_{\text{merged}} = W + BA \cdot \frac{\alpha}{r}" />
                ) and remove the adapter layers. Does inference speed change?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. The merged weight matrix has the same shape as the
                  original. Zero additional computation at inference time. This
                  is identical to what you learned in{' '}
                  <strong>LoRA and Quantization</strong> for LLMs—the merge
                  operation transfers directly.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 8: Style vs Subject LoRA */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Style vs Subject LoRA"
            subtitle="Same mechanism, different data strategies"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The LoRA mechanism is identical for style and subject adaptation.
              What differs is the training data strategy and how the model
              learns what you want it to learn.
            </p>
          </div>
          <div className="mt-4">
            <ComparisonRow
              left={{
                title: 'Style LoRA',
                color: 'violet',
                items: [
                  '50–200 diverse images in the target style',
                  'Varied subjects (landscapes, portraits, objects)',
                  'Captions describe both style and content',
                  'Model learns the stylistic delta across all subjects',
                  'Risk: style inconsistency if dataset is too varied',
                ],
              }}
              right={{
                title: 'Subject LoRA',
                color: 'emerald',
                items: [
                  '5–20 images of the specific subject',
                  'Varied poses, lighting, and backgrounds',
                  'Rare-token identifier (e.g., "a photo of sks dog")',
                  'Model learns the identity delta for one subject',
                  'Risk: face distortion, overfitting to training poses',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Subject LoRAs use a rare-token identifier like
              &ldquo;sks&rdquo; to avoid polluting common words. Without it,
              training on images of your dog with captions like &ldquo;a photo
              of a dog&rdquo; would shift the meaning of &ldquo;dog&rdquo; for{' '}
              <em>all</em> generations. Using &ldquo;a photo of sks dog&rdquo;
              anchors the new identity to an unused token, leaving
              &ldquo;dog&rdquo; intact.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why So Few Images?">
            The model already knows how to generate diverse images of dogs,
            landscapes, and faces. The LoRA only needs to encode the{' '}
            <em>delta</em>—the difference between the base model&rsquo;s
            understanding and your specific style or subject. Finetuning is a
            refinement, not a revolution.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Rank and Alpha for Diffusion */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Rank and Alpha for Diffusion"
            subtitle="Smaller delta, smaller rank"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="lora-and-quantization">LoRA and Quantization</LessonLink>, you learned that LLM
              LoRA commonly uses rank 8–64. Diffusion LoRA typically uses{' '}
              <strong>rank 4–8</strong>—lower. Why?
            </p>
            <p className="text-muted-foreground">
              Style adaptation is a smaller delta than instruction-following or
              chat behavior. The base model already generates high-quality
              images—you are only shifting the text-to-image mapping slightly.
              A lower rank constrains the adaptation to a smaller subspace,
              which acts as implicit regularization against overfitting on the
              small training datasets typical of diffusion LoRA.
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-3 mt-4">
            <GradientCard title="r = 1–2" color="amber">
              <p className="text-sm">Very constrained. Barely shifts the style. May underfit for complex style adaptation.</p>
            </GradientCard>
            <GradientCard title="r = 4–8" color="emerald">
              <p className="text-sm">The sweet spot for most style and subject LoRAs. Enough capacity without overfitting.</p>
            </GradientCard>
            <GradientCard title="r = 16–64" color="rose">
              <p className="text-sm">Risk of overfitting on small datasets. The model memorizes training images rather than learning the style.</p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Alpha Scaling">
            Alpha is typically set to 1.0 or equal to the rank. The{' '}
            <InlineMath math="\alpha / r" /> scaling factor controls how much
            the detour influences the output. With{' '}
            <InlineMath math="\alpha = r" />, the scaling is 1.0 and the
            learning rate alone controls the step size.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Multiple LoRA Composition */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multiple LoRA Composition"
            subtitle="Stacking adapters—plugins for a modular system"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember from{' '}
              <strong>The Stable Diffusion Pipeline</strong> that the SD
              components are modular—independently trained and swappable. LoRA
              adapters extend this modularity: you can apply a style LoRA{' '}
              <em>and</em> a subject LoRA simultaneously. Mechanically, both
              bypasses are summed:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <InlineMath math="W_{\text{combined}} = W + BA_{\text{style}} \cdot \frac{\alpha_1}{r_1} + BA_{\text{subject}} \cdot \frac{\alpha_2}{r_2}" />
            </div>
            <p className="text-muted-foreground">
              Each LoRA adapter is a small file (typically 2–50 MB) that can be
              loaded and unloaded independently. This is the same modularity
              principle from the SD pipeline—components connected by tensor
              interfaces—applied at a finer granularity.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Composition Interference">
            Two LoRAs can interfere if they modify the same projection matrices
            in conflicting directions. A style LoRA pushing toward
            &ldquo;watercolor softness&rdquo; and a subject LoRA pushing toward
            &ldquo;sharp facial features&rdquo; may produce blurry faces. Scale
            each LoRA&rsquo;s alpha to control the blend.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Overfitting and Timestep Effects */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practical Nuances"
            subtitle="Overfitting symptoms and timestep effects"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Overfitting:</strong> The most common failure mode.
              Generated images start looking like copies of the training
              images—same composition, same poses, same backgrounds—rather than
              novel compositions in the learned style. The model memorized
              specific images instead of learning the style delta.
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Fewer training steps (stop early)</li>
              <li>Lower rank (constrains the adaptation subspace)</li>
              <li>More diverse training images (harder to memorize)</li>
              <li>Lower learning rate (smaller updates per step)</li>
            </ul>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              <strong>LoRA&rsquo;s effect across timesteps:</strong> Remember
              the coarse-to-fine denoising progression from{' '}
              <strong>Sampling and Generation</strong>. Cross-attention has
              different influence at different noise levels. At high noise
              (t=900), spatial features are largely noise, so cross-attention
              influence is limited. At low noise (t=50), spatial features are
              well-formed and cross-attention steers fine details—this is where
              style LoRAs have their strongest visible effect, shaping textures,
              brushstrokes, and rendering style.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Coarse-to-Fine Interaction">
            LoRA modifies the same cross-attention projections at every
            timestep—the weights do not change between steps. But the{' '}
            <em>effect</em> is stronger at lower noise levels where spatial
            features are well-formed enough for cross-attention to meaningfully
            steer the rendering.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 9: Transfer Check */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Check"
            subtitle="Probing the boundaries"
          />
          <div className="space-y-4">
            <GradientCard title="Edge Case: Color Palette" color="cyan">
              <p className="text-sm">
                You want the model to generate images with a specific color
                palette (only blues and greens). Would LoRA on cross-attention
                projections be the best approach, or would you target different
                layers?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Color palette is arguably a lower-level visual property.
                  Cross-attention LoRA would work if the captions describe
                  colors, but targeting the decoder-side conv layers might be
                  more direct for pure color shifts. This is a genuine edge
                  case where the &ldquo;of course cross-attention&rdquo;
                  heuristic meets its boundary. Not every adaptation maps
                  cleanly to the text-to-image interface.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Mechanism vs Context" color="cyan">
              <p className="text-sm">
                Can you train a diffusion LoRA using the LLM LoRA training
                loop (next-token prediction on text)? Why or why not?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  No. The U-Net does not predict text tokens. It predicts noise
                  vectors. The loss function, input format, and output format
                  are all different. The LoRA mechanism (bypass architecture,
                  B=0 init, merge) transfers directly. The training procedure
                  (what data, what loss, what the model predicts) does not.
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
            title="Practice: LoRA Training and Composition"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook grounds the architectural understanding in real
                models, traces the training loop with real tensors, and
                culminates in training your own style LoRA.
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
                  <strong>Exercise 1 (Guided):</strong> Inspect LoRA target
                  layers. Load a pretrained SD model, list all cross-attention
                  projection layers by name, compute total LoRA params for
                  rank-4 vs rank-16, compare to total U-Net params.
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> One LoRA training step
                  by hand. Encode one image with VAE, sample a timestep, add
                  noise, run U-Net forward pass with LoRA adapters, compute MSE
                  loss, print gradients to verify base weights have none.
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Train a style LoRA.
                  Use diffusers + PEFT to train a LoRA on a small style dataset
                  (~20–50 images) for a few hundred steps. Generate with and
                  without the LoRA to see the effect. Compare different rank
                  values.
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> LoRA composition
                  experiment. Load two pre-trained LoRA adapters, apply
                  individually and together. Compare outputs. Experiment with
                  scaling alpha to control the blend.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Inspect the architecture (which layers, how many params)</li>
              <li>Trace one training step (verify gradients flow correctly)</li>
              <li>Train end-to-end (see the style effect)</li>
              <li>Compose and blend (advanced usage)</li>
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
                  'Same bypass mechanism, different target system.',
                description:
                  'LoRA for diffusion uses the same bypass architecture as LoRA for LLMs—the detour is the same, the highway is different. The LoRA mechanism (B=0 init, low-rank bypass, merge at inference) transfers directly from Module 4.4.',
              },
              {
                headline:
                  'Cross-attention projections are the primary target.',
                description:
                  'Style and subject live in the text-to-image mapping. Cross-attention is where text meaning meets spatial features. LoRA on W_Q, W_K, W_V, and W_out modifies that mapping. Conv layers handle low-level spatial processing and do not interact with text conditioning.',
              },
              {
                headline:
                  'The training loop is DDPM training with frozen base weights.',
                description:
                  'Same noise-prediction MSE loss. Same random timestep sampling. Same forward process. The only change: freeze the U-Net base weights and let only LoRA adapter parameters receive gradients and updates.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>Same detour, different highway.</strong> The LoRA bypass is
            the same trainable shortcut from Module 4.4. The highway it
            attaches to is the diffusion U-Net&rsquo;s cross-attention
            projections. The traffic is spatial features and text embeddings
            instead of token sequences. The detour mechanism is identical.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'LoRA: Low-Rank Adaptation of Large Language Models',
                authors: 'Hu et al., 2021',
                url: 'https://arxiv.org/abs/2106.09685',
                note: 'The original LoRA paper. You read the key results in Module 4.4. Sections 4.1–4.2 cover the design choices.',
              },
              {
                title: 'An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion',
                authors: 'Gal et al., 2022',
                url: 'https://arxiv.org/abs/2208.01618',
                note: 'Alternative approach to subject personalization. Optimizes a text embedding rather than model weights. Covered in a future lesson.',
              },
              {
                title: 'DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation',
                authors: 'Ruiz et al., 2022',
                url: 'https://arxiv.org/abs/2208.12242',
                note: 'Full fine-tuning approach for subject personalization. The rare-token identifier concept (e.g., "sks") originates here.',
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
            title="Up Next: Img2Img and Inpainting"
            description="LoRA customizes the model's weights. But you can also customize the inference process itself—start from a real image instead of pure noise, or selectively edit just part of an image."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
