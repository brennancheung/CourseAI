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
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-5-2-img2img-and-inpainting.ipynb'

/**
 * Img2Img and Inpainting
 *
 * Lesson 2 in Module 6.5 (Customization). Lesson 19 overall in Series 6.
 * Cognitive load: BUILD (2 new concepts).
 *
 * Two genuinely new concepts:
 *   1. Img2img as partial denoising (start from a noised real image instead
 *      of pure noise; strength parameter maps to a starting timestep)
 *   2. Inpainting as per-step spatial masking (apply a binary mask at each
 *      denoising step to control which regions are denoised vs preserved)
 *
 * Both are reconfigurations of inference mechanisms the student already
 * understands deeply. No new math, no training, no new architecture.
 *
 * Previous: LoRA Fine-Tuning (Module 6.5, Lesson 1)
 * Next: Textual Inversion (Module 6.5, Lesson 3)
 */

export function Img2imgAndInpaintingLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Img2Img and Inpainting"
            description="Same denoising process, different starting point. Reconfigure the inference pipeline you already know to edit real images and selectively replace regions—no training required."
            category="Customization"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Objective + Constraints
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how img2img starts the denoising loop from a noised real
            image instead of pure noise, and how inpainting applies a spatial
            mask at each denoising step to selectively edit regions of an
            image—both as purely inference-time modifications to the pipeline
            you already know.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You derived the forward process closed-form formula, built the
            denoising loop from scratch, explored the alpha-bar curve, and
            traced the full SD pipeline. This lesson reconfigures those
            pieces—no new math, no new architecture.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Img2img: starting the denoising loop from a noised real image',
              'The strength parameter: mapping to a starting timestep and nonlinear behavior',
              'Inpainting: per-step spatial masking to selectively edit regions',
              'Why inpainting boundaries blend seamlessly',
              'Practical use with StableDiffusionImg2ImgPipeline and StableDiffusionInpaintPipeline',
              'NOT: training any model (img2img and inpainting are inference-time only)',
              'NOT: ControlNet or structural conditioning (Series 7)',
              'NOT: specialized inpainting models with extra input channels (mentioned only)',
              'NOT: outpainting, SDEdit, depth-guided, or video inpainting',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Quick Recap
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Three pieces you already have"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Forward process closed-form</strong> (from{' '}
              <strong>The Forward Process</strong>): you can jump any image to
              any noise level in one step. The formula you derived:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon" />
            </div>
            <p className="text-muted-foreground">
              <strong>The denoising loop</strong> (from{' '}
              <strong>Build a Diffusion Model</strong>): start from pure noise{' '}
              <InlineMath math="z_T \sim \mathcal{N}(0, I)" />, iterate to{' '}
              <InlineMath math="z_0" />. You implemented this yourself.
            </p>
            <p className="text-muted-foreground">
              <strong>Alpha-bar as the signal-to-noise dial</strong> (from{' '}
              <strong>The Forward Process</strong>):{' '}
              <InlineMath math="\bar{\alpha}_t" /> near 1 means mostly
              signal; near 0 means mostly noise. You explored this
              interactively with the alpha-bar curve widget.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="All the Pieces">
            The forward process formula lets you noise any image to any level.
            The denoising loop removes that noise iteratively. The alpha-bar
            curve tells you what &ldquo;any level&rdquo; means. Img2img
            connects these three pieces in a way you could almost predict.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Hook—"What if you didn't start from pure noise?"
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What If You Didn't Start From Pure Noise?"
            subtitle="A challenge you can solve with what you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a photo of a landscape. You want to turn it into a
              watercolor painting of the same scene—same composition, same
              mountains, same sky, but in watercolor style. You could train a
              LoRA on watercolor images. Or you could do it{' '}
              <strong>right now</strong>, with zero training, using only the
              concepts you already have.
            </p>
            <GradientCard title="Think About It" color="cyan">
              <p className="text-sm">
                You know how to add a precise amount of noise to any image (the
                forward process formula). You know the model can denoise from
                any starting noise level. What if you noise your landscape
                photo partway, then let the model denoise with the prompt
                &ldquo;a watercolor painting of mountains&rdquo;?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal the insight
                </summary>
                <div className="mt-2 text-sm text-muted-foreground space-y-2">
                  <p>
                    The model starts denoising from your noised landscape
                    instead of from pure random noise. The original image
                    provides structural guidance—where the mountains are, where
                    the sky is—while the text prompt steers the style. More
                    noise means more creative freedom; less noise means more
                    faithfulness to the original.
                  </p>
                  <p>
                    That is img2img. You already had all the pieces.
                  </p>
                </div>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="No New Math">
            You are not learning a new algorithm. You are seeing a new
            application of the forward process formula and the denoising
            loop—two mechanisms you have used multiple times.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: Explain—Img2img Mechanism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Img2img Mechanism"
            subtitle="One change to the pipeline you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Standard text-to-image starts the denoising loop from pure noise:{' '}
              <InlineMath math="z_T \sim \mathcal{N}(0, I)" />. Img2img
              replaces that starting point with a noised version of a real
              image. Everything else—the sampler, CFG, the U-Net, the VAE
              decoder—stays identical.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="VAE Encoder Callback">
            In <strong>The Stable Diffusion Pipeline</strong>, we said the VAE
            encoder is not used during text-to-image inference. That was
            correct—text-to-image starts from random noise, so there is no
            image to encode. Img2img <em>does</em> have an input image, so
            the encoder is needed to convert it to latent space.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Pipeline comparison diagram */}
      <Row>
        <Row.Content>
          <div className="mt-2">
            <ComparisonRow
              left={{
                title: 'Text-to-Image Pipeline',
                color: 'blue',
                items: [
                  '1. Start: z_T ~ N(0, I) (pure noise)',
                  '2. Denoise from T to 0 (all steps)',
                  '3. VAE decode z_0 to pixel image',
                  'VAE encoder: NOT used',
                ],
              }}
              right={{
                title: 'Img2img Pipeline',
                color: 'violet',
                items: [
                  '1. VAE encode input image to z_0',
                  '2. Noise z_0 to z_{t_start} with forward process',
                  '3. Denoise from t_start to 0 (partial steps)',
                  '4. VAE decode z_0 to pixel image',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The <strong>one difference</strong>: the starting point of the
              denoising loop. Text-to-image starts from step T (pure noise).
              Img2img starts from step{' '}
              <InlineMath math="t_{\text{start}}" /> (partially noised real
              image). The denoising loop, CFG, the U-Net, the sampler—all
              identical.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Strength parameter */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Strength Parameter"
            subtitle="How much of the denoising process runs"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <strong>strength</strong> parameter determines where on the
              noise schedule you start. Strength = fraction of the denoising
              process that runs:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <strong>strength=0.8</strong>: 80% of the denoising steps run.
                Start at high noise—the model has wide creative latitude. Most
                of the original&rsquo;s structure is destroyed.
              </li>
              <li>
                <strong>strength=0.5</strong>: 50% of steps run. Moderate
                noise—broad composition preserved, significant detail changes.
              </li>
              <li>
                <strong>strength=0.2</strong>: 20% of steps run. Low
                noise—only fine details and textures change. The original is
                almost entirely preserved.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Alpha-Bar Connection">
            Map strength onto the alpha-bar curve you explored in{' '}
            <strong>The Forward Process</strong>. Low strength means starting
            where <InlineMath math="\bar{\alpha}" /> is high (mostly
            signal). High strength means starting where{' '}
            <InlineMath math="\bar{\alpha}" /> is low (mostly noise). The
            curve is nonlinear—and that explains why strength does not behave
            linearly.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Worked example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: Img2img End to End"
            subtitle="Tracing the pipeline with concrete values"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Input photo (512x512), DDIM sampler with 50 total steps,
              strength=0.7:
            </p>
          </div>
          <div className="mt-4">
            <CodeBlock
              code={`# 1. Encode input image with VAE
z_0 = vae.encode(image)               # [4, 64, 64]

# 2. Compute starting timestep from strength
#    strength=0.7 means 70% of steps run
#    With 50 DDIM steps, that's 35 denoising steps
#    t_start index = 50 - 35 = step 15
t_start = int(50 * 0.7)               # 35 steps to run

# 3. Noise z_0 to z_{t_start} using the forward process formula
epsilon = torch.randn_like(z_0)        # [4, 64, 64]
z_t_start = (sqrt(alpha_bar_t_start) * z_0
           + sqrt(1 - alpha_bar_t_start) * epsilon)

# 4. Denoise from t_start to t=0 (35 DDIM steps)
#    --- identical to standard txt2img from here ---
z = z_t_start
for t in schedule[15:]:                # steps 15 through 49
    noise_pred = unet(z, t, text_emb)  # predict noise
    z = ddim_step(z, noise_pred, t)    # DDIM update

# 5. VAE decode to pixel image
output = vae.decode(z)                 # [3, 512, 512]`}
              language="python"
              filename="img2img_trace.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Three new lines compared to text-to-image: VAE encode (line 2),
              compute starting timestep (line 6), forward process noise (lines
              9-11). The denoising loop (lines 14-17) is unchanged—it just
              starts partway through instead of at T.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Tensor Shape Check">
            <ul className="space-y-1 text-sm">
              <li>Input image: [3, 512, 512]</li>
              <li>Latent z_0: [4, 64, 64]</li>
              <li>Noise: [4, 64, 64]</li>
              <li>Noised latent z_{'{t_start}'}: [4, 64, 64]</li>
              <li>Output image: [3, 512, 512]</li>
            </ul>
            <p className="text-sm mt-2">
              Every shape is familiar from prior lessons.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Boundary cases and nonlinearity */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Boundary Cases"
            subtitle="The edges reveal the mechanism"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'strength = 1.0',
                color: 'amber',
                items: [
                  'Image noised to maximum (pure noise)',
                  'All denoising steps run',
                  'Original image completely destroyed',
                  'Output = standard text-to-image',
                  'The input image has zero influence',
                ],
              }}
              right={{
                title: 'strength = 0.0',
                color: 'emerald',
                items: [
                  'No noise added at all',
                  'Zero denoising steps run',
                  'Output = the original image unchanged',
                  'The text prompt has zero influence',
                  'Nothing happens',
                ],
              }}
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The strength=1.0 case is the key negative example. If img2img
              were &ldquo;modifying&rdquo; the image like a Photoshop filter,
              some input influence should remain even at maximum strength.
              Instead, the input is fully noised to random noise—
              indistinguishable from{' '}
              <InlineMath math="z_T \sim \mathcal{N}(0, I)" />. Img2img at
              strength=1.0 <strong>is</strong> standard text-to-image.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Image Blending">
            If img2img blended the original with the generated result
            (like alpha compositing), strength=0.5 would produce a ghostly
            double exposure—the same problem as pixel-space interpolation
            from <strong>Exploring Latent Spaces</strong>. Instead, img2img
            at strength=0.5 produces a single coherent image. The mechanism
            is <em>denoising</em>, not blending.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Why strength is nonlinear */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Strength Is Nonlinear"
            subtitle="The alpha-bar curve explains everything"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The strength parameter maps onto the alpha-bar curve—and that
              curve is nonlinear. Combined with the coarse-to-fine denoising
              progression from <strong>Sampling and Generation</strong>, the
              effect is qualitatively different at different strength values:
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-3 mt-4">
            <GradientCard title="Low Strength (0.1-0.3)" color="emerald">
              <ul className="space-y-1 text-sm">
                <li>Only detail-refinement steps run</li>
                <li>Textures and colors change</li>
                <li>Structure fully preserved</li>
                <li>Good for: color grading, style tinting</li>
              </ul>
            </GradientCard>
            <GradientCard title="Medium Strength (0.4-0.6)" color="blue">
              <ul className="space-y-1 text-sm">
                <li>Structure-setting + detail steps run</li>
                <li>Composition preserved, details change</li>
                <li>The &ldquo;sweet spot&rdquo; for editing</li>
                <li>Good for: style transfer, reinterpretation</li>
              </ul>
            </GradientCard>
            <GradientCard title="High Strength (0.7-0.9)" color="violet">
              <ul className="space-y-1 text-sm">
                <li>Most of the denoising process runs</li>
                <li>Only vague spatial hints survive</li>
                <li>Major creative reinterpretation</li>
                <li>Good for: sketch-to-image, dramatic restyling</li>
              </ul>
            </GradientCard>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The jump from strength=0.3 to strength=0.5 is qualitatively
              different from the jump from strength=0.7 to strength=0.9,
              because different denoising steps control different aspects of
              the image. Early steps (high noise) set structure. Late steps
              (low noise) refine details. The strength parameter determines
              which of these phases the model gets to run.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Coarse-to-Fine Explains It">
            Remember from <strong>Sampling and Generation</strong>: at t=900
            the model makes bold structural decisions; at t=50 it polishes
            textures. Strength determines which of these phases you allow the
            model to execute. Low strength = polishing only. High strength =
            structural reimagination.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Check—Predict-and-Verify for Img2img
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Img2img Predictions"
            subtitle="Predict, then verify"
          />
          <div className="space-y-4">
            <GradientCard title="Prediction 1: Dog to Cat" color="cyan">
              <p className="text-sm">
                You apply img2img with strength=0.5 to a photo of a dog, with
                the prompt &ldquo;a cat sitting on grass.&rdquo; What would you
                expect the output to look like?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The broad composition is preserved—similar pose, similar
                  background layout—but the subject is reinterpreted. At
                  strength=0.5, the structural steps have enough latitude to
                  change the animal, but the overall scene composition (grass,
                  sky, approximate object placement) is preserved from the
                  original.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 2: Same Dog, Low Strength" color="cyan">
              <p className="text-sm">
                You apply img2img with strength=0.1 to the same dog photo, same
                prompt (&ldquo;a cat sitting on grass&rdquo;). What changes?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Almost nothing. Only the finest details (color tints, texture
                  grain) change. The structure, shape, and identity of the dog
                  are fully preserved because the model only runs the
                  detail-refinement steps. The prompt &ldquo;a cat&rdquo; has
                  minimal influence because the coarse structure is locked in.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: Explain—Inpainting Mechanism
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Inpainting: Selective Editing"
            subtitle="What if you only want to change part of the image?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Img2img changes the entire image. But what if you only want to
              replace one object while keeping everything else? You need
              spatial selectivity—a way to tell the model &ldquo;change
              <em>this</em> region, preserve <em>everything else</em>.&rdquo;
            </p>
            <p className="text-muted-foreground">
              Imagine a landscape with a tree you want to replace with a
              fountain. At each denoising step, the model produces a prediction
              for the full image. You keep its prediction for the tree
              region—where the fountain will appear—and throw away its
              prediction for the rest of the image, replacing it with the
              original landscape re-noised to the current timestep. The model
              can only change the tree region. Everything else is preserved.
            </p>
            <p className="text-muted-foreground">
              In code, that is one line added to the denoising loop. At each
              step, after the model predicts the denoised latent, replace the{' '}
              <strong>unmasked</strong> regions with the original
              image&rsquo;s latent re-noised to the current timestep:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <BlockMath math="z_t^{\text{combined}} = m \cdot z_t^{\text{denoised}} + (1 - m) \cdot \text{forward}(z_0^{\text{original}}, t)" />
            </div>
            <p className="text-muted-foreground">
              Where <InlineMath math="m" /> is the binary mask (1 = edit this
              region, 0 = preserve). The denoised prediction fills the masked
              area. The re-noised original fills everything else.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Line">
            Inpainting adds exactly one operation to the denoising loop. The
            U-Net, the sampler, CFG, the VAE—all unchanged. The mask is the
            entire mechanism.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Why re-noise the original */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Why Re-Noise the Original for Unmasked Regions?" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  The denoising loop expects latents at the{' '}
                  <strong>current noise level</strong>. At step t, the
                  partially denoised latent has noise level t. If you inject
                  the clean original (<InlineMath math="z_0" />) into a tensor
                  at noise level t, the model encounters an inconsistency—one
                  region at noise level t, another at noise level 0—and
                  produces artifacts.
                </p>
                <p>
                  Re-noising the original to the current timestep using the
                  forward process formula keeps everything consistent. The
                  unmasked latents have the same noise level as the masked
                  latents at every step. As denoising progresses toward t=0,
                  both regions converge to clean latents together.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Formula Again">
            The re-noising step uses the exact same forward process
            closed-form formula from <strong>The Forward Process</strong>:{' '}
            <InlineMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon" />.
            Third context for this formula: (1) training, (2) img2img starting
            point, (3) inpainting per-step re-noising.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Inpainting pseudocode */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The One-Line Addition"
            subtitle="Img2img denoising loop + per-step mask"
          />
          <div className="mt-2">
            <CodeBlock
              code={`# Standard img2img denoising loop
z = z_t_start                          # noised input image

for t in schedule[start:]:
    # Model predicts noise (same as always)
    noise_pred = unet(z, t, text_emb)
    z_denoised = sampler_step(z, noise_pred, t)

    # --- THE ONE INPAINTING LINE ---
    # Replace unmasked regions with re-noised original
    z_original_at_t = forward_process(z_0_original, t)
    z = mask * z_denoised + (1 - mask) * z_original_at_t

output = vae.decode(z)                 # [3, 512, 512]`}
              language="python"
              filename="inpainting_loop.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Lines 10-11 are the entire inpainting mechanism. Everything else
              is identical to img2img. The mask controls which regions the
              model can change, and the forward process ensures consistency
              at every noise level.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Mask Convention">
            <InlineMath math="m = 1" /> means &ldquo;edit this region&rdquo;
            (use the model&rsquo;s prediction).{' '}
            <InlineMath math="m = 0" /> means &ldquo;preserve the
            original&rdquo; (use the re-noised input).
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Why boundaries blend */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Boundaries Blend Seamlessly"
            subtitle="The U-Net sees the full image at every step"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In traditional image editing, pasting a generated region into an
              existing image creates visible seams—color mismatches, edge
              artifacts. Inpainting is fundamentally different because of the
              denoising process itself.
            </p>
            <p className="text-muted-foreground">
              At each denoising step, the U-Net sees the{' '}
              <strong>full latent</strong>—both the denoised masked region and
              the original unmasked region. Remember from{' '}
              <strong>The U-Net Architecture</strong> that at the 8x8
              bottleneck resolution, each latent position has a global
              receptive field. The model&rsquo;s noise prediction for the
              masked region <em>accounts for the unmasked context</em>. It
              knows what surrounds the mask and produces predictions that
              match.
            </p>
            <p className="text-muted-foreground">
              This happens at every step—the model repeatedly sees the full
              image context and refines its predictions for the masked region
              accordingly. Seamless boundaries are a natural consequence of
              the denoising process, not a post-processing trick.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Cut-and-Paste">
            Cut-and-paste operates once, in pixel space, with no context.
            Inpainting operates iteratively, in latent space, with full
            context at every step. The U-Net&rsquo;s global receptive field
            at the bottleneck ensures the model &ldquo;sees&rdquo; the
            unmasked regions when predicting the masked ones.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Inpainting boundary cases */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Mask = Entire Image',
              color: 'amber',
              items: [
                'Every region is denoised',
                'No regions are preserved',
                'Collapses to standard img2img / txt2img',
                'The mask is the only difference between inpainting and standard denoising',
              ],
            }}
            right={{
              title: 'Mask = Nothing',
              color: 'emerald',
              items: [
                'No regions are denoised',
                'Every region is preserved',
                'Output = input image unchanged',
                'The model does no work',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 8: Check—Predict-and-Verify for Inpainting
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Inpainting Predictions"
            subtitle="Predict, then verify"
          />
          <div className="space-y-4">
            <GradientCard title="Prediction 1: Sky Replacement" color="cyan">
              <p className="text-sm">
                You inpaint the sky of a landscape photo with the prompt
                &ldquo;dramatic thunderstorm clouds.&rdquo; The mask covers
                only the sky. What happens at the boundary between sky and
                mountains?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The boundary blends naturally. The U-Net sees the mountain
                  latents in the unmasked region and the denoised sky in the
                  masked region at every step. Its predictions for the sky
                  near the boundary account for the mountains, producing
                  coherent lighting, color transitions, and overlap.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Prediction 2: Full-Image Mask" color="cyan">
              <p className="text-sm">
                What happens if you inpaint with a mask covering the entire
                image?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Every region is denoised, none are preserved. This is
                  identical to standard img2img. The mask is the entire
                  difference between inpainting and img2img—remove the mask,
                  and you get standard denoising.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Elaborate—Practical Considerations
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practical Considerations"
            subtitle="Making img2img and inpainting work well in practice"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Sketch-to-image:</strong> A rough hand-drawn sketch
              provides structural guidance at high strength (0.7-0.9); the
              model fills in realistic detail guided by the text prompt. The
              sketch does not need to be good—just enough to anchor the
              composition. This is one of img2img&rsquo;s most impressive
              practical applications.
            </p>
            <p className="text-muted-foreground">
              <strong>Mask sizing for inpainting:</strong> Too-tight masks
              leave insufficient context for the model to integrate new
              content with the surroundings. Slightly oversized masks produce
              better results because the model has room to create smooth
              transitions. Feathered (soft) mask edges can help with gradual
              blending.
            </p>
            <p className="text-muted-foreground">
              <strong>Prompt interaction with inpainting:</strong> The prompt
              guides what fills the masked region. Without a specific prompt,
              the model fills with contextually plausible content (context-
              aware fill). With a prompt, you direct the content—&ldquo;replace
              this tree with a fountain.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Mask Tip">
            When masking, err on the side of slightly too large rather than
            too tight. The model blends boundaries naturally—give it room to
            work. Padding the mask by 10-20% beyond the object boundary
            usually produces cleaner results.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Specialized inpainting models + customization spectrum */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Specialized Inpainting Models" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  Dedicated inpainting models exist (e.g., SD inpainting
                  models fine-tuned with an extra mask input channel to the
                  U-Net). These produce better results at mask boundaries
                  because the model explicitly sees the mask during the
                  forward pass—but the underlying mechanism is the same.
                  They are an optimization, not a fundamentally different
                  approach.
                </p>
              </div>
            </GradientCard>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              <strong>The customization spectrum:</strong> You now have three
              different customization strategies:
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-3 mt-2">
            <GradientCard title="LoRA (6.5.1)" color="violet">
              <ul className="space-y-1 text-sm">
                <li>Changes the model&rsquo;s <strong>weights</strong></li>
                <li>Requires training data + training loop</li>
                <li>Persistent style/subject changes</li>
              </ul>
            </GradientCard>
            <GradientCard title="Img2img / Inpainting" color="blue">
              <ul className="space-y-1 text-sm">
                <li>Changes the <strong>inference process</strong></li>
                <li>No training required</li>
                <li>Per-image editing control</li>
              </ul>
            </GradientCard>
            <GradientCard title="Textual Inversion (next)" color="cyan">
              <ul className="space-y-1 text-sm">
                <li>Changes the <strong>embeddings</strong></li>
                <li>Training but no weight changes</li>
                <li>New concept via a single token</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Knobs">
            LoRA: change the weights. Img2img/inpainting: change the
            inference. Textual inversion: change the embeddings. Three
            different knobs on three different parts of the pipeline.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: Check—Transfer Questions
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Check"
            subtitle="Apply the mechanism to new situations"
          />
          <div className="space-y-4">
            <GradientCard title="Outpainting" color="cyan">
              <p className="text-sm">
                A colleague says they want to &ldquo;outpaint&rdquo;—extend an
                image beyond its borders, adding new content that seamlessly
                continues the existing scene. You have not seen outpainting
                before. Using what you learned about inpainting, can you
                describe how it might work?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  Pad the image with blank/noise pixels, create a mask that
                  covers the new regions (and slightly overlaps the original
                  edges for blending), then run inpainting. The model denoises
                  the masked extension regions using the unmasked original as
                  context. It is inpainting with the mask on the outside
                  instead of the inside.
                </p>
              </details>
            </GradientCard>
            <GradientCard title="Sampler Choice" color="cyan">
              <p className="text-sm">
                Your friend asks: &ldquo;Does img2img work better with DDIM or
                DPM-Solver?&rdquo; How would you reason about this?
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary text-sm">
                  Reveal
                </summary>
                <p className="mt-2 text-sm text-muted-foreground">
                  The sampler is orthogonal to the img2img mechanism. Img2img
                  changes the starting point; the sampler determines how to
                  get from there to z_0. Any sampler works. The same practical
                  guidance from <strong>Samplers and Efficiency</strong>{' '}
                  applies: DPM-Solver++ at 20-30 steps for efficiency, DDIM
                  for reproducibility. The sampler does not &ldquo;know&rdquo;
                  whether it started from pure noise or a noised image.
                </p>
              </details>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: Practice—Notebook Exercises
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Img2img and Inpainting"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook grounds these mechanisms in real images. You will
                see the strength spectrum firsthand, trace the img2img
                pipeline by hand, create masks for inpainting, and combine
                both techniques in a creative workflow.
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
                  <strong>Exercise 1 (Guided):</strong> Img2img strength
                  exploration. Load StableDiffusionImg2ImgPipeline, run the
                  same image at strengths 0.1, 0.3, 0.5, 0.7, 0.9 with a
                  fixed prompt and seed. Display results in a grid. Before
                  running, predict which strength will preserve the most
                  original structure.
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> Trace the img2img
                  mechanism by hand. Encode an image with VAE, manually add
                  noise using the forward process formula at a specific
                  timestep (corresponding to strength=0.7), then run the
                  standard denoising loop from that timestep. Compare the
                  output to the img2img pipeline&rsquo;s output—they should
                  match.
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Inpainting with mask
                  creation. Load StableDiffusionInpaintPipeline, create a
                  binary mask for a specific region of an image, run
                  inpainting with a descriptive prompt. Experiment with tight
                  vs generous mask sizing and observe boundary quality.
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> Creative
                  application. Use img2img at high strength to turn a sketch
                  into a realistic image. Then use inpainting to modify a
                  specific element of the result. Combine both techniques in a
                  practical workflow: sketch, img2img, selective inpainting
                  refinement.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>See the strength spectrum (build intuition)</li>
              <li>Trace the mechanism (verify understanding)</li>
              <li>Inpainting hands-on (new technique)</li>
              <li>Combine both (creative workflow)</li>
            </ol>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Summary
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Img2img uses the forward process formula to noise a real image, then denoises from there.',
                description:
                  'Instead of starting from pure noise, start from a noised version of a real image. The strength parameter determines the starting point on the alpha-bar curve—low strength preserves structure, high strength allows creative reinterpretation.',
              },
              {
                headline:
                  'Inpainting adds a per-step spatial mask to the denoising loop.',
                description:
                  'At each step, masked regions use the model\'s denoised prediction while unmasked regions are replaced with the re-noised original. Boundaries blend naturally because the U-Net sees the full image at every step.',
              },
              {
                headline:
                  'Neither technique requires training—they are purely inference-time modifications.',
                description:
                  'No new math, no new architecture, no training loop. Just two clever reconfigurations of the denoising process you already know. The same U-Net, VAE, and CLIP from text-to-image work unchanged.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Mental model echo */}
      <Row>
        <Row.Content>
          <InsightBlock title="Mental Model">
            <strong>Same denoising process, different starting point (img2img)
            or selective application (inpainting).</strong> The pipeline you
            traced across 17 lessons is unchanged. Img2img moves the starting
            line. Inpainting adds a spatial filter. Both use the forward
            process formula you derived in <strong>The Forward Process</strong>
            —now in its third and fourth applications.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations',
                authors: 'Meng et al., 2021',
                url: 'https://arxiv.org/abs/2108.01073',
                note: 'The paper that formalized the "noise and denoise" approach to image editing. Section 3 describes the core mechanism used by img2img.',
              },
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach et al., 2022',
                url: 'https://arxiv.org/abs/2112.10752',
                note: 'The Stable Diffusion paper. Section 3.3 covers conditional latent diffusion for inpainting.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The original DDPM paper. The forward process formula used by both img2img and inpainting originates here.',
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
            title="Up Next: Textual Inversion"
            description="LoRA customized the model's weights. Img2img and inpainting customized the inference process. What if you could teach the model a new concept—your pet, your art style—without changing any weights at all? What if you could optimize a single embedding vector in CLIP's space to represent something the model has never seen?"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
