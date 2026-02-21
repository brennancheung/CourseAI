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
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-3-2-latent-consistency-and-turbo.ipynb'

/**
 * Latent Consistency & Turbo
 *
 * Lesson 2 in Module 7.3 (Fast Generation). Lesson 7 of 11 in Series 7.
 * Cognitive load: BUILD (2 concepts that apply known patterns at scale:
 * LCM as consistency distillation on latent diffusion, ADD as adversarial distillation).
 *
 * Core concepts:
 * 1. LCM / LCM-LoRA (consistency distillation applied to SD/SDXL latent diffusion) -- builds on DEVELOPED consistency distillation
 * 2. Adversarial diffusion distillation (ADD / SDXL Turbo) -- genuinely new adversarial training concept
 *
 * Builds on: consistency distillation from 7.3.1, latent diffusion from 6.3.5,
 * LoRA from 6.5.1 and 4.4.4, SD pipeline from 6.4.1.
 *
 * Previous: Consistency Models (Module 7.3, Lesson 1 / STRETCH)
 * Next: The Speed Landscape (Module 7.3, Lesson 3)
 */

export function LatentConsistencyAndTurboLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Latent Consistency & Turbo"
            description="Two production-ready approaches to 1-4 step image generation: Latent Consistency Models apply the consistency distillation you already know to Stable Diffusion's latent space, while SDXL Turbo adds a discriminator for sharper single-step results."
            category="Fast Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how consistency distillation scales from toy 2D data to
            real Stable Diffusion models (LCM and LCM-LoRA for plug-and-play 1-4
            step generation), and how adversarial diffusion distillation (ADD /
            SDXL Turbo) provides an alternative using a discriminator&rsquo;s
            realism signal instead of ODE consistency alone.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="BUILD Lesson">
            This lesson applies two concepts you already have at DEVELOPED
            depth—consistency distillation and LoRA—to a real-world system you
            already know (Stable Diffusion). The new idea is adversarial
            distillation, introduced at a conceptual level.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'LCM: how consistency distillation applies to SD/SDXL latent diffusion',
              'LCM-LoRA: capturing distillation as a plug-and-play LoRA adapter',
              'Adversarial diffusion distillation (ADD): the hybrid diffusion + adversarial loss',
              'ADD vs consistency distillation: different teacher signals, different failure modes',
              'NOT: Training LCM or LCM-LoRA from scratch',
              'NOT: Discriminator architecture or full ADD loss derivation',
              'NOT: GAN theory beyond the minimum needed for ADD',
              'NOT: SDXL architecture details (Module 7.4)',
              'NOT: Comprehensive speed comparison (next lesson: The Speed Landscape)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Three concepts you will build on in the next thirty minutes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>Consistency distillation</strong> (from{' '}
              <LessonLink slug="consistency-models">Consistency Models</LessonLink>). You trained a consistency
              model by distilling from a pretrained teacher. The teacher takes
              one ODE step between adjacent timesteps. The consistency model
              learns to map both points to the same endpoint. The result:
              generation in 1-4 steps instead of 50.
            </p>
            <p className="text-muted-foreground">
              <strong>Latent diffusion</strong> (from{' '}
              <strong>From Pixels to Latents</strong>). Stable Diffusion runs
              diffusion in the VAE&rsquo;s latent space: 64&times;64&times;4
              tensors instead of 512&times;512&times;3 images. The VAE encodes
              images to latents and decodes latents back to images. The U-Net
              denoises in latent space.
            </p>
            <p className="text-muted-foreground">
              <strong>LoRA</strong> (from{' '}
              <strong>LoRA Fine-Tuning</strong>). LoRA captures a behavior
              change as a low-rank bypass:{' '}
              <InlineMath math="Wx + BAx" />. The bypass is a small file (2-50
              MB) that can be loaded into any compatible base model. You have
              used it for style and subject adaptation.
            </p>
            <p className="text-muted-foreground">
              What happens when you apply consistency distillation to Stable
              Diffusion&rsquo;s latent space? And what if you captured that
              distillation as a LoRA?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Bridge">
            <LessonLink slug="consistency-models">Consistency Models</LessonLink> ended with a promise:
            &ldquo;The next lesson takes this to real scale.&rdquo; You have
            the theory from toy 2D data. Now we see it work on actual images.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — Before/After Reveal */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The 4 MB Difference"
            subtitle="What a tiny file can do to generation speed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider three scenarios, all using the same SD v1.5 model and
              the same prompt:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="50 Steps (Standard)" color="blue">
                <div className="space-y-2 text-sm">
                  <p>
                    The baseline you know. ~10 seconds. Good quality—the full
                    denoising trajectory from noise to image.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    50 U-Net evaluations.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="4 Steps (No Adapter)" color="rose">
                <div className="space-y-2 text-sm">
                  <p>
                    Same model, 4 steps. ~0.8 seconds. Blurry, incoherent. The
                    model was not trained for this—it expects 20-50 steps to
                    refine details.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    The model is out of its comfort zone.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="4 Steps + LCM-LoRA" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    Same model, one LoRA loaded, 4 steps. ~0.8 seconds.
                    Near-baseline quality. The LoRA changed what the model does
                    at each step.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    A 4 MB file makes the difference.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              The only difference between the second and third scenarios is a 4
              MB LoRA file. This lesson explains what that file contains, how it
              was created, and why it works. Then we see an alternative
              approach—adversarial distillation—that pushes 1-step quality even
              further.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Speed, Different Quality">
            Both 4-step runs take the same time (~0.8s). The difference is
            entirely in the model weights. LCM-LoRA modifies what the model
            does at each step so that 4 steps produce good results.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — LCM Part A: "Same recipe, bigger kitchen" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Latent Consistency Models (LCM)"
            subtitle="Same recipe, bigger kitchen"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consistency distillation on 2D point clouds worked like this: the
              teacher takes one ODE step, the student learns to match
              predictions at adjacent timesteps. LCM does the same thing, but
              the data points are latent codes from the SD VAE instead of 2D
              coordinates.
            </p>

            <GradientCard title="Everything Scales Up Cleanly" color="blue">
              <ul className="space-y-2 text-sm">
                <li>
                  &bull; The 2D noise vector becomes a{' '}
                  <InlineMath math="[4, 64, 64]" /> latent tensor
                </li>
                <li>
                  &bull; The 2D diffusion model becomes the SD U-Net (with text
                  conditioning)
                </li>
                <li>
                  &bull; The 2D ODE trajectory becomes the latent-space ODE
                  trajectory
                </li>
                <li>
                  &bull; The training procedure is structurally identical—the
                  same distillation loss{' '}
                  <InlineMath math="\mathcal{L}_\text{CD}" /> you saw in{' '}
                  <LessonLink slug="consistency-models">Consistency Models</LessonLink>
                </li>
              </ul>
            </GradientCard>

            <p className="text-muted-foreground">
              The key adaptations from pixel-space consistency models to LCM:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-3 ml-4">
              <li>
                <strong>Teacher model:</strong> A pretrained SD/SDXL model (the
                full U-Net with all conditioning)
              </li>
              <li>
                <strong>Student model:</strong> Same architecture, initialized
                from the teacher&rsquo;s weights
              </li>
              <li>
                <strong>ODE solver for teacher step:</strong> Uses an augmented
                PF-ODE that incorporates classifier-free guidance directly into
                the ODE (so the teacher&rsquo;s trajectory already accounts for
                the text prompt)
              </li>
              <li>
                <strong>Noise schedule:</strong> Adapted for the latent space
                statistics
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Augmented PF-ODE">
            One key innovation in LCM: the ODE trajectory used for distillation
            incorporates the CFG guidance direction. The consistency model learns
            to produce <strong>text-faithful</strong> images in one step, not
            just any plausible image.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Address Misconception: LCM is not a new architecture */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Despite the name &ldquo;Latent Consistency Model,&rdquo; LCM is
              not a new model architecture. It is a standard SD/SDXL U-Net whose
              weights have been modified through consistency distillation. The
              VAE and CLIP text encoder are completely unchanged. The
              architecture is the same—only the U-Net weights and the scheduler
              are different.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not a New Model—a Distilled Version">
            LCM starts from a pretrained SD checkpoint and applies consistency
            distillation to it. Remove the distillation and you have the
            original SD model back. The architecture is unchanged—the
            modularity from <LessonLink slug="stable-diffusion-architecture">Stable Diffusion Architecture</LessonLink>{' '}
            is preserved.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* LCM Part B: In practice — step counts */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LCM in Practice"
            subtitle="What 1, 2, 4, and 8 steps actually look like"
          />
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="1 Step" color="amber">
                <div className="space-y-2 text-sm">
                  <p>
                    Recognizable subject and composition. Soft textures, reduced
                    detail. Adequate for previews and rapid iteration.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="2 Steps" color="cyan">
                <div className="space-y-2 text-sm">
                  <p>
                    Significant quality improvement. Most textures resolved.
                    Occasional softness in fine details (hair strands, fabric
                    weave).
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="4 Steps" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    Near-baseline quality. Difficult to distinguish from the
                    50-step baseline in most cases. The practical sweet spot.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="8 Steps" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    Diminishing returns. Quality matches or exceeds baseline.
                    The consistency model was optimized for low step counts;
                    additional steps help little.
                  </p>
                </div>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Guidance Scale">
            LCM typically uses a reduced guidance scale (1.0-2.0) compared to
            standard SD (7.0-7.5). The augmented PF-ODE already incorporates
            guidance into the trajectory, so additional guidance is redundant
            and can cause oversaturation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Check #1 */}
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
                  LCM is applied to the SD U-Net. Does the VAE change?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The VAE is frozen and unchanged. LCM modifies only the
                    denoising process—the U-Net weights and the scheduler. The
                    VAE still encodes images to latents and decodes latents to
                    images. This is the modularity principle from{' '}
                    <strong>Stable Diffusion Architecture</strong> in action:
                    the three components (CLIP, U-Net, VAE) are independently
                    swappable.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  An LCM model trained on SD v1.5 produces 4-step results at
                  512&times;512. Would it work at 768&times;768?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Likely not well. The consistency model was distilled against
                    the teacher&rsquo;s behavior at 512&times;512 resolution. At
                    768&times;768, the latent dimensions change
                    (96&times;96&times;4 instead of 64&times;64&times;4), the
                    noise statistics change, and the model is out of
                    distribution. This is a resolution limitation, not a
                    fundamental limitation of the approach. SDXL-based LCMs work
                    at 1024&times;1024 because the teacher was trained at that
                    resolution.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — LCM-LoRA Part A: The key insight */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LCM-LoRA: Speed as a Skill"
            subtitle="What if you could capture 'generate fast' as a LoRA adapter?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Full LCM training modifies all the U-Net weights. The distilled
              model is a new checkpoint—2 GB or more. But what if you could
              capture the distillation as a LoRA adapter?
            </p>
            <p className="text-muted-foreground">
              The weight change from consistency distillation can be decomposed
              as a low-rank update:{' '}
              <InlineMath math="W_\text{distilled} \approx W_\text{original} + BA" />.
              This means you can:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                Start with the original SD model weights
              </li>
              <li>
                Train LoRA matrices (<InlineMath math="B" />,{' '}
                <InlineMath math="A" />) that approximate the distillation
                update
              </li>
              <li>
                Save the LoRA (~4 MB instead of a full 2 GB checkpoint)
              </li>
              <li>
                At inference: load ANY compatible SD model + the LCM-LoRA + LCM
                scheduler
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="A Different Kind of LoRA">
            You have used LoRA to teach a model{' '}
            <strong>what</strong> watercolor paintings look like (in{' '}
            <strong>LoRA Fine-Tuning</strong>). LCM-LoRA teaches a model{' '}
            <strong>how</strong> to generate in 4 steps. Same bypass mechanism,
            fundamentally different purpose.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* LCM-LoRA misconception: Style LoRA vs LCM-LoRA comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'Style LoRA',
                color: 'amber',
                items: [
                  'Learns a visual style or subject identity',
                  'Training data: 50-200 images of a style',
                  'Loss: MSE on noise prediction (same as DDPM)',
                  'Transfer: limited to similar fine-tunes',
                  'Composable with other LoRAs (sum bypasses)',
                  'File size: 2-50 MB',
                ],
              }}
              right={{
                title: 'LCM-LoRA',
                color: 'emerald',
                items: [
                  'Learns how to generate in fewer steps',
                  'Training data: noise-denoise pairs from teacher',
                  'Loss: consistency distillation loss',
                  'Transfer: designed for cross-model use',
                  'Composable (LCM-LoRA + style LoRA)',
                  'File size: ~4 MB',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Trained on Images">
            LCM-LoRA does NOT train on a dataset of images the way style LoRAs
            do. It distills from the teacher model&rsquo;s predictions. The LoRA
            learns to approximate the teacher&rsquo;s few-step behavior, not a
            specific visual style.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* LCM-LoRA Part B: Universality */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One Adapter, Many Models"
            subtitle="Why LCM-LoRA generalizes across checkpoints"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The LCM-LoRA trained on the base SD v1.5 checkpoint can be applied
              to Dreamshaper, Realistic Vision, Anything V5, or any other SD
              v1.5-compatible fine-tune. One 4 MB file turns all of them into
              4-step models.
            </p>
            <p className="text-muted-foreground">
              Why does this work? The LCM-LoRA captures a general behavior
              change: &ldquo;collapse the denoising trajectory into 4
              steps.&rdquo; This behavior change is about the{' '}
              <strong>dynamics of denoising</strong>, not about specific content
              or style. A watercolor-style fine-tune still follows the same ODE
              trajectory structure—it just arrives at a different endpoint. The
              LCM-LoRA teaches the model to teleport along this trajectory
              regardless of what the endpoint looks like.
            </p>
            <p className="text-muted-foreground">
              This universality has limits. The LCM-LoRA is{' '}
              <strong>architecture-specific</strong>. An LCM-LoRA trained on SD
              v1.5 does NOT work on SDXL because the U-Net architecture is
              different (different dimensions, different number of attention
              heads). Separate LCM-LoRA checkpoints exist for SD v1.5 and SDXL.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Adapter, Many Models">
            This is the same swappability pattern you know from{' '}
            <strong>LoRA Fine-Tuning</strong> and{' '}
            <strong>ControlNet</strong>. A small adapter file works across all
            compatible base models. LCM-LoRA extends this to speed: one speed
            adapter for an entire model family.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* LCM-LoRA Part C: Code comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="LCM-LoRA in Practice"
            subtitle="Three lines changed"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Standard SD generation:
            </p>
            <CodeBlock
              code={`pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
image = pipe(
    "a cat on a beach",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]`}
              language="python"
              filename="standard_sd.py"
            />
            <p className="text-muted-foreground">
              With LCM-LoRA:
            </p>
            <CodeBlock
              code={`pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
image = pipe(
    "a cat on a beach",
    num_inference_steps=4,
    guidance_scale=1.5,
).images[0]`}
              language="python"
              filename="lcm_lora_sd.py"
            />
            <p className="text-muted-foreground">
              Three lines changed: load the LoRA, swap the scheduler, reduce the
              step count and guidance scale. Everything else is identical. The
              pipeline is the same—the three translators (CLIP, U-Net, VAE) are
              unchanged. The denoising loop just runs fewer iterations with a
              model that knows how to handle it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Drop-In Acceleration">
            This is the practical payoff. Any existing SD generation workflow
            can be accelerated by adding two lines (load LoRA, swap scheduler)
            and changing two numbers (steps and guidance scale). No
            architectural changes, no retraining.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: LCM-LoRA"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague has a custom SD v1.5 model fine-tuned on
                  architectural photographs. They want 4-step generation. What
                  do you recommend: (a) train a full LCM from their fine-tune,
                  or (b) load the base LCM-LoRA?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Start with (b)—the base LCM-LoRA. It will likely work
                    because their fine-tune is SD v1.5-compatible. The LoRA
                    captures &ldquo;how to generate fast&rdquo; which
                    generalizes across fine-tunes. Only if the quality is
                    noticeably degraded should they consider (a). This is the
                    universality insight in action.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Can you load an LCM-LoRA AND a style LoRA at the same time?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes. LoRA weights are additive (from{' '}
                    <strong>LoRA Fine-Tuning</strong>). The style LoRA captures
                    &ldquo;generate in this style&rdquo; and the LCM-LoRA
                    captures &ldquo;generate in 4 steps.&rdquo; Both bypasses
                    are summed. In practice, you may need to adjust the LoRA
                    scale weights to balance the two influences. This is the
                    same composition pattern applied to a new type of LoRA.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Explain — ADD Part A: The problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Sharpness Problem"
            subtitle="Why consistency distillation is not enough at 1 step"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You saw in <LessonLink slug="consistency-models">Consistency Models</LessonLink> that 1-step
              consistency model output is soft. The images are recognizable but
              lack the crisp detail of multi-step diffusion. Why?
            </p>
            <p className="text-muted-foreground">
              Consistency distillation&rsquo;s teacher signal is &ldquo;be
              consistent with the ODE trajectory.&rdquo; This is a mathematical
              constraint: the predictions at adjacent timesteps should agree.
              But mathematical consistency does not guarantee perceptual
              sharpness. A blurry image can be perfectly consistent with the ODE
              trajectory—it is just a less accurate prediction of the endpoint.
            </p>
            <p className="text-muted-foreground">
              What if the training signal included &ldquo;look
              realistic&rdquo; in addition to &ldquo;be consistent&rdquo;?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Consistency vs Realism">
            Consistency distillation asks: &ldquo;Do these trajectory points
            agree?&rdquo; It does not ask: &ldquo;Does this output look like a
            real photograph?&rdquo; These are different questions with different
            answers.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ADD Part B: The discriminator concept (gap fill) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Discriminator"
            subtitle="A concept from a different branch of generative modeling"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              To understand adversarial diffusion distillation, you need one
              concept from a different branch of generative modeling: the
              discriminator.
            </p>

            <GradientCard title="What Is a Discriminator?" color="orange">
              <div className="space-y-2 text-sm">
                <p>
                  A discriminator is a classifier that distinguishes real images
                  from generated images. It receives an image and outputs a
                  probability: &ldquo;How likely is this to be a real photograph
                  vs. a generated sample?&rdquo;
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Think of it as a <strong>critic and artist</strong>. The
              generator is an artist producing images. The discriminator is an
              art critic who examines each piece and says &ldquo;this does not
              look like a real photograph—the textures are too smooth, the
              lighting is flat.&rdquo; The artist adjusts. Over many rounds,
              the artist learns to produce work that satisfies the critic.
            </p>
            <p className="text-muted-foreground">
              In adversarial training, these two networks compete:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                The <strong>artist</strong> (generator) tries to produce images
                that fool the critic
              </li>
              <li>
                The <strong>critic</strong> (discriminator) tries to correctly
                classify images as real or generated
              </li>
            </ul>
            <p className="text-muted-foreground">
              Each provides a gradient signal to the other. The artist receives
              gradients from the critic saying &ldquo;this output does not look
              real enough—here is what to fix.&rdquo; The critic receives real
              images and generated images, learning to tell them apart. In ADD,
              the artist must satisfy <strong>two</strong> judges: the diffusion
              teacher (&ldquo;be consistent with the trajectory&rdquo;) and the
              critic (&ldquo;look realistic&rdquo;).
            </p>
            <p className="text-muted-foreground">
              You have seen this pattern before, briefly: the SD VAE uses
              adversarial training for sharper reconstructions (from{' '}
              <strong>From Pixels to Latents</strong>). A discriminator
              penalizes blurry VAE outputs, pushing the decoder toward sharper
              images.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Critical Limitation">
            Pure adversarial training (GANs) suffers from{' '}
            <strong>mode collapse</strong>: the generator learns to produce a
            narrow set of outputs that reliably fool the discriminator,
            sacrificing diversity for safety. This is why ADD{' '}
            <strong>combines</strong> adversarial training with diffusion, not
            replaces diffusion with it.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ADD Part C: The hybrid loss */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ADD's Hybrid Loss"
            subtitle="Two teachers, two lessons"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Adversarial diffusion distillation combines two losses:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{ADD} = \mathcal{L}_\text{diffusion} + \lambda \cdot \mathcal{L}_\text{adversarial}" />
            </div>
            <p className="text-muted-foreground">
              The <strong>diffusion loss</strong> is a distillation objective
              similar to consistency distillation—the student learns from the
              teacher&rsquo;s predictions. This provides diversity and training
              stability.
            </p>
            <p className="text-muted-foreground">
              The <strong>adversarial loss</strong> comes from a discriminator
              that judges whether the few-step output looks realistic. This
              provides sharpness and perceptual quality.
            </p>
            <p className="text-muted-foreground">
              The <InlineMath math="\lambda" /> parameter controls the balance.
              Higher <InlineMath math="\lambda" /> means more discriminator
              influence (sharper but riskier). Lower{' '}
              <InlineMath math="\lambda" /> means more diffusion influence
              (softer but more stable). This is the same &ldquo;volume
              knob&rdquo; pattern you know from ControlNet&rsquo;s conditioning
              scale and IP-Adapter&rsquo;s scale parameter.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Two Teachers, Two Lessons">
            The diffusion teacher says: &ldquo;Your output should be consistent
            with what I would produce.&rdquo; The discriminator teacher says:
            &ldquo;Your output should look realistic.&rdquo; Neither alone is
            sufficient. Consistency without realism produces blurry outputs.
            Realism without consistency produces sharp but mode-collapsed
            outputs.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The discriminator is training-time only */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A crucial distinction: the discriminator is only used{' '}
              <strong>during training</strong> to provide gradient signal. At
              inference, only the generator (the distilled diffusion model)
              runs. The discriminator is not needed—it has already served its
              purpose by shaping the generator&rsquo;s weights during training.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Training-Time Teacher Only">
            The discriminator is a training-time teacher. At inference, only
            the generator runs. No extra model, no extra compute. The
            discriminator&rsquo;s influence is baked into the weights.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* ADD Part D: ADD vs Consistency Distillation comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ADD vs Consistency Distillation"
            subtitle="Different teacher signals, different outputs"
          />
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'Consistency Distillation (LCM)',
                color: 'emerald',
                items: [
                  'Teacher signal: ODE trajectory consistency',
                  '1-step quality: soft, reduced detail',
                  'Diversity: high (no mode collapse risk)',
                  'Training: stable (no adversarial dynamics)',
                  '4-step quality: near-baseline',
                  'Practical form: LCM-LoRA (drop-in adapter)',
                  'Plug-and-play: yes (one adapter, many models)',
                ],
              }}
              right={{
                title: 'Adversarial Distillation (ADD)',
                color: 'violet',
                items: [
                  'Teacher signal: ODE consistency + discriminator realism',
                  '1-step quality: sharp, detailed',
                  'Diversity: slightly lower (discriminator bias)',
                  'Training: more complex (balance two losses)',
                  '4-step quality: excellent',
                  'Practical form: SDXL Turbo (dedicated model)',
                  'Plug-and-play: no (full model retraining)',
                ],
              }}
            />

            {/* Contrastive quality comparison: what the outputs actually look like */}
            <p className="text-muted-foreground">
              The table says &ldquo;soft&rdquo; vs &ldquo;sharp.&rdquo; What
              does that actually mean? Here is what to expect when you run each
              approach at 1 step:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard
                title="1-Step Consistency Distillation (LCM)"
                color="emerald"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Overall impression:</strong> The subject, composition,
                    and colors are correct. The image looks like a watercolor
                    interpretation of the prompt rather than a photograph.
                  </p>
                  <p>
                    <strong>Textures:</strong> Hair renders as a smooth mass
                    rather than individual strands. Fabric appears flat—you lose
                    the weave and fold detail. Foliage becomes a soft green wash
                    without distinct leaves.
                  </p>
                  <p>
                    <strong>Edges:</strong> Boundaries between objects are
                    slightly blurred. Fine structures (eyelashes, fence wires,
                    tree branches against sky) tend to merge into their
                    background.
                  </p>
                  <p>
                    <strong>Why:</strong> The training signal is trajectory
                    consistency—&ldquo;do these ODE points agree?&rdquo; A
                    blurry-but-consistent prediction satisfies this constraint.
                    High-frequency detail is the first casualty.
                  </p>
                </div>
              </GradientCard>
              <GradientCard
                title="1-Step ADD (SDXL Turbo)"
                color="violet"
              >
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Overall impression:</strong> Immediately sharper.
                    Looks closer to a multi-step result. Details pop in a way
                    the consistency distillation output does not.
                  </p>
                  <p>
                    <strong>Textures:</strong> Hair shows individual strands.
                    Fabric has visible weave. Foliage has distinct leaf shapes.
                    The discriminator specifically penalizes the smoothed-out
                    look.
                  </p>
                  <p>
                    <strong>Edges:</strong> Crisp boundaries between objects.
                    Fine structures are preserved. However, you may notice
                    occasional texture repetition—small patches that look
                    unnaturally regular, as if stamped.
                  </p>
                  <p>
                    <strong>Why:</strong> The discriminator asks &ldquo;does
                    this look like a real photograph?&rdquo; Real photographs
                    have sharp detail. But the discriminator can be fooled by
                    plausible-looking patterns that repeat unnaturally.
                  </p>
                </div>
              </GradientCard>
            </div>

            {/* Address misconception: ADD is not just a GAN — three-way comparison */}
            <p className="text-muted-foreground">
              ADD is <strong>not</strong> just a GAN. To see why, compare what
              happens when you remove each component:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Pure GAN" color="rose">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>What it is:</strong> A feedforward generator maps
                    noise to an image in one pass. A discriminator judges
                    realism. No diffusion, no trajectory.
                  </p>
                  <p>
                    <strong>Sharpness:</strong> High—the discriminator demands
                    crisp detail.
                  </p>
                  <p>
                    <strong>Diversity:</strong> Low—mode collapse. The generator
                    learns a narrow set of &ldquo;safe&rdquo; outputs.
                  </p>
                  <p>
                    <strong>Stability:</strong> Fragile—adversarial training is
                    notoriously hard to balance.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Pure Consistency Distillation" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>What it is:</strong> A student learns the
                    teacher&rsquo;s ODE trajectory. No discriminator, no
                    adversarial loss. This is what LCM does.
                  </p>
                  <p>
                    <strong>Sharpness:</strong> Moderate—soft at 1 step because
                    trajectory consistency does not guarantee crisp detail.
                  </p>
                  <p>
                    <strong>Diversity:</strong> High—no mode collapse risk. The
                    diffusion trajectory preserves the full distribution.
                  </p>
                  <p>
                    <strong>Stability:</strong> Stable—no adversarial dynamics
                    to balance.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="ADD Hybrid" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>What it is:</strong> Diffusion distillation +
                    discriminator. Both teacher signals combined. This is what
                    SDXL Turbo does.
                  </p>
                  <p>
                    <strong>Sharpness:</strong> High—the discriminator pushes
                    for crisp detail.
                  </p>
                  <p>
                    <strong>Diversity:</strong> Good—the diffusion loss prevents
                    mode collapse.
                  </p>
                  <p>
                    <strong>Stability:</strong> More complex—must balance two
                    losses, but the diffusion component stabilizes training.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Remove the discriminator from ADD and you get consistency
              distillation (diverse but soft). Remove the diffusion loss and
              you get a GAN (sharp but mode-collapsed). ADD is the hybrid that
              gets both sharpness and diversity.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Problem, Different Solutions">
            Both LCM and ADD solve the same problem: fast generation from a
            pretrained diffusion model. LCM prioritizes universality and
            simplicity. ADD prioritizes output quality at very low step counts.
            The best choice depends on your use case.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 10: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: ADD and Tradeoffs"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  SDXL Turbo produces sharp 1-step images. Why not just always
                  use it instead of LCM?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2 text-muted-foreground space-y-2">
                    <p>Several reasons:</p>
                    <ul className="list-disc list-inside space-y-1 ml-2">
                      <li>
                        SDXL Turbo is a specific model, not an adapter—you
                        cannot apply it to your favorite fine-tuned model
                      </li>
                      <li>
                        LCM-LoRA works with any compatible base model, so all
                        community fine-tunes get 4-step generation for free
                      </li>
                      <li>
                        SDXL Turbo has slightly lower diversity due to the
                        adversarial component
                      </li>
                      <li>
                        For many applications, 4-step LCM-LoRA quality is
                        sufficient, and the flexibility is worth more than the
                        marginal quality gain
                      </li>
                    </ul>
                  </div>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Would it be possible to combine both approaches: consistency
                  distillation + adversarial loss?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes, and this is essentially what ADD does—it uses a
                    diffusion-based distillation objective alongside the
                    adversarial loss. The diffusion component in ADD provides
                    the consistency and diversity that pure adversarial training
                    lacks. The research direction is toward finding the right
                    balance between the two signals.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Elaborate — Placing LCM and ADD in the Speed Landscape */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Level 3 Expanded"
            subtitle="The trajectory bypass now has branches"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="consistency-models">Consistency Models</LessonLink>, Level 3 of the
              &ldquo;three levels of speed&rdquo; was a single entry:
              consistency models. Now Level 3 has two sub-approaches:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard
                title="Level 3a: Consistency-Based"
                color="violet"
              >
                <div className="space-y-2 text-sm">
                  <p>LCM, LCM-LoRA</p>
                  <p className="text-muted-foreground">
                    Teacher signal: ODE trajectory consistency
                  </p>
                  <p className="text-muted-foreground">
                    Strength: universality (one adapter, many models)
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    Best at: 4-step generation across many models.
                  </p>
                </div>
              </GradientCard>
              <GradientCard
                title="Level 3b: Adversarial"
                color="orange"
              >
                <div className="space-y-2 text-sm">
                  <p>SDXL Turbo (ADD)</p>
                  <p className="text-muted-foreground">
                    Teacher signal: ODE consistency + discriminator realism
                  </p>
                  <p className="text-muted-foreground">
                    Strength: sharp 1-step quality
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    Best at: single-step generation where maximum quality
                    matters.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Both are Level 3: both bypass the trajectory. The difference is
              what teacher signal guides the bypass.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Full Speed Picture">
            Level 1: smarter walking (DPM-Solver++). Level 2: straighten the
            road (flow matching). Level 3a: teleport via consistency (LCM).
            Level 3b: teleport via adversarial training (ADD). The next lesson
            maps the complete landscape.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* The practical decision */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Practical Decision"
            subtitle="When to use which approach"
          />
          <div className="space-y-4">
            <ul className="list-disc list-inside text-muted-foreground space-y-3 ml-4">
              <li>
                <strong>Need 4-step generation from a community model?</strong>{' '}
                LCM-LoRA. Drop-in, universal, composable with style LoRAs.
              </li>
              <li>
                <strong>
                  Need the sharpest possible 1-step result from SDXL?
                </strong>{' '}
                SDXL Turbo. Dedicated model, not an adapter.
              </li>
              <li>
                <strong>
                  Need flexibility and do not mind 20 steps?
                </strong>{' '}
                Standard SD with DPM-Solver++. No distillation needed.
              </li>
              <li>
                <strong>Building a real-time application?</strong> LCM-LoRA at
                4 steps or SDXL Turbo at 1 step, depending on quality
                requirements.
              </li>
            </ul>
          </div>
        </Row.Content>
      </Row>

      {/* Section 12: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: LCM-LoRA Hands-On"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes LCM-LoRA concrete—use it as a drop-in
                accelerator, test its universality across models, explore the
                step count and guidance scale space, and compose it with style
                LoRAs.
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
                    Exercise 1 (Guided): LCM-LoRA — 4-Step Generation.
                  </strong>{' '}
                  Load SD v1.5, load LCM-LoRA, swap scheduler to LCMScheduler.
                  Generate at 50 steps (standard), 4 steps (standard—bad
                  quality), and 4 steps with LCM-LoRA (good quality). Time each
                  generation. Predict before running: what will the standard
                  model at 4 steps look like?
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): LCM-LoRA Universality.
                  </strong>{' '}
                  Load a community fine-tune compatible with the base model.
                  Apply the SAME LCM-LoRA from Exercise 1. Generate at 4 steps
                  and compare quality. Predict: will the same LCM-LoRA work on
                  a different fine-tune?
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): Step Count and Guidance Scale.
                  </strong>{' '}
                  Using LCM-LoRA, generate the same prompt at 1, 2, 4, 8 steps.
                  At each step count, try guidance scales of 1.0, 1.5, 2.0,
                  4.0. Observe: quality plateau at 4 steps, oversaturation at
                  high guidance.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): LCM-LoRA + Style LoRA Composition.
                  </strong>{' '}
                  Load an LCM-LoRA + a style LoRA simultaneously. Generate at 4
                  steps. Compare: style LoRA alone at 50 steps vs LCM-LoRA +
                  style LoRA at 4 steps. Experiment with LoRA weight scales.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: see LCM-LoRA work</li>
              <li>Guided: test universality</li>
              <li>Supported: explore parameters</li>
              <li>Independent: compose LoRAs</li>
            </ol>
            <p className="text-sm mt-2">
              SDXL Turbo is not in the exercises because it requires a dedicated
              model. The notebook focuses on the more practically useful
              LCM-LoRA workflow.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 13: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Same recipe, bigger kitchen.',
                description:
                  'LCM applies consistency distillation to SD/SDXL latent diffusion. The procedure is structurally identical to what you trained in 2D—the teacher takes one ODE step, the student matches predictions. The only change is scale.',
              },
              {
                headline: 'Speed adapter.',
                description:
                  'LCM-LoRA captures "how to generate in 4 steps" as a LoRA bypass. One 4 MB file turns any compatible SD model into a 4-step model. Same LoRA mechanism, different skill: speed instead of style.',
              },
              {
                headline: 'Different teacher, different output.',
                description:
                  'Consistency distillation\'s teacher signal is ODE trajectory consistency (soft at 1 step). ADD\'s teacher signal adds a discriminator for realism (sharp at 1 step). Neither alone is ideal; each has tradeoffs.',
              },
              {
                headline: 'Universality vs quality.',
                description:
                  'LCM-LoRA wins on universality (one adapter, many models, composable). SDXL Turbo wins on 1-step quality (sharp, detailed). For most workflows, LCM-LoRA at 4 steps is the practical choice.',
              },
              {
                headline: 'Level 3 has branches.',
                description:
                  'The "bypass the trajectory" level splits into consistency-based (LCM) and adversarial (ADD). Both bypass the trajectory; they differ in what guides the bypass.',
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
                title: 'Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference',
                authors: 'Luo, Tan, Huang, Li & Zhao, 2023',
                url: 'https://arxiv.org/abs/2310.04378',
                note: 'The LCM paper. Sections 3-4 cover the augmented PF-ODE and latent-space consistency distillation. Section 5 introduces LCM-LoRA.',
              },
              {
                title: 'LCM-LoRA: A Universal Stable-Diffusion Acceleration Module',
                authors: 'Luo, Tan, Huang, Li & Zhao, 2023',
                url: 'https://arxiv.org/abs/2311.05556',
                note: 'The dedicated LCM-LoRA paper. Demonstrates universality across SD v1.5, SSD-1B, and SDXL checkpoints and fine-tunes.',
              },
              {
                title: 'Adversarial Diffusion Distillation',
                authors: 'Sauer, Lorenz, Blattmann & Rombach, 2023',
                url: 'https://arxiv.org/abs/2311.17042',
                note: 'The SDXL Turbo paper. Section 3 describes the hybrid adversarial + diffusion distillation loss. Section 4.2 has the ablation showing the contribution of each loss component.',
              },
              {
                title: 'Consistency Models',
                authors: 'Song, Dhariwal, Chen & Sutskever, 2023',
                url: 'https://arxiv.org/abs/2303.01469',
                note: 'The foundation paper from the previous lesson. LCM builds directly on this.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 14: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: The Speed Landscape"
            description="You now know two production-ready approaches to 1-4 step generation: LCM-LoRA for universal plug-and-play acceleration, and SDXL Turbo for maximum 1-step quality. The next lesson steps back and maps the complete picture. You have seen four acceleration strategies across Series 6 and 7: better ODE solvers, flow matching, consistency distillation, and adversarial distillation. The Speed Landscape organizes them into a taxonomy with clear quality-speed-flexibility tradeoffs—a map for choosing the right approach for any use case."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
