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
  NextStepBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'
import { ExternalLink } from 'lucide-react'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-4-3-generate-with-stable-diffusion.ipynb'

/**
 * Generate with Stable Diffusion
 *
 * Lesson 3 in Module 6.4 (Stable Diffusion). Lesson 17 overall in Series 6.
 * Cognitive load: CONSOLIDATE (zero new concepts).
 *
 * The payoff lesson: the student uses the real diffusers library and
 * understands every parameter because each one maps to a concept they
 * built from scratch across Modules 6.1-6.4.
 *
 * Previous: Samplers and Efficiency (Module 6.4, Lesson 2 / STRETCH)
 * Next: LoRA Fine-Tuning (Module 6.5, Lesson 1)
 */

export function GenerateWithStableDiffusionLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Generate with Stable Diffusion"
            description="Every parameter in the diffusers API maps to a concept you built from scratch. This is not a tutorial. This is you driving a machine you built."
            category="Stable Diffusion"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Use the diffusers library to generate images with Stable Diffusion,
            understanding what every parameter controls because each one maps to
            a concept you built across 16 lessons.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You traced the complete pipeline in{' '}
            <strong>The Stable Diffusion Pipeline</strong>. You implemented CFG
            in <strong>Text Conditioning &amp; Guidance</strong>. You chose
            samplers with understanding in{' '}
            <strong>Samplers and Efficiency</strong>. This lesson connects all
            of that to the real API.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Using the diffusers StableDiffusionPipeline to generate images',
              'Mapping every API parameter to its underlying concept and source lesson',
              'Exploring parameter effects through predict-then-verify experiments',
              'Building practical intuition for parameter selection',
              'NOT: new mathematical formulas or derivations (this is CONSOLIDATE)',
              'NOT: fine-tuning, LoRA, or customization (Module 6.5)',
              'NOT: img2img, inpainting, or ControlNet',
              'NOT: prompt engineering as a deep topic',
              'NOT: SD v1 vs v2 vs XL differences',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Hook — "You built this." */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You Built This"
            subtitle="Every parameter maps to a concept you learned"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For 16 lessons, you have been building. You learned to compress
              images (VAE), to destroy and reconstruct (diffusion), to condition
              on time and text (U-Net, CLIP, cross-attention, CFG), to work in
              latent space, to assemble the full pipeline, and to choose a
              sampler. Here is the payoff:
            </p>
          </div>
          <div className="mt-4">
            <CodeBlock
              code={`pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
image = pipe(
    prompt="a cat sitting on a beach at sunset",
    negative_prompt="blurry, low quality",
    guidance_scale=7.5,
    num_inference_steps=20,
    generator=torch.Generator().manual_seed(42),
    height=512,
    width=512,
).images[0]`}
              language="python"
              filename="generate.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Every parameter in this call maps to a concept you built from
              scratch. <code>guidance_scale</code> is the{' '}
              <InlineMath math="w" /> in a formula you implemented.{' '}
              <code>num_inference_steps</code> is the step count along a
              trajectory you learned to traverse.{' '}
              <code>negative_prompt</code> is a substitution in the CFG formula
              you already know.
            </p>
            <p className="text-muted-foreground">
              Most people use this API as a black box. You know what the machine
              does inside. The challenge: for each parameter, predict what it
              controls and what happens when you change it. Then run the code
              and verify.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a Tutorial">
            This is not &ldquo;copy these lines and get an image.&rdquo; This
            is you driving a car you built&mdash;the engine (diffusion), the
            transmission (sampler), the steering (CFG), the GPS (CLIP), and
            the dashboard (API). Every gauge on the dashboard corresponds to a
            part you assembled.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: The Parameter Map */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Parameter Map"
            subtitle="Every parameter, every concept, every source"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This table is the centerpiece of the lesson. Each row connects an
              API parameter to the internal behavior it controls, the concept
              behind it, and the lesson where you learned it:
            </p>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-muted">
                  <th className="text-left py-2 pr-3 text-foreground font-semibold">Parameter</th>
                  <th className="text-left py-2 pr-3 text-foreground font-semibold">What It Controls</th>
                  <th className="text-left py-2 pr-3 text-foreground font-semibold">Internal Behavior</th>
                  <th className="text-left py-2 text-foreground font-semibold">Source Lesson</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">prompt</td>
                  <td className="py-2 pr-3">What the image depicts</td>
                  <td className="py-2 pr-3">Tokenized (77 max), CLIP-encoded to [1, 77, 768], injected via cross-attention K/V</td>
                  <td className="py-2">CLIP, Text Conditioning &amp; Guidance</td>
                </tr>
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">negative_prompt</td>
                  <td className="py-2 pr-3">What to steer away from</td>
                  <td className="py-2 pr-3">Replaces empty-string unconditional embedding in CFG</td>
                  <td className="py-2">Text Conditioning &amp; Guidance, The SD Pipeline</td>
                </tr>
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">guidance_scale</td>
                  <td className="py-2 pr-3">How strongly to follow the prompt</td>
                  <td className="py-2 pr-3">The <InlineMath math="w" /> in CFG: amplifies difference between conditional and unconditional predictions</td>
                  <td className="py-2">Text Conditioning &amp; Guidance</td>
                </tr>
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">num_inference_steps</td>
                  <td className="py-2 pr-3">How many denoising steps</td>
                  <td className="py-2 pr-3">Steps from <InlineMath math="z_T" /> to <InlineMath math="z_0" />. Each step is 1&ndash;3 U-Net passes (doubled with CFG)</td>
                  <td className="py-2">Samplers and Efficiency</td>
                </tr>
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">scheduler</td>
                  <td className="py-2 pr-3">Which sampler algorithm</td>
                  <td className="py-2 pr-3">The ODE solver strategy: DPM-Solver++ (20&ndash;30), DDIM (50), Euler (30&ndash;50), DDPM (1000)</td>
                  <td className="py-2">Samplers and Efficiency</td>
                </tr>
                <tr className="border-b border-muted/50">
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">generator</td>
                  <td className="py-2 pr-3">Random seed for <InlineMath math="z_T" /></td>
                  <td className="py-2 pr-3">torch.Generator().manual_seed(N) determines the initial random latent [4, 64, 64]</td>
                  <td className="py-2">The SD Pipeline</td>
                </tr>
                <tr>
                  <td className="py-2 pr-3 font-mono text-xs text-foreground">height / width</td>
                  <td className="py-2 pr-3">Image dimensions in pixels</td>
                  <td className="py-2 pr-3">Must be multiples of 8 (VAE 8&times; downsampling). 512&times;512 &rarr; 64&times;64&times;4 latent</td>
                  <td className="py-2">From Pixels to Latents</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a Black Box">
            Every row in this table connects to a specific lesson. You built
            each of these concepts. The API is not a black box&mdash;it is a
            dashboard for a machine you understand.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Parameter-by-Parameter Exploration */}
      {/* 5a: guidance_scale */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="guidance_scale"
            subtitle="The contrast slider you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know this parameter as the <InlineMath math="w" /> in the CFG
              formula from <strong>Text Conditioning &amp; Guidance</strong>:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <InlineMath math="\varepsilon_{\text{cfg}} = \varepsilon_{\text{uncond}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})" />
            </div>
            <p className="text-muted-foreground">
              <strong>Predict before you run:</strong> At{' '}
              <InlineMath math="w = 3" />, the unconditional prediction has more
              weight. What does this mean for the image? At{' '}
              <InlineMath math="w = 25" />, the conditional-minus-unconditional
              direction is amplified 25&times;. What happens when you
              extrapolate that far?
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-5 mt-4">
            <GradientCard title="w = 1" color="amber">
              <p className="text-sm">No amplification. Conditional prediction used directly. Image may not match the prompt well.</p>
            </GradientCard>
            <GradientCard title="w = 3" color="cyan">
              <p className="text-sm">Mild amplification. Prompt partially followed. Soft, dreamlike quality.</p>
            </GradientCard>
            <GradientCard title="w = 7.5" color="emerald">
              <p className="text-sm">Balanced. Prompt-faithful with natural image quality. The typical default.</p>
            </GradientCard>
            <GradientCard title="w = 15" color="orange">
              <p className="text-sm">Strong amplification. Colors oversaturated. Contrast pushed hard.</p>
            </GradientCard>
            <GradientCard title="w = 25" color="rose">
              <p className="text-sm">Extreme. Distorted, unnatural features. The extrapolation overshoots.</p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Not a Quality Dial">
            <code>guidance_scale</code> is not a quality dial. It is a tradeoff
            between prompt fidelity and image naturalness. Higher values mean
            &ldquo;follow the text harder&rdquo;&mdash;not &ldquo;make the
            image better.&rdquo; Remember the &ldquo;contrast slider&rdquo;
            analogy: turning it too far in either direction hurts.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 5b: num_inference_steps */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="num_inference_steps"
            subtitle="The sweet spot you predicted in the last lesson"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You learned in <strong>Samplers and Efficiency</strong> that
              DPM-Solver++ plateaus around 20&ndash;30 steps. The model defines
              the trajectory; the sampler decides how to traverse it. More steps
              beyond the sweet spot means more compute for diminishing returns.
            </p>
            <p className="text-muted-foreground">
              <strong>Predict before you run:</strong> With DPM-Solver++, what
              quality difference do you expect between 20 and 100 steps?
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-5 mt-4">
            <GradientCard title="5 steps" color="rose">
              <p className="text-sm">Barely recognizable. Trajectory traversed too coarsely. Coarse structure only.</p>
            </GradientCard>
            <GradientCard title="10 steps" color="amber">
              <p className="text-sm">Recognizable but rough. Missing fine detail. Edges and textures incomplete.</p>
            </GradientCard>
            <GradientCard title="20 steps" color="emerald">
              <p className="text-sm">Good quality. The DPM-Solver++ sweet spot. Full detail, clean edges.</p>
            </GradientCard>
            <GradientCard title="50 steps" color="blue">
              <p className="text-sm">Indistinguishable from 20. More compute, no visible improvement.</p>
            </GradientCard>
            <GradientCard title="100 steps" color="violet">
              <p className="text-sm">Still indistinguishable. 5&times; the compute of 20 steps for zero visible gain.</p>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="More Steps Is Not More Quality">
            The default of 50 in many tutorials is a conservative holdover from
            DDIM. With DPM-Solver++, 20&ndash;25 steps is the sweet spot. Going
            to 100 or 200 wastes compute with negligible improvement&mdash;the
            quality curve plateaus.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 5c: scheduler */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="scheduler"
            subtitle="Same vehicle, different route"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Swapping schedulers is an inference-time decision that requires
              zero retraining. The model&rsquo;s job never changes: predict
              noise. The sampler&rsquo;s job is to decide what to do with that
              prediction. In diffusers, it is one line:
            </p>
          </div>
          <div className="mt-4">
            <CodeBlock
              code={`from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)`}
              language="python"
              filename="swap_scheduler.py"
            />
          </div>
          <div className="mt-4">
            <ComparisonRow
              left={{
                title: 'DPM-Solver++ at 20 steps',
                color: 'emerald',
                items: [
                  'Current standard',
                  'Best speed/quality balance',
                  'Higher-order solver (reads the road ahead)',
                  'Use this unless you have a reason not to',
                ],
              }}
              right={{
                title: 'DDIM at 50 steps',
                color: 'blue',
                items: [
                  'Deterministic (same seed = same image)',
                  'Best for reproducibility and A/B testing',
                  'First-order (predict-and-leap)',
                  'Slightly more steps for same quality',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Practical Default">
            DPM-Solver++ at 20&ndash;30 steps for speed. DDIM at 50 for
            reproducibility and interpolation experiments. Euler at 30&ndash;50
            for debugging. DDPM at 1000 only when speed does not matter. You
            learned these recommendations in{' '}
            <strong>Samplers and Efficiency</strong>&mdash;now you apply them.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 5d: negative_prompt */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="negative_prompt"
            subtitle="CFG steering, not image erasure"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know from{' '}
              <strong>The Stable Diffusion Pipeline</strong> that negative
              prompts replace the empty-string unconditional embedding in the
              CFG formula:
            </p>
            <div className="py-3 px-5 bg-muted/50 rounded-lg">
              <InlineMath math="\varepsilon_{\text{cfg}} = \varepsilon_{\text{neg}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{neg}})" />
            </div>
            <p className="text-muted-foreground">
              <strong>Predict before you run:</strong>{' '}
              <code>negative_prompt</code> replaces the empty string in CFG.
              What direction does this steer the generation? Think about the
              formula above&mdash;what role does{' '}
              <InlineMath math="\varepsilon_{\text{neg}}" /> play?
            </p>
            <details className="mt-2 mb-2">
              <summary className="font-medium cursor-pointer text-primary text-sm">
                Reveal
              </summary>
              <p className="mt-2 text-sm text-muted-foreground">
                It steers <strong>away</strong> from the negative prompt&rsquo;s
                semantic meaning, at every denoising step. Not just the final
                image&mdash;every step. The{' '}
                <InlineMath math="\varepsilon_{\text{neg}}" /> term replaces the
                unconditional baseline, so the CFG direction now points away
                from the negative prompt&rsquo;s meaning instead of away from
                &ldquo;nothing.&rdquo;
              </p>
            </details>
            <p className="text-muted-foreground">
              Generate with{' '}
              <code>negative_prompt=&quot;blurry, low quality, deformed,
              watermark&quot;</code> and observe: the image is not &ldquo;the
              same image minus blurriness.&rdquo; It is a fundamentally
              different generation steered away from blurry outputs from the
              first step onward.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Compass, Not Eraser">
            Negative prompts are not an eraser. They are a compass pointing
            away from undesirable directions in the generation trajectory. They
            are most effective for steering away from common failure modes
            (blurriness, deformation) rather than specific content. The
            steering is probabilistic, not absolute.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 5e: generator (seed) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="generator (seed)"
            subtitle="The starting point determines the journey"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You learned in{' '}
              <strong>The Stable Diffusion Pipeline</strong> that the seed
              determines <InlineMath math="z_T" />&mdash;the initial random
              latent tensor [4, 64, 64]. Same prompt + same seed + same
              everything else = same image.
            </p>
          </div>
          <div className="mt-4">
            <CodeBlock
              code={`# Fix the seed for reproducibility
generator = torch.Generator(device="cuda").manual_seed(42)

# Different seeds = different images of the same concept
for seed in [42, 123, 999, 7777]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt="a cat on a beach", generator=gen).images[0]`}
              language="python"
              filename="seed_exploration.py"
            />
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              Multiple seeds with the same prompt produce structurally different
              images of the same concept. Seeds control the &ldquo;identity&rdquo;
              of the image; prompts control the &ldquo;concept.&rdquo; Finding a
              good seed for a particular composition, then refining the prompt,
              is a practical workflow.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Seed and Structure">
            Remember from <strong>Sampling and Generation</strong>: early
            denoising steps create structure, late steps refine details. The
            seed determines the structural decisions&mdash;the overall
            composition, pose, and layout. Changing the seed changes the
            &ldquo;skeleton&rdquo; of the image.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 5f: height / width */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="height / width"
            subtitle="Latent dimensions via 8× downsampling"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know from <strong>From Pixels to Latents</strong> that the VAE
              downsamples 8&times; in each spatial dimension. Every pixel
              dimension must be a multiple of 8.
            </p>
            <p className="text-muted-foreground">
              <strong>Predict before you run:</strong> What latent tensor shape
              corresponds to a 768&times;512 image? Use the 8&times;
              downsampling factor and work it out before revealing the answer.
            </p>
            <details className="mt-2 mb-2">
              <summary className="font-medium cursor-pointer text-primary text-sm">
                Reveal
              </summary>
              <p className="mt-2 text-sm text-muted-foreground">
                <InlineMath math="96 \times 64 \times 4" /> because{' '}
                <InlineMath math="768 / 8 = 96" /> and{' '}
                <InlineMath math="512 / 8 = 64" />. The 4 channels come from
                the VAE&rsquo;s latent space dimensionality.
              </p>
            </details>
            <p className="text-muted-foreground">
              Larger images cost quadratically more. The latent spatial area
              scales with pixel area. A 1024&times;1024 image has 4&times; the
              latent area of 512&times;512, so each U-Net pass processes
              4&times; the data.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Non-Square Caution">
            SD v1.5 was trained on 512&times;512. Non-square or larger
            resolutions may produce artifacts (repeated subjects, poor
            composition) because the model was not trained on those aspect
            ratios. SD XL addresses this with multi-resolution
            training&mdash;a topic for a future module.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 5g: prompt */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="prompt"
            subtitle="Sentences, not keywords"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know from <strong>CLIP</strong> and{' '}
              <strong>Text Conditioning &amp; Guidance</strong> that the prompt
              is tokenized (77 tokens max, padded with SOT/EOT), encoded by
              CLIP&rsquo;s text transformer to [1, 77, 768] contextual
              embeddings, and injected via cross-attention at every denoising
              step.
            </p>
            <p className="text-muted-foreground">
              The key word is <strong>contextual</strong>. CLIP&rsquo;s
              transformer uses self-attention within the text encoder, making
              each token&rsquo;s representation depend on the surrounding
              tokens. Word order, articles, and phrasing all matter because the
              embeddings change with context.
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-3 mt-4">
            <GradientCard title="Structured Prompt" color="emerald">
              <p className="text-sm font-mono mb-2">&ldquo;a cat sitting on a beach at sunset&rdquo;</p>
              <p className="text-sm">
                CLIP&rsquo;s self-attention produces contextual embeddings where
                &ldquo;cat&rdquo; is understood as the subject, &ldquo;sitting&rdquo;
                as its action, &ldquo;beach&rdquo; as the location. Cross-attention
                maps each spatial position to the relevant tokens. Result: a
                coherent scene matching the description.
              </p>
            </GradientCard>
            <GradientCard title="Scrambled Prompt" color="amber">
              <p className="text-sm font-mono mb-2">&ldquo;cat beach sunset sitting a on&rdquo;</p>
              <p className="text-sm">
                CLIP&rsquo;s self-attention sees no grammatical structure, so
                each token&rsquo;s embedding lacks syntactic context. The result
                is often a beach scene without a clear cat, or a cat that is
                not sitting&mdash;because cross-attention patterns are disrupted
                by the different embeddings.
              </p>
            </GradientCard>
            <GradientCard title="Reversed Meaning" color="rose">
              <p className="text-sm font-mono mb-2">&ldquo;a beach sitting on a cat&rdquo;</p>
              <p className="text-sm">
                Grammatically valid but semantically inverted. CLIP&rsquo;s
                transformer encodes &ldquo;beach&rdquo; as the subject and
                &ldquo;cat&rdquo; as the surface. The image may show an
                oversized cat with landscape elements on top&mdash;word order
                changes <em>meaning</em>, not just quality.
              </p>
            </GradientCard>
          </div>
          <div className="space-y-4 mt-4">
            <p className="text-muted-foreground">
              The 77-token limit means roughly 50&ndash;60 words. Prompts
              longer than this are silently truncated. Front-load the most
              important concepts because of the 77-token limit and because
              early tokens tend to have stronger influence through
              cross-attention.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Prompts Are Sentences">
            Prompts are sentences because CLIP is a transformer. Its
            self-attention makes every token embedding context-dependent.
            &ldquo;A cat on a beach&rdquo; and &ldquo;a beach on a cat&rdquo;
            produce different embeddings because the syntactic structure guides
            cross-attention differently. This is not keyword matching&mdash;it
            is contextual language understanding.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Predict and Verify */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  You want to generate a detailed portrait but the result looks
                  oversaturated with unnatural colors. Which parameter is most
                  likely too high, and what value would you try?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    <strong>guidance_scale</strong> is probably too high. Try
                    reducing from the current value toward 7&ndash;8. The
                    oversaturation comes from CFG extrapolating too far beyond
                    the conditional prediction. The{' '}
                    <InlineMath math="w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})" />{' '}
                    term becomes too large.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  Your colleague says they get better results with 200 steps
                  than 20 steps using DPM-Solver++. What would you investigate?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    Compare the actual images at 20 and 200 steps side by side.
                    With DPM-Solver++, the quality difference should be
                    negligible. Check whether the scheduler is actually
                    DPM-Solver++ and not DDPM, and whether the seed is being
                    set consistently between runs. The perceived
                    &ldquo;improvement&rdquo; may be from a different random
                    seed, not more steps.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  You want the exact same image every time you run the code.
                  What two things must you set?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    A fixed seed via{' '}
                    <code>generator=torch.Generator().manual_seed(N)</code>, AND
                    a deterministic scheduler (DDIM with{' '}
                    <InlineMath math="\sigma = 0" /> or DPM-Solver++). DDPM
                    adds random noise at each step, so even with the same seed
                    for <InlineMath math="z_T" />, per-step stochastic noise
                    changes the result.
                  </p>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 7: Practical Patterns */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practical Patterns"
            subtitle="Systematic experimentation, not random exploration"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The best workflow is not random exploration. It is systematic: fix
              everything except one parameter, vary it, understand its effect,
              then move to the next. This is the same controlled experiment
              methodology from science: change one variable, observe the effect.
            </p>
            <p className="text-muted-foreground">
              Start with defaults: DPM-Solver++ at 20 steps,{' '}
              <code>guidance_scale=7.5</code>, 512&times;512. Then refine:
              adjust <code>guidance_scale</code> for the right fidelity/quality
              tradeoff, add a negative prompt if common artifacts appear, try
              different seeds for different compositions.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Experimental Workflow">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Fix a seed and prompt</li>
              <li>Set defaults (DPM-Solver++, 20 steps, w=7.5)</li>
              <li>Vary one parameter at a time</li>
              <li>Compare results systematically</li>
              <li>Refine toward your target</li>
            </ol>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Common Failure Modes */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Common Failure Modes"
            subtitle="Diagnose from the mechanism, not a troubleshooting guide"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each failure mode maps to a concept you know. You can diagnose
              these by reasoning from the underlying mechanisms:
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-2 mt-4">
            <GradientCard title="Oversaturated / Distorted" color="rose">
              <p className="text-sm">
                <strong>Cause:</strong> <code>guidance_scale</code> too high.
                CFG is extrapolating the conditional-minus-unconditional
                direction too aggressively.
              </p>
              <p className="text-sm mt-1">
                <strong>Fix:</strong> Reduce toward 7&ndash;8.
              </p>
            </GradientCard>
            <GradientCard title="Blurry / Vague" color="amber">
              <p className="text-sm">
                <strong>Cause:</strong> <code>guidance_scale</code> too low
                (unconditional prediction dominating) or too few steps (trajectory
                not fully traversed).
              </p>
              <p className="text-sm mt-1">
                <strong>Fix:</strong> Increase <code>guidance_scale</code> toward 7.5 or increase steps.
              </p>
            </GradientCard>
            <GradientCard title="Repeated Subjects" color="orange">
              <p className="text-sm">
                <strong>Cause:</strong> Non-square or large resolution. The model
                was trained on 512&times;512 and may repeat patterns at larger sizes.
              </p>
              <p className="text-sm mt-1">
                <strong>Fix:</strong> Use 512&times;512, or use SD XL which handles
                multi-resolution.
              </p>
            </GradientCard>
            <GradientCard title="Prompt Not Followed" color="violet">
              <p className="text-sm">
                <strong>Cause:</strong> Prompt too long (truncated past 77 tokens),{' '}
                <code>guidance_scale</code> too low, or conflicting concepts in the prompt.
              </p>
              <p className="text-sm mt-1">
                <strong>Fix:</strong> Shorten prompt, increase guidance, remove conflicts.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: What the API Hides */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What the API Hides"
            subtitle="One function call, five familiar stages"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <code>pipe()</code> call wraps five stages you have done
              manually: tokenization, CLIP encoding, <InlineMath math="z_T" />{' '}
              sampling, the full denoising loop with CFG, and VAE decoding. All
              in one function.
            </p>
            <p className="text-muted-foreground">
              You executed each of these steps individually in the{' '}
              <strong>The Stable Diffusion Pipeline</strong> notebook. The API
              is convenience, not mystery.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Recipe vs. Ingredients">
            The gap between &ldquo;using an API&rdquo; and &ldquo;understanding
            an API&rdquo; is the gap between following a recipe and knowing why
            each ingredient works. You have the ingredients.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 10: Transfer Questions */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Questions" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  Your friend is generating images with a model they downloaded,
                  and every image looks identical even though they change the
                  prompt. They are not setting a seed. What could explain this?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    The scheduler might be deterministic AND the default
                    seed/generator state might be the same every run. Also
                    check: is the prompt actually being encoded differently? If
                    CLIP encoding fails silently and always returns the same
                    tensor, changing the prompt would have no effect. The
                    modular pipeline means each component can fail
                    independently.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  You want to generate a 1024&times;1024 image with SD v1.5.
                  How much more compute will the denoising loop take compared
                  to 512&times;512?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    The latent tensor is 128&times;128&times;4 instead of
                    64&times;64&times;4. That is 4&times; the spatial area, so
                    each U-Net pass processes 4&times; the data. With the same
                    number of steps, the denoising loop takes roughly
                    4&times; longer. CLIP encoding and VAE decode also take
                    longer, but the denoising loop dominates.
                  </p>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 11: Practice — Notebook link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Generate and Explore"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook is the primary vehicle for this lesson. Generate
                real images, explore parameters, and verify your predictions
                with the actual Stable Diffusion model.
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
                  <strong>Exercise 1 (Guided):</strong> Load
                  StableDiffusionPipeline, generate your first image with
                  default parameters, then modify the prompt and generate again.
                  Your first experience of &ldquo;I typed words and got an
                  image, and I know what happened inside.&rdquo;
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> Guidance scale
                  sweep&mdash;fix seed and prompt, generate at{' '}
                  <code>guidance_scale</code> = 1, 3, 7.5, 15, 25. Display as a
                  grid. Predict before running: which value produces the most
                  prompt-faithful image?
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Step count and
                  scheduler comparison&mdash;generate at 5, 10, 20, 50, 100
                  steps with DPM-Solver++, then swap to DDIM at 50 steps and
                  Euler at 30. Display all results with generation times.
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> Design your own
                  experiment. Pick a parameter, form a hypothesis, design a
                  controlled comparison, generate, and write a one-sentence
                  conclusion. Suggested: negative prompts, seeds, non-square
                  resolutions, or prompt wording.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Predict Before Running">
            Each exercise includes a prediction prompt. Before running the
            code, write down what you expect. This is the deliberate practice
            pattern: predict, run, compare, revise your mental model.
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
                  'Every parameter maps to a concept you built from scratch.',
                description:
                  'prompt \u2192 CLIP encoding \u2192 cross-attention. guidance_scale \u2192 CFG w parameter. num_inference_steps \u2192 sampler step count. scheduler \u2192 sampler choice. negative_prompt \u2192 CFG unconditional substitution. generator \u2192 seed \u2192 z_T. height/width \u2192 latent dimensions via VAE.',
              },
              {
                headline:
                  'The API is a dashboard, not a black box.',
                description:
                  'The pipe() call wraps tokenization, CLIP encoding, z_T sampling, the denoising loop with CFG, and VAE decoding. You executed each of these steps manually. The API is convenience, not mystery.',
              },
              {
                headline:
                  'Systematic experimentation beats random exploration.',
                description:
                  'Fix everything except one parameter. Vary it. Understand its effect. Move to the next. Start with defaults (DPM-Solver++ at 20 steps, guidance_scale=7.5, 512\u00d7512) and refine from there.',
              },
              {
                headline:
                  'You did not follow a tutorial. You understood the system.',
                description:
                  'Every parameter change you made had a predictable effect because you built the underlying concepts. This is the difference between using a tool and understanding a tool.',
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
              The API is a dashboard. You built the machine behind it.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Module completion */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="6.4"
            title="Stable Diffusion"
            achievements={[
              'Complete pipeline trace from text prompt to generated image with tensor shapes at every stage',
              'Sampler understanding: DDIM predict-and-leap, ODE perspective, DPM-Solver higher-order methods',
              'Full parameter comprehension: every diffusers API parameter mapped to its underlying concept',
              'Practical generation skills: systematic experimentation, failure mode diagnosis, parameter selection',
            ]}
            nextModule="6.5"
            nextTitle="Customization & Fine-Tuning"
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Module 6.5 teaches you to customize this machine. The first lesson
              introduces LoRA: inject small trainable matrices into the frozen
              U-Net to teach it new styles or subjects, without retraining the
              860M-parameter model from scratch.
            </p>
            <p className="text-muted-foreground">
              Remember how the U-Net, CLIP, and VAE are independently trained
              and swappable? LoRA takes this modularity further: you can add and
              remove fine-tuning layers like plugins.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: LoRA Fine-Tuning"
            description="Inject small trainable matrices into the frozen U-Net to teach it new styles or subjects—without retraining from scratch."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
