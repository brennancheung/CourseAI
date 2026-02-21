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
  TryThisBlock,
  WarningBlock,
  SummaryBlock,
  NextStepBlock,
  GradientCard,
  ComparisonRow,
  PhaseCard,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { DenoisingTrajectoryWidget } from '@/components/widgets/DenoisingTrajectoryWidget'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Sampling and Generation (DDPM Reverse Process)
 *
 * Lesson 4 in Module 6.2 (Diffusion). Lesson 8 overall in Series 6.
 * Cognitive load: BUILD.
 *
 * Teaches the DDPM reverse process (sampling algorithm):
 * - Why naive one-shot denoising fails (negative example hook)
 * - The reverse step formula and what each term means
 * - Why noise is added back during sampling (stochastic sampling)
 * - The full sampling loop from t=T to t=0
 * - Visualizing the denoising trajectory (coarse-to-fine)
 *
 * Core concepts:
 * - Reverse step formula: DEVELOPED
 * - Stochastic noise injection during sampling: DEVELOPED
 * - Full sampling loop structure: DEVELOPED
 *
 * Previous: Learning to Denoise (module 6.2, lesson 3)
 * Next: Build a Diffusion Model (module 6.2, lesson 5)
 */

export function SamplingAndGenerationLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Sampling and Generation"
            description="The DDPM reverse process&mdash;start from pure noise, iteratively denoise, and generate an image that has never existed."
            category="Diffusion"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Context + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Trace the DDPM sampling algorithm step by step&mdash;understand how a
            trained noise-prediction network transforms pure Gaussian noise into a
            coherent image through iterative denoising.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <LessonLink slug="learning-to-denoise">Learning to Denoise</LessonLink>, you traced the complete
            training algorithm. The model can predict noise at any timestep. Now:
            how do you use that ability to generate?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The DDPM reverse step formula and what each term means',
              'Why noise is added back during sampling (stochastic vs deterministic)',
              'The full sampling loop from t=T to t=0',
              'Visualizing the denoising trajectory—coarse-to-fine progression',
              'NOT: implementing the algorithm in code—that is the capstone lesson',
              'NOT: DDIM or other accelerated samplers—those come later',
              'NOT: classifier-free guidance or conditional generation',
              'NOT: U-Net architecture internals',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: The Missing Piece */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Missing Piece"
            subtitle="You know training. Now: generation."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have a trained model that takes a noisy image{' '}
              <InlineMath math="x_t" /> and a timestep{' '}
              <InlineMath math="t" />, and predicts the noise{' '}
              <InlineMath math="\epsilon" /> that was added. The forward process
              formula tells you how any clean image maps to any noise level. You
              know how to destroy and you know how to train.
            </p>
            <p className="text-muted-foreground">
              But you have never seen the model <strong>create</strong>. Starting
              from pure Gaussian noise&mdash;an image with zero information&mdash;how
              do you turn a noise predictor into a generator?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Lessons of Buildup">
            The intuition (Lesson 1), the math (Lesson 2), the training (Lesson
            3)&mdash;all were building toward this moment. This is where the
            trained model finally gets used.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Negative Example&mdash;One-Shot Denoising */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Tempting Shortcut"
            subtitle="Just subtract the noise. What could go wrong?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Your first instinct might be: the model predicts{' '}
              <InlineMath math="\epsilon_\theta" />, and you know the forward
              formula. So rearrange and solve for{' '}
              <InlineMath math="x_0" /> directly:
            </p>
            <div className="py-3 px-4 bg-rose-500/5 border border-rose-500/20 rounded-lg">
              <BlockMath math="\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}" />
            </div>
            <p className="text-muted-foreground">
              One prediction, one formula, done. No iterating. Why would anyone
              need 1,000 steps?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="This Does Not Work">
            At <InlineMath math="t = 500" />, the one-shot attempt produces a
            foggy smear of gray&mdash;vaguely the right brightness but no
            edges, no texture, no coherent structure. The model tried to jump
            directly from static to image and produced something between the
            two. A single imperfect prediction cannot recover all the
            information that was destroyed.
          </WarningBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The problem: the model&rsquo;s prediction{' '}
              <InlineMath math="\epsilon_\theta" /> is an <em>approximation</em>.
              At high noise levels (say{' '}
              <InlineMath math="t = 900" />), the image is almost pure static.
              The model is essentially hallucinating what the noise pattern might
              be. That hallucination is useful but imperfect&mdash;and if you
              stake everything on a single imperfect prediction, errors compound
              catastrophically.
            </p>
            <p className="text-muted-foreground">
              Think of it this way: if you could perfectly predict{' '}
              <InlineMath math="\epsilon" /> in one shot, you would not need a
              neural network at all&mdash;you could just invert the forward
              process analytically. But the forward process is stochastic. You
              cannot rewind randomness. The noise that was added to THIS specific
              image is unknowable; the model can only estimate it.
            </p>
            <p className="text-muted-foreground">
              This is why diffusion uses 1,000 steps, not 1. Each step makes a
              small correction based on the model&rsquo;s best guess at the
              current noise level. The errors in each step are tiny. And crucially,
              the model gets a fresh look at a slightly cleaner image each time,
              so it can course-correct.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 5: The Reverse Step Formula */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Reverse Step"
            subtitle="One small step toward a cleaner image"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One crucial difference from training: in training, you knew the
              true noise <InlineMath math="\epsilon" />&mdash;it was the answer
              key on an open-book exam. In sampling, there is no answer key.
              The model&rsquo;s prediction{' '}
              <InlineMath math="\epsilon_\theta" /> is all you have. You must
              trust that prediction and use it to take one small step back
              toward a cleaner image.
            </p>
            <p className="text-muted-foreground">
              Instead of jumping from{' '}
              <InlineMath math="x_t" /> all the way to{' '}
              <InlineMath math="x_0" />, the reverse process takes one small
              step&mdash;from <InlineMath math="x_t" /> to{' '}
              <InlineMath math="x_{t-1}" />. Here is the formula:
            </p>
            <div className="py-4 px-6 bg-violet-500/10 border-2 border-violet-500/30 rounded-lg">
              <BlockMath math="x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z" />
            </div>
            <p className="text-muted-foreground text-sm">
              where{' '}
              <InlineMath math="z \sim \mathcal{N}(0, I)" /> when{' '}
              <InlineMath math="t > 1" />, and{' '}
              <InlineMath math="z = 0" /> when{' '}
              <InlineMath math="t = 1" />. And{' '}
              <InlineMath math="\sigma_t = \sqrt{\beta_t}" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Connection to Training">
            Training taught the model to predict{' '}
            <InlineMath math="\epsilon_\theta(x_t, t)" />. This formula is how
            you USE that prediction&mdash;plugging it in to take one reverse step.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Term-by-term breakdown */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let us break this apart, term by term:
            </p>
            <div className="space-y-3">
              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  <InlineMath math="\frac{1}{\sqrt{\alpha_t}}" />&mdash;Scaling factor
                </p>
                <p className="text-sm text-muted-foreground">
                  Compensates for how much the signal was scaled down at this
                  step. Since{' '}
                  <InlineMath math="\alpha_t = 1 - \beta_t" /> is close to 1,
                  this factor is close to 1 as well.
                </p>
              </div>

              <div className="rounded-lg bg-muted/30 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  <InlineMath math="x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta" />&mdash;Noise removal
                </p>
                <p className="text-sm text-muted-foreground">
                  Take the current noisy image and subtract the model&rsquo;s
                  predicted noise, scaled by the right amount. This is the
                  &ldquo;denoising&rdquo; part&mdash;the model&rsquo;s best
                  guess at a slightly cleaner image.
                </p>
              </div>

              <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-3 space-y-1">
                <p className="text-sm font-medium text-foreground">
                  <InlineMath math="\sigma_t \cdot z" />&mdash;Fresh noise injection
                </p>
                <p className="text-sm text-muted-foreground">
                  This is the surprising part. After partially denoising, you
                  add a <em>small amount of fresh noise</em> back in. This is
                  not a bug&mdash;it serves a crucial purpose. We will explain
                  why shortly.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Familiar Structure">
            Notice the pattern: <strong>deterministic mean + sigma &times;
            z</strong>. This is the same structure as the reparameterization
            trick from <LessonLink slug="variational-autoencoders">Variational Autoencoders</LessonLink>. A learned
            mean plus controlled randomness.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Connection to forward formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Where does this formula come from? The forward process says{' '}
              <InlineMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon" />.
              The reverse step formula is derived by rearranging this
              relationship, using the model&rsquo;s noise prediction{' '}
              <InlineMath math="\epsilon_\theta" /> as a stand-in for the true{' '}
              <InlineMath math="\epsilon" />.
            </p>
            <p className="text-muted-foreground">
              In <LessonLink slug="the-forward-process">The Forward Process</LessonLink>, you derived the formula
              for destroying an image. Now you are running it in
              reverse&mdash;using the model&rsquo;s noise prediction to
              approximately undo one step of that destruction.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 5b: Concrete Walkthrough at t=500 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Concrete Walkthrough"
            subtitle="Plugging in real numbers at t=500"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let us trace one reverse step with specific values. We will use{' '}
              <InlineMath math="t = 500" />&mdash;the same timestep from{' '}
              <strong>Learning to Denoise</strong>&mdash;with a linear noise
              schedule.
            </p>

            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Setup at t=500 (linear schedule)
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                <li>
                  <InlineMath math="\bar{\alpha}_{500} \approx 0.05" /> (only 5%
                  signal remains)
                </li>
                <li>
                  <InlineMath math="\beta_{500} \approx 0.01" /> (from linear
                  schedule)
                </li>
                <li>
                  <InlineMath math="\alpha_{500} = 1 - \beta_{500} = 0.99" />
                </li>
                <li>
                  <InlineMath math="\sigma_{500} = \sqrt{\beta_{500}} = \sqrt{0.01} = 0.1" />
                </li>
              </ul>
            </div>

            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Plugging into the formula
              </p>
              <div className="py-2 px-4 bg-muted/50 rounded-lg">
                <BlockMath math="x_{499} = \frac{1}{\sqrt{0.99}} \left( x_{500} - \frac{0.01}{\sqrt{0.95}} \cdot \epsilon_\theta \right) + 0.1 \cdot z" />
              </div>
              <div className="py-2 px-4 bg-violet-500/5 border border-violet-500/20 rounded-lg">
                <BlockMath math="x_{499} \approx 1.005 \cdot (x_{500} - 0.0103 \cdot \epsilon_\theta) + 0.1 \cdot z" />
              </div>
              <p className="text-sm text-muted-foreground">
                The mean is almost exactly{' '}
                <InlineMath math="x_{500}" /> with a tiny noise correction
                subtracted. The noise injection (0.1 &times; z) adds a small
                random perturbation. Each step makes a <em>small</em>{' '}
                adjustment&mdash;this is why you need 1,000 of them.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Continuity with Training">
            In <LessonLink slug="learning-to-denoise">Learning to Denoise</LessonLink>, you used{' '}
            <InlineMath math="t = 500" /> for the training walkthrough. Same
            timestep, same schedule&mdash;now from the sampling side.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Coefficient comparison at different timesteps */}
      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-lg bg-muted/30 p-4 space-y-2">
              <p className="text-sm font-medium text-foreground">
                At t=950 (near pure noise)
              </p>
              <p className="text-sm text-muted-foreground">
                <InlineMath math="\beta_{950} \approx 0.019" />. The noise
                correction is larger relative to the signal. The model is
                making bold structural decisions: is this a shoe or a shirt?
                The model hallucinates structure from almost nothing.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-4 space-y-2">
              <p className="text-sm font-medium text-foreground">
                At t=50 (near clean)
              </p>
              <p className="text-sm text-muted-foreground">
                <InlineMath math="\beta_{50} \approx 0.0005" />. The noise
                correction is tiny. The model is making minute adjustments,
                polishing textures and fine details that are almost
                imperceptible.
              </p>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-muted-foreground">
              Same formula at every step. The coefficients change with the
              schedule, and the model&rsquo;s behavior adapts via the timestep
              embedding&mdash;but the algorithm is identical.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 6: Check 1 */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                You are at <InlineMath math="t = 500" />. The model predicts{' '}
                <InlineMath math="\epsilon_\theta" />. You compute the
                predicted mean. <strong>How much of the original image&rsquo;s
                information is in that prediction?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Very little. At <InlineMath math="t = 500" />,{' '}
                    <InlineMath math="\bar{\alpha}_{500} \approx 0.05" />&mdash;only
                    about 5% of the original signal remains. The prediction is
                    almost entirely the model&rsquo;s best guess about what an
                    image should look like, not a recovery of information still
                    present in <InlineMath math="x_t" />. The model is
                    hallucinating most of the structure from learned patterns.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Memory Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                The sampling algorithm runs for 1,000 steps. <strong>Does
                memory usage grow with T?</strong> Do you need to store all
                1,000 intermediate images?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    No. The algorithm only ever uses{' '}
                    <InlineMath math="x_t" /> to compute{' '}
                    <InlineMath math="x_{t-1}" />. Each step overwrites the
                    previous image. You only need to store one image at a
                    time (plus the model weights and schedule parameters).
                    Memory cost is constant regardless of the number of steps.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 7: Denoising Visualization Widget */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: The Denoising Trajectory"
            subtitle="Watch structure emerge from pure noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Step through the denoising process. Watch the same procedural
              T-shirt image emerge from noise, one step at a time. Notice how
              different stages contribute different kinds of information.
            </p>
          </div>
          <ExercisePanel
            title="Denoising Trajectory"
            subtitle="Scrub the timeline or press play to watch generation unfold"
          >
            <DenoisingTrajectoryWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>
                &bull; Step through slowly. When does the T-shirt first become
                recognizable?
              </li>
              <li>
                &bull; Compare <strong>t&nbsp;=&nbsp;800</strong> to{' '}
                <strong>t&nbsp;=&nbsp;200</strong>. Which shows more dramatic
                change from its neighbor?
              </li>
              <li>
                &bull; Look at the last 100 steps (t&nbsp;=&nbsp;100 to t&nbsp;=&nbsp;0). How much
                visible change happens?
              </li>
              <li>
                &bull; Not all steps are created equal. Early steps create
                structure. Late steps polish details.
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Why Add Noise Back? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Add Noise Back?"
            subtitle="The counterintuitive step that makes generation work"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The <InlineMath math="\sigma_t \cdot z" /> term in the reverse
              step formula seems wrong. You are trying to remove noise. Why
              would you add more?
            </p>
            <p className="text-muted-foreground">
              Remember <strong>temperature</strong> in language models? At
              temperature = 0, the model always picks the most likely next
              token&mdash;safe, repetitive, boring. At temperature &gt; 0, it
              explores less likely options&mdash;diverse, creative, surprising.
              The noise injection in diffusion sampling is the same idea.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Idea as Temperature">
            Temperature in language models controls diversity. The{' '}
            <InlineMath math="\sigma_t \cdot z" /> term in diffusion sampling
            does the same thing&mdash;it controls how much the model explores
            vs commits.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The model&rsquo;s noise prediction at any step is imperfect. If
              you follow the predicted mean exactly&mdash;zero noise
              injection&mdash;you commit fully to that imperfect estimate. Errors
              compound across 1,000 steps, and the result converges to a blurry,
              averaged image. The model collapses to the &ldquo;safest&rdquo;
              interpretation.
            </p>
            <p className="text-muted-foreground">
              Adding noise keeps options open. Think of it as hiking toward a
              mountain. Your compass (the model) points toward the peak. But
              you add a small random jitter to each step. Different hikes reach
              different spots on the mountain. Without jitter, every hike
              follows the exact same path&mdash;and if the compass is slightly
              miscalibrated, you end up consistently in the wrong place.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Stochastic (DDPM)',
              color: 'emerald',
              items: [
                'Adds fresh noise at each step (σₜ > 0)',
                'Different noise draws → different images',
                'Rich diversity in generated outputs',
                'Requires all T steps for best quality',
              ],
            }}
            right={{
              title: 'Deterministic (DDIM)',
              color: 'amber',
              items: [
                'No noise injection (σₜ = 0)',
                'Same starting noise → same image every time',
                'Can skip steps for faster generation',
                'A later development—trades diversity for speed',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="DDIM Is Later">
            DDIM (Denoising Diffusion Implicit Models) sets{' '}
            <InlineMath math="\sigma_t = 0" /> for deterministic, faster
            sampling. You will encounter it when we discuss accelerated
            samplers. For now, focus on the stochastic (DDPM) version.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This also addresses a subtle point: the sampling process is{' '}
              <strong>not deterministic</strong>. The trained model is a
              deterministic function&mdash;same input, same output. But the
              noise injection at each step introduces randomness. Run the
              sampling algorithm twice from the same starting noise{' '}
              <InlineMath math="x_T" />, and if you inject fresh{' '}
              <InlineMath math="z" /> at each step, you get different images.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Check 2 */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Your colleague says: &ldquo;Diffusion sampling is just running
                the forward process backward&mdash;instead of adding noise,
                you subtract it.&rdquo; <strong>What is wrong with this
                claim?</strong>
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    The forward process is a known mathematical
                    transformation&mdash;you choose the noise and add it. The
                    reverse process requires a <strong>trained model</strong> to
                    predict the noise, because you do not know what noise was
                    added to any specific image. You cannot simply subtract
                    something you do not know.
                  </p>
                  <p>
                    Also, the reverse step <em>adds</em> fresh noise for
                    exploration&mdash;there is no analog of that in the forward
                    process. The forward process only destroys; the reverse
                    process creates and explores.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <GradientCard title="Diversity Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                If you wanted <strong>more diverse</strong> outputs from your
                diffusion model, would you increase or decrease{' '}
                <InlineMath math="\sigma_t" />?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    Increase. More noise = more exploration at each step =
                    more diverse generated images. This is directly analogous
                    to raising the temperature in a language model.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: The Full Sampling Algorithm */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Full Sampling Algorithm"
            subtitle="From pure noise to generated image"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete DDPM sampling algorithm. You have seen each
              piece&mdash;now see how they compose into the full loop.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-3">
            <PhaseCard
              number={1}
              title="Sample pure noise"
              subtitle="Starting point"
              color="violet"
            >
              <p className="text-sm">
                Draw <InlineMath math="x_T \sim \mathcal{N}(0, I)" />. A random
                image of pure Gaussian noise. This is the raw material&mdash;the
                uncarved marble.
              </p>
            </PhaseCard>

            <PhaseCard
              number={2}
              title="Loop from T down to 1"
              subtitle="The iterative reverse process"
              color="violet"
            >
              <div className="text-sm space-y-2">
                <p>
                  For each <InlineMath math="t = T, T-1, \ldots, 1" />:
                </p>
                <div className="pl-3 border-l-2 border-violet-500/30 space-y-2">
                  <p>
                    <strong>a.</strong> If{' '}
                    <InlineMath math="t > 1" />, sample{' '}
                    <InlineMath math="z \sim \mathcal{N}(0, I)" />. If{' '}
                    <InlineMath math="t = 1" />, set{' '}
                    <InlineMath math="z = 0" />.
                  </p>
                  <p>
                    <strong>b.</strong> Predict noise:{' '}
                    <InlineMath math="\epsilon_\theta = \text{model}(x_t, t)" />
                  </p>
                  <p>
                    <strong>c.</strong> Compute the reverse step:{' '}
                  </p>
                  <div className="py-2 px-3 bg-muted/50 rounded-lg">
                    <BlockMath math="x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta \right) + \sigma_t \cdot z" />
                  </div>
                </div>
              </div>
            </PhaseCard>

            <PhaseCard
              number={3}
              title="Return the generated image"
              subtitle="The final result"
              color="blue"
            >
              <p className="text-sm">
                Return <InlineMath math="x_0" />. The iteratively denoised
                result is your generated image&mdash;an image that has never
                existed in the training data.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The t=1 Special Case">
            At the final step (<InlineMath math="t = 1 \to x_0" />),{' '}
            <InlineMath math="z = 0" />. No noise is added because this is the
            final image&mdash;there is no subsequent step to correct errors.{' '}
            <strong>The last step commits. Every step before it explores.</strong>
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Computational cost */}
      <Row>
        <Row.Content>
          <GradientCard title="The Cost of 1,000 Steps" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                Generating one image requires{' '}
                <InlineMath math="T" /> forward passes through the neural
                network&mdash;one at every timestep. For{' '}
                <InlineMath math="T = 1000" /> and a U-Net, this takes seconds
                to minutes per image depending on resolution.
              </p>
              <p>
                Imagine waiting 30 seconds for every single image. Now imagine
                generating a grid of 64 images. This pain is real and motivates
                everything that comes after&mdash;accelerated samplers, DDIM,
                latent diffusion, consistency models.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Training vs Sampling">
            Training touches one random timestep per image&mdash;one forward
            pass, one gradient update. Sampling must visit <em>every single
            timestep</em>&mdash;T forward passes, no gradients. Training is
            cheap per step. Sampling is expensive per image.
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
                  'The trained model predicts noise. The sampling algorithm uses that prediction to take one small step toward a cleaner image.',
                description:
                  `The reverse step formula rearranges the forward process, using the model’s noise prediction in place of the true noise.`,
              },
              {
                headline:
                  'Noise is added back at each step (except the last) to maintain diversity and prevent premature convergence.',
                description:
                  'The same principle as temperature in language models—σₜ > 0 produces diverse outputs, σₜ = 0 produces deterministic (DDIM-style) outputs.',
              },
              {
                headline:
                  'The full algorithm iterates from t=T (pure noise) to t=0 (generated image), requiring T forward passes.',
                description:
                  '1,000 neural network evaluations per image. This is painfully slow and motivates everything that follows—accelerated samplers, latent diffusion, and beyond.',
              },
              {
                headline:
                  'Early steps create structure. Late steps refine details.',
                description:
                  'The coarse-to-fine progression is not uniform—the first few hundred steps do the most dramatic work, while the last steps make subtle refinements.',
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
              Destruction was easy and known. Creation requires a trained guide.
              At each step the guide points toward the clean image, but a small
              jitter keeps the path from collapsing to a single boring route.
              1,000 tiny corrections compose into something that never existed.
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* Section 12: Next Step */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You can now trace the full diffusion pipeline: forward process
              (destroy), training (learn to predict noise), and sampling
              (iteratively denoise to generate). Next lesson, you will build it.
              All of it. From scratch.
            </p>
            <p className="text-muted-foreground">
              You will implement the forward process, training loop, and sampling
              loop on real data. You will feel the slowness firsthand&mdash;and
              understand exactly why latent diffusion was invented.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Denoising Diffusion Implicit Models',
                authors: 'Song, Meng & Ermon, 2020',
                url: 'https://arxiv.org/abs/2010.02502',
                note: 'Introduces DDIM — deterministic sampling that enables faster generation by skipping timesteps.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Build a Diffusion Model"
            description="Implement the complete DDPM pipeline from scratch—forward process, training loop, and sampling—on real image data."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
