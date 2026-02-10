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
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { AlphaBarCurveWidget } from '@/components/widgets/AlphaBarCurveWidget'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * The Forward Process
 *
 * Lesson 2 in Module 6.2 (Diffusion).
 * Cognitive load: STRETCH (hardest in the module).
 *
 * Teaches the mathematical formulation of the forward process:
 * - Gaussian noise properties (addition, variance scaling)
 * - The noise schedule (beta_t) and variance-preserving formulation
 * - Alpha and alpha_bar notation
 * - The closed-form shortcut q(x_t|x_0)
 *
 * Core concepts:
 * - Gaussian noise properties: INTRODUCED
 * - Noise schedule / variance-preserving: DEVELOPED
 * - alpha_bar and closed-form shortcut: DEVELOPED
 *
 * Previous: The Diffusion Idea (module 6.2, lesson 1)
 * Next: The Training Objective (module 6.2, lesson 3)
 */

// ---------------------------------------------------------------------------
// Static SVG: Two Gaussians combining into a wider Gaussian
// ---------------------------------------------------------------------------

function gaussianY(x: number, variance: number): number {
  return Math.exp(-(x * x) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance)
}

function gaussianPath(
  variance: number,
  xMin: number,
  xMax: number,
  steps: number,
  scaleX: number,
  scaleY: number,
  offsetX: number,
  baseline: number,
): string {
  const points: string[] = []
  for (let i = 0; i <= steps; i++) {
    const x = xMin + (xMax - xMin) * (i / steps)
    const y = gaussianY(x, variance)
    const px = offsetX + x * scaleX
    const py = baseline - y * scaleY
    points.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`)
  }
  return points.join(' ')
}

function GaussianAdditionVisual() {
  const w = 360
  const h = 80
  const baseline = h - 10
  const scaleX = 30
  const scaleY = 160
  const centerX = w / 2
  const steps = 80

  const var1 = 0.6
  const var2 = 1.0
  const varSum = var1 + var2

  const path1 = gaussianPath(var1, -4, 4, steps, scaleX, scaleY, centerX, baseline)
  const path2 = gaussianPath(var2, -4, 4, steps, scaleX, scaleY, centerX, baseline)
  const pathSum = gaussianPath(varSum, -4, 4, steps, scaleX, scaleY, centerX, baseline)

  return (
    <div className="flex flex-col items-center gap-1 mt-2">
      <svg
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        className="overflow-visible"
      >
        {/* Baseline */}
        <line
          x1={20}
          y1={baseline}
          x2={w - 20}
          y2={baseline}
          stroke="currentColor"
          strokeOpacity={0.15}
          strokeWidth={1}
        />
        {/* Narrow Gaussian 1 (dashed) */}
        <path
          d={path1}
          fill="none"
          stroke="#38bdf8"
          strokeWidth={1.5}
          strokeDasharray="4,3"
          opacity={0.7}
        />
        {/* Narrow Gaussian 2 (dashed) */}
        <path
          d={path2}
          fill="none"
          stroke="#38bdf8"
          strokeWidth={1.5}
          strokeDasharray="4,3"
          opacity={0.7}
        />
        {/* Wider combined Gaussian (solid) */}
        <path
          d={pathSum}
          fill="none"
          stroke="#8b5cf6"
          strokeWidth={2.5}
          opacity={0.9}
        />
        {/* Labels */}
        <text
          x={centerX + 3.2 * scaleX}
          y={baseline - 6}
          className="fill-sky-400"
          fontSize={9}
          opacity={0.8}
        >
          N(0, 3) and N(0, 5)
        </text>
        <text
          x={centerX + 3.2 * scaleX}
          y={baseline - 22}
          className="fill-violet-400"
          fontSize={9}
          fontWeight="bold"
        >
          N(0, 8)
        </text>
      </svg>
      <p className="text-[10px] text-muted-foreground/60">
        N(0, 3) + N(0, 5) = N(0, 8)&mdash;variances add, not standard deviations.
      </p>
    </div>
  )
}

export function TheForwardProcessLesson() {
  return (
    <LessonLayout>
      {/* ================================================================
          Section 1: Header
          ================================================================ */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Forward Process"
            description="The math behind noise addition&mdash;noise schedules, alpha-bar, and an elegant shortcut that makes training practical."
            category="Diffusion"
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 2: Context + Constraints (Outline item 1)
          ================================================================ */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Derive and use the closed-form formula that lets you jump to any
            noise level in one step. Understand why each design choice in the
            forward process&mdash;Gaussian noise, variance-preserving scaling,
            the noise schedule&mdash;exists.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            In <strong>The Diffusion Idea</strong>, you watched images dissolve
            into noise and understood <em>why</em> small denoising steps make
            generation learnable. This lesson formalizes <em>how</em> that
            noise addition works mathematically.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The mathematical forward process: how noise is added step by step',
              'Gaussian properties needed for the derivation (addition, variance scaling)',
              'The noise schedule beta_t and the variance-preserving formulation',
              'Alpha-bar notation and the closed-form shortcut',
              'NOT: the reverse process, denoising, or the training loss\u2014those come next',
              'NOT: the sampling algorithm or code implementation',
              'NOT: U-Net architecture or any neural network details',
              'NOT: score matching, SDEs, or continuous-time formulations',
            ]}
          />
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 3: Hook / Puzzle (Outline item 2)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="A Practical Problem"
            subtitle="Why you need a mathematical shortcut"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>The Diffusion Idea</strong>, you learned that the
              training loop needs noisy images at <em>random</em> timesteps. To
              train, you pick a clean image, pick a random timestep{' '}
              <InlineMath math="t" />, and create the noisy version at step{' '}
              <InlineMath math="t" />.
            </p>
            <p className="text-muted-foreground">
              But the forward process is defined <strong>step by step</strong>.
              If you need the noisy image at step 500, do you really have to
              compute all 500 previous steps? With 1,000 timesteps and 50,000
              training images, that is 50 million noise-addition operations{' '}
              <em>per epoch</em>. Training would be painfully slow.
            </p>
            <p className="text-muted-foreground">
              There <strong>must</strong> be a shortcut&mdash;a formula that
              jumps directly to any timestep without iterating through all the
              intermediate steps. This lesson derives that formula. By the end,
              you will have one equation that teleports you from a clean image to
              any noise level in a single step.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Destination">
            One formula. Input: a clean image and a timestep. Output: the noisy
            image at that timestep. No iteration, no intermediate steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 4: Two Gaussian Properties (Outline item 3)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Gaussian Properties You Need"
            subtitle="Tools for the derivation ahead"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already know Gaussians&mdash;you sampled{' '}
              <InlineMath math="\epsilon \sim \mathcal{N}(0,1)" /> in the
              reparameterization trick. Here are two properties you will need in
              about five minutes. Think of these as tools, not deep theory.
            </p>

            <div className="rounded-lg bg-muted/30 p-4 space-y-4">
              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Property 1: Gaussians add to Gaussians
                </p>
                <p className="text-muted-foreground text-sm">
                  If you add two independent Gaussian samples, the result is
                  also Gaussian, and the <strong>variances add</strong>:
                </p>
                <div className="py-2 px-4 bg-muted/50 rounded-lg">
                  <BlockMath math="\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)" />
                </div>
                <p className="text-muted-foreground text-sm">
                  Concrete example: if you add a sample from{' '}
                  <InlineMath math="\mathcal{N}(0, 3)" /> and an independent
                  sample from <InlineMath math="\mathcal{N}(0, 5)" />, the
                  result comes from{' '}
                  <InlineMath math="\mathcal{N}(0, 8)" />. The variances
                  (3 + 5 = 8), not the standard deviations, are what add.
                </p>

                {/* Visual: two narrow Gaussians combining into a wider one */}
                <GaussianAdditionVisual />
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium text-foreground">
                  Property 2: Scaling scales variance
                </p>
                <p className="text-muted-foreground text-sm">
                  If you multiply a Gaussian sample by a constant{' '}
                  <InlineMath math="c" />, the variance scales by{' '}
                  <InlineMath math="c^2" />:
                </p>
                <div className="py-2 px-4 bg-muted/50 rounded-lg">
                  <BlockMath math="\text{If } X \sim \mathcal{N}(0,1), \text{ then } c \cdot X \sim \mathcal{N}(0, c^2)" />
                </div>
                <p className="text-muted-foreground text-sm">
                  You have already used this. In the reparameterization trick,{' '}
                  <InlineMath math="\sigma \cdot \epsilon" /> gave you a sample
                  with variance{' '}
                  <InlineMath math="\sigma^2" />&mdash;that is this property in
                  action.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Two facts. Gaussians add cleanly, and scaling a Gaussian scales
              its variance by the square of the scalar. Keep these in mind.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why These Matter">
            These two properties are what make the closed-form shortcut
            possible. Without them, you would be stuck iterating step by step.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 5: One Step of Noise (Outline item 4)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One Step of Noise"
            subtitle="Formalizing what the slider showed you"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <strong>The Diffusion Idea</strong>, you dragged a slider and
              watched an image dissolve into noise. Now: what formula was the
              slider computing at each step?
            </p>
            <p className="text-muted-foreground">
              Start with the simplest version. At each step, we add some noise.
              How much? Call it <InlineMath math="\beta_t" />&mdash;a small
              number that controls the noise amount at step{' '}
              <InlineMath math="t" />.
            </p>
            <p className="text-muted-foreground">
              The naive approach would be to just add noise directly:
            </p>
            <div className="py-3 px-4 bg-rose-500/5 border border-rose-500/20 rounded-lg">
              <BlockMath math="x_t = x_{t-1} + \sqrt{\beta_t} \cdot \epsilon \quad \text{(naive\u2014don't do this)}" />
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Beta">
            <InlineMath math="\beta_t" /> is the noise fraction at step{' '}
            <InlineMath math="t" />. A small number (typically 0.0001 to 0.02)
            that controls how much noise is added at each step.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Variance-exploding negative example */}
      <Row>
        <Row.Content>
          <GradientCard title="Why Naive Addition Fails" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                If you just add noise without scaling the signal down, the
                variance <strong>grows without bound</strong>. After 100 steps,
                pixel values blow up to ridiculous magnitudes. The image does
                not converge to neat Gaussian noise&mdash;it{' '}
                <strong>explodes</strong>.
              </p>
              <p>
                Try the mental math: if each step adds noise with variance{' '}
                <InlineMath math="\beta" />, after{' '}
                <InlineMath math="T" /> steps the total variance is the
                original image variance <em>plus</em>{' '}
                <InlineMath math="T \cdot \beta" />. For{' '}
                <InlineMath math="T = 1000" /> and{' '}
                <InlineMath math="\beta = 0.01" />, the final variance is
                roughly 11 times the original. Not useful.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Variance Exploding">
            The name is literal. Without controlling variance, pixel values
            grow without limit. The image does not become clean Gaussian
            noise&mdash;it becomes meaningless numbers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* The variance-preserving formula */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The fix: <strong>scale the signal down</strong> before adding
              noise. Think of it as mixing&mdash;each step blends old signal with
              new noise, keeping the total amount constant. Like mixing paint:
              remove some old paint before adding new color, and the total
              volume stays the same.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)" />
            </div>

            <p className="text-muted-foreground">
              Every piece exists for a reason:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="\sqrt{1 - \beta_t}" /> scales the signal
                down&mdash;the signal <em>shrinks</em> slightly each step
              </li>
              <li>
                <InlineMath math="\sqrt{\beta_t}" /> controls how much new noise
                enters
              </li>
              <li>
                <InlineMath math="\epsilon \sim \mathcal{N}(0, I)" />&mdash;fresh
                Gaussian noise, independent at each pixel
              </li>
            </ul>

            <p className="text-muted-foreground">
              Why these specific coefficients? Check the variance. If{' '}
              <InlineMath math="x_{t-1}" /> has variance 1:
            </p>

            <div className="py-3 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="\text{Var}(x_t) = (1 - \beta_t) \cdot \text{Var}(x_{t-1}) + \beta_t \cdot 1 = (1 - \beta_t) + \beta_t = 1" />
            </div>

            <p className="text-muted-foreground">
              The variance stays at 1. This is why it is called{' '}
              <strong>variance-preserving</strong>. The coefficients are not
              arbitrary&mdash;they are designed to keep the total variance
              constant throughout the entire process.
            </p>
            <p className="text-xs text-muted-foreground/80 italic">
              Why does <InlineMath math="x_0" /> have variance near 1? In
              practice, images are normalized (typically to [-1, 1]) before
              the forward process begins. As long as the input has variance
              approximately 1, the variance-preserving property keeps it near
              1 at every subsequent step.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Variance-Preserving">
            Each step is a weighted blend of signal and noise. The weights{' '}
            <InlineMath math="(1 - \beta_t) + \beta_t = 1" /> keep variance
            constant. You are not piling noise on top of signal&mdash;you are
            gradually <em>replacing</em> signal with noise.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 6: Predict-and-Verify (Outline item 5)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                <strong>
                  If <InlineMath math="\beta_t = 0" /> (no noise this step),
                  what does the formula give?
                </strong>
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <InlineMath math="x_t = \sqrt{1} \cdot x_{t-1} + \sqrt{0} \cdot \epsilon = x_{t-1}" />
                    . The image is unchanged. No noise, no change.
                  </p>
                </div>
              </details>

              <p className="mt-3">
                <strong>
                  If <InlineMath math="\beta_t = 1" /> (maximum noise), what
                  happens?
                </strong>
              </p>
              <details className="mt-2">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2">
                  <p>
                    <InlineMath math="x_t = \sqrt{0} \cdot x_{t-1} + \sqrt{1} \cdot \epsilon = \epsilon" />
                    . The image is replaced entirely by random noise. All signal
                    destroyed in one step.
                  </p>
                </div>
              </details>

              <p className="mt-3 text-muted-foreground">
                So <InlineMath math="\beta_t" /> controls the blend between
                signal and noise at each step. At the extremes, the formula does
                exactly what you would expect.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 7: The Noise Schedule (Outline item 6)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Noise Schedule"
            subtitle="How fast should we destroy the image?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <InlineMath math="\beta_t" /> is not a single number&mdash;it
              changes at each timestep. The sequence{' '}
              <InlineMath math="\{\beta_1, \beta_2, \ldots, \beta_T\}" /> is the{' '}
              <strong>noise schedule</strong>. It is a design choice that
              controls how fast the image is destroyed.
            </p>
            <p className="text-muted-foreground">
              The original DDPM paper used a <strong>linear schedule</strong>:{' '}
              <InlineMath math="\beta" /> starts very small (0.0001) and grows
              linearly to a larger value (0.02). Why not constant? Because the
              amount of signal remaining changes as the process proceeds. Early
              on, the image has lots of signal&mdash;a tiny{' '}
              <InlineMath math="\beta" /> is enough to start the process. Later,
              the image is mostly noise&mdash;a larger{' '}
              <InlineMath math="\beta" /> finishes the job.
            </p>
            <p className="text-muted-foreground">
              If <InlineMath math="\beta" /> were constant at, say, 0.01 for
              all 1,000 steps, the first 100 steps would destroy as much
              information as the last 100. But the early steps have far more
              signal to preserve&mdash;wasting large noise on them throws
              away fine detail that the model could have learned from. A
              constant schedule treats every stage of the process identically,
              which is the wrong tradeoff.
            </p>
            <p className="text-muted-foreground">
              Remember the widget from <strong>The Diffusion Idea</strong>? The
              first few steps barely changed the image. The last few steps
              erased the remaining traces. That was the schedule at work.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="A Design Choice">
            The noise schedule is not given by physics or mathematics. It is a
            design choice made by the researcher. Different schedules produce
            different results.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 8: Alpha Notation (Outline item 7)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Alpha Notation"
            subtitle="Progressive simplification, not progressive complexity"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The formula{' '}
              <InlineMath math="x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon" />{' '}
              works, but writing{' '}
              <InlineMath math="1 - \beta_t" /> everywhere gets cumbersome.
              Let us define a shorthand:
            </p>

            <div className="py-3 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="\alpha_t = 1 - \beta_t" />
            </div>

            <p className="text-muted-foreground">
              This is just a renaming. Where{' '}
              <InlineMath math="\beta_t" /> is the <strong>noise
              fraction</strong>, <InlineMath math="\alpha_t" /> is the{' '}
              <strong>signal fraction</strong>. The single-step formula becomes:
            </p>

            <div className="py-3 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \epsilon" />
            </div>

            <p className="text-muted-foreground">
              But the real payoff is{' '}
              <InlineMath math="\bar{\alpha}_t" /> (alpha-bar):
            </p>

            <div className="py-4 px-6 bg-violet-500/5 border border-violet-500/20 rounded-lg">
              <BlockMath math="\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdot \ldots \cdot \alpha_t = \prod_{i=1}^{t} \alpha_i" />
            </div>

            <p className="text-muted-foreground">
              <InlineMath math="\bar{\alpha}_t" /> is the{' '}
              <strong>cumulative signal fraction</strong>&mdash;how much of
              the original image survives after <InlineMath math="t" /> steps.
              It starts near 1 (image mostly preserved) and drops to near 0
              (image destroyed). This one number tells you everything about the
              forward process at any timestep.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The One Number That Matters">
            <InlineMath math="\bar{\alpha}_t" /> encodes the entire history of
            the noise schedule up to step{' '}
            <InlineMath math="t" />. Without it, you need all the individual{' '}
            <InlineMath math="\beta" /> values. With it, you need one number.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Misconception: notation is unnecessarily complex */}
      <Row>
        <Row.Content>
          <GradientCard title="Why Not Just Use Beta?" color="amber">
            <div className="space-y-3 text-sm">
              <p>
                The notation soup&mdash;
                <InlineMath math="\beta" />,{' '}
                <InlineMath math="\alpha" />,{' '}
                <InlineMath math="\bar{\alpha}" />&mdash;can feel like
                unnecessary layers of abstraction. Why not stick with{' '}
                <InlineMath math="\beta" />?
              </p>
              <p>
                Because <InlineMath math="\bar{\alpha}_t" /> is the punchline.
                It is the <em>only</em> number you need to describe the noise
                level at any timestep. The closed-form formula (coming next)
                uses only <InlineMath math="\bar{\alpha}_t" />&mdash;not any
                individual <InlineMath math="\beta" /> values. The apparent
                complexity dissolves once you see why each name exists.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 9: Interactive Widget (Outline item 8)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore: The Alpha-Bar Curve"
            subtitle="See what the formula produces at every timestep"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Drag the marker along the{' '}
              <InlineMath math="\bar{\alpha}_t" /> curve. Watch the image change
              and the formula coefficients update in real time. The curve IS the
              noise schedule&mdash;it shows exactly how much signal remains at
              every timestep.
            </p>
          </div>
          <ExercisePanel
            title="The Alpha-Bar Curve"
            subtitle="Click or drag the curve to explore different timesteps"
          >
            <AlphaBarCurveWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>
                &bull; Find the timestep where{' '}
                <InlineMath math="\bar{\alpha} = 0.5" /> (equal parts signal and
                noise). What does the image look like?
              </li>
              <li>
                &bull; Move to very early timesteps&mdash;
                <InlineMath math="\bar{\alpha}" /> near 1. Can you tell the
                image is noisy?
              </li>
              <li>
                &bull; Move to very late timesteps&mdash;
                <InlineMath math="\bar{\alpha}" /> near 0. Any trace of the
                original?
              </li>
              <li>
                &bull; Watch the coefficients: when{' '}
                <InlineMath math="\bar{\alpha}" /> is large, the{' '}
                <InlineMath math="x_0" /> term dominates. When it is small, the{' '}
                <InlineMath math="\epsilon" /> term takes over.
              </li>
              <li>
                &bull; Toggle between cosine and linear schedules. Notice how
                the linear schedule drops faster in early timesteps.
              </li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 10: The Closed-Form Shortcut (Outline item 9)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Closed-Form Shortcut"
            subtitle="From step-by-step to direct jump"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now for the payoff. The forward process is defined recursively:
              each step depends on the previous one. But you can unroll the
              recursion using the Gaussian properties from earlier.
            </p>
            <p className="text-muted-foreground">
              Start with one step:
            </p>
            <div className="py-2 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_1 = \sqrt{\alpha_1} \cdot x_0 + \sqrt{1 - \alpha_1} \cdot \epsilon_1" />
            </div>

            <p className="text-muted-foreground">
              Now a second step:
            </p>
            <div className="py-2 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_2 = \sqrt{\alpha_2} \cdot x_1 + \sqrt{1 - \alpha_2} \cdot \epsilon_2" />
            </div>

            <p className="text-muted-foreground">
              Substitute <InlineMath math="x_1" /> into the second equation:
            </p>
            <div className="py-2 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_2 = \sqrt{\alpha_2} \left( \sqrt{\alpha_1} \cdot x_0 + \sqrt{1 - \alpha_1} \cdot \epsilon_1 \right) + \sqrt{1 - \alpha_2} \cdot \epsilon_2" />
            </div>

            <p className="text-muted-foreground">
              Distribute and collect terms:
            </p>
            <div className="py-2 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_2 = \sqrt{\alpha_1 \alpha_2} \cdot x_0 + \left( \sqrt{\alpha_2(1 - \alpha_1)} \cdot \epsilon_1 + \sqrt{1 - \alpha_2} \cdot \epsilon_2 \right)" />
            </div>

            <p className="text-muted-foreground">
              Now apply the two Gaussian properties. Each noise term is a scaled{' '}
              <InlineMath math="\mathcal{N}(0,1)" /> sample. By{' '}
              <strong>Property 2</strong> (scaling scales variance):
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="\sqrt{\alpha_2(1 - \alpha_1)} \cdot \epsilon_1" />{' '}
                has variance{' '}
                <InlineMath math="\alpha_2(1 - \alpha_1) \cdot \text{Var}(\epsilon_1) = \alpha_2(1 - \alpha_1)" />
              </li>
              <li>
                <InlineMath math="\sqrt{1 - \alpha_2} \cdot \epsilon_2" />{' '}
                has variance{' '}
                <InlineMath math="(1 - \alpha_2) \cdot \text{Var}(\epsilon_2) = 1 - \alpha_2" />
              </li>
            </ul>
            <p className="text-muted-foreground">
              These two noise terms are independent Gaussians. By{' '}
              <strong>Property 1</strong> (variances add), the combined noise
              has variance:
            </p>
            <div className="py-2 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="\alpha_2(1 - \alpha_1) + (1 - \alpha_2) = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1 \alpha_2" />
            </div>

            <p className="text-muted-foreground">
              So we can replace the two noise terms with a single Gaussian
              sample:
            </p>
            <div className="py-3 px-4 bg-muted/50 rounded-lg">
              <BlockMath math="x_2 = \sqrt{\alpha_1 \alpha_2} \cdot x_0 + \sqrt{1 - \alpha_1 \alpha_2} \cdot \epsilon" />
            </div>

            <p className="text-muted-foreground">
              The product <InlineMath math="\alpha_1 \alpha_2" /> is{' '}
              <InlineMath math="\bar{\alpha}_2" />. And the pattern holds for
              any number of steps. The closed-form formula:
            </p>

            <div className="py-4 px-6 bg-violet-500/10 border-2 border-violet-500/30 rounded-lg">
              <BlockMath math="q(x_t | x_0) = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)" />
            </div>

            <p className="text-muted-foreground">
              Notice the structure. This is the{' '}
              <strong>reparameterization trick</strong> again: signal + noise_scale
              &times; epsilon. In the VAE, you wrote{' '}
              <InlineMath math="z = \mu + \sigma \cdot \epsilon" />. Here, the
              &ldquo;mu&rdquo; is{' '}
              <InlineMath math="\sqrt{\bar{\alpha}_t} \cdot x_0" /> and the
              &ldquo;sigma&rdquo; is{' '}
              <InlineMath math="\sqrt{1 - \bar{\alpha}_t}" />. Same pattern,
              different context. The closed-form formula is not a new thing to
              memorize&mdash;it is a variation on a structure you already know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern">
            Signal term times{' '}
            <InlineMath math="x_0" />, plus noise term times{' '}
            <InlineMath math="\epsilon" />. The reparameterization trick:{' '}
            <InlineMath math="z = \mu + \sigma \cdot \epsilon" />.
          </InsightBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This is the formula you saw updating in the widget. No iteration.
              Pick any timestep <InlineMath math="t" />, look up{' '}
              <InlineMath math="\bar{\alpha}_t" />, and compute the noisy image
              directly from the clean image. Step 500? One multiplication, one
              addition. Same as step 1. Same as step 999.
            </p>
            <p className="text-muted-foreground">
              Think of it like a map. The step-by-step definition tells you the
              route&mdash;walk one block north, then one block east, then
              another block north. The closed-form formula tells you the
              destination&mdash;you are at coordinates (2, 1). Same place,
              direct access.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 11: 1D Pixel Walkthrough (Outline item 10)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Verification: 1D Pixel Walkthrough"
            subtitle="Same destination, whether you walk or teleport"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let us verify with a single pixel. Start with a pixel value of{' '}
              <InlineMath math="0.8" />. Apply three steps with specific noise
              and epsilon values:
            </p>

            <div className="rounded-lg bg-muted/30 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Step-by-step approach
              </p>
              <div className="space-y-2 text-sm text-muted-foreground font-mono">
                <p>
                  <InlineMath math="\beta_1 = 0.01, \quad \beta_2 = 0.02, \quad \beta_3 = 0.03" />
                </p>
                <p>
                  <InlineMath math="\epsilon_1 = 0.5, \quad \epsilon_2 = -0.3, \quad \epsilon_3 = 0.8" />
                </p>
              </div>

              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <strong>Step 1:</strong>{' '}
                  <InlineMath math="x_1 = \sqrt{0.99} \cdot 0.8 + \sqrt{0.01} \cdot 0.5 = 0.995 \cdot 0.8 + 0.1 \cdot 0.5 = 0.846" />
                </p>
                <p>
                  <strong>Step 2:</strong>{' '}
                  <InlineMath math="x_2 = \sqrt{0.98} \cdot 0.846 + \sqrt{0.02} \cdot (-0.3) = 0.990 \cdot 0.846 + 0.141 \cdot (-0.3) = 0.795" />
                </p>
                <p>
                  <strong>Step 3:</strong>{' '}
                  <InlineMath math="x_3 = \sqrt{0.97} \cdot 0.795 + \sqrt{0.03} \cdot 0.8 = 0.985 \cdot 0.795 + 0.173 \cdot 0.8 = 0.921" />
                </p>
              </div>
            </div>

            <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-4 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Closed-form shortcut
              </p>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  <InlineMath math="\bar{\alpha}_3 = 0.99 \times 0.98 \times 0.97 = 0.9412" />
                </p>
                <p>
                  <InlineMath math="\sqrt{\bar{\alpha}_3} = 0.9702, \quad \sqrt{1 - \bar{\alpha}_3} = 0.2425" />
                </p>
                <p>
                  The closed-form says{' '}
                  <InlineMath math="x_3" /> is drawn from:
                </p>
                <div className="py-2 px-4 bg-muted/50 rounded-lg">
                  <BlockMath math="x_3 = 0.9702 \cdot x_0 + 0.2425 \cdot \epsilon" />
                </div>
                <p>
                  With <InlineMath math="x_0 = 0.8" />, this is a sample from{' '}
                  <InlineMath math="\mathcal{N}(0.9702 \times 0.8, \; 0.2425^2) = \mathcal{N}(0.776, \; 0.0588)" />.
                </p>
                <p>
                  Let us verify the step-by-step result lands in this distribution.
                  The three epsilon draws combine into a single effective epsilon.
                  From our step-by-step result:{' '}
                  <InlineMath math="x_3 = 0.921" />, so the
                  effective <InlineMath math="\epsilon" /> was:
                </p>
                <div className="py-2 px-4 bg-muted/50 rounded-lg">
                  <BlockMath math="\epsilon_{\text{eff}} = \frac{x_3 - 0.9702 \cdot x_0}{0.2425} = \frac{0.921 - 0.776}{0.2425} = 0.598" />
                </div>
                <p>
                  An epsilon of 0.598 is well within the range of{' '}
                  <InlineMath math="\mathcal{N}(0,1)" /> (less than one standard
                  deviation). Now plug it back into the closed-form formula:
                </p>
                <div className="py-2 px-4 bg-muted/50 rounded-lg">
                  <BlockMath math="0.9702 \times 0.8 + 0.2425 \times 0.598 = 0.776 + 0.145 = 0.921 \; \checkmark" />
                </div>
                <p>
                  Exactly the step-by-step result. The loop is closed&mdash;same
                  distribution, same answer.
                </p>
              </div>
            </div>

            <p className="text-muted-foreground">
              The step-by-step and closed-form give samples from the same
              distribution. Three separate epsilons combine into one effective
              epsilon&mdash;and the closed-form with that epsilon reproduces
              the step-by-step result exactly. The shortcut is not an
              approximation&mdash;it is <strong>exact</strong>. The Gaussian
              addition property guarantees this.
            </p>
            <p className="text-muted-foreground">
              And this generalizes beyond a single pixel. The widget you explored
              earlier IS the closed-form formula in action on a full image. Every
              pixel follows exactly the same math you just verified&mdash;the
              formula applies independently to each pixel. The 28&times;28 image
              at any timestep is just 784 copies of this same calculation, one per
              pixel, all using the same{' '}
              <InlineMath math="\bar{\alpha}_t" /> but independent epsilon draws.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Distribution">
            The step-by-step approach uses three independent epsilon values.
            The closed-form uses one. The results are different realizations,
            but drawn from the same probability distribution. For training,
            this is all that matters.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 12: Transfer Check (Outline item 11)
          ================================================================ */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Check" color="cyan">
            <div className="space-y-3 text-sm">
              <p>
                Your colleague says: &ldquo;The closed-form formula is just an
                approximation&mdash;it cannot give exactly the same result as
                running all the steps.&rdquo; Is this correct?
              </p>
              <details className="mt-3">
                <summary className="font-medium cursor-pointer text-primary">
                  Reveal
                </summary>
                <div className="mt-2 space-y-2">
                  <p>
                    No. The formula is <strong>exact</strong>, not approximate.
                    The Gaussian addition property is exact: the sum of
                    independent Gaussians IS Gaussian. The step-by-step and
                    closed-form produce samples from mathematically identical
                    distributions. The only difference is which epsilon you
                    use&mdash;but both are drawn from the same distribution.
                  </p>
                </div>
              </details>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 13: Why Gaussian? (Outline item 12)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Gaussian Noise?"
            subtitle="Not the only choice, but the best choice"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now that you have seen the closed-form formula, you can understand
              why Gaussian noise was chosen:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>The addition property:</strong> Gaussians add to
                Gaussians. This is what made the closed-form shortcut possible.
                If we used uniform noise, the sum of many steps would not
                collapse to a clean closed form.
              </li>
              <li>
                <strong>Central limit theorem:</strong> Even if individual steps
                were not exactly Gaussian, many steps would converge to
                Gaussian by the CLT. Nature gravitates toward Gaussians.
              </li>
              <li>
                <strong>Two-number description:</strong> A Gaussian is fully
                described by its mean and variance. Two numbers. This makes
                the math tractable at every step.
              </li>
            </ul>
            <p className="text-muted-foreground">
              Gaussian is not the only possible choice, but it is the choice
              that makes everything downstream elegant.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Elegance, Not Accident">
            Every design choice in the forward process&mdash;Gaussian noise,
            variance-preserving scaling, the specific coefficients&mdash;exists
            to enable the closed-form shortcut. The math is not arbitrary; it
            is carefully designed.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 14: Schedule Comparison (Outline item 13)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Schedule Comparison"
            subtitle="Not all destruction rates are equal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The linear schedule used in the original DDPM paper has a
              problem: it destroys information <strong>too quickly</strong> at
              early timesteps. The{' '}
              <InlineMath math="\bar{\alpha}" /> curve drops steeply at the
              start, leaving the model with very few &ldquo;easy&rdquo;
              denoising examples (images with light noise).
            </p>
            <p className="text-muted-foreground">
              The <strong>cosine schedule</strong> (introduced by Improved DDPM)
              spends more time at high{' '}
              <InlineMath math="\bar{\alpha}" /> values&mdash;low noise
              levels&mdash;giving the model more practice with fine details. If
              you tried the schedule toggle in the widget, you saw the
              difference: the cosine curve drops more gradually in early
              timesteps.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'Linear Schedule',
              color: 'amber',
              items: [
                'Beta grows linearly from 0.0001 to 0.02',
                'Alpha-bar drops steeply early on',
                'Fewer training samples at low noise',
                'Original DDPM paper (2020)',
              ],
            }}
            right={{
              title: 'Cosine Schedule',
              color: 'emerald',
              items: [
                'Alpha-bar follows a cosine curve',
                'Gentle decline at early timesteps',
                'More training samples at low noise',
                'Improved DDPM paper (2021)',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Widget Connection">
            The widget from <strong>The Diffusion Idea</strong> used a cosine
            schedule. That is why the image degraded slowly at first and then
            rapidly near the end. Now you know the math behind that behavior.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* ================================================================
          Section 15: Summary (Outline item 14)
          ================================================================ */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline:
                  'Each forward step blends signal and noise with variance-preserving coefficients.',
                description:
                  'The formula x_t = \u221A(1\u2212\u03B2_t) \u00B7 x_{t-1} + \u221A\u03B2_t \u00B7 \u03B5 scales the signal down before adding noise, keeping total variance at 1. The noise amount at each step is controlled by the schedule \u03B2_t.',
              },
              {
                headline:
                  '\u03B1\u0305_t is the one number that tells you everything about noise level t.',
                description:
                  'The cumulative product of signal fractions. Starts near 1 (clean) and drops to near 0 (pure noise). It encodes the entire history of the noise schedule.',
              },
              {
                headline:
                  'The closed-form formula lets you jump to ANY timestep in one step.',
                description:
                  'q(x_t|x_0) = \u221A\u03B1\u0305_t \u00B7 x_0 + \u221A(1\u2212\u03B1\u0305_t) \u00B7 \u03B5. No iteration needed. This is what makes training practical\u2014sample a random t and jump straight there.',
              },
              {
                headline:
                  'Every design choice exists for a reason.',
                description:
                  'Gaussian noise for the addition property (enables the shortcut). Variance-preserving scaling to keep values bounded. The schedule to control the destruction rate. Nothing is arbitrary.',
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
              &ldquo;
              <InlineMath math="\bar{\alpha}_t" /> is the signal-to-noise dial.
              The closed-form formula lets you turn that dial to any position
              without stepping through all the intermediate values.&rdquo;
            </strong>
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* ================================================================
          Section 16: Next Step (Outline item 15)
          ================================================================ */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now have the forward process&mdash;the mathematical machinery
              for creating noisy training examples at any timestep in one step.
              Next: what does the model actually learn? The answer is
              surprisingly simple: predict the noise. The training objective is
              MSE loss on noise predictions&mdash;the same MSE loss you have
              used since your very first lesson.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: The Training Objective"
            description="What the denoising network actually learns\u2014predict the noise, compare with MSE loss, update weights. The same training loop you already know."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
