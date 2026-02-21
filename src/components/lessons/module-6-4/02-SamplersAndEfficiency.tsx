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
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import { ExternalLink } from 'lucide-react'

/**
 * Samplers and Efficiency
 *
 * Lesson 2 in Module 6.4 (Stable Diffusion). Lesson 16 overall in Series 6.
 * Cognitive load: STRETCH (three genuinely new concepts: DDIM predict-and-leap,
 * ODE perspective on diffusion, higher-order solvers).
 *
 * Teaches why advanced samplers (DDIM, Euler, DPM-Solver) can generate images
 * in 20 steps using the exact same trained model that DDPM needs 1000 steps for:
 * - DDIM's predict-x₀-then-jump mechanism (DEVELOPED)
 * - The ODE perspective on diffusion (INTRODUCED)
 * - Euler method as ODE solver, bridged from gradient descent (INTRODUCED)
 * - DPM-Solver / higher-order methods (INTRODUCED)
 * - Practical sampler comparison and guidance (INTRODUCED)
 *
 * Previous: The Stable Diffusion Pipeline (Module 6.4, Lesson 1 / CONSOLIDATE)
 * Next: Generate with Stable Diffusion (Module 6.4, Lesson 3 / CONSOLIDATE)
 */

export function SamplersAndEfficiencyLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Samplers and Efficiency"
            description="The model predicts noise. The sampler decides what to do with that prediction. Same weights, different walkers&mdash;from 1000 steps to 20."
            category="Stable Diffusion"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Explain why advanced samplers (DDIM, Euler, DPM-Solver) can generate
            images in 20 steps using the exact same trained model that DDPM needs
            1000 steps for, and choose a sampler with understanding rather than
            blind defaults.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="What You Already Know">
            You implemented the DDPM sampling loop in{' '}
            <LessonLink slug="build-a-diffusion-model">Build a Diffusion Model</LessonLink> and felt the 1000-step cost.
            You derived the closed-form formula in{' '}
            <LessonLink slug="the-forward-process">The Forward Process</LessonLink>. You saw samplers as swappable
            components in{' '}
            <LessonLink slug="stable-diffusion-architecture">The Stable Diffusion Pipeline</LessonLink>. This lesson explains
            how those pieces fit together to make generation 50&times; faster.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'DDIM: predict-x₀-then-jump mechanism (DEVELOPED)',
              'ODE perspective: the model defines a trajectory, samplers follow it differently (INTRODUCED)',
              'Euler method as ODE solver, connected to gradient descent (INTRODUCED)',
              'DPM-Solver: higher-order solvers for even fewer steps (INTRODUCED)',
              'Practical sampler comparison: quality vs speed tradeoffs',
              'NOT: deriving DDIM or DPM-Solver from first principles',
              'NOT: score-based models or SDE/ODE duality in full rigor',
              'NOT: implementing samplers from scratch (notebook uses diffusers schedulers)',
              'NOT: ancestral sampling, Karras samplers, or UniPC',
              'NOT: training-based acceleration (distillation, consistency models)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Speed Problem"
            subtitle="You felt it. Now let's solve it."
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In <LessonLink slug="build-a-diffusion-model">Build a Diffusion Model</LessonLink>, you generated 64 MNIST
              digits and waited minutes. In{' '}
              <LessonLink slug="stable-diffusion-architecture">The Stable Diffusion Pipeline</LessonLink>, you saw that each
              denoising step requires a full U-Net forward pass&mdash;or two with
              CFG. At 50 steps, that is 100 forward passes through an 860M
              parameter U-Net.
            </p>
            <p className="text-muted-foreground">
              DDPM requires so many steps because its reverse formula assumes{' '}
              <strong>adjacent timesteps</strong>. Each step takes you from{' '}
              <InlineMath math="x_t" /> to <InlineMath math="x_{t-1}" />, with
              coefficients calibrated for that tiny transition. To go from pure
              noise to clean data, you must walk through every single timestep.
            </p>
            <p className="text-muted-foreground">
              But recall the closed-form formula from{' '}
              <LessonLink slug="the-forward-process">The Forward Process</LessonLink>:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon" />
            </div>
            <p className="text-muted-foreground">
              This formula lets you <strong>destroy</strong> an image to any noise
              level in one step. If destruction can skip steps, why can&rsquo;t
              creation?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Key Question">
            The model predicts <InlineMath math="\varepsilon" /> at every step.
            That prediction contains information about{' '}
            <InlineMath math="x_0" />&mdash;predicting the noise IS predicting
            the clean image. DDPM uses only a tiny fraction of this information
            to take a small step. What if we used more?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 3b: Hook — Same Model, Different Speed */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Same Model, Different Speed"
            subtitle="One model, three samplers, three step counts, comparable results"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The model was trained once. These three generation strategies use
              the same weights, the same noise prediction function. The{' '}
              <strong>only</strong> difference: how those predictions are used to
              take steps.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-3 mt-4">
            <GradientCard title="DDPM" color="amber">
              <ul className="space-y-1">
                <li>&bull; 1000 steps</li>
                <li>&bull; ~2 minutes (MNIST)</li>
                <li>&bull; Stochastic</li>
                <li>&bull; The original algorithm</li>
              </ul>
            </GradientCard>
            <GradientCard title="DDIM" color="blue">
              <ul className="space-y-1">
                <li>&bull; 50 steps</li>
                <li>&bull; ~6 seconds (MNIST)</li>
                <li>&bull; Deterministic</li>
                <li>&bull; 20&times; speedup</li>
              </ul>
            </GradientCard>
            <GradientCard title="DPM-Solver" color="emerald">
              <ul className="space-y-1">
                <li>&bull; 20 steps</li>
                <li>&bull; ~2.5 seconds (MNIST)</li>
                <li>&bull; Current standard</li>
                <li>&bull; 50&times; speedup</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="No Retraining">
            Swapping samplers requires zero retraining. In diffusers, it is
            literally one line:{' '}
            <code className="text-xs">pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)</code>.
            The model&rsquo;s job never changes: given a noisy input and a
            timestep, predict the noise. The sampler&rsquo;s job is to decide
            what to DO with that prediction.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: DDIM — Predict and Leap */}
      {/* 4a: Why DDPM is slow */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why DDPM Is Slow"
            subtitle="The Markov chain constraint"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DDPM&rsquo;s reverse step formula computes{' '}
              <InlineMath math="x_{t-1}" /> from <InlineMath math="x_t" />.
              The coefficients&mdash;<InlineMath math="1/\sqrt{\alpha_t}" />,{' '}
              <InlineMath math="\beta_t / \sqrt{1 - \bar{\alpha}_t}" />&mdash;are
              calibrated for the tiny transition from timestep{' '}
              <InlineMath math="t" /> to <InlineMath math="t-1" />, where{' '}
              <InlineMath math="\beta_t" /> is small (roughly 0.0001 to 0.02).
            </p>
            <p className="text-muted-foreground">
              What happens if you try to skip steps&mdash;applying the DDPM
              formula from <InlineMath math="t = 1000" /> to{' '}
              <InlineMath math="t = 950" />, jumping 50 timesteps at once?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Adjacent-Step Assumption">
            The DDPM formula uses <InlineMath math="\alpha_t" /> and{' '}
            <InlineMath math="\beta_t" />&mdash;single-step quantities. These
            encode the tiny noise change between adjacent timesteps. They do NOT
            encode the cumulative change across 50 or 200 timesteps.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: naive step-skipping */}
      <Row>
        <Row.Content>
          <GradientCard title="Negative Example: Naive Step-Skipping" color="rose">
            <div className="space-y-3 text-sm">
              <p>
                If you apply the DDPM reverse step formula at every 50th timestep
                (1000, 950, 900, ...), the result is <strong>garbage</strong>.
                The coefficients <InlineMath math="\alpha_t" /> and{' '}
                <InlineMath math="\beta_t" /> are calibrated for single-step
                transitions. Applying them across a 50-step gap produces
                catastrophically wrong scaling&mdash;the noise removal term
                undershoots, the signal preservation term overshoots, and errors
                compound at every jump.
              </p>
              <p>
                <strong>The lesson:</strong> faster sampling is NOT just
                &ldquo;do fewer iterations of the same formula.&rdquo; You need a
                different formula&mdash;one designed for arbitrary step sizes.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why It Fails">
            The DDPM formula is like a car with a fixed gear ratio optimized for
            city driving (tiny steps). If you try to drive on the highway (big
            jumps) in the same gear, the engine screams and the car
            breaks. You need a different gear&mdash;a different formula.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 4b: DDIM's insight */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="DDIM: Predict and Leap"
            subtitle="The same model, a different strategy"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DDIM (Song et al., 2020) starts with a simple observation. The
              model predicts <InlineMath math="\varepsilon_\theta(x_t, t)" />.
              But there is a direct relationship between{' '}
              <InlineMath math="\varepsilon" /> and{' '}
              <InlineMath math="x_0" />, from the closed-form formula you
              derived in <LessonLink slug="the-forward-process">The Forward Process</LessonLink>:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon_\theta}{\sqrt{\bar{\alpha}_t}}" />
            </div>
            <p className="text-muted-foreground">
              This <InlineMath math="\hat{x}_0" /> is the model&rsquo;s{' '}
              <strong>current best guess</strong> for the clean data. It is noisy
              and imperfect (especially at high{' '}
              <InlineMath math="t" /> when the input is mostly noise), but it
              contains information about the destination.
            </p>
            <p className="text-muted-foreground">
              DDIM&rsquo;s strategy: at each step, ask <strong>&ldquo;if my
              destination is <InlineMath math="\hat{x}_0" />, where should I be
              at the next timestep?&rdquo;</strong> Then use the closed-form
              formula to jump there.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Predict and Leap">
            DDIM is a navigator constantly recalculating the route. At each step
            it asks: &ldquo;Given my current position, where do I think the
            destination is?&rdquo; (predict{' '}
            <InlineMath math="\hat{x}_0" />). &ldquo;OK, given that destination,
            where should I be at the next checkpoint?&rdquo; (leap using the
            closed-form formula).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 4c: The DDIM step formula */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The DDIM Step Formula"
            subtitle="Side-by-side with DDPM"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The DDIM update (with <InlineMath math="\sigma = 0" /> for
              deterministic sampling) computes{' '}
              <InlineMath math="x_{t_\text{next}}" /> in two sub-steps:
            </p>
          </div>
          <div className="space-y-3 mt-4">
            <PhaseCard number={1} title="Predict the destination" subtitle="Extract x₀ from the noise prediction" color="blue">
              <div className="py-2">
                <BlockMath math="\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon_\theta}{\sqrt{\bar{\alpha}_t}}" />
              </div>
              <p className="text-sm text-muted-foreground">
                Rearrange the closed-form formula to solve for{' '}
                <InlineMath math="x_0" />. You already know this formula&mdash;it
                is the same rearrangement from{' '}
                <LessonLink slug="the-forward-process">The Forward Process</LessonLink>.
              </p>
            </PhaseCard>
            <PhaseCard number={2} title="Leap to the next timestep" subtitle="Use the closed-form formula in the forward direction" color="violet">
              <div className="py-2">
                <BlockMath math="x_{t_\text{next}} = \sqrt{\bar{\alpha}_{t_\text{next}}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t_\text{next}}} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} \cdot \hat{x}_0}{\sqrt{1 - \bar{\alpha}_t}}" />
              </div>
              <p className="text-sm text-muted-foreground">
                The first term places the predicted clean image at the target
                noise level. The second term is the &ldquo;direction pointing
                to <InlineMath math="x_t" />&rdquo;&mdash;it corrects the
                trajectory using the current position.
              </p>
            </PhaseCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Alpha-bar, Not Alpha">
            DDIM uses <InlineMath math="\bar{\alpha}" /> (cumulative signal
            fraction), not <InlineMath math="\alpha" /> (single-step). That is
            the difference between a step-by-step formula and a leap formula.{' '}
            <InlineMath math="\bar{\alpha}" /> encodes the entire schedule up to
            timestep <InlineMath math="t" />&mdash;the &ldquo;signal-to-noise
            dial&rdquo; from <LessonLink slug="the-forward-process">The Forward Process</LessonLink>.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* DDIM step diagram: the two-hop mechanism */}
      <Row>
        <Row.Content>
          <div className="rounded-lg bg-muted/30 p-4">
            <svg viewBox="0 0 520 160" className="w-full max-w-[540px] mx-auto" aria-label="DDIM two-hop step diagram: from x_t, predict backward to x_hat_0, then jump forward to x_t_next">
              <defs>
                <marker id="arrowBlue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#3b82f6" />
                </marker>
                <marker id="arrowVioletStep" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#8b5cf6" />
                </marker>
              </defs>

              {/* Timeline axis */}
              <line x1="40" y1="130" x2="480" y2="130" stroke="#64748b" strokeWidth="1" />
              <text x="30" y="148" fontSize="9" fill="#64748b" textAnchor="middle">t=0</text>
              <text x="260" y="148" fontSize="9" fill="#64748b" textAnchor="middle">t_next</text>
              <text x="480" y="148" fontSize="9" fill="#64748b" textAnchor="middle">t</text>

              {/* Tick marks */}
              <line x1="30" y1="126" x2="30" y2="134" stroke="#64748b" strokeWidth="1" />
              <line x1="260" y1="126" x2="260" y2="134" stroke="#64748b" strokeWidth="1" />
              <line x1="480" y1="126" x2="480" y2="134" stroke="#64748b" strokeWidth="1" />

              {/* x_t node (right, high noise) */}
              <circle cx="480" cy="80" r="8" fill="#f59e0b" fillOpacity="0.3" stroke="#f59e0b" strokeWidth="2" />
              <text x="480" y="65" fontSize="10" fill="#f59e0b" textAnchor="middle" fontWeight="bold">x_t</text>
              <text x="480" y="105" fontSize="8" fill="#64748b" textAnchor="middle">(noisy)</text>

              {/* x_hat_0 node (left, predicted clean) */}
              <circle cx="30" cy="40" r="8" fill="#3b82f6" fillOpacity="0.3" stroke="#3b82f6" strokeWidth="2" />
              <text x="30" y="25" fontSize="10" fill="#3b82f6" textAnchor="middle" fontWeight="bold">x̂₀</text>

              {/* x_t_next node (middle) */}
              <circle cx="260" cy="80" r="8" fill="#8b5cf6" fillOpacity="0.3" stroke="#8b5cf6" strokeWidth="2" />
              <text x="260" y="65" fontSize="10" fill="#8b5cf6" textAnchor="middle" fontWeight="bold">x_t_next</text>
              <text x="260" y="105" fontSize="8" fill="#64748b" textAnchor="middle">(less noisy)</text>

              {/* Hop 1: x_t -> x_hat_0 (predict backward) */}
              <path d="M472,76 Q250,10 38,40" fill="none" stroke="#3b82f6" strokeWidth="2" strokeDasharray="6,3" markerEnd="url(#arrowBlue)" />
              <text x="270" y="22" fontSize="9" fill="#3b82f6" textAnchor="middle" fontStyle="italic">1. Predict destination</text>

              {/* Hop 2: x_hat_0 -> x_t_next (jump forward via closed-form) */}
              <path d="M38,44 Q140,95 252,80" fill="none" stroke="#8b5cf6" strokeWidth="2.5" markerEnd="url(#arrowVioletStep)" />
              <text x="135" y="88" fontSize="9" fill="#8b5cf6" textAnchor="middle" fontStyle="italic">2. Leap via closed-form</text>
            </svg>
            <p className="text-xs text-muted-foreground text-center mt-2">
              DDIM&rsquo;s two-hop mechanism: predict the clean image{' '}
              <InlineMath math="\hat{x}_0" /> (dashed blue), then use the
              closed-form formula to leap to the target timestep (solid violet).
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Side-by-side comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'DDPM Step',
              color: 'amber',
              items: [
                'Uses α_t and β_t (adjacent-step quantities)',
                'Can only step from t to t−1',
                'Adds fresh noise z at every step (stochastic)',
                'Coefficients calibrated for tiny β_t ≈ 0.001',
                '1000 steps required',
              ],
            }}
            right={{
              title: 'DDIM Step',
              color: 'blue',
              items: [
                'Uses ᾱ_t and ᾱ_{t_next} (cumulative quantities)',
                'Can leap from t to ANY t_next',
                'No noise injection (deterministic with σ=0)',
                'Works with any step size via ᾱ values',
                '20–50 steps sufficient',
              ],
            }}
          />
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <InsightBlock title="The Closed-Form Formula: Destruction and Creation">
            The closed-form formula let you <strong>destroy</strong> an image to
            any noise level in one step. DDIM lets you <strong>reverse</strong>{' '}
            that destruction in large steps, using the same mathematical
            tool. The formula works in both directions because{' '}
            <InlineMath math="\bar{\alpha}" /> encodes the complete schedule at
            any timestep.
          </InsightBlock>
        </Row.Content>
      </Row>

      {/* 4d: Concrete example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One DDIM Step: t = 1000 to t = 800"
            subtitle="Traced with specific values"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Using a cosine schedule (the same one from{' '}
              <LessonLink slug="the-forward-process">The Forward Process</LessonLink>):
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="\bar{\alpha}_{1000} \approx 0.0001" />{' '}
                (almost pure noise, nearly all signal destroyed)
              </li>
              <li>
                <InlineMath math="\bar{\alpha}_{800} \approx 0.04" />{' '}
                (still very noisy, but 400&times; more signal than t=1000)
              </li>
            </ul>
            <p className="text-muted-foreground">
              <strong>Step 1&mdash;Predict the destination:</strong> The model
              predicts <InlineMath math="\varepsilon_\theta(x_{1000}, 1000)" />.
              Using the rearranged formula with{' '}
              <InlineMath math="\bar{\alpha}_{1000} = 0.0001" />, we compute{' '}
              <InlineMath math="\hat{x}_0" />. At this high noise level, the
              prediction is crude&mdash;just the rough outline of the image.
            </p>
            <p className="text-muted-foreground">
              <strong>Step 2&mdash;Leap to t=800:</strong> We use{' '}
              <InlineMath math="\bar{\alpha}_{800} = 0.04" /> to compute{' '}
              <InlineMath math="x_{800}" />. This places the predicted{' '}
              <InlineMath math="\hat{x}_0" /> at the noise level corresponding
              to t=800, with the direction correction from our current position.
            </p>
            <p className="text-muted-foreground">
              DDPM would need <strong>200 individual steps</strong> for the same
              transition from t=1000 to t=800. DDIM does it in one.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Not Jump Straight to t=0?">
            The <InlineMath math="\hat{x}_0" /> prediction at t=1000 is
            terrible&mdash;the input is almost pure noise. Each DDIM step
            refines the prediction: at t=800, the model works with a less noisy
            input and produces a better{' '}
            <InlineMath math="\hat{x}_0" /> estimate. The iterative refinement
            is still needed, just with fewer, larger steps.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 4d-2: Second concrete example: t=200 -> t=0 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="One DDIM Step: t = 200 to t = 0"
            subtitle="The same mechanism at low noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now the same mechanism near the end of the trajectory. At t=200,
              the image is mostly clean with some remaining noise:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <InlineMath math="\bar{\alpha}_{200} \approx 0.85" />{' '}
                (85% signal preserved&mdash;the image is mostly visible)
              </li>
              <li>
                <InlineMath math="\bar{\alpha}_{0} = 1.0" />{' '}
                (pure signal, no noise)
              </li>
            </ul>
            <p className="text-muted-foreground">
              <strong>Step 1&mdash;Predict the destination:</strong> The model
              predicts <InlineMath math="\varepsilon_\theta(x_{200}, 200)" />.
              With <InlineMath math="\bar{\alpha}_{200} = 0.85" />, the input
              is mostly clean. The model&rsquo;s{' '}
              <InlineMath math="\hat{x}_0" /> prediction is now{' '}
              <strong>much better</strong>&mdash;fine details, textures, edges
              are all captured.
            </p>
            <p className="text-muted-foreground">
              <strong>Step 2&mdash;Leap to t=0:</strong> We use{' '}
              <InlineMath math="\bar{\alpha}_{0} = 1.0" /> in the DDIM formula.
              The <InlineMath math="\hat{x}_0" /> prediction at t=200 is
              accurate enough that the final leap produces a clean image.
            </p>
            <p className="text-muted-foreground">
              Compare: at t=1000, the prediction was a crude outline (0.01%
              signal). At t=200, it captures fine detail (85% signal). This is
              why the iterative refinement converges&mdash;each step gives the
              model a cleaner input, producing a better prediction, enabling
              a more accurate leap.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Refinement Converges">
            At high t, the prediction is crude but the step moves you to a
            much less noisy region. At low t, the prediction is excellent and
            the step is a small polish. The predict-and-leap mechanism works
            at every noise level&mdash;the formula is the same, only the
            accuracy of <InlineMath math="\hat{x}_0" /> changes.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 4e: Deterministic generation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Deterministic Generation"
            subtitle="Same seed, same image—every time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DDIM with <InlineMath math="\sigma = 0" /> is fully deterministic:
              same starting noise <InlineMath math="z_T" />, same prompt, same
              image, every time, regardless of step count. At 20 steps or 50
              steps, the result converges to the same image.
            </p>
            <p className="text-muted-foreground">
              DDIM also has a tunable <InlineMath math="\sigma" /> parameter that
              interpolates between DDIM and DDPM behavior.{' '}
              <InlineMath math="\sigma = 0" /> gives deterministic DDIM.{' '}
              <InlineMath math="\sigma = 1" /> recovers the original DDPM
              algorithm.
            </p>
            <p className="text-muted-foreground">
              Remember the temperature analogy from{' '}
              <strong>Sampling and Generation</strong>?{' '}
              <InlineMath math="\sigma" /> is the temperature dial. DDPM always
              runs at temperature &gt; 0. DDIM can turn it to zero.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Deterministic Matters">
            Deterministic generation enables:
            <ul className="space-y-1 text-sm mt-2">
              <li>&bull; Reproducibility (debug, compare settings)</li>
              <li>&bull; Interpolation in <InlineMath math="z_T" /> space</li>
              <li>&bull; A/B testing (change one parameter, compare results)</li>
              <li>&bull; Editing workflows (consistent starting point)</li>
            </ul>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Predict and Verify #1 */}
      <Row>
        <Row.Content>
          <GradientCard title="Predict and Verify" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  Your DDIM sampler uses 50 steps for a 1000-timestep model.
                  What is the step size?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    <strong>20 timesteps per step.</strong> The sampler selects
                    50 evenly-spaced timesteps from [1000, 980, 960, ..., 20]
                    and takes one DDIM step between each pair.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  If DDIM predicts <InlineMath math="\hat{x}_0" /> at every
                  step, why not just use the prediction from the very first step
                  (t=1000)?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    The prediction at t=1000 is terrible because the input is
                    almost pure noise. The model has very little signal to work
                    with. Each step refines the prediction: at t=800, the model
                    works with less noisy input and produces a better{' '}
                    <InlineMath math="\hat{x}_0" /> estimate. Iterative
                    refinement is still needed&mdash;just with fewer, larger
                    steps.
                  </p>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 6: The ODE Perspective */}
      {/* 6a: Bridge from gradient descent */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The ODE Perspective"
            subtitle="You have been using Euler's method since Series 1"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              There is a deeper reason why DDIM works. To see it, we need to
              shift perspective&mdash;and the bridge is something you already
              know by heart.
            </p>
            <p className="text-muted-foreground">
              Gradient descent updates parameters like this:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\theta_{\text{new}} = \theta_{\text{old}} - \text{lr} \cdot \nabla L(\theta)" />
            </div>
            <p className="text-muted-foreground">
              Compute the direction (gradient), scale by a step size (learning
              rate), update. You have done this thousands of times since Series
              1. Here is the thing: <strong>that is Euler&rsquo;s method</strong>.
            </p>
            <p className="text-muted-foreground">
              Euler&rsquo;s method solves an ordinary differential equation
              (ODE) by stepping:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="x_{t+h} = x_t + h \cdot f(x_t, t)" />
            </div>
            <p className="text-muted-foreground">
              Compute the direction (<InlineMath math="f(x_t, t)" />), scale by
              a step size (<InlineMath math="h" />), update. Same structure.
              Same algorithm. Gradient descent IS Euler&rsquo;s method applied
              to the ODE of steepest descent on the loss surface.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="ODE in Plain Terms">
            An ODE says: &ldquo;at every point, the direction to move
            is...&rdquo; A solver (like Euler) says: &ldquo;OK, I will take a
            step of size <InlineMath math="h" /> in that direction, then check
            again.&rdquo; Gradient descent is exactly this, on the loss surface.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 6b: Diffusion as an ODE */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now apply the same lens to diffusion. At every point{' '}
              <InlineMath math="(x_t, t)" /> in noise-data space, the
              model&rsquo;s noise prediction defines a direction: &ldquo;to get
              closer to clean data, move this way.&rdquo; This direction at
              every point forms a <strong>vector field</strong>&mdash;a smooth
              landscape of arrows pointing from noise toward data.
            </p>
            <p className="text-muted-foreground">
              The collection of all these directions forms an ODE:{' '}
              <InlineMath math="dx/dt = f(x, t)" />, where{' '}
              <InlineMath math="f" /> is derived from the model&rsquo;s noise
              prediction. Following this ODE from <InlineMath math="t = T" />{' '}
              (noise) to <InlineMath math="t = 0" /> (data) traces a{' '}
              <strong>trajectory</strong> through the space. This trajectory IS
              the generation process.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Landscape, Different Lens">
            The Markov chain view says &ldquo;take small random steps.&rdquo;
            The ODE view says &ldquo;follow a smooth trajectory.&rdquo; Same
            landscape, same destination, different description of the path. The
            ODE view reveals why step-skipping works: smooth trajectories can be
            followed with large steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 6c: Why the ODE view enables efficiency */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              If the trajectory is smooth&mdash;and it is, because the noise
              schedule is smooth and the model is a smooth neural
              network&mdash;then you do not need to check the direction at every
              tiny step. You can take larger steps and still follow the
              trajectory accurately.
            </p>
            <p className="text-muted-foreground">
              Euler&rsquo;s method on the diffusion ODE: take a step of size{' '}
              <InlineMath math="h" /> in the direction the model gives you.
              Check again. Repeat. Fewer checks = fewer model evaluations =
              faster generation.
            </p>
            <p className="text-muted-foreground">
              <strong>DDIM IS approximately Euler&rsquo;s method on the
              diffusion ODE.</strong> It was derived independently by Song et
              al. (2020), but it turns out to be equivalent to a first-order ODE
              solver on what is called the &ldquo;probability flow ODE.&rdquo;
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Probability Flow ODE">
            Song et al. (2021) showed that the DDPM forward process can be
            described by a stochastic differential equation (SDE), and there is
            a corresponding ODE that traces the same distributions
            deterministically. This is the trajectory DDIM follows. If you see
            &ldquo;probability flow ODE&rdquo; in papers, this is what it
            means.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Trajectory visualization: DDPM vs DDIM vs DPM-Solver */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine the path from noise to data as a trajectory through
              space. Three samplers, three walking styles:
            </p>
            <div className="rounded-lg bg-muted/30 p-4">
              <svg viewBox="0 0 480 300" className="w-full max-w-[520px] mx-auto" aria-label="Trajectory diagram: three paths from noise to data. DDPM takes many jittery tiny steps. DDIM takes fewer smooth steps. DPM-Solver takes the fewest steps with curved arcs.">
                <defs>
                  <marker id="arrowAmberTraj" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                    <path d="M0,0 L5,2.5 L0,5 Z" fill="#f59e0b" />
                  </marker>
                  <marker id="arrowBlueTraj" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                    <path d="M0,0 L5,2.5 L0,5 Z" fill="#3b82f6" />
                  </marker>
                  <marker id="arrowEmeraldTraj" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                    <path d="M0,0 L5,2.5 L0,5 Z" fill="#10b981" />
                  </marker>
                </defs>

                {/* Axis labels */}
                <text x="440" y="285" fontSize="10" fill="#64748b" textAnchor="middle">clean data (t=0)</text>
                <text x="60" y="28" fontSize="10" fill="#64748b" textAnchor="middle">noise (t=T)</text>

                {/* Start and end markers */}
                <circle cx="60" cy="40" r="6" fill="#94a3b8" fillOpacity="0.4" stroke="#94a3b8" strokeWidth="1.5" />
                <circle cx="440" cy="270" r="6" fill="#22c55e" fillOpacity="0.4" stroke="#22c55e" strokeWidth="1.5" />
                <text x="440" y="264" fontSize="8" fill="#22c55e" textAnchor="middle">x₀</text>
                <text x="60" y="53" fontSize="8" fill="#94a3b8" textAnchor="middle">x_T</text>

                {/* DDPM path: many tiny steps with jitter (amber) */}
                <polyline
                  points="60,40 68,47 73,55 80,60 84,68 92,72 95,80 102,86 107,90 110,97 118,102 120,108 128,113 132,118 135,125 142,128 146,135 150,138 158,143 160,148 168,152 170,158 176,160 182,165 186,170 190,172 196,178 200,180 206,185 210,188 216,192 220,195 226,198 230,203 236,205 240,210 248,213 252,218 258,220 262,225 268,228 274,232 278,236 284,238 290,242 296,245 302,248 310,252 318,255 326,258 336,261 348,264 362,266 380,268 400,269 420,270 440,270"
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="1.5"
                  strokeOpacity="0.7"
                />
                {/* DDPM dots (many) */}
                <circle cx="80" cy="60" r="2" fill="#f59e0b" />
                <circle cx="95" cy="80" r="2" fill="#f59e0b" />
                <circle cx="110" cy="97" r="2" fill="#f59e0b" />
                <circle cx="128" cy="113" r="2" fill="#f59e0b" />
                <circle cx="142" cy="128" r="2" fill="#f59e0b" />
                <circle cx="158" cy="143" r="2" fill="#f59e0b" />
                <circle cx="170" cy="158" r="2" fill="#f59e0b" />
                <circle cx="186" cy="170" r="2" fill="#f59e0b" />
                <circle cx="200" cy="180" r="2" fill="#f59e0b" />
                <circle cx="216" cy="192" r="2" fill="#f59e0b" />
                <circle cx="230" cy="203" r="2" fill="#f59e0b" />
                <circle cx="248" cy="213" r="2" fill="#f59e0b" />
                <circle cx="262" cy="225" r="2" fill="#f59e0b" />
                <circle cx="278" cy="236" r="2" fill="#f59e0b" />
                <circle cx="296" cy="245" r="2" fill="#f59e0b" />
                <circle cx="318" cy="255" r="2" fill="#f59e0b" />
                <circle cx="348" cy="264" r="2" fill="#f59e0b" />
                <circle cx="380" cy="268" r="2" fill="#f59e0b" />
                <circle cx="420" cy="270" r="2" fill="#f59e0b" />
                {/* Jitter lines off the main path */}
                <line x1="95" y1="80" x2="99" y2="75" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />
                <line x1="128" y1="113" x2="124" y2="117" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />
                <line x1="170" y1="158" x2="175" y2="154" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />
                <line x1="216" y1="192" x2="212" y2="196" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />
                <line x1="262" y1="225" x2="266" y2="221" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />
                <line x1="318" y1="255" x2="314" y2="259" stroke="#f59e0b" strokeWidth="0.8" strokeOpacity="0.5" />

                {/* DDIM path: smooth curve, fewer points (blue) */}
                <path
                  d="M60,40 Q120,80 160,130 Q200,180 260,210 Q320,240 380,258 Q420,266 440,270"
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                />
                {/* DDIM dots (fewer, evenly spaced) */}
                <circle cx="60" cy="40" r="3.5" fill="#3b82f6" />
                <circle cx="130" cy="100" r="3.5" fill="#3b82f6" />
                <circle cx="195" cy="165" r="3.5" fill="#3b82f6" />
                <circle cx="270" cy="215" r="3.5" fill="#3b82f6" />
                <circle cx="355" cy="250" r="3.5" fill="#3b82f6" />
                <circle cx="440" cy="270" r="3.5" fill="#3b82f6" />

                {/* DPM-Solver path: fewest points, curved arcs (emerald) */}
                <path
                  d="M60,40 C130,60 170,140 220,190 C270,240 370,262 440,270"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.5"
                />
                {/* DPM-Solver dots (fewest) */}
                <circle cx="60" cy="40" r="4" fill="#10b981" />
                <circle cx="220" cy="190" r="4" fill="#10b981" />
                <circle cx="440" cy="270" r="4" fill="#10b981" />

                {/* Legend */}
                <line x1="30" y1="290" x2="50" y2="290" stroke="#f59e0b" strokeWidth="1.5" />
                <circle cx="40" cy="290" r="2" fill="#f59e0b" />
                <text x="55" y="293" fontSize="9" fill="#94a3b8">DDPM (many tiny jittery steps)</text>

                <line x1="225" y1="290" x2="245" y2="290" stroke="#3b82f6" strokeWidth="2" />
                <circle cx="235" cy="290" r="3" fill="#3b82f6" />
                <text x="250" y="293" fontSize="9" fill="#94a3b8">DDIM (fewer smooth steps)</text>

                <line x1="370" y1="290" x2="390" y2="290" stroke="#10b981" strokeWidth="2.5" />
                <circle cx="380" cy="290" r="3.5" fill="#10b981" />
                <text x="395" y="293" fontSize="9" fill="#94a3b8">DPM-Solver (fewest)</text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                All three start at the same noise and arrive at the same
                destination. The model defines where to go. The sampler defines
                how to get there. DDPM takes every back road (drunkard&rsquo;s walk).
                DDIM follows the highway. DPM-Solver uses GPS.
              </p>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Higher-Order Solvers */}
      {/* 7a: The problem with Euler */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Higher-Order Solvers: DPM-Solver"
            subtitle="Reading the road ahead"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Euler&rsquo;s method assumes the direction does not change between
              steps. It takes a straight-line step. If the trajectory curves,
              Euler overshoots. At low step counts (&lt;20 steps), quality drops
              because the trajectory <strong>does</strong> curve and Euler
              cannot account for it.
            </p>
            <p className="text-muted-foreground">
              Analogy: driving on a curvy road. If you can only check the road
              direction once per second (Euler), you miss turns at high speed.
              If you check multiple times per second (higher-order), you can
              anticipate curves and stay on the road.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Solver Order">
            <ul className="space-y-1 text-sm">
              <li><strong>First-order</strong> (Euler, DDIM): one model evaluation per step. Assumes straight-line trajectory between steps.</li>
              <li><strong>Second-order</strong>: two evaluations per step. Estimates how the direction is changing (curvature).</li>
              <li><strong>Third-order</strong>: three evaluations. Estimates how the curvature itself changes.</li>
            </ul>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 7b: DPM-Solver */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              DPM-Solver (Lu et al., 2022) evaluates the model at multiple
              nearby timesteps to estimate how the direction is changing. More
              evaluations per step, but far fewer steps total. The net result:
              fewer total model evaluations for the same quality.
            </p>
          </div>
          <div className="space-y-3 mt-4">
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                DPM-Solver-1 (first-order)
              </p>
              <p className="text-sm text-muted-foreground">
                Essentially Euler. One evaluation per step. Good at 50+ steps.
              </p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                DPM-Solver-2 (second-order)
              </p>
              <p className="text-sm text-muted-foreground">
                Two evaluations per step, accounts for trajectory curvature.
                Good at 20&ndash;30 steps. Matches DDIM at 50 steps.
              </p>
            </div>
            <div className="rounded-lg bg-violet-500/5 border border-violet-500/20 p-3 space-y-1">
              <p className="text-sm font-medium text-foreground">
                DPM-Solver++ (third-order, adaptive)
              </p>
              <p className="text-sm text-muted-foreground">
                Three evaluations at key points. Accounts for how the curvature
                itself changes. Excellent results at 15&ndash;20 steps. The
                current standard in most SD interfaces.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Net Evaluations">
            DPM-Solver-2 at 20 steps uses ~40 model evaluations (2 per step).
            DDIM at 50 steps uses 50 evaluations. Euler at 50 uses 50. DDPM at
            1000 uses 1000. DPM-Solver wins on total evaluations AND quality
            because each evaluation is used more efficiently.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Practical Guidance */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Choosing a Sampler"
            subtitle="Which sampler, how many steps, and when"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now you understand the mechanism behind each sampler. Here is the
              practical guidance:
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-2 mt-4">
            <GradientCard title="DPM-Solver++ (Default)" color="emerald">
              <ul className="space-y-1">
                <li>&bull; <strong>Steps:</strong> 20&ndash;30</li>
                <li>&bull; Best balance of speed and quality</li>
                <li>&bull; Current standard in most interfaces</li>
                <li>&bull; Use this unless you have a specific reason not to</li>
              </ul>
            </GradientCard>
            <GradientCard title="DDIM" color="blue">
              <ul className="space-y-1">
                <li>&bull; <strong>Steps:</strong> 50</li>
                <li>&bull; Deterministic (same seed = same image)</li>
                <li>&bull; Best for reproducibility, interpolation, editing</li>
                <li>&bull; Slightly more steps than DPM-Solver for same quality</li>
              </ul>
            </GradientCard>
            <GradientCard title="Euler" color="cyan">
              <ul className="space-y-1">
                <li>&bull; <strong>Steps:</strong> 30&ndash;50</li>
                <li>&bull; Simple, well-understood, easy to debug</li>
                <li>&bull; Good baseline for experimentation</li>
                <li>&bull; First-order (same as DDIM, different derivation)</li>
              </ul>
            </GradientCard>
            <GradientCard title="DDPM" color="amber">
              <ul className="space-y-1">
                <li>&bull; <strong>Steps:</strong> 1000</li>
                <li>&bull; Maximum quality, minimum speed</li>
                <li>&bull; Stochastic (diverse outputs)</li>
                <li>&bull; Only for research or when speed does not matter</li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <WarningBlock title="More Steps Does NOT Always Help">
            Advanced samplers have a sweet spot. DPM-Solver at 200 steps is not
            meaningfully better than at 25&mdash;you are paying more compute
            for diminishing returns. Use the sampler&rsquo;s recommended
            range. DDPM is the exception: it was designed for 1000 steps and
            actually needs them.
          </WarningBlock>
        </Row.Content>
      </Row>

      {/* Stochastic vs Deterministic */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Deterministic Samplers',
              color: 'blue',
              items: [
                'DDIM (σ=0), Euler',
                'Same seed = same image, always',
                'Reproducible: great for A/B testing',
                'Interpolation in z_T space works smoothly',
                'Slightly less diverse outputs',
              ],
            }}
            right={{
              title: 'Stochastic Samplers',
              color: 'violet',
              items: [
                'DDPM, DDIM (σ>0), some Euler variants',
                'Same seed ≠ same image (noise injected per step)',
                'More diverse outputs per seed',
                'Can sometimes produce higher quality',
                'Not reproducible without controlling all random state',
              ],
            }}
          />
          <p className="text-sm text-muted-foreground mt-4 italic">
            This is the same tradeoff as temperature in language models, applied
            to image generation. You saw this connection in{' '}
            <strong>Sampling and Generation</strong>.
          </p>
        </Row.Content>
      </Row>

      {/* Section 9: Transfer checks */}
      <Row>
        <Row.Content>
          <GradientCard title="Transfer Questions" color="cyan">
            <div className="space-y-4 text-sm">
              <div>
                <p className="font-medium">
                  Your colleague claims they found a way to make DPM-Solver even
                  faster by increasing the solver order to 10 (evaluating the
                  model 10 times per step). Why might this NOT work as expected?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    Each evaluation costs a full U-Net forward pass. At order
                    10, each step costs 10 forward passes. You need fewer
                    steps, but the total forward passes may not decrease. Also,
                    very high-order methods can be numerically unstable. In
                    practice, order 2&ndash;3 hits the sweet spot where fewer
                    total evaluations produce good results.
                  </p>
                </details>
              </div>
              <div>
                <p className="font-medium">
                  If you retrained the same U-Net with a different noise
                  schedule (say, sigmoid instead of cosine), would your DDIM
                  sampler still work?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2">
                    Yes, but you would need to recompute the{' '}
                    <InlineMath math="\bar{\alpha}" /> values. DDIM uses{' '}
                    <InlineMath math="\bar{\alpha}_t" /> at each scheduled
                    timestep. The formula is the same; only the numerical values
                    of <InlineMath math="\bar{\alpha}" /> change. The sampler
                    works with any noise schedule because it only needs the{' '}
                    <InlineMath math="\bar{\alpha}" /> values, not the specific
                    schedule shape.
                  </p>
                </details>
              </div>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: Practice — Notebook link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Compare Samplers"
            subtitle="Hands-on notebook exercises"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Swap samplers on the same model, compare outputs, and explore
                the quality-vs-speed tradeoff firsthand. Four exercises
                progressing from guided to independent.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/6-4-2-samplers-and-efficiency.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>
                  <strong>Exercise 1 (Guided):</strong> Same model, three
                  samplers&mdash;DDPM at 1000 steps, DDIM at 50, DPM-Solver at
                  20. Compare quality and timing.
                </li>
                <li>
                  <strong>Exercise 2 (Guided):</strong> DDIM
                  determinism&mdash;generate 3 images with the same seed and
                  verify they are identical. Compare to DDPM with the same seed.
                </li>
                <li>
                  <strong>Exercise 3 (Supported):</strong> Step count
                  exploration&mdash;generate at 5, 10, 20, 50, 100, 200 steps
                  with DPM-Solver. Find the sweet spot.
                </li>
                <li>
                  <strong>Exercise 4 (Independent):</strong> Inspect DDIM
                  intermediates at 10 steps&mdash;VAE-decode each intermediate
                  latent and observe the coarse-to-fine progression. Compare to
                  DDPM at 10 steps.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Prediction Before Running">
            Each exercise includes a prediction prompt. Before running the
            code, write down what you expect. This is the deliberate practice
            pattern from the course: predict, run, compare, revise your mental
            model.
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
                  'The model predicts noise. The sampler decides what to do with that prediction.',
                description:
                  'DDPM takes a tiny step to the adjacent timestep and adds fresh noise (1000 steps, stochastic). DDIM predicts x₀ from the noise prediction, then leaps to a distant timestep using the closed-form formula (50 steps, deterministic). DPM-Solver follows the trajectory using multiple evaluations to read the road ahead (20 steps, current standard).',
              },
              {
                headline:
                  'No retraining. Same weights. Different walkers.',
                description:
                  'Every sampler uses the exact same noise prediction model. Swapping samplers is an inference-time decision. The model\'s job (predict noise) never changes. The sampler\'s job (decide how to step) is the only variable.',
              },
              {
                headline:
                  'The ODE perspective reveals why step-skipping works.',
                description:
                  'The model\'s predictions at every point define a smooth trajectory from noise to data. Different samplers follow this trajectory with different step sizes and strategies. Gradient descent is Euler\'s method—you have been solving ODEs since Series 1.',
              },
              {
                headline:
                  'Use DPM-Solver++ at 20–30 steps as your default.',
                description:
                  'DDIM at 50 for reproducibility. Euler at 30–50 for debugging. DDPM at 1000 only when speed does not matter. More steps beyond the sweet spot yields diminishing returns.',
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
              The model defines where to go. The sampler defines how to get
              there.
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
                title: 'Denoising Diffusion Implicit Models',
                authors: 'Song, Meng & Ermon, 2020',
                url: 'https://arxiv.org/abs/2010.02502',
                note: 'The DDIM paper. Section 4 introduces the non-Markovian forward process and the deterministic sampling formula.',
              },
              {
                title: 'DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps',
                authors: 'Lu, Zhou, Bao, Chen, Li & Zhu, 2022',
                url: 'https://arxiv.org/abs/2206.00927',
                note: 'The DPM-Solver paper. Sections 3-4 explain the higher-order solver formulation. DPM-Solver++ extends it.',
              },
              {
                title: 'Score-Based Generative Modeling through Stochastic Differential Equations',
                authors: 'Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, 2021',
                url: 'https://arxiv.org/abs/2011.13456',
                note: 'The SDE/ODE unification paper. Section 3 introduces the probability flow ODE. Background reading for the ODE perspective.',
              },
              {
                title: 'Denoising Diffusion Probabilistic Models',
                authors: 'Ho, Jain & Abbeel, 2020',
                url: 'https://arxiv.org/abs/2006.11239',
                note: 'The DDPM paper. The baseline sampler that this lesson shows how to surpass.',
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
              You now understand every piece of the Stable Diffusion
              pipeline AND how to choose a sampler. The next lesson puts it all
              together: you will use the diffusers library to generate images,
              and you will know what every parameter means because you built
              each concept from scratch.
            </p>
            <p className="text-muted-foreground">
              Every API parameter maps to a concept you learned.{' '}
              <code>guidance_scale</code> is CFG from{' '}
              <strong>Text Conditioning &amp; Guidance</strong>.{' '}
              <code>num_inference_steps</code> is the sampler step count from
              this lesson. <code>scheduler</code> is your sampler choice.
              You will not be following a tutorial&mdash;you will be driving a
              machine you built.
            </p>
          </div>
        </Row.Content>
      </Row>

      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Up Next: Generate with Stable Diffusion"
            description="Use the diffusers library to generate images&mdash;and know what every parameter means because you built each concept from scratch."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
