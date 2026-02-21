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
  ModuleCompleteBlock,
  NextStepBlock,
  ReferencesBlock,
  LessonLink,
} from '@/components/lessons'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-2-2-flow-matching.ipynb'

/**
 * Flow Matching
 *
 * Lesson 2 in Module 7.2 (The Score-Based Perspective). Lesson 5 of 11 in Series 7.
 * Cognitive load: BUILD (follows STRETCH lesson on score functions and SDEs).
 *
 * Core concepts:
 * 1. Conditional flow matching (straight-line interpolation + velocity prediction) — NEW
 * 2. Rectified flow (iterative trajectory straightening) — NEW
 * Additionally: velocity prediction parameterization (natural consequence of CFM)
 *
 * Builds on: probability flow ODE from 7.2.1, Euler's method from 6.4.2,
 * DDPM training from 6.2.3, score-noise equivalence from 7.2.1.
 *
 * Previous: Score Functions & SDEs (Module 7.2, Lesson 1 / STRETCH)
 * Next: Module 7.3 (Fast Generation)
 */

export function FlowMatchingLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Flow Matching"
            description="What if we designed the generation trajectory to be straight? A simpler training objective, fewer sampling steps, and the reason SD3 and Flux abandoned noise prediction."
            category="The Score-Based Perspective"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand flow matching as a generative framework that defines
            straight-line trajectories between noise and data, replacing the
            curved diffusion paths with simpler, faster-to-follow paths via a
            velocity prediction training objective. See why this is the training
            objective behind SD3, Flux, and most modern diffusion models.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Payoff">
            The previous lesson was the most theoretical in Series 7. This
            lesson is the payoff: the SDE/ODE framework you just learned makes
            flow matching feel like an obvious simplification.
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'Why diffusion trajectories curve and why that is a problem',
              'Conditional flow matching: straight-line interpolation, velocity prediction, training objective',
              'Velocity prediction as a parameterization alongside noise and score prediction',
              'Rectified flow as trajectory straightening (intuition level)',
              'Why flow matching enables fewer sampling steps',
              'NOT: Optimal transport formulations (Lipman et al. 2023)',
              'NOT: The DiT architecture (Module 7.4)',
              'NOT: Consistency models (Module 7.3)',
              'NOT: Training from scratch beyond toy 2D examples',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Three concepts you will need in the next twenty minutes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>The probability flow ODE.</strong> In{' '}
              <strong>Score Functions &amp; SDEs</strong>, you learned that
              generation follows a deterministic trajectory from noise to
              data. The probability flow ODE defines this trajectory using the
              score function. DDIM was approximately solving this ODE.
            </p>
            <p className="text-muted-foreground">
              <strong>Euler&rsquo;s method.</strong> In{' '}
              <strong>Samplers and Efficiency</strong>, you learned the simplest
              ODE solver: compute the direction at your current point, take a
              step, repeat. Accuracy depends on step size and how much the
              trajectory curves.
            </p>
            <p className="text-muted-foreground">
              <strong>DDPM training.</strong> In{' '}
              <strong>Learning to Denoise</strong>, you learned the training
              loop: sample data, add noise at a random timestep, network
              predicts the noise, MSE loss. The same training structure will
              reappear with a different target.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Bridge">
            The previous lesson ended with a question: these trajectories{' '}
            <strong>curve</strong> through space. What if we could straighten
            them? Let&rsquo;s find out what happens.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Curved vs Straight"
            subtitle="A before/after that changes everything"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Look at the probability flow ODE trajectories from the previous
              lesson. Multiple paths from noise points to data points, each
              curving through space. The paths bend because the score field
              changes direction at different noise levels—at high noise the
              field is simple (nearly Gaussian), at low noise it is complex
              (multi-modal). The trajectory must follow this changing field,
              creating curves.
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Diffusion ODE Trajectories" color="amber">
                <div className="space-y-3 text-sm">
                  <pre className="font-mono text-xs leading-relaxed text-amber-200/80 whitespace-pre">
{`  noise ε                    noise ε
     ·                          ·
      ·                        ·
       ·  ←curved             · ←curved
        ·                    ·
         ·                 ·
          · · ·         · ·
                · · · ·
               data x₀          data x₀`}
                  </pre>
                  <p className="text-muted-foreground">
                    Paths bend through space. The score field changes direction at
                    each noise level, forcing the trajectory to curve. ODE solvers
                    must take small steps to follow the bends.
                  </p>
                  <p className="text-muted-foreground text-xs">
                    DDPM: 1000 steps | DDIM: 50 | DPM-Solver: 15-20
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Flow Matching Trajectories" color="emerald">
                <div className="space-y-3 text-sm">
                  <pre className="font-mono text-xs leading-relaxed text-emerald-200/80 whitespace-pre">
{`  noise ε                    noise ε
    |                            |
    |                            |
    |  ←straight                 |  ←straight
    |                            |
    |                            |
    |                            |
    |                            |
  data x₀                     data x₀`}
                  </pre>
                  <p className="text-muted-foreground">
                    Paths are straight lines from data to noise. Velocity is
                    constant along each path. Euler&rsquo;s method follows a
                    straight line with trivial accuracy—even one step is exact.
                  </p>
                  <p className="text-muted-foreground text-xs">
                    SD3/Flux: 20-30 steps for high quality
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-xs text-muted-foreground">
              You will see this difference plotted precisely in Exercises 1 and 2
              of the notebook.
            </p>
            <p className="text-muted-foreground">
              This is not wishful thinking. This is what flow matching does.
              And the math is embarrassingly simple.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Question">
            Every improvement to sampling speed—DDIM, DPM-Solver++—fights the{' '}
            <strong>curvature</strong> with smarter solvers. Flow matching asks:
            what if we eliminated the curvature at the source?
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — Why Trajectories Curve */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Trajectories Curve"
            subtitle="The problem that flow matching solves"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know Euler&rsquo;s method from{' '}
              <strong>Samplers and Efficiency</strong>: compute the direction at
              your current point, take a step. But what if the trajectory curves
              right after you step? Your step overshoots—you expected the
              trajectory to keep going straight, but it turned. The more the
              trajectory curves, the smaller your steps must be to stay on
              track.
            </p>
            <p className="text-muted-foreground">
              This is <strong>why</strong> diffusion needs many steps. Not
              because generation is inherently complex, but because the
              probability flow ODE trajectory <strong>curves</strong>, and ODE
              solvers need small steps to follow curves accurately.
              Higher-order solvers like DPM-Solver++ handle curvature better,
              but they are fighting the symptom, not the cause.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Symptom vs Cause">
            Better ODE solvers = treating the symptom (curvature). Flow
            matching = treating the cause (the trajectory itself).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Why the curves exist */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Why does the probability flow ODE trajectory curve? Because the
              score function—the vector field that guides the trajectory—changes
              dramatically as noise level changes. At high noise (
              <InlineMath math="t" /> near 1), the score field is smooth and
              nearly Gaussian. At low noise (<InlineMath math="t" /> near 0),
              the score field is complex and multi-modal. The trajectory must
              navigate this changing landscape, and the change in direction
              creates curvature.
            </p>
            <GradientCard title="GPS Recalculating vs Straight Highway" color="blue">
              <div className="space-y-2 text-sm">
                <p>
                  On a winding road (curved diffusion trajectory), your GPS must
                  constantly recalculate as the road turns—miss a turn and you
                  are off course. On a straight highway (flow matching
                  trajectory), you point the car in the right direction and
                  drive—no recalculation needed, no chance of missing a turn.
                </p>
                <p className="text-muted-foreground">
                  Fewer steps (GPS recalculations) needed because the path has
                  no curves to navigate.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Floor on Step Count">
            The curvature of the diffusion ODE trajectory sets a floor on how
            few steps <strong>any</strong> solver can take while staying
            accurate. DPM-Solver needs 15-20 steps, not 1, because the path
            bends.
          </InsightBlock>
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
                  If the probability flow ODE trajectory were perfectly
                  straight, how many Euler steps would you need to follow it
                  exactly?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    One. Euler&rsquo;s method extrapolates linearly from the
                    current point. If the trajectory IS a straight line, the
                    extrapolation is exact regardless of step size.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Looking at the score field panels from the previous lesson—the
                  field changes from simple (high noise) to complex (low noise).
                  Where does the probability flow ODE trajectory curve the most?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    The trajectory curves most where the score field changes the
                    fastest—in the transition from high noise to low noise. This
                    is the middle region where the field structure is emerging.
                    At the extremes (pure noise or nearly clean data), the field
                    is relatively stable.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — Conditional Flow Matching */}
      {/* Part A: The key idea */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Conditional Flow Matching"
            subtitle="Choose the path first, then learn to follow it"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Diffusion starts with a training objective (predict noise) and
              discovers that the resulting trajectories curve. Flow matching
              flips the order:{' '}
              <strong>choose the trajectory first</strong> (make it straight),
              then design a training objective to learn it.
            </p>
            <p className="text-muted-foreground">
              This is the conceptual shift. In DDPM, the forward process (add
              noise) determines the trajectory shape. In flow matching, you
              define the trajectory shape directly and build a training
              objective around it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Flip">
            DDPM: design training objective &rarr; discover trajectory shape.
            Flow matching: design trajectory shape &rarr; derive training
            objective.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Part B: The simplest possible interpolation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Simplest Possible Interpolation"
            subtitle="Linear coefficients instead of nonlinear schedules"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The DDPM forward process uses a complex interpolation:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon" />
            </div>
            <p className="text-muted-foreground">
              Nonlinear coefficients. Variance-preserving (the squared
              coefficients sum to 1). The path from{' '}
              <InlineMath math="x_0" /> to noise{' '}
              <strong>curves</strong> because of the nonlinear coefficients.
            </p>
            <p className="text-muted-foreground">
              Flow matching replaces this with the simplest possible
              interpolation:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="x_t = (1-t)\, x_0 + t\, \epsilon" />
            </div>
            <p className="text-muted-foreground">
              Linear coefficients. At{' '}
              <InlineMath math="t=0" />: <InlineMath math="x_t = x_0" /> (pure
              data). At <InlineMath math="t=1" />:{' '}
              <InlineMath math="x_t = \epsilon" /> (pure noise). The path from{' '}
              <InlineMath math="x_0" /> to <InlineMath math="\epsilon" /> is a{' '}
              <strong>straight line</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Idea, Simpler Coefficients">
            Both formulas interpolate between data and noise. DDPM uses{' '}
            <InlineMath math="\sqrt{\bar\alpha_t}" /> and{' '}
            <InlineMath math="\sqrt{1-\bar\alpha_t}" /> (nonlinear). Flow
            matching uses <InlineMath math="1-t" /> and{' '}
            <InlineMath math="t" /> (linear). That is the entire change.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Comparison table */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'DDPM Interpolation',
              color: 'amber',
              items: [
                'x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon',
                'Nonlinear coefficients (sqrt of cumulative product)',
                'Curved path through data space',
                'At midpoint: NOT a 50/50 mix (depends on schedule)',
              ],
            }}
            right={{
              title: 'Flow Matching Interpolation',
              color: 'emerald',
              items: [
                'x_t = (1-t) * x_0 + t * epsilon',
                'Linear coefficients (1-t, t)',
                'Straight line through data space',
                'At midpoint (t=0.5): exactly 50/50 mix',
              ],
            }}
          />
          <div className="mt-4">
            <GradientCard title="The Simplicity IS the Advantage" color="violet">
              <p className="text-sm">
                The DDPM noise schedule was carefully designed to make certain
                mathematical properties hold (variance-preserving, tractable
                posteriors). Flow matching sidesteps all of that. The linear
                interpolation is simple and the resulting training objective is
                simple. No{' '}
                <InlineMath math="\bar\alpha" />, no cumulative products, no
                special schedule.
              </p>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Part C: The velocity field */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Velocity Field"
            subtitle="What is the direction along a straight line?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The path is a straight line from{' '}
              <InlineMath math="x_0" /> to <InlineMath math="\epsilon" />.
              What is the velocity along this path? Take the derivative:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="v = \frac{dx_t}{dt} = \frac{d}{dt}\left[(1-t)\, x_0 + t\, \epsilon\right] = \epsilon - x_0" />
            </div>
            <p className="text-muted-foreground">
              The velocity is <strong>constant</strong>. It does not depend on{' '}
              <InlineMath math="t" />. At every point along the straight-line
              path, the velocity is the same:{' '}
              <InlineMath math="\epsilon - x_0" /> (the direction from data to
              noise).
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Constant Velocity">
            The velocity is constant because the path is a straight line. A
            straight line has no curvature, so the tangent direction never
            changes. Same reason constant velocity means zero acceleration in
            physics.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Part D: The training objective */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Objective"
            subtitle="Predict the velocity, not the noise"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Train a network{' '}
              <InlineMath math="v_\theta(x_t, t)" /> to predict the velocity{' '}
              <InlineMath math="\epsilon - x_0" />. The flow matching training
              step:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                Sample a data point <InlineMath math="x_0" />
              </li>
              <li>
                Sample random noise{' '}
                <InlineMath math="\epsilon \sim \mathcal{N}(0, I)" />
              </li>
              <li>
                Sample a random time{' '}
                <InlineMath math="t \sim \text{Uniform}(0, 1)" />
              </li>
              <li>
                Compute{' '}
                <InlineMath math="x_t = (1-t)\, x_0 + t\, \epsilon" />
              </li>
              <li>
                Target: <InlineMath math="v = \epsilon - x_0" />
              </li>
              <li>
                Loss ={' '}
                <InlineMath math="\text{MSE}(v_\theta(x_t, t),\; \epsilon - x_0)" />
              </li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Where Is x₀ at Inference?">
            If we need <InlineMath math="x_0" /> to compute the target velocity
            (<InlineMath math="v = \epsilon - x_0" />), how is this useful? The
            network gets (<InlineMath math="x_t" />,{' '}
            <InlineMath math="t" />) as input, NOT{' '}
            <InlineMath math="x_0" />. Just like DDPM: the training target uses{' '}
            <InlineMath math="x_0" /> and <InlineMath math="\epsilon" /> (both
            known during training), but the network only sees{' '}
            <InlineMath math="x_t" /> and <InlineMath math="t" />. At
            inference, we follow the learned velocity field from noise to
            generate <InlineMath math="x_0" />.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Training comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'DDPM Training',
              color: 'amber',
              items: [
                'x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon',
                'Target: epsilon (the noise)',
                'Loss: MSE(epsilon_theta, epsilon)',
                'Schedule: alpha_bar_t (carefully designed)',
              ],
            }}
            right={{
              title: 'Flow Matching Training',
              color: 'emerald',
              items: [
                'x_t = (1-t) * x_0 + t * epsilon',
                'Target: epsilon - x_0 (the velocity)',
                'Loss: MSE(v_theta, epsilon - x_0)',
                'Schedule: uniform t in [0, 1]',
              ],
            }}
          />
        </Row.Content>
      </Row>

      {/* Part E: Concrete worked example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Worked Example: One Training Step"
            subtitle="Flow matching with actual numbers"
          />
          <div className="space-y-4">
            <GradientCard title="Flow Matching Training Step" color="blue">
              <div className="space-y-3 text-sm">
                <p>
                  <strong>Data point:</strong>{' '}
                  <InlineMath math="x_0 = [3.0,\; 1.0]" />
                </p>
                <p>
                  <strong>Sampled noise:</strong>{' '}
                  <InlineMath math="\epsilon = [-1.0,\; 2.0]" />
                </p>
                <p>
                  <strong>Random time:</strong>{' '}
                  <InlineMath math="t = 0.3" />
                </p>
                <p>
                  <strong>Interpolated point:</strong>
                </p>
                <div className="py-2 px-4 bg-muted/30 rounded">
                  <BlockMath math="x_t = 0.7 \times [3.0,\; 1.0] + 0.3 \times [-1.0,\; 2.0] = [2.1 - 0.3,\; 0.7 + 0.6] = [1.8,\; 1.3]" />
                </div>
                <p>
                  <strong>Target velocity:</strong>
                </p>
                <div className="py-2 px-4 bg-muted/30 rounded">
                  <BlockMath math="v = \epsilon - x_0 = [-1.0,\; 2.0] - [3.0,\; 1.0] = [-4.0,\; 1.0]" />
                </div>
                <p>
                  <strong>Network input:</strong>{' '}
                  <InlineMath math="([1.8,\; 1.3],\; t=0.3)" />
                </p>
                <p>
                  <strong>Network target:</strong>{' '}
                  <InlineMath math="[-4.0,\; 1.0]" />
                </p>
                <p>
                  <strong>Loss:</strong>{' '}
                  <InlineMath math="\text{MSE}(v_\theta([1.8,\; 1.3],\; 0.3),\;\; [-4.0,\; 1.0])" />
                </p>
                <p className="text-muted-foreground mt-2">
                  Compare this to the DDPM training step: no{' '}
                  <InlineMath math="\sqrt{\bar\alpha}" />, no
                  variance-preserving rescaling, no noise schedule lookup. Just
                  a weighted average for{' '}
                  <InlineMath math="x_t" /> and a subtraction for the target.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Velocity Does Not Depend on t">
            The target velocity{' '}
            <InlineMath math="[-4.0,\; 1.0]" /> does not depend on{' '}
            <InlineMath math="t" />. The same{' '}
            <InlineMath math="(x_0, \epsilon)" /> pair produces the same target
            velocity at every timestep. The only thing{' '}
            <InlineMath math="t" /> changes is WHERE along the straight line
            (<InlineMath math="x_t" />) the network is asked to make its
            prediction.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* The "of course" chain */}
      <Row>
        <Row.Content>
          <GradientCard title="The 'Of Course' Chain" color="violet">
            <div className="space-y-2 text-sm">
              <ol className="list-decimal list-inside space-y-1">
                <li>We want to go from noise to data.</li>
                <li>The simplest path is a straight line.</li>
                <li>A straight line has constant velocity.</li>
                <li>Constant velocity is trivially easy to predict.</li>
                <li>So train a network to predict the velocity.</li>
              </ol>
              <p className="text-muted-foreground mt-2">
                Each step follows inevitably from the previous one. Given the
                ODE framework from the previous lesson, flow matching is the
                obvious simplification.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 8: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: The Interpolation"
            subtitle="Verify your understanding of flow matching"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In flow matching, what happens if{' '}
                  <InlineMath math="t=0.5" /> in the interpolation{' '}
                  <InlineMath math="x_t = (1-t)\, x_0 + t\, \epsilon" />?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    <InlineMath math="x_t = 0.5 \cdot x_0 + 0.5 \cdot \epsilon" />—exactly
                    a 50/50 mix of data and noise. In DDPM, the midpoint depends
                    on the noise schedule and is NOT a 50/50 mix.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague says: &ldquo;Flow matching is simpler because it
                  has no noise schedule.&rdquo; Is this correct?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Partially. Flow matching has a schedule—it is just the
                    trivial schedule{' '}
                    <InlineMath math="t \in [0, 1]" /> with uniform sampling.
                    The key simplification is that this schedule is linear,
                    requiring no{' '}
                    <InlineMath math="\bar\alpha" /> computation, no cumulative
                    products, no careful tuning. The &ldquo;schedule&rdquo; is
                    implicit in the linear interpolation.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Three Parameterizations */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Velocity, Noise, and Score"
            subtitle="Three parameterizations of the same vector field"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The previous lesson showed that noise prediction is a scaled score
              function. Velocity prediction is the third member of this family.
            </p>
            <p className="text-muted-foreground">
              All three are vector fields: at every point{' '}
              <InlineMath math="(x_t, t)" />, the network outputs a vector. The
              vectors differ in direction and magnitude, but they encode the
              same information. You can convert between them:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="One Network, Three Languages">
            The same U-Net or transformer architecture works for all three. You
            change the training target and loss function, not the model.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <div className="grid gap-4 md:grid-cols-3">
            <GradientCard title="Noise Prediction" color="amber">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-amber-300/90">
                  &ldquo;What noise was added?&rdquo;
                </p>
                <p className="text-muted-foreground">
                  Network outputs <InlineMath math="\epsilon_\theta" />—the
                  noise that was added. DDPM&rsquo;s language. The original
                  parameterization you learned in{' '}
                  <strong>Learning to Denoise</strong>.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Score Prediction" color="violet">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-violet-300/90">
                  &ldquo;Which way is higher probability?&rdquo;
                </p>
                <p className="text-muted-foreground">
                  Network outputs <InlineMath math="s_\theta" />—the gradient of
                  log probability. Score-based diffusion&rsquo;s language.
                  Connected to noise prediction by a scale factor from the
                  previous lesson.
                </p>
              </div>
            </GradientCard>
            <GradientCard title="Velocity Prediction" color="emerald">
              <div className="space-y-2 text-sm">
                <p className="font-medium text-emerald-300/90">
                  &ldquo;Which way is the trajectory heading?&rdquo;
                </p>
                <p className="text-muted-foreground">
                  Network outputs <InlineMath math="v_\theta" />—the tangent
                  direction along the trajectory. Flow matching&rsquo;s
                  language. The direction from data to noise:{' '}
                  <InlineMath math="v = \epsilon - x_0" />.
                </p>
              </div>
            </GradientCard>
          </div>
          <div className="mt-4 space-y-4">
            <p className="text-muted-foreground">
              Key conversions between them:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                From noise to score:{' '}
                <InlineMath math="\text{score} = -\epsilon / \sqrt{1-\bar\alpha_t}" />{' '}
                (from <LessonLink slug="score-functions-and-sdes">Score Functions &amp; SDEs</LessonLink>)
              </li>
              <li>
                From velocity to noise:{' '}
                since <InlineMath math="x_t = (1-t)\,x_0 + t\,\epsilon" /> and{' '}
                <InlineMath math="v = \epsilon - x_0" />, rearranging gives:
              </li>
            </ul>
            <div className="py-3 px-6 bg-muted/50 rounded-lg space-y-2">
              <BlockMath math="\epsilon = x_t + (1-t)\,v" />
              <BlockMath math="x_0 = x_t - t\,v" />
            </div>
            <p className="text-muted-foreground text-sm ml-4">
              Given the model&rsquo;s velocity prediction{' '}
              <InlineMath math="v_\theta(x_t, t)" /> and the current point{' '}
              <InlineMath math="x_t" />, you can recover both the predicted noise
              and the predicted clean data with one line of algebra each.
            </p>
            <p className="text-muted-foreground">
              Same trained network architecture. Same kind of output (a vector
              at each point). Different training objectives and different
              interpretations.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Address Misconception #3 */}
      <Row>
        <Row.Content>
          <GradientCard title="No New Architecture Required" color="emerald">
            <p className="text-sm">
              Velocity prediction does not require a different architecture. The
              same U-Net or transformer that predicts noise can predict velocity
              instead. You change the training target and the loss function, not
              the model architecture. The shift from U-Net to DiT (which
              SD3/Flux also do) is a separate architectural choice that happened
              to coincide with the shift to flow matching.
            </p>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Section 10: Check #3 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Parameterizations"
            subtitle="Verify your understanding of the three languages"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  If you have a model trained with noise prediction (DDPM-style)
                  and a model trained with velocity prediction (flow
                  matching-style), can you use the same ODE solver for both?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Yes. Both models produce a vector field. The solver follows
                    the trajectory defined by that field. The difference is that
                    the flow matching model&rsquo;s trajectories tend to be
                    straighter, so the solver needs fewer steps.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A paper says a model was trained with &ldquo;v-prediction.&rdquo;
                  Is this the same as flow matching?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Not exactly. V-prediction can be used within the DDPM
                    framework too—it is a parameterization choice. Flow matching
                    specifically refers to the combination of linear
                    interpolation AND velocity prediction AND uniform time
                    sampling. V-prediction within a DDPM noise schedule is
                    different from flow matching&rsquo;s v-prediction with
                    linear interpolation. The terminology is often used loosely
                    in practice.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Rectified Flow */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Rectified Flow"
            subtitle="When straight paths are not straight enough"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We said flow matching defines straight paths between paired{' '}
              <InlineMath math="(x_0, \epsilon)" /> points. But we train the
              network on <strong>many</strong> such pairs. The aggregate
              velocity field—what the network actually learns—is an average of
              all these individual straight-line velocities. This average can
              reintroduce curvature.
            </p>
            <p className="text-muted-foreground">
              Think of it this way: at a given point{' '}
              <InlineMath math="x_t" />, many different{' '}
              <InlineMath math="(x_0, \epsilon)" /> pairs could have passed
              through this point. The network&rsquo;s prediction at{' '}
              <InlineMath math="x_t" /> is an average of all their velocities.
              If different pairs have very different velocities at the same
              point, the average introduces curvature in the aggregate
              trajectory.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Individual vs Aggregate">
            Each individual path (one data point, one noise sample) is perfectly
            straight. But the <strong>learned</strong> velocity field averages
            over many paths, and the average can curve.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Rectified flow process */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Rectified flow is a simple but clever fix: take the learned flow,
              generate <InlineMath math="(\epsilon, x_0)" /> pairs using it,
              and re-train on the new pairs.
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <PhaseCard number={1} title="Initial Training" subtitle="Random pairing" color="cyan">
                <p className="text-sm">
                  Train on random (data, noise) pairs. Individual paths are
                  straight, but the aggregate field has curvature from
                  averaging.
                </p>
              </PhaseCard>
              <PhaseCard number={2} title="Generate Aligned Pairs" subtitle="Use the model" color="blue">
                <p className="text-sm">
                  Use the model to generate{' '}
                  <InlineMath math="(\epsilon, x_0)" /> pairs. These pairs are
                  connected by the learned flow, not random pairing.
                </p>
              </PhaseCard>
              <PhaseCard number={3} title="Retrain" subtitle="Straighter paths" color="emerald">
                <p className="text-sm">
                  Train on the aligned pairs. The averaging now introduces less
                  curvature because the pairs are naturally aligned by the
                  flow.
                </p>
              </PhaseCard>
            </div>
            <p className="text-muted-foreground">
              Each round of rectification straightens the trajectories further.
              In practice, 1-2 rounds of rectification significantly reduce the
              number of steps needed.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Refinement, Not Replacement">
            Rectified flow is like retracing a hand-drawn line with a ruler. The
            first pass gives you the approximate shape; the second pass
            straightens it. The first model is good; the rectified model is
            straighter and needs fewer steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 12: Check #4 (Transfer) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Questions"
            subtitle="Apply flow matching to the bigger picture"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1: SD3 and Flux" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  SD3 and Flux use flow matching. Based on what you know, what
                  advantage does this give them over a DDPM-trained model with
                  the same architecture?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Straighter trajectories mean fewer sampling steps for the
                    same quality. SD3 can generate in 20-30 steps where a
                    DDPM-trained model of similar size might need 50+. The
                    training objective is also simpler—no carefully tuned noise
                    schedule, just linear interpolation.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2: DPM-Solver++ on Flow Matching" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague proposes: &ldquo;Instead of rectified flow, why
                  not just use DPM-Solver++ on the flow matching model?&rdquo;
                  Would this help? Why or why not?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    It would help, but less than you might think. DPM-Solver++
                    handles curvature at inference time by estimating it with
                    multiple function evaluations. But if the trajectories are
                    already nearly straight from flow matching, there is little
                    curvature to handle. Rectified flow eliminates curvature at
                    the source; DPM-Solver++ compensates for it at inference.
                    Both help, but the combination gives diminishing returns
                    compared to either alone.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 13: Elaborate — Why Flow Matching Won */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Flow Matching Won"
            subtitle="Three reasons it became the standard"
          />
          <div className="space-y-4">
            <GradientCard title="Simpler Training" color="emerald">
              <ul className="space-y-1 text-sm">
                <li>
                  &bull; No noise schedule to tune (no{' '}
                  <InlineMath math="\bar\alpha" />, no{' '}
                  <InlineMath math="\beta_t" />, no cosine vs linear debate)
                </li>
                <li>
                  &bull; Linear interpolation instead of variance-preserving
                  formulation
                </li>
                <li>
                  &bull; Uniform time sampling instead of importance-weighted
                  timestep selection
                </li>
                <li>
                  &bull; The entire training loop fits in fewer lines of code
                </li>
              </ul>
            </GradientCard>
            <GradientCard title="Fewer Steps" color="blue">
              <ul className="space-y-1 text-sm">
                <li>
                  &bull; Straighter trajectories mean ODE solvers converge in
                  fewer steps
                </li>
                <li>
                  &bull; Euler&rsquo;s method is nearly exact on straight paths
                </li>
                <li>
                  &bull; Practical impact: 20-30 steps for SD3/Flux vs 50+ for
                  DDPM-style models
                </li>
                <li>
                  &bull; Rectified flow can push this even lower
                </li>
              </ul>
            </GradientCard>
            <GradientCard title="Architecture Independence" color="violet">
              <ul className="space-y-1 text-sm">
                <li>
                  &bull; Flow matching works with U-Nets, transformers, or any
                  architecture
                </li>
                <li>
                  &bull; The training objective does not depend on architectural
                  choices
                </li>
                <li>
                  &bull; This independence made it easy to adopt when the field
                  shifted from U-Net to DiT (Module 7.4)
                </li>
              </ul>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Same Family, Different Member">
            Flow matching is not a different paradigm from diffusion. It
            produces a vector field that maps noise to data, just like the
            probability flow ODE. The difference is HOW the vector field is
            trained (velocity vs score/noise) and WHAT trajectory it defines
            (straight vs curved). Same family, different member.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Address Misconception #4 */}
      <Row>
        <Row.Content>
          <GradientCard title="Two Independent Changes" color="rose">
            <div className="space-y-2 text-sm">
              <p>
                SD3 and Flux made two changes simultaneously: they switched from
                U-Net to DiT (<strong>architecture</strong> change) AND from
                noise prediction to flow matching (
                <strong>training objective</strong> change). These are
                independent choices.
              </p>
              <p className="text-muted-foreground">
                You could use flow matching with a U-Net, or noise prediction
                with a DiT. The changes happened to coincide, but they solve
                different problems. The architecture change is covered in
                Module 7.4.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
      </Row>

      {/* Analogy extension: "Same landscape, different lens" */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember &ldquo;same landscape, different lens&rdquo; from{' '}
              <strong>Score Functions &amp; SDEs</strong>? Now extend it to
              three lenses:
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Diffusion SDE" color="amber">
                <p className="text-sm">
                  Stochastic, curved paths. DDPM sampling. Many steps for
                  diversity.
                </p>
              </GradientCard>
              <GradientCard title="Probability Flow ODE" color="blue">
                <p className="text-sm">
                  Deterministic, curved paths. DDIM sampling. Fewer steps but
                  still fighting curvature.
                </p>
              </GradientCard>
              <GradientCard title="Flow Matching ODE" color="emerald">
                <p className="text-sm">
                  Deterministic, straight paths. Velocity-guided. Fewest steps
                  because curvature is eliminated by construction.
                </p>
              </GradientCard>
            </div>
            <p className="text-sm text-muted-foreground">
              Same start (noise), same end (data), different routes. Flow
              matching is not a &ldquo;better solver for the same
              trajectory.&rdquo; It is a different trajectory entirely.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Negative example: distinguish from "staircase to ramp" */}
      <Row>
        <Row.Content>
          <GradientCard title="Not a Smoother Staircase" color="rose">
            <div className="space-y-2 text-sm">
              <p>
                The &ldquo;staircase to ramp&rdquo; analogy from the previous
                lesson could mislead here. Flow matching is NOT about making
                the staircase smoother (that was the SDE). The ramp (SDE) still
                curves. Flow matching replaces the curved ramp with a{' '}
                <strong>straight line</strong>.
              </p>
              <p className="text-muted-foreground">
                It is not a better solver for the same trajectory. It is a
                different trajectory entirely.
              </p>
            </div>
          </GradientCard>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Velocity ≠ Score">
            The velocity field does NOT point toward high probability—it points
            along the straight-line trajectory from data to noise. The score
            field points toward high probability. The two fields agree on
            endpoints but differ in direction at intermediate points.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 14: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Flow Matching Hands-On"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes flow matching concrete—compare interpolation
                schemes, see why straight paths help ODE solvers, train a flow
                matching model from scratch on 2D data, and compare it head to
                head against DDPM.
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
                  <strong>Exercise 1 (Guided): Flow Matching vs DDPM
                  Interpolation.</strong>{' '}
                  Implement both interpolation schemes in 1D. Plot the paths from{' '}
                  <InlineMath math="x_0 = 5.0" /> to{' '}
                  <InlineMath math="\epsilon = -2.0" />. Predict: which path is
                  straighter? Plot the velocity (derivative) along each path.
                  Flow matching: constant. DDPM: varies with{' '}
                  <InlineMath math="t" />.
                </li>
                <li>
                  <strong>Exercise 2 (Guided): Euler Steps on Curved vs
                  Straight Paths.</strong>{' '}
                  Apply Euler&rsquo;s method with <InlineMath math="N=5" />{' '}
                  steps to a curved 2D trajectory and a straight one. Observe:
                  Euler on the curved path drifts. Euler on the straight path is
                  exact. Repeat with <InlineMath math="N=1" />. The straight
                  path is still exact.
                </li>
                <li>
                  <strong>Exercise 3 (Supported): Train a Flow Matching
                  Model on 2D Data.</strong>{' '}
                  Define a simple 2D target distribution (e.g., two-moons).
                  Implement the flow matching training loop. Train a small MLP.
                  Generate samples by Euler ODE solving. Predict: with 5 Euler
                  steps, will the samples be recognizable?
                </li>
                <li>
                  <strong>Exercise 4 (Independent): Compare Flow Matching to
                  DDPM.</strong>{' '}
                  Train both on the same 2D distribution. Generate at varying
                  step counts (1, 5, 10, 20, 50). Plot sample quality vs
                  number of steps. Bonus: try one round of rectified flow.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: compare interpolations</li>
              <li>Guided: Euler on curved vs straight</li>
              <li>Supported: train flow matching from scratch</li>
              <li>Independent: flow matching vs DDPM comparison</li>
            </ol>
            <p className="text-sm mt-2">
              Exercises mirror the lesson arc: see the difference, understand
              why it matters, build it yourself, compare head to head.
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
                headline: 'Curved vs straight.',
                description:
                  'Diffusion ODE trajectories curve because the score field changes with noise level. Flow matching trajectories are straight by construction (linear interpolation between data and noise).',
              },
              {
                headline: 'Velocity prediction.',
                description:
                  'Flow matching trains a network to predict velocity (v = epsilon - x_0) instead of noise or score. Same kind of network, different target, straight trajectories.',
              },
              {
                headline: 'Simpler training.',
                description:
                  'No noise schedule, no alpha_bar, no variance-preserving formulation. Just a weighted average (x_t = (1-t)*x_0 + t*epsilon) and a subtraction (v = epsilon - x_0).',
              },
              {
                headline: 'Fewer steps.',
                description:
                  "On a straight path, Euler's method is exact in one step. Flow matching models generate high-quality samples in 20-30 steps where DDPM-style models need 50+.",
              },
              {
                headline: 'Same family.',
                description:
                  'Noise prediction, score prediction, and velocity prediction are three parameterizations of the same underlying vector field. Flow matching is a member of the diffusion family, not a replacement.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="7.2"
            title="The Score-Based Perspective"
            achievements={[
              'Score function as gradient of log probability',
              'SDE/ODE duality for diffusion',
              'Score-noise equivalence',
              'Flow matching and velocity prediction',
              'Rectified flow for trajectory straightening',
            ]}
            nextModule="7.3"
            nextTitle="Fast Generation"
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Flow Matching for Generative Modeling',
                authors: 'Lipman, Chen, Ben-Hamu, Nickel & Le, 2023',
                url: 'https://arxiv.org/abs/2210.02747',
                note: 'The paper that formalized conditional flow matching. Section 3 covers the key idea: define straight-line conditional paths, derive the velocity target.',
              },
              {
                title: 'Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow',
                authors: 'Liu, Gong & Liu, 2023',
                url: 'https://arxiv.org/abs/2209.03003',
                note: 'Introduces rectified flow: iterative trajectory straightening by re-training on model-generated pairs. Section 3 covers the straightening procedure.',
              },
              {
                title: 'Scaling Rectified Flow Transformers for High-Resolution Image Synthesis',
                authors: 'Esser, Kulal, Blattmann, Entezari, Müller, Saini, Levi, Noroozi, Ommer, Rombach, 2024',
                url: 'https://arxiv.org/abs/2403.03206',
                note: 'The SD3 paper. Uses flow matching as the training objective. Section 2 connects flow matching to the practical system.',
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
            title="Next: Consistency Models"
            description="You now understand two ways to define the generation trajectory: the diffusion ODE (curved, score-guided) and flow matching (straight, velocity-guided). Both produce high-quality samples, but flow matching gets there in fewer steps. But what if you could collapse the ENTIRE trajectory into a SINGLE step? Not 20 steps, not 5 steps—one step, directly from noise to data. That is the promise of consistency models."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
