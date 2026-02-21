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
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

const NOTEBOOK_URL =
  'https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/7-3-1-consistency-models.ipynb'

/**
 * Consistency Models
 *
 * Lesson 1 in Module 7.3 (Fast Generation). Lesson 6 of 11 in Series 7.
 * Cognitive load: STRETCH (2 genuinely new concepts: self-consistency property, consistency distillation).
 *
 * Core concepts:
 * 1. Self-consistency property of ODE trajectories and its use as a training objective -- NEW
 * 2. Consistency distillation (teacher-student pattern for consistency models) -- NEW
 * Additionally: consistency training (without teacher), multi-step consistency
 *
 * Builds on: probability flow ODE from 7.2.1, ODE trajectory from 6.4.2,
 * Euler's method from 6.4.2, flow matching from 7.2.2, DDIM predict-and-leap from 6.4.2.
 *
 * Previous: Flow Matching (Module 7.2, Lesson 2 / BUILD)
 * Next: Latent Consistency Models & Adversarial Distillation (Module 7.3, Lesson 2)
 */

export function ConsistencyModelsLesson() {
  return (
    <LessonLayout>
      {/* Section 1: Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Consistency Models"
            description="What if you could collapse an entire ODE trajectory into a single function evaluation? Consistency models bypass trajectory-following entirely—mapping any noisy input directly to the clean endpoint."
            category="Fast Generation"
          />
        </Row.Content>
      </Row>

      {/* Section 2: Objective + Constraints */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand the self-consistency property of ODE trajectories—any
            point on the same trajectory maps to the same clean endpoint—and how
            this property becomes a training objective for a neural network that
            generates images in a single step, bypassing multi-step ODE solving
            entirely.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Module 7.3: Fast Generation">
            The previous module gave you the theoretical tools—score functions,
            ODE trajectories, flow matching. This module uses those tools to
            answer: how fast can we actually generate?
          </TipBlock>
        </Row.Aside>
      </Row>

      <Row>
        <Row.Content>
          <ConstraintBlock
            title="Scope for This Lesson"
            items={[
              'The self-consistency property and how it becomes a training objective',
              'Consistency distillation (learning from a pretrained teacher model)',
              'Consistency training without a teacher (conceptual comparison)',
              'Multi-step consistency as a quality-speed tradeoff',
              'NOT: Latent Consistency Models or LCM-LoRA (next lesson)',
              'NOT: Adversarial diffusion distillation / SDXL Turbo (next lesson)',
              'NOT: Full mathematical derivations or architecture details',
              'NOT: Production-scale consistency model training',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 3: Recap */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Recap"
            subtitle="Four concepts you will need in the next twenty minutes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <strong>The probability flow ODE.</strong> In{' '}
              <strong>Score Functions and SDEs</strong>, you learned that
              generation follows a deterministic trajectory from noise to data.
              Because it is deterministic, the same starting noise always leads
              to the same clean image.
            </p>
            <p className="text-muted-foreground">
              <strong>DDIM predict-and-leap.</strong> In{' '}
              <strong>Samplers and Efficiency</strong>, you learned that DDIM
              predicts the clean image <InlineMath math="x_0" /> from the
              current noisy image, then leaps toward it. But this prediction is
              approximate—it gets better as you get closer to{' '}
              <InlineMath math="x_0" />. That is why DDIM still needs multiple
              steps.
            </p>
            <p className="text-muted-foreground">
              <strong>The speed progression so far.</strong> Better solvers
              (DPM-Solver++, 15-20 steps). Straighter paths (flow matching,
              20-30 steps). Both still require{' '}
              <strong>stepping along a trajectory</strong>.
            </p>
            <p className="text-muted-foreground">
              <strong>Knowledge distillation.</strong> A large, expensive model
              (the teacher) generates examples or targets. A smaller or faster
              model (the student) learns to match the teacher&rsquo;s outputs
              rather than learning from scratch. The teacher does not need to be
              larger—it just needs to have knowledge that is expensive to
              produce. A diffusion model that requires 50 ODE steps to generate
              an image has expensive knowledge: the full trajectory from noise
              to data. A student model that learns to match the teacher&rsquo;s
              output in fewer steps is distilling that expensive knowledge into
              a cheaper form.
            </p>
            <p className="text-muted-foreground">
              Why does this work? The teacher has already spent millions of
              training steps learning where the ODE trajectory goes at every
              noise level. That knowledge is baked into its weights. The student
              does not need to rediscover this trajectory from scratch—it can
              learn from the teacher&rsquo;s established predictions. The
              teacher provides a shortcut: instead of raw data, the student
              learns from the teacher&rsquo;s processed understanding.
            </p>
            <p className="text-muted-foreground">
              You have seen this pattern before. LoRA adapts a pretrained
              model&rsquo;s knowledge for a specific task—the pretrained model
              is the &ldquo;teacher&rdquo; in spirit, and LoRA learns a
              compressed adaptation on top of it. Distillation is the broader
              version: a new model learns to reproduce a pretrained model&rsquo;s
              behavior directly. This pattern will be central to everything that
              follows.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Bridge">
            <strong>Flow Matching</strong> ended with a question: &ldquo;What
            if you could collapse the ENTIRE trajectory into a SINGLE
            step?&rdquo; Not 20 steps, not 5 steps—one step. Let&rsquo;s find
            out how.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Hook — Three Levels */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Three Levels of Speed"
            subtitle="A framework for everything you have learned about faster generation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You have learned two strategies for faster generation. Think of
              them as levels.
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              <GradientCard title="Level 1: Smarter Walking" color="amber">
                <div className="space-y-2 text-sm">
                  <p>
                    DPM-Solver++ follows the same curved path but takes larger,
                    smarter steps.
                  </p>
                  <p className="text-muted-foreground">
                    Result: 15-20 steps instead of 1000.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    You are still walking the path.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Level 2: Straighten the Road" color="emerald">
                <div className="space-y-2 text-sm">
                  <p>
                    Flow matching replaces the curved path with a straight one.
                    Euler&rsquo;s method is nearly exact.
                  </p>
                  <p className="text-muted-foreground">
                    Result: 20-30 steps.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    The road is straighter, so you walk faster.
                  </p>
                </div>
              </GradientCard>
              <GradientCard title="Level 3: ???" color="violet">
                <div className="space-y-2 text-sm">
                  <p>
                    What if you did not walk the path at all?
                  </p>
                  <p className="text-muted-foreground">
                    Result: 1 step.
                  </p>
                  <p className="text-xs text-muted-foreground italic">
                    You teleport to the destination.
                  </p>
                </div>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Level 3 is the idea behind consistency models. You do not step
              along the trajectory—you jump directly to the endpoint.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Progression">
            Better solvers treat the <strong>symptom</strong> (curvature). Flow
            matching treats the <strong>cause</strong> (the trajectory itself).
            Consistency models bypass the problem entirely—no trajectory, no
            curvature, no steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Explain — The Self-Consistency Property */}
      {/* Part A: A trivial property of deterministic ODEs */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Self-Consistency Property"
            subtitle="A trivial fact about deterministic ODEs, used in a non-trivial way"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Look at the probability flow ODE trajectory. It runs from{' '}
              <InlineMath math="x_T" /> (pure noise) to{' '}
              <InlineMath math="x_0" /> (clean data). Pick three points along
              it: one at high noise (<InlineMath math="t = 0.8" />), one at
              medium noise (<InlineMath math="t = 0.5" />), and one at low noise
              (<InlineMath math="t = 0.2" />).
            </p>

            <GradientCard title="One Trajectory, One Destination" color="blue">
              <div className="space-y-3 text-sm">
                <pre className="font-mono text-xs leading-relaxed text-blue-200/80 whitespace-pre">
{`  x_T (noise)
   |
   * ← x_t at t=0.8    ──→  all three map
   |                          to the same
   * ← x_t at t=0.5    ──→  endpoint x₀
   |
   * ← x_t at t=0.2    ──→
   |
  x₀ (clean data)`}
                </pre>
                <p className="text-muted-foreground">
                  All three points sit on the same trajectory, heading to the
                  same destination.
                </p>
              </div>
            </GradientCard>

            <p className="text-muted-foreground">
              Here is a fact so obvious it might not seem useful:{' '}
              <strong>
                all three of these points end up at the same place.
              </strong>
            </p>
            <p className="text-muted-foreground">
              Of course they do. The ODE is deterministic. If you start at any
              point on this trajectory and run the ODE forward to{' '}
              <InlineMath math="t = 0" />, you always arrive at{' '}
              <InlineMath math="x_0" />. The point at{' '}
              <InlineMath math="t = 0.8" /> and the point at{' '}
              <InlineMath math="t = 0.2" /> are on the same trajectory—they{' '}
              <strong>must</strong> have the same destination.
            </p>

            <p className="text-muted-foreground">
              To make this concrete, imagine a 2D diffusion model trained on the
              two-moons distribution. One ODE trajectory passes through these
              points:
            </p>

            <GradientCard title="Worked Example: Two Points, One Endpoint" color="orange">
              <div className="space-y-3 text-sm">
                <ul className="space-y-2">
                  <li>
                    &bull; At <InlineMath math="t = 0.7" />, the trajectory
                    passes through <InlineMath math="x_t = (1.3,\; {-}0.8)" />
                  </li>
                  <li>
                    &bull; At <InlineMath math="t = 0.3" />, the same
                    trajectory passes through{' '}
                    <InlineMath math="x_t = (0.5,\; {-}0.3)" />
                  </li>
                  <li>
                    &bull; The trajectory ends at the clean data point{' '}
                    <InlineMath math="x_0 = (0.2,\; {-}0.1)" />
                  </li>
                </ul>
                <p className="text-muted-foreground">
                  A consistency function <InlineMath math="f" /> should satisfy:
                </p>
                <div className="py-2 px-4 bg-muted/30 rounded">
                  <InlineMath math="f\!\left((1.3, {-}0.8),\; 0.7\right) = f\!\left((0.5, {-}0.3),\; 0.3\right) = (0.2, {-}0.1)" />
                </div>
                <p className="text-muted-foreground">
                  Both points map to the same endpoint because they are on the
                  same trajectory. Later, the consistency distillation training
                  loss for this pair would be{' '}
                  <InlineMath math="d\!\left(f(1.3, {-}0.8,\; 0.7),\;\; f(0.5, {-}0.3,\; 0.3)\right)" />—the
                  distance between the two predictions. If the model is
                  well-trained, this distance is near zero.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a New Mathematical Result">
            This is not a discovery about ODEs. It is the{' '}
            <strong>definition</strong> of a deterministic ODE. The insight is
            not the property—it is what we can <strong>do</strong> with it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Part B: The consistency function */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Consistency Function"
            subtitle="From a trivial property to a powerful training objective"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Define a function <InlineMath math="f(x_t, t)" /> that maps any
              point on a trajectory to its endpoint:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="f(x_t, t) = x_0 \quad \text{for all } t, \text{ where } x_t \text{ lies on the ODE trajectory ending at } x_0" />
            </div>
            <p className="text-muted-foreground">
              The self-consistency constraint: for any two points on the same
              trajectory,
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="f(x_t, t) = f(x_{t'}, t') \quad \text{for all } t, t' \text{ on the same ODE trajectory}" />
            </div>
            <p className="text-muted-foreground">
              And the boundary condition:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="f(x, \epsilon) = x \quad \text{(at noise level near 0, you are already at the endpoint)}" />
            </div>
            <p className="text-muted-foreground">
              If a neural network could learn this function{' '}
              <InlineMath math="f" />, generation would be trivial: sample noise{' '}
              <InlineMath math="x_T" />, call{' '}
              <InlineMath math="f(x_T, T)" />, get a clean image.{' '}
              <strong>
                One function evaluation. No ODE solver. No trajectory stepping.
              </strong>
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Boundary Condition">
            <InlineMath math="f(x, \epsilon) = x" /> anchors the function: at
            noise level near zero, the input is already nearly clean, so the
            model should not change it. This prevents the consistency function
            from &ldquo;hallucinating&rdquo; changes to already-clean images.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Address Misconception #1: Not the same as 1-step ODE */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <ComparisonRow
              left={{
                title: 'ODE Solver (1 Step)',
                color: 'amber',
                items: [
                  'Computes direction at x_T (score/noise/velocity)',
                  'Takes one massive step along that direction',
                  'Terrible accuracy (massive curvature error)',
                  'No special training needed',
                ],
              }}
              right={{
                title: 'Consistency Model',
                color: 'emerald',
                items: [
                  'Computes f(x_T, T) directly—the endpoint',
                  'No direction, no step—a direct mapping',
                  'Trained specifically for single-step accuracy',
                  'Requires consistency training objective',
                ],
              }}
            />
            <p className="text-sm text-muted-foreground">
              This is <strong>not</strong> the same as running an ODE solver with
              1 step. An ODE solver with 1 step computes a direction and takes a
              single massive step. The consistency model does not compute a
              direction—it maps{' '}
              <InlineMath math="x_T" /> directly to{' '}
              <InlineMath math="x_0" />. No direction, no step. A direct
              mapping.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Fewer Steps—No Steps">
            The consistency model is not an ODE solver with fewer steps. It
            bypasses ODE solving entirely. You cannot &ldquo;pause&rdquo; a
            consistency model mid-generation—there is no intermediate trajectory
            to inspect.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Part C: DDIM comparison */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Connection to DDIM"
            subtitle="The closest thing you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              This should remind you of DDIM&rsquo;s predict-and-leap.
              DDIM also predicts <InlineMath math="x_0" /> from{' '}
              <InlineMath math="x_t" /> at each step. But DDIM&rsquo;s
              prediction is approximate—it uses the noise prediction{' '}
              <InlineMath math="\epsilon_\theta(x_t, t)" /> to estimate{' '}
              <InlineMath math="x_0" />, and this estimate improves as{' '}
              <InlineMath math="t" /> decreases (as you get closer to the clean
              image). That is why DDIM still needs multiple predict-and-leap
              iterations.
            </p>
            <p className="text-muted-foreground">
              The consistency model is trained specifically so that{' '}
              <InlineMath math="f(x_t, t)" /> is accurate{' '}
              <strong>even when <InlineMath math="t" /> is large</strong> (far
              from the clean image). DDIM predicts{' '}
              <InlineMath math="x_0" /> and iterates. The consistency model
              predicts <InlineMath math="x_0" /> and <strong>stops</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Predict-and-Leap, Perfected">
            DDIM: predict <InlineMath math="x_0" />, leap, predict again, leap
            again... The consistency model: predict{' '}
            <InlineMath math="x_0" />, done. The ultimate predict-and-leap—one
            prediction, one leap.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Important clarification: not memorization */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One thing to keep straight: at inference time, you start from{' '}
              <strong>pure random noise</strong>—not a point that lies on any
              specific ODE trajectory from the training data. The consistency
              model must generalize beyond the trajectories it trained on. The
              self-consistency property is the <strong>training signal</strong>,
              not a runtime constraint. The model does not memorize
              trajectory-endpoint pairs—it learns a continuous mapping from
              (noisy input, noise level) to clean data that works for arbitrary
              noise it has never seen before.
            </p>
          </div>
        </Row.Content>
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
                  Two points <InlineMath math="x_{t_1}" /> and{' '}
                  <InlineMath math="x_{t_2}" /> are on{' '}
                  <strong>different</strong> ODE trajectories (heading to
                  different clean images). Should{' '}
                  <InlineMath math="f(x_{t_1}, t_1)" /> equal{' '}
                  <InlineMath math="f(x_{t_2}, t_2)" />?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The self-consistency property only holds for points on
                    the <strong>same</strong> trajectory. Different trajectories
                    have different endpoints.{' '}
                    <InlineMath math="f(x_{t_1}, t_1)" /> and{' '}
                    <InlineMath math="f(x_{t_2}, t_2)" /> should be different
                    clean images.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague says: &ldquo;The consistency model just memorizes
                  which noise maps to which clean image.&rdquo; Is this correct?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    No. The model learns a continuous mapping from (noisy image,
                    noise level) to clean image. It handles noise inputs it has
                    never seen during training, including noise that does not lie
                    on any specific ODE trajectory from the training data. The
                    self-consistency property is the training signal—at
                    inference, the model generalizes.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 3" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  What does the boundary condition{' '}
                  <InlineMath math="f(x, \epsilon) = x" /> accomplish?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    It anchors the function at near-clean inputs. If{' '}
                    <InlineMath math="x" /> is already nearly clean, the model
                    should not change it. This prevents the consistency function
                    from hallucinating changes to already-clean images and
                    provides a stable reference point that the rest of the
                    function must be consistent with.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 7: Explain — Consistency Distillation */}
      {/* Part A: The teacher-student setup */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Consistency Distillation"
            subtitle="Using a pretrained model to teach the consistency function"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              How do you train a network to learn the consistency function? The
              self-consistency property says{' '}
              <InlineMath math="f" /> should be constant along any ODE
              trajectory. But you need to know where the trajectory goes to
              enforce this constraint.
            </p>
            <p className="text-muted-foreground">
              <strong>Consistency distillation</strong> uses a pretrained
              diffusion model as the teacher. The teacher already knows the ODE
              trajectory—its noise predictions define the vector field that the
              probability flow ODE follows.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Teacher Knows the Path">
            The pretrained diffusion model has been trained for millions of
            steps. It knows where the ODE trajectory goes at every point. The
            consistency model learns from this established knowledge rather
            than discovering the trajectory from scratch.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Training procedure */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Procedure"
            subtitle="Four steps to distill a trajectory into a single function call"
          />
          <div className="space-y-4">
            <ol className="list-decimal list-inside text-muted-foreground space-y-3 ml-4">
              <li>
                Sample a data point <InlineMath math="x_0" />, add noise to
                get <InlineMath math="x_{t_{n+1}}" /> (a point at a higher
                noise level)
              </li>
              <li>
                Use the pretrained teacher to estimate where{' '}
                <InlineMath math="x_{t_{n+1}}" /> would be at the next lower
                noise level <InlineMath math="t_n" /> (one ODE step):{' '}
                <InlineMath math="\hat{x}_{t_n} = \text{ODE\_step}(x_{t_{n+1}}, t_{n+1}, t_n, \text{teacher})" />
              </li>
              <li>
                Train the consistency model so that{' '}
                <InlineMath math="f_\theta(x_{t_{n+1}}, t_{n+1})" /> is close
                to{' '}
                <InlineMath math="f_{\theta^-}(\hat{x}_{t_n}, t_n)" />
              </li>
              <li>
                <InlineMath math="\theta^-" /> is an exponential moving average
                (EMA) of <InlineMath math="\theta" />—a slowly-moving copy
                that stabilizes training
              </li>
            </ol>
            <p className="text-muted-foreground">
              In words: two points on the same trajectory (estimated by the
              teacher) should produce the same output from the consistency
              model.
            </p>

            <GradientCard title="The Distillation Picture" color="blue">
              <div className="space-y-3 text-sm">
                <pre className="font-mono text-xs leading-relaxed text-blue-200/80 whitespace-pre">
{`  ODE trajectory
  ──────────────────────────────────────
  x_{t_{n+1}}                  x_{t_n}
      ●─── teacher ODE step ───→ ●
      │                           │
      │ f_θ                       │ f_{θ⁻}
      ▼                           ▼
    pred₁         loss ≈ 0      pred₂
      └──── should match ─────────┘`}
                </pre>
                <p className="text-muted-foreground">
                  The teacher connects two points on the same trajectory (one
                  ODE step). Both points are fed through the consistency
                  model—<InlineMath math="f_\theta" /> for the noisier point,{' '}
                  <InlineMath math="f_{\theta^-}" /> (EMA copy) for the
                  teacher-estimated cleaner point. The loss penalizes any
                  difference between the two predictions.
                </p>
              </div>
            </GradientCard>

            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\mathcal{L}_\text{CD} = d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat{x}_{t_n}, t_n)\right)" />
            </div>
            <p className="text-sm text-muted-foreground ml-4">
              where <InlineMath math="d" /> is a distance metric (e.g., L2 or
              LPIPS) and <InlineMath math="\hat{x}_{t_n}" /> is the
              teacher&rsquo;s one-step ODE estimate.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Pattern as Before">
            The training loop structure is familiar: sample a random point in
            time, compute a target, minimize a distance metric. The key
            difference: the target comes from the{' '}
            <strong>model&rsquo;s own predictions</strong> at an adjacent
            timestep (via the EMA copy), not from the ground truth noise or
            velocity.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Part C: The EMA target */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Two Copies of the Model?"
            subtitle="The EMA target prevents collapse"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The consistency model appears on <strong>both sides</strong> of
              the loss: <InlineMath math="f_\theta" /> on the left (for the
              higher-noise point) and{' '}
              <InlineMath math="f_{\theta^-}" /> on the right (for the
              teacher-estimated lower-noise point). If both sides used the
              same <InlineMath math="\theta" />, the model could cheat by
              making <InlineMath math="f" /> constant everywhere—outputting the
              same image regardless of input.
            </p>
            <p className="text-muted-foreground">
              The EMA target <InlineMath math="\theta^-" /> is a slowly-moving
              copy of <InlineMath math="\theta" />. It provides a stable target
              that the model must match, preventing collapse.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="A Common Pattern">
            This EMA target pattern appears throughout self-supervised learning
            (momentum contrast, BYOL). The idea is always the same: when both
            sides of the loss come from the same model, make one side lag
            behind to prevent trivial solutions.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 8: Explain — Consistency Training (Without a Teacher) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Consistency Training (Without a Teacher)"
            subtitle="What if you do not have a pretrained model?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if you do not have a pretrained teacher model? Consistency
              training learns the consistency function directly, without a
              teacher providing the ODE trajectory.
            </p>
            <p className="text-muted-foreground">
              Instead of using a teacher&rsquo;s one-step ODE estimate,
              consistency training uses the model&rsquo;s <strong>own</strong>{' '}
              predictions at adjacent timesteps. The training signal comes from
              the model&rsquo;s self-consistency: adjacent timesteps should
              produce similar outputs, and this similarity improves as training
              progresses.
            </p>
            <p className="text-muted-foreground">
              The key difference: in consistency distillation, the teacher
              provides a good estimate of{' '}
              <InlineMath math="\hat{x}_{t_n}" /> from the start. In
              consistency training, the estimate starts poor and improves as the
              model trains. This makes consistency training slower and harder to
              converge.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Discovery vs Learning">
            Distillation: the teacher already knows the trajectory. The
            consistency model <strong>learns</strong> from it. Training: the
            consistency model must <strong>discover</strong> the trajectory
            while simultaneously learning to collapse it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Comparison */}
      <Row>
        <Row.Content>
          <ComparisonRow
            left={{
              title: 'Consistency Distillation',
              color: 'emerald',
              items: [
                'Requires a pretrained teacher model',
                'Teacher provides accurate ODE step estimates',
                'Faster training, better sample quality',
                'Most common in practice',
                'Tied to a specific teacher model',
              ],
            }}
            right={{
              title: 'Consistency Training',
              color: 'amber',
              items: [
                'No teacher model required',
                'Model discovers trajectory from scratch',
                'Slower convergence, lower quality at 1-2 steps',
                'Useful when no teacher is available',
                'Independent—no external dependency',
              ],
            }}
          />
          <div className="mt-4">
            <p className="text-sm text-muted-foreground">
              These are <strong>not</strong> interchangeable. In the original
              paper, consistency distillation achieves FID ~3.5 on ImageNet
              64&times;64 while consistency training achieves FID ~6.2 at
              similar step counts. The teacher&rsquo;s knowledge makes a real
              difference.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 9: Check #2 */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Check: Distillation vs Training"
            subtitle="Predict before you read the answers"
          />
          <div className="space-y-4">
            <GradientCard title="Question 1" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  In consistency distillation, the teacher takes one ODE step
                  from <InlineMath math="x_{t_{n+1}}" /> to estimate{' '}
                  <InlineMath math="\hat{x}_{t_n}" />. Why only one step? Why
                  not run the full ODE to get the exact endpoint?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Running the full ODE would be prohibitively expensive during
                    training—you would need 10-50 model evaluations per training
                    step. One ODE step is cheap (one teacher evaluation) and
                    provides a reasonable estimate, especially between adjacent
                    timesteps where the trajectory changes little. The
                    consistency model learns to make these local consistency
                    constraints imply global consistency.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Question 2" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  If you had a flow matching model instead of a DDPM model as
                  the teacher for consistency distillation, would you expect
                  better or worse results?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    Likely better. The flow matching model defines straighter ODE
                    trajectories. Straighter trajectories mean the one-step ODE
                    estimate <InlineMath math="\hat{x}_{t_n}" /> is more
                    accurate (less curvature error between adjacent timesteps). A
                    more accurate estimate provides a better training signal. This
                    is one reason flow matching and consistency distillation are
                    complementary, not competing.
                  </p>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 10: Explain — Multi-Step Consistency */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multi-Step Consistency"
            subtitle="When one step is not quite enough"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One-step consistency model generation is fast but not as good as
              multi-step diffusion. The single-step prediction from pure noise
              is a hard problem—the model must map from a maximally uncertain
              input to a clean image.
            </p>
            <p className="text-muted-foreground">
              Multi-step consistency is a middle ground: run the consistency
              model at 2-4 noise levels instead of just 1. At each step, the
              model maps from the current noise level directly to{' '}
              <InlineMath math="x_0" /> (using the consistency function), then{' '}
              <strong>re-noises</strong> <InlineMath math="x_0" /> to a lower
              noise level, and runs the consistency function again.
            </p>

            <GradientCard title="Multi-Step Consistency Procedure" color="blue">
              <ol className="list-decimal list-inside space-y-2 text-sm">
                <li>
                  Start at <InlineMath math="x_T" /> (pure noise)
                </li>
                <li>
                  Apply <InlineMath math="f(x_T, T)" /> to get a clean
                  estimate <InlineMath math="\hat{x}_0" />
                </li>
                <li>
                  Add noise to <InlineMath math="\hat{x}_0" /> to get{' '}
                  <InlineMath math="x_{t_2}" /> at a lower noise level{' '}
                  <InlineMath math="t_2" />
                </li>
                <li>
                  Apply <InlineMath math="f(x_{t_2}, t_2)" /> to get a better
                  clean estimate
                </li>
                <li>Repeat for 1-3 more noise levels</li>
              </ol>
            </GradientCard>

            <p className="text-muted-foreground">
              Each application of <InlineMath math="f" /> starts from a{' '}
              <strong>less noisy</strong> input, so its prediction is more
              accurate. 2-4 steps of this process recovers much of the quality
              gap between 1-step consistency and 50-step diffusion.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Quality-Speed Tradeoff">
            One-step consistency models are <strong>not</strong> as good as
            50-step diffusion. There is a real quality tradeoff. But 4-step
            consistency often approaches 50-step quality—still a massive
            speedup from 50 to 4.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* How multi-step differs from ODE solving */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <GradientCard title="Multi-Step Consistency ≠ ODE Solving" color="rose">
              <div className="space-y-2 text-sm">
                <p>
                  Multi-step consistency is <strong>not</strong> the same as
                  running the ODE with 2-4 steps. In ODE solving, each step{' '}
                  <strong>continues</strong> from where the previous step left
                  off—you walk along the trajectory. In multi-step consistency,
                  each step <strong>restarts</strong>: jump to{' '}
                  <InlineMath math="x_0" />, re-noise, jump to{' '}
                  <InlineMath math="x_0" /> again.
                </p>
                <p className="text-muted-foreground">
                  Each step is an independent application of the consistency
                  function, not a continuation of a trajectory.
                </p>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 11: Check #3 (Transfer) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Transfer Questions"
            subtitle="Apply consistency models to the bigger picture"
          />
          <div className="space-y-4">
            <GradientCard title="Scenario 1: Where Does Multi-Step Fit?" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  Based on the three levels framework (smarter walking,
                  straighten the road, teleport), where does multi-step
                  consistency fit? Is it Level 2 or Level 3?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <p className="mt-2 text-muted-foreground">
                    It is Level 3 with a refinement flavor. Each individual step{' '}
                    <strong>is</strong> a teleportation—the consistency function
                    maps directly to <InlineMath math="x_0" />, not along a
                    trajectory. But the re-noising and re-applying pattern gives
                    it an iterative feel. The key distinction: there is no
                    trajectory being followed between steps. Each step is an
                    independent jump to the endpoint.
                  </p>
                </details>
              </div>
            </GradientCard>

            <GradientCard title="Scenario 2: Always Use 1 Step?" color="cyan">
              <div className="space-y-2 text-sm">
                <p>
                  A colleague suggests: &ldquo;We should just distill any
                  diffusion model into a consistency model and always use 1
                  step.&rdquo; What tradeoffs should they consider?
                </p>
                <details className="mt-2">
                  <summary className="font-medium cursor-pointer text-primary">
                    Reveal
                  </summary>
                  <div className="mt-2 text-muted-foreground space-y-2">
                    <p>Several tradeoffs:</p>
                    <ul className="list-disc list-inside space-y-1 ml-2">
                      <li>
                        1-step quality is noticeably lower—for art or medical
                        imaging, this matters
                      </li>
                      <li>
                        Consistency distillation requires a teacher model and
                        training time—not free
                      </li>
                      <li>
                        The consistency model is bounded by the teacher&rsquo;s
                        quality ceiling
                      </li>
                      <li>
                        Multi-step consistency (2-4 steps) is often the sweet
                        spot: much faster than 50-step diffusion, close in
                        quality
                      </li>
                      <li>
                        A regular diffusion model can generate at any step
                        count—a consistency model is specialized for low-step
                        generation
                      </li>
                    </ul>
                  </div>
                </details>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
      </Row>

      {/* Section 12: Elaborate — Positioning */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Where Consistency Models Fit"
            subtitle="Complementary, not competing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is the complete Level 3:
            </p>
            <GradientCard title="Level 3: Bypass the Trajectory" color="violet">
              <ul className="space-y-1 text-sm">
                <li>
                  &bull; Consistency models learn a direct mapping: any noisy
                  input to the clean endpoint
                </li>
                <li>
                  &bull; No ODE solver needed (in single-step mode)
                </li>
                <li>
                  &bull; Trained using the self-consistency property of the
                  underlying ODE
                </li>
                <li>
                  &bull; Quality-speed tradeoff: 1 step (fast, softer), 2-4
                  steps (fast, near-diffusion quality)
                </li>
              </ul>
            </GradientCard>
            <p className="text-muted-foreground">
              Flow matching and consistency models are{' '}
              <strong>complementary, not competing</strong>. Flow matching makes
              the trajectory straighter (better training signal, more stable
              convergence). Consistency models bypass the trajectory at inference
              (fewer steps needed). A flow matching model can serve as the
              teacher for consistency distillation, getting the best of both:
              stable training from straight trajectories, plus single-step
              inference from the consistency function.
            </p>
            <p className="text-muted-foreground">
              This combination is exactly what Latent Consistency Models use—the
              subject of the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Same Landscape, Another Lens">
            Diffusion SDE: walk the stochastic path. Probability flow ODE:
            drive the deterministic path. Flow matching: straighten the road.
            Consistency models: teleport to the destination. Same landscape,
            four ways to traverse it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* What the consistency model actually learns */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The consistency model learns a mapping from (noisy image, noise
              level) to clean image. This is similar to what the noise
              prediction network already does—given a noisy input, estimate the
              clean image. The difference is in <strong>how</strong> it is
              trained. The noise prediction network is trained on{' '}
              <InlineMath math="\text{MSE}(\epsilon_\theta, \epsilon)" />. The
              consistency model is trained on the self-consistency constraint:
              adjacent points on the trajectory should agree.
            </p>
            <p className="text-muted-foreground">
              Song et al. show that consistency models can be parameterized
              using the same architecture as the original diffusion model, with
              a specific skip connection structure that enforces the boundary
              condition <InlineMath math="f(x, \epsilon) = x" />.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 13: Practice — Notebook Link */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Practice: Consistency Models Hands-On"
            subtitle="Four exercises from guided to independent"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                The notebook makes consistency models concrete—verify the
                self-consistency property on real ODE trajectories, see why
                single-step ODE methods fail, train a toy consistency model from
                scratch, and compare multi-step consistency to multi-step ODE
                solving.
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
                    Exercise 1 (Guided): Visualize Self-Consistency on ODE
                    Trajectories.
                  </strong>{' '}
                  Use a pretrained 2D diffusion model to generate ODE
                  trajectories. Pick one trajectory and highlight 5 points at
                  different noise levels. Verify: running the ODE from each
                  point arrives at the same endpoint.
                </li>
                <li>
                  <strong>
                    Exercise 2 (Guided): One-Step ODE vs Consistency Model
                    Prediction.
                  </strong>{' '}
                  Compare the true ODE endpoint, a single Euler step from{' '}
                  <InlineMath math="x_T" />, and DDIM&rsquo;s 1-step{' '}
                  <InlineMath math="x_0" /> prediction. Predict: how far off
                  will the 1-step Euler estimate be?
                </li>
                <li>
                  <strong>
                    Exercise 3 (Supported): Train a Toy Consistency Model.
                  </strong>{' '}
                  Implement consistency distillation on the 2D two-moons
                  dataset. Use the pretrained model as teacher. Generate 1-step
                  samples and compare to the teacher&rsquo;s multi-step output.
                </li>
                <li>
                  <strong>
                    Exercise 4 (Independent): Multi-Step Consistency and Quality
                    Comparison.
                  </strong>{' '}
                  Generate samples at 1, 2, 4, and 8 consistency steps. Compare
                  to the teacher at 1, 5, 10, 20, 50 ODE steps. Bonus: try
                  consistency training (without a teacher) and compare convergence.
                </li>
              </ul>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Exercise Progression">
            <ol className="space-y-1 text-sm list-decimal list-inside">
              <li>Guided: verify the self-consistency property</li>
              <li>Guided: see why 1-step ODE fails</li>
              <li>Supported: train consistency distillation</li>
              <li>Independent: multi-step comparison</li>
            </ol>
            <p className="text-sm mt-2">
              Exercises mirror the lesson arc: see the property, understand why
              it matters, build it yourself, evaluate the tradeoffs.
            </p>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 14: Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Self-consistency.',
                description:
                  'Any point on the same ODE trajectory maps to the same clean endpoint. This trivial property of deterministic ODEs becomes a powerful training objective: train a network so that f(x_t, t) is constant along any trajectory.',
              },
              {
                headline: 'Three levels of speed.',
                description:
                  'Better solvers (walk the path smarter). Straighter paths (straighten the road). Consistency models (bypass the road entirely—teleport to the destination).',
              },
              {
                headline: 'Distillation vs training.',
                description:
                  'Consistency distillation uses a pretrained teacher to provide the trajectory. Consistency training discovers the trajectory on its own. Distillation is faster and better, but requires an existing model.',
              },
              {
                headline: 'Quality-speed tradeoff.',
                description:
                  'One-step consistency is fast but softer. Multi-step consistency (2-4 steps) recovers most of the quality. The right number of steps depends on the application.',
              },
              {
                headline: 'Complementary, not competing.',
                description:
                  'Flow matching makes trajectories straighter; consistency models bypass them. The combination (flow matching teacher + consistency distillation) gets the best of both.',
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
                title: 'Consistency Models',
                authors: 'Song, Dhariwal, Chen & Sutskever, 2023',
                url: 'https://arxiv.org/abs/2303.01469',
                note: 'The original consistency models paper. Section 3 defines the self-consistency property and derives both consistency distillation and consistency training. Section 5 has the ImageNet experiments.',
              },
              {
                title: 'Improved Consistency Training',
                authors: 'Song & Dhariwal, 2023',
                url: 'https://arxiv.org/abs/2310.14189',
                note: 'Follow-up that significantly improves consistency training (without a teacher). Closes much of the gap with consistency distillation.',
              },
              {
                title: 'Latent Consistency Models',
                authors: 'Luo, Tan, Huang, Li & Zhao, 2023',
                url: 'https://arxiv.org/abs/2310.04378',
                note: 'Applies consistency distillation to latent diffusion models (Stable Diffusion). This is the focus of the next lesson.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 15: Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: Fast Generation in Practice"
            description="You now understand the consistency model idea: collapse an entire ODE trajectory into a single function evaluation. But we have been working with toy 2D data. The next lesson takes this to real scale: Latent Consistency Models (LCM) apply consistency distillation to Stable Diffusion and SDXL, enabling 1-4 step generation from existing checkpoints. We will also see a different approach—adversarial diffusion distillation in SDXL Turbo, which uses a GAN-style discriminator instead of self-consistency."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
