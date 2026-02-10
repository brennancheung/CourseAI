'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  TryThisBlock,
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
  ConstraintBlock,
  ComparisonRow,
  GradientCard,
  ReferencesBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { OptimizerExplorer } from '@/components/widgets/OptimizerExplorer'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Optimizers — Lesson 5 of Module 1.3 (Training Neural Networks)
 *
 * Teaches the student to understand:
 * - Why vanilla SGD struggles on certain loss landscapes (ravine problem)
 * - How momentum smooths gradient direction via EMA
 * - How RMSProp adapts learning rates per parameter
 * - How Adam combines momentum + RMSProp
 * - When to use which optimizer (no free lunch)
 *
 * Target depths:
 * - Momentum: DEVELOPED
 * - Adam: DEVELOPED
 * - RMSProp: INTRODUCED
 * - Exponential moving average: DEVELOPED
 * - Per-parameter learning rates: INTRODUCED
 */

export function OptimizersLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Optimizers"
            description="Why vanilla SGD struggles in ravines—and how momentum, RMSProp, and Adam fix it."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand what modern optimizers do and why they exist. By the end,
            you&apos;ll know what Adam is actually doing—not just &ldquo;use Adam&rdquo;—and
            when SGD might be the better choice.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Update Rule">
            Every optimizer in this lesson is a modification of the same gradient
            descent update:{' '}
            <InlineMath math="\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot (\text{something})" />.
            The &ldquo;something&rdquo; is what changes.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'We focus on 3 optimizers: Momentum, RMSProp, Adam. There are dozens of others—these 3 cover the core ideas.',
              'No code implementation—that comes in the PyTorch series',
              'No convergence proofs or theoretical guarantees',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — The Ravine Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Ravine Problem"
            subtitle="Not all loss landscapes are created equal"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Batching and SGD, you watched optimization paths on a loss landscape with two
              bowl-shaped minima. The gradients pointed roughly toward the nearest minimum, and
              SGD got there—maybe with some noise, but it got there.
            </p>
            <p className="text-muted-foreground">
              Now imagine a different shape: a long, narrow valley—like a canyon with steep walls
              but a gentle downward slope along the floor. This is called a <strong>ravine</strong>,
              and most real neural network loss landscapes have exactly this shape, because different
              parameters operate at wildly different scales.
            </p>
            <p className="text-muted-foreground">
              What happens when vanilla SGD encounters a ravine? The gradient points mostly{' '}
              <em>across</em> the ravine (the steep direction), not <em>along</em> it (the
              gentle direction). So SGD bounces back and forth between the canyon walls,
              barely making progress toward the actual minimum.
            </p>

            {/* Zigzag vs smooth path illustration */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <svg viewBox="0 0 400 140" className="w-full" style={{ maxHeight: 160 }}>
                {/* Ravine shape — elongated ellipse contours */}
                <ellipse cx="200" cy="70" rx="170" ry="40" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.15" />
                <ellipse cx="200" cy="70" rx="140" ry="32" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.2" />
                <ellipse cx="200" cy="70" rx="110" ry="24" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.25" />
                <ellipse cx="200" cy="70" rx="80" ry="16" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.3" />
                <ellipse cx="200" cy="70" rx="50" ry="9" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.35" />
                <ellipse cx="200" cy="70" rx="20" ry="4" fill="none" stroke="#6366f1" strokeWidth="1" opacity="0.4" />

                {/* Vanilla SGD path — zigzag */}
                <polyline
                  points="50,35 80,105 115,40 145,100 170,45 195,90 215,55 235,80 250,62 265,75 275,68"
                  fill="none" stroke="#ef4444" strokeWidth="2" opacity="0.8"
                  strokeLinejoin="round"
                />
                <circle cx="50" cy="35" r="4" fill="#ef4444" />
                <circle cx="275" cy="68" r="4" fill="#ef4444" />

                {/* Momentum path — smooth curve */}
                <path
                  d="M 50,105 Q 100,85 150,75 Q 200,68 250,70 Q 280,70 300,70"
                  fill="none" stroke="#22c55e" strokeWidth="2" opacity="0.8"
                />
                <circle cx="50" cy="105" r="4" fill="#22c55e" />
                <circle cx="300" cy="70" r="4" fill="#22c55e" />

                {/* Minimum dot */}
                <circle cx="200" cy="70" r="3" fill="#f97316" />
                <text x="200" y="60" textAnchor="middle" fill="#f97316" fontSize="9" fontWeight="500">min</text>

                {/* Legend */}
                <line x1="20" y1="130" x2="40" y2="130" stroke="#ef4444" strokeWidth="2" />
                <text x="44" y="133" fill="#ef4444" fontSize="9">Vanilla SGD (zigzag)</text>
                <line x1="200" y1="130" x2="220" y2="130" stroke="#22c55e" strokeWidth="2" />
                <text x="224" y="133" fill="#22c55e" fontSize="9">Momentum (smooth)</text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                Vanilla SGD wastes energy oscillating across the ravine. Momentum cuts straight through.
              </p>
            </div>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5 text-center">
              <p className="text-amber-400 font-semibold">
                What if the ball could remember where it&apos;s been going?
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Ravines Exist">
            Different parameters in a neural network have different sensitivities.
            One weight might need tiny adjustments while another needs large ones.
            This creates loss landscapes where the curvature varies dramatically
            across directions—a ravine.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. EMA Primer */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Tool We Need: Exponential Moving Average"
            subtitle="A way to smooth noisy signals"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before we fix SGD, we need one mathematical tool: the{' '}
              <strong>exponential moving average</strong> (EMA). Once you see how it works,
              momentum and Adam will make immediate sense.
            </p>
            <p className="text-muted-foreground">
              Imagine you&apos;re forecasting tomorrow&apos;s temperature. You could average
              all past days (slow to react to recent changes), or just use today&apos;s reading
              (too noisy—one cold day doesn&apos;t mean winter). EMA finds the sweet spot:
              mostly trust the recent history, but let today&apos;s reading nudge it.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">
                The EMA formula:
              </p>
              <BlockMath math="v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot x_t" />
              <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
                <li><InlineMath math="v_t" /> = the smoothed value at step <InlineMath math="t" /></li>
                <li><InlineMath math="x_t" /> = the new observation (today&apos;s reading)</li>
                <li><InlineMath math="\beta" /> = how much to trust the history (0.9 = &ldquo;90% old, 10% new&rdquo;)</li>
              </ul>
            </div>

            <p className="text-muted-foreground">
              Let&apos;s trace through a concrete example. Suppose you receive 5 gradient values
              and compute the EMA with <InlineMath math="\beta = 0.9" />, starting
              from <InlineMath math="v_0 = 0" />:
            </p>

            {/* EMA concrete walkthrough table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Step</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">
                      <InlineMath math="x_t" /> (gradient)
                    </th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Computation</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">
                      <InlineMath math="v_t" /> (EMA)
                    </th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 font-mono">1</td>
                    <td className="py-2 px-3 font-mono">2.0</td>
                    <td className="py-2 px-3 font-mono text-xs">0.9(0) + 0.1(2.0)</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.20</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 font-mono">2</td>
                    <td className="py-2 px-3 font-mono">-1.5</td>
                    <td className="py-2 px-3 font-mono text-xs">0.9(0.20) + 0.1(-1.5)</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.03</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 font-mono">3</td>
                    <td className="py-2 px-3 font-mono">3.0</td>
                    <td className="py-2 px-3 font-mono text-xs">0.9(0.03) + 0.1(3.0)</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.33</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 font-mono">4</td>
                    <td className="py-2 px-3 font-mono">-0.5</td>
                    <td className="py-2 px-3 font-mono text-xs">0.9(0.33) + 0.1(-0.5)</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.24</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-mono">5</td>
                    <td className="py-2 px-3 font-mono">2.5</td>
                    <td className="py-2 px-3 font-mono text-xs">0.9(0.24) + 0.1(2.5)</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.47</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Notice: the raw gradients swing wildly (2.0, -1.5, 3.0, -0.5, 2.5), but the EMA
              stays relatively stable. The negative values barely dent it because the positive
              trend dominates. That&apos;s exactly what we want for optimization—a smoothed
              signal that filters out the noise while preserving the trend.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Beta Controls Memory">
            <p className="text-sm">
              <InlineMath math="\beta = 0.9" /> means &ldquo;90% old, 10% new.&rdquo;
            </p>
            <p className="text-sm mt-2">
              Higher <InlineMath math="\beta" /> = smoother, slower to react.
              Lower <InlineMath math="\beta" /> = noisier, quick to react.
            </p>
            <p className="text-sm mt-2">
              Rule of thumb: EMA roughly averages over the
              last <InlineMath math="1/(1-\beta)" /> values. So{' '}
              <InlineMath math="\beta=0.9" /> averages ~10 steps.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Momentum */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Momentum: Give the Ball Some Weight"
            subtitle="Smooth the gradient direction with EMA"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now let&apos;s put EMA to work. Remember{' '}
              <InlineMath math="v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot x_t" />?
              Replace <InlineMath math="x_t" /> with today&apos;s gradient, and you get
              momentum—a smoothed gradient signal instead of the raw, noisy one.
            </p>
            <p className="text-muted-foreground">
              Think about the difference between rolling a tennis ball and a bowling ball
              down a bumpy hill. The tennis ball bounces off every little bump, changing
              direction constantly. The bowling ball has <strong>inertia</strong>—it plows
              through small bumps and maintains its overall direction.
              Remember the mini-batch noise that makes the hill shake? Momentum smooths
              that out—a heavier ball is less affected by vibrations.
            </p>
            <p className="text-muted-foreground">
              That&apos;s exactly what momentum does to gradient descent. Instead of using
              the raw gradient at each step (the tennis ball), we keep a running average
              of past gradients (the bowling ball):
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">SGD with Momentum:</p>
              <BlockMath math="v_t = \beta \cdot v_{t-1} + g_t" />
              <BlockMath math="\theta_{t+1} = \theta_t - \alpha \cdot v_t" />
              <p className="text-xs text-muted-foreground">
                <InlineMath math="v_t" /> is the <strong>velocity</strong> (accumulated
                momentum), <InlineMath math="g_t" /> is the current gradient,{' '}
                <InlineMath math="\beta" /> is typically 0.9.
              </p>
            </div>

            <div className="rounded-md border border-sky-500/30 bg-sky-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-sky-400">Wait—where did the (1-&beta;) go?</strong>{' '}
              The EMA formula was <InlineMath math="v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot x_t" />,
              but classical momentum uses <InlineMath math="v_t = \beta \cdot v_{t-1} + g_t" /> without
              the <InlineMath math="(1-\beta)" /> factor. This is a deliberate variant: classical momentum
              accumulates gradients rather than averaging them, and the scaling difference gets absorbed
              into the learning rate <InlineMath math="\alpha" />. The two forms are mathematically
              equivalent up to that scaling—same idea, slightly different bookkeeping. Adam uses the
              EMA form with <InlineMath math="(1-\beta)" />.
            </div>

            <p className="text-muted-foreground">
              Now revisit the ravine. The gradient across the ravine oscillates: positive,
              negative, positive, negative. When momentum averages these, they{' '}
              <strong>cancel out</strong>. Meanwhile, the gradient along the ravine floor
              consistently points the same direction. When momentum averages these, they{' '}
              <strong>accumulate</strong>.
            </p>

            <ComparisonRow
              left={{
                title: 'Across the ravine',
                color: 'rose',
                items: [
                  'Gradient oscillates: +5, -5, +5, -5...',
                  'EMA: values cancel toward 0',
                  'Momentum dampens the zigzag',
                ],
              }}
              right={{
                title: 'Along the ravine',
                color: 'emerald',
                items: [
                  'Gradient is consistent: +0.1, +0.1, +0.1...',
                  'EMA: values accumulate',
                  'Momentum amplifies the signal',
                ],
              }}
            />

            <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
              <p className="text-violet-400 font-semibold">
                Momentum doesn&apos;t just make training faster. It changes <em>which direction</em> the
                optimizer moves—dampening oscillation and amplifying consistent signal.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not Just Speed">
            A common misconception: &ldquo;momentum just makes training faster.&rdquo;
            That misses the key insight. Momentum changes the <em>direction</em> of travel
            by filtering out the oscillating component. Speed is a side effect—direction
            is the point.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              If the ravine curves to the right at the bottom, what happens to the momentum ball?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about inertia. The ball has been building up velocity in one direction...
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  It <strong>overshoots the curve</strong>. Momentum has inertia—the accumulated
                  velocity keeps pushing in the old direction even after the landscape turns.
                  The ball will swing past the curve and then correct.
                </p>
                <p className="text-muted-foreground">
                  This is a genuine downside of momentum on curved landscapes. Higher{' '}
                  <InlineMath math="\beta" /> makes the overshoot worse (more inertia).
                  It&apos;s the same tradeoff as a real bowling ball—great at going straight,
                  bad at sharp turns.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 6. RMSProp */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="RMSProp: A Volume Knob for Each Parameter"
            subtitle="Adapting the learning rate per parameter"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Momentum solves the direction problem, but there&apos;s a separate issue:
              different parameters can have vastly different gradient magnitudes. Imagine
              a network where the gradient for weight <InlineMath math="w_1" /> is 0.001 and
              the gradient for weight <InlineMath math="w_2" /> is 50. One learning rate
              cannot serve both.
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <GradientCard title="lr = 0.01 for both" color="rose">
                <ul className="space-y-1 text-sm">
                  <li>&bull; <InlineMath math="w_1" /> step: 0.01 &times; 0.001 = 0.00001 (crawls)</li>
                  <li>&bull; <InlineMath math="w_2" /> step: 0.01 &times; 50 = 0.5 (overshoots)</li>
                </ul>
              </GradientCard>
              <GradientCard title="What we want" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>&bull; <InlineMath math="w_1" />: bigger effective learning rate</li>
                  <li>&bull; <InlineMath math="w_2" />: smaller effective learning rate</li>
                  <li>&bull; Each parameter gets its own &ldquo;volume knob&rdquo;</li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Think of it like mixing a song. Each instrument (parameter) gets its own volume
              control (learning rate). You don&apos;t turn up the master volume and hope for the
              best—you adjust each channel individually. RMSProp does this automatically.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">RMSProp update:</p>
              <BlockMath math="s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot g_t^2" />
              <BlockMath math="\theta_{t+1} = \theta_t - \frac{\alpha \cdot g_t}{\sqrt{s_t} + \epsilon}" />
              <p className="text-xs text-muted-foreground">
                <InlineMath math="s_t" /> tracks the EMA of <strong>squared</strong> gradients—a
                measure of how large gradients have been recently. Dividing
                by <InlineMath math="\sqrt{s_t}" /> normalizes the update: large gradients get
                divided by a large number (smaller step), small gradients by a small number
                (larger step). <InlineMath math="\epsilon" /> (typically 10<sup>-8</sup>)
                prevents division by zero.
              </p>
            </div>

            <p className="text-muted-foreground">
              Let&apos;s trace through one step with concrete numbers. Start
              from <InlineMath math="s_0 = 0" />, <InlineMath math="\beta = 0.999" />,{' '}
              <InlineMath math="\alpha = 0.01" />:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Parameter</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium"><InlineMath math="g_t" /></th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium"><InlineMath math="s_1 = 0.001 \cdot g_t^2" /></th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Step = <InlineMath math="\alpha \cdot g_t / \sqrt{s_1}" /></th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 font-mono"><InlineMath math="w_1" /></td>
                    <td className="py-2 px-3 font-mono">0.001</td>
                    <td className="py-2 px-3 font-mono text-xs">0.001 &times; 0.000001 = 10<sup>-9</sup></td>
                    <td className="py-2 px-3 font-mono font-semibold">0.01 &times; 0.001 / 3.2&times;10<sup>-5</sup> &asymp; <span className="text-emerald-400">0.32</span></td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-mono"><InlineMath math="w_2" /></td>
                    <td className="py-2 px-3 font-mono">50</td>
                    <td className="py-2 px-3 font-mono text-xs">0.001 &times; 2500 = 2.5</td>
                    <td className="py-2 px-3 font-mono font-semibold">0.01 &times; 50 / 1.58 &asymp; <span className="text-emerald-400">0.32</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Despite 50,000x different gradient magnitudes, both parameters take steps of
              similar size. RMSProp is <strong>self-normalizing</strong>. Parameters with
              consistently large gradients automatically get a smaller effective learning rate.
              Parameters with small gradients automatically get a larger one. No manual tuning
              per parameter required.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Squared Gradients?">
            We track <InlineMath math="g_t^2" /> (squared gradients) because we care about the{' '}
            <em>magnitude</em> of recent gradients, regardless of sign. A gradient of -50 is
            just as &ldquo;big&rdquo; as +50 for the purpose of scaling the learning rate.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7. Adam */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Adam: The Best of Both Worlds"
            subtitle="Momentum + RMSProp combined"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Momentum smooths the gradient <em>direction</em>. RMSProp normalizes the
              gradient <em>magnitude</em> per parameter. What if we combine both ideas?
              That&apos;s Adam (&ldquo;Adaptive Moment Estimation&rdquo;).
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">Adam update:</p>
              <BlockMath math="m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad \text{(momentum)}" />
              <BlockMath math="v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad \text{(RMSProp)}" />
              <BlockMath math="\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}" />
              <BlockMath math="\theta_{t+1} = \theta_t - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}" />
            </div>

            <p className="text-muted-foreground">
              Each line adds one piece to the vanilla SGD update. The bias correction
              step fixes a cold-start problem: since <InlineMath math="m_0 = 0" /> and{' '}
              <InlineMath math="v_0 = 0" />, the first few estimates are too small. Dividing
              by <InlineMath math="1 - \beta^t" /> compensates—as <InlineMath math="t" /> grows,
              this correction fades to 1.
            </p>

            <p className="text-muted-foreground font-semibold">
              Standard defaults:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><InlineMath math="\alpha = 0.001" /> (learning rate)</li>
              <li><InlineMath math="\beta_1 = 0.9" /> (momentum decay)</li>
              <li><InlineMath math="\beta_2 = 0.999" /> (RMSProp decay)</li>
              <li><InlineMath math="\epsilon = 10^{-8}" /></li>
            </ul>

            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">These defaults are starting points, not universal solutions.</strong>{' '}
              &ldquo;Adaptive&rdquo; does not mean &ldquo;no tuning needed.&rdquo; The learning rate
              still matters enormously—Adam with a bad LR will diverge or stall just like SGD. The
              beta values matter for specific domains (e.g., <InlineMath math="\beta_2 = 0.98" /> is
              common for transformers). Treat these defaults as a first experiment, not a final answer.
            </div>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-amber-400 font-semibold mb-2">
                The learning rate is NOT the same across optimizers
              </p>
              <p className="text-sm text-muted-foreground">
                A common mistake: using the same learning rate for SGD and Adam. But Adam
                internally rescales gradients, so its effective step size is very different.
                Typical values: <InlineMath math="\alpha = 0.01" /> for SGD
                vs <InlineMath math="\alpha = 0.001" /> for Adam. Plugging SGD&apos;s learning
                rate into Adam will often diverge.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Side-by-Side Summary">
            <p className="text-sm"><strong>SGD:</strong> step = <InlineMath math="\alpha \cdot g_t" /></p>
            <p className="text-sm mt-1"><strong>+Momentum:</strong> smooth <InlineMath math="g_t" /> with EMA</p>
            <p className="text-sm mt-1"><strong>+RMSProp:</strong> divide by <InlineMath math="\sqrt{\text{EMA}(g_t^2)}" /></p>
            <p className="text-sm mt-1"><strong>Adam:</strong> both + bias correction</p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 8. Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Optimizer Explorer"
            subtitle="Compare optimizer trajectories on the ravine landscape"
          >
            <OptimizerExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Guided Experiments">
            <ol className="space-y-2 text-sm list-decimal list-inside">
              <li>Run <strong>Vanilla SGD</strong>. Watch the zigzag across the ravine.</li>
              <li>Switch to <strong>Momentum</strong>. Watch the path smooth out and cut through the ravine.</li>
              <li>Try <strong>RMSProp</strong>. Notice how it adapts to the ravine shape differently—shorter steps in the steep direction.</li>
              <li>Now try <strong>Adam</strong>. It combines both effects—smooth AND adaptive.</li>
              <li>Increase the learning rate. Which optimizer handles it most gracefully?</li>
            </ol>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 9. Elaborate — "Adam Always Beats SGD" misconception */}
      <Row>
        <Row.Content>
          <SectionHeader
            title='The "Adam Always Wins" Myth'
            subtitle="Faster convergence does not mean better generalization"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Adam is a great default—it converges quickly and requires minimal tuning.
              So why doesn&apos;t everyone just use Adam for everything?
            </p>
            <p className="text-muted-foreground">
              Because converging faster to <em>a</em> minimum doesn&apos;t mean finding
              the <em>best</em> minimum. Research has consistently shown that SGD with momentum
              and a tuned learning rate schedule often <strong>generalizes better</strong> than
              Adam on well-understood problems—especially in computer vision.
            </p>

            <ComparisonRow
              left={{
                title: 'Adam',
                color: 'cyan',
                items: [
                  'Converges fast (fewer epochs)',
                  'Good defaults, less tuning needed',
                  'Adaptive LR navigates into tight valleys',
                  'Can converge to sharper minima',
                  '3x memory per parameter (stores m, v)',
                ],
              }}
              right={{
                title: 'SGD + Momentum',
                color: 'emerald',
                items: [
                  'Slower convergence (more epochs)',
                  'Requires LR schedule tuning',
                  'Gradient noise bounces out of sharp valleys',
                  'Often finds wider, better-generalizing minima',
                  'Lower memory footprint',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Remember sharp vs wide minima from Batching and SGD? Adam&apos;s adaptive
              learning rate lets it navigate into tight spaces that SGD would bounce out of.
              That&apos;s great for convergence speed, but those tight spaces are often
              sharp minima that don&apos;t generalize well.
            </p>

            <div className="rounded-md border border-violet-500/30 bg-violet-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-violet-400">In practice:</strong> Use Adam as your default when
              starting a new problem. If you&apos;re pushing for the best possible performance on a
              well-understood task, try SGD + momentum with a cosine learning rate schedule. Many
              state-of-the-art vision models still use SGD + momentum.
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="No Free Lunch">
            More sophisticated does not mean universally better. Adam has more things
            that <em>can</em> go wrong (3 hyperparameters instead of 1). Simpler optimizers
            are easier to debug.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 10. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              You&apos;re training a new model. Adam converges in 50 epochs but SGD
              hasn&apos;t converged after 100. Your colleague says &ldquo;Adam is clearly
              better.&rdquo; What might they be missing?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about convergence speed vs final quality. What should you actually compare?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  Convergence speed is not the same as final quality. Check the{' '}
                  <strong>validation loss</strong>, not just the training loss. SGD might end
                  up at a better minimum if given enough time with a good learning rate schedule.
                </p>
                <p className="text-muted-foreground">
                  Also: 100 epochs with SGD might not be enough. SGD often needs 3-5x more
                  epochs than Adam to converge, but the final model can be better. The fair
                  comparison is at their best final performance, not at the same number of epochs.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Real Metric">
            Training loss tells you how well the model fits the training data. Validation
            loss tells you how well it generalizes. Always compare optimizers on validation
            performance.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Momentum smooths gradient direction',
                description:
                  'EMA of gradients dampens oscillation across ravines and amplifies consistent signal along them. The bowling ball analogy: inertia plows through noise.',
              },
              {
                headline: 'RMSProp adapts learning rate per parameter',
                description:
                  'EMA of squared gradients normalizes update size. Parameters with large gradients get smaller steps; parameters with small gradients get larger steps.',
              },
              {
                headline: 'Adam = Momentum + RMSProp',
                description:
                  'Combines both ideas with bias correction. A great default (lr=0.001, beta1=0.9, beta2=0.999), but not always the best final answer.',
              },
              {
                headline: 'No free lunch: faster convergence ≠ better generalization',
                description:
                  'Adam converges faster but can settle in sharper minima. SGD + momentum with a tuned schedule often generalizes better on well-understood problems.',
              },
              {
                headline: 'Learning rate means different things for different optimizers',
                description:
                  'lr=0.01 for SGD and lr=0.001 for Adam produce similar-scale updates. Don’t swap one into the other.',
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
                title: 'Adam: A Method for Stochastic Optimization',
                authors: 'Kingma & Ba, 2014',
                url: 'https://arxiv.org/abs/1412.6980',
                note: 'Introduced Adam, the most widely used optimizer in deep learning, combining momentum and adaptive learning rates.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 12. Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/training-dynamics"
            title="Next: Training Dynamics"
            description="Now you know how to optimize. Next: why training sometimes fails completely—vanishing gradients, exploding gradients, and the initialization strategies that prevent them."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
