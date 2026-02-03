'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  WarningBlock,
  TryThisBlock,
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { Exercise } from '@/lib/exercises'
import { GradientDescentExplorer } from '@/components/widgets/GradientDescentExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Lesson 1.1.4: Gradient Descent - Following the Slope
 *
 * This lesson introduces the core optimization algorithm.
 * Key concepts:
 * - Gradient as slope/direction of steepest ascent
 * - Update rule: move opposite to gradient
 * - Learning rate controls step size
 */

export const gradientDescentExercise: Exercise = {
  slug: 'gradient-descent',
  title: 'Gradient Descent: Following the Slope',
  description:
    'Learn the fundamental algorithm for finding minimum loss — following the gradient downhill.',
  category: 'Fundamentals',
  duration: '20 min',
  constraints: [
    'Focus on intuition first',
    'Math is secondary',
    'Watch the animation many times',
  ],
  steps: [
    'Understand the ball-rolling-downhill analogy',
    'Learn what the gradient tells us',
    'See the update rule in action',
    'Experiment with different learning rates',
  ],
  skills: ['gradient-descent', 'optimization'],
  prerequisites: ['loss-functions'],
}

export function GradientDescentLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title={gradientDescentExercise.title}
            description={gradientDescentExercise.description}
            category={gradientDescentExercise.category}
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand gradient descent — the algorithm that finds parameters
            minimizing loss by iteratively moving downhill.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Core Algorithm">
            Gradient descent is THE fundamental optimization algorithm in machine
            learning. Everything else builds on this.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Intuition */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Intuition: Ball Rolling Downhill"
            subtitle="Finding the lowest point"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine you&apos;re blindfolded on a hilly landscape. You want to find
              the lowest point. What do you do?
            </p>
            <p className="text-muted-foreground">
              You feel which way the ground slopes, then take a step downhill.
              Repeat until you stop descending.
            </p>
            <p className="text-muted-foreground">
              That&apos;s gradient descent:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4">
              <li>Check the slope at your current position</li>
              <li>Take a step in the direction that goes downhill</li>
              <li>Repeat until you reach a flat spot (minimum)</li>
            </ol>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Gradient">
            The <strong>gradient</strong> is just a fancy word for &quot;slope&quot;
            (in one dimension) or &quot;direction of steepest ascent&quot; (in multiple
            dimensions). We go the opposite way.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: What is the Gradient? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What is the Gradient?"
            subtitle="The derivative tells us the slope"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember calculus? The derivative of a function at a point tells you
              the slope of the tangent line.
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Positive derivative:</strong> Function is increasing
                (going uphill to the right)
              </li>
              <li>
                <strong>Negative derivative:</strong> Function is decreasing
                (going downhill to the right)
              </li>
              <li>
                <strong>Zero derivative:</strong> Flat spot (could be minimum,
                maximum, or inflection)
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              For the loss function <InlineMath math="L(\theta)" />, the gradient
              <InlineMath math="\nabla L" /> (or <InlineMath math="\frac{dL}{d\theta}" /> in 1D)
              tells us which way is &quot;uphill.&quot;
            </p>
            <p className="text-muted-foreground">
              To go downhill, we move in the <strong>opposite direction</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Notation">
            <p className="mb-2">
              <InlineMath math="\nabla L" /> = &quot;nabla L&quot; = gradient of L
            </p>
            <p>
              In 1D, this is just <InlineMath math="\frac{dL}{d\theta}" />
            </p>
            <p className="mt-2">
              In multiple dimensions, it&apos;s a vector of partial derivatives.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: The Update Rule */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Update Rule"
            subtitle="How we actually move"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The gradient descent update rule is beautifully simple:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\theta_{new} = \theta_{old} - \alpha \nabla L" />
            </div>
            <p className="text-muted-foreground">
              Let&apos;s break this down:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="\theta_{old}" /> — Current parameter value
              </li>
              <li>
                <InlineMath math="\nabla L" /> — Gradient (which way is uphill)
              </li>
              <li>
                <InlineMath math="-" /> — Minus sign (go opposite direction = downhill)
              </li>
              <li>
                <InlineMath math="\alpha" /> — Learning rate (how big a step)
              </li>
              <li>
                <InlineMath math="\theta_{new}" /> — New parameter value after the step
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Minus Sign">
            The key is the minus sign! The gradient points <em>uphill</em>.
            We want to go <em>downhill</em>. So we subtract.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Gradient Descent Animation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h3 className="font-semibold">Watch Gradient Descent in Action</h3>
            <ExercisePanel title="Gradient Descent Animation">
              <GradientDescentExplorer
                initialPosition={-2}
                initialLearningRate={0.3}
                showLearningRateSlider={true}
                showGradientArrow={true}
              />
            </ExercisePanel>
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Click &quot;Step&quot; to see one update at a time</li>
              <li>• Watch the update equation at the bottom</li>
              <li>• Try different learning rates</li>
              <li>• Notice: big LR = big steps, small LR = small steps</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Learning Rate Preview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Learning Rate α"
            subtitle="How big a step to take"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The learning rate <InlineMath math="\alpha" /> controls step size.
              Try adjusting it in the visualization above:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <ConceptBlock title="Too Big (α > 0.7)">
                <p>
                  Takes huge steps. Might overshoot the minimum and bounce around.
                  Can even diverge (get worse!).
                </p>
              </ConceptBlock>
              <ConceptBlock title="Too Small (α < 0.1)">
                <p>
                  Takes tiny steps. Very slow to converge. Might get stuck or take
                  forever to reach minimum.
                </p>
              </ConceptBlock>
            </div>
            <p className="text-muted-foreground mt-4">
              Finding the right learning rate is crucial. We&apos;ll dive deeper
              into this in the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Goldilocks Zone">
            The learning rate must be &quot;just right.&quot; Too big and you overshoot.
            Too small and you take forever. There&apos;s no universal best value — it
            depends on the problem.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Why It Works */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Does This Work?"
            subtitle="The math behind the magic"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Gradient descent works because of a fundamental property of derivatives:
            </p>
            <p className="text-muted-foreground">
              Near any point, the best linear approximation of a function is its
              tangent line. The derivative tells us exactly which direction decreases
              the function <em>fastest</em>.
            </p>
            <p className="text-muted-foreground">
              By taking small steps in the negative gradient direction, we&apos;re
              guaranteed to decrease the loss (as long as our step isn&apos;t too big).
            </p>
            <p className="text-muted-foreground mt-4">
              For convex functions (bowl-shaped, like MSE for linear regression),
              there&apos;s exactly one minimum, and gradient descent will find it.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Convexity Matters">
            MSE for linear regression is <strong>convex</strong> — there&apos;s only
            one minimum. Neural networks are <em>not</em> convex, so there can be
            many local minima. Still, gradient descent works surprisingly well!
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Gradient = direction of steepest ascent.',
                description: 'It tells us which way is "uphill" on the loss surface.',
              },
              {
                headline: 'We go opposite to the gradient.',
                description:
                  'The minus sign makes us go downhill, decreasing loss.',
              },
              {
                headline: 'Learning rate controls step size.',
                description:
                  'Too big = overshoot. Too small = slow. Must be tuned.',
              },
              {
                headline: 'Iterate until convergence.',
                description:
                  'Keep stepping until the gradient is near zero (flat spot).',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/learning-rate"
            title="Ready for the next lesson?"
            description="The learning rate is crucial. Let's explore what happens when it's wrong."
            buttonText="Continue to Learning Rate Deep Dive"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
