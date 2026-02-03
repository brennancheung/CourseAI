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
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { Exercise } from '@/lib/exercises'
import { LearningRateExplorer } from '@/components/widgets/LearningRateExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Lesson 1.1.5: Learning Rate Deep Dive
 *
 * This lesson explores the critical hyperparameter α.
 * Key concepts:
 * - What happens when LR is too big/small
 * - Overshooting and divergence
 * - Learning rate schedules (preview)
 */

export const learningRateExercise: Exercise = {
  slug: 'learning-rate',
  title: 'Learning Rate Deep Dive',
  description:
    'Understand the most important hyperparameter in gradient descent — the learning rate.',
  category: 'Fundamentals',
  duration: '15 min',
  constraints: [
    'Focus on developing intuition',
    'Experiment with different values',
    'Watch the behaviors carefully',
  ],
  steps: [
    'See what happens with learning rate too small',
    'See what happens with learning rate too large',
    'Understand overshooting and divergence',
    'Preview learning rate schedules',
  ],
  skills: ['learning-rate', 'hyperparameters'],
  prerequisites: ['gradient-descent'],
}

export function LearningRateLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title={learningRateExercise.title}
            description={learningRateExercise.description}
            category={learningRateExercise.category}
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Develop intuition for how the learning rate affects training —
            what happens when it&apos;s too big, too small, or just right.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Hyperparameter">
            The learning rate is a <strong>hyperparameter</strong> — a value
            YOU choose, not one the model learns. Choosing good hyperparameters
            is a major part of ML in practice.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Goldilocks Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Goldilocks Problem"
            subtitle="Not too big, not too small"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The learning rate <InlineMath math="\alpha" /> controls how big a
              step we take each iteration:
            </p>
            <div className="py-4">
              <BlockMath math="\theta_{new} = \theta_{old} - \alpha \nabla L" />
            </div>
            <p className="text-muted-foreground">
              If <InlineMath math="\alpha" /> is too small, we take tiny steps
              and converge very slowly.
            </p>
            <p className="text-muted-foreground">
              If <InlineMath math="\alpha" /> is too large, we take huge steps
              and might overshoot the minimum — or even diverge!
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="No Universal Answer">
            There&apos;s no single &quot;best&quot; learning rate. It depends on
            the problem, the loss surface shape, and even where you start.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Side by Side Comparison */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h3 className="font-semibold">Compare Learning Rates</h3>
            <LearningRateExplorer mode="comparison" />
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Observe">
            <ul className="space-y-2 text-sm">
              <li>• <strong>Too Small (0.05):</strong> Watch how slow it is</li>
              <li>• <strong>Just Right (0.5):</strong> Smooth, fast convergence</li>
              <li>• <strong>Too Large (0.9):</strong> See the oscillation</li>
              <li>• <strong>Way Too Large (1.1):</strong> Diverges completely!</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: Too Small */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Too Small: Slow Convergence"
            subtitle="When patience runs out before training finishes"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              With a very small learning rate (like 0.01 or 0.001):
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Each step barely moves the parameters</li>
              <li>Convergence takes many, many iterations</li>
              <li>Training is slow and expensive</li>
              <li>Might get &quot;stuck&quot; in shallow local minima</li>
            </ul>
            <p className="text-muted-foreground mt-4">
              This isn&apos;t catastrophic — you&apos;ll eventually converge. But you&apos;ll
              waste time and compute.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="When Small is OK">
            Sometimes a small learning rate is intentional — especially for
            fine-tuning pretrained models where you don&apos;t want to disturb the
            learned weights too much.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: Too Large */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Too Large: Oscillation and Divergence"
            subtitle="When the cure is worse than the disease"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              With a learning rate that&apos;s too large:
            </p>

            <ComparisonRow
              left={{
                title: 'Oscillation',
                color: 'orange',
                items: [
                  'Steps overshoot the minimum',
                  'Bounces back and forth',
                  'May eventually settle down',
                  'Wastes computation',
                ],
              }}
              right={{
                title: 'Divergence',
                color: 'rose',
                items: [
                  'Each step makes things worse',
                  'Loss increases instead of decreasing',
                  'Parameters grow without bound',
                  'Training completely fails',
                ],
              }}
            />

            <p className="text-muted-foreground mt-4">
              The critical threshold depends on the loss surface curvature.
              Steeper surfaces require smaller learning rates.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Divergence is Bad">
            If your loss suddenly shoots up to NaN or infinity, learning rate
            is often the culprit. Try reducing it by a factor of 10.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Interactive Exploration */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <SectionHeader
              title="Find the Sweet Spot"
              subtitle="Experiment with different learning rates"
            />
            <ExercisePanel title="Find the Sweet Spot">
              <LearningRateExplorer mode="interactive" />
            </ExercisePanel>
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Try These">
            <ul className="space-y-2 text-sm">
              <li>• Set α = 0.1 and run (slow but stable)</li>
              <li>• Set α = 0.5 and run (fast and smooth)</li>
              <li>• Set α = 1.0 and run (right at the edge)</li>
              <li>• Set α = 1.05 and run (diverges!)</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Learning Rate Schedules */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Learning Rate Schedules (Preview)"
            subtitle="Why use one value when you can use many?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In practice, we often <strong>change</strong> the learning rate
              during training:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Start high:</strong> Take big steps to make fast progress
              </li>
              <li>
                <strong>Decay over time:</strong> Smaller steps for fine-tuning
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              Common schedules:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><strong>Step decay:</strong> Reduce by factor every N epochs</li>
              <li><strong>Exponential decay:</strong> Smooth exponential decrease</li>
              <li><strong>Cosine annealing:</strong> Smooth cosine-shaped decay</li>
              <li><strong>Warmup:</strong> Start small, ramp up, then decay</li>
            </ul>
            <p className="text-muted-foreground mt-4 text-sm">
              We&apos;ll cover schedules in detail in Module 1.7: Practical Training.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Modern Practice">
            Most modern training uses learning rate schedules. The &quot;warmup + cosine
            decay&quot; pattern is especially popular for transformer training.
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
                headline: 'Learning rate controls step size.',
                description: 'It multiplies the gradient to determine how far to move.',
              },
              {
                headline: 'Too small = slow convergence.',
                description: 'Training works but takes forever.',
              },
              {
                headline: 'Too large = oscillation or divergence.',
                description: 'Overshooting the minimum, possibly getting worse.',
              },
              {
                headline: 'Schedules change LR during training.',
                description: 'Start large for speed, decay for precision.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/implementing-linear-regression"
            title="Ready to code?"
            description="Time to implement everything we've learned in Python — linear regression from scratch."
            buttonText="Continue to Implementation"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
