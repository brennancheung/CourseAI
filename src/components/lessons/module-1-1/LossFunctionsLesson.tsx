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
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { Exercise } from '@/lib/exercises'
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'
import { LossSurfaceExplorer } from '@/components/widgets/LossSurfaceExplorer'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Lesson 1.1.3: Loss Functions - Measuring "Wrongness"
 *
 * This lesson introduces loss functions as the way to measure model quality.
 * Key concepts:
 * - Residuals: the difference between predicted and actual
 * - MSE: Mean Squared Error
 * - Loss landscape visualization
 */

export const lossFunctionsExercise: Exercise = {
  slug: 'loss-functions',
  title: 'Loss Functions: Measuring "Wrongness"',
  description:
    'Learn how we quantify how wrong our model is using loss functions.',
  category: 'Fundamentals',
  duration: '20 min',
  constraints: [
    'Understand the intuition first',
    'Then the math',
    'Play with both visualizations',
  ],
  steps: [
    'Understand residuals (prediction errors)',
    'Learn why we square the errors (MSE)',
    'See the loss as a landscape',
    'Understand why we need to minimize loss',
  ],
  skills: ['loss-functions', 'mse'],
  prerequisites: ['linear-regression'],
}

export function LossFunctionsLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title={lossFunctionsExercise.title}
            description={lossFunctionsExercise.description}
            category={lossFunctionsExercise.category}
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how we measure model quality with a single number — the loss —
            and why Mean Squared Error is the go-to choice.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why This Matters">
            The loss function is what we minimize during training. Choose the wrong
            loss and your model learns the wrong thing.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: Residuals */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Residuals: Measuring Individual Errors"
            subtitle="How wrong is each prediction?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For each data point, we can measure how far off our prediction is:
            </p>
            <div className="py-4">
              <BlockMath math="\text{residual}_i = y_i - \hat{y}_i" />
            </div>
            <p className="text-muted-foreground">
              This is called the <strong>residual</strong> (or error) for point <InlineMath math="i" />.
              It&apos;s simply the actual value minus the predicted value.
            </p>
            <p className="text-muted-foreground">
              Try moving the line and watch the red residual lines change:
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Residual = Error">
            Positive residual: we predicted too low.
            Negative residual: we predicted too high.
            Zero: perfect prediction for that point.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Residuals */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h3 className="font-semibold">Visualizing Residuals</h3>
            <LinearFitExplorer
              initialSlope={0.3}
              initialIntercept={-0.5}
              showResiduals={true}
              showMSE={true}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Watch the red lines — those are residuals</li>
              <li>• Try to make all red lines as short as possible</li>
              <li>• Notice the MSE number changing</li>
              <li>• Can you get the MSE below 0.5?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: Why Square? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Mean SQUARED Error?"
            subtitle="The reasoning behind squaring"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We could just add up all the residuals. But there&apos;s a problem:
              positive and negative errors cancel out! A prediction 5 too high and
              5 too low would average to 0 error.
            </p>
            <p className="text-muted-foreground">
              Two common solutions:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Mean Absolute Error (MAE):</strong> Take absolute value
                <InlineMath math="|y_i - \hat{y}_i|" />
              </li>
              <li>
                <strong>Mean Squared Error (MSE):</strong> Square it
                <InlineMath math="(y_i - \hat{y}_i)^2" />
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              MSE is more popular because:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>Squaring is smooth and differentiable (important for calculus)</li>
              <li>Large errors get penalized more (10² = 100, not just 10)</li>
              <li>The math works out nicely for optimization</li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Large Errors Hurt">
            Because of squaring, one very wrong prediction hurts more than two
            slightly wrong predictions. This is usually what we want!
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: MSE Formula */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The MSE Formula"
            subtitle="Putting it all together"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Mean Squared Error for <InlineMath math="n" /> data points:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />
            </div>
            <p className="text-muted-foreground">
              Let&apos;s break this down:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <InlineMath math="y_i - \hat{y}_i" /> — the residual for point <InlineMath math="i" />
              </li>
              <li>
                <InlineMath math="(\cdot)^2" /> — square it (makes all values positive, penalizes large errors)
              </li>
              <li>
                <InlineMath math="\sum" /> — sum over all points
              </li>
              <li>
                <InlineMath math="\frac{1}{n}" /> — divide by count to get the mean
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              The result <InlineMath math="L" /> is a single number: the <strong>loss</strong>.
              Smaller is better. Zero means perfect predictions on all points.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Remember">
            <InlineMath math="L" /> is our &quot;wrongness score&quot;.
            Training a model means finding parameters that make <InlineMath math="L" /> as small as possible.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Loss Landscape */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Loss Landscape"
            subtitle="Visualizing loss as a surface"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s a powerful mental model: imagine plotting the loss for
              <em> every possible</em> combination of slope and intercept.
            </p>
            <p className="text-muted-foreground">
              You get a <strong>loss surface</strong> — a landscape where:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>X-axis: slope (<InlineMath math="w" />)</li>
              <li>Y-axis: intercept (<InlineMath math="b" />)</li>
              <li>Height: loss (<InlineMath math="L" />)</li>
            </ul>
            <p className="text-muted-foreground mt-4">
              Training is finding the lowest point on this landscape — the valley
              where loss is minimized.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Valley">
            For linear regression with MSE, the loss surface is always a
            &quot;bowl&quot; shape (convex). There&apos;s exactly one lowest point,
            and we can always find it.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Loss Surface */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h3 className="font-semibold">Explore the Loss Surface</h3>
            <LossSurfaceExplorer />
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Drag the point on the surface</li>
              <li>• Watch the line and MSE change</li>
              <li>• Find the bottom of the bowl</li>
              <li>• Notice: there&apos;s exactly one minimum</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Residual = actual - predicted.',
                description: 'How wrong we are for each individual point.',
              },
              {
                headline: 'MSE squares and averages errors.',
                description:
                  'Makes all errors positive and penalizes large errors more.',
              },
              {
                headline: 'Loss is a single number.',
                description:
                  'It tells us how wrong our model is overall. Lower is better.',
              },
              {
                headline: 'Loss landscape = all possible losses.',
                description:
                  'Training is finding the lowest point on this surface.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/gradient-descent"
            title="Ready for the next lesson?"
            description="Now that we can measure wrongness, how do we find the parameters that minimize it? Enter gradient descent."
            buttonText="Continue to Gradient Descent"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
