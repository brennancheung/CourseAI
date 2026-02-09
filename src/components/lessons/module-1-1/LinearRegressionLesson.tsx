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
  ConceptBlock,
  TryThisBlock,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Lesson 1.1.2: Linear Regression from Scratch
 *
 * This lesson introduces the simplest possible model—a line.
 * Key concepts:
 * - Parameters: slope and intercept (weight and bias)
 * - What "fitting" means
 * - Interactive exploration of line fitting
 */

export function LinearRegressionLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Linear Regression from Scratch"
            description="Understand the simplest machine learning model: fitting a line to data points."
            category="Fundamentals"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand how a simple line can be a &quot;model&quot; and what it
            means to &quot;fit&quot; that line to data.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Building Blocks">
            Every neural network starts with exactly this—a line. Master this
            and you have the fundamental building block.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Scope */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            title="This Lesson"
            items={[
              'The simplest model—no optimization yet',
              'Understanding parameters and fitting intuitively',
              'Does NOT cover how to find the best fit (that\'s next)',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Section 1: The Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="A Prediction Problem"
            subtitle="Can you predict a price from a size?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Imagine you&apos;re selling your apartment. You know the square footage of
              several recent sales and what they sold for. Now you want to predict
              what YOUR apartment is worth based on its size.
            </p>
            <p className="text-muted-foreground">
              This is function approximation in action: you have examples
              (size &rarr; price), and you need to predict for a new input. But what kind
              of function should you use?
            </p>
            <p className="text-muted-foreground">
              The simplest possible answer: <strong>a straight line</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Function Approximation">
            Last lesson: ML is finding an unknown function from examples.
            Now we pick the simplest function to try — a line.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: The Model — Familiar Form */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Simplest Model"
            subtitle="A line through points"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You already know this equation from algebra:
            </p>
            <div className="py-4">
              <BlockMath math="y = mx + b" />
            </div>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><InlineMath math="x" /> is the input (what we know—e.g., square footage)</li>
              <li><InlineMath math="y" /> is the output (what we predict — e.g., price)</li>
              <li><InlineMath math="m" /> is the <strong>slope</strong> (how steep the line is)</li>
              <li><InlineMath math="b" /> is the <strong>intercept</strong> (where the line crosses the y-axis)</li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Parameters">
            <InlineMath math="m" /> and <InlineMath math="b" /> are called{' '}
            <strong>parameters</strong> — the numbers the model
            &quot;learns.&quot; Training means finding the best values for them.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2b: ML Notation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In machine learning, we write the same equation with different names:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\hat{y} = wx + b" />
            </div>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><InlineMath math="w" /> is the <strong>weight</strong> (same as slope—how much the input matters)</li>
              <li><InlineMath math="b" /> is the <strong>bias</strong> (same as intercept—the baseline)</li>
              <li><InlineMath math="\hat{y}" /> (y-hat) is our <strong>prediction</strong></li>
              <li><InlineMath math="y" /> (no hat) is the <strong>actual value</strong></li>
            </ul>
            <p className="text-muted-foreground mt-4">
              Let&apos;s see it work. Using simple numbers (the same principle applies
              to square footage, temperature, or any input): if{' '}
              <InlineMath math="w = 0.5" /> and{' '}
              <InlineMath math="b = 1" />, then for{' '}
              <InlineMath math="x = 2" />:
            </p>
            <div className="py-3 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\hat{y} = 0.5 \times 2 + 1 = 2" />
            </div>
            <p className="text-muted-foreground">
              Our model predicts <InlineMath math="2" />. If the actual value was{' '}
              <InlineMath math="y = 2.3" />, we were off by{' '}
              <InlineMath math="0.3" />. The difference between{' '}
              <InlineMath math="\hat{y}" /> and <InlineMath math="y" /> is the{' '}
              <strong>error</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Hat Matters">
            <InlineMath math="\hat{y}" /> vs <InlineMath math="y" /> — always
            distinguish prediction from truth. The gap between them is the
            <strong> error</strong>.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: What Fitting Means */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Does 'Fitting' Mean?"
            subtitle="Finding the right weight and bias"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have data points — past measurements of inputs and outputs.
              We want to find values for <InlineMath math="w" /> and{' '}
              <InlineMath math="b" /> that make our line pass as close to those
              points as possible. This is called <strong>fitting</strong> the
              line to the data.
            </p>
            <p className="text-muted-foreground">
              Try it yourself — adjust the weight and bias to get the line close
              to all the points:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Try Fitting a Line">
            <LinearFitExplorer
              initialSlope={0.3}
              initialIntercept={-0.5}
              showResiduals={false}
              showMSE={false}
            />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>
                • Drag the <span className="text-orange-400 font-medium">orange</span> and{' '}
                <span className="text-violet-400 font-medium">purple</span> numbers
                in the equation to adjust the line
              </li>
              <li>• Or use the sliders below for fine control</li>
              <li>• Try to get the line as close to all points as possible</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Post-widget insight */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You probably noticed: no matter how you adjust the line, you
              can&apos;t make it pass through every point. That&apos;s not a failure —
              real data is noisy, and a straight line is deliberately simple. The
              gap between the line and each point is the <strong>error</strong> for
              that prediction.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why Not Hit Every Point?">
            A line that bends to hit every single data point would be
            overfitting — remember the wiggly line from last lesson? Simplicity
            is a feature, not a bug.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: The Problem with Guessing */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem with Guessing"
            subtitle="We need to measure 'goodness'"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You probably found a reasonable-looking fit. But how do you{' '}
              <em>know</em> it&apos;s the best? If someone else adjusted the
              sliders differently, how would you compare?
            </p>
            <p className="text-muted-foreground">
              We need a way to <strong>measure</strong> how good a fit is — a
              single number that tells us how wrong our predictions are. That
              number is called the <strong>loss</strong>.
            </p>
            <p className="text-muted-foreground">
              That&apos;s exactly what we&apos;ll build in the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Preview">
            The distance from each point to the line tells us how wrong we are
            for that point. We&apos;ll combine these distances into a single
            number called the <strong>loss</strong>.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            title="Key Takeaways"
            items={[
              {
                headline: 'Linear regression is the simplest model.',
                description: 'A line: ŷ = wx + b. Two parameters: weight (slope) and bias (intercept).',
              },
              {
                headline: 'Parameters are what we learn.',
                description:
                  'The weight and bias are adjusted to fit the data. Training = finding good parameter values.',
              },
              {
                headline: 'A perfect fit isn\'t the goal.',
                description:
                  'A line can\'t pass through every noisy data point — and that\'s by design.',
              },
              {
                headline: 'We need to measure goodness.',
                description:
                  'Eyeballing isn\'t enough. We need a number — the loss — to compare different fits.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/loss-functions"
            title="Ready for the next lesson?"
            description="Now let's define exactly what 'good fit' means with loss functions."
            buttonText="Continue to Loss Functions"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
