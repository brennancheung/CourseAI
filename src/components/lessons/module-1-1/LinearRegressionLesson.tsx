'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  ConceptBlock,
  TryThisBlock,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { Exercise } from '@/lib/exercises'
import { LinearFitExplorer } from '@/components/widgets/LinearFitExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Lesson 1.1.2: Linear Regression from Scratch
 *
 * This lesson introduces the simplest possible model — a line.
 * Key concepts:
 * - Parameters: slope and intercept
 * - What "fitting" means
 * - Interactive exploration of line fitting
 */

export const linearRegressionExercise: Exercise = {
  slug: 'linear-regression',
  title: 'Linear Regression from Scratch',
  description:
    'Understand the simplest machine learning model: fitting a line to data points.',
  category: 'Fundamentals',
  duration: '20 min',
  constraints: [
    'Focus on intuition first',
    'Math is secondary to understanding',
    'Play with the interactive widget',
  ],
  steps: [
    'Understand what a linear model is',
    'Learn what parameters (slope, intercept) do',
    'Explore fitting interactively',
    'See why we need a way to measure "goodness" of fit',
  ],
  skills: ['linear-models', 'parameters'],
  prerequisites: ['what-is-learning'],
}

export function LinearRegressionLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title={linearRegressionExercise.title}
            description={linearRegressionExercise.description}
            category={linearRegressionExercise.category}
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
            Linear regression is the foundation. Neural networks are just many
            linear operations combined with non-linear activations.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Simplest Model */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Simplest Model"
            subtitle="A line through points"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember: machine learning is function approximation. What&apos;s the
              simplest function we can use?
            </p>
            <p className="text-muted-foreground">
              A straight line. In math terms:
            </p>
            <div className="py-4">
              <BlockMath math="y = mx + b" />
            </div>
            <p className="text-muted-foreground">
              Where:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><InlineMath math="x" /> is the input (what we know)</li>
              <li><InlineMath math="y" /> is the output (what we predict)</li>
              <li><InlineMath math="m" /> is the <strong>slope</strong> (how steep the line is)</li>
              <li><InlineMath math="b" /> is the <strong>intercept</strong> (where the line crosses y-axis)</li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Parameters">
            <InlineMath math="m" /> and <InlineMath math="b" /> are called{' '}
            <strong>parameters</strong>. These are the numbers the model
            &quot;learns&quot;. The goal of training is to find the best values
            for these parameters.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: What Fitting Means */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Does 'Fitting' Mean?"
            subtitle="Finding the right slope and intercept"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have data points. We want to find a line that goes through them
              as well as possible. This is called &quot;fitting&quot; the line to
              the data.
            </p>
            <p className="text-muted-foreground">
              But what does &quot;as well as possible&quot; mean? We need to define
              it precisely. For now, let&apos;s just use our intuition: we want
              the line close to all the points.
            </p>
            <p className="text-muted-foreground">
              Try it yourself:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Interactive Widget */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h3 className="font-semibold">Try Fitting a Line</h3>
            <ExercisePanel title="Try Fitting a Line">
              <LinearFitExplorer
                initialSlope={0.3}
                initialIntercept={-0.5}
                showResiduals={false}
                showMSE={false}
              />
            </ExercisePanel>
          </div>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Drag the <strong>intercept</strong> point up and down</li>
              <li>• Drag the <strong>slope</strong> point to change the angle</li>
              <li>• Try to get the line as close to all points as possible</li>
              <li>• Notice: you can&apos;t hit every point perfectly!</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: The Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem with Guessing"
            subtitle="We need to measure 'goodness'"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You probably found a reasonable-looking fit. But how do you know
              it&apos;s the best? How would you compare two different fits?
            </p>
            <p className="text-muted-foreground">
              We need a way to <strong>measure</strong> how good a fit is — a
              single number that tells us how wrong our predictions are.
            </p>
            <p className="text-muted-foreground">
              This brings us to <strong>loss functions</strong>, which we&apos;ll
              cover in the next lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Preview">
            The distance from each point to the line tells us how wrong we are for
            that point. We&apos;ll combine these distances into a single number
            called the <strong>loss</strong>.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Notation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ML Notation"
            subtitle="How we'll write this going forward"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In machine learning, we typically write the same equation slightly
              differently:
            </p>
            <div className="py-4">
              <BlockMath math="\hat{y} = wx + b" />
            </div>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li><InlineMath math="\hat{y}" /> (y-hat) is our <strong>prediction</strong></li>
              <li><InlineMath math="y" /> (without hat) is the <strong>actual value</strong></li>
              <li><InlineMath math="w" /> is the <strong>weight</strong> (same as slope)</li>
              <li><InlineMath math="b" /> is the <strong>bias</strong> (same as intercept)</li>
            </ul>
            <p className="text-muted-foreground mt-4">
              Why &quot;weight&quot; and &quot;bias&quot;? These terms generalize better to
              neural networks. A neural network has many weights, but the concept
              is the same: learnable parameters that control the output.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Hat Matters">
            <InlineMath math="\hat{y}" /> vs <InlineMath math="y" /> — always
            distinguish prediction from truth. The difference between them is the
            <strong> error</strong>.
          </TipBlock>
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
                description: 'A line: ŷ = wx + b. Two parameters: weight and bias.',
              },
              {
                headline: 'Parameters are what we learn.',
                description:
                  'The weight (slope) and bias (intercept) are adjusted to fit the data.',
              },
              {
                headline: '"Fitting" means finding good parameters.',
                description:
                  'We want the line close to the data points.',
              },
              {
                headline: 'We need to measure goodness.',
                description:
                  'Visual estimation is not enough. We need a number: the loss.',
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
