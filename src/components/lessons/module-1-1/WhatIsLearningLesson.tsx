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
  ComparisonRow,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { OverfittingWidget } from '@/components/widgets/OverfittingWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'

/**
 * Lesson 1.1.1: What is Learning?
 *
 * This foundational lesson establishes the mental model for machine learning:
 * - ML as function approximation
 * - Generalization vs memorization
 * - Bias-variance tradeoff (intuitive)
 * - Train/val/test splits
 */

export function WhatIsLearningLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="What is Learning?"
            description="Understand machine learning as function approximation and why generalization matters."
            category="Fundamentals"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Build a clear mental model for what machine learning actually does —
            and why it&apos;s different from traditional programming.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why This Matters">
            Everything else in deep learning builds on these concepts. Get them
            right now and the rest will click faster.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Core Insight */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Core Insight"
            subtitle="What does a machine actually 'learn'?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Traditional programming: You write rules. &quot;If input is X, output Y.&quot;
            </p>
            <p className="text-muted-foreground">
              Machine learning: You provide examples. The machine figures out the rules.
            </p>
            <p className="text-muted-foreground">
              More precisely: <strong>machine learning is function approximation</strong>.
              You have some unknown function that maps inputs to outputs. You can&apos;t
              see the function directly, but you have examples of input-output pairs.
              The goal is to find a function that produces the same outputs for the same
              inputs — and, crucially, works on new inputs you&apos;ve never seen.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Function Approximation">
            Think of it like this: there&apos;s a &quot;true&quot; function out there
            (e.g., &quot;is this email spam?&quot;). We can&apos;t write it explicitly,
            but we can approximate it by learning from examples.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: Generalization vs Memorization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Generalization vs Memorization"
            subtitle="The difference between learning and cheating"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A model could simply memorize all the training examples. See a dog?
              Remember it&apos;s a dog. Perfect accuracy on training data!
            </p>
            <p className="text-muted-foreground">
              But that&apos;s useless. The whole point is to handle{' '}
              <strong>new data you&apos;ve never seen</strong>.
            </p>
            <p className="text-muted-foreground">
              <strong>Generalization</strong> = the ability to perform well on new,
              unseen data. This is what we actually care about.
            </p>
          </div>
          <ComparisonRow
            left={{
              title: 'Memorization',
              color: 'rose',
              items: [
                'Perfect on training data',
                'Fails on new data',
                'Learned the noise, not the pattern',
                'Like cramming answers before a test',
              ],
            }}
            right={{
              title: 'Generalization',
              color: 'emerald',
              items: [
                'Good on training AND new data',
                'Learned the underlying pattern',
                'Can handle variations',
                'Like understanding the concept',
              ],
            }}
          />
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Think About It">
            Imagine you&apos;re studying for a test. If you memorize every practice
            problem exactly, you might fail when the wording changes. But if you
            understand the concept, you can solve new problems too.
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: Bias-Variance Tradeoff */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bias-Variance Tradeoff"
            subtitle="Finding the sweet spot"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When a model fails to generalize, it&apos;s usually one of two problems:
            </p>

            <div className="grid gap-4 md:grid-cols-2">
              <ConceptBlock title="High Bias (Underfitting)">
                <p>
                  The model is too simple. It can&apos;t capture the pattern even in
                  training data. Like trying to fit a curve with a straight line.
                </p>
                <p className="mt-2 font-medium text-foreground">
                  Sign: Bad on training data AND test data
                </p>
              </ConceptBlock>

              <ConceptBlock title="High Variance (Overfitting)">
                <p>
                  The model is too complex. It fits the training data perfectly,
                  including the noise. Like connecting every dot with a wiggly line.
                </p>
                <p className="mt-2 font-medium text-foreground">
                  Sign: Great on training, bad on test
                </p>
              </ConceptBlock>
            </div>

            <p className="text-muted-foreground">
              The goal is to find a model complex enough to capture the real pattern,
              but not so complex that it memorizes noise.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Sweet Spot">
            You want a model that&apos;s &quot;just right&quot; — complex enough to
            learn, simple enough to generalize. We&apos;ll see how to find this
            balance throughout the course.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Overfitting vs Underfitting */}
      <Row>
        <Row.Content>
          <ExercisePanel title="See It Visually">
            <OverfittingWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Compare">
            <ul className="space-y-2 text-sm">
              <li>• <strong>Underfitting:</strong> The line misses the curve entirely</li>
              <li>• <strong>Good Fit:</strong> Captures the pattern, ignores noise</li>
              <li>• <strong>Overfitting:</strong> Wiggles through every point</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Train/Val/Test */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Train, Validation, Test"
            subtitle="Why three datasets?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              To detect if we&apos;re overfitting, we need to check performance on
              data the model hasn&apos;t seen. This leads to splitting our data:
            </p>

            <div className="space-y-3">
              <ConceptBlock title="Training Set (~70%)">
                The model learns from this. Like the textbook you study.
              </ConceptBlock>

              <ConceptBlock title="Validation Set (~15%)">
                Check progress during training. Like practice tests — you can peek
                and adjust your studying. Used to tune hyperparameters.
              </ConceptBlock>

              <ConceptBlock title="Test Set (~15%)">
                Final evaluation ONCE. Like the real exam — you only see it once.
                Never use it to make training decisions.
              </ConceptBlock>
            </div>

            <p className="text-muted-foreground">
              <strong>Critical rule:</strong> The test set must stay untouched until
              you&apos;re completely done. If you peek at it during training,
              you&apos;ll subtly optimize for it and lose your honest evaluation.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Common Mistake">
            &quot;I&apos;ll just check the test set once to see how I&apos;m doing.&quot;
            Nope! Each peek biases your decisions. The test set is sacred.
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
                headline: 'ML is function approximation.',
                description:
                  'We learn an approximate function from examples, not write rules explicitly.',
              },
              {
                headline: 'Generalization is the goal.',
                description:
                  'We care about performance on new data, not training data.',
              },
              {
                headline: 'Bias-variance tradeoff.',
                description:
                  'Models can be too simple (underfit) or too complex (overfit).',
              },
              {
                headline: 'Three data splits.',
                description:
                  'Train to learn, validate to tune, test to evaluate (once!).',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/linear-regression"
            title="Ready for the next lesson?"
            description="Now that you understand what learning means, let's see the simplest possible example: linear regression."
            buttonText="Continue to Linear Regression"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
