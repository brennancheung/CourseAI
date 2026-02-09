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
  ComparisonRow,
  WarningBlock,
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
            Build a clear mental model for what machine learning actually does—and why it&apos;s different from traditional programming.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why This Matters">
            Everything else in deep learning builds on these concepts. Get them
            right now and the rest will click faster.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Scope */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            title="This Lesson"
            items={[
              'Pure intuition—no math, no code',
              'Building the mental model for what ML is',
              'Does NOT cover specific algorithms or how training works',
            ]}
          />
        </Row.Content>
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
              Imagine you&apos;re building a spam filter. You could try writing rules:
              &quot;If subject contains &apos;FREE MONEY&apos;, it&apos;s spam.&quot;
              But spammers adapt. They write &quot;FR3E M0NEY&quot; or skip the subject
              line entirely. Every new trick requires a new rule. You&apos;re playing
              whack-a-mole, and the moles are winning.
            </p>
            <p className="text-muted-foreground">
              This is the fundamental problem with traditional programming: you write
              rules, but the real world has too many edge cases to enumerate.
            </p>
            <p className="text-muted-foreground">
              Machine learning flips this around. Instead of writing rules, you provide
              thousands of examples—emails labeled &quot;spam&quot; or &quot;not spam&quot;—and the machine figures out the rules itself.
            </p>
            <p className="text-muted-foreground">
              More precisely: <strong>machine learning is function approximation</strong>.
              There&apos;s some unknown function that maps inputs (emails) to outputs
              (spam or not). You can&apos;t see this function directly, but you have
              examples of input-output pairs. The goal is to find a function that
              produces the same outputs—and, crucially, works on new emails
              you&apos;ve never seen.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Function Approximation">
            The same pattern applies everywhere: house prices (input = features,
            output = price), medical diagnosis (input = symptoms, output = condition),
            image recognition (input = pixels, output = label). In each case,
            there&apos;s a &quot;true&quot; function we can&apos;t write by hand but can
            approximate from examples.
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
              A model could simply memorize all the training examples. Show it a
              photo of a golden retriever? It remembers: &quot;that&apos;s a dog.&quot;
              Perfect accuracy on training data!
            </p>
            <p className="text-muted-foreground">
              But now show it a golden retriever from a different angle, in different
              lighting, or a breed it&apos;s never seen before. It&apos;s completely
              lost—it only memorized specific images, not what makes a dog a dog.
            </p>
            <p className="text-muted-foreground">
              This is why <strong>generalization</strong>—the ability to perform
              well on new, unseen data—is what we actually care about. A model that
              memorizes is useless. A model that generalizes has learned the real
              pattern.
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
                  training data. Like trying to fit a curve with a straight line —
                  the model has a strong <strong>bias</strong> toward simplicity and
                  can&apos;t bend enough to match reality.
                </p>
                <p className="mt-2 font-medium text-foreground">
                  Sign: Bad on training data AND test data
                </p>
              </ConceptBlock>

              <ConceptBlock title="High Variance (Overfitting)">
                <p>
                  The model is too complex. It fits the training data perfectly,
                  including the noise. Like connecting every dot with a wiggly line —
                  train it on slightly different data and you&apos;d get a wildly
                  different result. That instability is <strong>variance</strong>.
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
            You want a model that&apos;s &quot;just right&quot;—complex enough to
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

      {/* Why each fit matters — tied back to memorization vs generalization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Goes Wrong"
            subtitle="Connecting what you just saw to memorization and generalization"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the distinction between memorization and generalization? What you
              just saw in the widget is exactly that tradeoff playing out visually.
            </p>

            <ConceptBlock title="Underfitting—Hasn't Learned Yet">
              The model hasn&apos;t captured the pattern yet. It&apos;s not memorizing
              and it&apos;s not generalizing—it&apos;s just wrong. Even the examples it
              trained on get bad predictions, and new data is no better.
            </ConceptBlock>

            <ConceptBlock title="Good Fit—Actual Generalization">
              The model learned the real pattern, not the individual data points.
              It won&apos;t be perfect on training data—and that&apos;s a good sign.
              It means the model is capturing what&apos;s actually there instead of
              memorizing what&apos;s not.
            </ConceptBlock>

            <ConceptBlock title="Overfitting—Memorization in Disguise">
              This is the memorization problem made visible. The model memorized every
              training point—including the quirks—instead of learning the real
              pattern. It looks perfect on training data, which is what makes it
              dangerous: you think it&apos;s working, but show it new data and the
              predictions fall apart.
            </ConceptBlock>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Pattern">
            Underfitting: hasn&apos;t learned yet.
            Good fit: actually generalized.
            Overfitting: memorized instead of learned.
          </InsightBlock>
          <WarningBlock title="The Perfect Fit Trap">
            Your instinct will be to fit the data as closely as possible—if
            loss measures error, zero error must be the goal, right? But
            imagine a stock trading strategy that perfectly predicts last
            year&apos;s prices. Every peak, every dip—nailed. Run it on
            tomorrow&apos;s market and it falls apart. A perfect fit on the
            past is not a good predictor of the future.
          </WarningBlock>
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
              So overfitting means great performance on training data but poor
              performance in the real world. How do we actually detect that? We
              can&apos;t wait until deployment to find out. We need to simulate
              &quot;the real world&quot; by holding back some data the model never
              gets to train on. This leads to splitting our data:
            </p>

            <div className="space-y-3">
              <ConceptBlock title="Training Set (~70%)">
                The model learns from this. Like the textbook you study.
              </ConceptBlock>

              <ConceptBlock title="Validation Set (~15%)">
                Check progress during training. Like practice tests—you can peek
                and adjust your approach. Used to make decisions like &quot;is my
                model too simple or too complex?&quot;
              </ConceptBlock>

              <ConceptBlock title="Test Set (~15%)">
                Final evaluation ONCE. Like the real exam—you only see it once.
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
          <WarningBlock title="Why Not Train on Everything?">
            <ul className="space-y-1">
              <li>• More data is generally <em>good</em>—it helps generalize</li>
              <li>• The problem isn&apos;t that more data hurts the model</li>
              <li>• The problem is you lose your ability to <em>measure</em> whether it works</li>
            </ul>
          </WarningBlock>
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
