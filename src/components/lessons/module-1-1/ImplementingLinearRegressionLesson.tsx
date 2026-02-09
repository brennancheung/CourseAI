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
  ConceptBlock,
  StepList,
  SummaryBlock,
  NextStepBlock,
  ModuleCompleteBlock,
} from '@/components/lessons'
import 'katex/dist/katex.min.css'
import { BlockMath, InlineMath } from 'react-katex'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { TrainingLoopExplorer } from '@/components/widgets/TrainingLoopExplorer'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'

/**
 * Lesson 1.1.6: Implementing Linear Regression
 *
 * This lesson bridges theory to code.
 * Key concepts:
 * - NumPy implementation
 * - Computing gradients by hand
 * - Training loop structure
 * - Interactive training visualization
 */

export function ImplementingLinearRegressionLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Implementing Linear Regression"
            description="Put it all together—implement linear regression with gradient descent from scratch in Python."
            category="Fundamentals"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Implement linear regression with gradient descent from scratch using
            only NumPy—no ML frameworks. Understand every line.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why From Scratch?">
            Frameworks hide the math. Building it yourself ensures you
            understand what&apos;s happening. This foundation helps debug issues
            later.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Motivating Hook */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Over the last five lessons, you&apos;ve learned the pieces: a
              model (<InlineMath math="\hat{y} = wx + b" />), a way to measure
              how wrong it is (MSE loss), a way to improve (gradient descent),
              and how to control the improvement speed (learning rate).
            </p>
            <p className="text-muted-foreground">
              Now we wire them all together and watch a model learn from
              scratch.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Payoff">
            This is the moment where five lessons of theory become a working
            system. Every line of code maps to something you already understand.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Training Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Training Loop"
            subtitle="The universal pattern for training ML models"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every ML training procedure follows this pattern:
            </p>
            <StepList
              title="Training Loop"
              steps={[
                'Initialize parameters (weights and bias) randomly or to zero',
                'Forward pass: Compute predictions using current parameters',
                'Compute loss: How wrong are our predictions?',
                'Backward pass: Compute gradients (how to improve)',
                'Update: Adjust parameters in direction that reduces loss',
                'Repeat steps 2-5 until loss is low enough',
              ]}
            />
            <p className="text-muted-foreground mt-4">
              This loop is the heartbeat of all neural network training. Learn
              it well.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern Everywhere">
            PyTorch, TensorFlow, JAX—all frameworks implement this same loop.
            The details change, but the structure is universal.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: The Math */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Computing Gradients"
            subtitle="The math we need to implement"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We need the gradient for each parameter separately. The
              symbol <InlineMath math="\partial L / \partial w" /> means
              &quot;how much does the loss change when we
              change <InlineMath math="w" />?&quot; It&apos;s the same idea as
              the slope of the loss curve from lesson 4, but now we have one
              slope per parameter.
            </p>

            <div className="space-y-4">
              <ConceptBlock title="The Model">
                <BlockMath math="\hat{y} = wx + b" />
                <p className="mt-2">Prediction is weight times input plus bias.</p>
              </ConceptBlock>

              <ConceptBlock title="The Loss (MSE)">
                <BlockMath math="L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />
                <p className="mt-2">Average squared error.</p>
              </ConceptBlock>

              <ConceptBlock title="Gradient w.r.t. Weight">
                <BlockMath math="\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n} -2x_i(y_i - \hat{y}_i)" />
                <p className="mt-2">How loss changes as weight changes.</p>
              </ConceptBlock>

              <ConceptBlock title="Gradient w.r.t. Bias">
                <BlockMath math="\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n} -2(y_i - \hat{y}_i)" />
                <p className="mt-2">How loss changes as bias changes.</p>
              </ConceptBlock>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Chain Rule">
            These gradients come from applying the chain rule to the MSE
            formula. The key: derivative of squared error flows back through
            the prediction. We&apos;ll unpack this in Module 1.3.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 2b: Worked Example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tracing One Iteration"
            subtitle="What actually happens in one step"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before we see the code, let&apos;s trace through one training
              step with real numbers. Suppose we have just 3 data points and
              start with <InlineMath math="w = 0" />,{' '}
              <InlineMath math="b = 0" />,{' '}
              <InlineMath math="\text{lr} = 0.1" />:
            </p>
            <div className="space-y-3 py-3 px-4 bg-muted/50 rounded-lg text-sm">
              <div>
                <p className="text-muted-foreground font-medium">
                  Data: <InlineMath math="x = [1, 2, 3]" />,{' '}
                  <InlineMath math="y = [3, 5, 7]" />
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">
                  <strong>Forward:</strong>{' '}
                  <InlineMath math="\hat{y} = 0 \cdot [1,2,3] + 0 = [0, 0, 0]" />
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">
                  <strong>Loss:</strong>{' '}
                  <InlineMath math="L = \frac{1}{3}(9 + 25 + 49) = 27.67" />
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">
                  <strong>Gradients:</strong>{' '}
                  <InlineMath math="\frac{\partial L}{\partial w} = \frac{1}{3}(-2 \cdot 1 \cdot 3 + -2 \cdot 2 \cdot 5 + -2 \cdot 3 \cdot 7) = -22.67" />
                </p>
                <p className="text-muted-foreground ml-4">
                  <InlineMath math="\frac{\partial L}{\partial b} = \frac{1}{3}(-6 - 10 - 14) = -10" />
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">
                  <strong>Update:</strong>{' '}
                  <InlineMath math="w = 0 - 0.1 \times (-22.67) = 2.267" />
                </p>
                <p className="text-muted-foreground ml-4">
                  <InlineMath math="b = 0 - 0.1 \times (-10) = 1.0" />
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              One step, and the parameters already jumped from zero toward the
              true values. The negative gradients told us &quot;loss decreases
              when you increase both <InlineMath math="w" /> and{' '}
              <InlineMath math="b" />,&quot; so the update rule increased them.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Negative = Go Up">
            A negative gradient means the loss goes down when the parameter
            goes up. The minus sign in the update rule flips it:{' '}
            <InlineMath math="0 - 0.1 \times (-22.67)" /> becomes addition.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: The Code */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Code"
            subtitle="Everything above in ~15 lines of Python"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Each section of this code maps directly to a step in the training
              loop:
            </p>
            <CodeBlock
              language="python"
              code={`# Generate synthetic data
X, y = generate_data(100)

# Initialize parameters
w, b = 0.0, 0.0
lr = 0.01

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = w * X + b

    # Compute loss
    loss = np.mean((y - y_pred) ** 2)

    # Compute gradients
    dw = np.mean(-2 * X * (y - y_pred))
    db = np.mean(-2 * (y - y_pred))

    # Update parameters
    w = w - lr * dw
    b = b - lr * db`}
            />
            <p className="text-muted-foreground mt-4">
              That&apos;s it. The same pattern you just traced by hand, running
              1000 times. This is the foundation of all deep learning.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Common Bug">
            Watch the signs! The gradient formula has a negative sign from the
            chain rule, and the update rule subtracts. Getting these wrong
            makes the model diverge.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Watch It Learn */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Watch It Learn"
            subtitle="The training loop running in real time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now that you understand every piece, watch the full training loop
              in action. Click &quot;Step&quot; to run one gradient descent
              update, or &quot;Train&quot; to animate:
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Training Loop Explorer">
            <TrainingLoopExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="What to Watch For">
            <ul className="space-y-2 text-sm">
              <li>
                • The <span className="text-red-400">red loss curve</span> drops
                as training progresses
              </li>
              <li>
                • <span className="text-orange-400">w</span> and{' '}
                <span className="text-orange-400">b</span> approach their true
                values in parentheses
              </li>
              <li>
                • The <span className="text-orange-400">orange line</span> moves
                toward the dashed{' '}
                <span className="text-green-400/60">green target</span>
              </li>
              <li>• Try different learning rates and see the speed change</li>
            </ul>
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Post-widget insight */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every neural network—from linear regression to GPT—follows this
              exact pattern. Forward, loss, backward, update. Over and over.
              The model you just watched train is the same process that
              trains billion-parameter language models.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Try It Yourself - Colab */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Implement the training loop in Python"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                You&apos;ve seen it work. Now write the code yourself in a
                Jupyter notebook.
              </p>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/1-1-6-linear-regression.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                The notebook includes exercises: try different learning rates,
                add more noise, use fewer samples. See how each change affects
                training.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Save a Copy">
            Colab opens in read-only mode. Click &quot;Copy to Drive&quot; to
            save your own version that you can edit and experiment with.
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
                headline: 'Training loop: forward, loss, backward, update.',
                description:
                  'This pattern is universal. Every ML model trains this way.',
              },
              {
                headline: 'Gradients tell us which way to adjust.',
                description:
                  'Negative gradient means "increase this parameter to reduce loss."',
              },
              {
                headline: 'The entire implementation is ~15 lines.',
                description:
                  'No magic. Every line maps to a concept you already understand.',
              },
              {
                headline: 'This pattern scales to GPT.',
                description:
                  'Linear regression and billion-parameter models share the same training loop.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="1.1"
            title="The Learning Problem"
            achievements={[
              'ML as function approximation',
              'Generalization vs memorization',
              'Linear models and parameters',
              'Loss functions (MSE)',
              'Gradient descent optimization',
              'Learning rate dynamics',
              'Implementation from scratch',
            ]}
            nextModule="1.2"
            nextTitle="From Linear to Neural"
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/limits-of-linearity"
            title="Ready for neural networks?"
            description="Linear models can only do so much. Let's see their limits and why we need something more."
            buttonText="Continue to Module 1.2"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
