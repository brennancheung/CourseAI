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
} from '@/components/lessons'
import { Exercise } from '@/lib/exercises'
import 'katex/dist/katex.min.css'
import { BlockMath } from 'react-katex'
import { ExternalLink, Code2 } from 'lucide-react'

/**
 * Lesson 1.1.6: Implementing Linear Regression
 *
 * This lesson bridges theory to code.
 * Key concepts:
 * - NumPy implementation
 * - Computing gradients by hand
 * - Training loop structure
 * - Google Colab notebook
 */

export const implementingLinearRegressionExercise: Exercise = {
  slug: 'implementing-linear-regression',
  title: 'Implementing Linear Regression',
  description:
    'Put it all together — implement linear regression with gradient descent from scratch in Python.',
  category: 'Fundamentals',
  duration: '30 min',
  constraints: [
    'Pure Python + NumPy only',
    'No sklearn or PyTorch',
    'Compute every gradient by hand',
  ],
  steps: [
    'Understand the training loop structure',
    'Derive the gradient formulas',
    'Implement in Python',
    'Watch the model learn',
  ],
  skills: ['python', 'numpy', 'implementation'],
  prerequisites: ['learning-rate'],
}

export function ImplementingLinearRegressionLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title={implementingLinearRegressionExercise.title}
            description={implementingLinearRegressionExercise.description}
            category={implementingLinearRegressionExercise.category}
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Implement linear regression with gradient descent from scratch using
            only NumPy — no ML frameworks. Understand every line.
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

      {/* Colab Link */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                <Code2 className="w-6 h-6 text-primary" />
              </div>
              <div className="space-y-3">
                <div>
                  <h3 className="font-semibold text-lg">Open the Notebook</h3>
                  <p className="text-sm text-muted-foreground">
                    This lesson is built around a hands-on Colab notebook. Open it
                    to follow along and run the code yourself.
                  </p>
                </div>
                <a
                  href="https://colab.research.google.com/github/YOUR_USERNAME/CourseAI-notebooks/blob/main/1-1-6-linear-regression.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Google Colab
                </a>
                <p className="text-xs text-muted-foreground">
                  The notebook will open in Colab. Save a copy to your Drive to
                  edit and run the cells.
                </p>
              </div>
            </div>
          </div>
        </Row.Content>
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
              This loop is the heartbeat of all neural network training. Learn it well.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Pattern Everywhere">
            PyTorch, TensorFlow, JAX — all frameworks implement this same loop.
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
              For linear regression with MSE loss, here are the gradient formulas:
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
            formula. In the notebook, we derive them step by step.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: The Code */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Code Structure"
            subtitle="What you'll implement"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the notebook, you&apos;ll implement:
            </p>
            <div className="rounded-lg bg-muted/50 p-4 font-mono text-sm space-y-2">
              <p className="text-muted-foreground"># Generate synthetic data</p>
              <p>X, y = generate_data(100)</p>
              <p></p>
              <p className="text-muted-foreground"># Initialize parameters</p>
              <p>w, b = 0.0, 0.0</p>
              <p>lr = 0.01</p>
              <p></p>
              <p className="text-muted-foreground"># Training loop</p>
              <p>for epoch in range(1000):</p>
              <p className="pl-4"># Forward pass</p>
              <p className="pl-4">y_pred = w * X + b</p>
              <p className="pl-4"></p>
              <p className="pl-4"># Compute loss</p>
              <p className="pl-4">loss = np.mean((y - y_pred) ** 2)</p>
              <p className="pl-4"></p>
              <p className="pl-4"># Compute gradients</p>
              <p className="pl-4">dw = np.mean(-2 * X * (y - y_pred))</p>
              <p className="pl-4">db = np.mean(-2 * (y - y_pred))</p>
              <p className="pl-4"></p>
              <p className="pl-4"># Update parameters</p>
              <p className="pl-4">w = w - lr * dw</p>
              <p className="pl-4">b = b - lr * db</p>
            </div>
            <p className="text-muted-foreground mt-4">
              That&apos;s it! This simple loop is the foundation of all deep
              learning.
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

      {/* Section 4: What You'll See */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What You'll Observe"
            subtitle="The model learning in real time"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              When you run the notebook, you&apos;ll see:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Loss decreasing:</strong> The MSE drops from high to low
              </li>
              <li>
                <strong>Parameters converging:</strong> w and b approach their
                optimal values
              </li>
              <li>
                <strong>The line fitting:</strong> Animated plot shows the line
                getting better
              </li>
              <li>
                <strong>Learning rate effects:</strong> Try different LRs and
                see convergence change
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              This is the &quot;aha moment&quot; where theory becomes real. Watch
              the numbers change and the line move.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="That's Deep Learning">
            Every neural network — from linear regression to GPT — follows this
            exact pattern. Forward, loss, backward, update. Over and over.
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
                headline: 'Training loop: forward, loss, backward, update.',
                description: 'This pattern is universal in ML.',
              },
              {
                headline: 'Gradients tell us which way to adjust.',
                description: 'Derived from chain rule applied to loss.',
              },
              {
                headline: 'NumPy handles the vector math.',
                description: 'Operations work on entire arrays at once.',
              },
              {
                headline: 'Watch the model learn.',
                description: 'Seeing loss drop and parameters converge builds intuition.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module Complete */}
      <Row>
        <Row.Content>
          <div className="rounded-lg bg-gradient-to-br from-emerald-500/10 via-emerald-500/5 to-transparent border border-emerald-500/20 p-6">
            <h3 className="font-semibold text-lg text-emerald-400 mb-2">
              🎉 Module 1.1 Complete!
            </h3>
            <p className="text-muted-foreground mb-4">
              You&apos;ve learned the core concepts of the learning problem:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 mb-4">
              <li>ML as function approximation</li>
              <li>Generalization vs memorization</li>
              <li>Linear models and parameters</li>
              <li>Loss functions (MSE)</li>
              <li>Gradient descent optimization</li>
              <li>Learning rate dynamics</li>
              <li>Implementation from scratch</li>
            </ul>
            <p className="text-muted-foreground">
              Next up: <strong>Module 1.2 — From Linear to Neural</strong>.
              We&apos;ll see why linear models aren&apos;t enough and introduce
              nonlinearity.
            </p>
          </div>
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
