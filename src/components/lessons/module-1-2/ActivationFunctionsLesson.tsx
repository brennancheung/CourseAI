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
  ComparisonRow,
  GradientCard,
} from '@/components/lessons'
import { ActivationFunctionExplorer } from '@/components/widgets/ActivationFunctionExplorer'
import { XORTransformationWidget } from '@/components/widgets/XORTransformationWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Activation Functions - The Missing Ingredient
 *
 * Follows "The Neuron" and "Limits of Linearity" lessons.
 * This lesson adds activation functions and shows XOR being solved.
 *
 * Key concepts:
 * - Activation functions add nonlinearity after each neuron
 * - One neuron = one decision boundary
 * - Common activations: ReLU, sigmoid
 * - With activations, two neurons can solve XOR
 *
 * Next: Activation Functions Reference (visual guide to all activations)
 */

export function ActivationFunctionsLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Activation Functions: The Missing Ingredient"
            description="Add the nonlinearity that transforms our limited linear network into something that can solve XOR—and much more."
            category="From Linear to Neural"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Add the missing ingredient—the activation function—and finally
            solve XOR. Then explore the different activation functions available.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Payoff">
            We&apos;ve built up to this: neurons, layers, networks, the XOR
            problem. Now we add ONE ingredient and everything works.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section: The Fix - Adding Activation */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Fix: Add Nonlinearity"
            subtitle="One small change, huge difference"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have neurons that compute <InlineMath math="y = w \cdot x + b" />.
              We can stack them into layers and networks. But stacking linear
              operations is still linear—that&apos;s why XOR was impossible.
            </p>
            <p className="text-muted-foreground">
              The fix is simple: after each linear computation, apply a{' '}
              <strong>nonlinear function</strong>:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{output} = \sigma(w \cdot x + b)" />
            </div>
            <p className="text-muted-foreground">
              The function <InlineMath math="\sigma" /> is called the{' '}
              <strong>activation function</strong>. It&apos;s applied to each
              neuron&apos;s output, breaking the linearity.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Full Neuron">
            <p className="mb-2">Now a neuron computes:</p>
            <p className="font-mono text-sm">output = σ(w·x + b)</p>
            <p className="mt-2 text-sm">
              Linear combination, <em>then</em> activation.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section: Why Not Just Use Two Lines? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Wait—Why Not Just Use Two Lines?"
            subtitle="The crucial insight"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You might be thinking: &quot;If one line can&apos;t solve XOR, why
              not just use two lines?&quot; Good intuition! But there&apos;s a catch.
            </p>
            <p className="text-muted-foreground">
              Suppose we have two neurons computing two lines, then combine them:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-mono text-muted-foreground">
                h₁ = A + B - 0.5 <span className="text-blue-400">(line 1)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                h₂ = A + B - 1.5 <span className="text-blue-400">(line 2)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                output = w₁·h₁ + w₂·h₂ + b
              </p>
            </div>
            <p className="text-muted-foreground">
              But expand that output and you get:
            </p>
            <div className="py-4 px-6 bg-rose-500/10 rounded-lg border border-rose-500/20">
              <p className="text-sm font-mono">
                output = (w₁+w₂)A + (w₁+w₂)B + constant = <strong className="text-rose-400">ONE LINE!</strong>
              </p>
            </div>
            <p className="text-muted-foreground">
              Linearly combining linear functions always gives you another linear
              function. No matter how many neurons you stack, without activation
              functions it all collapses to <strong>one line</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="The Collapse">
            This is exactly what we saw in the Neurons lesson: linear × linear = linear.
            The &quot;two lines&quot; idea is right, but we need a way to combine
            them <em>non-linearly</em>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section: Activation Functions Enable Thresholds */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Activation Functions Enable Thresholds"
            subtitle="How two lines actually combine"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s what activation functions actually do. Consider ReLU applied
              to our two neurons:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-mono text-muted-foreground">
                h₁ = <span className="text-emerald-400">ReLU</span>(A + B - 0.5)
                <span className="text-blue-400 ml-2">fires when at least one input ≈ 1</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                h₂ = <span className="text-emerald-400">ReLU</span>(A + B - 1.5)
                <span className="text-blue-400 ml-2">fires only when both inputs ≈ 1</span>
              </p>
            </div>
            <p className="text-muted-foreground">
              Now h₁ is <strong>zero</strong> when A+B {'<'} 0.5, and positive
              otherwise. And h₂ is <strong>zero</strong> unless A+B {'>'} 1.5
              (both inputs are 1).
            </p>
            <p className="text-muted-foreground">
              The output neuron gives <strong>positive weight</strong> to h₁
              and <strong>negative weight</strong> to h₂. Since h₂ only turns
              on when both inputs are 1, the negative weight penalizes
              exactly that case. The result: positive output only when at
              least one input is 1 but not both. That&apos;s XOR!
            </p>
            <div className="py-4 px-6 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
              <p className="text-sm">
                <strong className="text-emerald-400">The key:</strong> Activation
                functions create <em>thresholds</em> that can&apos;t be collapsed.
                ReLU&apos;s &quot;zero below, linear above&quot; behavior is fundamentally
                non-linear—h₂ is completely silent for three of the four
                inputs, then &quot;turns on&quot; for (1,1). No linear function
                can do that.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Threshold Trick">
            Without activation, h₂ is always a linear mix of A and B.
            With ReLU, h₂ is <em>zero</em> for most inputs and only
            &quot;turns on&quot; for (1,1). That selective silence is
            something no linear function can replicate.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: The Classic - Sigmoid */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Classic: Sigmoid"
            subtitle="The original activation function"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\sigma(x) = \frac{1}{1 + e^{-x}}" />
            </div>
            <p className="text-muted-foreground">
              Sigmoid squashes any input into the range (0, 1). It was the
              original activation function, inspired by biological neurons.
            </p>
            <ComparisonRow
              left={{
                title: 'Pros',
                color: 'emerald',
                items: [
                  'Smooth and differentiable',
                  'Output always between 0 and 1 (useful for yes/no decisions)',
                  'Historically important',
                ],
              }}
              right={{
                title: 'Cons',
                color: 'rose',
                items: [
                  'Vanishing gradients at extremes',
                  'Not zero-centered',
                  'Computationally expensive (exp)',
                ],
              }}
            />
            <p className="text-muted-foreground">
              Today, sigmoid is mainly used in the <strong>output layer</strong> for
              binary classification, where we want a probability. For hidden
              layers, we use better alternatives.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Range (0, 1)">
            No matter what you input—even ±1000—sigmoid outputs a value
            between 0 and 1. Large positive inputs → ~1. Large negative → ~0.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: The Modern Default - ReLU */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Modern Default: ReLU"
            subtitle="Rectified Linear Unit"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{ReLU}(x) = \max(0, x)" />
            </div>
            <p className="text-muted-foreground">
              ReLU is dead simple: if the input is positive, pass it through.
              If negative, output zero. That&apos;s it.
            </p>
            <ComparisonRow
              left={{
                title: 'Pros',
                color: 'emerald',
                items: [
                  'Extremely fast to compute',
                  'No vanishing gradient for positive values',
                  'Sparse activation (zeros help efficiency)',
                ],
              }}
              right={{
                title: 'Cons',
                color: 'rose',
                items: [
                  '"Dying ReLU" - neurons stuck at 0',
                  'Not differentiable at exactly 0',
                  'Not zero-centered',
                ],
              }}
            />
            <p className="text-muted-foreground">
              ReLU is the default choice for most neural networks. It&apos;s fast,
              works well, and is easy to understand.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why ReLU Won">
            In 2012, deep networks trained with ReLU significantly outperformed
            sigmoid-based networks. The simple &quot;max&quot; operation turned out to
            be better than the biologically-inspired sigmoid.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Activation Function Explorer */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Explore Activation Functions"
            subtitle="See how different functions transform inputs"
          />
          <ExercisePanel title="Activation Function Explorer">
            <ActivationFunctionExplorer
              defaultFunction="relu"
              showDerivatives={false}
              visibleFunctions={['linear', 'sigmoid', 'relu']}
            />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Select &quot;Linear&quot; to see the baseline</li>
              <li>• Compare sigmoid&apos;s S-curve with ReLU&apos;s hinge</li>
              <li>• Move the input slider to extreme values</li>
              <li>• Enable derivatives to see gradients</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: What Makes a Good Activation? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Makes a Good Activation Function?"
            subtitle="Now that you've seen two, what do they have in common?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Sigmoid and ReLU look very different, but they share key properties.
              Here&apos;s what makes an activation function work:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="Must Have" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>• <strong>Nonlinear</strong>—otherwise pointless</li>
                  <li>• <strong>Differentiable</strong>—for gradient descent</li>
                  <li>• <strong>Computationally cheap</strong>—applied billions of times</li>
                </ul>
              </GradientCard>
              <GradientCard title="Nice to Have" color="blue">
                <ul className="space-y-1 text-sm">
                  <li>• <strong>Zero-centered</strong>—helps optimization</li>
                  <li>• <strong>No vanishing gradients</strong>—for deep networks</li>
                  <li>• <strong>Sparse activation</strong>—efficiency</li>
                </ul>
              </GradientCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Vanishing Gradients">
            Notice how sigmoid&apos;s curve flattens at the extremes? That means
            near-zero gradients for large inputs. This causes problems in deep
            networks—we&apos;ll explore this more later.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section: XOR Solved */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="XOR: Solved!"
            subtitle="The hidden layer transforms the space"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now for the &quot;aha&quot; moment. Watch what the hidden layer actually does:
              it <strong>moves the points</strong> to new positions where they
              become linearly separable.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Real Trick">
            Neural networks don&apos;t draw multiple lines in your input space.
            They <em>transform</em> the space so that one line is enough.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: XOR Transformation */}
      <Row>
        <Row.Content>
          <ExercisePanel title="The Space Transformation">
            <XORTransformationWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="What You&apos;re Seeing">
            <ul className="space-y-2 text-sm">
              <li>• <strong>Left:</strong> Original XOR—no line works</li>
              <li>• <strong>Right:</strong> After hidden layer—one line works!</li>
              <li>• The hidden layer <em>moved the points</em></li>
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
                headline: 'Two lines alone don\'t help.',
                description:
                  'Without activation, combining two linear neurons gives you one line—the "collapse" problem.',
              },
              {
                headline: 'Activation functions create thresholds.',
                description:
                  'ReLU outputs zero below a line, positive above. This threshold behavior can\'t be collapsed.',
              },
              {
                headline: 'Thresholds let the output neuron discriminate.',
                description:
                  'h₁ fires broadly, h₂ fires only for (1,1). The output neuron weights them to penalize the "both on" case—impossible with pure linearity.',
              },
              {
                headline: 'That\'s how XOR is solved.',
                description:
                  'Positive weight on h₁ (at least one is 1), negative weight on h₂ (both are 1). The network uses selective silence—h₂ being zero—to separate the cases.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/activation-functions-deep-dive"
            title="Want to go deeper?"
            description="Explore all the major activation functions: ReLU variants, GELU, Swish, and when to use each."
            buttonText="Continue to Activation Functions Deep Dive"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
