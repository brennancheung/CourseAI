'use client'

import { LessonLayout } from '@/components/lessons/LessonLayout'
import { Row } from '@/components/layout/Row'
import {
  LessonHeader,
  ObjectiveBlock,
  SectionHeader,
  InsightBlock,
  TipBlock,
  TryThisBlock,
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
} from '@/components/lessons'
import { SingleNeuronExplorer } from '@/components/widgets/SingleNeuronExplorer'
import { NetworkDiagramWidget } from '@/components/widgets/NetworkDiagramWidget'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * The Neuron - Building Block of Neural Networks
 *
 * First lesson in "From Linear to Neural" module.
 * Key insight: a neuron without activation is just linear regression!
 *
 * This lesson covers:
 * - What a neuron computes (weighted sum + bias)
 * - Connection to linear regression
 * - Multiple neurons = a layer
 * - Stacking layers = a network
 * - The limitation: it's all still linear
 *
 * Next: The Limits of Linearity (XOR problem)
 */

export function NeuronBasicsLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Neuron: Building Block of Neural Networks"
            description="Discover that a neuron is just weighted sum + bias — the same linear regression you already know!"
            category="From Linear to Neural"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand what a neuron actually computes — and realize it&apos;s
            something you already know from Module 1.1.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Good News">
            If you understood linear regression, you already understand 90% of
            what a neuron does. The rest is just connecting them together.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section: What is a Neuron? */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What is a Neuron?"
            subtitle="The fundamental unit of neural networks"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A <strong>neuron</strong> (also called a <em>unit</em> or <em>node</em>)
              takes some inputs, combines them, and produces an output:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="y = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b" />
            </div>
            <p className="text-muted-foreground">
              That&apos;s it. Each input <InlineMath math="x_i" /> gets multiplied by
              a <strong>weight</strong> <InlineMath math="w_i" />, everything gets
              added together, and we add a <strong>bias</strong> <InlineMath math="b" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Wait...">
            Does this formula look familiar? It should!
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: It's Just Linear Regression! */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="It's Just Linear Regression!"
            subtitle="You already know this"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Compare the neuron formula to what you learned in Module 1.1:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="p-4 rounded-lg border bg-muted/30">
                <p className="font-medium text-blue-400 mb-2">Linear Regression</p>
                <p className="font-mono text-sm">y = wx + b</p>
                <p className="text-xs text-muted-foreground mt-2">
                  One input, one weight, one bias
                </p>
              </div>
              <div className="p-4 rounded-lg border bg-muted/30">
                <p className="font-medium text-emerald-400 mb-2">Neuron</p>
                <p className="font-mono text-sm">y = w₁x₁ + w₂x₂ + ... + b</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Multiple inputs, multiple weights, one bias
                </p>
              </div>
            </div>
            <p className="text-muted-foreground">
              A neuron is just <strong>multi-input linear regression</strong>. Instead
              of one input feature, it takes many. But the computation is identical:
              weighted sum plus bias.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Weights = Parameters">
            The weights and bias are the <strong>learnable parameters</strong>.
            During training, the network adjusts these values to minimize error —
            exactly like gradient descent in linear regression.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Single Neuron Explorer */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Explore a Single Neuron">
            <SingleNeuronExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Set w₁=1, w₂=0, b=0 — now y = x₁</li>
              <li>• Try w₁=0.5, w₂=0.5 — averaging inputs</li>
              <li>• What happens with negative weights?</li>
              <li>• How does the bias shift the output?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: Multiple Neurons = A Layer */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Multiple Neurons = A Layer"
            subtitle="Combining neurons together"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              One neuron computes one output. But what if we want multiple outputs?
              We use multiple neurons, each with its own weights:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-mono">
                <span className="text-blue-400">Neuron 1:</span> y₁ = w₁₁x₁ + w₁₂x₂ + b₁
              </p>
              <p className="text-sm font-mono">
                <span className="text-emerald-400">Neuron 2:</span> y₂ = w₂₁x₁ + w₂₂x₂ + b₂
              </p>
              <p className="text-sm font-mono">
                <span className="text-orange-400">Neuron 3:</span> y₃ = w₃₁x₁ + w₃₂x₂ + b₃
              </p>
            </div>
            <p className="text-muted-foreground">
              A group of neurons that all receive the same inputs is called a{' '}
              <strong>layer</strong>. Each neuron in the layer computes a different
              weighted combination of the inputs.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Matrix Form">
            This is just matrix multiplication! A layer with 3 neurons taking 2
            inputs is a 3×2 weight matrix times the input vector, plus a bias vector.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section: Stacking Layers = A Network */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Stacking Layers = A Network"
            subtitle="The 'deep' in deep learning"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now for the key idea: we can <strong>stack layers</strong>. The output
              of one layer becomes the input to the next:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center gap-4 text-sm">
                <div className="text-center">
                  <div className="font-medium mb-1">Input</div>
                  <div className="text-muted-foreground">(x₁, x₂)</div>
                </div>
                <div className="text-muted-foreground">→</div>
                <div className="text-center">
                  <div className="font-medium mb-1 text-blue-400">Layer 1</div>
                  <div className="text-muted-foreground">3 neurons</div>
                </div>
                <div className="text-muted-foreground">→</div>
                <div className="text-center">
                  <div className="font-medium mb-1 text-emerald-400">Layer 2</div>
                  <div className="text-muted-foreground">2 neurons</div>
                </div>
                <div className="text-muted-foreground">→</div>
                <div className="text-center">
                  <div className="font-medium mb-1">Output</div>
                  <div className="text-muted-foreground">(y₁, y₂)</div>
                </div>
              </div>
            </div>
            <p className="text-muted-foreground">
              This is a <strong>neural network</strong>! The layers between input
              and output are called <strong>hidden layers</strong>. A network with
              many hidden layers is called a <strong>deep</strong> neural network.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why 'Hidden'?">
            We don&apos;t directly observe the values in hidden layers — only the
            final output. The hidden layers learn internal representations that
            help solve the task.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Network Diagram */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Watch Data Flow Through a Network">
            <NetworkDiagramWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Observe">
            <ul className="space-y-2 text-sm">
              <li>• Change the inputs and watch values propagate</li>
              <li>• Each hidden neuron computes something different</li>
              <li>• The outputs depend on ALL the hidden values</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: The Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="But There's a Problem..."
            subtitle="Linear layers are still linear"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s the catch: if each neuron just computes a weighted sum,
              then <strong>stacking layers doesn&apos;t add any power</strong>.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm text-muted-foreground">
                Layer 1: <InlineMath math="h = W_1 x + b_1" />
              </p>
              <p className="text-sm text-muted-foreground">
                Layer 2: <InlineMath math="y = W_2 h + b_2" />
              </p>
              <p className="text-sm text-muted-foreground">
                Combined: <InlineMath math="y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)" />
              </p>
            </div>
            <p className="text-muted-foreground">
              The result is still just <InlineMath math="y = Wx + b" /> — a single
              linear transformation! We could collapse any number of linear layers
              into one.
            </p>
            <p className="text-muted-foreground font-medium">
              A 100-layer linear network is no more powerful than a 1-layer linear
              network.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Linear × Linear = Linear">
            This is the fundamental limitation. To make deep networks useful, we
            need something that <em>breaks</em> linearity. That&apos;s coming in
            the next lessons.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section: What We Have So Far */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What We Have So Far"
            subtitle="The structure without the magic"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We now understand the <strong>architecture</strong> of neural networks:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Neuron:</strong> Weighted sum of inputs plus bias
              </li>
              <li>
                <strong>Layer:</strong> Multiple neurons processing the same inputs
              </li>
              <li>
                <strong>Network:</strong> Layers stacked together, output → input
              </li>
              <li>
                <strong>Parameters:</strong> All the weights and biases (learned during training)
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              But right now, this whole structure is just a fancy way of doing
              linear regression. To unlock the true power of neural networks, we
              need one more ingredient.
            </p>
            <p className="text-muted-foreground">
              First, let&apos;s see exactly <em>why</em> linearity is a problem...
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Coming Up">
            Next: A simple problem that linear models <em>cannot</em> solve, no
            matter how many layers you stack.
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
                headline: 'A neuron = weighted sum + bias.',
                description:
                  'It\'s just multi-input linear regression. You already know this!',
              },
              {
                headline: 'Layers group neurons together.',
                description:
                  'Each neuron computes a different weighted combination of the inputs.',
              },
              {
                headline: 'Networks stack layers.',
                description:
                  'The output of one layer feeds into the next. Hidden layers learn internal representations.',
              },
              {
                headline: 'But linear layers collapse.',
                description:
                  'Stacking linear operations is still linear. We need something more.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/limits-of-linearity"
            title="See the Limitation in Action"
            description="Next, we'll encounter a simple problem that exposes exactly why linear networks aren't enough."
            buttonText="Continue to The Limits of Linearity"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
