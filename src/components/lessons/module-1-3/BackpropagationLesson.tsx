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
  WarningBlock,
  ConceptBlock,
  SummaryBlock,
  NextStepBlock,
  PhaseCard,
  ReferencesBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { BackpropFlowExplorer } from '@/components/widgets/BackpropFlowExplorer'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Backpropagation - How Neural Networks Learn
 *
 * First lesson in the "Training Neural Networks" module.
 *
 * Key concepts:
 * - The chain rule from calculus
 * - Forward pass: computing the output
 * - Backward pass: computing gradients
 * - How gradients flow through layers
 *
 * This lesson focuses on intuition and the "why" before diving into math.
 */

export function BackpropagationLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Backpropagation: How Networks Learn"
            description="Understand how neural networks compute gradients — the ‘backward pass’ that makes learning possible."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand backpropagation — the algorithm that computes how each
            parameter in a neural network should change to reduce loss.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Core of Deep Learning">
            Backpropagation is what makes deep learning work. Without it, we
            couldn&apos;t train networks with millions of parameters.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 1: The Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Problem: Many Parameters"
            subtitle="How do we know which way to adjust each one?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In linear regression, we had two parameters: weight and bias.
              Computing gradients was straightforward.
            </p>
            <p className="text-muted-foreground">
              But a neural network has <strong>thousands to billions</strong> of
              parameters. Each weight in each layer affects the final output.
              How do we figure out how to adjust them all?
            </p>
            <p className="text-muted-foreground">
              We need a way to compute:{' '}
              <InlineMath math="\frac{\partial L}{\partial w}" /> for every
              single weight <InlineMath math="w" /> in the network. The{' '}
              <InlineMath math="\partial" /> symbol means &quot;partial
              derivative&quot;—how does <InlineMath math="L" /> change when
              we nudge just this one weight, holding everything else fixed?
            </p>
            <p className="text-muted-foreground">
              That&apos;s what <strong>backpropagation</strong> does — it computes
              all these gradients efficiently in one backward pass.
              Backpropagation isn&apos;t a separate learning algorithm from
              gradient descent. It&apos;s the computation step <em>inside</em>{' '}
              gradient descent that figures out which direction to go.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Scale">
            GPT-3 has 175 billion parameters. Backpropagation computes a gradient
            for every single one, every training step. The algorithm is remarkably
            efficient.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: The Chain Rule */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Key Idea: Chain Rule"
            subtitle="Calculus to the rescue"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the chain rule from calculus? If{' '}
              <InlineMath math="y = f(g(x))" />, then:
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}" />
            </div>
            <p className="text-muted-foreground">
              In words: the rate of change of <InlineMath math="y" /> with
              respect to <InlineMath math="x" /> equals the rate of change of{' '}
              <InlineMath math="y" /> with respect to <InlineMath math="g" />,
              times the rate of change of <InlineMath math="g" /> with respect
              to <InlineMath math="x" />.
            </p>
            <p className="text-muted-foreground">
              <strong>This is the entire idea behind backpropagation.</strong>{' '}
              A neural network is just a chain of functions. We can compute how
              the output changes with respect to any input by multiplying
              derivatives along the chain.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Intuition">
            If doubling <InlineMath math="g" /> doubles <InlineMath math="y" />,
            and doubling <InlineMath math="x" /> triples <InlineMath math="g" />,
            then doubling <InlineMath math="x" /> will 6× <InlineMath math="y" />.
            <br /><br />
            Effects multiply through the chain.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: Forward and Backward */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Two Passes: Forward and Backward"
            subtitle="Computing outputs, then computing gradients"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Training involves two passes through the network:
            </p>
            <div className="space-y-4">
              <PhaseCard number={1} title="Forward Pass" subtitle="Input → Output" color="blue">
                <p>
                  Push input through the network layer by layer, computing each
                  intermediate value. Save these values — we&apos;ll need them.
                  End with the prediction and loss.
                </p>
              </PhaseCard>

              <PhaseCard number={2} title="Backward Pass" subtitle="Gradient flows back" color="violet">
                <p>
                  Start from the loss. Work backward through each layer, using
                  the chain rule to compute how much each weight contributed to
                  the loss. This gives us the gradients.
                </p>
              </PhaseCard>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why &apos;Back&apos;?">
            Gradients flow <em>backward</em> from the loss to the inputs.
            That&apos;s why it&apos;s called back-propagation — the error signal
            propagates back through the network.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: A Concrete Example */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="A Concrete Example"
            subtitle="Walking through a simple network"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&apos;s trace through the simplest possible network: one neuron
              with one input.
            </p>
            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-mono text-muted-foreground">
                <span className="text-blue-400">Forward:</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4">
                z = w·x + b <span className="text-muted-foreground/60">(linear combination)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4">
                a = ReLU(z) <span className="text-muted-foreground/60">(activation)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4">
                L = (y - a)² <span className="text-muted-foreground/60">(loss)</span>
              </p>
            </div>
            <p className="text-muted-foreground">
              Now for the backward pass. We want{' '}
              <InlineMath math="\frac{\partial L}{\partial w}" /> — how does
              changing <InlineMath math="w" /> affect the loss?
            </p>
            <div className="py-4 px-6 bg-violet-500/10 rounded-lg border border-violet-500/20 space-y-2">
              <p className="text-sm font-mono text-violet-300">
                <span className="text-violet-400">Backward (chain rule):</span>
              </p>
              <p className="text-sm font-mono text-violet-300 ml-4">
                ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4 mt-3">
                <span className="text-violet-400">Step 1:</span>{' '}
                ∂L/∂a = <InlineMath math="-2(y-a)" />{' '}
                <span className="text-muted-foreground/60">(loss derivative)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4">
                <span className="text-violet-400">Step 2:</span>{' '}
                ∂a/∂z ={' '}
                <span className="text-violet-300">1 if z {'>'} 0, else 0</span>{' '}
                <span className="text-muted-foreground/60">(ReLU derivative)</span>
              </p>
              <p className="text-sm font-mono text-muted-foreground ml-4">
                <span className="text-violet-400">Step 3:</span>{' '}
                ∂z/∂w = <InlineMath math="x" />{' '}
                <span className="text-muted-foreground/60">(linear derivative)</span>
              </p>
              <p className="text-sm font-mono text-violet-300 ml-4 mt-2">
                Multiply: ∂L/∂w = <InlineMath math="-2(y-a)" /> ·{' '}
                <span className="text-violet-300">(1 if z{'>'} 0, else 0)</span> ·{' '}
                <InlineMath math="x" />
              </p>
            </div>
            <p className="text-muted-foreground">
              Each term is the local derivative at that step. We multiply them
              together to get the gradient of the loss with respect to{' '}
              <InlineMath math="w" />.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Local × Local × Local">
            Each layer only needs to know its own local derivative and the
            gradient arriving from the layer above. Layer 2 doesn&apos;t need
            to know anything about layer 1&apos;s weights — it just passes
            the gradient backward. This locality is why backprop scales.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Backprop Flow Explorer */}
      <Row>
        <Row.Content>
          <ExercisePanel title="See Gradients Flow">
            <BackpropFlowExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>• Click <strong>Step</strong> to advance one node at a time</li>
              <li>• Click <strong>Run</strong> to watch the full forward → backward → update loop</li>
              <li>• Watch the loss decrease over epochs as weights update</li>
              <li>• Try x={'<'}0 to see ReLU block gradients in the backward pass</li>
              <li>• Adjust the <strong>learning rate</strong>—too high overshoots, too low barely moves</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: The Key Insight */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Key Insight"
            subtitle="Errors flow backward, gradients multiply"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s what backpropagation really does:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Start at the loss.</strong> Compute how the loss changes
                with the network&apos;s output:{' '}
                <InlineMath math="\frac{\partial L}{\partial \text{output}}" />
              </li>
              <li>
                <strong>Flow backward.</strong> At each layer, multiply by the
                local derivative to get the gradient for the previous layer.
              </li>
              <li>
                <strong>Collect gradients.</strong> At each layer, use the
                incoming gradient to compute{' '}
                <InlineMath math="\frac{\partial L}{\partial w}" /> for that
                layer&apos;s weights.
              </li>
            </ol>
            <p className="text-muted-foreground mt-4">
              The &quot;error signal&quot; from the loss propagates backward
              through every layer, telling each weight how to change.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Vanishing Gradients">
            If local derivatives are small ({'<'}1), they multiply to get
            even smaller. Deep networks can suffer from gradients that
            &quot;vanish&quot; to near-zero. This is why ReLU is preferred over
            sigmoid — its derivative is either 0 or 1, not a small fraction.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Why This Matters */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why Backprop Changed Everything"
            subtitle="Efficient gradient computation at scale"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Before backprop was popularized (1986), training neural networks
              was impractical. Computing gradients naively requires:
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <ConceptBlock title="Naive Approach">
                <p>
                  Perturb each weight slightly, recompute the loss, measure the
                  change. For N weights, requires N forward passes.
                </p>
                <p className="mt-2 font-medium text-rose-400">
                  O(N) forward passes per gradient update
                </p>
              </ConceptBlock>

              <ConceptBlock title="Backpropagation">
                <p>
                  One forward pass + one backward pass gives gradients for ALL
                  weights at once, using the chain rule.
                </p>
                <p className="mt-2 font-medium text-emerald-400">
                  O(1) forward + backward passes
                </p>
              </ConceptBlock>
            </div>
            <p className="text-muted-foreground mt-4">
              For a network with 1 million weights, backprop is a million times
              faster than naive gradient computation. This efficiency unlocks
              deep learning.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Automatic Differentiation">
            Modern frameworks (PyTorch, TensorFlow) implement backprop
            automatically. You define the forward pass, they compute gradients.
            This is called &quot;autograd&quot; or &quot;automatic differentiation.&quot;
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
                headline: 'Backprop computes all gradients efficiently.',
                description:
                  'One forward pass + one backward pass gives gradients for every parameter.',
              },
              {
                headline: 'The chain rule is the key.',
                description:
                  'Multiply local derivatives along the path from loss to each weight.',
              },
              {
                headline: 'Gradients flow backward.',
                description:
                  'The error signal propagates from the loss back through each layer.',
              },
              {
                headline: 'This enables deep learning.',
                description:
                  'Without backprop, training networks with millions of parameters would be impractical.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* References */}
      <Row>
        <Row.Content>
          <ReferencesBlock
            references={[
              {
                title: 'Learning representations by back-propagating errors',
                authors: 'Rumelhart, Hinton & Williams, 1986',
                url: 'https://www.nature.com/articles/323533a0',
                note: 'The 1986 Nature paper that formalized backpropagation for training multi-layer neural networks.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="What's Next?"
            description="You now understand the core algorithm that trains neural networks. Next, we'll explore training in practice — batching, optimizers, and more."
            buttonText="Back to Home"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
