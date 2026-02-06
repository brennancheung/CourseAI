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
  ComparisonRow,
  GradientCard,
  ModuleCompleteBlock,
} from '@/components/lessons'
import { ActivationFunctionExplorer } from '@/components/widgets/ActivationFunctionExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { BlockMath } from 'react-katex'

/**
 * Activation Functions - Visual Reference
 *
 * Final lesson in "From Linear to Neural" module.
 * A visual reference for activation function shapes and basic properties.
 *
 * Note: The "why" behind the trade-offs (vanishing gradients, etc.)
 * is deferred to after backprop is taught.
 *
 * Focus: What do they look like? What's the intuition? When to use?
 */

export function ActivationFunctionsDeepDiveLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Activation Functions: Visual Reference"
            description="A visual tour of activation functions — sigmoid, tanh, ReLU, GELU, and more. See their shapes and learn when to use each."
            category="From Linear to Neural"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Build visual intuition for the major activation functions. This is
            your reference guide — you&apos;ll come back to it later.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Coming Later">
            Why do some activations work better than others? That question
            requires understanding <em>backpropagation</em> (how networks
            learn). We&apos;ll revisit this in Module 1.3.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section: Sigmoid */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Sigmoid"
            subtitle="Squashes everything to (0, 1)"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\sigma(x) = \frac{1}{1 + e^{-x}}" />
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <ConceptBlock title="Output Range">
                <p className="font-mono text-lg">(0, 1)</p>
                <p className="text-xs mt-1">Always between 0 and 1</p>
              </ConceptBlock>
              <ConceptBlock title="Shape">
                <p>S-curve (sigmoid)</p>
                <p className="text-xs mt-1">Smooth transition</p>
              </ConceptBlock>
              <ConceptBlock title="Main Use">
                <p>Binary probability</p>
                <p className="text-xs mt-1">Output layer for yes/no</p>
              </ConceptBlock>
            </div>
            <p className="text-muted-foreground">
              <strong>Intuition:</strong> Sigmoid interprets its input as a
              &quot;confidence score&quot; and outputs a probability. Large positive
              inputs → close to 1. Large negative → close to 0.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Historical Note">
            Sigmoid was the original activation, inspired by biological neurons.
            It&apos;s now mainly used for output layers, not hidden layers.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: Tanh */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Tanh"
            subtitle="Squashes to (-1, 1) — zero-centered"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}" />
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <ConceptBlock title="Output Range">
                <p className="font-mono text-lg">(-1, 1)</p>
                <p className="text-xs mt-1">Can be negative</p>
              </ConceptBlock>
              <ConceptBlock title="Shape">
                <p>S-curve centered at 0</p>
                <p className="text-xs mt-1">Similar to sigmoid</p>
              </ConceptBlock>
              <ConceptBlock title="Main Use">
                <p>RNNs / LSTMs</p>
                <p className="text-xs mt-1">When you need ± values</p>
              </ConceptBlock>
            </div>
            <p className="text-muted-foreground">
              <strong>Intuition:</strong> Like sigmoid, but centered at zero.
              Outputs can be positive or negative, which is often more useful
              for hidden layers.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="vs Sigmoid">
            Tanh is often preferred over sigmoid for hidden layers because
            it&apos;s zero-centered. But both have been largely replaced by ReLU.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Sigmoid vs Tanh */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Compare Sigmoid and Tanh">
            <ActivationFunctionExplorer
              defaultFunction="sigmoid"
              showDerivatives={false}
            />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Observe">
            <ul className="space-y-2 text-sm">
              <li>• Both have the S-curve shape</li>
              <li>• Sigmoid outputs 0.5 at x=0; tanh outputs 0</li>
              <li>• Move to extreme values — both saturate</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: ReLU */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="ReLU"
            subtitle="The modern default"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{ReLU}(x) = \max(0, x)" />
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <ConceptBlock title="Output Range">
                <p className="font-mono text-lg">[0, ∞)</p>
                <p className="text-xs mt-1">Zero or positive</p>
              </ConceptBlock>
              <ConceptBlock title="Shape">
                <p>Hinge at zero</p>
                <p className="text-xs mt-1">Flat left, linear right</p>
              </ConceptBlock>
              <ConceptBlock title="Main Use">
                <p>Default for most networks</p>
                <p className="text-xs mt-1">CNNs, feedforward</p>
              </ConceptBlock>
            </div>
            <p className="text-muted-foreground">
              <strong>Intuition:</strong> Dead simple — pass positive values
              through, block negative ones. Despite its simplicity, ReLU works
              remarkably well and is computationally cheap.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Surprise">
            ReLU&apos;s simplicity was initially seen as a weakness. But it
            turns out that this &quot;less is more&quot; approach works better
            than the biologically-inspired sigmoid in most cases.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: Leaky ReLU */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Leaky ReLU"
            subtitle="ReLU with a small slope for negatives"
          />
          <div className="space-y-4">
            <div className="py-4 px-6 bg-muted/50 rounded-lg">
              <BlockMath math="\text{LeakyReLU}(x) = \max(0.01x, x)" />
            </div>
            <ComparisonRow
              left={{
                title: 'ReLU',
                color: 'emerald',
                items: [
                  'Zero for negative inputs',
                  'Creates sparsity',
                  'Can "die" (stuck at 0)',
                ],
              }}
              right={{
                title: 'Leaky ReLU',
                color: 'cyan',
                items: [
                  'Small slope for negatives',
                  'Never completely blocks',
                  'Slightly more computation',
                ],
              }}
            />
            <p className="text-muted-foreground">
              <strong>When to use:</strong> If ReLU neurons are &quot;dying&quot;
              (always outputting zero), try Leaky ReLU as a drop-in replacement.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The 0.01">
            The small negative slope (0.01) is a hyperparameter. Some variants
            (PReLU) learn this value. But 0.01 works well as a default.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: ReLU vs Leaky */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Compare ReLU and Leaky ReLU">
            <ActivationFunctionExplorer
              defaultFunction="relu"
              showDerivatives={false}
            />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Observe">
            <ul className="space-y-2 text-sm">
              <li>• Toggle &quot;Linear&quot; to see the baseline</li>
              <li>• ReLU is flat for x {'<'} 0</li>
              <li>• Leaky ReLU has a small slope for x {'<'} 0</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: GELU and Swish */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="GELU and Swish"
            subtitle="Smooth modern alternatives"
          />
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <GradientCard title="GELU" color="rose">
                <p className="font-mono text-sm mb-2">x · Φ(x)</p>
                <p className="text-sm">Used in transformers (GPT, BERT)</p>
              </GradientCard>
              <GradientCard title="Swish" color="amber">
                <p className="font-mono text-sm mb-2">x · σ(x)</p>
                <p className="text-sm">Used in vision (EfficientNet)</p>
              </GradientCard>
            </div>
            <p className="text-muted-foreground">
              Both GELU and Swish are smooth versions of ReLU. They can output
              small negative values and have no sharp corners. They&apos;re
              nearly identical in shape.
            </p>
            <p className="text-muted-foreground">
              <strong>When to use:</strong> GELU for language models/transformers.
              Swish for computer vision. But honestly, they&apos;re interchangeable
              for most purposes.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Smooth?">
            GELU and Swish don&apos;t have the sharp &quot;kink&quot; at zero that
            ReLU has. Whether this matters depends on the problem. For
            transformers, it seems to help.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: Modern activations */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Compare Modern Activations">
            <ActivationFunctionExplorer
              defaultFunction="relu"
              showDerivatives={false}
            />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Observe">
            <ul className="space-y-2 text-sm">
              <li>• GELU and Swish look nearly identical</li>
              <li>• Both are smooth where ReLU has a corner</li>
              <li>• All three are similar for positive inputs</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Section: Quick Decision Guide */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Quick Decision Guide"
            subtitle="What to use when"
          />
          <div className="space-y-3">
            <div className="p-4 rounded-lg border bg-muted/30">
              <p className="font-medium text-emerald-400 mb-1">
                Default choice: ReLU
              </p>
              <p className="text-sm text-muted-foreground">
                Start with ReLU for feedforward and convolutional networks.
                It&apos;s fast, simple, and works well.
              </p>
            </div>

            <div className="p-4 rounded-lg border bg-muted/30">
              <p className="font-medium text-rose-400 mb-1">
                Transformers/LLMs: GELU
              </p>
              <p className="text-sm text-muted-foreground">
                GELU is the standard for BERT, GPT, and most language models.
              </p>
            </div>

            <div className="p-4 rounded-lg border bg-muted/30">
              <p className="font-medium text-orange-400 mb-1">
                Output layer (binary): Sigmoid
              </p>
              <p className="text-sm text-muted-foreground">
                When you need a probability between 0 and 1 for yes/no questions.
              </p>
            </div>

            <div className="p-4 rounded-lg border bg-muted/30">
              <p className="font-medium text-violet-400 mb-1">
                RNNs: Tanh
              </p>
              <p className="text-sm text-muted-foreground">
                Tanh is still common in recurrent networks like LSTMs.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Don&apos;t Overthink">
            The activation function matters, but it&apos;s rarely the
            bottleneck. Pick something reasonable based on your architecture
            and move on.
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
                headline: 'Sigmoid/Tanh: S-curves that squash to bounded ranges.',
                description:
                  'Sigmoid → (0,1), Tanh → (-1,1). Mostly for output layers or RNNs now.',
              },
              {
                headline: 'ReLU: Simple hinge that passes positive values.',
                description:
                  'The default for most networks. Fast and effective.',
              },
              {
                headline: 'GELU/Swish: Smooth ReLU alternatives.',
                description:
                  'Used in modern architectures like transformers. Nearly identical to each other.',
              },
              {
                headline: 'Why some work better than others?',
                description:
                  'That requires understanding backpropagation — coming in Module 1.3!',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="1.2"
            title="From Linear to Neural"
            achievements={[
              'Why linear models fail on XOR',
              'How activation functions add nonlinearity',
              'The shapes of sigmoid, tanh, ReLU, GELU, Swish',
              'When to use each activation function',
            ]}
            nextModule="1.3"
            nextTitle="Backpropagation"
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Module 1.2 Complete!"
            description="You now know the building blocks of neural networks. Next: How do they actually learn? (Backpropagation)"
            buttonText="Back to Home"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
