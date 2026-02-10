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
  ReferencesBlock,
} from '@/components/lessons'
import { XORClassifierExplorer } from '@/components/widgets/XORClassifierExplorer'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'

/**
 * The Limits of Linearity
 *
 * Follows "The Neuron" lesson. Now that we understand neurons and networks,
 * this lesson shows WHY that structure alone isn't enough.
 *
 * Key concepts:
 * - Linear decision boundaries can only separate linearly separable data
 * - XOR is the canonical example of non-linearly separable data
 * - Even multi-layer LINEAR networks can't solve XOR
 * - This motivates the need for activation functions
 *
 * Next: Activation Functions (the fix)
 */

export function LimitsOfLinearityLesson() {
  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Limits of Linearity"
            description="See why the neural network structure alone isn't enough—linear networks fail on simple problems like XOR."
            category="From Linear to Neural"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            See exactly where linear neural networks break down—and understand
            why the structure we just learned isn&apos;t enough on its own.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Remember">
            In the previous lesson, we saw that stacking linear layers is still
            linear. Now we&apos;ll see a concrete problem that proves why that&apos;s
            a fatal limitation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section: What is XOR? - FIRST, before showing the graph */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="XOR: A Simple Function"
            subtitle="Can our neural network learn this?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              XOR (exclusive or) is a simple function with two inputs and one output.
              It returns 1 when the inputs are <em>different</em>, and 0 when they&apos;re the same:
            </p>
            <div className="overflow-x-auto">
              <table className="text-sm border-collapse w-full max-w-sm">
                <thead>
                  <tr className="border-b border-muted">
                    <th className="p-2 text-left text-muted-foreground">Input A</th>
                    <th className="p-2 text-left text-muted-foreground">Input B</th>
                    <th className="p-2 text-left text-muted-foreground">XOR(A, B)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-muted/50">
                    <td className="p-2 font-mono">0</td>
                    <td className="p-2 font-mono">0</td>
                    <td className="p-2 font-mono font-bold text-blue-400">0</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="p-2 font-mono">0</td>
                    <td className="p-2 font-mono">1</td>
                    <td className="p-2 font-mono font-bold text-orange-400">1</td>
                  </tr>
                  <tr className="border-b border-muted/50">
                    <td className="p-2 font-mono">1</td>
                    <td className="p-2 font-mono">0</td>
                    <td className="p-2 font-mono font-bold text-orange-400">1</td>
                  </tr>
                  <tr>
                    <td className="p-2 font-mono">1</td>
                    <td className="p-2 font-mono">1</td>
                    <td className="p-2 font-mono font-bold text-blue-400">0</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-muted-foreground">
              Let&apos;s try to learn this function. We have 4 training examples
              (each row in the table), and we want our model to predict the correct output.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Challenge">
            We want our neural network to take (A, B) as input and output the
            correct XOR value. Can the linear network we just learned do this?
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section: Visualizing XOR as a Classification Problem */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Plotting XOR"
            subtitle="Turning the truth table into a graph"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              To visualize this, we&apos;ll plot each row as a point on a graph:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>X-axis:</strong> Input A (0 or 1)
              </li>
              <li>
                <strong>Y-axis:</strong> Input B (0 or 1)
              </li>
              <li>
                <strong>Color:</strong> The output—<span className="text-blue-400 font-medium">blue for 0</span>,{' '}
                <span className="text-orange-400 font-medium">orange for 1</span>
              </li>
            </ul>
            <p className="text-muted-foreground">
              This gives us 4 points at the corners of a square. Now the question becomes:
              can we draw a <strong>single straight line</strong> that separates
              the blue points from the orange points?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Classification">
            We&apos;ve turned &quot;learning XOR&quot; into a <strong>classification problem</strong>:
            given a point (A, B), which side of the line is it on? That determines
            our prediction.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The XOR Challenge - NOW show the interactive */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The XOR Challenge"
            subtitle="Try to separate the two classes with a single line"
          />
          <p className="text-muted-foreground mb-4">
            Your goal: <strong>draw a line that perfectly separates the classes</strong>.
            All blue points (output = 0) should be on one side, all orange points
            (output = 1) on the other.
          </p>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Go Ahead">
            <ul className="space-y-2 text-sm">
              <li>• Adjust the slope and intercept</li>
              <li>• Try to get 100% accuracy</li>
              <li>• Spend at least a minute trying</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Interactive: XOR Explorer */}
      <Row>
        <Row.Content>
          <ExercisePanel title="XOR Classification Challenge">
            <XORClassifierExplorer />
          </ExercisePanel>
        </Row.Content>
      </Row>

      {/* Why It's Impossible */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why It's Impossible"
            subtitle="The diagonal problem"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Look at where the same-class points are:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <span className="text-blue-400 font-medium">Blue (output 0)</span>: corners (0,0) and (1,1)—<strong>diagonal</strong>
              </li>
              <li>
                <span className="text-orange-400 font-medium">Orange (output 1)</span>: corners (0,1) and (1,0)—<strong>opposite diagonal</strong>
              </li>
            </ul>
            <p className="text-muted-foreground">
              A straight line can&apos;t separate points on opposite diagonals.
              No matter how you rotate or position it, one point from each class
              will always end up on the wrong side.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Core Issue">
            XOR is <strong>not linearly separable</strong>. This isn&apos;t about
            finding better parameters—no linear model can learn XOR. We need
            something fundamentally different.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: Linear Decision Boundaries */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Linear Decision Boundaries"
            subtitle="What a line can and can't do"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A linear classifier divides the input space with a straight line
              (in 2D) or a flat plane (in higher dimensions). Everything on one
              side gets one label; everything on the other side gets the other.
            </p>
            <p className="text-muted-foreground">
              Consider AND: (0,0)=0, (0,1)=0, (1,0)=0, (1,1)=1. The single
              output-1 point sits in one corner, all the output-0 points in the
              other three. A line easily separates them. Now compare that to
              XOR, where the same-class points sit on <em>opposite</em> diagonals.
            </p>
            <ComparisonRow
              left={{
                title: 'Linearly Separable',
                color: 'emerald',
                items: [
                  'Same-class points cluster together',
                  'A single line can separate them',
                  'AND: one corner vs three corners',
                ],
              }}
              right={{
                title: 'Not Linearly Separable',
                color: 'rose',
                items: [
                  'Same-class points on opposite diagonals',
                  'No single line works',
                  'XOR: two corners vs two corners',
                ],
              }}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Higher Dimensions">
            In 3D, a linear boundary is a plane. In higher dimensions, it&apos;s
            called a hyperplane. But the limitation is the same—it can only
            make one straight cut.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section: Why This Matters */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Why This Matters"
            subtitle="Real patterns are rarely linear"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              XOR might seem like a toy problem, but the pattern shows up everywhere:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>
                <strong>Images:</strong> &quot;Is this a cat?&quot; can&apos;t be answered
                by drawing a line through pixel values
              </li>
              <li>
                <strong>Language:</strong> The meaning of &quot;bank&quot; depends on
                context in nonlinear ways
              </li>
              <li>
                <strong>Science:</strong> Many relationships are inherently
                nonlinear (e.g., dose-response curves)
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              If we could only use linear models, machine learning would be very limited.
              We need a way to create <strong>curved decision boundaries</strong>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Motivation">
            This is exactly why neural networks were invented. By combining
            linear operations with nonlinear <em>activation functions</em>, we
            can learn any decision boundary—including the one that solves XOR.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section: The Solution Preview */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Missing Ingredient"
            subtitle="We have the structure—we need one more thing"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We already have the neural network structure: neurons, layers,
              connections. But as we saw in the previous lesson, stacking linear
              layers is still linear.
            </p>
            <p className="text-muted-foreground">
              The solution isn&apos;t more layers or bigger networks—it&apos;s
              adding <strong>nonlinearity</strong> after each linear operation.
            </p>
            <p className="text-muted-foreground">
              When we add a nonlinear <strong>activation function</strong> to each
              neuron, suddenly:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
              <li>Each neuron creates one decision boundary</li>
              <li>Multiple neurons = multiple boundaries</li>
              <li>Combined, they can carve out any region—including XOR</li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Two Neurons, Two Lines">
            Here&apos;s a hint: XOR can be solved with two neurons creating two
            lines. The region <em>between</em> them captures exactly the orange
            points. We&apos;ll see this visually in the next lesson.
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
                headline: 'Linear networks draw straight boundaries.',
                description:
                  'No matter how many layers, a linear network can only make one straight cut.',
              },
              {
                headline: 'XOR is not linearly separable.',
                description:
                  'Same-class points are on opposite diagonals—no single line can separate them.',
              },
              {
                headline: 'This limitation is fundamental.',
                description:
                  'It\'s not about finding better weights. Linear networks mathematically cannot solve XOR.',
              },
              {
                headline: 'We need nonlinearity.',
                description:
                  'Adding a nonlinear activation function after each neuron unlocks the full power of neural networks.',
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
                note: 'The seminal paper that demonstrated XOR as the canonical example of why neural networks need nonlinear hidden layers.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/activation-functions"
            title="Add the Missing Ingredient"
            description="Next: activation functions—the simple nonlinearity that transforms our limited linear network into a universal function approximator."
            buttonText="Continue to Activation Functions"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
