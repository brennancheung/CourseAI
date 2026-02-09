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
  ConstraintBlock,
  ComparisonRow,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { ComputationalGraphExplorer } from '@/components/widgets/ComputationalGraphExplorer'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Computational Graphs — Lesson 3 of Module 1.3 (Training Neural Networks)
 *
 * Teaches the student to represent neural network computations as visual
 * computational graphs and use them to trace the chain rule.
 *
 * Key concepts:
 * - Computational graph notation (nodes = operations, edges = data flow)
 * - Forward pass as left-to-right traversal
 * - Backward pass as right-to-left traversal with "incoming x local" rule
 * - Fan-out rule (gradients sum when a value feeds multiple paths)
 * - Connection to automatic differentiation / PyTorch's autograd
 */

export function ComputationalGraphsLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Computational Graphs"
            description="A visual notation that makes gradient bookkeeping automatic."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Read and trace computational graphs—the visual tool that PyTorch uses to compute
            every gradient automatically.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Same Math, New View">
            You already know how to compute gradients by hand. This lesson gives you
            the <strong>visual map</strong> that organizes that same computation.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'We are learning notation, not a new algorithm',
              'Same chain rule, same numbers—now drawn as a graph',
              'No code—just the visual framework that code uses',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook (Before/After) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Bookkeeping Problem"
            subtitle="You've done the math—but can you scale it?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Backprop by the Numbers, you traced 7 backward steps through a 4-parameter
              network. You tracked which saved value went with which derivative, which gradient
              to pass to the next layer, and how the chain rule connected everything.
            </p>
            <p className="text-muted-foreground">
              It worked. But be honest—the bookkeeping was a lot. And that was for
              4 parameters. Real networks have <strong>millions</strong>.
            </p>

            <ComparisonRow
              left={{
                title: 'Step-by-step list',
                color: 'amber',
                items: [
                  'Step 1: dL/dy\u0302 = -2(y - y\u0302)',
                  'Step 2: dL/dw\u2082 = dL/dy\u0302 \u00D7 a\u2081',
                  'Step 3: dL/db\u2082 = dL/dy\u0302 \u00D7 1',
                  'Step 4: dL/da\u2081 = dL/dy\u0302 \u00D7 w\u2082',
                  '\u2026 (3 more steps)',
                ],
              }}
              right={{
                title: 'As a graph',
                color: 'blue',
                items: [
                  'Each operation is a node',
                  'Values flow left \u2192 right',
                  'Gradients flow right \u2192 left',
                  'Every node: incoming \u00D7 local',
                  'The same math—visible at a glance',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Which one would you rather debug when there are 50 layers? The graph wins.
              That&apos;s what this lesson builds.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a New Algorithm">
            Computational graphs don&apos;t replace the chain rule—they <em>draw</em> it.
            You could compute every gradient without ever drawing a graph (you already did).
            But graphs make the computation visible and systematic.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain: The Simplest Graph */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Simplest Graph"
            subtitle="Two operations, one input, one output"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Let&apos;s start with the simplest possible example:{' '}
              <InlineMath math="f(x) = (x + 1)^2" /> with <InlineMath math="x = 3" />.
            </p>

            <p className="text-muted-foreground">
              This computation has two operations: add 1, then square. We draw each
              operation as a <strong>node</strong>, with arrows showing data flowing
              left to right:
            </p>

            <div className="rounded-lg border bg-muted/30 p-6">
              <div className="flex items-center justify-center gap-4 text-sm flex-wrap">
                <div className="rounded-full bg-blue-500/20 px-4 py-2 text-center border border-blue-500/40">
                  <div className="text-xs text-muted-foreground">input</div>
                  <div className="font-mono font-bold text-blue-400">x = 3</div>
                </div>
                <span className="text-muted-foreground font-mono">&rarr;</span>
                <div className="rounded-lg bg-muted px-4 py-2 text-center border border-muted-foreground/30">
                  <div className="text-xs text-muted-foreground">op</div>
                  <div className="font-mono font-bold text-muted-foreground">+ 1</div>
                </div>
                <span className="text-muted-foreground font-mono">&rarr;</span>
                <div className="rounded-lg bg-muted px-4 py-2 text-center border border-muted-foreground/30">
                  <div className="text-xs text-muted-foreground">op</div>
                  <div className="font-mono font-bold text-muted-foreground">x&sup2;</div>
                </div>
                <span className="text-muted-foreground font-mono">&rarr;</span>
                <div className="rounded-full bg-emerald-500/20 px-4 py-2 text-center border border-emerald-500/40">
                  <div className="text-xs text-muted-foreground">output</div>
                  <div className="font-mono font-bold text-emerald-400">f</div>
                </div>
              </div>
            </div>

            <p className="text-muted-foreground">
              <strong>Forward pass</strong> (left to right): Push the value through each node.
            </p>

            <div className="rounded-md bg-muted/50 px-4 py-3 font-mono text-sm space-y-1">
              <p>x = <span className="text-blue-400">3</span></p>
              <p>After +1: <span className="text-blue-400">4</span></p>
              <p>After square: <span className="text-blue-400">16</span></p>
            </div>

            <p className="text-muted-foreground">
              <strong>Backward pass</strong> (right to left): Start at the output with gradient
              = 1, then apply &ldquo;incoming gradient &times; local derivative&rdquo; at each node.
            </p>

            <div className="rounded-md bg-rose-500/5 border border-rose-500/20 px-4 py-3 font-mono text-sm space-y-1">
              <p><span className="text-rose-400">&nabla;f</span> = 1 (start here)</p>
              <p>Through square: <span className="text-rose-400">1</span> &times; 2&middot;(4) = <span className="text-rose-400">8</span></p>
              <p>Through +1: <span className="text-rose-400">8</span> &times; 1 = <span className="text-rose-400">8</span></p>
            </div>

            <p className="text-muted-foreground">
              Verify: <InlineMath math="\frac{d}{dx}(x+1)^2 = 2(x+1) = 2 \cdot 4 = 8" />.
              It matches. The graph traces the same chain rule you already know.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Graph Anatomy">
            <ul className="space-y-1 text-sm">
              <li><strong>Nodes</strong> = operations (add, multiply, ReLU, etc.)</li>
              <li><strong>Edges</strong> = data flowing between operations</li>
              <li><strong className="text-blue-400">Blue values</strong> = forward pass (above edges)</li>
              <li><strong className="text-rose-400">Red values</strong> = gradients (below edges)</li>
              <li><strong>Upstream</strong> = toward the output; <strong>downstream</strong> = toward the inputs</li>
            </ul>
          </ConceptBlock>
          <WarningBlock title="Opposite Directions">
            Forward values travel <em>with</em> the arrows (left to right).
            Gradients travel <em>against</em> the arrows (right to left).
            Same graph, opposite directions—that&apos;s why it&apos;s called the
            <strong> backward</strong> pass.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 4. Check: Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              If <InlineMath math="f(x) = (x + 2)^2" /> and <InlineMath math="x = 1" />,
              what is the gradient <InlineMath math="df/dx" />?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Pause. Draw the graph in your head (or on paper): two nodes, +2 then square.
              Trace forward, then backward. What gradient arrives at x?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show solution
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  Trace the graph: <InlineMath math="1 \xrightarrow{+2} 3 \xrightarrow{\text{square}} 9" />
                </p>
                <p className="text-muted-foreground">
                  Backward: through square: <InlineMath math="1 \times 2 \cdot 3 = 6" />,
                  through +2: <InlineMath math="6 \times 1 = 6" />.
                </p>
                <p className="text-sm font-mono text-emerald-400">
                  Verify: d/dx (x+2)&sup2; = 2(x+2) = 2 &middot; 3 = 6 &check;
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 5. Explain: The Lesson-2 Network as a Graph */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Your Network, as a Graph"
            subtitle="The same 2-layer network from Backprop by the Numbers"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember the network from Backprop by the Numbers?{' '}
              <InlineMath math="x = 2" />, <InlineMath math="w_1 = 0.5" />,{' '}
              <InlineMath math="b_1 = 0.1" />, <InlineMath math="w_2 = -0.3" />,{' '}
              <InlineMath math="b_2 = 0.2" />, <InlineMath math="y = 1" />.
              Every computation maps to a node:
            </p>

            <div className="rounded-lg border bg-muted/30 p-4 overflow-x-auto">
              <div className="flex items-center gap-2 text-xs font-mono min-w-[500px] justify-center flex-wrap">
                <span className="text-blue-400">x</span>
                <span className="text-muted-foreground">&rarr;</span>
                <span className="rounded border border-muted-foreground/30 px-2 py-1">{"×w₁"}</span>
                <span className="text-muted-foreground">{"→"}</span>
                <span className="rounded border border-muted-foreground/30 px-2 py-1">{"+b₁"}</span>
                <span className="text-muted-foreground">{"→"}</span>
                <span className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-amber-400">ReLU</span>
                <span className="text-muted-foreground">{"→"}</span>
                <span className="rounded border border-muted-foreground/30 px-2 py-1">{"×w₂"}</span>
                <span className="text-muted-foreground">{"→"}</span>
                <span className="rounded border border-muted-foreground/30 px-2 py-1">{"+b₂"}</span>
                <span className="text-muted-foreground">&rarr;</span>
                <span className="rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-rose-400">MSE</span>
                <span className="text-muted-foreground">&rarr;</span>
                <span className="text-emerald-400">L</span>
              </div>
            </div>

            <p className="text-muted-foreground">
              The forward values are the same ones you already computed:
            </p>

            <div className="grid grid-cols-3 gap-2 sm:grid-cols-6 text-center text-xs">
              {[
                { label: 'x', val: '2' },
                { label: '\u00D7w\u2081', val: '1.0' },
                { label: '+b\u2081', val: '1.1' },
                { label: 'ReLU', val: '1.1' },
                { label: '\u00D7w\u2082', val: '-0.33' },
                { label: '+b\u2082', val: '-0.13' },
              ].map(item => (
                <div key={item.label} className="rounded-md bg-blue-500/10 border border-blue-500/20 p-2">
                  <div className="text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-blue-400 font-bold">{item.val}</div>
                </div>
              ))}
            </div>

            <p className="text-muted-foreground">
              And the backward gradients are the same too. Step 3 in Backprop by the Numbers
              was <InlineMath math="dL/dw_2 = -2.26 \times 1.1 = -2.486" />. On the graph,
              that&apos;s the gradient arriving at the <InlineMath math="\times w_2" /> node
              (<InlineMath math="-2.26" />) times the local derivative (<InlineMath math="a_1 = 1.1" />).
            </p>

            <p className="text-muted-foreground">
              Same numbers. Same computation. The graph just draws it instead of listing it.
            </p>

            <p className="text-muted-foreground font-semibold mt-2">
              The chain rule, hop by hop
            </p>
            <p className="text-muted-foreground">
              Each factor in the chain rule corresponds to one hop backward through the graph.
              Here&apos;s the full expansion for <InlineMath math="dL/dw_1" />:
            </p>

            <div className="rounded-lg bg-muted/50 p-4 space-y-3">
              <BlockMath math="\frac{dL}{dw_1} = \underbrace{\frac{dL}{d\hat{y}}}_{\text{MSE} \to \text{+}b_2} \times \underbrace{\frac{d\hat{y}}{da_1}}_{\text{+}b_2 \to \times w_2 \to \text{ReLU}} \times \underbrace{\frac{da_1}{dz_1}}_{\text{ReLU}} \times \underbrace{\frac{dz_1}{dw_1}}_{\text{+}b_1 \to \times w_1}" />
              <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                <div className="rounded bg-rose-500/10 border border-rose-500/20 p-2 text-center">
                  <div className="text-muted-foreground">Hop 1</div>
                  <div className="font-mono text-rose-400">-2.26</div>
                </div>
                <div className="rounded bg-rose-500/10 border border-rose-500/20 p-2 text-center">
                  <div className="text-muted-foreground">Hop 2</div>
                  <div className="font-mono text-rose-400">-0.3</div>
                </div>
                <div className="rounded bg-rose-500/10 border border-rose-500/20 p-2 text-center">
                  <div className="text-muted-foreground">Hop 3</div>
                  <div className="font-mono text-rose-400">1</div>
                </div>
                <div className="rounded bg-rose-500/10 border border-rose-500/20 p-2 text-center">
                  <div className="text-muted-foreground">Hop 4</div>
                  <div className="font-mono text-rose-400">2</div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground text-center">
                4 meaningful multiplications backward through the graph ={' '}
                <span className="font-mono text-rose-400">1.356</span>
              </p>
              <p className="text-xs text-muted-foreground">
                Why 4 factors when the graph has more than 4 nodes? Addition nodes
                (like +b{"₁"} and +b{"₂"}) have local derivative 1, so they pass gradients
                through unchanged. They&apos;re real graph nodes, but they don&apos;t add a
                new factor to the product.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Recognition, Not Novelty">
            You&apos;ve already computed every one of these numbers. The graph reorganizes
            your existing knowledge—it doesn&apos;t add new knowledge. If the numbers
            look familiar, that&apos;s the point.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* The "one rule" crystallization */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s the single rule that governs every node in the graph:
            </p>
            <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
              <p className="text-lg font-semibold text-violet-400">
                incoming gradient &times; local derivative = outgoing gradient
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Every node follows this rule. The only thing that changes is what
                the local derivative is.
              </p>
            </div>
            <p className="text-muted-foreground">
              For the <InlineMath math="+" /> node: local derivative is 1.
              For <InlineMath math="\times" />: local derivative is the other input.
              For ReLU: local derivative is 1 (active) or 0 (dead).
              For MSE: local derivative is <InlineMath math="-2(y - \hat{y})" />.
              You already know all of these from the previous lesson.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Not &ldquo;Different Rules&rdquo;">
            Graphs look complex because they have many different node types. But every node
            follows the <em>same</em> rule. The only thing that varies is the local derivative
            formula, which you already learned for each operation.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6. Elaborate: The Fan-Out Rule */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When Paths Split"
            subtitle="The one genuinely new piece: gradients sum at fan-out"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Every example so far had a simple chain—each value flows into one operation.
              But what happens when <strong>one value feeds into two different operations</strong>?
            </p>

            <p className="text-muted-foreground">
              Consider <InlineMath math="f(x) = x \cdot (x + 1)" /> with <InlineMath math="x = 3" />.
              The variable <InlineMath math="x" /> fans out into two paths: one goes directly to
              the multiply node, the other goes through the add node first.
            </p>

            <div className="rounded-lg border bg-muted/30 p-6">
              <div className="flex items-center justify-center text-sm">
                <div className="relative flex flex-col items-center gap-12">
                  {/* x node */}
                  <div className="absolute left-0 top-1/2 -translate-y-1/2">
                    <div className="rounded-full bg-blue-500/20 px-4 py-2 border border-blue-500/40">
                      <div className="text-xs text-muted-foreground text-center">input</div>
                      <div className="font-mono font-bold text-blue-400 text-center">x = 3</div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col items-center gap-8 ml-4">
                  {/* Top path: direct to multiply */}
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground font-mono text-xs">&rarr;</span>
                    <span className="text-xs text-muted-foreground italic">value: 3</span>
                    <span className="text-muted-foreground font-mono text-xs">&rarr;</span>
                  </div>
                  {/* Bottom path: through +1 */}
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground font-mono text-xs">&rarr;</span>
                    <div className="rounded-lg bg-muted px-3 py-1.5 border border-muted-foreground/30 text-center">
                      <div className="text-xs text-muted-foreground">op</div>
                      <div className="font-mono font-bold text-muted-foreground">+ 1</div>
                    </div>
                    <span className="text-muted-foreground font-mono text-xs">&rarr;</span>
                    <span className="text-xs text-muted-foreground italic">4</span>
                    <span className="text-muted-foreground font-mono text-xs">&rarr;</span>
                  </div>
                </div>

                {/* Multiply node */}
                <div className="flex items-center gap-3 ml-2">
                  <div className="rounded-lg bg-muted px-3 py-1.5 border border-muted-foreground/30 text-center">
                    <div className="text-xs text-muted-foreground">op</div>
                    <div className="font-mono font-bold text-muted-foreground">&times;</div>
                  </div>
                  <span className="text-muted-foreground font-mono">&rarr;</span>
                  <div className="rounded-full bg-emerald-500/20 px-4 py-2 border border-emerald-500/40">
                    <div className="text-xs text-muted-foreground text-center">output</div>
                    <div className="font-mono font-bold text-emerald-400 text-center">f = 12</div>
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground text-center mt-3">
                x splits into two paths that both converge at the &times; node
              </p>
            </div>

            <p className="text-muted-foreground">
              Forward: <InlineMath math="3 \times (3 + 1) = 3 \times 4 = 12" />.
            </p>

            <p className="text-muted-foreground">
              Backward through the &times; node: the multiply operation sends gradient
              = 4 to the top path (value of the <em>other</em> input) and gradient
              = 3 to the bottom path. The bottom path passes through +1 (local derivative 1),
              giving gradient = 3.
            </p>

            <p className="text-muted-foreground">
              At <InlineMath math="x" />, two gradients arrive: <strong>4 from the top path</strong> and{' '}
              <strong>3 from the bottom path</strong>. The rule:
            </p>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5 text-center">
              <p className="text-lg font-semibold text-amber-400">
                When a value fans out, <strong>sum</strong> the gradients from each path.
              </p>
              <div className="mt-3 font-mono text-sm">
                <InlineMath math="\nabla x = 4 + 3 = 7" />
              </div>
            </div>

            <p className="text-muted-foreground">
              Verify: <InlineMath math="\frac{d}{dx}(x^2 + x) = 2x + 1 = 2 \cdot 3 + 1 = 7" />.
              The sum of path gradients matches the true derivative.
            </p>

            <p className="text-muted-foreground">
              This matters in real networks: when one neuron&apos;s output feeds into{' '}
              <em>every</em> neuron in the next layer, its gradient is the sum of gradients
              from all those downstream neurons.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Why Sum?">
            If <InlineMath math="x" /> affects the output through two paths, each path
            contributes independently to how the output changes. The total effect is
            the sum—just like how the total force on an object is the sum of individual
            forces.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7. Negative Example: No Path = No Gradient */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="No Path, No Gradient"
            subtitle="What happens when a variable isn't in the graph?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Consider <InlineMath math="f(a, b) = a + b" />. What is{' '}
              <InlineMath math="df/dc" />?
            </p>

            <p className="text-muted-foreground">
              There is no node involving <InlineMath math="c" />. No path from{' '}
              <InlineMath math="c" /> to the output means no gradient can flow to it.
              The answer: <InlineMath math="df/dc = 0" />.
            </p>

            <p className="text-muted-foreground">
              This isn&apos;t just a theoretical edge case. Remember the dying ReLU from
              Backprop by the Numbers? When a ReLU&apos;s input is negative, it puts a{' '}
              <strong>zero</strong> on the backward edge. That zero has the same effect
              as cutting the path—everything downstream gets zero gradient.
            </p>

            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">The principle:</strong> A parameter&apos;s
              gradient depends on the path from that parameter to the loss. No path = no
              gradient = that parameter doesn&apos;t learn.
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Broken Paths">
            In a computational graph, a zero on any edge in a path zeros out everything
            downstream of it. This is why dying ReLU is visible on the graph as a
            &ldquo;broken link&rdquo; in the gradient chain.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8. Check: Transfer */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              Given <InlineMath math="f(x, y) = x \cdot y + y" /> with{' '}
              <InlineMath math="x = 2, y = 3" />: find{' '}
              <InlineMath math="df/dx" /> and <InlineMath math="df/dy" />.
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Hint: which variable fans out to multiple paths? Draw the graph,
              trace forward values, then trace gradients backward. Sum where paths merge.
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show solution
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  Forward: <InlineMath math="x \cdot y = 6" />,{' '}
                  <InlineMath math="6 + y = 6 + 3 = 9" />.
                </p>
                <p className="text-muted-foreground">
                  Backward for <InlineMath math="df/dx" />: through + (gradient 1),
                  through &times; (local derivative = y = 3).{' '}
                  <span className="font-mono text-emerald-400">df/dx = 3</span>. Only one path.
                </p>
                <p className="text-muted-foreground">
                  Backward for <InlineMath math="df/dy" />: <InlineMath math="y" /> fans out to both
                  the &times; node and the + node. From &times;: gradient = x = 2. From +:
                  gradient = 1. Sum:{' '}
                  <span className="font-mono text-emerald-400">df/dy = 2 + 1 = 3</span>.
                </p>
                <p className="text-sm font-mono text-emerald-400">
                  Verify: d/dx(xy + y) = y = 3 &check; &nbsp; d/dy(xy + y) = x + 1 = 3 &check;
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Fan-Out in Action">
            The variable <InlineMath math="y" /> appears in two places, so it gets gradients
            from two paths. This is the fan-out rule in a new context.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 9. Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Computational Graph Explorer"
            subtitle="Watch values and gradients flow through the graph"
          >
            <ComputationalGraphExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>&bull; Start with f(x) = (x+1)&sup2; and change x—watch both the forward values and gradients update</li>
              <li>&bull; Switch to the 2-Layer Network—recognize the numbers from Backprop by the Numbers</li>
              <li>&bull; Try f(x) = x&middot;(x+1)—hover or tap x to see gradients arriving from two paths and summing</li>
              <li>&bull; Toggle between Graph View and Step-by-Step to see they produce the same numbers</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 10. Connection to Autograd */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="This Is What PyTorch Does"
            subtitle="From visual notation to automatic differentiation"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Backpropagation: How Networks Learn, we mentioned that frameworks compute
              gradients automatically. Now you know <em>how</em>.
            </p>

            <p className="text-muted-foreground">
              When you write <code className="text-sm bg-muted px-1.5 py-0.5 rounded">y = w * x + b</code> in
              PyTorch, the framework builds exactly the kind of graph you&apos;ve been
              tracing—a node for the multiply, a node for the add. When you call{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">loss.backward()</code>,
              PyTorch walks the graph backward, applying &ldquo;incoming &times; local&rdquo; at
              every node. That&apos;s <strong>automatic differentiation</strong>—the framework
              does the graph traversal for you.
            </p>

            <p className="text-muted-foreground">
              You don&apos;t need to implement this yourself. But understanding it means you
              know what&apos;s happening when training seems broken—when gradients explode,
              vanish, or mysteriously become zero. The graph is where you&apos;ll debug.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Automatic Differentiation">
            Build the graph during the forward pass. Walk it backward to compute all
            gradients. That&apos;s autograd in a sentence.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 11. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Operations become nodes, data flow becomes edges',
                description: 'Any sequence of operations can be drawn as a graph, making gradient flow visible.',
              },
              {
                headline: 'Forward: left to right',
                description: 'Push values through the graph, saving each intermediate result.',
              },
              {
                headline: 'Backward: right to left',
                description: 'At every node: incoming gradient \u00D7 local derivative = outgoing gradient.',
              },
              {
                headline: 'Fan-out: sum the gradients',
                description: 'When one value feeds multiple paths, its total gradient is the sum from all paths.',
              },
              {
                headline: 'This is how PyTorch computes gradients',
                description: 'loss.backward() walks a computational graph—the same process you just traced by hand.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 12. Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: Batching and SGD"
            description="From one input at a time to training on real datasets"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
