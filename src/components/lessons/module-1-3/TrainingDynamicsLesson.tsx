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
  GradientCard,
  ReferencesBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { TrainingDynamicsExplorer } from '@/components/widgets/TrainingDynamicsExplorer'
import { cn } from '@/lib/utils'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Training Dynamics — Lesson 6 of Module 1.3 (Training Neural Networks)
 *
 * Teaches the student to:
 * - Diagnose why deep networks fail to train (vanishing/exploding gradients)
 * - Understand how weight initialization prevents these failures (Xavier/He)
 * - Recognize batch normalization as a technique that stabilizes gradient flow
 *
 * Target depths:
 * - Vanishing gradients: DEVELOPED (from INTRODUCED)
 * - Exploding gradients: DEVELOPED (new)
 * - Weight initialization (Xavier/He): DEVELOPED (new)
 * - Batch normalization: INTRODUCED (new)
 * - Gradient clipping: MENTIONED
 * - Skip connections / ResNets: MENTIONED
 */

export function TrainingDynamicsLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Training Dynamics"
            description="Why deep networks fail to train—and the three ideas that fix it."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Understand why stacking more layers makes training <em>harder</em>, not
            easier—and learn the three ideas (understanding gradient pathologies,
            preventing them with initialization, and stabilizing training with batch
            normalization) that turned &ldquo;deep learning&rdquo; from a theoretical idea
            into a practical reality.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Chain Rule, Again">
            Everything in this lesson follows from one fact you already know: gradients are
            products of local derivatives through the chain rule. The question is just:{' '}
            <em>what happens when you multiply many factors together?</em>
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'We cover three techniques: understanding the problem (gradient pathologies), prevention (initialization), and runtime stabilization (batch normalization)',
              'No code implementation—that comes in the PyTorch series',
              'Skip connections and gradient clipping are mentioned but not developed',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — The Depth Paradox */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Depth Paradox"
            subtitle="Deep learning is named for deep networks—so why does going deeper break training?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You now have everything you need to train a neural network: backpropagation to compute
              gradients, mini-batch SGD to handle real datasets, and Adam to navigate tricky loss
              landscapes. You could train a 2-layer network right now, and it would work.
            </p>
            <p className="text-muted-foreground">
              But here is the paradox. The field is called <strong>deep</strong> learning—meaning
              many layers. Yet if you take your 2-layer network that trains beautifully and stack it
              to 10 layers, training collapses. Same optimizer, same data, same learning rate. The
              only change is depth.
            </p>

            {/* Before/after training curves */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <svg viewBox="0 0 400 120" className="w-full" style={{ maxHeight: 140 }}>
                {/* Axes */}
                <line x1="40" y1="100" x2="380" y2="100" stroke="#666" strokeWidth="1" />
                <line x1="40" y1="10" x2="40" y2="100" stroke="#666" strokeWidth="1" />
                <text x="210" y="118" textAnchor="middle" fill="#888" fontSize="9">Epoch</text>
                <text x="12" y="55" textAnchor="middle" fill="#888" fontSize="9" transform="rotate(-90, 12, 55)">Loss</text>

                {/* 2-layer network: healthy training curve */}
                <path
                  d="M 50,90 Q 100,60 150,35 Q 200,22 250,18 Q 300,16 370,15"
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="2"
                />

                {/* 10-layer network: flatline */}
                <path
                  d="M 50,88 Q 100,86 150,85 Q 200,84.5 250,84 Q 300,83.8 370,83.5"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                />

                {/* Legend */}
                <line x1="110" y1="8" x2="130" y2="8" stroke="#22c55e" strokeWidth="2" />
                <text x="134" y="11" fill="#22c55e" fontSize="9">2-layer network</text>
                <line x1="250" y1="8" x2="270" y2="8" stroke="#ef4444" strokeWidth="2" />
                <text x="274" y="11" fill="#ef4444" fontSize="9">10-layer network</text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                Same architecture, same optimizer, same data. The 2-layer network learns normally.
                The 10-layer network barely improves.
              </p>
            </div>

            <div className="rounded-lg border-2 border-amber-500/30 bg-amber-500/5 p-5 text-center">
              <p className="text-amber-400 font-semibold">
                Why does going deeper make things <em>worse</em>?
              </p>
            </div>

            {/* Misconception #5: deeper is always better */}
            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">
                Deeper is not automatically better.
              </strong>{' '}
              Without the techniques in this lesson, going deeper consistently made performance{' '}
              <em>worse</em>—not better. Before ResNet (2015), adding layers beyond a point hurt
              accuracy. Depth is powerful, but only when you have the tools to maintain gradient
              flow.
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a Bug">
            This is not a bug in your code. Before 2012, researchers routinely
            failed to train networks deeper than a few layers. The techniques in this
            lesson are what changed that.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Recap — Chain Rule Products */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Chain Rule Products"
            subtitle="A quick reminder of what you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember from Backpropagation: the gradient for a weight in layer 1 is a{' '}
              <strong>product</strong> of local derivatives through every layer between it and
              the loss. You computed this concretely in Backprop by the Numbers for 2 layers.
              Here is the general form for <InlineMath math="N" /> layers:
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <BlockMath math="\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial a_N}}_{\text{loss derivative}} \cdot \underbrace{\frac{\partial a_N}{\partial z_N}}_{\text{activation}_{N}} \cdot \frac{\partial z_N}{\partial a_{N-1}} \cdots \underbrace{\frac{\partial a_1}{\partial z_1}}_{\text{activation}_{1}} \cdot \frac{\partial z_1}{\partial w_1}" />
              <p className="text-xs text-muted-foreground">
                Each <InlineMath math="\frac{\partial a_i}{\partial z_i}" /> is the derivative of
                the activation function at layer <InlineMath math="i" />. In Backprop by the Numbers,
                you called this &ldquo;incoming gradient times local derivative.&rdquo;
              </p>
            </div>

            <p className="text-muted-foreground">
              You did this for 2 layers and it worked. Now imagine 10 layers—or 100. Each{' '}
              <InlineMath math="\frac{\partial a_i}{\partial z_i}" /> is one factor in the product.
              What happens when you multiply many numbers together?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="The Key Equation">
            <p className="text-sm">
              The gradient at layer <InlineMath math="k" /> is:
            </p>
            <p className="text-sm mt-1">
              <InlineMath math="|{\nabla}_{k}| \approx \prod_{i=k}^{N} |\sigma'(z_i)|" />
            </p>
            <p className="text-sm mt-2">
              A <strong>product</strong> of <InlineMath math="N - k" /> activation derivatives.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Vanishing Gradients */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Vanishing Gradients"
            subtitle="When small factors multiply to near-zero"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Sigmoid&apos;s derivative has a maximum value of <strong>0.25</strong> (at{' '}
              <InlineMath math="z = 0" />)—and is usually smaller. Here is the derivative curve:
            </p>

            {/* Sigmoid derivative curve — bell-shaped, peaking at 0.25 at z=0 */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <svg viewBox="0 0 400 130" className="w-full" style={{ maxHeight: 150 }}>
                {/* Axes */}
                <line x1="40" y1="110" x2="380" y2="110" stroke="#666" strokeWidth="1" />
                <line x1="200" y1="10" x2="200" y2="110" stroke="#666" strokeWidth="0.5" strokeDasharray="4 4" />
                <text x="385" y="113" fill="#888" fontSize="9">z</text>
                <text x="15" y="18" fill="#888" fontSize="9">&sigma;&apos;(z)</text>

                {/* Z-axis labels */}
                <text x="60" y="122" textAnchor="middle" fill="#888" fontSize="8">-4</text>
                <text x="130" y="122" textAnchor="middle" fill="#888" fontSize="8">-2</text>
                <text x="200" y="122" textAnchor="middle" fill="#888" fontSize="8">0</text>
                <text x="270" y="122" textAnchor="middle" fill="#888" fontSize="8">2</text>
                <text x="340" y="122" textAnchor="middle" fill="#888" fontSize="8">4</text>

                {/* Shaded region where derivative < 0.1 — left tail */}
                <path
                  d={`M 40,110 ${[
                    [-4.6, 0.0098], [-4, 0.0177], [-3.5, 0.0291], [-3, 0.0452],
                    [-2.5, 0.0664], [-2, 0.105],
                  ].map(([z, d]) => {
                    const x = 200 + z * 35
                    const y = 110 - d * 360
                    return `L ${x},${y}`
                  }).join(' ')} L ${200 + -2 * 35},110 Z`}
                  fill="#ef4444"
                  opacity="0.1"
                />

                {/* Shaded region where derivative < 0.1 — right tail */}
                <path
                  d={`M ${200 + 2 * 35},110 ${[
                    [2, 0.105], [2.5, 0.0664], [3, 0.0452],
                    [3.5, 0.0291], [4, 0.0177], [4.6, 0.0098],
                  ].map(([z, d]) => {
                    const x = 200 + z * 35
                    const y = 110 - d * 360
                    return `L ${x},${y}`
                  }).join(' ')} L 380,110 Z`}
                  fill="#ef4444"
                  opacity="0.1"
                />

                {/* Derivative curve: sigma'(z) = sigma(z)(1-sigma(z)) */}
                {(() => {
                  const derivPoints: [number, number][] = [
                    [-4, 0.0177], [-3.5, 0.0291], [-3, 0.0452], [-2.5, 0.0664],
                    [-2, 0.105], [-1.5, 0.1491], [-1, 0.1966], [-0.5, 0.2350],
                    [0, 0.25],
                    [0.5, 0.2350], [1, 0.1966], [1.5, 0.1491],
                    [2, 0.105], [2.5, 0.0664], [3, 0.0452], [3.5, 0.0291],
                    [4, 0.0177], [4.6, 0.0098],
                  ]
                  const segments = derivPoints.map(([z, val]) =>
                    'L ' + (200 + z * 35) + ',' + (110 - val * 360)
                  )
                  const derivPathD = 'M ' + (200 + -4.6 * 35) + ',' + (110 - 0.0098 * 360) + ' ' + segments.join(' ')
                  return <path d={derivPathD} fill="none" stroke="#a78bfa" strokeWidth="2" />
                })()}

                {/* Peak annotation at z=0, y=0.25 */}
                <circle cx="200" cy={110 - 0.25 * 360} r="3" fill="#a78bfa" />
                <line x1="200" y1={110 - 0.25 * 360 - 5} x2="200" y2={110 - 0.25 * 360 - 20} stroke="#a78bfa" strokeWidth="1" />
                <text x="200" y={110 - 0.25 * 360 - 23} textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="600">
                  max = 0.25
                </text>

                {/* "< 0.1" label on tails */}
                <text x="75" y="100" textAnchor="middle" fill="#ef4444" fontSize="8" opacity="0.8">
                  &lt; 0.1
                </text>
                <text x="325" y="100" textAnchor="middle" fill="#ef4444" fontSize="8" opacity="0.8">
                  &lt; 0.1
                </text>

                {/* 0.1 reference line */}
                <line
                  x1="40" y1={110 - 0.1 * 360} x2="380" y2={110 - 0.1 * 360}
                  stroke="#ef4444" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.4"
                />
                <text x="38" y={110 - 0.1 * 360 - 3} textAnchor="end" fill="#ef4444" fontSize="7" opacity="0.6">
                  0.1
                </text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                Sigmoid&apos;s derivative peaks at just 0.25 and only near{' '}
                <InlineMath math="z = 0" />. In the <span className="text-red-400">shaded tails</span>,
                the derivative drops below 0.1—meaning most inputs produce factors far smaller than 0.25.
              </p>
            </div>

            <p className="text-muted-foreground">
              That means every factor in our product is at most 0.25—and usually much less.
              What does a product of ten 0.25s look like?
            </p>

            <p className="text-muted-foreground">
              Think of the <strong>telephone game</strong>. A message is passed through 10 people.
              Each person loses some fidelity—like multiplying by 0.25. By person 10, the message
              is unintelligible. That is exactly what happens to the error signal as it flows
              backward through the network.
            </p>

            {/* Layer-by-layer gradient table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Layer</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Hops from output</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Gradient magnitude</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  {[
                    { layer: 10, hops: 0, grad: '1.0' },
                    { layer: 9, hops: 1, grad: '0.25' },
                    { layer: 8, hops: 2, grad: '0.0625' },
                    { layer: 7, hops: 3, grad: '0.0156' },
                    { layer: 6, hops: 4, grad: '0.00391' },
                    { layer: 5, hops: 5, grad: '9.77 × 10⁻⁴' },
                    { layer: 4, hops: 6, grad: '2.44 × 10⁻⁴' },
                    { layer: 3, hops: 7, grad: '6.10 × 10⁻⁵' },
                    { layer: 2, hops: 8, grad: '1.53 × 10⁻⁵' },
                    { layer: 1, hops: 9, grad: '9.54 × 10⁻⁷' },
                  ].map((row) => (
                    <tr key={row.layer} className="border-b border-border/50">
                      <td className="py-2 px-3 font-mono">{row.layer}</td>
                      <td className="py-2 px-3 font-mono">{row.hops}</td>
                      <td className={cn(
                        'py-2 px-3 font-mono',
                        row.hops >= 7 ? 'text-red-400' : row.hops >= 4 ? 'text-yellow-400' : ''
                      )}>
                        {row.grad}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Layer 10 receives a gradient of 1.0. Layer 1 receives{' '}
              <InlineMath math="0.25^9 \approx 0.00000095" />. Layer 1 learns{' '}
              <strong>a million times slower</strong> than layer 10. The network is not
              broken—layer 10 learns fine. But layer 1 is effectively frozen.
            </p>

            {/* Misconception #1: vanishing != zero */}
            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">Vanishing does not mean zero.</strong>{' '}
              The gradient is 0.00000095—not zero. Weight updates happen, but they are so
              small that the weights barely change. Dying ReLU is different: when ReLU outputs
              zero, the gradient is <em>exactly</em> zero. That is instant death. Vanishing
              gradients is death by a thousand multiplications—slow, silent, and easy to miss.
            </div>

            {/* Misconception #2: just increase the learning rate */}
            <div className="space-y-2 mt-4">
              <p className="text-muted-foreground">
                Intuitive fix: if gradients are too small, just increase the learning rate?
                Let&apos;s try it:
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">Layer</th>
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">Gradient</th>
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">Step (lr=100)</th>
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">Result</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted-foreground">
                    <tr className="border-b border-border/50">
                      <td className="py-2 px-3 font-mono">10</td>
                      <td className="py-2 px-3 font-mono">1.0</td>
                      <td className="py-2 px-3 font-mono">100</td>
                      <td className="py-2 px-3 text-red-400 font-semibold">Diverges!</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-mono">1</td>
                      <td className="py-2 px-3 font-mono">0.00001</td>
                      <td className="py-2 px-3 font-mono">0.001</td>
                      <td className="py-2 px-3 text-emerald-400">Finally reasonable</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p className="text-muted-foreground">
                A learning rate that helps layer 1 destroys layer 10. A <strong>global</strong>{' '}
                knob cannot fix a <strong>per-layer</strong> problem. You saw this same principle
                in Optimizers—one learning rate cannot serve all parameters. Same idea, different
                scale.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Telephone Game">
            Each layer is a person passing a message. Sigmoid&apos;s max derivative (0.25) means
            each person passes at most 25% of the message. After 10 people:{' '}
            <InlineMath math="0.25^{10}" /> of the original signal remains.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Exploding Gradients */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Exploding Gradients"
            subtitle="The mirror image: when large factors multiply to infinity"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Remember, the local derivative at each layer is not just the activation
              derivative—it also includes the weight magnitudes (the{' '}
              <InlineMath math="\frac{\partial z_i}{\partial a_{i-1}} = w_i" /> terms in the chain
              rule). Sigmoid caps the activation derivative at 0.25, but if the weights are large
              enough, the combined factor per layer can exceed 1.0. For ReLU, the activation
              derivative is exactly 1.0 for active neurons, so the weight term is the dominant
              factor—large weights directly produce large local derivatives.
            </p>

            <p className="text-muted-foreground">
              What if each local derivative is <strong>2.0</strong> instead of 0.25? Same product,
              opposite direction:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Layer</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Hops from output</th>
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">Gradient magnitude</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  {[
                    { layer: 10, hops: 0, grad: '1' },
                    { layer: 9, hops: 1, grad: '2' },
                    { layer: 8, hops: 2, grad: '4' },
                    { layer: 7, hops: 3, grad: '8' },
                    { layer: 6, hops: 4, grad: '16' },
                    { layer: 5, hops: 5, grad: '32' },
                    { layer: 4, hops: 6, grad: '64' },
                    { layer: 3, hops: 7, grad: '128' },
                    { layer: 2, hops: 8, grad: '256' },
                    { layer: 1, hops: 9, grad: '1,024' },
                  ].map((row) => (
                    <tr key={row.layer} className="border-b border-border/50">
                      <td className="py-2 px-3 font-mono">{row.layer}</td>
                      <td className="py-2 px-3 font-mono">{row.hops}</td>
                      <td className={cn(
                        'py-2 px-3 font-mono',
                        row.hops >= 7 ? 'text-red-400' : row.hops >= 4 ? 'text-yellow-400' : ''
                      )}>
                        {row.grad}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Gradients double at each layer. After 10 layers, they are <strong>1,024x</strong>{' '}
              larger. The weights update by enormous amounts, overshooting catastrophically.
            </p>

            <p className="text-muted-foreground">
              Think of the telephone game again. Instead of each person losing fidelity, each
              person <strong>amplifies</strong> the message—shouting louder than the last. After
              10 people, the original whisper has become a deafening scream. That is what happens
              to your gradients.
            </p>

            <p className="text-muted-foreground">
              <strong>Practical symptom:</strong> your loss is decreasing nicely for hundreds of
              epochs. Then suddenly: <span className="text-red-400 font-mono font-semibold">NaN</span>.
              Training has collapsed because a gradient explosion pushed weights to infinity. In
              floating point, infinity times anything is NaN, and NaN propagates through
              everything.
            </p>

            {/* Unifying frame */}
            <div className="rounded-lg border-2 border-violet-500/30 bg-violet-500/5 p-5 text-center">
              <p className="text-violet-400 font-semibold">
                Vanishing and exploding are the <em>same</em> problem. The product of local
                derivatives is unstable—it either shrinks or grows exponentially. The only stable
                case is when each factor is close to 1.0.
              </p>
            </div>

            <p className="text-muted-foreground text-sm">
              <strong>Gradient clipping</strong> is a practical band-aid for exploding gradients:
              if the gradient magnitude exceeds a threshold, scale it down. It does not fix the root
              cause, but it prevents NaN. We will not develop it further here—just know it exists
              as a safety net.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="NaN = Explosion">
            If you see <span className="font-mono">loss: NaN</span> during training, your
            first thought should be &ldquo;exploding gradients.&rdquo; Vanishing gradients
            produce a flatline, not NaN.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 6. Check 1 — ReLU insight */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              ReLU&apos;s derivative is <InlineMath math="1" /> for positive inputs and{' '}
              <InlineMath math="0" /> for negative inputs. Based on what you just learned, why
              does ReLU reduce the vanishing gradient problem compared to sigmoid?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about the product of local derivatives. What is{' '}
              <InlineMath math="1^{10}" />? What is{' '}
              <InlineMath math="0.25^{10}" />?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  ReLU&apos;s derivative is <strong>1</strong> for positive inputs—not 0.25.
                  So the product <InlineMath math="1^{10} = 1" />: gradients pass through
                  without shrinking. Compare
                  to sigmoid: <InlineMath math="0.25^{10} \approx 10^{-6}" />.
                </p>
                <p className="text-muted-foreground">
                  But the <InlineMath math="0" />-derivative case (dying ReLU) means some paths
                  die completely. ReLU trades <em>gradual vanishing</em> for{' '}
                  <em>binary alive/dead</em>. This is a better tradeoff in practice: most neurons
                  stay alive (derivative = 1, perfect gradient flow), and the alive ones have no
                  gradient degradation at all.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Weight Initialization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Weight Initialization: The Prevention"
            subtitle="If each factor should be near 1.0, choose the right starting weights"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The vanishing/exploding problem depends on the magnitude of local derivatives. The
              magnitude of local derivatives depends on the activations. The activations depend
              on the weights. So the <strong>initial weights</strong> determine whether gradients
              vanish, explode, or flow healthily.
            </p>

            <p className="text-muted-foreground">
              In Backprop by the Numbers, we chose{' '}
              <InlineMath math="w_1 = 0.5" /> by hand. But a real network has millions of weights.
              How do you set them all? Random numbers—but from what distribution?
            </p>

            {/* Misconception #3: naive random init */}
            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">
                Your instinct might be to use random numbers between 0 and 1.
              </strong>{' '}
              That fails catastrophically. All-positive weights with mean 0.5 cause
              activations to grow at each layer. After 10 layers, outputs are in the
              thousands—and gradients are equally extreme.
            </div>

            <p className="text-muted-foreground">
              Using a normal (bell-curve) distribution centered at zero is better—at least
              the weights are not all positive. But the variance is still wrong: it does not
              account for how many inputs each layer has, so the signal still drifts as it
              passes through layers.
            </p>

            <p className="text-muted-foreground">
              The principle is simple: <strong>each layer should neither amplify nor dampen the
              signal</strong>. If the variance of the output matches the variance of the input,
              the signal is preserved.
            </p>

            <p className="text-muted-foreground">
              <strong>Variance</strong> measures how spread out values are—like the gradient
              magnitudes you just saw across layers, where some were tiny and some were huge.
              If inputs to a layer have a certain spread, the outputs should have roughly the
              same spread. Too much spread means activations explode. Too little means
              activations collapse to zero.
            </p>

            <p className="text-muted-foreground">
              The formula follows directly: a layer sums{' '}
              <InlineMath math="n_{\text{in}}" /> terms (<InlineMath math="w_i \cdot x_i" />),
              and the variance of that sum is{' '}
              <InlineMath math="n_{\text{in}} \cdot \text{Var}(w)" />. To keep output variance
              equal to input variance, we need{' '}
              <InlineMath math="n_{\text{in}} \cdot \text{Var}(w) = 1" />, which gives{' '}
              <InlineMath math="\text{Var}(w) = 1/n_{\text{in}}" />.
            </p>

            <ComparisonRow
              left={{
                title: 'Xavier Initialization',
                color: 'blue',
                items: [
                  'Var(w) = 1/n_in',
                  'Scale weights inversely with input count',
                  'More inputs → smaller weights → same output spread',
                  'Designed for sigmoid and tanh',
                ],
              }}
              right={{
                title: 'He Initialization',
                color: 'emerald',
                items: [
                  'Var(w) = 2/n_in',
                  'Same idea, 2x the variance',
                  'Accounts for ReLU zeroing ~50% of neurons',
                  'Designed for ReLU activations',
                ],
              }}
            />

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">
                Xavier initialization (for sigmoid/tanh):
              </p>
              <BlockMath math="\text{Var}(w) = \frac{1}{n_{\text{in}}}" />
              <p className="text-sm text-muted-foreground font-semibold mt-2">
                He initialization (for ReLU):
              </p>
              <BlockMath math="\text{Var}(w) = \frac{2}{n_{\text{in}}}" />
              <p className="text-xs text-muted-foreground">
                <InlineMath math="n_{\text{in}}" /> is the number of inputs to the layer. More
                inputs means each weight should be smaller, so the weighted sum stays in a
                reasonable range.
              </p>
            </div>

            <p className="text-muted-foreground">
              Why the extra factor of 2 for He? ReLU zeros out approximately half the neurons
              (negative inputs become 0). That halves the effective signal power. Doubling the
              variance compensates for the lost half.
            </p>

            {/* Before/after comparison */}
            <GradientCard title="Before and After: Layer Output Variance Through 10 Layers" color="blue">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-sm font-semibold text-red-400 mb-1">Naive Uniform [0, 1]</p>
                  <p className="text-xs text-muted-foreground">
                    Layer 1: 1.0 &rarr; Layer 3: 8.2 &rarr; Layer 5: 340 &rarr; Layer 8: 28,000 &rarr; Layer 10: 1.2M
                  </p>
                  <p className="text-xs text-red-400 mt-1">Signal explodes. Training is impossible.</p>
                </div>
                <div>
                  <p className="text-sm font-semibold text-emerald-400 mb-1">Xavier</p>
                  <p className="text-xs text-muted-foreground">
                    Layer 1: 1.0 &rarr; Layer 3: 0.95 &rarr; Layer 5: 0.91 &rarr; Layer 8: 0.88 &rarr; Layer 10: 0.85
                  </p>
                  <p className="text-xs text-emerald-400 mt-1">Signal stays stable. Gradients flow healthily.</p>
                </div>
              </div>
            </GradientCard>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Design Principle">
            Xavier and He are not magic numbers. They follow from one principle:{' '}
            <strong>each layer should preserve the signal magnitude</strong>. If the product
            of all scaling factors is near 1.0, gradients neither vanish nor explode.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 8. Batch Normalization */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Batch Normalization"
            subtitle="Stabilizing gradients during training, not just at startup"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Initialization sets the right starting point. But during training, weights change—and
              as layers 1 through 4 update their weights, the input distribution to layer 5 shifts.
              Layer 5 is trying to learn a moving target. This makes deep training unstable even
              with good initialization.
            </p>

            <p className="text-muted-foreground">
              You already normalize input data before training—subtract the mean, divide by the
              standard deviation. <strong>Batch normalization</strong> does the same thing, but{' '}
              <em>between every layer</em>, <em>during training</em>.
            </p>

            <div className="py-4 px-6 bg-muted/50 rounded-lg space-y-3">
              <p className="text-sm text-muted-foreground font-semibold">What batch norm does at each layer:</p>
              <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-1 ml-2">
                <li>Compute the mean and variance of activations across the current mini-batch</li>
                <li>Normalize to mean = 0, variance = 1</li>
                <li>Apply learned scale (<InlineMath math="\gamma" />) and shift (<InlineMath math="\beta" />) parameters</li>
              </ol>
              <BlockMath math="\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta" />
              <p className="text-xs text-muted-foreground">
                <InlineMath math="\gamma" /> and <InlineMath math="\beta" /> are <strong>learned</strong>{' '}
                during training. Forcing everything to mean=0, variance=1 might be too restrictive—sometimes the
                network needs a different distribution. These parameters let the network learn the optimal normalization.
              </p>
            </div>

            {/* Misconception #4 */}
            <div className="rounded-md border border-rose-500/30 bg-rose-500/5 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-rose-400">
                Batch norm is NOT just data preprocessing.
              </strong>{' '}
              Input normalization happens <em>once</em>, before training, on the raw data.
              Batch norm happens at <em>every layer</em>, at <em>every training step</em>,
              and the normalization parameters are <em>learned</em>. If it were just
              preprocessing, you would only need it at the input—but the whole point is that
              intermediate layers have shifting distributions.
            </div>

            <GradientCard title="Practical Impact" color="emerald">
              <ul className="space-y-1 text-sm">
                <li>&bull; Train deeper networks that would otherwise fail</li>
                <li>&bull; Use higher learning rates (the normalization absorbs large updates)</li>
                <li>&bull; Less sensitive to initialization choice</li>
                <li>&bull; Acts as a mild regularizer (the per-batch statistics add noise)</li>
              </ul>
            </GradientCard>

            {/* Before/after: 20-layer training with vs without batch norm */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <svg viewBox="0 0 400 120" className="w-full" style={{ maxHeight: 140 }}>
                {/* Axes */}
                <line x1="40" y1="100" x2="380" y2="100" stroke="#666" strokeWidth="1" />
                <line x1="40" y1="10" x2="40" y2="100" stroke="#666" strokeWidth="1" />
                <text x="210" y="118" textAnchor="middle" fill="#888" fontSize="9">Epoch</text>
                <text x="12" y="55" textAnchor="middle" fill="#888" fontSize="9" transform="rotate(-90, 12, 55)">Loss</text>

                {/* Without BN: 20-layer ReLU+He, slow convergence / flatline */}
                <path
                  d="M 50,88 Q 100,82 150,78 Q 200,75 250,73 Q 300,72 370,71"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                />

                {/* With BN: 20-layer ReLU+He, healthy convergence */}
                <path
                  d="M 50,90 Q 100,55 150,32 Q 200,22 250,18 Q 300,16 370,15"
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="2"
                />

                {/* Legend */}
                <line x1="80" y1="8" x2="100" y2="8" stroke="#ef4444" strokeWidth="2" />
                <text x="104" y="11" fill="#ef4444" fontSize="9">20-layer, no BN</text>
                <line x1="230" y1="8" x2="250" y2="8" stroke="#22c55e" strokeWidth="2" />
                <text x="254" y="11" fill="#22c55e" fontSize="9">20-layer + batch norm</text>
              </svg>
              <p className="text-xs text-muted-foreground text-center mt-2">
                Same 20-layer ReLU network with He initialization. Without batch norm, the network
                barely converges. With batch norm, training progresses normally.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Learned Parameters">
            <p className="text-sm">
              Batch norm adds two parameters per layer: <InlineMath math="\gamma" /> (scale)
              and <InlineMath math="\beta" /> (shift).
            </p>
            <p className="text-sm mt-2">
              If the optimal behavior is &ldquo;no normalization,&rdquo; the network can learn{' '}
              <InlineMath math="\gamma = \sigma" /> and <InlineMath math="\beta = \mu" /> to undo
              the normalization entirely.
            </p>
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 9. Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel
            title="Training Dynamics Explorer"
            subtitle="See how depth, activation, initialization, and batch norm affect gradient flow"
          >
            <TrainingDynamicsExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Guided Experiments">
            <ol className="space-y-2 text-sm list-decimal list-inside">
              <li>Set <strong>10 layers, Sigmoid, Naive Uniform</strong>. Watch the gradients vanish (early layers are red).</li>
              <li>Switch to <strong>Xavier</strong> initialization. Watch the gradient bars become green and uniform.</li>
              <li>Switch to <strong>ReLU + Xavier</strong>. Notice some bars might be larger—ReLU passes gradients of 1.0, not 0.25.</li>
              <li>Switch to <strong>He</strong> initialization with ReLU. Gradients stabilize.</li>
              <li>Set <strong>20 layers, ReLU, He</strong>. Now turn on <strong>Batch Normalization</strong>. Compare the training curves.</li>
            </ol>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 10. The Historical Story */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Historical Arc"
            subtitle="How three breakthroughs unlocked depth"
          />
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-3">
              <GradientCard title="Pre-2012" color="amber">
                <ul className="space-y-1 text-sm">
                  <li>&bull; Sigmoid activations</li>
                  <li>&bull; Random initialization</li>
                  <li>&bull; 2-3 layers max</li>
                  <li>&bull; Vanishing gradients everywhere</li>
                </ul>
              </GradientCard>
              <GradientCard title="2012: AlexNet" color="blue">
                <ul className="space-y-1 text-sm">
                  <li>&bull; ReLU replaces sigmoid</li>
                  <li>&bull; Gradient flow improved</li>
                  <li>&bull; 8 layers, GPU training</li>
                  <li>&bull; Deep learning takes off</li>
                </ul>
              </GradientCard>
              <GradientCard title="2015: ResNet" color="emerald">
                <ul className="space-y-1 text-sm">
                  <li>&bull; He initialization + BN</li>
                  <li>&bull; Skip connections (teased)</li>
                  <li>&bull; 152 layers (!)</li>
                  <li>&bull; Deeper than ever before</li>
                </ul>
              </GradientCard>
            </div>

            <p className="text-muted-foreground">
              Each technique in this lesson solved a real barrier. Sigmoid to ReLU fixed gradient
              shrinking. Random init to Xavier/He fixed signal degradation at startup. Batch norm
              fixed training instability. Together, they turned &ldquo;deep learning&rdquo; from
              2-3 layers into hundreds of layers.
            </p>

            <p className="text-muted-foreground text-sm">
              There is one more technique—<strong>skip connections</strong>—that pushed depth even
              further by letting gradients flow directly through shortcut paths. We will encounter
              that in a future module.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="The Modern Baseline">
            <strong>ReLU + He initialization + batch normalization</strong> is the starting point
            for most deep networks today. Start here, then adjust based on your specific problem.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              You are training a 15-layer network with ReLU activations and He initialization.
              Training loss decreases for 100 epochs, then suddenly becomes{' '}
              <span className="font-mono text-red-400">NaN</span>. Your colleague says
              &ldquo;the gradients are vanishing.&rdquo; What is actually happening, and what
              would you try?
            </p>
            <p className="text-sm text-muted-foreground mb-3 italic">
              Think about which symptom matches which problem. Flatline vs NaN.
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  This is <strong>exploding</strong>, not vanishing. NaN means gradients grew
                  to infinity. Vanishing gradients produce a <em>flatline</em> (loss barely
                  decreases), not a sudden collapse to NaN.
                </p>
                <p className="text-muted-foreground">
                  What to try: (1) reduce the learning rate, (2) add gradient clipping as a
                  safety net, (3) add batch normalization if you have not already, (4) check
                  for numerical issues in the data (e.g., very large input values).
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Symptom Guide">
            <p className="text-sm"><strong>Flatline</strong> = vanishing (early layers frozen)</p>
            <p className="text-sm mt-1"><strong>NaN</strong> = exploding (gradients hit infinity)</p>
            <p className="text-sm mt-1">Getting these backwards wastes debugging time.</p>
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 12. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Vanishing gradients: small factors multiply to near-zero',
                description:
                  'Sigmoid’s max derivative is 0.25. After 10 layers: 0.25¹⁰ ≈ 10⁻⁶. Early layers learn a million times slower than later layers.',
              },
              {
                headline: 'Exploding gradients: large factors multiply to infinity',
                description:
                  'If local derivatives exceed 1.0, gradients grow exponentially. Symptom: loss suddenly becomes NaN.',
              },
              {
                headline: 'Both are the same root cause',
                description:
                  'Gradients are products of local derivatives. The product is only stable when each factor is close to 1.0.',
              },
              {
                headline: 'Xavier (1/nᵢₙ) and He (2/nᵢₙ) initialization preserve signal',
                description:
                  'Xavier for sigmoid/tanh, He for ReLU. They ensure each layer neither amplifies nor dampens the signal, keeping gradient products near 1.0.',
              },
              {
                headline: 'Batch normalization stabilizes during training',
                description:
                  'Normalizes activations between layers at every step. Learned γ and β parameters let the network find the right distribution. Allows deeper networks and higher learning rates.',
              },
              {
                headline: 'ReLU + He init + batch norm = modern baseline',
                description:
                  'This combination turned deep learning from 2-3 layers into hundreds. Start here for any new deep network.',
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
                title: 'Understanding the difficulty of training deep feedforward neural networks',
                authors: 'Glorot & Bengio, 2010',
                url: 'http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf',
                note: 'Introduced Xavier initialization and analyzed vanishing/exploding gradients in deep networks.',
              },
              {
                title: 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification',
                authors: 'He, Zhang, Ren & Sun, 2015',
                url: 'https://arxiv.org/abs/1502.01852',
                note: 'Introduced He initialization for ReLU networks, enabling training of much deeper architectures.',
              },
              {
                title: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift',
                authors: 'Ioffe & Szegedy, 2015',
                url: 'https://arxiv.org/abs/1502.03167',
                note: 'Introduced batch normalization, stabilizing training and allowing higher learning rates.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 13. Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app"
            title="Next: Overfitting and Regularization"
            description="Your deep network trains now. But training well is not the same as generalizing well. Next: when your network memorizes the training data instead of learning the underlying pattern—and the techniques that prevent it."
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
