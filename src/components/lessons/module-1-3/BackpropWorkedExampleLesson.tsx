'use client'

import { useState } from 'react'
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
  ConstraintBlock,
} from '@/components/lessons'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

/**
 * Backprop by the Numbers — Worked Example
 *
 * Lesson 2 of Module 1.3 (Training Neural Networks)
 *
 * Takes the conceptual framework from lesson 1 (chain rule, forward/backward
 * passes) and makes it concrete: every computation traced with actual numbers
 * through a 2-layer network.
 *
 * Key concepts:
 * - Multi-layer gradient computation with real numbers
 * - Numerical gradient verification
 * - Forward pass -> backward pass -> update with concrete values
 */

// Network computation helpers
function relu(x: number): number {
  return Math.max(0, x)
}

function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0
}

function computeForwardPass(x: number, w1: number, b1: number, w2: number, b2: number) {
  const z1 = w1 * x + b1
  const a1 = relu(z1)
  const yHat = w2 * a1 + b2
  return { z1, a1, yHat }
}

function computeBackwardPass(
  x: number,
  y: number,
  w2: number,
  z1: number,
  a1: number,
  yHat: number,
) {
  const dLdyHat = -2 * (y - yHat)
  const dLdw2 = dLdyHat * a1
  const dLdb2 = dLdyHat
  const dLda1 = dLdyHat * w2
  const dLdz1 = dLda1 * reluDerivative(z1)
  const dLdw1 = dLdz1 * x
  const dLdb1 = dLdz1
  return { dLdyHat, dLdw2, dLdb2, dLda1, dLdz1, dLdw1, dLdb1 }
}

function fmt(n: number, decimals = 4): string {
  return n.toFixed(decimals)
}

function ValueBox({ label, value, color = 'blue' }: { label: string; value: string; color?: string }) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    emerald: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    rose: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
    amber: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
    violet: 'bg-violet-500/10 border-violet-500/30 text-violet-400',
  }
  return (
    <div className={`rounded-md border px-3 py-1.5 text-sm font-mono ${colorMap[color] ?? colorMap.blue}`}>
      <span className="text-muted-foreground text-xs">{label}</span>
      <span className="ml-2 font-semibold">{value}</span>
    </div>
  )
}

function NumberStep({
  step,
  formula,
  result,
  explanation,
}: {
  step: number
  formula: string
  result: string
  explanation?: string
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-bold text-primary">
        {step}
      </div>
      <div className="space-y-1">
        <div className="font-mono text-sm">
          <InlineMath math={`${formula} = ${result}`} />
        </div>
        {explanation && (
          <p className="text-xs text-muted-foreground">{explanation}</p>
        )}
      </div>
    </div>
  )
}

function BackpropCalculatorWidget() {
  const [w1, setW1] = useState(0.5)
  const [b1, setB1] = useState(0.1)
  const [w2, setW2] = useState(-0.3)
  const [b2, setB2] = useState(0.2)
  const [xVal] = useState(2)
  const [yVal] = useState(1)
  const [showBackward, setShowBackward] = useState(false)
  const [stepCount, setStepCount] = useState(0)

  const lr = 0.1
  const { z1, a1, yHat } = computeForwardPass(xVal, w1, b1, w2, b2)
  const loss = (yVal - yHat) ** 2
  const backward = computeBackwardPass(xVal, yVal, w2, z1, a1, yHat)

  function handleStep() {
    setW1(prev => prev - lr * backward.dLdw1)
    setB1(prev => prev - lr * backward.dLdb1)
    setW2(prev => prev - lr * backward.dLdw2)
    setB2(prev => prev - lr * backward.dLdb2)
    setStepCount(prev => prev + 1)
  }

  function handleReset() {
    setW1(0.5)
    setB1(0.1)
    setW2(-0.3)
    setB2(0.2)
    setStepCount(0)
    setShowBackward(false)
  }

  return (
    <div className="space-y-4">
      {/* Network diagram */}
      <div className="rounded-lg border bg-muted/30 p-4">
        <div className="flex items-center justify-between gap-2 text-sm flex-wrap">
          <div className="flex items-center gap-1">
            <div className="rounded bg-blue-500/20 px-2 py-1 font-mono text-blue-400">
              x={xVal}
            </div>
            <span className="text-muted-foreground">→</span>
          </div>
          <div className="flex flex-col items-center gap-0.5">
            <span className="text-xs text-muted-foreground">Layer 1</span>
            <div className="rounded border border-violet-500/30 bg-violet-500/10 px-2 py-1 font-mono text-xs text-violet-400">
              w1={fmt(w1,3)} b1={fmt(b1,3)}
            </div>
          </div>
          <div className="flex flex-col items-center gap-0.5">
            <span className="text-xs text-muted-foreground">ReLU</span>
            <div className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 font-mono text-xs text-amber-400">
              {fmt(z1,3)} → {fmt(a1,3)}
            </div>
          </div>
          <div className="flex flex-col items-center gap-0.5">
            <span className="text-xs text-muted-foreground">Layer 2</span>
            <div className="rounded border border-violet-500/30 bg-violet-500/10 px-2 py-1 font-mono text-xs text-violet-400">
              w2={fmt(w2,3)} b2={fmt(b2,3)}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">→</span>
            <div className="rounded bg-emerald-500/20 px-2 py-1 font-mono text-emerald-400">
              y&#x302;={fmt(yHat,3)}
            </div>
          </div>
        </div>
      </div>

      {/* Forward pass values */}
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-muted-foreground">Forward Pass</h4>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5">
          <ValueBox label="z1" value={fmt(z1)} color="blue" />
          <ValueBox label="a1" value={fmt(a1)} color="blue" />
          <ValueBox label="y&#x302;" value={fmt(yHat)} color="emerald" />
          <ValueBox label="Loss" value={fmt(loss)} color={loss < 0.01 ? 'emerald' : 'rose'} />
          <ValueBox label="Target" value={String(yVal)} color="amber" />
        </div>
      </div>

      {/* Backward pass values */}
      {showBackward && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-muted-foreground">Backward Pass (Gradients)</h4>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">
            <ValueBox label="dL/dy&#x302;" value={fmt(backward.dLdyHat)} color="rose" />
            <ValueBox label="dL/dw2" value={fmt(backward.dLdw2)} color="rose" />
            <ValueBox label="dL/db2" value={fmt(backward.dLdb2)} color="rose" />
            <ValueBox label="dL/da1" value={fmt(backward.dLda1)} color="rose" />
            <ValueBox label="dL/dz1" value={fmt(backward.dLdz1)} color="rose" />
            <ValueBox label="dL/dw1" value={fmt(backward.dLdw1)} color="rose" />
            <ValueBox label="dL/db1" value={fmt(backward.dLdb1)} color="rose" />
          </div>
          {z1 <= 0 && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-400">
              ReLU killed the gradient! z1 = {fmt(z1)} ≤ 0, so Layer 1 gets zero gradient.
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        {!showBackward && (
          <button
            onClick={() => setShowBackward(true)}
            className="cursor-pointer rounded-md bg-rose-500/20 px-3 py-1.5 text-sm font-medium text-rose-400 hover:bg-rose-500/30 transition-colors"
          >
            Run Backward Pass
          </button>
        )}
        {showBackward && (
          <button
            onClick={handleStep}
            className="cursor-pointer rounded-md bg-emerald-500/20 px-3 py-1.5 text-sm font-medium text-emerald-400 hover:bg-emerald-500/30 transition-colors"
          >
            Apply Update (lr={lr})
          </button>
        )}
        <button
          onClick={handleReset}
          className="cursor-pointer rounded-md bg-muted px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-muted/80 transition-colors"
        >
          Reset
        </button>
        {stepCount > 0 && (
          <span className="text-xs text-muted-foreground">
            Step {stepCount}—Loss: {fmt(loss)}
          </span>
        )}
      </div>

      {/* Parameter sliders */}
      <details className="text-sm">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
          Adjust parameters manually
        </summary>
        <div className="mt-3 grid grid-cols-2 gap-4">
          {[
            { label: 'w1', value: w1, setter: setW1, min: -2, max: 2 },
            { label: 'b1', value: b1, setter: setB1, min: -2, max: 2 },
            { label: 'w2', value: w2, setter: setW2, min: -2, max: 2 },
            { label: 'b2', value: b2, setter: setB2, min: -2, max: 2 },
          ].map(({ label, value, setter, min, max }) => (
            <div key={label} className="space-y-1">
              <label className="flex justify-between text-xs text-muted-foreground">
                <span>{label}</span>
                <span className="font-mono">{fmt(value, 3)}</span>
              </label>
              <input
                type="range"
                min={min}
                max={max}
                step={0.01}
                value={value}
                onChange={e => {
                  setter(parseFloat(e.target.value))
                  setShowBackward(false)
                  setStepCount(0)
                }}
                className="w-full cursor-pointer"
              />
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}

export function BackpropWorkedExampleLesson() {
  // Pre-computed values for the worked example
  const x = 2, w1 = 0.5, b1 = 0.1, w2 = -0.3, b2 = 0.2, y = 1
  const z1 = w1 * x + b1 // = 1.1
  const a1 = relu(z1)     // = 1.1
  const yHat = w2 * a1 + b2 // = -0.13
  const loss = (y - yHat) ** 2 // = 1.2769

  const dLdyHat = -2 * (y - yHat) // = -2.26
  const dLdw2 = dLdyHat * a1       // = -2.486
  const dLdb2 = dLdyHat            // = -2.26
  const dLda1 = dLdyHat * w2       // = 0.678
  const dLdz1 = dLda1 * 1          // = 0.678 (ReLU deriv = 1 since z1 > 0)
  const dLdw1 = dLdz1 * x          // = 1.356
  const dLdb1 = dLdz1              // = 0.678

  // Updated weights
  const lr = 0.1
  const w1New = w1 - lr * dLdw1
  const b1New = b1 - lr * dLdb1
  const w2New = w2 - lr * dLdw2
  const b2New = b2 - lr * dLdb2

  // Recompute loss after update
  const { yHat: yHatNew } = computeForwardPass(x, w1New, b1New, w2New, b2New)
  const lossNew = (y - yHatNew) ** 2

  // Numerical gradient check
  const eps = 0.001
  const { yHat: yHatEps } = computeForwardPass(x, w1 + eps, b1, w2, b2)
  const lossEps = (y - yHatEps) ** 2
  const numericalGrad = (lossEps - loss) / eps

  return (
    <LessonLayout>
      {/* Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Backprop by the Numbers"
            description="Trace every gradient through a real neural network—with actual numbers, not just symbols."
            category="Training Neural Networks"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Compute every gradient in a 2-layer neural network by hand, using real
            numbers, and verify the results.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="From Concept to Concrete">
            In Backpropagation: How Networks Learn, you saw the algorithm
            symbolically. Now you&apos;ll trace it with actual numbers and see
            exactly why it works.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'One tiny network: 2 layers, 1 neuron each, 4 parameters',
              'Real numbers at every step—nothing hidden',
              'No matrices needed—each layer has one neuron',
            ]}
          />
        </Row.Content>
      </Row>

      {/* Motivation */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Backpropagation: How Networks Learn, you saw the algorithm symbolically—the
              chain rule, forward and backward passes, the &quot;Local &times; Local &times;
              Local&quot; insight. You understood the structure.
            </p>
            <p className="text-muted-foreground">
              But there&apos;s a difference between watching a magic trick and performing
              one yourself. Symbols show the pattern; <strong>numbers prove you can do
              it</strong>. This lesson is where you perform the trick—we&apos;ll trace
              every computation with actual numbers, and by the end you&apos;ll have
              computed 4 gradients by hand and verified them.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Section 1: The Network */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Network"
            subtitle="Our test subject: the simplest deep network"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here&apos;s our network. It&apos;s the smallest network that qualifies
              as &quot;deep&quot;—it has a hidden layer between input and output:
            </p>

            <div className="rounded-lg border bg-muted/30 p-6">
              <div className="flex items-center justify-center gap-3 text-sm flex-wrap">
                <div className="rounded-lg bg-blue-500/20 px-4 py-2 text-center">
                  <div className="text-xs text-muted-foreground">Input</div>
                  <div className="font-mono font-bold text-blue-400">x = {x}</div>
                </div>
                <span className="text-muted-foreground font-mono">→ ×w1 + b1 →</span>
                <div className="rounded-lg bg-amber-500/20 px-4 py-2 text-center">
                  <div className="text-xs text-muted-foreground">ReLU</div>
                  <div className="font-mono text-amber-400">max(0, z1)</div>
                </div>
                <span className="text-muted-foreground font-mono">→ ×w2 + b2 →</span>
                <div className="rounded-lg bg-emerald-500/20 px-4 py-2 text-center">
                  <div className="text-xs text-muted-foreground">Output</div>
                  <div className="font-mono font-bold text-emerald-400">y&#x302;</div>
                </div>
                <span className="text-muted-foreground font-mono">→ L = (y - y&#x302;)²</span>
              </div>
            </div>

            <p className="text-muted-foreground">
              We have <strong>4 parameters</strong> to learn: {' '}
              <InlineMath math="w_1, b_1, w_2, b_2" />. Let&apos;s assign starting values:
            </p>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-md border bg-violet-500/10 border-violet-500/30 px-3 py-2 text-center">
                <div className="text-xs text-muted-foreground">w1</div>
                <div className="font-mono font-bold text-violet-400">{w1}</div>
              </div>
              <div className="rounded-md border bg-violet-500/10 border-violet-500/30 px-3 py-2 text-center">
                <div className="text-xs text-muted-foreground">b1</div>
                <div className="font-mono font-bold text-violet-400">{b1}</div>
              </div>
              <div className="rounded-md border bg-violet-500/10 border-violet-500/30 px-3 py-2 text-center">
                <div className="text-xs text-muted-foreground">w2</div>
                <div className="font-mono font-bold text-violet-400">{w2}</div>
              </div>
              <div className="rounded-md border bg-violet-500/10 border-violet-500/30 px-3 py-2 text-center">
                <div className="text-xs text-muted-foreground">b2</div>
                <div className="font-mono font-bold text-violet-400">{b2}</div>
              </div>
            </div>

            <p className="text-muted-foreground">
              Our target output is <InlineMath math="y = 1" />. Right now, the network
              will predict something wrong. Our job: compute exactly how each parameter
              should change.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Why 1 Neuron per Layer?">
            Real networks have many neurons per layer (requiring matrix math). But
            the gradient computation principle is identical—you just repeat it for
            more paths. One neuron per layer isolates the chain rule mechanics.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* Section 2: Forward Pass */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Forward Pass"
            subtitle="Compute the prediction, saving every intermediate value"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              The forward pass pushes the input through the network to get a
              prediction. We save every intermediate value—we&apos;ll need them
              in the backward pass.
            </p>

            <PhaseCard number={1} title="Forward Pass" subtitle="Input → Prediction" color="blue">
              <div className="space-y-3">
                <NumberStep
                  step={1}
                  formula="z_1 = w_1 \\cdot x + b_1 = 0.5 \\times 2 + 0.1"
                  result="1.1"
                  explanation="Linear combination in layer 1"
                />
                <NumberStep
                  step={2}
                  formula="a_1 = \\text{ReLU}(1.1) = \\max(0, 1.1)"
                  result="1.1"
                  explanation="Activation: 1.1 > 0, so ReLU passes it through"
                />
                <NumberStep
                  step={3}
                  formula="\\hat{y} = w_2 \\cdot a_1 + b_2 = -0.3 \\times 1.1 + 0.2"
                  result="-0.13"
                  explanation="Layer 2 output (no activation on the output for regression)"
                />
                <NumberStep
                  step={4}
                  formula="L = (y - \\hat{y})^2 = (1 - (-0.13))^2 = (1.13)^2"
                  result="1.2769"
                  explanation="MSE loss—how wrong we are"
                />
              </div>
            </PhaseCard>

            <p className="text-muted-foreground">
              The network predicted <InlineMath math="\hat{y} = -0.13" /> when the
              target is <InlineMath math="y = 1" />. Not great. The loss of{' '}
              <InlineMath math="1.2769" /> tells us how wrong we are.
            </p>

            <p className="text-muted-foreground">
              Notice we saved <InlineMath math="z_1, a_1, \hat{y}" />. These
              intermediate values are crucial—the backward pass will use every
              single one.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Why Save Values?">
            Every intermediate value becomes an input to a derivative in the
            backward pass. <InlineMath math="a_1" /> shows up in{' '}
            <InlineMath math="\partial L / \partial w_2" />,{' '}
            <InlineMath math="z_1" /> determines the ReLU derivative,
            and <InlineMath math="\hat{y}" /> appears in the loss derivative.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 3: Backward Pass */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Backward Pass"
            subtitle="Trace the gradients from loss back through every layer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now the interesting part. We start at the loss and work backward,
              computing the gradient at each step. Each step uses the chain rule:
              multiply the incoming gradient by the local derivative.
            </p>

            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-muted-foreground">
              <strong className="text-amber-400">Don&apos;t confuse &quot;backward&quot; with
              &quot;reverse.&quot;</strong> The forward pass computes activations
              (multiply, add, ReLU). The backward pass computes <em>derivatives</em>—completely
              different operations. &quot;Backward&quot; refers to the direction
              (from loss toward input), not running the same operations in reverse.
            </div>

            <PhaseCard number={2} title="Backward Pass" subtitle="Loss → Gradients" color="rose">
              <div className="space-y-4">
                {/* Step 1: Loss derivative */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Start at the loss
                  </h4>
                  <NumberStep
                    step={1}
                    formula="\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(1 - (-0.13))"
                    result="-2.26"
                    explanation="How does the loss change when y-hat changes? This is the 'error signal' that flows backward."
                  />
                </div>

                {/* Step 2: Layer 2 gradients */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Layer 2 gradients
                  </h4>
                  <NumberStep
                    step={2}
                    formula="\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot a_1 = -2.26 \times 1.1"
                    result="-2.486"
                    explanation="Chain rule: error signal × local derivative. The local derivative of y-hat = w2·a1 + b2 with respect to w2 is a1."
                  />
                  <NumberStep
                    step={3}
                    formula="\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial \hat{y}} \cdot 1"
                    result="-2.26"
                    explanation="For the bias, the local derivative is always 1."
                  />
                </div>

                {/* Step 3: Pass gradient to layer 1 */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Pass the gradient to Layer 1
                  </h4>
                  <NumberStep
                    step={4}
                    formula="\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial \hat{y}} \cdot w_2 = -2.26 \times (-0.3)"
                    result="0.678"
                    explanation="This is the gradient flowing INTO layer 1—what layer 1 'receives from above.'"
                  />
                </div>

                {/* Step 4: Through ReLU */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Through the ReLU gate
                  </h4>
                  <NumberStep
                    step={5}
                    formula="\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \times 1"
                    result="0.678"
                    explanation={`ReLU derivative is 1 (since z1 = ${fmt(z1)} > 0). The gradient passes through unchanged.`}
                  />
                </div>

                {/* Step 5: Layer 1 gradients */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-muted-foreground">
                    Layer 1 gradients
                  </h4>
                  <NumberStep
                    step={6}
                    formula="\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \cdot x = 0.678 \times 2"
                    result="1.356"
                    explanation="Same pattern as layer 2: incoming gradient × local derivative."
                  />
                  <NumberStep
                    step={7}
                    formula="\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot 1"
                    result="0.678"
                    explanation="Bias gradient: incoming gradient × 1."
                  />
                </div>
              </div>
            </PhaseCard>

            <p className="text-muted-foreground">
              That&apos;s it. Seven steps, and we have gradients for all 4 parameters.
              Every step followed the same three-part recipe:
            </p>
            <ol className="list-decimal list-inside text-muted-foreground space-y-1 ml-4 text-sm">
              <li>Receive the gradient from the layer above</li>
              <li>Compute the local derivative (how does this operation&apos;s output change with respect to its input?)</li>
              <li>Multiply them together—that&apos;s your gradient to pass down or use for the weight update</li>
            </ol>
            <p className="text-muted-foreground">
              This recipe works for <em>any</em> network, <em>any</em> size. The same three steps, repeated for every layer.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Pattern">
            Every backward step follows the same recipe:
            <ol className="mt-2 space-y-1 text-sm list-decimal list-inside">
              <li>Receive gradient from above</li>
              <li>Compute local derivative</li>
              <li>Multiply them together</li>
            </ol>
            That&apos;s it. Any network, any size, same three steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Check 1: Trace the chain */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              The gradient <InlineMath math="\partial L / \partial w_1" /> required
              multiplying 4 local derivatives together. Can you trace them?
            </p>
            <div className="rounded-md bg-muted/50 p-3">
              <BlockMath math="\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial \hat{y}}}_{-2.26} \cdot \underbrace{\frac{\partial \hat{y}}{\partial a_1}}_{w_2 = -0.3} \cdot \underbrace{\frac{\partial a_1}{\partial z_1}}_{1} \cdot \underbrace{\frac{\partial z_1}{\partial w_1}}_{x = 2}" />
              <p className="text-xs text-muted-foreground text-center mt-2">
                Local &times; Local &times; Local &times; Local = {fmt(dLdw1)}
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Local &times; Local &times; Local">
            This is the &quot;Local &times; Local &times; Local&quot; pattern from
            Backpropagation: How Networks Learn—now
            with actual numbers. Each factor is a local derivative that only depends
            on its own layer&apos;s values.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Section 4: Dead ReLU */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="When Gradients Die"
            subtitle="What happens when ReLU blocks the signal?"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What if <InlineMath math="w_1" /> were <InlineMath math="-1" /> instead
              of <InlineMath math="0.5" />? Let&apos;s trace the forward pass:
            </p>

            <div className="rounded-md border bg-muted/30 p-4 font-mono text-sm space-y-1">
              <p>z1 = (-1) &times; 2 + 0.1 = <span className="text-rose-400 font-bold">-1.9</span></p>
              <p>a1 = ReLU(-1.9) = <span className="text-rose-400 font-bold">0</span></p>
            </div>

            <p className="text-muted-foreground">
              ReLU outputs zero because <InlineMath math="z_1 = -1.9 < 0" />. And
              since the ReLU derivative is 0 when its input is negative:
            </p>

            <div className="rounded-md bg-rose-500/10 border border-rose-500/30 p-4">
              <BlockMath math="\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \times 0 = 0" />
              <p className="text-sm text-rose-400 text-center mt-2">
                Layer 1&apos;s gradients are <strong>all zero</strong>.{' '}
                <InlineMath math="w_1" /> and <InlineMath math="b_1" /> get no
                update—they can&apos;t learn.
              </p>
            </div>

            <p className="text-muted-foreground">
              This is the &quot;dying ReLU&quot; problem mentioned in Activation Functions. When
              a ReLU neuron&apos;s pre-activation is negative, it blocks the gradient
              completely. The neuron is effectively dead—it can&apos;t recover through
              gradient descent because the gradient is exactly zero.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Dying ReLU">
            This isn&apos;t just theoretical. In real networks, neurons can die
            during training if a large gradient update pushes the pre-activation
            permanently negative. This is one reason for alternatives like Leaky
            ReLU.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Section 5: Numerical Verification */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Trust but Verify"
            subtitle="Proving backprop gives exact gradients"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              How do we know the gradients we computed are correct? There&apos;s a
              beautifully simple check: <strong>numerical differentiation</strong>.
              Nudge one parameter by a tiny amount, recompute the loss, and see if
              the change matches the gradient.
            </p>

            <div className="rounded-lg border bg-muted/30 p-4 space-y-3 text-sm">
              <p className="text-muted-foreground">
                Let&apos;s verify <InlineMath math="\partial L / \partial w_1 = 1.356" />:
              </p>
              <div className="font-mono space-y-1">
                <p>Original loss: L(w1=0.5) = {fmt(loss)}</p>
                <p>Nudged loss:  L(w1=0.501) = {fmt(lossEps)}</p>
                <p className="border-t border-muted pt-1">
                  Numerical gradient: ({fmt(lossEps)} - {fmt(loss)}) / 0.001 ={' '}
                  <span className="text-emerald-400 font-bold">{fmt(numericalGrad)}</span>
                </p>
                <p>
                  Backprop gradient: {' '}
                  <span className="text-emerald-400 font-bold">{fmt(dLdw1)}</span>
                </p>
              </div>
              <p className="text-emerald-400 text-xs">
                They match (up to floating-point precision). Backprop is exact.
              </p>
            </div>

            <p className="text-muted-foreground">
              This &quot;gradient check&quot; is a real debugging tool.
              When implementing backprop (or when a framework gives surprising
              gradients), comparing analytical gradients to numerical ones is the
              standard way to verify correctness. You&apos;ll use this technique
              again.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Gradient Checking in Practice">
            Numerical gradients are too slow for training (they need one forward
            pass per parameter). But they&apos;re perfect for debugging—compare
            them against your analytical gradients to catch implementation bugs.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Section 6: Weight Update */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Completing the Loop"
            subtitle="Apply the gradients and watch the loss drop"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              We have all 4 gradients. Now apply the update rule from Gradient Descent with a
              learning rate <InlineMath math="\alpha = 0.1" />:
            </p>

            <div className="rounded-md bg-muted/50 p-4">
              <BlockMath math="\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial \theta}" />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-md border bg-muted/30 p-3 text-sm font-mono space-y-1">
                <p className="text-xs text-muted-foreground font-sans font-semibold">Layer 1</p>
                <p>w1: {w1} - 0.1 &times; {fmt(dLdw1)} = <strong className="text-emerald-400">{fmt(w1New)}</strong></p>
                <p>b1: {b1} - 0.1 &times; {fmt(dLdb1)} = <strong className="text-emerald-400">{fmt(b1New)}</strong></p>
              </div>
              <div className="rounded-md border bg-muted/30 p-3 text-sm font-mono space-y-1">
                <p className="text-xs text-muted-foreground font-sans font-semibold">Layer 2</p>
                <p>w2: {w2} - 0.1 &times; ({fmt(dLdw2)}) = <strong className="text-emerald-400">{fmt(w2New)}</strong></p>
                <p>b2: {b2} - 0.1 &times; ({fmt(dLdb2)}) = <strong className="text-emerald-400">{fmt(b2New)}</strong></p>
              </div>
            </div>

            <p className="text-muted-foreground">
              Did it work? Rerun the forward pass with the new weights:
            </p>

            <div className="rounded-md border-2 border-emerald-500/30 bg-emerald-500/10 p-4 text-sm">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <span className="text-muted-foreground">Before: </span>
                  <span className="font-mono">Loss = {fmt(loss)}, y&#x302; = {fmt(yHat)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">After: </span>
                  <span className="font-mono text-emerald-400 font-bold">
                    Loss = {fmt(lossNew)}, y&#x302; = {fmt(yHatNew)}
                  </span>
                </div>
              </div>
              <p className="text-emerald-400 text-xs mt-2">
                Loss dropped from {fmt(loss)} to {fmt(lossNew)}. The prediction
                moved closer to the target. The network learned.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="One Complete Training Step">
            You just completed one full cycle of the training loop:
            <ol className="mt-2 space-y-0.5 text-sm list-decimal list-inside">
              <li>Forward pass → prediction</li>
              <li>Compute loss</li>
              <li>Backward pass → gradients</li>
              <li>Update parameters</li>
            </ol>
            Repeat this thousands of times and the network converges.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Backprop Calculator" subtitle="Explore how gradients flow through the network">
            <BackpropCalculatorWidget />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiments">
            <ul className="space-y-2 text-sm">
              <li>Click &quot;Run Backward Pass&quot; to see all gradients</li>
              <li>Click &quot;Apply Update&quot; repeatedly—watch the loss shrink</li>
              <li>Set w1 to a negative value—can you trigger the dead ReLU?</li>
              <li>After a dead ReLU, try clicking Update—does it recover?</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'Chain rule with real numbers',
                description: 'Backprop multiplies local derivatives step by step—the same pattern at every layer.',
              },
              {
                headline: 'No global information needed',
                description: 'Each layer only needs the gradient from above and its own local values.',
              },
              {
                headline: 'Numerical gradient checks',
                description: 'Nudge a parameter, measure the loss change, compare to backprop—a debugging tool you\'ll use forever.',
              },
              {
                headline: 'One complete training step',
                description: 'Forward + backward + update = one step. Repeat thousands of times and the network converges.',
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
            title="Next: Computational Graphs"
            description="A visual notation that makes this bookkeeping automatic"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
