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
  SummaryBlock,
  ConstraintBlock,
  ComparisonRow,
  NextStepBlock,
  TryThisBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExercisePanel } from '@/components/widgets/ExercisePanel'
import { AutogradExplorer } from '@/components/widgets/AutogradExplorer'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Autograd — Lesson 2 of Module 2.1 (PyTorch Core)
 *
 * Teaches the student to:
 * - Use requires_grad to tell PyTorch to track operations
 * - Call backward() to compute gradients automatically
 * - Access .grad to read the stored gradients
 * - Understand and handle gradient accumulation (zero_grad)
 * - Use torch.no_grad() to disable tracking during updates
 * - Use .detach() to sever a tensor from the computational graph
 *
 * Target depths:
 * - requires_grad / backward() / .grad: DEVELOPED
 * - zero_grad() / no_grad() / detach(): DEVELOPED
 *
 * Central insight: backward() performs the exact same computation
 * the student did by hand in Backprop by the Numbers. Same algorithm,
 * same numbers, different executor.
 */

export function AutogradLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="Autograd"
            description="PyTorch computes gradients for you — using the exact algorithm you already know."
            category="PyTorch Core"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Use PyTorch&apos;s autograd system to compute gradients automatically. By the
            end, you can set up tensors with <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad</code>,
            call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code>, and read
            the results from <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.grad</code>&mdash;understanding
            that this is the same computational graph traversal you did by hand in Backprop by the Numbers.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Building on Series 1">
            Everything here connects to the computational graphs and backpropagation
            you learned in Training Neural Networks. Autograd is the API for
            the algorithm you already understand.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'Autograd only — no nn.Module or layer abstractions (that\'s the next lesson)',
              'No optimizers like Adam or SGD objects (lesson 4)',
              'No full training loops — we show one manual update step as a preview',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook — "Remember Computing These By Hand?" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Remember Computing These By Hand?"
            subtitle="Four gradients, seven steps, one very long lesson"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Backprop by the Numbers, you traced through a 2-layer network with
              real numbers. Seven steps of &ldquo;incoming gradient times local
              derivative.&rdquo; You computed four parameter gradients:
            </p>

            <div className="grid gap-2 grid-cols-2 sm:grid-cols-4">
              <div className="rounded-md border bg-rose-500/10 border-rose-500/30 px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">dL/dw1</p>
                <p className="font-mono text-sm font-semibold text-rose-400">1.3560</p>
              </div>
              <div className="rounded-md border bg-rose-500/10 border-rose-500/30 px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">dL/db1</p>
                <p className="font-mono text-sm font-semibold text-rose-400">0.6780</p>
              </div>
              <div className="rounded-md border bg-rose-500/10 border-rose-500/30 px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">dL/dw2</p>
                <p className="font-mono text-sm font-semibold text-rose-400">-2.4860</p>
              </div>
              <div className="rounded-md border bg-rose-500/10 border-rose-500/30 px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">dL/db2</p>
                <p className="font-mono text-sm font-semibold text-rose-400">-2.2600</p>
              </div>
            </div>

            <p className="text-muted-foreground">
              What if all seven steps happened in <strong>one line of code</strong>?
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="The Payoff">
            You did that manual work for a reason. Every gradient you computed by
            hand is about to appear from a single function call&mdash;and you will
            know exactly what happened inside.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain Part 1 — requires_grad */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="requires_grad: Press Record"
            subtitle="Tell PyTorch to start tracking operations"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Tensors, you created PyTorch tensors and did math with them. But
              PyTorch was not recording anything. To compute gradients, you first
              need to tell PyTorch to <strong>watch</strong>:
            </p>

            <CodeBlock
              language="python"
              filename="requires_grad.py"
              code={`import torch

# A regular tensor — PyTorch does NOT track operations
x = torch.tensor(3.0)
y = x ** 2
print(y)  # tensor(9.)  — no grad_fn, nothing recorded

# A tensor with requires_grad=True — PyTorch IS tracking
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(y)  # tensor(9., grad_fn=<PowBackward0>)
#                       ^^^^^^^^^^^^^^^^^^^^^^
#          This is the computational graph being built!`}
            />

            <p className="text-muted-foreground">
              Think of <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad=True</code> like
              pressing <strong>Record</strong> on a video camera. The tensor does the same math
              either way&mdash;but when Recording is on, every operation is being logged.
              The <code className="text-sm bg-muted px-1.5 py-0.5 rounded">grad_fn</code> is the log entry:
              it tells PyTorch &ldquo;this value came from a <code className="text-sm bg-muted px-1.5 py-0.5 rounded">Pow</code> operation.&rdquo;
            </p>

            <p className="text-muted-foreground">
              This is exactly the computational graph from Computational Graphs. Each
              operation creates a node. Each <code className="text-sm bg-muted px-1.5 py-0.5 rounded">grad_fn</code> is
              a node in that graph, built live as you write forward-pass code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Not a Different Tensor">
            A tensor with <code className="text-xs">requires_grad=True</code> is
            not a special type. It has the same shape, dtype, and device. The only
            difference: PyTorch records operations on it. The flag does not change
            the data&mdash;it tells PyTorch to watch.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Explain Part 2 — backward() and .grad */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="backward() and .grad: Press Rewind"
            subtitle="Walk the graph backward, store the results"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> on
              the output. This is pressing <strong>Rewind</strong> on the recording.
              PyTorch walks through every recorded operation in reverse, applying the chain
              rule at each step:
            </p>

            <CodeBlock
              language="python"
              filename="backward.py"
              code={`x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

y.backward()     # Walk the graph backward

print(x.grad)    # tensor(6.)
# dy/dx = 2x = 2(3) = 6  ✓`}
            />

            <p className="text-muted-foreground">
              Two things to notice:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.grad</code> is <strong>where the result is stored</strong>.
                It is an attribute on the leaf tensor (the one with <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad=True</code>).
                Not a return value&mdash;an attribute.
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> is called on the <strong>output</strong> (the
                loss), not on the parameters. Gradients flow backward from the output to the leaves.
              </li>
            </ul>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Algorithm">
            <code className="text-xs">backward()</code> does exactly what you did
            in Computational Graphs: at every node, multiply the incoming gradient
            by the local derivative to get the outgoing gradient. The &ldquo;recipe&rdquo;
            is identical.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 — Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              Before running this code, predict: what
              is <code className="text-sm bg-muted px-1.5 py-0.5 rounded">x.grad</code> after <code className="text-sm bg-muted px-1.5 py-0.5 rounded">z.backward()</code>?
            </p>
            <CodeBlock
              language="python"
              code={`x = torch.tensor(2.0, requires_grad=True)
y = 3 * x + 1    # y = 7
z = y ** 2        # z = 49

z.backward()
print(x.grad)     # ?`}
            />
            <p className="text-muted-foreground text-xs mt-2 mb-3">
              Hint: trace the chain rule. <InlineMath math="\frac{dz}{dy} = 2y = 14" />,{' '}
              <InlineMath math="\frac{dy}{dx} = 3" />, so{' '}
              <InlineMath math="\frac{dz}{dx} = 14 \times 3 = \text{?}" />
            </p>
            <details className="group mt-4">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <CodeBlock
                  language="python"
                  code={`print(x.grad)  # tensor(42.)`}
                />
                <p className="text-muted-foreground">
                  <InlineMath math="dz/dx = dz/dy \times dy/dx = 2y \times 3 = 2(7) \times 3 = 42" />.
                  The chain rule in action&mdash;exactly the same computation that <code className="text-xs bg-muted px-1 rounded">backward()</code> performed
                  by walking the graph.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 6. Explain Part 3 — The Payoff */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Payoff: Same Numbers, One Line"
            subtitle="Reproducing Backprop by the Numbers in PyTorch"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now for the moment this lesson has been building toward. Set up the exact
              same network from Backprop by the Numbers&mdash;same weights, same input,
              same target:
            </p>

            <CodeBlock
              language="python"
              filename="reproduce_backprop.py"
              code={`import torch

# Same initial weights from Backprop by the Numbers
w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(-0.3, requires_grad=True)
b2 = torch.tensor(0.2, requires_grad=True)

# Same input and target
x = torch.tensor(2.0)
y_true = torch.tensor(1.0)

# Forward pass (same computation)
z1 = w1 * x + b1              # 0.5 * 2 + 0.1 = 1.1
a1 = torch.clamp(z1, min=0)   # relu(1.1) = 1.1
y_hat = w2 * a1 + b2          # -0.3 * 1.1 + 0.2 = -0.13
loss = (y_true - y_hat) ** 2  # (1 - (-0.13))^2 = 1.2769

# ONE LINE — all seven backward steps
loss.backward()

# Check the results
print(f"w1.grad = {w1.grad:.4f}")   # 1.3560
print(f"b1.grad = {b1.grad:.4f}")   # 0.6780
print(f"w2.grad = {w2.grad:.4f}")   # -2.4860
print(f"b2.grad = {b2.grad:.4f}")   # -2.2600`}
            />

            <p className="text-muted-foreground">
              Compare these to the values you computed by hand:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-muted-foreground border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4 font-semibold">Parameter</th>
                    <th className="text-left py-2 pr-4 font-semibold">Hand-computed</th>
                    <th className="text-left py-2 pr-4 font-semibold">PyTorch .grad</th>
                    <th className="text-left py-2 pr-4 font-semibold">Match?</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">w1</td>
                    <td className="py-2 pr-4 font-mono">1.3560</td>
                    <td className="py-2 pr-4 font-mono">1.3560</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">b1</td>
                    <td className="py-2 pr-4 font-mono">0.6780</td>
                    <td className="py-2 pr-4 font-mono">0.6780</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">w2</td>
                    <td className="py-2 pr-4 font-mono">-2.4860</td>
                    <td className="py-2 pr-4 font-mono">-2.4860</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-mono">b2</td>
                    <td className="py-2 pr-4 font-mono">-2.2600</td>
                    <td className="py-2 pr-4 font-mono">-2.2600</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              All that manual work&mdash;the seven steps, the local derivatives, tracking
              every intermediate value&mdash;happens inside <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code>.
              Same numbers. Same algorithm. One line.
            </p>

            <p className="text-muted-foreground">
              Here is the computational graph that PyTorch built during the forward pass.
              Forward values flow left to right (blue). When you
              call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code>, PyTorch
              walks this same graph right to left, computing gradients (red) at every node:
            </p>

            <MermaidDiagram chart={`
              graph LR
                x["x = 2.0"] --> mul1["w1*x = 1.0"]
                w1["w1 = 0.5<br/><span style='color:#f87171'>grad: 1.3560</span>"] --> mul1
                mul1 --> add1["z1 = 1.1"]
                b1["b1 = 0.1<br/><span style='color:#f87171'>grad: 0.6780</span>"] --> add1
                add1 --> relu["ReLU<br/>a1 = 1.1"]
                relu --> mul2["w2*a1 = -0.33"]
                w2["w2 = -0.3<br/><span style='color:#f87171'>grad: -2.4860</span>"] --> mul2
                mul2 --> add2["y-hat = -0.13"]
                b2["b2 = 0.2<br/><span style='color:#f87171'>grad: -2.2600</span>"] --> add2
                add2 --> mse["MSE Loss<br/>1.2769"]

                style w1 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style b1 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style w2 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style b2 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style x fill:#172554,stroke:#3b82f6,color:#e2e8f0
                style mse fill:#4c0519,stroke:#f43f5e,color:#e2e8f0
                style mul1 fill:#1e293b,stroke:#475569,color:#e2e8f0
                style add1 fill:#1e293b,stroke:#475569,color:#e2e8f0
                style relu fill:#1e293b,stroke:#475569,color:#e2e8f0
                style mul2 fill:#1e293b,stroke:#475569,color:#e2e8f0
                style add2 fill:#1e293b,stroke:#475569,color:#e2e8f0
            `} />

            <p className="text-muted-foreground text-sm">
              Parameter nodes (purple) have <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad=True</code>&mdash;these
              are the leaf tensors PyTorch accumulates gradients into. Each operation node (slate) has
              a <code className="text-sm bg-muted px-1.5 py-0.5 rounded">grad_fn</code> recording how
              it was computed. This is exactly the graph from Computational Graphs, with
              PyTorch API labels mapped on.
            </p>

            <p className="text-muted-foreground">
              Of course these are the same&mdash;<code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> IS
              walking the computational graph. You already drew this graph. You already computed
              these values. The only thing that changed is who is doing the arithmetic.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Magic&mdash;Automation">
            Autograd does not replace understanding. It automates the algorithm
            you already know. The manual work in Backprop by the Numbers was worth
            doing&mdash;now you understand what is inside the black box.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 7. Interactive Widget */}
      <Row>
        <Row.Content>
          <ExercisePanel title="Autograd Explorer" subtitle="Compare manual backprop to autograd">
            <AutogradExplorer />
          </ExercisePanel>
        </Row.Content>
        <Row.Aside>
          <TryThisBlock title="Experiment">
            <ul className="space-y-2 text-sm">
              <li>Step through manual mode, then switch to autograd and click backward(). Same numbers.</li>
              <li>Adjust the weights with the sliders. Verify the match holds for different values.</li>
              <li>Set w1 to a negative value so z1 becomes negative. Watch the ReLU kill the gradient for Layer 1.</li>
            </ul>
          </TryThisBlock>
        </Row.Aside>
      </Row>

      {/* 8. Elaborate — The Gotchas */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Three Gotchas"
            subtitle="Behaviors that trip up every beginner"
          />

          {/* Gotcha 1: Gradient Accumulation */}
          <div className="space-y-4 mt-4">
            <h4 className="text-sm font-bold text-amber-400">Gotcha 1: Gradient Accumulation</h4>
            <p className="text-muted-foreground">
              What happens if you call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> twice?
              Predict the output before reading:
            </p>

            <CodeBlock
              language="python"
              filename="accumulation_trap.py"
              code={`x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

# retain_graph=True keeps the graph so we can call backward() again.
# Normally backward() destroys the graph after running (to free memory).
# In real training you don't need this — each forward pass builds a new graph.
y.backward(retain_graph=True)
print(x.grad)    # tensor(6.)  — correct, dy/dx = 2(3) = 6

y.backward()
print(x.grad)    # tensor(12.) — DOUBLED! Not 6.`}
            />

            <p className="text-muted-foreground">
              If you predicted 6.0 the second time, you fell into the trap. PyTorch <strong>adds</strong> to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.grad</code> by
              default&mdash;it does not replace the old value. The gradient <em>accumulated</em>.
            </p>

            <p className="text-muted-foreground">
              This is not a bug. Remember fan-out from Computational Graphs? When a value
              feeds into <strong>two</strong> operations, the gradient from each path sums.
              Gradient accumulation is the same idea. If a parameter appears in multiple
              computations (like a shared weight used twice), the gradients from each use
              should <strong>add</strong>. PyTorch defaults to accumulation because it handles
              this case correctly. For the common single-use case, clear the gradients
              between steps:
            </p>

            <CodeBlock
              language="python"
              code={`# The fix: zero_grad() before each backward pass
x.grad.zero_()   # Clear the accumulated gradient
y.backward()
print(x.grad)    # tensor(6.) — fresh gradient`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="#1 Beginner Bug">
            Forgetting <code className="text-xs">zero_grad()</code> is the single most
            common PyTorch bug. Gradients accumulate silently, causing the model
            to diverge after a few steps.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* Gotcha 2: no_grad() */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h4 className="text-sm font-bold text-amber-400">Gotcha 2: torch.no_grad()</h4>
            <p className="text-muted-foreground">
              The parameter update <code className="text-sm bg-muted px-1.5 py-0.5 rounded">w = w - lr * w.grad</code> is
              an operation on a <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad</code> tensor.
              Without protection, PyTorch would <strong>record</strong> this update too&mdash;building
              a graph on top of the graph. The update step would become part of the next
              computation.
            </p>

            <CodeBlock
              language="python"
              filename="no_grad.py"
              code={`# WITHOUT no_grad — the update is recorded (BAD)
w = w - lr * w.grad  # PyTorch builds a new graph node for this!

# WITH no_grad — the update is NOT recorded (CORRECT)
with torch.no_grad():
    w = w - lr * w.grad  # No recording. Just math.`}
            />

            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.no_grad()</code> is not just
              an optimization&mdash;it is semantically necessary. The recording metaphor:
              you need to <strong>pause Recording</strong> while updating the weights,
              then resume for the next forward pass.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Pause, Don&apos;t Stop">
            <code className="text-xs">no_grad()</code> is a context manager&mdash;it
            pauses recording inside
            the <code className="text-xs">with</code> block, then resumes after.
            Parameters still have <code className="text-xs">requires_grad=True</code>;
            tracking is temporarily suspended.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Gotcha 3: detach() */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <h4 className="text-sm font-bold text-amber-400">Gotcha 3: .detach()</h4>
            <p className="text-muted-foreground">
              Remember <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.detach().cpu().numpy()</code> from
              Tensors? Now it makes sense. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.detach()</code> creates
              a new tensor that shares the same data but is <strong>severed from the
              computational graph</strong>:
            </p>

            <CodeBlock
              language="python"
              filename="detach.py"
              code={`x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Detach y from the graph
y_detached = y.detach()
print(y_detached)          # tensor(4.) — same value
print(y_detached.grad_fn)  # None — no graph connection

# Now .detach().cpu().numpy() makes sense:
# 1. detach() — sever from graph (NumPy has no graph)
# 2. cpu()    — move to CPU (NumPy has no GPU)
# 3. numpy()  — convert to NumPy array`}
            />

            <p className="text-muted-foreground">
              This is &ldquo;no path = no gradient&rdquo; from Computational Graphs,
              implemented as an API call. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">detach()</code> snips
              the tape. Parameters feeding through a detached path get no gradient.
            </p>

            <CodeBlock
              language="python"
              code={`# Demonstration: detach blocks gradient flow
a = torch.tensor(2.0, requires_grad=True)
b = a * 3          # b depends on a
c = b.detach() * 5 # c depends on b's VALUE, not on a

c.backward()
print(a.grad)  # None — the path was severed by detach()`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Snip the Tape">
            <code className="text-xs">.detach()</code> creates a tensor with the same
            data but no graph connection. Think of it as cutting the recording tape&mdash;the
            value survives, but the history is gone. Any path through a detached tensor
            is a dead end for gradients.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 9. Check 2 — Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              A colleague writes a training step but forgets <code className="text-sm bg-muted px-1.5 py-0.5 rounded">zero_grad()</code>.
              After 10 iterations, they notice the gradients are enormous and the model
              diverges. They think there is a bug in their loss function. What is actually
              happening, and how do you fix it?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>Gradients are accumulating.</strong> Without <code className="text-xs bg-muted px-1 rounded">zero_grad()</code>,
                  each call to <code className="text-xs bg-muted px-1 rounded">backward()</code> adds
                  to the existing <code className="text-xs bg-muted px-1 rounded">.grad</code> values.
                  After 10 iterations, the gradients are ~10x too large, causing the weight
                  updates to overshoot wildly.
                </p>
                <p className="text-muted-foreground">
                  <strong>Fix:</strong> Add <code className="text-xs bg-muted px-1 rounded">zero_grad()</code> (or <code className="text-xs bg-muted px-1 rounded">param.grad.zero_()</code> for
                  each parameter) before every <code className="text-xs bg-muted px-1 rounded">backward()</code> call.
                  The loss function is fine&mdash;the bug is in the gradient hygiene.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* Side-by-side manual vs autograd training step */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Preview: A Complete Training Step"
            subtitle="Manual NumPy vs PyTorch autograd, side by side"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Implementing Linear Regression, you had to manually derive the gradient
              formula for each parameter, then implement that formula in NumPy code. With
              autograd, that entire step collapses to a single line. Here is that training
              step next to its PyTorch equivalent:
            </p>

            <ComparisonRow
              left={{
                title: 'Manual (NumPy)',
                color: 'amber',
                items: [
                  'Compute forward pass',
                  'Compute loss',
                  'Manually derive dL/dw and dL/db',
                  'Implement gradient formulas',
                  'Update: w = w - lr * dL_dw',
                ],
              }}
              right={{
                title: 'Autograd (PyTorch)',
                color: 'blue',
                items: [
                  'Compute forward pass',
                  'Compute loss',
                  'loss.backward()  ← one line',
                  'Read w.grad, b.grad',
                  'Update: w -= lr * w.grad (in no_grad)',
                ],
              }}
            />

            <CodeBlock
              language="python"
              filename="manual_training_step.py"
              code={`# A single training step with autograd
x = torch.tensor(2.0)
y_true = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
lr = 0.01

# Forward
y_hat = w * x + b
loss = (y_true - y_hat) ** 2

# Backward (autograd does the chain rule)
loss.backward()

# Update (pause recording, then clear gradients)
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

w.grad.zero_()
b.grad.zero_()`}
            />

            <p className="text-muted-foreground text-sm">
              This is the complete pattern: forward, backward, update (in no_grad), zero_grad.
              The next lessons will package this into cleaner abstractions&mdash;but the
              mechanics are exactly what you see here.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Coming Next">
            Writing <code className="text-xs">w</code> and <code className="text-xs">b</code> as
            individual tensors gets tedious. <strong>nn.Module</strong> (next lesson)
            packages parameters into reusable building blocks.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Practice — Colab */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Build muscle memory with real autograd code"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Reading about autograd is not enough. Open the notebook and
                complete these exercises:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>Compute gradients for a polynomial function, verify by hand</li>
                <li>Reproduce the backprop-worked-example network and compare <code className="text-xs bg-muted px-1 rounded">.grad</code> values to manual calculation</li>
                <li>Demonstrate the accumulation trap&mdash;predict, run, fix with <code className="text-xs bg-muted px-1 rounded">zero_grad()</code></li>
                <li>Write a single manual training step: forward, backward, update with <code className="text-xs bg-muted px-1 rounded">no_grad()</code>, <code className="text-xs bg-muted px-1 rounded">zero_grad()</code></li>
                <li>(Stretch) Use <code className="text-xs bg-muted px-1 rounded">detach()</code> to stop gradients flowing to part of a computation. Verify affected variables have <code className="text-xs bg-muted px-1 rounded">.grad = None</code></li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-1-2-autograd.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercises 1&ndash;3 are guided with starter code and hints. Exercise 4 is
                supported (template provided). Exercise 5 is independent.
              </p>
            </div>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Predict First">
            For every code cell, predict the output before running it. Wrong predictions
            are more valuable than correct ones&mdash;they reveal gaps in your mental model.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 11. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'requires_grad = press Record',
                description:
                  'Tells PyTorch to track operations on this tensor. Every operation creates a node in the computational graph. The tensor itself is unchanged — PyTorch just watches.',
              },
              {
                headline: 'backward() = press Rewind',
                description:
                  'Walks the computational graph backward, applying the chain rule at each node. Exactly the same algorithm you did by hand in Backprop by the Numbers.',
              },
              {
                headline: '.grad = the result, stored as an attribute',
                description:
                  'Gradients are stored on the leaf tensors (the parameters), not returned from backward(). This is where you read the results.',
              },
              {
                headline: 'zero_grad() = clear the tape for the next step',
                description:
                  'Gradients accumulate by default (correct for shared parameters, like fan-out in a computational graph). For the common case, call zero_grad() before every backward() to prevent silent accumulation.',
              },
              {
                headline: 'no_grad() = pause Recording during updates',
                description:
                  'Wrap parameter updates in torch.no_grad() to prevent the update step from being recorded. Not just optimization — semantically necessary.',
              },
              {
                headline: 'detach() = snip the tape',
                description:
                  'Severs a tensor from the computational graph. "No path = no gradient." Now .detach().cpu().numpy() makes sense: detach from graph, move to CPU, convert to array.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/nn-module"
            title="Ready to package neurons into reusable layers?"
            description="You can now compute gradients for any computation PyTorch can express. Next: nn.Module — package parameters, layers, and forward passes into building blocks."
            buttonText="Continue to nn.Module"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
