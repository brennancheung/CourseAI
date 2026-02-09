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
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { MermaidDiagram } from '@/components/widgets/MermaidDiagram'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * nn.Module -- Lesson 3 of Module 2.1 (PyTorch Core)
 *
 * Teaches the student to:
 * - Use nn.Linear as a layer that IS w*x + b
 * - Define custom nn.Module subclasses with __init__ + forward()
 * - Collect all parameters with model.parameters()
 * - Use nn.Sequential for simple layer stacks
 * - Understand why custom forward() is needed for non-sequential architectures
 *
 * Target depths:
 * - nn.Module subclass pattern: DEVELOPED
 * - nn.Linear: DEVELOPED
 * - nn.Sequential: INTRODUCED
 * - model.parameters(): DEVELOPED
 * - nn.ReLU as a layer: INTRODUCED
 *
 * Central insight: nn.Module packages the exact same computation the student
 * already knows (neurons, layers, activations) into reusable building blocks
 * with automatic parameter management. The math does not change. The
 * organization does.
 */

export function NnModuleLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="nn.Module"
            description="Package neurons, layers, and networks into reusable building blocks with automatic parameter management."
            category="PyTorch Core"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Define neural networks using <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Module</code>.
            By the end, you can build a model as a Python class with{' '}
            <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__init__</code> and{' '}
            <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code>, verify
            that <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code> is
            the same <InlineMath math="w \cdot x + b" /> you already know, and collect all
            learnable parameters with a single call.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Building on Autograd">
            Everything here builds on the raw tensors from Autograd.
            You already know how to create parameters, compute gradients, and
            clear them. Now you learn how to organize all of that.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'nn.Module and nn.Linear only — no optimizers like Adam or SGD (next lesson)',
              'No loss function objects like nn.MSELoss (next lesson)',
              'No full training loops — we verify forward + backward, not training',
              'No convolutional or recurrent layers — just the linear building block',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook -- "Imagine 100 Neurons" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title={`Imagine 100 Neurons`}
            subtitle="The bookkeeping problem that nn.Module solves"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Autograd, you built a 2-layer network with four individual tensors:
            </p>

            <CodeBlock
              language="python"
              filename="from_autograd.py"
              code={`# From Autograd — four individual tensors
w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(-0.3, requires_grad=True)
b2 = torch.tensor(0.2, requires_grad=True)

# Manually wired forward pass
z1 = w1 * x + b1
a1 = torch.clamp(z1, min=0)
y_hat = w2 * a1 + b2`}
            />

            <p className="text-muted-foreground">
              This worked for 4 parameters. Now consider a 3-layer network with 64
              neurons per layer. That is:
            </p>

            <div className="rounded-md border bg-muted/50 px-4 py-3">
              <p className="text-sm text-muted-foreground font-mono text-center">
                <InlineMath math="64 \times 3 + 64 \times 64 + 64 \times 1 = 4{,}353" /> parameters
              </p>
            </div>

            <p className="text-muted-foreground">
              Are you going to create <strong>4,353 individual tensors</strong>, each
              with <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad=True</code>,
              each needing <code className="text-sm bg-muted px-1.5 py-0.5 rounded">zero_grad()</code>?
              The math does not get harder. The bookkeeping becomes impossible.
            </p>

            <p className="text-muted-foreground">
              There is a better way.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Math, Better Organization">
            The concepts do not change. Neurons still compute{' '}
            <InlineMath math="w \cdot x + b" />. Layers are still groups of neurons.
            What changes is how you manage the parameters.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain Part 1 -- nn.Linear (The Building Block) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="nn.Linear: A Neuron You Already Know"
            subtitle="The same w*x + b, packaged as a layer"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Start with the simplest possible example. One neuron, three inputs:
            </p>

            <CodeBlock
              language="python"
              filename="nn_linear_basics.py"
              code={`import torch
import torch.nn as nn

# One neuron with 3 inputs
layer = nn.Linear(3, 1)

print(layer.weight)        # tensor([[ 0.xxxx,  0.xxxx,  0.xxxx]])
print(layer.weight.shape)  # torch.Size([1, 3])
print(layer.bias)          # tensor([0.xxxx])
print(layer.bias.shape)    # torch.Size([1])`}
            />

            <p className="text-muted-foreground">
              That is a weight matrix and a bias vector&mdash;exactly
              the <InlineMath math="w \cdot x + b" /> from Neuron Basics. The numbers
              are random because PyTorch initializes them for you (using Kaiming uniform
              by default&mdash;a variant of He initialization, which you learned preserves
              signal magnitude across layers in Training Dynamics).
            </p>

            <p className="text-muted-foreground">
              Check the gradient flag:
            </p>

            <CodeBlock
              language="python"
              code={`print(layer.weight.requires_grad)  # True
print(layer.bias.requires_grad)    # True`}
            />

            <p className="text-muted-foreground">
              Both are already <code className="text-sm bg-muted px-1.5 py-0.5 rounded">True</code>.
              In Autograd, you set <code className="text-sm bg-muted px-1.5 py-0.5 rounded">requires_grad=True</code> on
              every tensor manually. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code> handles
              that for you. The Recording is on from the start.
            </p>

            <p className="text-muted-foreground">
              Now verify that this layer does the same computation you know:
            </p>

            <CodeBlock
              language="python"
              filename="verify_computation.py"
              code={`x = torch.tensor([[1.0, 2.0, 3.0]])  # shape: (1, 3)

# Using the layer
output_layer = layer(x)

# Manual computation — same formula
output_manual = x @ layer.weight.T + layer.bias

print(output_layer)   # tensor([[...]])
print(output_manual)  # tensor([[...]])  — same numbers!`}
            />

            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code> does
              not do something mysterious. It IS the matrix multiply you already know.
              The <code className="text-sm bg-muted px-1.5 py-0.5 rounded">@</code> operator from
              Tensors, the <InlineMath math="w \cdot x + b" /> from Neuron Basics&mdash;same
              computation, packaged into a reusable object.
            </p>

            <p className="text-muted-foreground">
              Extend to multiple neurons: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear(3, 4)</code> creates
              4 neurons, each with 3 inputs:
            </p>

            <CodeBlock
              language="python"
              code={`layer = nn.Linear(3, 4)
print(layer.weight.shape)  # torch.Size([4, 3]) — 4 neurons, 3 inputs each
print(layer.bias.shape)    # torch.Size([4])    — one bias per neuron`}
            />

            <p className="text-muted-foreground">
              This is the &ldquo;layer = group of neurons&rdquo; concept from Neuron
              Basics&mdash;same inputs, different weight sets&mdash;now expressed as a single
              object with a <code className="text-sm bg-muted px-1.5 py-0.5 rounded">(4, 3)</code> weight
              matrix.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="nn.Parameter">
            Under the hood, <code className="text-xs">nn.Linear</code> wraps its
            tensors in <code className="text-xs">nn.Parameter</code>&mdash;a thin
            wrapper that sets <code className="text-xs">requires_grad=True</code> by
            default and registers the tensor for parameter collection. You do not need
            to create Parameters manually for standard layers.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 4. Explain Part 2 -- nn.Module (The Packaging System) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="nn.Module: The Packaging System"
            subtitle="A class that holds layers and defines how data flows through them"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              A single <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code> is
              one layer. To build a network, you compose multiple layers into
              an <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Module</code> subclass.
              Think of it like a LEGO brick&mdash;each brick has a specific shape
              (computation) and connection points (inputs/outputs). You snap them together
              to build something larger.
            </p>

            <p className="text-muted-foreground">
              Here is the same 2-layer network from Autograd, expressed as a Module:
            </p>

            <CodeBlock
              language="python"
              filename="two_layer_net.py"
              code={`class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)  # 1 input, 1 output
        self.layer2 = nn.Linear(1, 1)  # 1 input, 1 output

    def forward(self, x):
        x = self.layer1(x)
        x = torch.clamp(x, min=0)  # ReLU
        x = self.layer2(x)
        return x

model = TwoLayerNet()`}
            />

            <p className="text-muted-foreground">
              Two methods, two jobs:
            </p>
            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__init__</code> defines
                the <strong>what</strong>&mdash;which layers exist
              </li>
              <li>
                <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code> defines
                the <strong>how</strong>&mdash;how data flows through them
              </li>
            </ul>

            <p className="text-muted-foreground">
              This is Python. You are defining a class. The magic is minimal:{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">super().__init__()</code> registers
              the module, and any <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Module</code> or{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Parameter</code> assigned
              to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">self</code> is automatically tracked.
            </p>

            <p className="text-muted-foreground">
              Every call to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model(x)</code> builds
              a fresh computational graph. That means <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code> can
              contain any Python logic&mdash;conditionals, loops, anything:
            </p>

            <CodeBlock
              language="python"
              filename="dynamic_forward.py"
              code={`# forward() is not a static description — it runs every time
def forward(self, x):
    x = self.layer1(x)
    if x.mean() > 0:       # different path per input!
        x = self.layer2(x)
    return x`}
            />

            <p className="text-muted-foreground text-sm">
              A container could never do this. Each input can follow a different computation
              path. This is what makes PyTorch a <strong>dynamic graph</strong> framework.
            </p>

            <p className="text-muted-foreground">
              Now the &ldquo;parameters are knobs&rdquo; metaphor from Linear Regression
              becomes a concrete API:
            </p>

            <CodeBlock
              language="python"
              filename="parameters.py"
              code={`# Iterate all learnable parameters
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# Output:
# layer1.weight: shape=torch.Size([1, 1]), requires_grad=True
# layer1.bias: shape=torch.Size([1]), requires_grad=True
# layer2.weight: shape=torch.Size([1, 1]), requires_grad=True
# layer2.bias: shape=torch.Size([1]), requires_grad=True`}
            />

            <p className="text-muted-foreground">
              One call&mdash;<code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.parameters()</code>&mdash;collects
              every learnable tensor in the model. And{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.zero_grad()</code> clears
              all their gradients at once. In Autograd, you called{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">zero_grad()</code> on
              each tensor individually.
            </p>

            <p className="text-muted-foreground">
              One important convention: always
              call <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model(x)</code>, not{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.forward(x)</code>.
              They look like they do the same thing&mdash;and for the output, they
              do. But <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model(x)</code> calls{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__call__</code>, which
              runs internal hooks before and after{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code>. Calling{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code> directly
              skips those hooks. For now, just follow the convention.
            </p>

            <CodeBlock
              language="python"
              code={`# Always use model(x), not model.forward(x)
# model(x) runs hooks + forward(); model.forward(x) skips hooks
y_hat = model(x)  # correct`}
            />

            <p className="text-muted-foreground">
              Here is how the code maps to the architecture:
            </p>

            <MermaidDiagram chart={`
              graph LR
                Input["x<br/>(input)"] --> L1["self.layer1<br/>nn.Linear(1,1)"]
                L1 --> ReLU["torch.clamp<br/>ReLU"]
                ReLU --> L2["self.layer2<br/>nn.Linear(1,1)"]
                L2 --> Output["y_hat<br/>(output)"]

                style Input fill:#172554,stroke:#3b82f6,color:#e2e8f0
                style L1 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style ReLU fill:#1e293b,stroke:#475569,color:#e2e8f0
                style L2 fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style Output fill:#172554,stroke:#3b82f6,color:#e2e8f0
            `} />

            <p className="text-muted-foreground text-sm">
              Purple blocks are learnable layers (parameters live here). Slate block is the
              activation function (no parameters). Each block maps one-to-one to a line in{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">__init__</code> and a
              line in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not Just a Container">
            <code className="text-xs">nn.Module</code> is more than a dictionary of
            parameters. It manages the computational graph:{' '}
            <code className="text-xs">model(x)</code> runs <code className="text-xs">forward()</code>,
            which builds the autograd graph. Parameters are tracked,
            gradients are managed, and the computation can change per input
            (conditionals, loops&mdash;anything Python can do).
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 -- Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <p className="text-muted-foreground text-sm mb-3">
              You create <code className="text-sm bg-muted px-1.5 py-0.5 rounded">layer = nn.Linear(5, 3)</code>.
              How many parameters does this layer have?
            </p>
            <p className="text-muted-foreground text-sm mb-3">
              Hint: 3 neurons, each with 5 weights + 1 bias.
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <InlineMath math="5 \times 3 + 3 = 18" /> parameters.
                  The weight matrix is <code className="text-xs bg-muted px-1 rounded">(3, 5)</code> = 15
                  values, plus 3 biases.
                </p>
                <CodeBlock
                  language="python"
                  code={`layer = nn.Linear(5, 3)
total = sum(p.numel() for p in layer.parameters())
print(total)  # 18`}
                />
                <p className="text-muted-foreground mt-2">
                  Follow-up: how many parameters
                  in <code className="text-xs bg-muted px-1 rounded">nn.Linear(100, 50)</code>?
                </p>
                <p className="text-muted-foreground">
                  <InlineMath math="100 \times 50 + 50 = 5{,}050" />.
                </p>
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 6. Explain Part 3 -- The Payoff (Reproducing the Autograd Network) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Payoff: Same Numbers, Better Organization"
            subtitle="Reproducing the Autograd network with nn.Module"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Now for the proof. Set up the exact same network from Autograd&mdash;same
              weights (w1=0.5, b1=0.1, w2=-0.3, b2=0.2), same input (x=2), same
              target (y=1):
            </p>

            <CodeBlock
              language="python"
              filename="reproduce_autograd.py"
              code={`import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.clamp(x, min=0)  # ReLU
        x = self.layer2(x)
        return x

model = TwoLayerNet()

# Set the SAME weights from Autograd
with torch.no_grad():
    model.layer1.weight.fill_(0.5)
    model.layer1.bias.fill_(0.1)
    model.layer2.weight.fill_(-0.3)
    model.layer2.bias.fill_(0.2)

# Same input and target
x = torch.tensor([[2.0]])
y_true = torch.tensor([[1.0]])

# Forward pass
y_hat = model(x)
loss = (y_true - y_hat) ** 2

# Clear gradients, then backward pass
model.zero_grad()  # clears ALL parameter gradients at once
loss.backward()

# Check the gradients
print(f"layer1.weight.grad = {model.layer1.weight.grad.item():.4f}")  # 1.3560
print(f"layer1.bias.grad   = {model.layer1.bias.grad.item():.4f}")    # 0.6780
print(f"layer2.weight.grad = {model.layer2.weight.grad.item():.4f}")  # -2.4860
print(f"layer2.bias.grad   = {model.layer2.bias.grad.item():.4f}")    # -2.2600`}
            />

            <p className="text-muted-foreground">
              Compare these to the values from Autograd and Backprop by the Numbers:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-muted-foreground border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4 font-semibold">Parameter</th>
                    <th className="text-left py-2 pr-4 font-semibold">Raw tensors (Autograd)</th>
                    <th className="text-left py-2 pr-4 font-semibold">nn.Module</th>
                    <th className="text-left py-2 pr-4 font-semibold">Match?</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">w1</td>
                    <td className="py-2 pr-4 font-mono">1.3560</td>
                    <td className="py-2 pr-4 font-mono">layer1.weight.grad = 1.3560</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">b1</td>
                    <td className="py-2 pr-4 font-mono">0.6780</td>
                    <td className="py-2 pr-4 font-mono">layer1.bias.grad = 0.6780</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 pr-4 font-mono">w2</td>
                    <td className="py-2 pr-4 font-mono">-2.4860</td>
                    <td className="py-2 pr-4 font-mono">layer2.weight.grad = -2.4860</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-mono">b2</td>
                    <td className="py-2 pr-4 font-mono">-2.2600</td>
                    <td className="py-2 pr-4 font-mono">layer2.bias.grad = -2.2600</td>
                    <td className="py-2 pr-4 text-emerald-400">Yes</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="text-muted-foreground">
              Same computation. Same gradients. The only thing that changed is how
              the parameters are organized.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Three Representations">
            You have now seen this exact network expressed three ways: math in
            Backprop by the Numbers, raw tensors in Autograd, and nn.Module here.
            The numbers are the same every time. Each representation is a different
            level of abstraction for the same computation.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Side-by-side: raw tensors vs nn.Module */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Here is what nn.Module handles for you:
            </p>

            <ComparisonRow
              left={{
                title: 'Raw Tensors (Autograd)',
                color: 'amber',
                items: [
                  'Create each tensor with requires_grad=True',
                  'Wire forward pass manually',
                  'Call zero_grad() on each tensor',
                  'Track parameters yourself',
                  '4 lines just for parameter setup',
                ],
              }}
              right={{
                title: 'nn.Module',
                color: 'blue',
                items: [
                  'requires_grad set automatically',
                  'forward() defines the wiring once',
                  'model.zero_grad() clears all at once',
                  'model.parameters() collects all',
                  '1 class, reusable for any input',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Of course you would package this. You already know the math. Now the code
              matches how you think about it: <strong>layers</strong>, not individual weights.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Explain Part 4 -- nn.Sequential (The Shortcut) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="nn.Sequential: The Shortcut"
            subtitle="For simple layer stacks, skip the class"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              For networks that are just layers in a straight line, you do not need
              a class at all:
            </p>

            <CodeBlock
              language="python"
              filename="sequential.py"
              code={`model = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

# That's it. PyTorch runs them in order.
x = torch.tensor([[2.0]])
y_hat = model(x)  # Linear -> ReLU -> Linear`}
            />

            <p className="text-muted-foreground">
              Instead of writing a class, you list the layers in order. PyTorch runs
              them sequentially.
            </p>

            <p className="text-muted-foreground">
              Notice <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.ReLU()</code>&mdash;the
              same <InlineMath math="\max(0, x)" /> you know from Activation Functions.
              As a module, it can be placed in a Sequential stack. No parameters, no
              weights&mdash;just the nonlinear function applied element-wise.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="nn.ReLU vs torch.clamp">
            In the custom Module above, we
            used <code className="text-xs">torch.clamp(x, min=0)</code> for
            ReLU&mdash;a function call. <code className="text-xs">nn.ReLU()</code> is
            the same operation wrapped as a module, so it can slot into Sequential.
            Both produce identical output.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Negative example: linear collapse */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              But be careful. What happens if you stack linear layers <strong>without</strong> activations?
            </p>

            <CodeBlock
              language="python"
              filename="linear_collapse.py"
              code={`# No activation between linear layers
model_no_act = nn.Sequential(
    nn.Linear(2, 4),
    nn.Linear(4, 1),
)

x = torch.randn(1, 2)

# Prove collapse: compute the effective single-layer equivalent
W1 = model_no_act[0].weight  # shape (4, 2)
b1 = model_no_act[0].bias    # shape (4,)
W2 = model_no_act[1].weight  # shape (1, 4)
b2 = model_no_act[1].bias    # shape (1,)

W_eff = W2 @ W1               # shape (1, 2) — one linear layer!
b_eff = W2 @ b1 + b2          # shape (1,)

# Compare: 2-layer model vs single effective layer
print(model_no_act(x))            # tensor([[...]])
print(x @ W_eff.T + b_eff)       # tensor([[...]])  — same numbers!

# Now add activation and compare
model_with_act = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
with torch.no_grad():
    model_with_act[0].weight.copy_(W1)
    model_with_act[0].bias.copy_(b1)
    model_with_act[2].weight.copy_(W2)
    model_with_act[2].bias.copy_(b2)

print(model_with_act(x))  # Different — ReLU breaks the collapse`}
            />

            <p className="text-muted-foreground">
              Without activations, two stacked linear layers collapse into one.
              This is the same &ldquo;100-layer linear = 1-layer linear&rdquo; insight
              from Neuron Basics. The math
              is <InlineMath math="W_2(W_1 x + b_1) + b_2 = W_{\text{eff}} x + b_{\text{eff}}" />&mdash;a
              single linear transformation, no matter how many layers you stack.
            </p>

            <p className="text-muted-foreground">
              Activations between linear layers prevent collapse. This is why
              every <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Sequential</code> model
              interleaves <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Linear</code> with{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.ReLU()</code> (or
              another activation). Same insight from Series 1, expressed in code.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="Linear Collapse">
            Stacking <code className="text-xs">nn.Linear</code> layers without
            activation functions gives you a deeper model with more parameters
            but <strong>no</strong> additional representational power. Always place
            an activation between linear layers.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8. Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              A colleague builds a model:
            </p>
            <CodeBlock
              language="python"
              code={`model = nn.Sequential(
    nn.Linear(10, 50),
    nn.Linear(50, 50),
    nn.Linear(50, 1),
)`}
            />
            <p className="text-muted-foreground text-sm mt-3 mb-3">
              They are confused why their &ldquo;deep network&rdquo; performs no better
              than a simple linear regression. What is the problem, and how do they fix it?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>No activation functions between linear layers.</strong> Without
                  nonlinearities, the three linear layers collapse to a single linear
                  transformation. The model has more parameters but no more expressive power
                  than <code className="text-xs bg-muted px-1 rounded">nn.Linear(10, 1)</code>.
                </p>
                <p className="text-muted-foreground">
                  <strong>Fix:</strong> Add <code className="text-xs bg-muted px-1 rounded">nn.ReLU()</code> between
                  each pair of linear layers:
                </p>
                <CodeBlock
                  language="python"
                  code={`model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
)`}
                />
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 9. Elaborate -- Beyond Sequential (Custom forward()) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Beyond Sequential"
            subtitle="When your architecture is not a simple stack"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Sequential</code> works
              when data flows straight through: layer 1, then layer 2, then layer 3.
              But what if the architecture is not a straight line?
            </p>

            <p className="text-muted-foreground">
              A <strong>skip connection</strong> (or residual connection) adds the
              input directly to the output. The input bypasses the layer and is added back:
            </p>

            <CodeBlock
              language="python"
              filename="skip_connection.py"
              code={`class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(torch.clamp(x, min=0)) + x`}
            />

            <p className="text-muted-foreground">
              The <code className="text-sm bg-muted px-1.5 py-0.5 rounded">+ x</code> at the end
              cannot be expressed
              in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Sequential</code> because
              the input needs to skip ahead past the layer. This is
              why <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code> exists:
              it gives you full control over the computation graph.
            </p>

            <MermaidDiagram chart={`
              graph LR
                X["x"] --> ReLU["ReLU<br/>torch.clamp(x, min=0)"]
                X --> Add["+ (add)"]
                ReLU --> Linear["nn.Linear(size, size)"]
                Linear --> Add
                Add --> Out["output"]

                style X fill:#172554,stroke:#3b82f6,color:#e2e8f0
                style Linear fill:#2e1065,stroke:#7c3aed,color:#e2e8f0
                style ReLU fill:#1e293b,stroke:#475569,color:#e2e8f0
                style Add fill:#1e293b,stroke:#475569,color:#e2e8f0
                style Out fill:#172554,stroke:#3b82f6,color:#e2e8f0
            `} />

            <p className="text-muted-foreground">
              Skip connections were mentioned in Training Dynamics as the technique
              that pushed networks to 152 layers (ResNets). Now you see the code
              pattern. The LEGO analogy extends: while Sequential snaps bricks in
              a straight tower, custom <code className="text-sm bg-muted px-1.5 py-0.5 rounded">forward()</code> lets
              you design any assembly you want.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Sequential vs Custom">
            <strong>Sequential</strong> = simple stacks where each layer feeds the
            next. <strong>Custom Module</strong> = anything else: skip connections,
            branching, conditional logic, shared layers.
            Most real architectures need custom Modules.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 10. Practice -- Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Build muscle memory with nn.Module"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Reading about nn.Module is not enough. Open the notebook and
                complete these exercises:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>Create and inspect nn.Linear layers</strong>&mdash;create layers of
                  various sizes, print weight shapes, verify parameter counts (guided)
                </li>
                <li>
                  <strong>Verify nn.Linear IS w*x + b</strong>&mdash;manually
                  compute <code className="text-xs bg-muted px-1 rounded">x @ weight.T + bias</code> and
                  compare to <code className="text-xs bg-muted px-1 rounded">layer(x)</code> (guided)
                </li>
                <li>
                  <strong>Build a 2-layer nn.Module subclass</strong>&mdash;define{' '}
                  <code className="text-xs bg-muted px-1 rounded">__init__</code> and{' '}
                  <code className="text-xs bg-muted px-1 rounded">forward()</code>, run
                  forward pass, verify output (supported)
                </li>
                <li>
                  <strong>Convert to nn.Sequential</strong>&mdash;rewrite the same network
                  using Sequential, verify same output (supported)
                </li>
                <li>
                  <strong>Linear collapse experiment</strong>&mdash;build Sequential with and
                  without activations, compare (supported)
                </li>
                <li>
                  <strong>(Stretch) Build a skip-connection Module</strong>&mdash;implement
                  the ResidualBlock from the lesson, verify forward pass includes the skip
                  (independent)
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-1-3-nn-module.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Open in Google Colab
              </a>
              <p className="text-xs text-muted-foreground">
                Exercises 1&ndash;2 are guided with starter code and hints. Exercises 3&ndash;5
                are supported (template provided). Exercise 6 is independent.
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
                headline: 'nn.Linear(in, out) = a layer of neurons',
                description:
                  'Each neuron computes w*x + b — the same formula from Neuron Basics. The weight matrix is (out, in), the bias is (out,). requires_grad is True by default.',
              },
              {
                headline: 'nn.Module = computation + parameter management',
                description:
                  'A class with __init__ (what layers exist) and forward() (how data flows through them). Any nn.Module or nn.Parameter assigned to self is automatically tracked.',
              },
              {
                headline: 'model.parameters() = collect all knobs',
                description:
                  'One call returns every learnable tensor in the model. model.zero_grad() clears all gradients at once. The "parameters are knobs" metaphor is now a concrete API.',
              },
              {
                headline: 'nn.Sequential = layers in a straight line',
                description:
                  'List layers in order, PyTorch runs them sequentially. Great for simple stacks. But if your architecture needs skip connections, branching, or conditionals — use a custom Module.',
              },
              {
                headline: 'Custom forward() = any computation you want',
                description:
                  'Skip connections, conditional logic, shared layers — forward() gives you full control. Most real architectures need custom Modules, not Sequential.',
              },
              {
                headline: 'The math did not change. The organization did.',
                description:
                  'You defined neurons mathematically in Series 1. You wired them with raw tensors in Autograd. Now you package them into models. Same computation at every level of abstraction.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* 12. Next Step */}
      <Row>
        <Row.Content>
          <NextStepBlock
            href="/app/lesson/training-loop"
            title="Ready to put it all together?"
            description="You have tensors, autograd, and models. Next: write a complete training loop — the same forward, loss, backward, update pattern from Series 1, now with nn.Module models, loss function objects, and torch.optim optimizers."
            buttonText="Continue to Training Loop"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
