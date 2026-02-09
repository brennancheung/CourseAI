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
  ModuleCompleteBlock,
} from '@/components/lessons'
import { CodeBlock } from '@/components/common/CodeBlock'
import { ExternalLink } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath } from 'react-katex'

/**
 * Training Loop — Lesson 4 of Module 2.1 (PyTorch Core)
 *
 * CONSOLIDATE capstone: assembles tensors + autograd + nn.Module into
 * a complete PyTorch training loop. Two genuinely new API concepts:
 *   1. Loss function objects (nn.MSELoss)
 *   2. Optimizer objects (torch.optim.SGD, torch.optim.Adam)
 * Everything else is reconnecting known pieces.
 *
 * Target depths:
 * - nn.MSELoss: DEVELOPED
 * - torch.optim.SGD / Adam: DEVELOPED
 * - optimizer.step(): DEVELOPED
 * - optimizer.zero_grad(): DEVELOPED
 * - Complete PyTorch training loop: DEVELOPED
 * - model.train() / model.eval(): MENTIONED
 *
 * Central insight: the training loop pattern is IDENTICAL to the one from
 * Implementing Linear Regression — forward, loss, backward, update. The
 * only change is that PyTorch gives you objects for each step.
 */

export function TrainingLoopLesson() {
  return (
    <LessonLayout>
      {/* 1. Header */}
      <Row>
        <Row.Content>
          <LessonHeader
            title="The Training Loop"
            description="Assemble tensors, autograd, and nn.Module into a complete PyTorch training loop — the same pattern you already know."
            category="PyTorch Core"
          />
        </Row.Content>
      </Row>

      {/* Objective */}
      <Row>
        <Row.Content>
          <ObjectiveBlock>
            Write a complete PyTorch training loop from scratch. By the end, you
            can connect an <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.Module</code> model,
            a loss function object, and a{' '}
            <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.optim</code> optimizer into the
            same forward-loss-backward-update pattern you implemented in Implementing Linear Regression.
          </ObjectiveBlock>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Module 2.1 Capstone">
            This is the payoff for everything in PyTorch Core. You have
            tensors, autograd, and nn.Module. Now you put them all together.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* Constraints */}
      <Row>
        <Row.Content>
          <ConstraintBlock
            items={[
              'No DataLoader or Dataset objects—raw tensors for data',
              'No validation loops or train/val split in code',
              'No model.train() / model.eval() in practice—just mentioned',
              'No learning rate schedulers or gradient clipping',
              'No GPU training—CPU only for clarity',
            ]}
          />
        </Row.Content>
      </Row>

      {/* 2. Hook -- "You Have Already Written This" */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="You Have Already Written This"
            subtitle="The training loop you know, mapped to PyTorch"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Implementing Linear Regression, you wrote a training loop from scratch
              in NumPy. Six steps, repeated for every epoch:
            </p>

            <CodeBlock
              language="python"
              filename="from_implementing_linear_regression.py"
              code={`# The universal training loop (NumPy, from Series 1)
for epoch in range(num_epochs):
    y_hat = w * x + b              # 1. Forward pass
    loss = ((y_hat - y)**2).mean() # 2. Compute loss
    dw = ...                       # 3. Compute gradients (by hand)
    db = ...
    w -= lr * dw                   # 4. Update parameters
    b -= lr * db
    # (gradients recomputed each iteration — no accumulation in NumPy)`}
            />

            <p className="text-muted-foreground">
              Every one of these steps has a PyTorch API call. You already know what
              each call does&mdash;you just have not put them together yet:
            </p>

            <CodeBlock
              language="python"
              filename="pytorch_training_loop.py"
              code={`# The same loop in PyTorch
for epoch in range(num_epochs):
    y_hat = model(x)               # 1. nn.Module forward pass
    loss = criterion(y_hat, y)     # 2. nn.MSELoss — same formula
    optimizer.zero_grad()          # 3a. Clear accumulated gradients
    loss.backward()                # 3b. Compute all gradients (autograd)
    optimizer.step()               # 4. Update all parameters (torch.optim)`}
            />

            <p className="text-muted-foreground">
              Same heartbeat. Same order. The only new names
              are <code className="text-sm bg-muted px-1.5 py-0.5 rounded">criterion</code> (the
              loss function object) and <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer</code> (the
              optimizer object). Let&rsquo;s meet them.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Same Heartbeat, New Instruments">
            The training loop is the heartbeat: forward, loss, backward, update.
            In Series 1, you played each beat by hand. Now PyTorch gives you
            instruments: <code className="text-xs">nn.MSELoss</code> plays the loss beat,{' '}
            <code className="text-xs">optimizer.step()</code> plays the update beat.
            The rhythm does not change.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 3. Explain Part 1 -- Loss Function Objects (nn.MSELoss) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Loss Function Objects"
            subtitle="nn.MSELoss wraps the formula you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You know this formula well from Loss Functions:{' '}
              <InlineMath math="L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2" />. In the
              Autograd lesson, you computed it directly:
            </p>

            <CodeBlock
              language="python"
              filename="manual_mse.py"
              code={`# Manual MSE — from the Autograd lesson
loss = ((y_hat - y) ** 2).mean()`}
            />

            <p className="text-muted-foreground">
              PyTorch wraps this exact computation in an object:
            </p>

            <CodeBlock
              language="python"
              filename="nn_mse_loss.py"
              code={`import torch.nn as nn

criterion = nn.MSELoss()
loss = criterion(y_hat, y)  # same formula, wrapped as an object`}
            />

            <p className="text-muted-foreground">
              Verify they produce the same value:
            </p>

            <CodeBlock
              language="python"
              filename="verify_mse.py"
              code={`import torch

y_hat = torch.tensor([2.5, 0.5, 1.0])
y     = torch.tensor([3.0, 0.0, 1.0])

manual_loss = ((y_hat - y) ** 2).mean()
criterion = nn.MSELoss()
pytorch_loss = criterion(y_hat, y)

print(f"Manual:  {manual_loss.item():.6f}")   # 0.166667
print(f"PyTorch: {pytorch_loss.item():.6f}")  # 0.166667 — identical`}
            />

            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.MSELoss()</code> is{' '}
              <strong>stateless</strong>. It computes{' '}
              <InlineMath math="\text{mean}((\hat{y} - y)^2)" /> and returns a scalar tensor.
              Nothing is stored between calls. It exists as an object for two reasons:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4">
              <li>
                It is <strong>configurable</strong>&mdash;you can
                set <code className="text-sm bg-muted px-1.5 py-0.5 rounded">reduction=&apos;sum&apos;</code> instead
                of the default <code className="text-sm bg-muted px-1.5 py-0.5 rounded">reduction=&apos;mean&apos;</code>
              </li>
              <li>
                It plugs cleanly into the training loop as a <strong>callable</strong>&mdash;one
                line: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">loss = criterion(y_hat, y)</code>
              </li>
            </ul>

            <p className="text-muted-foreground">
              Other loss functions follow the same pattern:{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.CrossEntropyLoss()</code> for
              classification, <code className="text-sm bg-muted px-1.5 py-0.5 rounded">nn.L1Loss()</code> for
              mean absolute error. The API is always the
              same: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">criterion = nn.SomeLoss()</code>,
              then <code className="text-sm bg-muted px-1.5 py-0.5 rounded">loss = criterion(predictions, targets)</code>.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="No Hidden State">
            <code className="text-xs">nn.MSELoss()</code> does NOT track history,
            cache results, or hold internal state. It is a pure function wrapped
            in an object. <code className="text-xs">criterion(y_hat, y)</code> is
            exactly <code className="text-xs">((y_hat - y)**2).mean()</code>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 4. Explain Part 2 -- Optimizer Objects (torch.optim) */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Optimizer Objects"
            subtitle="torch.optim wraps the update rule you already know"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In the Autograd lesson, you performed the parameter update manually:
            </p>

            <CodeBlock
              language="python"
              filename="manual_update.py"
              code={`# Manual update from Autograd — the same formula from Gradient Descent
with torch.no_grad():
    for param in model.parameters():
        param -= lr * param.grad`}
            />

            <p className="text-muted-foreground">
              This is{' '}
              <InlineMath math="\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla L" />&mdash;the
              update rule from Gradient Descent. PyTorch wraps it in an optimizer:
            </p>

            <CodeBlock
              language="python"
              filename="optimizer_sgd.py"
              code={`import torch.optim as optim

# Give the optimizer the model's parameters and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# After loss.backward():
optimizer.step()  # performs param -= lr * param.grad for ALL parameters`}
            />

            <p className="text-muted-foreground">
              What does <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer.step()</code> actually
              do for SGD? Exactly this:
            </p>

            <CodeBlock
              language="python"
              filename="what_step_does.py"
              code={`# What optimizer.step() does internally (for SGD):
for param in model.parameters():
    param.data -= lr * param.grad
# Same formula. Same tensors. Just wrapped in a method call.`}
            />

            <p className="text-muted-foreground">
              The optimizer received <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.parameters()</code> at
              construction&mdash;it holds <strong>references</strong> to the same tensors the model
              owns. When it calls <code className="text-sm bg-muted px-1.5 py-0.5 rounded">param.data -= lr * param.grad</code>,
              it is modifying the model&rsquo;s weights directly.
            </p>

            <p className="text-muted-foreground">
              And <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer.zero_grad()</code> clears
              gradients on every parameter the optimizer was given. Same
              operation as <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.zero_grad()</code>&mdash;they
              act on the same parameter tensors.
            </p>

            <CodeBlock
              language="python"
              filename="zero_grad_equivalence.py"
              code={`# Proof: they act on the SAME tensors
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(10, 1)
y = torch.randn(10, 1)

# After a backward pass, gradients exist:
loss = ((model(x) - y) ** 2).mean()
loss.backward()
print(model.weight.grad)   # tensor([[-1.2345]])

# Clear via the optimizer:
optimizer.zero_grad()
print(model.weight.grad)   # tensor([[0.]])

# Run another backward pass:
loss = ((model(x) - y) ** 2).mean()
loss.backward()
print(model.weight.grad)   # tensor([[-1.2345]])

# Clear via the model — same effect:
model.zero_grad()
print(model.weight.grad)   # tensor([[0.]])

# Same tensors, same result. The optimizer was given model.parameters()
# at construction — they share references to the same .grad attributes.`}
            />

            <p className="text-muted-foreground">
              For Adam, <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer.step()</code> does
              more&mdash;it applies the momentum + RMSProp + bias correction you learned in
              the Optimizers lesson. But the interface is identical: call{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.step()</code> and the parameters
              are updated. The algorithm is encapsulated inside the object.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Not a Black Box">
            <code className="text-xs">optimizer.step()</code> for SGD
            is <code className="text-xs">param -= lr * grad</code>&mdash;the same update
            rule from Gradient Descent. For
            Adam, it applies momentum + adaptive rates from the Optimizers lesson.
            The abstraction is convenient, not opaque.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* 5. Check 1 -- Predict and Verify */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Check Your Understanding</h3>
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                You create <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer = torch.optim.SGD(model.parameters(), lr=0.1)</code>.
                A parameter starts at <strong>5.0</strong>. After one training step (forward, backward, step),
                the gradient is <strong>2.0</strong>. What is the new parameter value?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <p className="text-muted-foreground">
                    <InlineMath math="5.0 - 0.1 \times 2.0 = 4.8" />. Same formula
                    from Gradient Descent:{' '}
                    <InlineMath math="\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla L" />.
                  </p>
                </div>
              </details>
              <p className="text-muted-foreground text-sm mt-3">
                <strong>Follow-up:</strong> What if you forgot{' '}
                <code className="text-xs bg-muted px-1 rounded">optimizer.zero_grad()</code> and
                ran a second step where the new gradient is 3.0? What gradient does the
                optimizer see?
              </p>
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                  Show follow-up answer
                </summary>
                <div className="mt-3 text-sm border-t border-primary/20 pt-3">
                  <p className="text-muted-foreground">
                    <InlineMath math="2.0 + 3.0 = 5.0" /> (accumulated). The update
                    uses 5.0, not 3.0. The parameter becomes{' '}
                    <InlineMath math="4.8 - 0.1 \times 5.0 = 4.3" /> instead of the
                    correct <InlineMath math="4.8 - 0.1 \times 3.0 = 4.5" />. Same
                    bug from Autograd&mdash;still the #1 beginner mistake.
                  </p>
                </div>
              </details>
            </div>
          </div>
        </Row.Content>
      </Row>

      {/* 6. Explain Part 3 -- The Complete Training Loop */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Complete Training Loop"
            subtitle="The Module 2.1 capstone — all the pieces, assembled"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Time to put it all together. You will train a model on the same
              problem from Implementing Linear Regression: learn{' '}
              <InlineMath math="y = 2x + 1" /> from synthetic data.
            </p>

            <CodeBlock
              language="python"
              filename="complete_training_loop.py"
              code={`import torch
import torch.nn as nn
import torch.optim as optim

# --- Data: same y = 2x + 1 from Implementing Linear Regression ---
torch.manual_seed(42)
x = torch.randn(100, 1)          # 100 data points
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)  # true relationship + noise

# --- Model: simplest possible — one linear neuron ---
model = nn.Linear(1, 1)

# --- Loss function: MSE, same formula as always ---
criterion = nn.MSELoss()

# --- Optimizer: SGD with lr=0.1 ---
optimizer = optim.SGD(model.parameters(), lr=0.1)

# --- Training loop ---
for epoch in range(100):
    # 1. Forward pass (nn.Module)
    y_hat = model(x)

    # 2. Compute loss (nn.MSELoss)
    loss = criterion(y_hat, y)

    # 3. Clear gradients (same zero_grad from Autograd)
    optimizer.zero_grad()

    # 4. Compute all gradients (autograd backward)
    loss.backward()

    # 5. Update all parameters (optimizer step)
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss = {loss.item():.4f}")

# Check what the model learned
print(f"\\nLearned weight: {model.weight.item():.4f}")  # should be ~2.0
print(f"Learned bias:   {model.bias.item():.4f}")       # should be ~1.0`}
            />

            <p className="text-muted-foreground">
              Every line maps to a lesson you have already taken:
            </p>

            <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-4 text-sm">
              <li>
                <code className="text-xs bg-muted px-1 rounded">model(x)</code>&mdash;nn.Module
                forward pass (nn.Module)
              </li>
              <li>
                <code className="text-xs bg-muted px-1 rounded">criterion(y_hat, y)</code>&mdash;MSE
                loss (Loss Functions, this lesson)
              </li>
              <li>
                <code className="text-xs bg-muted px-1 rounded">optimizer.zero_grad()</code>&mdash;clear
                accumulated gradients (Autograd)
              </li>
              <li>
                <code className="text-xs bg-muted px-1 rounded">loss.backward()</code>&mdash;compute
                all gradients via autograd (Autograd)
              </li>
              <li>
                <code className="text-xs bg-muted px-1 rounded">optimizer.step()</code>&mdash;update
                all parameters (Gradient Descent, this lesson)
              </li>
            </ul>

            <p className="text-muted-foreground">
              The loss decreases. The weight converges to ~2.0. The bias converges
              to ~1.0. The model learned{' '}
              <InlineMath math="y = 2x + 1" />&mdash;the same function, the same result.
            </p>

            <p className="text-muted-foreground">
              Notice what you do <strong>not</strong> need to manage between iterations:
              the computational graph. Each call
              to <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model(x)</code> builds a
              fresh graph. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">loss.backward()</code> walks
              it, populates <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.grad</code>, and
              the graph is released. No cleanup needed&mdash;the next forward pass creates a
              new one. You only need to manually
              clear <em>gradients</em> (<code className="text-sm bg-muted px-1.5 py-0.5 rounded">zero_grad</code>),
              not the graph itself.
            </p>

            <p className="text-muted-foreground">
              You knew this would work. Every piece has been verified: the model
              computes the right forward pass (nn.Module), autograd computes the
              right gradients (Autograd), and the optimizer performs the right
              update (Gradient Descent). The training loop just runs those verified
              pieces in sequence.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <InsightBlock title="Of Course It Works">
            Nothing here is new. You verified the forward pass produces correct
            outputs in nn.Module. You verified autograd produces correct gradients
            in Autograd. The training loop just repeats those verified steps.
          </InsightBlock>
        </Row.Aside>
      </Row>

      {/* Side-by-side: NumPy vs PyTorch training loop */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Compare the two implementations of the same training loop:
            </p>

            <ComparisonRow
              left={{
                title: 'NumPy (Implementing Linear Regression)',
                color: 'amber',
                items: [
                  '1. Forward: y_hat = w * x + b',
                  '2. Loss: loss = ((y_hat - y)**2).mean()',
                  '3. Gradients: compute dw, db by hand',
                  '4. Update: w -= lr * dw; b -= lr * db',
                  '5. Clear: gradients recomputed each time (implicit)',
                ],
              }}
              right={{
                title: 'PyTorch (This Lesson)',
                color: 'blue',
                items: [
                  '1. Forward: y_hat = model(x)',
                  '2. Loss: loss = criterion(y_hat, y)',
                  '3. Gradients: loss.backward()',
                  '4. Update: optimizer.step()',
                  '5. Clear: optimizer.zero_grad()',
                ],
              }}
            />

            <p className="text-muted-foreground">
              Same structure. Same five steps. Same order. The PyTorch version
              scales to models with millions of parameters&mdash;the loop body
              does not change.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 7. Negative Example -- The Accumulation Trap Revisited */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="The Accumulation Trap, Revisited"
            subtitle="Same bug from Autograd — still the #1 beginner mistake"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What happens if you forget{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">optimizer.zero_grad()</code>?
            </p>

            <CodeBlock
              language="python"
              filename="accumulation_bug.py"
              code={`# BUG: missing optimizer.zero_grad()
model_buggy = nn.Linear(1, 1)
optimizer_buggy = optim.SGD(model_buggy.parameters(), lr=0.1)
criterion = nn.MSELoss()

for epoch in range(5):
    y_hat = model_buggy(x)
    loss = criterion(y_hat, y)
    # optimizer_buggy.zero_grad()  # MISSING!
    loss.backward()
    optimizer_buggy.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}, "
          f"weight.grad = {model_buggy.weight.grad.item():.4f}")`}
            />

            <p className="text-muted-foreground">
              The gradients <strong>grow</strong> each iteration because they accumulate.
              The loss oscillates or diverges instead of decreasing smoothly. You
              saw this exact behavior with raw tensors in Autograd. The fix is the
              same: clear gradients before computing new ones.
            </p>

            <p className="text-muted-foreground">
              The canonical order
              is <strong>zero_grad before backward</strong>:
            </p>

            <CodeBlock
              language="python"
              code={`optimizer.zero_grad()   # clear
loss.backward()         # compute
optimizer.step()        # use`}
            />

            <p className="text-muted-foreground">
              Clear, compute, use. This order keeps the mental model
              clean&mdash;you always start with a blank slate.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <WarningBlock title="#1 Beginner Bug">
            Whether you use raw tensors (Autograd) or an optimizer (this lesson),
            gradients accumulate by default. Always
            call <code className="text-xs">zero_grad()</code> before{' '}
            <code className="text-xs">backward()</code>.
          </WarningBlock>
        </Row.Aside>
      </Row>

      {/* 8. Explain Part 4 -- Swapping Optimizers */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Swapping Optimizers"
            subtitle="One line changes — everything else stays the same"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              Replace SGD with Adam:
            </p>

            <CodeBlock
              language="python"
              filename="swap_optimizer.py"
              code={`# Change ONE line:
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Everything else stays IDENTICAL:
for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
            />

            <p className="text-muted-foreground">
              The training loop did not change. Only the constructor
              call. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.step()</code> and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">.zero_grad()</code> work
              identically for every optimizer. The algorithm&mdash;SGD, momentum, Adam&mdash;is
              encapsulated inside the object.
            </p>

            <p className="text-muted-foreground">
              Remember Adam&rsquo;s defaults from the Optimizers
              lesson? <InlineMath math="\text{lr}=0.001" />,{' '}
              <InlineMath math="\beta_1=0.9" />,{' '}
              <InlineMath math="\beta_2=0.999" />. Those are the exact defaults
              in <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.optim.Adam</code>:
            </p>

            <CodeBlock
              language="python"
              code={`# Adam's signature — same defaults you learned
optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)`}
            />

            <p className="text-muted-foreground">
              One more thing: the <code className="text-sm bg-muted px-1.5 py-0.5 rounded">weight_decay</code> parameter.
              This is the L2 regularization / weight decay you learned in Overfitting and
              Regularization. In PyTorch, it is a single parameter:
            </p>

            <CodeBlock
              language="python"
              code={`optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)`}
            />

            <p className="text-muted-foreground text-sm">
              We will not develop this further here&mdash;just know it exists. When you need
              regularization in practice, this is where it lives.
            </p>
          </div>
        </Row.Content>
        <Row.Aside>
          <ConceptBlock title="Uniform Interface">
            The <code className="text-xs">torch.optim</code> interface is
            uniform: every optimizer
            has <code className="text-xs">.step()</code> and{' '}
            <code className="text-xs">.zero_grad()</code>. SGD, Adam, AdamW,
            RMSProp&mdash;same API, different algorithms. Swap freely.
          </ConceptBlock>
        </Row.Aside>
      </Row>

      {/* 9. Negative Example -- backward() Needs a Scalar */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="backward() Needs a Scalar"
            subtitle="The loss must be one number, not a vector"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              What happens if the loss is not reduced to a single number?
            </p>

            <CodeBlock
              language="python"
              filename="scalar_error.py"
              code={`criterion = nn.MSELoss(reduction='none')  # no reduction!
loss = criterion(y_hat, y)  # shape: (100, 1) — one loss per sample

loss.backward()  # RuntimeError: grad can be implicitly created
                 # only for scalar outputs`}
            />

            <p className="text-muted-foreground">
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> expects a
              scalar&mdash;one number that summarizes how wrong the model is.
              This is the &ldquo;one number that summarizes your wrongness&rdquo;
              from Loss Functions. Per-sample losses are useful for inspection, but
              you must reduce them (mean or sum) before calling backward.
            </p>

            <CodeBlock
              language="python"
              filename="fix_scalar.py"
              code={`# Fix: use default reduction='mean' (recommended)
criterion = nn.MSELoss()  # reduction='mean' by default

# Or reduce manually:
criterion = nn.MSELoss(reduction='none')
per_sample_losses = criterion(y_hat, y)
loss = per_sample_losses.mean()  # now a scalar
loss.backward()  # works`}
            />
          </div>
        </Row.Content>
        <Row.Aside>
          <TipBlock title="Default Is Usually Right">
            <code className="text-xs">reduction=&apos;mean&apos;</code> is the
            default for all PyTorch loss functions.
            Use <code className="text-xs">reduction=&apos;none&apos;</code> only
            when you need per-sample losses for debugging or weighted loss
            computations&mdash;and always reduce before backward.
          </TipBlock>
        </Row.Aside>
      </Row>

      {/* 10. Check 2 -- Transfer Question */}
      <Row>
        <Row.Content>
          <div className="rounded-lg border-2 border-primary/30 bg-primary/5 p-5">
            <h3 className="text-sm font-bold text-primary mb-2">Transfer Question</h3>
            <p className="text-muted-foreground text-sm mb-3">
              A colleague writes this training loop and says it is not learning:
            </p>
            <CodeBlock
              language="python"
              code={`for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()`}
            />
            <p className="text-muted-foreground text-sm mt-3 mb-3">
              What is the bug? How do you fix it?
            </p>
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-primary hover:text-primary/80 transition-colors">
                Show answer
              </summary>
              <div className="mt-3 space-y-2 text-sm border-t border-primary/20 pt-3">
                <p className="text-muted-foreground">
                  <strong>Missing <code className="text-xs bg-muted px-1 rounded">optimizer.zero_grad()</code>.</strong>{' '}
                  Gradients accumulate across epochs, so each step uses the sum of
                  all previous gradients instead of just the current one. The model
                  overshoots and oscillates.
                </p>
                <p className="text-muted-foreground">
                  <strong>Fix:</strong> Add <code className="text-xs bg-muted px-1 rounded">optimizer.zero_grad()</code> before{' '}
                  <code className="text-xs bg-muted px-1 rounded">loss.backward()</code>:
                </p>
                <CodeBlock
                  language="python"
                  code={`for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()   # <-- the fix
    loss.backward()
    optimizer.step()`}
                />
              </div>
            </details>
          </div>
        </Row.Content>
      </Row>

      {/* 11. Practice -- Colab Notebook */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="Try It Yourself"
            subtitle="Build muscle memory with the training loop"
          />
          <div className="rounded-lg border-2 border-primary/50 bg-primary/5 p-6">
            <div className="space-y-4">
              <p className="text-muted-foreground">
                Reading about training loops is not enough. Open the notebook and
                complete these exercises:
              </p>
              <ol className="list-decimal list-inside text-muted-foreground space-y-2 ml-2 text-sm">
                <li>
                  <strong>Verify nn.MSELoss matches manual MSE</strong>&mdash;compute both,
                  compare values (guided)
                </li>
                <li>
                  <strong>Verify optimizer.step() matches manual update</strong>&mdash;single
                  SGD step, compare parameter values before and after (guided)
                </li>
                <li>
                  <strong>Train linear regression in PyTorch</strong>&mdash;complete loop
                  from scratch on <InlineMath math="y = 3x - 2" /> data (supported)
                </li>
                <li>
                  <strong>Swap SGD for Adam</strong>&mdash;change one line, observe
                  convergence difference (supported)
                </li>
                <li>
                  <strong>Diagnose the accumulation bug</strong>&mdash;given a broken loop (no
                  zero_grad), predict behavior, run, fix (supported)
                </li>
                <li>
                  <strong>(Stretch) Train a 2-layer network on nonlinear data</strong>&mdash;generate{' '}
                  <InlineMath math="y = x^2" /> data, build nn.Sequential with ReLU,
                  train, verify it learns the curve (independent)
                </li>
              </ol>
              <a
                href="https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-1-4-training-loop.ipynb"
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

      {/* 12. Summary */}
      <Row>
        <Row.Content>
          <SummaryBlock
            items={[
              {
                headline: 'nn.MSELoss() = same MSE formula, wrapped as an object',
                description:
                  'Stateless, configurable, callable. criterion(y_hat, y) computes ((y_hat - y)**2).mean()—the same formula from Loss Functions.',
              },
              {
                headline: 'torch.optim.SGD = same update rule from Gradient Descent',
                description:
                  'optimizer.step() performs param -= lr * grad for every parameter. Same formula, applied to all model.parameters() at once.',
              },
              {
                headline: 'torch.optim.Adam = same algorithm from the Optimizers lesson',
                description:
                  'Momentum + RMSProp + bias correction, with the same defaults you learned: lr=0.001, beta1=0.9, beta2=0.999. One-line swap from SGD.',
              },
              {
                headline: 'optimizer.zero_grad() = still the #1 beginner bug if forgotten',
                description:
                  'Clears all parameter gradients. Same operation as model.zero_grad(), called from the optimizer. Always call before backward().',
              },
              {
                headline: 'The training loop is IDENTICAL to Implementing Linear Regression',
                description:
                  'Forward → loss → backward → update. Same pattern, same order. PyTorch gives you objects for each step, but the heartbeat does not change.',
              },
              {
                headline: 'Swapping optimizers changes ONE line',
                description:
                  'The torch.optim interface is uniform. SGD, Adam, AdamW—same .step(), same .zero_grad(). The algorithm is encapsulated; the loop stays the same.',
              },
            ]}
          />
        </Row.Content>
      </Row>

      {/* Summary paragraph */}
      <Row>
        <Row.Content>
          <div className="space-y-4">
            <p className="text-muted-foreground">
              In Series 1, you built the training loop from scratch in NumPy. In
              Autograd, you used <code className="text-sm bg-muted px-1.5 py-0.5 rounded">backward()</code> to
              automate gradient computation. In nn.Module, you packaged neurons
              into models. Now you have assembled all the pieces. The same heartbeat&mdash;forward,
              loss, backward, update&mdash;now plays with PyTorch instruments. You can
              train real models.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* 13. Next Step -- Preview What Comes Next */}
      <Row>
        <Row.Content>
          <SectionHeader
            title="What Comes Next"
            subtitle="Preview of upcoming topics"
          />
          <div className="space-y-4">
            <p className="text-muted-foreground">
              You trained a model using raw tensors as data. But real datasets have
              thousands or millions of examples that do not fit in memory at once.
              Next: <code className="text-sm bg-muted px-1.5 py-0.5 rounded">torch.utils.data.Dataset</code> and{' '}
              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">DataLoader</code>&mdash;PyTorch&rsquo;s
              system for batching, shuffling, and feeding data to your training loop.
              The mini-batch SGD you learned in Batching and SGD becomes a
              concrete <code className="text-sm bg-muted px-1.5 py-0.5 rounded">for</code> loop
              over DataLoader batches.
            </p>

            <p className="text-muted-foreground text-sm">
              One more thing: when you add dropout or batch norm to your models,
              PyTorch needs to know if you are training or
              evaluating. <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.train()</code> enables
              training behavior (dropout active, batch norm
              updating); <code className="text-sm bg-muted px-1.5 py-0.5 rounded">model.eval()</code> switches
              to inference behavior. For now, with just nn.Linear layers, this
              distinction does not matter.
            </p>
          </div>
        </Row.Content>
      </Row>

      {/* Module Complete */}
      <Row>
        <Row.Content>
          <ModuleCompleteBlock
            module="2.1"
            title="PyTorch Core"
            achievements={[
              'Create and manipulate tensors (shape, dtype, device)',
              'Use autograd for automatic gradient computation',
              'Define models with nn.Module and nn.Linear',
              'Write complete training loops with loss functions and optimizers',
              'Swap optimizers with a one-line change',
            ]}
            nextModule="2.2"
            nextTitle="Datasets & DataLoaders"
          />
        </Row.Content>
      </Row>
    </LessonLayout>
  )
}
