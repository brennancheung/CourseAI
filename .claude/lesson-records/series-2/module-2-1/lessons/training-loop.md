# Lesson: training-loop (Module 2.1, Lesson 4)

**Type:** CONSOLIDATE
**Previous lesson:** nn-module (BUILD)
**Next lesson:** First lesson of Module 2.2 (Datasets & DataLoaders, or next series module)

---

## Phase 1: Student State (Orient)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Training loop (universal pattern) | DEVELOPED | implementing-linear-regression (1.1) | Forward -> loss -> backward -> update; "heartbeat of training"; implemented from scratch in NumPy ~15 lines |
| MSE loss function | DEVELOPED | loss-functions (1.1) | Formula, squaring rationale, "wrongness score" |
| Gradient descent update rule | DEVELOPED | gradient-descent (1.1) | theta_new = theta_old - alpha * grad_L; ball-on-hill analogy |
| Learning rate as step size | DEVELOPED | learning-rate (1.1) | Goldilocks zone, oscillation/divergence failure modes |
| Mini-batch SGD | DEVELOPED | batching-and-sgd (1.3) | Polling analogy; batch_size as hyperparameter; shuffle + split + process per epoch |
| Epochs | DEVELOPED | batching-and-sgd (1.3) | One pass through all data; iterations per epoch = N/B |
| Adam optimizer (concept) | DEVELOPED | optimizers (1.3) | Combines momentum + RMSProp + bias correction; defaults lr=0.001, beta1=0.9, beta2=0.999 |
| SGD with momentum (concept) | DEVELOPED | optimizers (1.3) | EMA of gradients smooths direction; bowling ball analogy |
| Overfitting / training curves | DEVELOPED | overfitting-and-regularization (1.3) | Scissors pattern; train vs val divergence; three phases |
| Early stopping (concept) | DEVELOPED | overfitting-and-regularization (1.3) | Monitor val loss, patience hyperparameter, save best weights |
| Dropout (concept) | DEVELOPED | overfitting-and-regularization (1.3) | Randomly zero neurons during training, all active at inference |
| Weight decay / L2 regularization (concept) | DEVELOPED | overfitting-and-regularization (1.3) | Penalty on large weights; AdamW MENTIONED as practical default |
| "The complete training recipe" | INTRODUCED | overfitting-and-regularization (1.3) | Xavier/He init + batch norm + AdamW + dropout (if needed) + early stopping |
| `requires_grad` flag | DEVELOPED | autograd (2.1) | "Press Record" -- enables gradient tracking |
| `backward()` method | DEVELOPED | autograd (2.1) | "Press Rewind" -- walks computational graph backward |
| `.grad` attribute | DEVELOPED | autograd (2.1) | Gradient results stored on leaf tensors (parameters) |
| Gradient accumulation / `zero_grad()` | DEVELOPED | autograd (2.1) | Gradients accumulate by default; must clear before each step; "#1 beginner bug" |
| `torch.no_grad()` context manager | DEVELOPED | autograd (2.1) | Pauses recording; semantically necessary during parameter updates |
| Manual training step with autograd | INTRODUCED | autograd (2.1) | Forward, backward, update (in no_grad), zero_grad -- complete pattern shown, not practiced in loop |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1) | __init__ + forward(); layers in __init__, computation in forward() |
| nn.Linear | DEVELOPED | nn-module (2.1) | IS w*x + b; weight shape (out, in), bias shape (out,); requires_grad=True via nn.Parameter |
| model.parameters() | DEVELOPED | nn-module (2.1) | Collects all learnable tensors; "parameters are knobs" as concrete API |
| model.zero_grad() | DEVELOPED | nn-module (2.1) | Clears gradients on ALL parameters at once |
| nn.Sequential | INTRODUCED | nn-module (2.1) | Simple layer stacks; cannot express skip connections |
| nn.ReLU as module | INTRODUCED | nn-module (2.1) | Same max(0, x) from 1.2, wrapped as module for Sequential |
| Linear collapse in PyTorch | DEVELOPED | nn-module (2.1) | Proved: stacking nn.Linear without activations collapses; ReLU breaks collapse |
| Dynamic computational graph | INTRODUCED | nn-module (2.1) | Each model(x) call builds a fresh graph |
| PyTorch tensor creation API | DEVELOPED | tensors (2.1) | torch.tensor(), torch.randn(), etc. |
| Tensor attributes: shape, dtype, device | DEVELOPED | tensors (2.1) | "Shape, dtype, device -- check these first" |
| Matrix multiplication with `@` | DEVELOPED | tensors (2.1) | y_hat = X @ w + b |
| Device management (.to(), .cpu()) | INTRODUCED | tensors (2.1) | Standard pattern for GPU placement |

### Mental Models Already Established

- **"Training loop = forward -> loss -> backward -> update"** -- the universal pattern from Series 1; "heartbeat of training"
- **"Parameters are knobs the model learns"** -- now concrete as model.parameters()
- **"requires_grad = press Record"** -- autograd tracking
- **"backward() = press Rewind"** -- graph traversal for gradients
- **"zero_grad() = clear the tape"** -- gradients accumulate; clear before each step
- **"no_grad() = pause Recording"** -- semantically necessary during updates
- **"Same math, better organization"** -- nn.Module packages known computation into reusable blocks
- **"nn.Linear IS w*x + b"** -- the abstraction does not change the math
- **"Not magic -- automation"** -- autograd automates what the student did manually
- **"Three representations of the same network"** -- math (1.3) -> raw tensors (autograd) -> nn.Module; same numbers
- **"The complete training recipe"** -- Xavier/He init + batch norm + AdamW + dropout + early stopping
- **"Ball rolling downhill"** / loss landscape -- gradient descent intuition
- **"Polling analogy"** -- mini-batch gradient approximates full gradient
- **"Adam = momentum + RMSProp"** -- smooth direction + normalize magnitude
- **"Scissors pattern"** -- train/val divergence = overfitting
- **"Regularization increases training loss -- that is the point"** -- val loss is what matters

### What Was Explicitly NOT Covered (Relevant Here)

- `torch.optim` or any optimizer objects (explicitly deferred from autograd AND nn-module)
- Loss function objects like `nn.MSELoss` (explicitly deferred from autograd AND nn-module)
- Full training loops in PyTorch (autograd showed one step; nn-module verified forward+backward only)
- `model.train()` vs `model.eval()` mode switching (deferred from nn-module)
- Saving/loading models (`torch.save`, `state_dict`)
- GPU placement of models (`.to(device)` on a model)
- DataLoader / Dataset objects (not yet in curriculum)
- Learning rate schedulers in PyTorch
- Cross-entropy loss / multi-class classification

### Readiness Assessment

The student is fully prepared. This is a CONSOLIDATE lesson -- the capstone of Module 2.1. Every conceptual building block is already in place at DEVELOPED depth: the training loop pattern, loss functions, optimizers, autograd, nn.Module. The student has separately learned (a) what a training loop does (Series 1), (b) how PyTorch computes gradients (autograd), and (c) how to define models (nn.Module). The only genuinely new content is the PyTorch API for two things they already understand: loss function objects (`nn.MSELoss`) and optimizer objects (`torch.optim.SGD`, `torch.optim.Adam`). Everything else is assembling known pieces. The emotional arc is integration and payoff: "I can put it all together and train a real model in PyTorch."

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to write a complete PyTorch training loop by connecting nn.Module models, loss function objects, and torch.optim optimizers into the same forward-loss-backward-update pattern they already know.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Training loop (universal pattern) | DEVELOPED | DEVELOPED | implementing-linear-regression (1.1) | OK | This lesson reimplements the same pattern in PyTorch; student must be fluent in forward -> loss -> backward -> update |
| MSE loss function (concept + formula) | DEVELOPED | DEVELOPED | loss-functions (1.1) | OK | `nn.MSELoss` wraps the same formula; student must recognize the math inside the object |
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent (1.1) | OK | `optimizer.step()` performs the same theta_new = theta_old - lr * grad; student must see this as the same operation |
| Mini-batch SGD (concept) | DEVELOPED | DEVELOPED | batching-and-sgd (1.3) | OK | The training loop iterates over batches; student needs the epoch/batch mental model |
| Adam optimizer (concept) | DEVELOPED | DEVELOPED | optimizers (1.3) | OK | `torch.optim.Adam` wraps the algorithm the student learned; must recognize defaults (lr=0.001, beta1=0.9, beta2=0.999) |
| nn.Module subclass pattern | DEVELOPED | DEVELOPED | nn-module (2.1) | OK | The model being trained is an nn.Module; student must be fluent in defining and calling models |
| model.parameters() | DEVELOPED | DEVELOPED | nn-module (2.1) | OK | Passed to the optimizer constructor; student must understand this returns all learnable tensors |
| `backward()` method | DEVELOPED | DEVELOPED | autograd (2.1) | OK | Called on the loss tensor; student understands this walks the graph and populates .grad |
| Gradient accumulation / `zero_grad()` | DEVELOPED | DEVELOPED | autograd (2.1) | OK | `optimizer.zero_grad()` replaces `model.zero_grad()` and individual `tensor.zero_()` -- student must understand WHY it is called before backward |
| `torch.no_grad()` | DEVELOPED | DEVELOPED | autograd (2.1) | OK | Used during evaluation/inference; student understands this pauses recording |
| nn.Linear | DEVELOPED | DEVELOPED | nn-module (2.1) | OK | Used in the model definition; student can inspect weights and verify computation |
| Overfitting / training curves | DEVELOPED | DEVELOPED | overfitting-and-regularization (1.3) | OK | Printing/plotting train loss is the minimal version of this; student knows what the curve should look like |

**Gap resolution:** No gaps. All prerequisites are at or above required depth. This is the CONSOLIDATE lesson -- every piece has been taught individually. The new content is purely the PyTorch API (`nn.MSELoss`, `torch.optim.SGD`, `torch.optim.Adam`, `optimizer.step()`, `optimizer.zero_grad()`) wrapping concepts the student already understands deeply.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **`optimizer.zero_grad()` is different from `model.zero_grad()`** | Two different objects (optimizer vs model) calling zero_grad(). "The optimizer has its own gradient clearing mechanism." | Show that `optimizer.zero_grad()` literally calls `param.grad.zero_()` on the same parameter tensors that `model.parameters()` returns. Print `.grad` before and after both calls -- same effect. The optimizer was given the model's parameters at construction time; they share the same tensors. | Early, when introducing the optimizer and showing the training step. Dispel immediately because the autograd lesson drilled zero_grad as critical. |
| **`loss.backward()` in a loop recomputes the entire graph from scratch** | The student knows "each model(x) call builds a fresh graph" from nn-module. They might think backward() needs the full graph to be rebuilt. "Does backward destroy the graph? Do I need to do something special?" | Show the lifecycle: (1) `y_hat = model(x)` builds the graph, (2) `loss = criterion(y_hat, y)` extends it, (3) `loss.backward()` walks it backward and populates .grad, (4) the graph is released (freed). Next iteration, step 1 builds a fresh graph. There is no graph to preserve between iterations. The loop naturally handles this: each forward pass creates a new graph, each backward pass consumes it. | During the annotated training loop walkthrough, when explaining what happens per iteration. |
| **`optimizer.step()` replaces the manual update rule entirely -- "I do not need to understand what it does"** | Software engineers trust abstractions. "I just call step() and it works." This undermines the connection to Series 1. | Show what `optimizer.step()` does for SGD: it executes `param.data -= lr * param.grad` for every parameter. This is the SAME update rule from gradient-descent (1.1): theta_new = theta_old - alpha * grad. For Adam, it additionally applies the momentum and RMSProp calculations from the optimizers lesson (1.3). The abstraction is a convenience, not a black box. | Right after introducing optimizer.step(), with a concrete comparison to the manual update from the autograd lesson. |
| **You need separate code for SGD vs Adam -- they are fundamentally different** | Different algorithms with different formulas. "Adam has momentum and adaptive rates, surely the training loop is different." | The training loop is IDENTICAL for both. Only the constructor call changes: `optim.SGD(model.parameters(), lr=0.01)` vs `optim.Adam(model.parameters(), lr=0.001)`. Every other line stays the same. This is the power of the optimizer abstraction: the algorithm is encapsulated behind `.step()`. | When swapping SGD for Adam in the second training example. The student should feel surprise at how little changes. |
| **Loss function objects do more than compute a number** | `nn.MSELoss()` is a class instance. Software engineers expect objects to hold state, cache results, etc. "Is it tracking something internally?" | `nn.MSELoss()` is stateless. It computes `mean((y_hat - y)^2)` and returns a scalar tensor. Nothing is stored between calls. It exists as an object so it can (a) participate in the computational graph via autograd and (b) be configured once (e.g., `reduction='sum'` vs `reduction='mean'`). Show: `criterion = nn.MSELoss()` then `criterion(y_hat, y)` produces the same number as `((y_hat - y)**2).mean()`. | When introducing nn.MSELoss, immediately after showing the manual MSE computation for comparison. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Manual MSE vs nn.MSELoss side-by-side** | Positive (introductory) | Show that `nn.MSELoss` computes the exact same value as `((y_hat - y)**2).mean()`. Demystify the loss function object. | Directly connects to the MSE formula from loss-functions (1.1). Minimal code, one comparison. Establishes the pattern: "PyTorch object wraps known math." |
| **Manual update vs optimizer.step() side-by-side** | Positive (introductory) | Show that `optimizer.step()` for SGD performs `param.data -= lr * param.grad`, the identical update rule from gradient-descent (1.1). | Mirrors the autograd lesson's manual update step. The student sees three representations of the same operation: math (1.1) -> raw tensors (autograd) -> optimizer object (here). Completes the "three abstractions" arc. |
| **Complete PyTorch training loop on synthetic linear regression** | Positive (primary) | Full training loop: create synthetic data, define model, define loss, define optimizer, train for N epochs, print loss decreasing. This is the Module 2.1 capstone moment. | Reimplements implementing-linear-regression (1.1) in PyTorch. Same task, same data generation pattern, new tools. The student sees the familiar problem expressed with every PyTorch concept from this module. Full circle. |
| **Swap SGD for Adam with one line change** | Positive (stretch) | Change the optimizer constructor from `optim.SGD(...)` to `optim.Adam(...)`. Rest of the loop is identical. Train on the same problem, compare convergence speed. | Disproves "you need separate code for different optimizers." Demonstrates the torch.optim abstraction's power. Connects to the optimizers lesson (1.3) where Adam was taught conceptually. |
| **Forgetting optimizer.zero_grad() -- the accumulation trap revisited** | Negative | Remove `optimizer.zero_grad()` from the loop. Show gradients growing, loss diverging or oscillating. Compare to the correct loop. | Direct callback to the autograd lesson's "#1 beginner bug." The student has seen this trap before with raw tensors. Seeing it again in the training loop context reinforces the lesson and shows the bug manifests the same way regardless of abstraction level. |
| **Calling backward() without a scalar loss** | Negative | Attempt `loss.backward()` where loss has shape `(batch_size,)` instead of a scalar `()`. PyTorch raises an error. Show that `reduction='none'` produces per-sample losses (not reduced), and explain why backward() requires a scalar. | Defines a boundary: backward() operates on a scalar loss that represents the single "wrongness score." This connects to the loss function lesson's "one number that summarizes how wrong you are." Prevents a common debugging frustration. |

---

## Phase 3: Design

### Narrative Arc

You have learned every piece of the PyTorch puzzle separately. You created tensors and moved them between devices. You used autograd to compute gradients automatically -- the same gradients you computed by hand in backprop-worked-example. You packaged neurons and layers into nn.Module, verifying the same math at every abstraction level. But you have not yet put it all together. In the autograd lesson, you saw a preview: one forward pass, one backward pass, one manual parameter update. In the nn-module lesson, you verified gradients but never actually trained anything. The individual pieces work. The question is: what does a COMPLETE training loop look like when you use all of them together? The answer will feel familiar. The pattern is the same one you implemented from scratch in implementing-linear-regression: forward, compute loss, backward, update. The heartbeat does not change. What changes is that PyTorch gives you objects for each step: `nn.MSELoss` computes the loss, `loss.backward()` computes all gradients, and `optimizer.step()` performs the update. Two new API concepts, zero new algorithmic concepts. After this lesson, you can train real models in PyTorch -- the capstone moment for Module 2.1.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example with real numbers** | Reproduce the same linear regression task from implementing-linear-regression (1.1) in PyTorch. Generate the same y = 2x + 1 synthetic data, train, watch loss decrease, verify the model learns w ~= 2 and b ~= 1. | The student has a personal connection to this specific task -- it was the capstone of Module 1.1. Seeing the same problem solved in PyTorch makes the connection visceral: "same task, new tools, same result." The specific parameter values (w converging to 2, b to 1) provide concrete verification. |
| **Symbolic / Code** | Progressive code examples: (1) manual MSE vs nn.MSELoss, (2) manual update vs optimizer.step(), (3) complete training loop, (4) same loop with Adam. The code IS the lesson -- this is an API lesson. | This is fundamentally an API lesson. Every concept has a direct code translation. The progression from manual -> wrapped manages cognitive load while building toward the full picture. The student reads code as fluently as prose. |
| **Visual (annotated diagram)** | Side-by-side comparison: the NumPy training loop from implementing-linear-regression (1.1) vs the PyTorch training loop from this lesson, with colored annotations mapping each line to its counterpart. | The visual modality here is structural comparison, not a plot or graph. The student needs to SEE that the structure is identical -- same 4 steps, same order, different syntax. Color-coding makes the 1:1 mapping unmissable. |
| **Verbal / Analogy** | "Same heartbeat, new instruments." The training loop is the heartbeat (forward-loss-backward-update). In Series 1, you played each beat by hand. Now PyTorch gives you instruments: nn.MSELoss plays the loss beat, optimizer.step() plays the update beat. The rhythm does not change. The instruments make it easier to play at scale. | Extends the "heartbeat of training" metaphor from implementing-linear-regression. Adding "instruments" maps cleanly: each PyTorch API object is an instrument that plays one part of the pattern. Does not conflict with prior analogies. |
| **Intuitive ("of course" feeling)** | After the complete training loop runs and loss decreases: "You knew this would work. Every piece has been verified: the model computes the right forward pass (nn-module), autograd computes the right gradients (autograd), and the optimizer performs the right update (gradient-descent from 1.1). The training loop just runs those verified pieces in sequence." | Lands the CONSOLIDATE payoff. The student should feel zero surprise that the loop works -- every component was individually proven. The "of course" is the emotional signal that understanding is solid. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. `nn.MSELoss` (and loss function objects generally) -- wraps the MSE formula the student knows into a callable object
  2. `torch.optim.SGD` / `torch.optim.Adam` (and `optimizer.step()`, `optimizer.zero_grad()`) -- wraps the update rule the student knows into an optimizer object
- **Previous lesson load:** BUILD (nn-module -- 2-3 new concepts, API-focused)
- **This lesson's load:** CONSOLIDATE -- appropriate as module capstone after BUILD
- **Assessment:** Two new API concepts, both wrapping deeply understood algorithms. The cognitive load is minimal because the student has the universal training loop at DEVELOPED depth, MSE at DEVELOPED depth, and both SGD and Adam at DEVELOPED depth. The only new thing is the PyTorch syntax for calling these known operations. This is the lightest lesson in the module, which is correct for a CONSOLIDATE lesson -- the weight is on integration and payoff, not new ideas.

### Connections to Prior Concepts

| Prior Concept | How It Connects |
|--------------|-----------------|
| Training loop from implementing-linear-regression (1.1) | This lesson reimplements the same task (y = 2x + 1 linear regression) with PyTorch. Same structure, same steps, same result. "Same heartbeat, new instruments." |
| MSE loss from loss-functions (1.1) | `nn.MSELoss` computes the identical formula: mean((y_hat - y)^2). Show side-by-side to prove it. |
| Gradient descent update from gradient-descent (1.1) | `optimizer.step()` for SGD does param -= lr * grad. Same formula, wrapped in a method call. |
| Manual update step from autograd (2.1) | The autograd lesson showed `with torch.no_grad(): param -= lr * param.grad`. The optimizer replaces this with `optimizer.step()`. The student sees three representations: math -> manual tensor update -> optimizer.step(). |
| zero_grad() from autograd (2.1) | `optimizer.zero_grad()` replaces `model.zero_grad()` and individual tensor zero_grad(). Same operation, different caller. The "#1 beginner bug" stays the #1 beginner bug. |
| Adam from optimizers (1.3) | `torch.optim.Adam(model.parameters(), lr=0.001)` instantiates the algorithm the student learned. Same defaults (lr=0.001, betas=(0.9, 0.999)). One line swap from SGD. |
| model.parameters() from nn-module (2.1) | Passed directly to the optimizer constructor. "Here are my knobs; please learn them for me." The optimizer needs to know which parameters to update. |
| Overfitting / training curves from overfitting-and-regularization (1.3) | Printing loss per epoch is the minimal training curve. The student watches loss decrease and connects it to the training curve concept. |
| "Three representations of the same network" from nn-module (2.1) | Extended here: the training loop itself now has three representations: NumPy from-scratch (1.1), manual autograd (2.1 autograd), and full PyTorch (this lesson). Same algorithm at every abstraction level. |

**Potentially misleading prior analogies:** None significant. The "heartbeat" metaphor extends cleanly. The "press Record / Rewind" metaphor from autograd applies directly -- `loss.backward()` is still "press Rewind." The main risk is the student thinking the optimizer does something fundamentally different from the manual update; the side-by-side comparison addresses this directly.

### Scope Boundaries

**This lesson IS about:**
- `nn.MSELoss` as a loss function object (comparison to manual MSE)
- `torch.optim.SGD` and `torch.optim.Adam` as optimizer objects
- `optimizer.step()` as the update step
- `optimizer.zero_grad()` as gradient clearing (connecting to autograd's zero_grad)
- The complete PyTorch training loop: forward -> loss -> backward -> step
- Swapping optimizers with a one-line change
- The gradient accumulation trap revisited in the training loop context
- Reimplementing the linear regression task from 1.1 in PyTorch

**This lesson is NOT about:**
- DataLoader or Dataset objects (future lesson / module)
- Validation loops or train/val split in code (concept known from 1.3, PyTorch implementation deferred)
- `model.train()` vs `model.eval()` mode switching (brief mention for completeness, not developed)
- Learning rate schedulers in code (`torch.optim.lr_scheduler`)
- Cross-entropy loss or multi-class classification
- Saving/loading models (`torch.save`, `state_dict`)
- GPU training (model and data on GPU) -- briefly mentioned, not practiced
- Regularization in code (dropout layers, weight_decay parameter) -- the concepts are known; the PyTorch API is for a future lesson
- Batch training with DataLoader (this lesson uses full-batch for simplicity; mini-batch is deferred to when DataLoader is taught)
- Custom loss functions
- Gradient clipping in code

**Target depth:**
- `nn.MSELoss`: DEVELOPED -- student understands it wraps MSE, can compare to manual, knows it is stateless
- `torch.optim.SGD` / `torch.optim.Adam`: DEVELOPED -- student can construct, call step(), call zero_grad(), swap between them
- Complete PyTorch training loop: DEVELOPED -- student can write the full loop from memory
- `optimizer.step()`: DEVELOPED -- student knows it performs the update rule
- `optimizer.zero_grad()`: DEVELOPED -- student knows it clears parameter gradients, connects to autograd lesson
- `model.train()` / `model.eval()`: MENTIONED -- named, not practiced

### Lesson Outline

#### 1. Context + Constraints
What this lesson is: the capstone of Module 2.1, putting tensors + autograd + nn.Module together into a complete training loop. What it is NOT: no DataLoaders, no validation loops, no GPU training, no regularization code. Frame: "You have all the pieces. This lesson assembles them."

#### 2. Hook -- "You Have Already Written This"
Type: recognition callback. Show the 6-step training loop from implementing-linear-regression (1.1):

```
1. Forward pass: y_hat = w * x + b
2. Compute loss: L = mean((y_hat - y)^2)
3. Compute gradients (by hand / NumPy)
4. Update parameters: w -= lr * dw, b -= lr * db
5. Clear gradients (implicit in NumPy -- no accumulation)
6. Repeat
```

Then: "Every one of these steps has a PyTorch API call. You already know what each call does -- you just have not put them together yet." Map each step to its PyTorch equivalent:

```
1. y_hat = model(x)           # nn.Module forward
2. loss = criterion(y_hat, y)  # nn.MSELoss
3. loss.backward()             # autograd
4. optimizer.step()            # torch.optim
5. optimizer.zero_grad()       # clears .grad on all parameters
6. Repeat
```

The student should immediately see: "I know all of these." The only new names are `criterion` (loss object) and `optimizer` (optimizer object).

#### 3. Explain Part 1 -- Loss Function Objects (nn.MSELoss)
Start with what the student knows: `loss = ((y_hat - y)**2).mean()` from the loss-functions lesson and the autograd lesson.

Then show the PyTorch equivalent:
```python
criterion = nn.MSELoss()
loss = criterion(y_hat, y)
```

Verify they produce the same value. The object is stateless -- it just computes the formula. It exists as an object because (a) it is configurable (`reduction='mean'` vs `reduction='sum'`), and (b) it plugs cleanly into the training loop as a callable.

Address misconception: "Loss function objects do more than compute a number." They do not. `nn.MSELoss()` is `((y_hat - y)**2).mean()` with no hidden state.

Brief mention of other loss functions: `nn.CrossEntropyLoss` for classification (not taught here -- just named so the student knows it exists). The pattern is the same: `criterion = nn.SomeLoss()`, then `criterion(predictions, targets)`.

#### 4. Explain Part 2 -- Optimizer Objects (torch.optim)
Start with what the student knows: the manual update from the autograd lesson.

```python
with torch.no_grad():
    for param in model.parameters():
        param -= lr * param.grad
```

Then show the PyTorch equivalent:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# ... later, after backward:
optimizer.step()
```

Show what `optimizer.step()` does for SGD: exactly the same loop (`param.data -= lr * param.grad`). The optimizer was given `model.parameters()` at construction -- it holds references to the same tensors.

Show `optimizer.zero_grad()`: clears `.grad` on every parameter the optimizer was given. Same operation as `model.zero_grad()`, called from the optimizer instead.

Address misconception: "optimizer.zero_grad() is different from model.zero_grad()." They operate on the same parameter tensors. The optimizer received them via `model.parameters()`.

Address misconception: "optimizer.step() is a black box." For SGD, it is literally `param -= lr * grad`. For Adam, it applies the momentum + RMSProp + bias correction the student learned in the optimizers lesson. The abstraction is convenient, not opaque.

#### 5. Check 1 -- Predict and Verify
"You create `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`. After one training step (forward, backward, step), the gradient for a parameter is 2.0. What is the new parameter value if it started at 5.0?"

Student must reason: `5.0 - 0.1 * 2.0 = 4.8`. Same formula as gradient-descent (1.1). Then verify with code.

Follow-up: "What if you forgot `optimizer.zero_grad()` and ran a second step where the new gradient is 3.0? What gradient does the optimizer see?" Answer: `2.0 + 3.0 = 5.0` (accumulated). The update uses 5.0, not 3.0. Callback to the "#1 beginner bug."

#### 6. Explain Part 3 -- The Complete Training Loop (The Payoff)
This is the capstone moment. Build it step by step:

1. **Data:** Generate synthetic y = 2x + 1 data (same as implementing-linear-regression)
2. **Model:** `model = nn.Linear(1, 1)` (simplest possible -- one neuron)
3. **Loss:** `criterion = nn.MSELoss()`
4. **Optimizer:** `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`
5. **Training loop:**
```python
for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

Annotate every line, connecting each to the lesson where the concept was taught:
- `model(x)` -- nn.Module forward pass (nn-module)
- `criterion(y_hat, y)` -- MSE loss (loss-functions 1.1, this lesson)
- `optimizer.zero_grad()` -- clear accumulated gradients (autograd)
- `loss.backward()` -- compute all gradients via autograd (autograd)
- `optimizer.step()` -- update all parameters (gradient-descent 1.1, this lesson)

Run it. Loss decreases. Print `model.weight` and `model.bias` -- they converge to ~2.0 and ~1.0. The model learned the same function.

Side-by-side comparison: the NumPy loop from implementing-linear-regression vs this PyTorch loop. Color-annotated to show 1:1 correspondence.

#### 7. Negative Example -- The Accumulation Trap Revisited
Remove `optimizer.zero_grad()` from the loop. Run again. Loss oscillates or diverges instead of decreasing smoothly. Print `.grad` values to show they grow each iteration.

"Same bug, same fix. Whether you are using raw tensors (autograd lesson) or an optimizer (this lesson), gradients accumulate by default. Always call zero_grad() before backward()."

Note: Show `optimizer.zero_grad()` placement. The canonical order is: zero_grad BEFORE backward, not after step. Explain why either placement works but zero_grad-before-backward is the convention (cleaner mental model: "clear, compute, use").

#### 8. Explain Part 4 -- Swapping Optimizers
Change one line:
```python
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Everything else stays the same. Train again. The student should notice: same loop structure, potentially faster convergence.

Address misconception: "SGD and Adam need different training loops." They do not. The `torch.optim` interface is uniform: `.step()` and `.zero_grad()` work identically. The algorithm is encapsulated inside the optimizer object.

Connect to the optimizers lesson: "Remember Adam's defaults? lr=0.001, beta1=0.9, beta2=0.999. Those are the exact defaults in `torch.optim.Adam`."

Brief mention: `weight_decay` parameter exists in both SGD and Adam (AdamW). "This is the weight decay / L2 regularization you learned in overfitting-and-regularization. In PyTorch, it is a single parameter: `optim.Adam(params, lr=0.001, weight_decay=1e-4)`." Do not develop -- just plant the seed.

#### 9. Negative Example -- backward() Needs a Scalar
Show what happens when `reduction='none'` is used:
```python
criterion = nn.MSELoss(reduction='none')
loss = criterion(y_hat, y)  # shape: (batch_size,)
loss.backward()  # RuntimeError
```

The error occurs because `backward()` expects a scalar (one number). The loss must be reduced (mean or sum) to a single value before calling backward. This connects to "one number that summarizes how wrong you are" from loss-functions (1.1).

Fix: use the default `reduction='mean'`, or call `.mean()` on the unreduced loss.

#### 10. Check 2 -- Transfer Question
"A colleague writes this training loop and says it is not learning:
```python
for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
```
What is the bug? How do you fix it?"

The student must identify: missing `optimizer.zero_grad()`. Gradients accumulate across epochs. Fix: add `optimizer.zero_grad()` before `loss.backward()`.

#### 11. Practice -- Colab Notebook
Exercises:

1. **Verify nn.MSELoss matches manual MSE** -- compute both, compare values (guided)
2. **Verify optimizer.step() matches manual update** -- single SGD step, compare parameter values before/after (guided)
3. **Train linear regression in PyTorch** -- complete loop from scratch on y = 3x - 2 data (supported)
4. **Swap SGD for Adam** -- change one line, observe convergence difference (supported)
5. **Diagnose the accumulation bug** -- given a broken loop (no zero_grad), predict behavior, run, fix (supported)
6. **(Stretch) Train a 2-layer network on nonlinear data** -- generate y = x^2 data, build nn.Sequential with ReLU, train, verify it learns the curve (independent)

#### 12. Summarize -- Mental Model Echo
- `nn.MSELoss()` = the same MSE formula from Series 1, wrapped as a callable object
- `torch.optim.SGD(model.parameters(), lr)` = the same update rule from gradient-descent, applied to all model parameters
- `torch.optim.Adam(model.parameters(), lr)` = the same Adam algorithm from the optimizers lesson
- `optimizer.step()` = perform one parameter update (the "update" beat)
- `optimizer.zero_grad()` = clear all gradients (the "clear" beat -- still the #1 beginner bug if forgotten)
- The training loop pattern is IDENTICAL to implementing-linear-regression: forward -> loss -> backward -> update
- Swapping optimizers changes ONE line; the loop stays the same

"In Series 1, you built the training loop from scratch in NumPy. In the autograd lesson, you used backward() to automate gradient computation. In the nn-module lesson, you packaged neurons into models. Now you have assembled all the pieces. The same heartbeat -- forward, loss, backward, update -- now plays with PyTorch instruments. You can train real models."

#### 13. Next Step -- Preview What Comes Next
"You trained a model using raw tensors as data. But real datasets have thousands or millions of examples that do not fit in memory at once. The next step: `torch.utils.data.Dataset` and `DataLoader` -- PyTorch's system for batching, shuffling, and feeding data to your training loop. The mini-batch SGD you learned in batching-and-sgd becomes a concrete for-loop over DataLoader batches."

Brief mention of `model.train()` / `model.eval()`: "When you add dropout or batch norm to your models, PyTorch needs to know if you are training or evaluating. `model.train()` enables training behavior (dropout active, batch norm updating); `model.eval()` switches to inference behavior. For now, with just nn.Linear layers, this distinction does not matter."

ModuleCompleteBlock: Module 2.1 (PyTorch Core) complete. The student can build and train neural networks in PyTorch.

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues -- the student will not get lost or form wrong mental models. However, four improvement-level findings would significantly strengthen the lesson if addressed.

### Findings

#### [IMPROVEMENT] — Misconception #2 (graph lifecycle) is not addressed

**Location:** Sections 3-6 (Loss Function Objects through Complete Training Loop)
**Issue:** The planning document identifies misconception #2: "loss.backward() in a loop recomputes the entire graph from scratch" -- the student may worry about graph persistence between iterations. The plan specifies addressing this "during the annotated training loop walkthrough, when explaining what happens per iteration." The built lesson never addresses graph lifecycle. There is no mention of the graph being built fresh each forward pass and freed after backward(). The nn-module lesson INTRODUCED dynamic computational graphs ("each model(x) call builds a fresh graph"), but that was the previous lesson and the student may not connect it to the training loop context.
**Student impact:** A student who internalized the "dynamic graph" concept from nn-module may be fine. But a student who is shaky on this will have a nagging question: "What happens to the graph between iterations? Do I need to do something to reset it?" This is especially relevant because the lesson explicitly covers zero_grad (clearing gradients between iterations) but says nothing about the graph itself, which could make the student think: "If I need to manually clear gradients, do I also need to manually clear the graph?"
**Suggested fix:** Add 1-2 sentences in the "Complete Training Loop" section (section 6), right after the annotated code, explaining the lifecycle: "Each call to `model(x)` builds a fresh computational graph. `loss.backward()` walks it and then the graph is released. No cleanup needed -- the next forward pass creates a new graph. You only need to manually clear *gradients* (zero_grad), not the graph itself."

#### [IMPROVEMENT] — Misconception #1 (optimizer.zero_grad vs model.zero_grad) addressed weakly

**Location:** Section 4 (Optimizer Objects), the zero_grad equivalence code block
**Issue:** The planning document identifies misconception #1: "optimizer.zero_grad() is different from model.zero_grad()." The plan specifies showing that they operate on the same parameter tensors. The built lesson includes a code block showing `model.zero_grad()` and `optimizer.zero_grad()` with a comment that they work on the same tensors. However, the plan called for printing `.grad` before and after both calls to prove they have the same effect. The lesson shows the equivalence declaratively (comments) rather than demonstrating it empirically. The "show, don't tell" principle means the student should SEE the proof, not just be told.
**Student impact:** A software engineer who reads comments may still wonder: "But how do I know they really do the same thing?" The declarative statement is likely sufficient for most students given the context, but empirical verification was planned for good reason.
**Suggested fix:** Either (a) add a brief code snippet showing `.grad` values before and after both calls, or (b) add this as a note that the Colab exercise "Verify optimizer.step() matches manual update" also covers this. Option (b) is lighter and may be more appropriate for a CONSOLIDATE lesson.

#### [IMPROVEMENT] — First example for optimizer.step() could be more concrete

**Location:** Section 4 (Optimizer Objects), the "what_step_does.py" code block
**Issue:** The code showing what optimizer.step() does internally uses `self.param_groups[0]['params']` which is the actual PyTorch internal API. While the comment says "Same formula. Same tensors. Just wrapped in a method call," the use of `param_groups` introduces an unfamiliar data structure that the student has never seen. The planning document says to show `param.data -= lr * param.grad` as the internal operation. The built lesson does this, but wraps it in the param_groups iteration which is unnecessarily complex for a "demystification" moment.
**Student impact:** The student may think "wait, what is param_groups? Is there something more complex going on here?" The very thing this example is trying to disprove (that optimizer.step is a black box) is slightly undermined by exposing internal implementation detail that was not taught.
**Suggested fix:** Simplify the internal code to use the same iteration pattern the student knows:
```python
# What optimizer.step() does internally (for SGD):
for param in model.parameters():
    param.data -= lr * param.grad
# Same formula. Same tensors. Just wrapped in a method call.
```
This uses `model.parameters()` which the student knows from nn-module, rather than the internal `param_groups` API.

#### [IMPROVEMENT] — Missing modality: no visual diagram of the training loop lifecycle

**Location:** Whole lesson
**Issue:** The planning document identifies 5 modalities, including "Visual (annotated diagram)" described as "side-by-side comparison: the NumPy training loop from implementing-linear-regression vs the PyTorch training loop from this lesson, with colored annotations mapping each line to its counterpart." The built lesson uses a ComparisonRow component for this comparison (section after the complete training loop), which provides a structured but text-only comparison. ComparisonRow lists bullet points, not a color-annotated code comparison. The plan specifically called for "colored annotations mapping each line to its counterpart." While the ComparisonRow is a reasonable implementation, it is a weaker visual modality than what was planned.
**Student impact:** The ComparisonRow works well enough. The student can see the mapping. But a true side-by-side code comparison with highlighted correspondences would make the structural identity more visceral. The current version requires the student to mentally map between the bullet points, rather than seeing the code lines side-by-side with color links.
**Suggested fix:** This is a judgment call. The ComparisonRow is adequate for a CONSOLIDATE lesson where the student already knows both patterns. A possible enhancement would be to place the two code blocks (NumPy and PyTorch) side-by-side with matching line comments (e.g., "# 1. Forward" on both), but this may not be worth the effort for this iteration. Consider whether the existing ComparisonRow is "good enough" or whether the visual could be strengthened.

#### [POLISH] — Spaced em dashes in ConstraintBlock items and SummaryBlock descriptions

**Location:** ConstraintBlock (line 83-87), SummaryBlock (lines 861, 886)
**Issue:** Several ConstraintBlock items and SummaryBlock description strings use ` — ` (spaced em dash) rather than the `&mdash;` (no spaces) convention. Examples: `'No DataLoader or Dataset objects — raw tensors for data'`, `'criterion(y_hat, y) computes ((y_hat - y)**2).mean() — the same formula from Loss Functions.'`. The writing style rule says em dashes must have no spaces: `word—word` not `word — word`.
**Student impact:** Minimal -- this is a consistency issue. The student likely will not notice, but it deviates from the style guide.
**Suggested fix:** Replace ` — ` with `—` in all ConstraintBlock items and SummaryBlock description strings. These are string literals so use the literal `—` character rather than `&mdash;`.

#### [POLISH] — Colab notebook link may not exist yet

**Location:** Section 11 (Practice), the Colab link
**Issue:** The link points to `notebooks/2-1-4-training-loop.ipynb`. Checking the git status, this file is not listed among tracked or untracked files. The exercises are well-designed in the lesson text, but the notebook may not exist yet.
**Student impact:** If the student clicks the link and gets a 404, the practice section -- arguably the most important part of a CONSOLIDATE lesson -- becomes unusable.
**Suggested fix:** Create the notebook, or add a note in the planning document that the notebook needs to be created before this lesson ships.

#### [POLISH] — Check 1 follow-up answer reveals before student reflects

**Location:** Section 5 (Check 1), the follow-up inside the details/summary
**Issue:** The follow-up question ("What if you forgot optimizer.zero_grad()...") is inside the same `<details>` block as the answer to the first question. The student opens the details to check their answer to question 1 and immediately sees the follow-up question AND its answer. They cannot attempt the follow-up before seeing the solution. The pedagogical intent (reinforcing the accumulation bug) is undermined by the layout: the student reads the answer passively rather than predicting.
**Student impact:** The follow-up becomes passive reading rather than active recall. The student sees "2.0 + 3.0 = 5.0" before having a chance to reason about it.
**Suggested fix:** Move the follow-up question outside the `<details>` block (visible after the student opens the first answer), and wrap the follow-up's answer in its own `<details>` block. This preserves the predict-then-verify flow for both questions.

### Review Notes

**What works well:**
- The central narrative arc is strong. The "same heartbeat, new instruments" framing is consistently applied and reinforces the CONSOLIDATE pattern.
- Connection to prior lessons is excellent throughout. Every new API concept is explicitly mapped to its Series 1 or earlier Module 2.1 equivalent.
- The negative examples (accumulation trap, scalar loss) are well-chosen and well-placed.
- Cognitive load is appropriate for a CONSOLIDATE lesson -- two genuinely new API concepts, everything else is assembly.
- The exercise design in the Practice section is well-scaffolded (guided -> supported -> independent).
- The ModuleCompleteBlock is a satisfying capstone moment.
- Em dashes in prose are consistently correct (no spaces around `&mdash;`).
- Scope boundaries are respected -- the lesson does not creep into DataLoader, validation, or GPU training.

**Patterns to watch:**
- The lesson leans heavily on code blocks as the primary modality (9 CodeBlock instances). For this API-focused CONSOLIDATE lesson, code IS the right modality. But the visual comparison (planned as color-annotated) ended up as a ComparisonRow, which is weaker.
- The planned misconception about graph lifecycle is the most significant omission. The other 4 misconceptions are addressed.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All four improvement findings from iteration 1 were addressed. The lesson is pedagogically sound, correctly scoped, and well-connected to prior material. The two remaining polish items are minor and do not require another review pass.

### Iteration 1 Fix Verification

| Iteration 1 Finding | Severity | Status | Notes |
|---------------------|----------|--------|-------|
| Graph lifecycle not addressed | IMPROVEMENT | FIXED | Lines 506-515 now explain that each forward pass builds a fresh graph, backward walks and releases it, and only gradients need manual clearing. Clear and well-placed. |
| optimizer.zero_grad vs model.zero_grad addressed weakly | IMPROVEMENT | FIXED | Lines 317-342 now show a full empirical proof: backward, print grad, clear via optimizer, print, backward again, print, clear via model, print. Same result both ways. Strong. |
| optimizer.step() example too complex (param_groups) | IMPROVEMENT | FIXED | Lines 296-299 now use `for param in model.parameters(): param.data -= lr * param.grad`—the iteration pattern the student knows from nn-module. Clean. |
| Visual modality (ComparisonRow vs color-annotated code) | IMPROVEMENT | ACCEPTED | Not changed, which was explicitly allowed by the iteration 1 review as a judgment call. The ComparisonRow is adequate for a CONSOLIDATE lesson. |
| Em dashes in string literals | POLISH | FIXED | ConstraintBlock items and SummaryBlock descriptions now use unspaced em dashes. |
| Colab notebook may not exist | POLISH | STILL OPEN | Notebook at `notebooks/2-1-4-training-loop.ipynb` still does not exist. Expected—notebook creation is a separate deliverable. |
| Check 1 follow-up reveals before student reflects | POLISH | FIXED | Follow-up question is now outside the first details block, with its own separate details/summary for the answer. Predict-then-verify flow preserved. |

### Findings

#### [POLISH] — Colab notebook still does not exist

**Location:** Section 11 (Practice), the Colab link
**Issue:** Carried forward from iteration 1. The link points to `notebooks/2-1-4-training-loop.ipynb`, which does not exist in the repository. The exercise descriptions in the lesson text are well-designed, but the notebook needs to be created before the lesson ships.
**Student impact:** Clicking the link produces a 404. The practice section—the most important part of a CONSOLIDATE lesson—becomes unusable.
**Suggested fix:** Create the notebook as a separate task. The lesson text already specifies all six exercises with scaffolding levels (guided/supported/independent), so the notebook content is fully specified.

#### [POLISH] — zero_grad equivalence code uses undefined x and y variables

**Location:** Section 4 (Optimizer Objects), the `zero_grad_equivalence.py` code block (lines 319-342)
**Issue:** The code snippet references `x` and `y` on lines 324 and 333 (`loss = ((model(x) - y) ** 2).mean()`) without defining them. While the student can infer these are tensor variables from context (they have been used throughout the lesson), the code block is labeled with a filename (`zero_grad_equivalence.py`) which implies it is a self-contained runnable example. All other named code blocks in the lesson are either self-contained or clearly labeled as fragments.
**Student impact:** A student who tries to copy and run this code block in isolation would get a `NameError`. The pedagogical intent (proving optimizer.zero_grad and model.zero_grad are equivalent) is not undermined—the student understands the proof from the print statements. This is cosmetic.
**Suggested fix:** Add two lines at the top of the code block defining x and y, e.g., `x = torch.randn(10, 1)` and `y = torch.randn(10, 1)`. This makes the snippet self-contained without adding cognitive load.

### Review Notes

**What works well:**
- All iteration 1 improvements were addressed effectively. The graph lifecycle explanation (lines 506-515) is particularly well-placed—it comes right after the student would naturally wonder "what about the graph?" because they just learned about zero_grad (clearing gradients between iterations).
- The zero_grad equivalence proof (lines 317-342) is now one of the strongest sections in the lesson. The empirical verification pattern (backward, print, clear, print, backward, print, clear, print) is methodical and convincing.
- The optimizer.step() demystification (lines 296-299) now uses `model.parameters()` which the student has used since nn-module. Clean and familiar.
- The Check 1 restructuring (two separate details blocks) properly preserves predict-then-verify for both questions. This is a small but meaningful pedagogical improvement.
- The central narrative arc remains the lesson's greatest strength. "Same heartbeat, new instruments" is consistently applied from hook through summary. Every new API concept is immediately connected to its known equivalent.
- The five misconceptions from the planning document are all addressed at appropriate points in the lesson.
- Scope discipline is excellent—the lesson never strays into DataLoader, validation loops, GPU training, or other deferred topics.
- The ModuleCompleteBlock provides a satisfying capstone for Module 2.1.

**Overall assessment:** This lesson is ready to ship (pending notebook creation). It is a strong CONSOLIDATE lesson that delivers on its promise: the student assembles known pieces into a working PyTorch training loop with minimal new cognitive load. The emotional arc—"I already know this"—is the correct feeling for the capstone of Module 2.1.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept (5 planned), each with rationale
- [x] At least 2 positive examples + 1 negative example (4 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 identified)
- [x] Cognitive load = 2 new concepts (within budget -- loss objects + optimizer objects)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
