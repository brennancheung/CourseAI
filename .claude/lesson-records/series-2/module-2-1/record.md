# Module 2.1: PyTorch Core — Record

**Goal:** The student can build, train, and debug a neural network in PyTorch by understanding tensors, autograd, nn.Module, and the training loop — translating every concept from Series 1 into working code.
**Status:** Complete (4 of 4 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| PyTorch tensor creation API | DEVELOPED | tensors | `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`, `torch.from_numpy()` — mapped 1:1 to NumPy equivalents |
| Tensor attributes: shape, dtype, device | DEVELOPED | tensors | "Three attributes that determine everything"; `.shape`, `.dtype`, `.device` — debugging first instinct |
| Tensor reshaping (view/reshape) | INTRODUCED | tensors | `view()` vs `reshape()` — "use reshape() unless you have a specific reason"; `-1` auto-dimension |
| Element-wise arithmetic on tensors | DEVELOPED | tensors | `+`, `*`, `**` — same as NumPy; `*` vs `@` distinction called out explicitly |
| Matrix multiplication with `@` operator | DEVELOPED | tensors | `@` is shorthand for `torch.matmul()`; used in forward pass `y_hat = X @ w + b` |
| Broadcasting (PyTorch) | INTRODUCED | tensors | "Same rules as NumPy, same behavior" — bias addition example |
| Tensor dimension vocabulary (scalar/vector/matrix/batch) | INTRODUCED | tensors | 0D through 3D with code examples; 3D = "batch of matrices" |
| PyTorch float32 default (vs NumPy float64) | DEVELOPED | tensors | "Measuring a rough sketch with a micrometer" analogy; mini-batch gradients are approximate, float64 is wasted precision |
| PyTorch dtype system | INTRODUCED | tensors | float32, float64, float16, int64, bool — table with use cases; `.float()`, `.half()` conversion methods |
| GPU as parallel processor | INTRODUCED | tensors | CPU (8-16 fast cores) vs GPU (thousands of simple cores); ComparisonRow visual |
| Device management (`.to()`, `.cpu()`) | INTRODUCED | tensors | Standard pattern: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| GPU transfer overhead | INTRODUCED | tensors | Small tensors faster on CPU; timing comparison showing 10-element (CPU 12x faster) vs 10M-element (GPU 350x faster) |
| Device mismatch error | INTRODUCED | tensors | "All tensors in an operation must be on the same device"; fix with `.to(X.device)` |
| NumPy-PyTorch interop | DEVELOPED | tensors | `torch.from_numpy()` (shared memory), `.numpy()` (shared memory), `torch.tensor()` (copies) |
| Shared memory between NumPy and tensors | DEVELOPED | tensors | Zero-copy conversion; modifying one changes the other; explicit warning about silent bugs |
| `.detach().cpu().numpy()` chain | MENTIONED | tensors | Previewed as "the most common pattern"; `.detach()` explained in next lesson |
| PyTorch shape argument convention | INTRODUCED | tensors | `torch.zeros(3, 4)` not `np.zeros((3, 4))` — WarningBlock aside |
| `requires_grad` flag | DEVELOPED | autograd | Tells PyTorch to record operations on a tensor; "press Record" analogy; does not change tensor type/data — only enables tracking |
| `backward()` method | DEVELOPED | autograd | Walks the computational graph backward applying the chain rule; "press Rewind" analogy; called on the output (loss), not on parameters |
| `.grad` attribute | DEVELOPED | autograd | Where gradient results are stored — on the leaf tensors (parameters), not returned from backward(); an attribute, not a return value |
| Gradient accumulation (PyTorch default) | DEVELOPED | autograd | `.grad` adds by default, does not replace; correct for fan-out/shared params; must call `zero_grad()` between training steps |
| `zero_grad()` for clearing accumulated gradients | DEVELOPED | autograd | Resets `.grad` to zero before each backward pass; forgetting this is the #1 beginner PyTorch bug |
| `torch.no_grad()` context manager | DEVELOPED | autograd | Pauses operation recording; semantically necessary during parameter updates to prevent recording the update as part of the graph |
| `.detach()` method | DEVELOPED | autograd | Severs a tensor from the computational graph; "snip the tape" analogy; connects to "no path = no gradient" from computational-graphs (1.3) |
| `.detach().cpu().numpy()` chain (explained) | DEVELOPED | autograd | Full explanation: detach from graph, move to CPU, convert to array — completes the MENTIONED preview from tensors |
| `grad_fn` attribute on result tensors | INTRODUCED | autograd | Operation nodes have `grad_fn` recording what computation produced them; leaf parameters have `grad_fn=None` |
| `retain_graph=True` argument | MENTIONED | autograd | Prevents graph destruction after backward(); normally not needed because each forward pass builds a new graph |
| Autograd as API for computational graph traversal | DEVELOPED | autograd | backward() performs the identical backward traversal the student did by hand in backprop-worked-example (1.3); same numbers, same algorithm |
| Manual training step with autograd | INTRODUCED | autograd | Forward, backward, update (in no_grad), zero_grad — complete pattern shown but not practiced in a loop (deferred to training-loop) |
| nn.Module subclass pattern (__init__ + forward()) | DEVELOPED | nn-module | Define layers in __init__, wire data flow in forward(); any nn.Module/nn.Parameter assigned to self is auto-tracked; model(x) builds a fresh computational graph each call |
| nn.Linear as w*x + b | DEVELOPED | nn-module | nn.Linear(in, out) = out neurons each computing w*x + b; weight shape (out, in), bias shape (out,); requires_grad=True by default via nn.Parameter |
| nn.Parameter (auto requires_grad wrapper) | INTRODUCED | nn-module | Thin wrapper around tensor that sets requires_grad=True and registers for parameter collection; student does not create manually for standard layers |
| model.parameters() for collecting all learnable tensors | DEVELOPED | nn-module | One call returns every nn.Parameter in the model; model.zero_grad() clears all their gradients at once; "parameters are knobs" metaphor becomes a concrete API |
| model.zero_grad() as batch zero_grad | DEVELOPED | nn-module | Clears gradients on ALL parameters at once; replaces per-tensor zero_grad() from autograd |
| nn.Sequential for simple layer stacks | INTRODUCED | nn-module | List layers in order, PyTorch runs them sequentially; cannot express skip connections, branching, or conditional logic |
| nn.ReLU as a module layer | INTRODUCED | nn-module | Same max(0, x) from activation-functions (1.2), wrapped as a module for use in Sequential; no parameters |
| Linear collapse in PyTorch code | DEVELOPED | nn-module | Stacking nn.Linear without activations collapses to single linear transformation; proved by computing W_eff = W2 @ W1 and showing identical output |
| Skip/residual connection code pattern | INTRODUCED | nn-module | forward() returns self.linear(relu(x)) + x; cannot be expressed in Sequential; previews ResNets from training-dynamics (1.3) |
| model(x) vs model.forward(x) convention | INTRODUCED | nn-module | model(x) calls __call__ which runs hooks then forward(); calling forward() directly skips hooks; always use model(x) |
| Dynamic computational graph (per-call) | INTRODUCED | nn-module | Each call to model(x) builds a fresh graph; forward() can contain conditionals and loops — different path per input |
| nn.MSELoss (loss function object) | DEVELOPED | training-loop | Wraps the same MSE formula from loss-functions (1.1) as a stateless callable object; `criterion(y_hat, y)` = `((y_hat - y)**2).mean()`; configurable via `reduction` parameter |
| torch.optim.SGD (optimizer object) | DEVELOPED | training-loop | Wraps the gradient descent update rule: `optimizer.step()` performs `param.data -= lr * param.grad` for all parameters; constructed with `model.parameters()` and `lr` |
| torch.optim.Adam (optimizer object) | DEVELOPED | training-loop | Same momentum + RMSProp + bias correction from optimizers (1.3); defaults lr=0.001, betas=(0.9, 0.999); one-line swap from SGD |
| optimizer.step() (parameter update) | DEVELOPED | training-loop | Performs the update rule for all parameters the optimizer was given; for SGD = `param -= lr * grad`; for Adam = momentum + adaptive rates; same formula from gradient-descent (1.1) wrapped in a method |
| optimizer.zero_grad() (gradient clearing) | DEVELOPED | training-loop | Clears `.grad` on every parameter the optimizer holds; operates on the same tensors as `model.zero_grad()` (proven empirically); canonical placement: before `backward()` |
| Complete PyTorch training loop pattern | DEVELOPED | training-loop | forward -> loss -> backward -> update = same heartbeat from implementing-linear-regression (1.1); `model(x)`, `criterion(y_hat, y)`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` |
| Uniform optimizer interface (torch.optim) | DEVELOPED | training-loop | All optimizers share `.step()` and `.zero_grad()`; swapping SGD for Adam changes one constructor line; the algorithm is encapsulated inside the object |
| Graph lifecycle in the training loop | INTRODUCED | training-loop | Each `model(x)` builds a fresh graph; `loss.backward()` walks it and releases it; no cleanup needed between iterations; only gradients need manual clearing |
| backward() requires a scalar loss | INTRODUCED | training-loop | `loss.backward()` expects a scalar (one number); `reduction='none'` produces per-sample losses that must be reduced before backward; connects to "one number that summarizes wrongness" from loss-functions (1.1) |
| model.train() / model.eval() mode switching | MENTIONED | training-loop | Named as relevant when dropout or batch norm are present; not practiced (only nn.Linear layers used in this module) |
| weight_decay optimizer parameter | MENTIONED | training-loop | L2 regularization from overfitting-and-regularization (1.3) lives in the optimizer constructor; `optim.Adam(params, weight_decay=1e-4)` |

## Per-Lesson Summaries

### tensors (Lesson 1 — BUILD)

**Concepts taught:** PyTorch tensor creation/manipulation API, tensor attributes (shape/dtype/device), reshaping, arithmetic, matrix multiply, float32 default, GPU placement, NumPy interop.

**Mental models established:**
- "Tensors are NumPy arrays that know where they live" — the core framing for the entire lesson
- "Same interface, different engine" — PyTorch mirrors NumPy intentionally; the new 20% is device management and autograd
- "Shape, dtype, device — check these first" — the debugging trinity
- "Measuring a rough sketch with a micrometer" — why float32 is right-sized, not less precise
- "GPU wins at scale, CPU wins for small operations" — transfer overhead as the deciding factor

**Analogies used:**
- "NumPy arrays that can ride the GPU" — tensors as containers that know their location
- "Rough sketch with a micrometer" — float64 overkill for approximate gradients
- CPU vs GPU comparison as "8 powerful workers vs thousands of simple workers"

**How concepts were taught:**
- Hook: side-by-side NumPy vs PyTorch code doing the SAME linear regression data creation from Series 1. Differences are cosmetic. Low activation energy.
- Tensor creation: code-first with NumPy equivalents commented inline; four creation patterns
- Shapes: visual grid showing 0D/1D/2D/3D tensors with code and dimension labels
- dtypes: table format with use cases; code showing creation and conversion
- GPU: ComparisonRow visual (CPU vs GPU), then timing code showing empirical speedup/slowdown by tensor size
- NumPy interop: code demonstrating shared memory surprise, with warning about silent mutation bugs
- Practice: Colab notebook with guided exercises (no custom widget — code IS the modality)

**Checks and exercises:**
- Check 1: Predict-and-verify — translate NumPy forward pass to PyTorch, predict shape/dtype/device
- Check 2: Transfer question — "colleague says GPU is always faster, what do you tell them?"
- Colab exercises: create tensors, move to GPU, compute forward pass, reshape images

**What is NOT covered (explicitly deferred):**
- Autograd / `requires_grad` (lesson 2: autograd)
- `torch.no_grad()` (lesson 2: autograd)
- `nn.Module` or layers (lesson 3: nn-module)
- Training loops (lesson 4: training-loop)
- Advanced indexing, scatter/gather
- Memory management, pinned memory, CUDA streams

### autograd (Lesson 2 — STRETCH)

**Concepts taught:** `requires_grad`, `backward()`, `.grad`, gradient accumulation and `zero_grad()`, `torch.no_grad()`, `.detach()`, `grad_fn` on operation nodes, reproducing hand-computed backprop in PyTorch.

**Mental models established:**
- "requires_grad = press Record" — tells PyTorch to track operations; the tensor does the same math, but now operations are logged
- "backward() = press Rewind" — walks the computational graph backward, applying the chain rule at each node
- ".grad = the result, stored as an attribute" — gradients live on leaf tensors, not returned from backward()
- "zero_grad() = clear the tape for the next step" — gradients accumulate by default (correct for fan-out); clear before each backward pass
- "no_grad() = pause Recording" — semantically necessary during parameter updates, not just an optimization
- "detach() = snip the tape" — severs a tensor from the graph; completes "no path = no gradient" from computational-graphs

**Analogies used:**
- "Press Record / Rewind / Snip the tape" — unified recording metaphor across all six autograd API concepts
- "Not magic — automation" — autograd does not replace understanding, it automates the algorithm the student already knows
- Fan-out gradient summation (from computational-graphs 1.3) to explain why gradient accumulation is the default behavior

**How concepts were taught:**
- Hook: showed the 4 exact gradient values from backprop-worked-example, asked "what if all seven steps happened in one line?"
- requires_grad: simplest scalar example (x=3.0, y=x^2), showed `grad_fn=<PowBackward0>` as evidence of recording, contrast with untracked tensor
- backward()/grad: called backward() on scalar, verified dy/dx = 2x = 6 by hand
- The Payoff (central moment): reproduced the exact 2-layer network from backprop-worked-example with same weights/input/target; backward() produces identical gradients; side-by-side comparison table
- Mermaid computational graph diagram: proper node-and-edge graph with parameter nodes (purple), operation nodes (slate), gradient annotations (red), matching Series 1.3 visual conventions
- AutogradExplorer widget: two modes — manual (step-by-step with "incoming x local = outgoing" real numbers) vs autograd (one-click backward()); adjustable parameters to verify match holds
- Three gotchas: gradient accumulation (predict-then-reveal trap), no_grad (semantic necessity, not just optimization), detach (completes tensors lesson promise)
- Training step preview: side-by-side ComparisonRow of Manual(NumPy) vs Autograd(PyTorch); complete code block showing forward/backward/update/zero_grad pattern

**Checks and exercises:**
- Check 1: Predict x.grad for z = (3x+1)^2 at x=2 — tests chain rule tracing through the API (answer: 42)
- Check 2: Transfer question — colleague forgets zero_grad(), gradients grow, model diverges; diagnose as accumulation bug
- Colab exercises: polynomial gradients (guided), reproduce backprop network (guided), accumulation trap (guided), manual training step (supported), detach blocking gradients (independent)

**What is NOT covered (explicitly deferred):**
- nn.Module, nn.Linear, or any layer abstractions (lesson 3: nn-module)
- torch.optim or optimizer objects (lesson 4: training-loop)
- Loss function objects like nn.MSELoss (lesson 4: training-loop)
- Higher-order gradients / torch.autograd.grad()
- Custom autograd functions (torch.autograd.Function)
- Full training loops (one manual update shown as preview only)
- Dynamic vs static graph comparison

### nn-module (Lesson 3 — BUILD)

**Concepts taught:** nn.Module subclass pattern (__init__ + forward()), nn.Linear as w*x + b, nn.Parameter, model.parameters(), model.zero_grad(), nn.Sequential, nn.ReLU as a module, linear collapse in PyTorch, skip/residual connection pattern, model(x) vs model.forward(x), dynamic computational graphs.

**Mental models established:**
- "Same math, better organization" — nn.Module packages the exact same computation (neurons, layers, activations) into reusable building blocks; the math does not change, the organization does
- "nn.Module is not just a container" — forward() builds a fresh computational graph each call; conditionals and loops mean each input can follow a different path (dynamic graph)
- "nn.Linear IS w*x + b" — not a mysterious abstraction; the exact same matrix multiply and bias addition the student already knows
- "LEGO bricks" — nn.Linear is the basic brick (specific computation + connection points); nn.Sequential snaps them in a tower; custom forward() lets you design any assembly

**Analogies used:**
- LEGO bricks for nn.Module composition — individual bricks (layers), straight towers (Sequential), custom assemblies (Module subclass with forward())
- "The Recording is on from the start" — callback to autograd's "press Record" metaphor; nn.Parameter sets requires_grad=True automatically
- "Three representations of the same network" — math (backprop-worked-example), raw tensors (autograd), nn.Module (this lesson); same numbers every time

**How concepts were taught:**
- Hook: callback to autograd's 4 individual tensors, then "imagine 4,353 parameters" escalation; problem before solution
- nn.Linear: simplest example first (nn.Linear(3,1) = one neuron), then manual verification that layer(x) == x @ weight.T + bias produces same numbers; misconception addressed immediately ("not mysterious")
- nn.Module: TwoLayerNet class reproducing the autograd lesson's network; __init__ = what, forward() = how; dynamic forward() example with conditionals to disprove "just a container" misconception
- The Payoff: same weights (w1=0.5, b1=0.1, w2=-0.3, b2=0.2), same input, same target — gradients match autograd exactly; side-by-side comparison table; ComparisonRow of raw tensors vs nn.Module
- nn.Sequential: shown as convenience for simple stacks; nn.ReLU() introduced as module form of max(0,x)
- Linear collapse negative example: built 2-layer Sequential without activations, computed W_eff = W2 @ W1 to prove mathematical equivalence to single layer, then showed ReLU breaks the collapse
- Skip connection: custom Module with forward() returning self.linear(relu(x)) + x; Mermaid diagram showing bypass path; explains why Sequential is insufficient
- model(x) vs model.forward(x): brief code block showing convention; hooks explanation kept minimal ("trust the convention")
- Architecture diagrams: two Mermaid diagrams — basic 2-layer network (code-to-architecture mapping) and skip connection (bypass path visualization)
- Practice: Colab notebook with 6 exercises (2 guided, 3 supported, 1 independent stretch)

**Checks and exercises:**
- Check 1: Predict parameter count for nn.Linear(5, 3) — tests understanding of weight matrix shape (5*3 + 3 = 18); follow-up with nn.Linear(100, 50)
- Check 2: Transfer question — colleague's 3-layer Sequential without activations performs like linear regression; diagnose as linear collapse, fix by adding nn.ReLU()
- Colab exercises: inspect nn.Linear layers (guided), verify nn.Linear IS w*x+b (guided), build 2-layer Module subclass (supported), convert to Sequential (supported), linear collapse experiment (supported), skip-connection Module (independent stretch)

**What is NOT covered (explicitly deferred):**
- torch.optim or optimizer objects (lesson 4: training-loop)
- Loss function objects like nn.MSELoss (lesson 4: training-loop)
- Full training loops (forward + backward verified, not trained)
- model.train() vs model.eval() mode switching (lesson 4 or later)
- Saving/loading models (torch.save, state_dict)
- GPU placement of models (.to(device) on a model)
- Convolutional, recurrent, or other architecture-specific layers
- Hooks (pre-forward, post-forward) — mentioned only to justify model(x) convention
- Custom autograd functions
- nn.Dropout as a module (mentioned in planning, intentionally omitted to avoid clutter)

### training-loop (Lesson 4 — CONSOLIDATE)

**Concepts taught:** nn.MSELoss as a loss function object, torch.optim.SGD and torch.optim.Adam as optimizer objects, optimizer.step() and optimizer.zero_grad(), complete PyTorch training loop, swapping optimizers with one line, graph lifecycle in the training loop, backward() requiring a scalar loss.

**Mental models established:**
- "Same heartbeat, new instruments" — the training loop is the heartbeat (forward-loss-backward-update); PyTorch gives you instruments (nn.MSELoss, optimizer.step()) for each beat; the rhythm does not change
- "Of course it works" — every piece was individually verified (forward pass in nn-module, gradients in autograd, update rule in gradient-descent); the training loop just runs verified pieces in sequence
- "Not a black box" — optimizer.step() for SGD is param -= lr * grad (the exact update rule from gradient-descent); for Adam, it applies the momentum + adaptive rates from the optimizers lesson; the abstraction is convenient, not opaque
- "Uniform interface" — all torch.optim optimizers share .step() and .zero_grad(); the algorithm is encapsulated; swapping SGD for Adam changes one constructor line

**Analogies used:**
- "Same heartbeat, new instruments" — extends the "heartbeat of training" metaphor from implementing-linear-regression (1.1); each PyTorch object is an instrument that plays one part of the pattern
- "Clear, compute, use" — the canonical order for gradients: zero_grad (clear), backward (compute), step (use)

**How concepts were taught:**
- Hook: showed the NumPy training loop from implementing-linear-regression side-by-side with its PyTorch equivalent; 1:1 mapping; "you already know all of these"
- nn.MSELoss: manual MSE vs nn.MSELoss producing identical values; addressed misconception that loss objects hold state (they are stateless)
- torch.optim: manual update vs optimizer.step() producing identical results; empirical proof that optimizer.zero_grad() and model.zero_grad() operate on the same tensors (printed .grad before and after both)
- Complete training loop: reimplemented the y=2x+1 linear regression from Series 1; every line annotated with its source lesson; ComparisonRow showing NumPy vs PyTorch structure
- Graph lifecycle: explained that each forward pass builds a fresh graph, backward walks and releases it, no cleanup needed between iterations
- Swapping optimizers: changed one constructor line from SGD to Adam; loop body identical; connected Adam defaults to the optimizers lesson
- Negative examples: (1) accumulation trap revisited — missing zero_grad causes growing gradients and divergence; (2) backward() on non-scalar loss raises RuntimeError
- weight_decay mentioned as where L2 regularization lives in PyTorch; not developed
- model.train()/model.eval() mentioned for dropout/batch norm context; not practiced

**Checks and exercises:**
- Check 1: Predict parameter value after one SGD step (5.0 - 0.1 * 2.0 = 4.8); follow-up: predict accumulated gradient when zero_grad is missing (2.0 + 3.0 = 5.0)
- Check 2: Transfer question — diagnose missing zero_grad in colleague's training loop
- Colab exercises: verify nn.MSELoss matches manual MSE (guided), verify optimizer.step() matches manual update (guided), train linear regression from scratch (supported), swap SGD for Adam (supported), diagnose accumulation bug (supported), train 2-layer network on nonlinear data (independent stretch)

**What is NOT covered (explicitly deferred):**
- DataLoader or Dataset objects (Module 2.2)
- Validation loops or train/val split in code (concept known from 1.3, implementation deferred)
- model.train() / model.eval() in practice (mentioned only)
- Learning rate schedulers (torch.optim.lr_scheduler)
- Cross-entropy loss or multi-class classification
- Saving/loading models (torch.save, state_dict)
- GPU training (model and data on GPU)
- Regularization in code (dropout layers, weight_decay developed)
- Batch training with DataLoader (full-batch used for simplicity; mini-batch deferred to Module 2.2)
- Custom loss functions
- Gradient clipping

## Key Mental Models and Analogies

| Model/Analogy | Established In | Available For |
|---------------|---------------|---------------|
| "Tensors are NumPy arrays that know where they live" | tensors | autograd, nn-module, training-loop |
| "Same interface, different engine" | tensors | autograd (extends: same graph traversal, automated) |
| "Shape, dtype, device — check these first" | tensors | All future lessons (debugging pattern) |
| "Rough sketch with micrometer" (float32 sufficiency) | tensors | Mixed precision training (Series 2.3) |
| "GPU wins at scale" | tensors | GPU training lesson (2.3), batch size decisions |
| "Parameters are knobs" (from Series 1) | what-is-learning (1.1) | Extended here: tensors are the containers holding those knobs |
| "Training loop = forward -> loss -> backward -> update" (from Series 1) | implementing-linear-regression (1.1) | Foreshadowed: "By lesson 4, you'll rewrite this in PyTorch" |
| "requires_grad = press Record" | autograd | Available for: nn-module (parameters auto-set requires_grad), training-loop |
| "backward() = press Rewind" | autograd | Available for: training-loop, debugging-and-visualization |
| ".grad = result stored as attribute" | autograd | Available for: nn-module (accessing parameter gradients), training-loop |
| "zero_grad() = clear the tape" | autograd | Available for: training-loop (optimizer.zero_grad()), debugging |
| "no_grad() = pause Recording" | autograd | Available for: training-loop (evaluation mode), inference |
| "detach() = snip the tape" | autograd | Available for: nn-module, training-loop (metrics logging), debugging |
| "Not magic — automation" (autograd) | autograd | Reinforces: the manual work in Series 1.3 was worth doing; understanding what is inside the black box |
| Fan-out explains gradient accumulation default | autograd | Connects computational-graphs (1.3) to PyTorch API behavior |
| "Same math, better organization" (nn.Module) | nn-module | Available for: training-loop (model is the organized container), all future lessons |
| "nn.Module is not just a container" (dynamic graph) | nn-module | Available for: training-loop, advanced architectures; each forward() call builds a fresh graph |
| "nn.Linear IS w*x + b" | nn-module | Available for: training-loop, datasets-and-dataloaders, all architecture lessons |
| "LEGO bricks" (modular composition) | nn-module | Available for: building larger architectures (CNNs, transformers); bricks compose into bigger structures |
| "Three representations of the same network" | nn-module | Capstone connection: math (1.3) -> raw tensors (autograd) -> nn.Module; same numbers at every abstraction level |
| "Same heartbeat, new instruments" | training-loop | Extends the "heartbeat of training" metaphor; PyTorch objects are instruments for each beat of forward-loss-backward-update |
| "Of course it works" | training-loop | Every piece was individually verified; the training loop just runs verified pieces in sequence; the capstone emotional payoff |
| "Not a black box" (optimizer.step) | training-loop | optimizer.step() for SGD = param -= lr * grad; for Adam = momentum + adaptive rates; the abstraction is convenient, not opaque |
| "Clear, compute, use" (gradient lifecycle) | training-loop | Canonical ordering: zero_grad (clear), backward (compute), step (use); keeps the mental model clean |
| "Uniform interface" (torch.optim) | training-loop | All optimizers share .step() and .zero_grad(); swap freely; the algorithm is encapsulated inside the object |
