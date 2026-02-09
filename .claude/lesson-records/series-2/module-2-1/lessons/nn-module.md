# Lesson: nn-module (Module 2.1, Lesson 3)

**Type:** BUILD
**Previous lesson:** autograd (STRETCH)
**Next lesson:** training-loop (CONSOLIDATE)

---

## Phase 1: Student State (Orient)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Neuron = weighted sum + bias | DEVELOPED | neuron-basics (1.2) | Multi-input linear regression; formula `output = w*x + b`; widget-explored |
| Layer = group of neurons | INTRODUCED | neuron-basics (1.2) | Same inputs, different weight sets; matrix form mentioned but not practiced |
| Network = stacked layers | INTRODUCED | neuron-basics (1.2) | Output of one layer feeds input of next |
| Activation function (concept) | DEVELOPED | activation-functions (1.2) | `output = sigma(w*x + b)` -- nonlinear function after linear combination |
| ReLU formula and shape | DEVELOPED | activation-functions (1.2), activation-functions-deep-dive (1.2) | `max(0, x)`, range [0, inf), hinge at zero |
| Linear collapse | DEVELOPED | neuron-basics (1.2) | `W2(W1x + b1) + b2 = Wx + b` -- stacking linear layers collapses to one |
| `requires_grad` flag | DEVELOPED | autograd (2.1) | "Press Record" -- tells PyTorch to track operations |
| `backward()` method | DEVELOPED | autograd (2.1) | "Press Rewind" -- walks computational graph backward, applying chain rule |
| `.grad` attribute | DEVELOPED | autograd (2.1) | Gradient results stored on leaf tensors (parameters), not returned |
| `zero_grad()` for clearing gradients | DEVELOPED | autograd (2.1) | Resets `.grad` to zero; forgetting = #1 beginner bug |
| `torch.no_grad()` context manager | DEVELOPED | autograd (2.1) | Pauses recording; semantically necessary during parameter updates |
| `.detach()` method | DEVELOPED | autograd (2.1) | Severs tensor from computational graph; "snip the tape" |
| Manual training step with autograd | INTRODUCED | autograd (2.1) | Forward, backward, update (in no_grad), zero_grad -- complete pattern shown but not practiced in loop |
| PyTorch tensor creation API | DEVELOPED | tensors (2.1) | `torch.tensor()`, `torch.zeros()`, `torch.randn()`, etc. |
| Tensor attributes: shape, dtype, device | DEVELOPED | tensors (2.1) | "Shape, dtype, device -- check these first" debugging trinity |
| Matrix multiplication with `@` operator | DEVELOPED | tensors (2.1) | `y_hat = X @ w + b` |
| Training loop (universal pattern) | DEVELOPED | implementing-linear-regression (1.1) | Forward -> loss -> backward -> update; "heartbeat of training" |
| Parameters as "knobs the model learns" | DEVELOPED | linear-regression (1.1) | The values that get adjusted via gradient descent |
| Weight initialization (general principle) | DEVELOPED | training-dynamics (1.3) | Each layer should preserve signal; Xavier and He initialization |
| Dropout (concept) | DEVELOPED | overfitting-and-regularization (1.3) | Randomly zero neurons during training; `p=0.5` default |
| Batch normalization (concept) | INTRODUCED | training-dynamics (1.3) | Normalize activations between layers; learned gamma/beta |

### Mental Models Already Established

- **"Neuron = multi-input linear regression"** -- the mathematical definition of a neuron
- **"100-layer linear = 1-layer linear"** -- linear collapse, why activation functions exist
- **"Networks TRANSFORM the space"** -- hidden layers move data points to new positions
- **"Parameters are knobs the model learns"** -- the values gradient descent adjusts
- **"requires_grad = press Record"** -- tells PyTorch to track operations on a tensor
- **"backward() = press Rewind"** -- walks the graph backward, storing gradients in `.grad`
- **"zero_grad() = clear the tape"** -- gradients accumulate by default; clear before each step
- **"Not magic -- automation"** -- autograd automates the same algorithm the student did manually
- **"Tensors are NumPy arrays that know where they live"** -- core tensor framing
- **"Same interface, different engine"** -- PyTorch mirrors NumPy; new parts are device + autograd
- **"Training loop = forward -> loss -> backward -> update"** -- universal pattern from Series 1

### What Was Explicitly NOT Covered (Relevant Here)

- `nn.Module`, `nn.Linear`, or any layer abstractions (explicitly deferred from autograd)
- `torch.optim` or optimizer objects (deferred to training-loop)
- Loss function objects like `nn.MSELoss` (deferred to training-loop)
- Matrix form of layers (multi-neuron layers as matrix operations) -- INTRODUCED conceptually in neuron-basics but never implemented
- `nn.Sequential` -- not mentioned anywhere
- `model.parameters()` -- the autograd lesson used individual tensors with `requires_grad=True`
- How PyTorch layers handle initialization internally
- `nn.ReLU` vs `torch.relu` vs `torch.clamp` (autograd used `torch.clamp` for ReLU)

### Readiness Assessment

The student is well-prepared. They have the conceptual foundation from Series 1 (neurons, layers, networks, activation functions, linear collapse) and the PyTorch mechanics from this module (tensors, autograd). The critical bridge: in autograd, they created individual tensors (`w1 = torch.tensor(0.5, requires_grad=True)`, etc.) and manually wired them together. This is exactly the pain point nn.Module solves. They already feel the friction of managing individual parameters -- "imagine doing this for 100 neurons" is the natural motivation. This is a BUILD lesson because the underlying concepts (neurons, layers, parameters, gradients) are all review; the new content is purely the PyTorch API for packaging them.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to define neural networks using `nn.Module`, understanding that layers like `nn.Linear` package the exact same `w*x + b` computation they already know into reusable building blocks with automatic parameter management.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Neuron = weighted sum + bias | DEVELOPED | DEVELOPED | neuron-basics (1.2) | OK | `nn.Linear` IS this computation; student must recognize that `nn.Linear(3, 1)` computes `x @ W.T + b` |
| Layer = group of neurons | INTRODUCED | INTRODUCED | neuron-basics (1.2) | OK | `nn.Linear(in, out)` is a layer of `out` neurons, each taking `in` inputs; student needs conceptual recognition, not matrix implementation |
| Activation function (concept) | DEVELOPED | DEVELOPED | activation-functions (1.2) | OK | `nn.ReLU()` wraps the same `max(0, x)` the student knows; connection must be explicit |
| Linear collapse | INTRODUCED | DEVELOPED | neuron-basics (1.2) | OK | Motivates WHY activations are placed between linear layers in `nn.Sequential`; actual depth exceeds requirement |
| `requires_grad` flag | DEVELOPED | DEVELOPED | autograd (2.1) | OK | `nn.Parameter` automatically sets `requires_grad=True`; student must see this is the same flag, not a new concept |
| `backward()` and `.grad` | DEVELOPED | DEVELOPED | autograd (2.1) | OK | Model parameters are leaf tensors with `.grad`; backward() works identically on nn.Module-based networks |
| `zero_grad()` | INTRODUCED | DEVELOPED | autograd (2.1) | OK | `model.zero_grad()` calls the same operation on all parameters at once; student recognizes the pattern |
| Matrix multiplication with `@` | DEVELOPED | DEVELOPED | tensors (2.1) | OK | `nn.Linear` internally does `x @ weight.T + bias`; student should see this is the same `@` they know |
| Manual training step with autograd | INTRODUCED | INTRODUCED | autograd (2.1) | OK | The autograd lesson previewed the pattern; nn.Module replaces individual tensors with organized structure |
| Parameters as "knobs" | INTRODUCED | DEVELOPED | linear-regression (1.1) | OK | `model.parameters()` returns all learnable knobs; reinforces the metaphor with a concrete API |

**Gap resolution:** No gaps. All prerequisites are at or above required depth. The student has both the conceptual understanding (neurons, layers, activations from Series 1) and the mechanical foundation (tensors, autograd from this module). The new content is the organizational API -- packaging known concepts into reusable structures.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **`nn.Linear` does something different from `x @ W + b`** | The abstraction hides the math. Software engineers know abstractions can do more than the simple description suggests. "Surely a PyTorch layer is more complex than matrix multiplication?" | Create an `nn.Linear(3, 1)` layer, manually extract `.weight` and `.bias`, then compute `x @ layer.weight.T + layer.bias` by hand. Compare to `layer(x)`. Same numbers. The layer IS the matrix multiply the student already knows. | Early, right after introducing `nn.Linear` -- before the student has time to build a mystery around it |
| **`nn.Module` is just a container / dictionary of parameters** | Coming from software engineering, classes that hold data feel like data structures. "It's just a namespace for weights." | Show that `nn.Module` manages the computational graph: calling `model(x)` runs the `forward()` method, which builds the autograd graph. Parameters are automatically tracked, `model.parameters()` iterates them, `model.zero_grad()` clears all their gradients. It's a computation + parameter management system, not just storage. Show a Module with `forward()` that does conditional logic -- the computation can change per input. | After the first `nn.Module` subclass, when explaining `forward()` |
| **You need to call `requires_grad=True` on `nn.Linear` parameters** | They just spent a lesson manually setting `requires_grad=True` on every tensor. The habit is fresh. "I need to mark these as learnable." | Print `layer.weight.requires_grad` and `layer.bias.requires_grad` -- both already `True`. `nn.Parameter` wraps tensors with `requires_grad=True` by default. This is the convenience: nn.Module handles what you did manually. | Immediately after showing `nn.Linear` parameters |
| **`nn.Sequential` is the only way to build models** | Sequential is introduced as a convenience, and students may over-generalize. "Every model is a stack of layers." | Show a model that cannot be expressed as Sequential: a skip connection where the input bypasses a layer and is added to the output. This requires a custom `forward()` method. Sequential is for simple stacks; real architectures need custom Module subclasses. | After introducing Sequential, as the "but what about..." transition |
| **Calling `model(x)` and `model.forward(x)` are the same thing** | They look like they do the same thing -- `model(x)` calls `forward()`. "It's just syntactic sugar." | `model(x)` calls `__call__`, which runs hooks (pre-forward, post-forward) and then calls `forward()`. Calling `forward()` directly skips the hooks. For now hooks are not important, but the convention matters: always use `model(x)`, never `model.forward(x)`. This is a "trust the convention" moment -- the student knows it matters but does not need to understand hooks yet. | Brief aside when first showing `model(x)`, not a full section |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Reproduce the autograd lesson's manual network using nn.Linear** | Positive (primary) | Show that the same 2-layer, 4-parameter network from autograd (and originally from backprop-worked-example) can be expressed with `nn.Linear`. Same forward pass, same `backward()`, same gradients. The parameters are now managed by the Module, not by the student. | Continuity. The student has a personal connection to these specific parameters (w1=0.5, b1=0.1, w2=-0.3, b2=0.2). Seeing the same network expressed three ways (math in 1.3, raw tensors in autograd, nn.Module here) makes the abstraction layers visible and concrete. |
| **A single `nn.Linear(3, 1)` layer** | Positive (introductory) | Simplest possible nn.Linear usage: 3 inputs, 1 output. Show that it is a neuron (multi-input weighted sum + bias). Print `.weight` and `.bias` to show it is literally `w*x + b`. Compare `layer(x)` to manual `x @ layer.weight.T + layer.bias`. | Minimal first example. Removes network complexity. Directly connects to "neuron = weighted sum + bias" from neuron-basics. The student can verify the computation by hand. |
| **Sequential model that collapses without activations** | Negative | Build `nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))` (no activations). Show that it is equivalent to a single linear layer -- the same collapse the student proved in neuron-basics. Then add `nn.ReLU()` between them and show the output changes. | Reinforces linear collapse from Series 1. Demonstrates WHY activation layers appear in Sequential. Prevents the misconception that stacking linear layers adds power. The student has this concept at DEVELOPED depth -- seeing it happen in PyTorch code cements the connection. |
| **Skip connection that cannot be expressed as Sequential** | Negative | Build a custom Module with a skip connection: `out = self.layer(x) + x`. This cannot be written as `nn.Sequential(...)` because the input must bypass the layer. | Defines the boundary of Sequential. Prevents over-reliance on Sequential for all architectures. Previews residual connections (mentioned in training-dynamics as "skip connections / ResNets"). The student sees why custom `forward()` methods exist. |
| **Comparing raw-tensor network vs nn.Module network (side-by-side)** | Positive (stretch) | Side-by-side comparison: the manual 4-tensor approach from autograd vs the nn.Module approach. Same computation, but nn.Module handles `requires_grad`, parameter collection, zero_grad, and device management automatically. | Makes the "packaging" value proposition concrete. The student sees the exact code reduction and organizational benefit. Bridges directly from the previous lesson's approach to this lesson's approach. |

---

## Phase 3: Design

### Narrative Arc

In the autograd lesson, you built a 2-layer network by creating four individual tensors -- `w1`, `b1`, `w2`, `b2` -- each with `requires_grad=True`. You wired them together manually: `z1 = w1 * x + b1`, `a1 = torch.clamp(z1, min=0)`, `y_hat = w2 * a1 + b2`. It worked. But imagine doing this for a network with 100 neurons across 5 layers. That is 1,000 weight values and 100 biases -- 1,100 individual tensors, each needing `requires_grad=True`, each needing `zero_grad()`, each needing manual wiring. The math does not get harder. The bookkeeping becomes impossible. This is exactly the problem `nn.Module` solves. It packages the computation you already understand (weighted sum + bias + activation) into reusable building blocks that automatically manage their parameters. `nn.Linear` is a layer of neurons. `nn.Sequential` is a stack of layers. `model.parameters()` collects every learnable tensor in one call. You are not learning new math. You are learning PyTorch's organizational system for the math you already know.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example with real numbers** | Reproduce the autograd lesson's network using `nn.Linear`, manually set the same weights (w1=0.5, b1=0.1, w2=-0.3, b2=0.2), run forward + backward, verify same gradients appear in `layer.weight.grad` and `layer.bias.grad`. | The student has computed these specific gradients three times now (by hand in 1.3, with raw tensors in autograd). Seeing the same values emerge from an nn.Module-based network proves the abstraction does not change the computation. Personal connection to specific numbers drives the "of course" feeling. |
| **Visual (architecture diagram)** | Show a simple network diagram (input -> Linear -> ReLU -> Linear -> output) with annotations mapping each block to the nn.Module code. Parameter counts labeled on each layer. Color-code to match the code: layer names in the diagram match `self.layer1`, `self.relu`, `self.layer2` in the class definition. | The student has seen network diagrams in Series 1. Mapping a diagram directly to code -- where each visual block has a one-to-one correspondence with a line in `__init__` and a line in `forward()` -- bridges the visual and symbolic modalities. |
| **Symbolic / Code** | Progressive code examples: (1) single `nn.Linear` with manual verification, (2) custom `nn.Module` subclass for 2-layer network, (3) same network as `nn.Sequential`, (4) skip connection that requires custom `forward()`. | This is fundamentally a code lesson -- the student is learning an API. The progression from single layer to custom module to Sequential to beyond-Sequential manages cognitive load while building up the full picture. |
| **Verbal / Analogy** | "nn.Module is like a LEGO brick. Each brick has a specific shape (computation) and connection points (inputs/outputs). You can snap them together in different configurations. nn.Linear is the most basic brick -- it computes `w*x + b`. nn.Sequential is a straight tower of bricks. But for more complex builds, you design your own assembly instructions (the `forward()` method)." | LEGO analogy maps well: individual bricks (layers), compositions (Sequential), custom builds (Module subclass). It captures both the modularity and the compositional nature. Does not conflict with prior analogies. |
| **Intuitive ("of course" feeling)** | After the side-by-side comparison of raw tensors vs nn.Module: "Of course you would package this. You already know the math. Now the code matches how you think about it: layers, not individual weights." | Lands the insight that nn.Module is organizational, not conceptual. The student should feel that this is the natural next step -- not a new concept, but a better way to express concepts they already have. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3
  1. `nn.Module` / `nn.Parameter` -- the packaging system (how to define a model as a class with `__init__` + `forward()`)
  2. `nn.Linear` -- a concrete layer that IS `w*x + b` (the most-used building block)
  3. `nn.Sequential` -- convenience for simple layer stacks (lighter concept, depends on the first two)
- **Previous lesson load:** STRETCH (autograd -- new mechanism, 3 API concepts)
- **This lesson's load:** BUILD -- appropriate after a STRETCH lesson
- **Assessment:** Two core concepts (Module subclass pattern, Linear layer) plus one convenience concept (Sequential). The conceptual load is low because every idea maps directly to something the student already knows: neurons become `nn.Linear`, layers become modules, stacking becomes `Sequential`, `requires_grad` becomes automatic. The novelty is the organizational pattern (Python class with `__init__` and `forward()`), not the math.

### Connections to Prior Concepts

| Prior Concept | How It Connects |
|--------------|-----------------|
| "Neuron = weighted sum + bias" from neuron-basics (1.2) | `nn.Linear(in, out)` is EXACTLY this: `out` neurons, each computing `w*x + b`. The formula did not change. The packaging did. |
| "Linear collapse: 100 layers = 1 layer" from neuron-basics (1.2) | `nn.Sequential(nn.Linear(2,4), nn.Linear(4,1))` without activations collapses. Student proves the same result in PyTorch code. |
| "requires_grad = press Record" from autograd (2.1) | `nn.Parameter` sets `requires_grad=True` automatically. The Recording is still happening -- it is just handled for you. |
| "backward() = press Rewind" from autograd (2.1) | Calling `loss.backward()` still walks the graph. The `.grad` values end up on `model.layer1.weight.grad` instead of on standalone tensors. Same mechanism, better organization. |
| "zero_grad() = clear the tape" from autograd (2.1) | `model.zero_grad()` calls `zero_grad()` on every parameter. Same operation, one call instead of N. |
| Manual training step from autograd (2.1) | The autograd lesson previewed: "forward, backward, update, zero_grad." This lesson upgrades the "forward" step from manual wiring to `model(x)`. Next lesson (training-loop) completes the picture with optimizers. |
| "Parameters are knobs" from linear-regression (1.1) | `model.parameters()` literally returns all the knobs. The metaphor becomes a concrete API call. |
| Weight initialization (He/Xavier) from training-dynamics (1.3) | `nn.Linear` uses Kaiming uniform initialization by default. Briefly note this -- the student already knows WHY initialization matters. |
| Dropout from overfitting-and-regularization (1.3) | `nn.Dropout(p=0.5)` is the code equivalent. Can be placed in Sequential. Brief preview -- not the focus. |

**Potentially misleading prior analogies:** The "LEGO brick" analogy for nn.Module could suggest that modules are static, pre-defined blocks with no flexibility. In reality, `forward()` can contain arbitrary Python logic (conditionals, loops, dynamic graphs). Address this when showing the skip-connection example: "LEGO bricks snap together in fixed ways; nn.Module's `forward()` method lets you define any assembly logic you want."

### Scope Boundaries

**This lesson IS about:**
- `nn.Module` as the base class for all neural network components (how to subclass it)
- `nn.Linear` as a concrete layer that computes `y = x @ W.T + b`
- `nn.Parameter` and automatic `requires_grad=True`
- `model.parameters()` for collecting all learnable tensors
- `nn.Sequential` for simple layer stacks
- `nn.ReLU` as an activation layer (connecting to the concept from Series 1.2)
- The `__init__` + `forward()` pattern for defining models
- `model.zero_grad()` as a convenience over per-parameter zero_grad

**This lesson is NOT about:**
- `torch.optim` or any optimizer objects (lesson 4: training-loop)
- Loss function objects like `nn.MSELoss` (lesson 4: training-loop)
- Full training loops (we show forward + backward to verify gradients, not training)
- Convolutional layers, recurrent layers, or any architecture-specific layers
- `model.train()` vs `model.eval()` mode switching (lesson 4: training-loop, or later)
- Saving/loading models (`torch.save`, `state_dict`)
- GPU placement of models (`.to(device)` on a model -- briefly mentioned, not developed)
- Custom autograd functions
- Hooks (pre-forward, post-forward) -- mentioned only to justify `model(x)` over `model.forward(x)`

**Target depth:**
- `nn.Module` subclass pattern (`__init__` + `forward()`): DEVELOPED -- student can write a custom Module from scratch
- `nn.Linear`: DEVELOPED -- student understands it IS `w*x + b`, can inspect and verify
- `nn.Sequential`: INTRODUCED -- student can use it for simple stacks, knows its limitations
- `model.parameters()`: DEVELOPED -- student can iterate parameters, understands it collects all `nn.Parameter` instances
- `nn.ReLU` as a layer: INTRODUCED -- student recognizes it as the same function, knows it goes between linear layers

### Lesson Outline

#### 1. Context + Constraints
What this lesson is (PyTorch's system for packaging neurons, layers, and networks into reusable components) and what it is NOT (no optimizers, no loss objects, no full training loops -- those are next). Frame: "You already know the math of neurons and layers. You already know how PyTorch computes gradients. Now you learn how to organize all of that into real, reusable models."

#### 2. Hook -- "Imagine 100 Neurons"
Type: Pain-point callback. Show a snippet from the autograd lesson: four individual tensors, each with `requires_grad=True`, manually wired together. Then ask: "This worked for 4 parameters. What happens with a 3-layer network with 64 neurons per layer? That is 64*3 + 64*64 + 64*1 = 4,353 parameters. Are you going to create 4,353 individual tensors?" The student feels the organizational pain immediately. Then: "There is a better way."

#### 3. Explain Part 1 -- nn.Linear (The Building Block)
Start with the simplest possible example: `layer = nn.Linear(3, 1)`. This is one neuron with 3 inputs. Print `layer.weight` and `layer.bias` -- they are tensors. Print `layer.weight.requires_grad` -- it is `True`. "nn.Linear creates the weight and bias tensors for you, with requires_grad already set."

Verify it IS `w*x + b`: create an input `x = torch.tensor([1.0, 2.0, 3.0])`, run `layer(x)`, then manually compute `x @ layer.weight.T + layer.bias`. Same result.

Address misconception: "nn.Linear does not do something mysterious. It IS the matrix multiply you already know. The only difference: it creates and manages the tensors for you."

Extend to multiple neurons: `layer = nn.Linear(3, 4)` creates 4 neurons, each with 3 inputs. Print `layer.weight.shape` -- it is `(4, 3)`. This is the matrix form mentioned in neuron-basics but never implemented.

Briefly note: "nn.Linear initializes weights using Kaiming uniform by default -- a variant of He initialization, which you learned preserves signal magnitude across layers."

#### 4. Explain Part 2 -- nn.Module (The Packaging System)
Show how to build the same 2-layer network from autograd as an `nn.Module` subclass:

```python
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
```

Walk through the pattern: `__init__` defines the layers (the "what"), `forward()` defines the computation (the "how"). Explain: "This is Python -- you are defining a class. The magic is minimal: `super().__init__()` registers the module, and any `nn.Module` or `nn.Parameter` assigned to `self` is automatically tracked."

Show `model.parameters()`: iterate and print shapes. "This is the 'parameters are knobs' idea made concrete. One call collects every learnable tensor in the model."

Show `model.zero_grad()`: "Same zero_grad, but applied to ALL parameters at once. In autograd, you called zero_grad on each tensor individually."

Architecture diagram: input -> Linear(1,1) -> ReLU -> Linear(1,1) -> output, with annotations mapping each block to the code.

#### 5. Check 1 -- Predict and Verify
"You create `layer = nn.Linear(5, 3)`. How many parameters does this layer have?" Student must reason: 3 neurons, each with 5 weights + 1 bias = 18 total. `5*3 + 3 = 18`. Verify with `sum(p.numel() for p in layer.parameters())`. Then: "How many parameters in `nn.Linear(100, 50)`?" Answer: 100*50 + 50 = 5,050.

#### 6. Explain Part 3 -- The Payoff (Reproducing the Autograd Network)
Same emotional callback as autograd: set up the 2-layer network from backprop-worked-example using nn.Module. Manually set the weights to w1=0.5, b1=0.1, w2=-0.3, b2=0.2 (using `with torch.no_grad(): layer.weight.fill_(...)` or `nn.Parameter(torch.tensor(...))`). Run the same input (x=2, y=1). Call `loss.backward()`. Compare gradients:

| Parameter | Raw tensors (autograd) | nn.Module | Match? |
|-----------|----------------------|-----------|--------|
| w1 | (value) | model.layer1.weight.grad | Yes |
| b1 | (value) | model.layer1.bias.grad | Yes |
| w2 | (value) | model.layer2.weight.grad | Yes |
| b2 | (value) | model.layer2.bias.grad | Yes |

"Same computation. Same gradients. The only thing that changed is how the parameters are organized."

Side-by-side comparison: the 4-tensor approach from autograd vs the nn.Module approach. Highlight what nn.Module handles: `requires_grad` (automatic), parameter collection (one call), zero_grad (one call), organization (named attributes).

#### 7. Explain Part 4 -- nn.Sequential (The Shortcut)
For simple layer stacks, show `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
```

"Instead of writing a class, you list the layers in order. PyTorch runs them sequentially."

Introduce `nn.ReLU()` as a module: "Same `max(0, x)` you know from activation-functions. As a module, it can be placed in a Sequential stack."

Negative example: linear collapse. Build `nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))` without activations. Show it is equivalent to one linear layer -- the same collapse from neuron-basics, now in PyTorch code.

Then add `nn.ReLU()` between them. Show the output changes. "Activations between linear layers prevent collapse. This is the same insight from Series 1.2, expressed in code."

#### 8. Check 2 -- Transfer Question
"A colleague builds a model with `nn.Sequential(nn.Linear(10, 50), nn.Linear(50, 50), nn.Linear(50, 1))` and is confused why their deep network performs no better than a simple linear regression. What is the problem, and how do they fix it?"

Student must recognize: no activation functions between linear layers = linear collapse. Fix: add `nn.ReLU()` between each pair of linear layers.

#### 9. Elaborate -- Beyond Sequential (Custom forward())
"What if your architecture is not a simple stack?"

Show a skip connection:

```python
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layer = nn.Linear(size, size)

    def forward(self, x):
        return self.layer(torch.clamp(x, min=0)) + x
```

"The input `x` bypasses the layer and is added to the output. You cannot express this in nn.Sequential because the input needs to skip ahead." This is why `forward()` exists: it gives you full control over the computation graph.

Connect to prior: "Skip connections were mentioned in Training Dynamics as the technique that pushed networks to 152 layers (ResNets). Now you see the code pattern."

Briefly note `model(x)` vs `model.forward(x)`: "Always use `model(x)`. Calling `forward()` directly skips hooks that PyTorch uses internally. For now, just follow the convention."

#### 10. Practice -- Colab Notebook
Exercises:

1. **Create and inspect nn.Linear layers** -- create layers of various sizes, print weight shapes, verify parameter counts (guided)
2. **Verify nn.Linear IS w*x + b** -- manually compute `x @ weight.T + bias` and compare to `layer(x)` (guided)
3. **Build a 2-layer nn.Module subclass** -- define `__init__` and `forward()`, run forward pass, verify output (supported)
4. **Convert to nn.Sequential** -- rewrite the same network using Sequential, verify same output (supported)
5. **Linear collapse experiment** -- build Sequential with and without activations, compare (supported)
6. **(Stretch) Build a skip-connection Module** -- implement the ResidualBlock from the lesson, verify forward pass includes the skip (independent)

#### 11. Summarize -- Mental Model Echo
- `nn.Linear(in, out)` = a layer of `out` neurons, each computing `w*x + b` -- the same neuron from Series 1.2
- `nn.Module` = a class that packages computation (`forward()`) and parameters (`__init__`) into a reusable building block
- `nn.Parameter` = a tensor with `requires_grad=True` -- same flag, automatic management
- `model.parameters()` = collect all learnable knobs in one call
- `nn.Sequential` = snap layers together in a straight line (simple stacks only)
- Custom `forward()` = any computation you want (skip connections, conditionals, anything)

"You defined neurons mathematically in Series 1. You wired them with raw tensors in autograd. Now you package them into models. The math did not change. The organization did."

#### 12. Next Step -- Preview Training Loop
"You have tensors, autograd, and models. The last piece: putting it all together. In the next lesson, you will write a complete training loop -- the same forward -> loss -> backward -> update pattern from Series 1, but now with `nn.Module` models, loss function objects, and `torch.optim` optimizers. The heartbeat of training, expressed in PyTorch."

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
- [x] At least 2 positive examples + 1 negative example (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 identified)
- [x] Cognitive load = 2-3 new concepts (within budget -- Module pattern + Linear + Sequential)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

One critical finding (skip-connection diagram contradicts the code) plus four improvement findings that would meaningfully strengthen the lesson.

### Findings

#### [CRITICAL] — Skip-connection Mermaid diagram contradicts the code

**Location:** "Beyond Sequential" section, Mermaid diagram (~line 812-825)
**Issue:** The Mermaid diagram shows the data flow as: x -> Linear -> ReLU -> Add(+x) -> output. But the actual code on line 800 is `return self.linear(torch.clamp(x, min=0)) + x`, which means the data flows as: x -> ReLU -> Linear -> Add(+x) -> output. The ReLU is applied to x BEFORE the linear layer, not after it. The diagram has the order of Linear and ReLU reversed.
**Student impact:** The student would see code that says `self.linear(torch.clamp(x, min=0))` (ReLU first, then Linear) but the diagram shows Linear first then ReLU. This directly contradicts the code and will confuse any student who tries to trace through both. Since the lesson is explicitly teaching students to read `forward()` methods, a mismatch between code and diagram undermines the core skill being taught.
**Suggested fix:** Either (a) swap the diagram to show x -> ReLU -> Linear -> Add -> output, matching the code, or (b) rewrite the code to `x_out = self.linear(x); return torch.clamp(x_out, min=0) + x` and keep the diagram as-is. Option (a) is simpler and matches the planning document's code exactly. Note: the planning document (Phase 3, section 9) also has the code as `self.layer(torch.clamp(x, min=0)) + x`, so the code is intentional and the diagram is wrong.

---

#### [IMPROVEMENT] — Linear collapse negative example does not actually prove collapse

**Location:** "nn.Sequential: The Shortcut" section, linear collapse code block (~line 660-710)
**Issue:** The code creates two models (with and without activations), copies weights from one to the other, and prints outputs. The student sees two different numbers. The lesson then says "Without activations, two stacked linear layers collapse into one." But the code does not actually prove collapse. It only shows that adding ReLU produces a different output, which any function inserted between layers would do (even adding 1). The student never sees evidence that the no-activation model is mathematically equivalent to a single linear layer. The planned example in the design document says "show it is equivalent to one linear layer" but the implementation does not do this.
**Student impact:** A careful student would think: "OK, the outputs differ. But that does not prove the first model collapsed to a single linear layer. Maybe it just computed something different." The claim is not supported by the evidence shown.
**Suggested fix:** Add a verification step: compute the effective single-layer equivalent (`W_eff = W2 @ W1`, `b_eff = W2 @ b1 + b2`) and show that `x @ W_eff.T + b_eff` produces the same output as the 2-layer model. This makes the collapse concrete and verifiable, matching the formula already shown in the lesson text (`W_2(W_1x + b_1) + b_2 = W_eff x + b_eff`).

---

#### [IMPROVEMENT] — Misconception #2 ("Module is just a container") addressed weakly

**Location:** "nn.Module: The Packaging System" section, aside block (~line 394-403)
**Issue:** The planning document identifies a key misconception: "nn.Module is just a container / dictionary of parameters." The plan calls for showing that Module manages the computational graph, and specifically mentions showing "a Module with forward() that does conditional logic -- the computation can change per input." In the built lesson, this misconception is addressed only in a sidebar InsightBlock that says "nn.Module is more than a dictionary of parameters. It manages the computational graph." This is a claim, not a demonstration. No conditional-logic example is shown. The student has no concrete evidence to distinguish Module from a simple parameter container.
**Student impact:** Software engineers coming from OOP backgrounds will naturally pattern-match nn.Module to "a class that holds data" (i.e., a container). The aside text tells them it is more, but does not show them why. The skip-connection example later partially addresses this (forward() has non-trivial logic), but it comes much later and its purpose is framed around Sequential's limitations, not about Module being more than a container.
**Suggested fix:** Either (a) add a brief inline example showing conditional logic in forward() (e.g., a Module that applies dropout only when a flag is set, or one that uses different layers based on input size), or (b) when introducing the first Module subclass, explicitly highlight that forward() builds a NEW computational graph each time it runs (dynamic graph), which a static container could never do. Option (b) is lighter and may suffice -- one sentence like: "Every call to model(x) builds a fresh computational graph. If you added an if-statement in forward(), each input could follow a different path. This is not a container -- it is a computation factory."

---

#### [IMPROVEMENT] — No `model.zero_grad()` demonstration in code

**Location:** "nn.Module: The Packaging System" section (~line 346-352)
**Issue:** The lesson text says: "model.zero_grad() clears all their gradients at once. In Autograd, you called zero_grad() on each tensor individually." But this is stated as prose with no code example. The student never sees `model.zero_grad()` called in any code block. The Autograd lesson made `zero_grad()` a critical concept (the "#1 beginner bug"). Showing its nn.Module equivalent only in prose, never in runnable code, is a missed opportunity to reinforce and connect.
**Student impact:** The student understands zero_grad conceptually but does not see the nn.Module version in action. When they encounter it in the training-loop lesson, it will feel like a new thing rather than a familiar pattern in new packaging.
**Suggested fix:** In the "reproduce autograd network" code block (section 6, ~line 462-503), add `model.zero_grad()` before the backward pass, or add a brief follow-up code block showing `model.zero_grad()` and printing the gradients after clearing. This keeps it concrete and directly parallels the autograd lesson's emphasis.

---

#### [IMPROVEMENT] — `model(x)` vs `model.forward(x)` convention explained but not demonstrated

**Location:** "nn.Module: The Packaging System" section (~line 355-366)
**Issue:** The lesson explains the `model(x)` vs `model.forward(x)` distinction in prose and says "always use model(x)." However, the planning document lists this as a misconception (#5: "They look like they do the same thing") and recommends addressing it as "a brief aside when first showing model(x)." The built lesson places it well (right after the first model(x) usage), but it is pure prose -- no code demonstrating the difference. The student is told about hooks but never sees what happens (even a simple print). For a convention that the lesson says is important, there is no evidence-based persuasion.
**Student impact:** The student will likely follow the convention out of trust ("the lesson said so") but will not understand why. This is acceptable for now (the plan says "trust the convention" moment), so this is an improvement, not critical. A one-line code comment or a brief code block showing both calls producing the same output with a comment about hooks would strengthen the point.
**Suggested fix:** Add a minimal code block: `# Always use model(x), not model.forward(x)\n# model(x) runs hooks + forward(); model.forward(x) skips hooks\ny_hat = model(x)  # correct`. This is light, reinforces the convention visually, and gives the student something to remember.

---

#### [POLISH] — Dropout preview mentioned in planning but absent from lesson

**Location:** Planning document, "Connections to Prior Concepts" and "Scope Boundaries" sections
**Issue:** The planning document notes that `nn.Dropout(p=0.5)` could be briefly previewed as the code equivalent of the dropout concept from overfitting-and-regularization (1.3). The built lesson does not mention dropout at all. This is not a gap -- it was listed as "brief preview, not the focus" -- but it is a deviation from the plan.
**Student impact:** Minimal. The student does not need dropout here. Noting it as a deviation for documentation purposes.
**Suggested fix:** If easy, add a one-line mention in the nn.Sequential section or an aside: "You will also see nn.Dropout -- the same random neuron zeroing from Overfitting & Regularization, expressed as a layer you can slot into Sequential." If this adds clutter, skip it.

---

#### [POLISH] — Parameter count formula in the hook section uses a non-standard network shape

**Location:** "Imagine 100 Neurons" section (~line 129-133)
**Issue:** The formula shown is `64 x 3 + 64 x 64 + 64 x 1 = 4,353`. This represents a network with layers of sizes [3, 64, 64, 1], where "3 inputs, 64 hidden, 64 hidden, 1 output." But the framing says "3-layer network with 64 neurons per layer." A "3-layer network with 64 neurons per layer" would typically mean 3 hidden layers of 64, not a 3-input, 64, 64, 1-output network. The formula also omits biases (a 3-layer network with the described structure has 64+64+1 = 129 bias parameters, making the true count 4,482). This is a minor inaccuracy in a motivational section -- the exact number does not matter for the hook -- but a careful student might try to verify and get confused.
**Student impact:** Low. The hook's purpose is "imagine lots of parameters" not "compute the exact count." But since the lesson later teaches parameter counting (Check 1), a student who tries to apply the counting formula to the hook example will get a different answer because biases are omitted.
**Suggested fix:** Either (a) include biases: `64*3 + 64 + 64*64 + 64 + 64*1 + 1 = 4,353 + 129 = 4,482`, or (b) simplify the framing: "a network with thousands of parameters" without a specific formula. Option (b) keeps the hook clean without introducing a counting exercise before the concept is taught.

---

#### [POLISH] — "The Recording is on from the start" capitalization

**Location:** nn.Linear section (~line 205)
**Issue:** The sentence "The Recording is on from the start" capitalizes "Recording" as if it is a proper noun. The autograd lesson established the metaphor "requires_grad = press Record" but did not capitalize "recording" as a term of art. This inconsistency is minor but could confuse a student who tries to search for "Recording" as a PyTorch concept.
**Student impact:** Negligible. The meaning is clear from context.
**Suggested fix:** Lowercase: "The recording is on from the start." Or if the capitalization is intentional (treating it as a callback to the metaphor), add quotes: 'The "Recording" is on from the start.'

---

### Review Notes

**What works well:**
- The hook is excellent. Callback to the autograd lesson's specific code, then the "imagine 4,353 tensors" escalation, creates genuine motivation. Problem before solution is well-executed.
- The "same numbers" payoff (section 6) is the strongest moment in the lesson. Reproducing the exact gradients from autograd with nn.Module drives the central insight home with concrete evidence.
- The side-by-side ComparisonRow (raw tensors vs nn.Module) is well-placed and clearly communicates the organizational benefit.
- Connection to prior concepts is strong throughout. Nearly every paragraph references something the student already knows. The lesson genuinely feels like "recognition, not new learning."
- The Mermaid diagram for the basic 2-layer network (section 4) is a good code-to-architecture bridge.
- Scope boundaries are well-maintained. The lesson stays within its stated scope -- no optimizer objects, no loss functions, no full training loops.
- The two comprehension checks are well-designed: Check 1 (parameter counting) tests understanding of nn.Linear internals, Check 2 (transfer question about missing activations) tests application.
- The LEGO analogy is used sparingly and appropriately. It does not overextend.

**Patterns to watch:**
- The lesson is code-heavy (9 code blocks in the main content), which is appropriate for a BUILD lesson teaching an API. But the modality balance tilts toward symbolic/code and verbal, with the visual modality carried only by two Mermaid diagrams. The planning document lists 5 modalities; the lesson delivers 4 (concrete example, visual, symbolic, verbal/analogy, intuitive). This is adequate but the visual modality could be stronger -- the Mermaid diagrams are small and schematic. No interactive widget is present, which is fine for this lesson type but worth noting.
- The lesson runs long. 12 sections with substantial code blocks may exceed comfortable reading time for an ADHD-friendly format. Consider whether any sections could be tightened.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All critical and improvement findings from iteration 1 have been correctly fixed. The lesson is pedagogically sound, well-connected to prior knowledge, and maintains its scope boundaries. Two minor polish items noted below.

### Iteration 1 Fix Verification

| Iteration 1 Finding | Status | Notes |
|---------------------|--------|-------|
| CRITICAL: Skip-connection diagram contradicts code | FIXED | Diagram now correctly shows x -> ReLU -> Linear -> Add(+x) -> output, matching the code `self.linear(torch.clamp(x, min=0)) + x` |
| IMPROVEMENT: Linear collapse does not prove collapse | FIXED | Code now computes W_eff = W2 @ W1, b_eff = W2 @ b1 + b2, and shows `x @ W_eff.T + b_eff` produces identical output to the 2-layer model. Then shows ReLU breaks the collapse. Rigorous and correct. |
| IMPROVEMENT: "Module is just a container" misconception weak | FIXED | New inline code example shows conditional logic in forward() with text "A container could never do this. Each input can follow a different computation path." Directly demonstrates the dynamic computation that distinguishes Module from a container. |
| IMPROVEMENT: No model.zero_grad() in code | FIXED | `model.zero_grad()` now appears in the reproduce-autograd code block (line 526) with comment "clears ALL parameter gradients at once." |
| IMPROVEMENT: model(x) vs model.forward(x) not demonstrated | FIXED | New code block (lines 391-395) shows the convention with comments explaining hooks. Light and appropriate for a "trust the convention" moment. |
| POLISH: Dropout preview absent | SKIPPED (intentional) | Acceptable -- avoids clutter in an already long lesson. |
| POLISH: Parameter count omits biases | SKIPPED (intentional) | Acceptable -- the hook's purpose is motivation, not exact counting. |
| POLISH: "Recording" capitalization | SKIPPED (intentional) | Acceptable -- meaning is clear from context. |

### Findings

#### [POLISH] — Spaced em dashes in string props

**Location:** ConstraintBlock items (lines 88-92), SummaryBlock descriptions (lines 961-981), NextStepBlock description (line 999)
**Issue:** JSX prose correctly uses `&mdash;` (no spaces), but plain string props use ` — ` (space-dash-space). Both render as lesson text visible to the student. The writing style rule requires em dashes with no surrounding spaces.
**Student impact:** Negligible. The content reads fine either way.
**Suggested fix:** Replace ` — ` with `—` (or use unicode em dash without spaces) in ConstraintBlock items, SummaryBlock descriptions, and NextStepBlock description strings. Low priority.

---

#### [POLISH] — "Dynamic graph" term introduced without definition

**Location:** "nn.Module: The Packaging System" section, line 346
**Issue:** The sentence "This is what makes PyTorch a **dynamic graph** framework" introduces a technical term that was explicitly deferred from the autograd lesson (autograd record: "Dynamic vs static graph comparison" listed under "What is NOT covered"). The term is used as a label for the demonstrated behavior (forward() can use conditionals), and the meaning is inferable from context. But a student searching for "dynamic graph framework" would find explanations involving static graph comparison (TensorFlow 1.x vs PyTorch) that go beyond what this lesson covers.
**Student impact:** Low. The demonstration is clear even without the label. A curious student might Google the term and encounter unnecessary complexity.
**Suggested fix:** Either (a) drop the term and say "This is what makes PyTorch flexible" or (b) keep the term but add a brief parenthetical: "This is what makes PyTorch a **dynamic graph** framework (the graph is rebuilt from scratch each time forward() runs, so it can change per input)." Option (b) is slightly better since it grounds the term in the lesson's own demonstration.

---

### Review Notes

**What works well (carries forward from iteration 1):**
- The hook and payoff remain the lesson's strongest moments. The "same numbers" proof is convincing and personal.
- Connection to prior concepts is pervasive and well-executed. Almost every paragraph links back to something the student has seen.
- The 5 planned misconceptions are all addressed with concrete demonstrations, not just assertions.
- Scope boundaries are maintained cleanly throughout.

**Improvements since iteration 1:**
- The linear collapse proof is now rigorous. Computing W_eff and showing equivalence is exactly what was needed.
- The dynamic forward() example is a well-chosen, lightweight way to disprove the "container" misconception. It adds just one code block and two sentences.
- model.zero_grad() in the reproduce-autograd block is natural and well-placed. It does not feel forced.
- The model(x) vs model.forward(x) code block is appropriately minimal.
- The skip-connection diagram is now correct and matches the code exactly.

**No new issues introduced by the fixes.** All five fixes are clean and well-integrated. The lesson is ready to ship.
