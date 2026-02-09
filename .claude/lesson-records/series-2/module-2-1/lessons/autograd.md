# Lesson: autograd (Module 2.1, Lesson 2)

**Type:** STRETCH
**Previous lesson:** tensors (BUILD)
**Next lesson:** nn-module (BUILD)

---

## Phase 1: Student State (Orient)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Computational graph notation | DEVELOPED | computational-graphs (1.3) | Nodes = operations, edges = data flow; forward left-to-right, backward right-to-left |
| Forward pass as graph traversal | DEVELOPED | computational-graphs (1.3) | Push values through nodes, saving intermediate results |
| Backward pass as graph traversal | DEVELOPED | computational-graphs (1.3) | At every node: incoming gradient x local derivative = outgoing gradient |
| Fan-out gradient summation | DEVELOPED | computational-graphs (1.3) | When a value feeds multiple paths, sum the gradients |
| Automatic differentiation (mechanism) | INTRODUCED | computational-graphs (1.3) | "PyTorch builds a computational graph during forward pass, walks it backward for loss.backward()" |
| Multi-layer gradient computation (by hand) | DEVELOPED | backprop-worked-example (1.3) | 4 gradients through 2 layers, all hand-computed with real numbers |
| Weight update with real numbers | DEVELOPED | backprop-worked-example (1.3) | theta_new = theta_old - lr * gradient; verified loss decreases |
| Chain rule for composed functions | DEVELOPED | backpropagation (1.3) | dy/dx = dy/dg * dg/dx; "effects multiply through the chain" |
| PyTorch tensor creation API | DEVELOPED | tensors (2.1) | torch.tensor(), torch.zeros(), torch.randn(), etc. |
| Tensor attributes: shape, dtype, device | DEVELOPED | tensors (2.1) | "Shape, dtype, device — check these first" debugging trinity |
| Matrix multiplication with @ | DEVELOPED | tensors (2.1) | y_hat = X @ w + b |
| `.detach().cpu().numpy()` chain | MENTIONED | tensors (2.1) | Previewed as "the most common pattern"; .detach() explained in next lesson |
| Training loop (universal pattern) | DEVELOPED | implementing-linear-regression (1.1) | Forward -> loss -> backward -> update |
| Gradient descent update rule | DEVELOPED | gradient-descent (1.1) | theta_new = theta_old - alpha * gradient |
| NumPy-PyTorch interop | DEVELOPED | tensors (2.1) | torch.from_numpy() (shared memory), .numpy() (shared memory) |

### Mental Models Already Established

- **"The graph IS the chain rule, drawn instead of written"** — computational graphs are notation for the chain rule
- **"Incoming gradient x local derivative = outgoing gradient"** — the recipe for every backward step
- **"No path = no gradient = doesn't learn"** — graph structure determines which parameters get gradients
- **"Fan-out = sum the gradients"** — when a value feeds multiple paths, total gradient is the sum
- **"Tensors are NumPy arrays that know where they live"** — core framing from tensors lesson
- **"Same interface, different engine"** — PyTorch mirrors NumPy; the new parts are device management and autograd
- **"Training loop = forward -> loss -> backward -> update"** — universal pattern established in Series 1

### What Was Explicitly NOT Covered (Relevant Here)

- Autograd / `requires_grad` (deferred from tensors to this lesson)
- `torch.no_grad()` (deferred from tensors to this lesson)
- `.detach()` was previewed but NOT explained in tensors
- Implementing gradients in PyTorch (all gradient work was NumPy or conceptual)
- Dynamic vs static computational graphs (not mentioned)

### Readiness Assessment

The student is well-prepared. They have the complete conceptual foundation: they understand computational graphs at DEVELOPED depth, they have hand-computed gradients through a 2-layer network, and they know that "PyTorch builds a graph during forward pass and walks it backward" (INTRODUCED in computational-graphs). They also have the tensor API from last lesson. The gap is purely operational: they have never used `requires_grad`, `backward()`, or `.grad` in code. This is a STRETCH lesson because the autograd API is a genuinely new mechanism, but the underlying concepts are solid review.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to use PyTorch's autograd system to compute gradients automatically, understanding that `loss.backward()` performs exactly the same computational graph traversal they did by hand in Series 1.3.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Computational graph (nodes, edges, traversal) | DEVELOPED | DEVELOPED | computational-graphs (1.3) | OK | Student must recognize that autograd builds/traverses the same graph they drew by hand |
| Backward pass as graph traversal | DEVELOPED | DEVELOPED | computational-graphs (1.3) | OK | loss.backward() IS this traversal — student must connect API to concept |
| Hand-computed gradients through a network | DEVELOPED | DEVELOPED | backprop-worked-example (1.3) | OK | The "aha" requires having done it manually first |
| Chain rule | DEVELOPED | DEVELOPED | backpropagation (1.3) | OK | Autograd applies chain rule; student must see it happening |
| PyTorch tensor creation and manipulation | DEVELOPED | DEVELOPED | tensors (2.1) | OK | All autograd operations are on tensors |
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent (1.1) | OK | Manual update with .grad values uses this rule |
| Training loop pattern | DEVELOPED | DEVELOPED | implementing-linear-regression (1.1) | OK | The backward step in the loop IS loss.backward() |
| Autograd as concept (PyTorch builds graph, walks backward) | INTRODUCED | INTRODUCED | computational-graphs (1.3) | OK | Student has the idea; this lesson develops it into working knowledge |
| Fan-out gradient summation | INTRODUCED | DEVELOPED | computational-graphs (1.3) | OK | Needed for understanding gradient accumulation behavior |

**Gap resolution:** No gaps. All prerequisites are at or above required depth. The student has a strong foundation from hand-computing gradients and understanding computational graphs conceptually. This lesson elevates autograd from INTRODUCED to DEVELOPED/APPLIED.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **`backward()` does something different from the manual backprop they learned** | The API is so simple (one line) that it feels like magic — surely it can't be the same tedious 7-step process? | Reproduce the exact 2-layer network from backprop-worked-example in PyTorch. Run backward(). Compare .grad values to the hand-computed gradients from that lesson. Numbers match exactly. Same computation, different executor. | Core explain section — the central "aha" moment |
| **`requires_grad` makes tensors "gradient tensors" — a different kind of tensor** | The flag feels like it changes what the tensor IS, like a different data type. Software engineers think of flags as type discriminators. | Show that a tensor with requires_grad=True still does all the same arithmetic, has the same shape/dtype/device. The only difference: PyTorch records operations on it. Print the tensor — same numbers. The flag doesn't change the data; it tells PyTorch to watch. | Early in the lesson, right after introducing requires_grad |
| **Gradients accumulate because PyTorch is buggy or poorly designed** | Coming from NumPy where variables get overwritten, accumulation feels wrong. "Why wouldn't it just replace the old gradient?" | Show the RNN use case: same parameters used at multiple time steps, gradients from each step should ADD (fan-out from computational-graphs). Accumulation is correct behavior for shared parameters. Then show that for the common non-shared case, you call zero_grad(). | After first backward() call, when showing gradient accumulation |
| **`torch.no_grad()` is just an optimization detail / boilerplate** | It looks like a performance hint, not semantically meaningful. "The code works the same without it, just slower." | Show that without no_grad(), modifying parameters DURING the update step would be recorded by autograd, building a second graph on top of the first. The update step would become part of the next computation. It's not just faster — it's semantically necessary for correct training. | After the core explain, in the elaboration section |
| **`.grad` is recomputed fresh every time you call `backward()`** | Natural assumption from other programming: calling a function returns a fresh result. | Call backward() twice without zero_grad(). Show that .grad doubled. The gradient ACCUMULATED, it wasn't replaced. This is the single most common PyTorch bug for beginners. | Immediately after first successful backward() call — make them predict before revealing |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Reproduce backprop-worked-example in PyTorch** | Positive (primary) | Show that loss.backward() produces the EXACT same gradients the student computed by hand in Series 1.3. Same network (2 layers, 4 params), same input (x=2, y=1), same initial weights. | This IS the lesson's central "aha." The student has a personal connection to those specific numbers — they spent a full lesson computing them. Seeing the same values appear from one line of code is the emotional payoff for all that manual work. |
| **Simple scalar: f(x) = x^2 at x=3** | Positive (introductory) | Minimal first example to show requires_grad, backward(), and .grad before any network complexity. Gradient should be 6.0 — easy to verify mentally. | Simplest possible case. Removes all network complexity. Student can verify the gradient by calculus (df/dx = 2x = 6). Establishes the API pattern before adding conceptual load. |
| **Gradient accumulation trap** | Negative | Show that calling backward() twice without zero_grad() doubles the gradient. Student predicts "6.0 again" but gets "12.0". | This is the #1 beginner bug. Making them predict-then-see creates a lasting memory. Connects to fan-out gradient summation from computational-graphs — accumulation is sometimes correct (RNN), but you must be intentional. |
| **Detached tensor has no gradient** | Negative | Create a computation where one intermediate value is detached. Show that parameters feeding through the detached path get no gradient. | Connects to "no path = no gradient" from computational-graphs. Makes .detach() concrete — it severs the graph. Also fulfills the promise from tensors lesson where .detach() was previewed. |
| **Manual update vs. the autograd way (side-by-side)** | Positive (stretch) | Show the complete manual training step (forward, compute loss, manually compute gradients, update) next to the autograd version. Same result, fraction of the code. | Bridges Series 1 implementation to Series 2. The student literally sees their NumPy training loop become a PyTorch training loop. Previews the training-loop lesson. |

---

## Phase 3: Design

### Narrative Arc

In the backprop-worked-example lesson, you sat down and hand-computed 4 gradients through a 2-layer network. Seven steps of "incoming gradient times local derivative," tracking values at every node. In computational-graphs, you learned that PyTorch builds this exact graph during the forward pass and walks it backward. But you've never actually SEEN it happen. This lesson closes that loop. You will set up the same network with the same weights, call `loss.backward()`, and watch the exact same gradient values appear — the ones you computed by hand — from a single line of code. The point isn't that autograd is magic. The point is that autograd is the thing you already understand, automated. Every concept from Series 1.3 — the computational graph, the backward traversal, "incoming times local," fan-out summation — is running inside `backward()`. You're not learning a new algorithm. You're learning PyTorch's API for the algorithm you already know.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example with real numbers** | Reproduce the backprop-worked-example network (w1=0.5, b1=0.1, w2=-0.3, b2=0.2, x=2, y=1) in PyTorch. Run backward(). Compare .grad values to the hand-computed values from that lesson. | The central "aha" requires SEEING the same numbers. Abstract explanation that "autograd does backprop" doesn't land. The student needs to match specific values they personally computed to specific .grad attributes. |
| **Visual (computational graph diagram)** | Show the computational graph that PyTorch builds internally, annotated with the same forward values (blue) and gradient values (red) from the computational-graphs lesson. Overlay which parts requires_grad controls (which nodes are tracked) and what backward() traverses. | The student already has the mental model of computational graphs with color-coded values. Showing the SAME diagram with PyTorch API labels mapped onto it (requires_grad = "this node is tracked", backward() = "walk right to left") bridges concept to API. |
| **Symbolic / Code** | Progressive code examples: (1) scalar x^2, (2) backprop-worked-example reproduction, (3) gradient accumulation trap, (4) manual vs autograd training step. | This is fundamentally a code lesson — the student is learning an API. Every concept must be grounded in runnable code. The progression from trivial (scalar) to meaningful (full network) manages cognitive load. |
| **Verbal / Analogy** | "requires_grad is like pressing Record on a video camera. The tensor does the same things either way — but when Recording is on, every operation is being tracked. backward() is pressing Rewind and watching the tape in reverse." | Extends the "same interface, different engine" model from tensors. The recording metaphor captures what autograd actually does (records operations on the tape/graph) and what backward does (replays in reverse). |
| **Intuitive (the "of course" feeling)** | After showing the number match: "Of course these are the same — backward() IS walking the computational graph. You already drew this graph. You already computed these values. The only thing that changed is who's doing the arithmetic." | Lands the central insight that autograd doesn't replace understanding — it automates it. Reinforces that the manual work in Series 1.3 was worth doing because now they UNDERSTAND what's inside the black box. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3
  1. `requires_grad` flag and its effect (tells PyTorch to track operations)
  2. `backward()` / `.grad` (triggers backward pass, stores results)
  3. `no_grad()` / `zero_grad()` / `detach()` (controlling the tracking)
- **Previous lesson load:** BUILD (tensors — familiar territory, low stretch)
- **This lesson's load:** STRETCH — appropriate after a BUILD lesson
- **Assessment:** Three new concepts is at the ceiling but manageable because all three are API mechanics for a single underlying concept (autograd) that the student already understands theoretically. The conceptual load is low (review); the novelty is purely in the API surface. The progression scalar -> network -> gotchas manages the introduction carefully.

### Connections to Prior Concepts

| Prior Concept | How It Connects |
|--------------|-----------------|
| Computational graph from 1.3 | `requires_grad` tells PyTorch which nodes to include in the graph. `backward()` walks the graph right-to-left. This IS the same graph they drew. |
| "Incoming gradient x local derivative" from backprop-worked-example | Each node in backward() computes exactly this. The .grad values are the final accumulated results. |
| Fan-out gradient summation from computational-graphs | Gradient accumulation in PyTorch is the code-level manifestation of fan-out summation. zero_grad() resets between training steps because each step is a new graph. |
| "No path = no gradient" from computational-graphs | .detach() severs the path. A detached tensor is a node with no backward edge. Connects the conceptual model to the API call. |
| Training loop from 1.1 | The "backward" step in forward->loss->backward->update IS loss.backward(). The "update" step uses .grad. This lesson fills in the PyTorch implementation of steps 3 and 4. |
| `.detach().cpu().numpy()` from tensors | Now .detach() makes sense: it removes the tensor from the computational graph so NumPy (which has no graph) can use it. |

**Potentially misleading prior analogies:** The "ball rolling downhill" analogy for gradient descent is fine — it describes the optimization. It shouldn't be confused with autograd, which is about COMPUTING the gradient (knowing which direction is downhill), not about the descent itself. Clarify if needed: autograd tells you the slope; gradient descent uses the slope to step.

### Scope Boundaries

**This lesson IS about:**
- What requires_grad does (records operations for gradient computation)
- How backward() works (walks the computational graph, storing gradients in .grad)
- The gradient accumulation default and zero_grad()
- torch.no_grad() for disabling tracking
- .detach() for severing the graph
- Connecting these API calls to the computational graph concepts from Series 1.3

**This lesson is NOT about:**
- nn.Module, nn.Linear, or any layer abstractions (lesson 3: nn-module)
- torch.optim or optimizer objects (lesson 4: training-loop)
- Loss function objects like nn.MSELoss (lesson 4: training-loop)
- Higher-order gradients / torch.autograd.grad()
- Custom autograd functions (torch.autograd.Function)
- JIT compilation or torch.compile
- Dynamic vs static graph comparison (TensorFlow)
- Memory management or gradient checkpointing
- Full training loops (we show one manual update step as preview, not the full pattern)

**Target depth:** Autograd concepts (requires_grad, backward, grad, no_grad, zero_grad, detach) at DEVELOPED. The student can explain what each does and predict behavior in novel situations. Full APPLIED depth comes in lesson 4 (training-loop) when they use autograd inside a real training loop.

### Lesson Outline

#### 1. Context + Constraints
What this lesson is (the PyTorch API for automatic gradient computation) and what it is NOT (no nn.Module, no optimizers, no full training loops yet). Frame: "You already know HOW gradients are computed. Today you learn how to TELL PyTorch to compute them."

#### 2. Hook — "Remember Computing These By Hand?"
Type: Before/after callback. Show the 7-step gradient computation from backprop-worked-example (just the summary — the 4 final gradient values). Then: "What if I told you all seven steps happen in one line?" Don't reveal the code yet — create anticipation.

#### 3. Explain Part 1 — requires_grad (The Record Button)
Introduce requires_grad with the simplest possible example: a scalar tensor x = torch.tensor(3.0, requires_grad=True). Explain: "This tells PyTorch to start recording. Every operation on x is tracked." Recording metaphor: pressing Record on a camera. The tensor does the same math — but now operations are being logged.

Show: y = x ** 2. Print y — it shows `tensor(9., grad_fn=<PowBackward0>)`. The `grad_fn` is the recording. Point out: this is the computational graph being built, live, as you write forward-pass code.

Negative example: a tensor WITHOUT requires_grad. Same computation, no grad_fn. PyTorch didn't record anything — nothing to walk backward through.

#### 4. Explain Part 2 — backward() and .grad (Pressing Rewind)
Call y.backward(). Then print x.grad — it shows 6.0. Verify: dy/dx = 2x = 2(3) = 6. Correct.

Analogy: backward() is pressing Rewind on the recording. PyTorch walks through every recorded operation in reverse, applying the chain rule at each step.

Key point: .grad is WHERE the result is stored. It's an attribute on the leaf tensor (the one with requires_grad=True). Not a return value — an attribute.

#### 5. Check 1 — Predict and Verify
Give the student: x = torch.tensor(2.0, requires_grad=True), y = 3 * x + 1, z = y ** 2. Ask: "What is x.grad after z.backward()?" Student must trace: dz/dy = 2y = 2(7) = 14, dy/dx = 3, so dz/dx = 14 * 3 = 42. Reveal: x.grad = 42.0. This checks they can connect the chain rule to the API.

#### 6. Explain Part 3 — The Payoff (Reproducing Backprop-Worked-Example)
THIS is the lesson's emotional center. Set up the exact same network from backprop-worked-example:
- w1 = 0.5, b1 = 0.1, w2 = -0.3, b2 = 0.2 (all with requires_grad=True)
- x = 2.0, y_true = 1.0
- Forward pass: z1 = w1*x + b1, a1 = relu(z1), y_hat = w2*a1 + b2, loss = (y_true - y_hat)^2

Call loss.backward(). Print all four .grad values. Compare to the hand-computed values from backprop-worked-example side-by-side:

| Parameter | Hand-computed (1.3) | PyTorch .grad | Match? |
|-----------|-------------------|---------------|--------|
| w1 | (value) | (value) | Yes |
| b1 | (value) | (value) | Yes |
| w2 | (value) | (value) | Yes |
| b2 | (value) | (value) | Yes |

"All that manual work — the seven steps, the local derivatives, tracking every intermediate value — happens inside backward(). Same numbers. Same algorithm. One line."

Show the computational graph diagram with PyTorch API labels mapped onto it.

#### 7. Explore — Interactive Widget (Autograd Explorer)
Widget showing a small computational graph (the 2-layer network). Two modes:
- **Manual mode:** Student clicks through backward steps one at a time (like backprop-worked-example), seeing "incoming x local = outgoing" at each node
- **Autograd mode:** Student clicks one "backward()" button and all gradients appear simultaneously

The same graph, the same numbers — just different levels of automation. Student can toggle between modes to see the correspondence. Adjustable parameters let them verify the match holds for different initial weights.

Purpose: Makes the "same algorithm, different executor" insight interactive and verifiable.

#### 8. Elaborate — The Gotchas
Three critical behaviors that trip up beginners:

**Gradient accumulation:** Call backward() twice without zero_grad(). Student predicts x.grad, gets double. Explain: PyTorch ADDS to .grad by default. Connect to fan-out summation — this is correct for shared parameters (RNNs). For the common case, call zero_grad() between steps.

**no_grad():** Show that the parameter update `w = w - lr * w.grad` would be recorded by autograd (it's an operation on a requires_grad tensor). Wrap in `with torch.no_grad():` to prevent this. Not just optimization — semantically necessary.

**detach():** Fulfill the promise from tensors lesson. Show that .detach() creates a tensor sharing the same data but severed from the graph. "No path = no gradient" from computational-graphs, implemented as an API call. Now .detach().cpu().numpy() makes sense: detach from graph, move to CPU, convert to NumPy.

#### 9. Check 2 — Transfer Question
"A colleague writes a training step but forgets zero_grad(). After 10 iterations, they notice the gradients are enormous and the model diverges. They think there's a bug in their loss function. What's actually happening, and how do you fix it?"

Student must connect gradient accumulation to the symptom (growing gradients) and prescribe zero_grad().

#### 10. Practice — Colab Notebook
Guided exercises:
1. Compute gradients for a polynomial function, verify by hand
2. Reproduce a simple forward/backward pass, compare to manual calculation
3. Demonstrate the accumulation trap — predict, run, fix with zero_grad()
4. Write a single manual training step: forward, backward, update with no_grad(), zero_grad(). See loss decrease.
5. (Stretch) Use detach() to stop gradients flowing to part of a network. Verify affected parameters have .grad = None or 0.

Scaffolding level: Guided for exercises 1-3, supported for 4, independent for 5.

#### 11. Summarize — Mental Model Echo
- `requires_grad` = press Record (tell PyTorch to track operations)
- `backward()` = press Rewind (walk the graph backward, applying chain rule)
- `.grad` = the result, stored as a tensor attribute (not a return value)
- `zero_grad()` = clear the tape for the next step (gradients accumulate by default)
- `no_grad()` = pause Recording (needed during parameter updates)
- `detach()` = snip the tape (sever a tensor from the graph)

"You already knew the algorithm. Now you know the API."

#### 12. Next Step — Preview nn.Module
"You can now compute gradients for any computation PyTorch can express. But writing w1, b1, w2, b2 as individual tensors gets tedious. What if we could package neurons, layers, and their parameters into reusable building blocks? That's nn.Module."

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
- [x] Cognitive load = 3 new concepts (at ceiling but justified — all are API surface for one understood concept)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

Critical findings exist. Must fix before this lesson is usable.

### Findings

#### [CRITICAL] — Planned visual modality (computational graph diagram) is entirely missing

**Location:** Section 6 ("The Payoff") and throughout
**Issue:** The planning document explicitly calls for a "Visual (computational graph diagram)" modality: "Show the computational graph that PyTorch builds internally, annotated with the same forward values (blue) and gradient values (red) from the computational-graphs lesson. Overlay which parts requires_grad controls (which nodes are tracked) and what backward() traverses." The built lesson has zero visual representation of the computational graph. The AutogradExplorer widget shows gradient badge values in a horizontal flow of text badges, but this is NOT a computational graph diagram. It is a flat list of forward values and gradient values with chevron arrows between them. There are no nodes, no edges, no spatial graph structure, no color-coded forward/backward annotations.
**Student impact:** The student learned computational graphs with a proper node-and-edge visual in Series 1.3. The central claim of this lesson is "autograd walks the SAME graph." Without showing that graph, the lesson asks the student to take this on faith. The bridge between concept (graph) and API (backward()) lacks its strongest modality. The lesson drops from 5 planned modalities to 4, and the missing one is the visual modality that would most directly connect to the student's existing mental model.
**Suggested fix:** Add an inline SVG or diagram (similar to what computational-graphs used) showing the 2-layer network as a proper computational graph with nodes for each operation (multiply, add, ReLU, MSE), forward values in blue, and gradient values in red. Annotate which nodes have `grad_fn` attributes. This could be a static SVG placed in the "Payoff" section before or alongside the comparison table. Alternatively, upgrade the AutogradExplorer widget to render an actual graph layout rather than a horizontal badge row.

#### [CRITICAL] — `import torch.nn.functional as F` used without introduction

**Location:** Section 6, code block `reproduce_backprop.py`, line 2
**Issue:** The code imports `torch.nn.functional as F` and uses `F.relu(z1)`. The student has never seen `torch.nn.functional`, `F` as an alias, or any `nn` submodule. The lesson's own scope boundaries say "no nn.Module or layer abstractions (that's the next lesson)." While `F.relu` is a function not a layer, the import path `torch.nn.functional` introduces the `nn` namespace without explanation, creating confusion about scope boundaries. The student will see `nn` and wonder if this contradicts the constraint block.
**Student impact:** The student encounters an unexplained import. They will either (a) gloss over it and lose the ability to reproduce the code, (b) wonder what `torch.nn.functional` is and feel that the lesson has an unstated prerequisite, or (c) think the constraint block was misleading. Any of these breaks flow at the lesson's most important moment.
**Suggested fix:** Replace `F.relu(z1)` with a manual ReLU: `a1 = torch.clamp(z1, min=0)` or `a1 = z1 * (z1 > 0).float()` or even a simple `a1 = torch.relu(z1)` (which is in the top-level `torch` namespace and does not require the `nn` import). Alternatively, add a one-line explanation: "We borrow one function from torch.nn.functional for ReLU; we will explore this module properly in the next lesson." But the cleaner fix is to avoid the import entirely.

#### [IMPROVEMENT] — Manual mode in AutogradExplorer does not show "incoming x local = outgoing" breakdown

**Location:** AutogradExplorer widget, manual mode step descriptions
**Issue:** The planning document specifies: "Manual mode: Student clicks through backward steps one at a time, seeing 'incoming x local = outgoing' at each node." The built widget shows step descriptions like "Step 2: Layer 2 gradients. dL/dw2 = dL/dy-hat x a1. dL/db2 = dL/dy-hat x 1." These are formula labels, not the concrete "incoming x local = outgoing" computation with real numbers that the student practiced in backprop-worked-example. The student cannot verify the math at each step because the widget only shows the final gradient values (as badges that appear), not the intermediate multiplication.
**Student impact:** The widget's manual mode becomes a passive "click to reveal" exercise rather than an active "trace the computation" exercise. The student clicks 5 times and sees numbers appear, but doesn't engage with the chain rule at each step. This weakens the "same algorithm" insight because the student cannot verify that each step matches what they did by hand.
**Suggested fix:** For each manual step, show the actual multiplication with concrete numbers. For example, Step 2 should show: "dL/dw2 = dL/dy-hat x a1 = 2.2600 x 1.1000 = 2.4860" (with the sign). Display this as a small equation below the step description, using the current network values. This makes each step verifiable and reinforces the "incoming x local" pattern.

#### [IMPROVEMENT] — Gradient accumulation connection to fan-out is mentioned but not grounded

**Location:** Gotcha 1 section, the paragraph explaining WHY accumulation exists
**Issue:** The lesson says "Remember fan-out gradient summation from Computational Graphs? When a parameter is used at multiple time steps in an RNN, gradients from each step should add." This is a correct connection, but it introduces the RNN concept without any foundation. The student has never seen an RNN, does not know what "time steps" means in this context, and has no concrete picture of "a parameter used at multiple time steps." The fan-out connection is there in name but not in substance.
**Student impact:** The student reads "RNN" and "time steps" and has no referent. The justification for gradient accumulation feels hand-wavy rather than grounded. They may accept it on authority ("I guess that makes sense") but do not genuinely understand WHY accumulation is the default.
**Suggested fix:** Replace the RNN reference with the fan-out example the student already knows from computational-graphs: "Remember fan-out from Computational Graphs? When x feeds into TWO operations, the gradient from each path sums. Gradient accumulation is the same idea. If a parameter appears in multiple computations (like a shared weight used twice), the gradients from each use should add. PyTorch defaults to accumulation because it handles this case correctly. For the common single-use case, you clear with zero_grad()." This grounds the explanation in a concept the student has at DEVELOPED depth.

#### [IMPROVEMENT] — `retain_graph=True` appears without explanation

**Location:** Gotcha 1 section, code block `accumulation_trap.py`, line 4
**Issue:** The code calls `y.backward(retain_graph=True)` on the first backward call. The student has never seen `retain_graph`. The lesson does not explain what it does or why it is needed. Without it, the second `y.backward()` would error because PyTorch destroys the graph after the first backward pass by default. But the student does not know this.
**Student impact:** The student sees an unexplained argument and is left wondering what `retain_graph=True` does. If they try to reproduce the code and forget it, they get an error they do not understand. If they include it without understanding, they are cargo-culting. Either way, this is an untaught concept appearing in a code example.
**Suggested fix:** Add a brief parenthetical or aside: "By default, backward() destroys the graph after running (to free memory). retain_graph=True tells PyTorch to keep the graph so we can call backward() again. In normal training, you do not need this because each forward pass builds a new graph." This is a one-sentence explanation that prevents confusion without expanding scope.

#### [IMPROVEMENT] — Side-by-side comparison (Section 10) uses unexplained comparison labels

**Location:** ComparisonRow component, left side labeled "Manual (NumPy)"
**Issue:** The left column says "Manual (NumPy)" and lists "Manually derive dL/dw and dL/db" and "Implement gradient formulas" as steps. But the student's NumPy experience from implementing-linear-regression was 8+ lessons ago (all of Series 1.1 plus all of Series 1.3 plus tensors). The comparison assumes the student remembers the specific workflow from that lesson. While the training loop pattern is at DEVELOPED depth, the specific NumPy implementation details may be fading.
**Student impact:** The comparison is slightly less impactful because the "before" picture is fuzzy. The student remembers the pattern (forward -> loss -> backward -> update) but may not vividly recall "manually derive dL/dw" as a distinct painful step.
**Suggested fix:** Add one sentence before the ComparisonRow: "In Implementing Linear Regression (Series 1), you had to manually derive the gradient formula for each parameter, then implement that formula in NumPy code. With autograd, step 3 collapses to a single line." This brief recap refreshes the memory before the comparison.

#### [POLISH] — Em dash spacing in InsightBlock title

**Location:** Section 6 aside, InsightBlock title: `Not Magic&mdash;Automation`
**Issue:** The `&mdash;` entity renders correctly as an em dash without spaces, which is the correct style per the writing rules. However, in the rendered HTML, this will display as "Not Magic---Automation" depending on font metrics. This is actually correct. No fix needed upon closer inspection. RETRACTED.

#### [POLISH] — Missing "predict before reading" prompt in Check 1

**Location:** Section 5, Check Your Understanding block
**Issue:** The check says "Before running this code, predict: what is x.grad after z.backward()?" This is good. But then immediately provides a hint with the chain rule steps and the final "14 x 3 = ?" which nearly gives away the answer. The planning document says the student "must trace" the chain rule. The hint reduces the tracing to a single multiplication.
**Student impact:** The hint is generous enough that the student does not need to independently trace the chain rule. They just compute 14 x 3 = 42. The check becomes arithmetic rather than chain-rule application.
**Suggested fix:** Split the hint into two levels. First hint: "Trace the chain rule: what is dz/dy? What is dy/dx?" Second hint (more detailed): the current content. This gives the student a chance to work through it before getting the near-answer.

#### [POLISH] — `w -= lr * w.grad` in no_grad block uses in-place subtraction inconsistently

**Location:** Section 8 (Gotcha 2), code block `no_grad.py`
**Issue:** The "BAD" example uses `w = w - lr * w.grad` (non-in-place) while the "CORRECT" example uses `w -= lr * w.grad` (in-place). This inconsistency might confuse the student into thinking the fix is the in-place operator rather than the `no_grad()` context manager. The actual fix is the `with torch.no_grad():` wrapper, not the choice between `=` and `-=`.
**Student impact:** Minor risk of learning the wrong lesson from the comparison. The student might think "oh, I should use -= instead of = for updates" rather than "I should wrap updates in no_grad()."
**Suggested fix:** Use the same operator in both examples. Either both `w = w - lr * w.grad` or both `w -= lr * w.grad`. The difference should be solely the `no_grad()` wrapper.

#### [POLISH] — Training step code block uses undefined `x` and `y_true`

**Location:** Section 10, code block `manual_training_step.py`, lines 6-7
**Issue:** The code references `x` and `y_true` without defining them. They were defined in the earlier `reproduce_backprop.py` block but not in this one. The code is presented as a standalone block with its own filename. A student trying to run this in isolation would get a NameError.
**Student impact:** Minor. The student would likely understand from context. But if they try to run the code in Colab, it fails without the missing definitions.
**Suggested fix:** Add `x = torch.tensor(2.0)` and `y_true = torch.tensor(1.0)` to the top of the code block, or add a comment: `# Using x and y_true from the previous code block`.

### Review Notes

**What works well:**
- The hook is excellent. Showing the 4 hand-computed gradient values, then asking "what if all seven steps happened in one line?" creates genuine anticipation. The callback to specific numbers the student computed personally is emotionally effective.
- The narrative arc faithfully follows the plan: scalar example -> full network reproduction -> gotchas -> training step preview. The pacing builds well.
- The "Record / Rewind / Snip" analogy is coherent and extends naturally across requires_grad, backward(), and detach(). It is a genuinely useful mental model.
- The constraint block is clear and sets appropriate expectations.
- The summary block perfectly echoes the analogy framework established throughout the lesson.
- The exercise panel in the Colab section is well-scaffolded with progressive difficulty.
- Both comprehension checks (predict-and-verify + transfer question) are well-designed and test different levels of understanding.

**Systemic issue:**
The AutogradExplorer widget is functional but underwhelming compared to widgets in other lessons (e.g., the ComputationalGraphExplorer in 1.3 which had graph view vs step-by-step toggle with node detail). The widget renders as a badge dashboard rather than an interactive graph. Given that this lesson's entire thesis is "autograd walks the same computational graph," the widget should probably SHOW that graph. This connects to the Critical finding about the missing visual modality. Upgrading the widget would address both findings simultaneously.

**Pattern to watch:**
The lesson introduces `torch.nn.functional` and `retain_graph` without explanation. Both are minor in isolation, but together they suggest a builder tendency to reach for "what works in real PyTorch" without filtering through "what the student knows." Every import, every argument, every API surface in a code block is something the student sees. If it is not explained, it is an untaught concept.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

No critical findings. One improvement finding introduced by the diagram fix needs to be addressed before this lesson is finalized.

### Iteration 1 Fix Verification

All 9 findings from iteration 1 have been addressed:

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | CRITICAL | Missing computational graph diagram | **FIXED** — MermaidDiagram added in section 6 with proper node-and-edge graph, parameter nodes in purple, forward values, gradient annotations in red |
| 2 | CRITICAL | `import torch.nn.functional as F` used without introduction | **FIXED** — Replaced `F.relu(z1)` with `torch.clamp(z1, min=0)`, removing the unexplained import entirely |
| 3 | IMPROVEMENT | Manual mode doesn't show "incoming x local = outgoing" with real numbers | **FIXED** — Widget step descriptions now show concrete numbers (e.g., `dL/dw2 = dL/dy-hat x a1 = 2.2600 x 1.1000 = -2.4860`) |
| 4 | IMPROVEMENT | Gradient accumulation fan-out connection references RNNs | **FIXED** — Replaced RNN reference with fan-out example the student already knows. Now says "When a value feeds into two operations, the gradient from each path sums." |
| 5 | IMPROVEMENT | `retain_graph=True` appears without explanation | **FIXED** — Added inline comment: "retain_graph=True keeps the graph so we can call backward() again. Normally backward() destroys the graph after running (to free memory). In real training you don't need this — each forward pass builds a new graph." |
| 6 | IMPROVEMENT | Side-by-side comparison lacks context | **FIXED** — Added introductory paragraph: "In Implementing Linear Regression, you had to manually derive the gradient formula for each parameter, then implement that formula in NumPy code. With autograd, that entire step collapses to a single line." |
| 7 | POLISH (RETRACTED) | Em dash spacing | Was retracted during iteration 1 — no fix needed |
| 8 | POLISH | Check 1 hint too generous | **NOT FIXED** — Hint still shows `14 x 3 = ?`. Low severity, does not affect verdict. |
| 9 | POLISH | `w -= lr * w.grad` inconsistency in no_grad examples | **FIXED** — Both BAD and CORRECT examples now use `w = w - lr * w.grad` |
| 10 | POLISH | Training step code block missing variable definitions | **FIXED** — Code block now includes `x = torch.tensor(2.0)` and `y_true = torch.tensor(1.0)` |

### Findings

#### [IMPROVEMENT] — Paragraph after Mermaid diagram incorrectly attributes grad_fn to parameter nodes

**Location:** Section 6 ("The Payoff"), paragraph immediately after the MermaidDiagram (line ~422)
**Issue:** The text reads: "Parameter nodes (purple) have `requires_grad=True` — these are the nodes PyTorch is tracking. Each has a `grad_fn` connecting it to the operations in the graph." In PyTorch, leaf tensors (parameters created with `requires_grad=True`) have `grad_fn=None`. It is the result/intermediate tensors (like `y = x ** 2`) that have `grad_fn` attributes (like `PowBackward0`). The lesson itself correctly teaches this distinction earlier: when printing `y`, it shows `grad_fn=<PowBackward0>` — the grad_fn is on the result, not on x. But this paragraph reverses it by saying parameter nodes "each has a grad_fn."
**Student impact:** The student already learned (correctly) from the requires_grad section that `grad_fn` appears on the *output* of operations, not on the input parameters. Reading this paragraph, they would either (a) form a contradictory mental model ("wait, do parameters have grad_fn?"), or (b) accept the new statement and overwrite the correct understanding. Either outcome weakens the lesson's clarity at its most important moment.
**Suggested fix:** Rewrite to: "Parameter nodes (purple) have `requires_grad=True` — these are the leaf tensors PyTorch is tracking. Each operation node has a `grad_fn` recording what computation produced it. This is exactly the graph from Computational Graphs, with PyTorch API labels mapped on." This correctly attributes grad_fn to operation nodes (which DO have grad_fn) rather than to parameter leaf nodes (which do NOT).

#### [POLISH] — Check 1 hint still gives away the answer (unfixed from iteration 1)

**Location:** Section 5, Check Your Understanding, hint text (line ~270)
**Issue:** The hint shows `dz/dy = 2y = 14`, `dy/dx = 3`, so `dz/dx = 14 x 3 = ?`. This reduces the chain rule exercise to a single multiplication. The planning document says the student "must trace" the chain rule, but the hint does the tracing for them.
**Student impact:** Minor. The student computes 14 x 3 = 42 instead of independently identifying the intermediate derivatives. The check tests arithmetic rather than chain-rule application.
**Suggested fix:** Either (a) remove the numerical evaluation from the hint, leaving only the symbolic form: "Hint: trace the chain rule. What is dz/dy? What is dy/dx? Multiply them." Or (b) put the current detailed hint inside its own `<details>` element as a second-level hint, with a first hint that only says "Trace the chain rule backward: z depends on y, y depends on x."

### Review Notes

**What was fixed well:**
- The MermaidDiagram addition is the standout fix. It provides a proper node-and-edge computational graph with color-coded parameter nodes (purple), operation nodes (slate), and gradient annotations (red). The graph layout matches the left-to-right flow convention established in computational-graphs (Series 1.3), making the visual bridge explicit.
- The `torch.clamp(z1, min=0)` replacement is clean — it avoids the `nn` namespace entirely while being a standard PyTorch function the student can understand from context.
- The widget step descriptions now show concrete "incoming x local = outgoing" computations with real numbers, making manual mode genuinely interactive rather than a passive reveal.
- The fan-out connection in Gotcha 1 is now grounded in concepts the student has at DEVELOPED depth, eliminating the unexplained RNN reference.
- The `retain_graph=True` comment is appropriately brief — it explains what the argument does without expanding scope.

**New issue introduced by fixes:**
The only new issue is the grad_fn attribution error in the paragraph after the Mermaid diagram. This was introduced when the diagram and its surrounding prose were added to fix CRITICAL #1. It is an IMPROVEMENT level finding because the incorrect statement contradicts the correct teaching earlier in the lesson.

**Overall assessment:**
The lesson is in strong shape after the iteration 1 fixes. The two critical findings are fully resolved. The four improvement findings are fully resolved. The remaining improvement finding (grad_fn attribution) is a localized text error that requires a single sentence rewrite. After that fix, the lesson should be ready for a final review pass.

---

## Review — 2026-02-09 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

No critical or improvement findings. One polish item carried over from iteration 2 (Check 1 hint). Per the relaxation rule for iteration 3, the lesson passes with notes.

### Iteration 2 Fix Verification

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | IMPROVEMENT | Paragraph after Mermaid diagram incorrectly attributes grad_fn to parameter nodes | **FIXED** — Text now reads: "Parameter nodes (purple) have `requires_grad=True` — these are the leaf tensors PyTorch accumulates gradients into. Each operation node (slate) has a `grad_fn` recording how it was computed." This correctly attributes grad_fn to operation nodes and requires_grad/grad storage to leaf parameter nodes. |
| 2 | POLISH | Check 1 hint still gives away the answer | **NOT FIXED** — Carried forward. See findings below. |

### Findings

#### [POLISH] — Check 1 hint reduces chain rule exercise to single multiplication (carried from iteration 1)

**Location:** Section 5, Check Your Understanding, hint text (line ~270)
**Issue:** The hint shows `dz/dy = 2y = 14`, `dy/dx = 3`, so `dz/dx = 14 x 3 = ?`. This does the chain rule tracing for the student, reducing the exercise to computing 14 x 3.
**Student impact:** Minor. The student computes a multiplication instead of independently identifying the intermediate derivatives. The check tests arithmetic rather than chain-rule application.
**Suggested fix:** Put the current detailed hint inside a nested `<details>` as a second-level hint, with a first hint that only says "Trace the chain rule backward: what is dz/dy? What is dy/dx?"

### Review Notes

**Iteration 2 fix landed correctly:**
The grad_fn attribution fix (the sole IMPROVEMENT finding from iteration 2) is accurate. Lines 421-427 now correctly distinguish between parameter leaf nodes (which have `requires_grad=True` and accumulate gradients in `.grad`) and operation nodes (which have `grad_fn` attributes recording their computation). This matches the correct teaching earlier in the lesson (section 3) where `grad_fn=<PowBackward0>` is shown on the result `y`, not on the input `x`.

**Full lesson assessment:**
The lesson is pedagogically sound. All 9 findings from iteration 1 have been resolved (2 critical, 4 improvement, 3 polish — with one polish retracted and one polish deliberately left unfixed). The 1 improvement finding from iteration 2 has been resolved. The only remaining item is a polish-level hint that is slightly too generous in Check 1. The lesson can ship as-is.

**Strengths worth noting:**
- The hook is genuinely compelling — showing the exact gradient values the student hand-computed and asking "what if this was one line?" creates real anticipation
- The "Record / Rewind / Snip" analogy framework is coherent across all six API concepts and echoes cleanly in the summary
- The Mermaid diagram bridges the visual modality gap effectively — proper graph structure with color-coded nodes matching the Series 1.3 conventions
- The AutogradExplorer widget's manual mode now shows concrete "incoming x local = outgoing" computations, making it actively instructive rather than a passive reveal
- The gradient accumulation explanation grounds the design decision in fan-out (a concept at DEVELOPED depth) rather than RNNs (unknown to the student)
- The three gotchas are well-sequenced: accumulation (most common bug), no_grad (semantic necessity), detach (completes the tensors lesson promise)
- The training step preview correctly places zero_grad() after the no_grad block, showing the complete and correct pattern

**Ready for Phase 5 (Record).**
