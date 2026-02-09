# Lesson: The Neuron — Building Block of Neural Networks

**Module:** 1.2 — From Linear to Neural
**Slug:** neuron-basics
**Position:** Lesson 1 of 4
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Linear model (y-hat = wx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| MSE loss function | DEVELOPED | loss-functions |
| Gradient descent | DEVELOPED | gradient-descent |
| Learning rate | DEVELOPED | learning-rate |
| Training loop (forward → loss → backward → update) | DEVELOPED | implementing-linear-regression |
| Implementation from scratch (Python/NumPy) | APPLIED | implementing-linear-regression |

Mental models: "Training loop is forward → loss → backward → update." "Parameters are learned via gradient descent." "Every neural network follows this pattern." Student has implemented linear regression end-to-end. Ready to extend to neural network architecture.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student that a neuron is just multi-input linear regression (weighted sum + bias), and that networks are layers of neurons stacked together — but that linear layers collapse to a single linear transformation.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Linear model y-hat = wx + b | DEVELOPED | DEVELOPED | linear-regression | OK | Core connection: "this IS a neuron" |
| Parameters = learnable values | DEVELOPED | DEVELOPED | linear-regression | OK | Weights and biases extend directly |
| Matrix multiplication | INTRODUCED | APPLIED | Prior education | OK | Software engineer — mentioned but not required for understanding |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Neurons are mysterious / brain-like | Pop culture, name implies biology | "It's just weighted sum + bias — the same thing as linear regression!" | Section 2: "It's Just Linear Regression!" |
| More layers = more powerful (always) | Intuition that stacking = strength | W2(W1x + b1) + b2 collapses to Wx + b — 100 linear layers = 1 linear layer | Section 6: "The Problem" |
| "Deep" means something magical | Buzzword | "Deep" just means many hidden layers — doesn't help without nonlinearity | Section 5: Stacking Layers |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| Linear regression vs neuron side-by-side | Positive | Shows they're the same computation |
| SingleNeuronExplorer widget | Interactive positive | Manipulate weights/bias, see output |
| 3-neuron layer with formulas | Positive | Shows how multiple neurons form a layer |
| Network architecture diagram (text-based) | Positive | Input → Layer 1 → Layer 2 → Output |
| NetworkDiagramWidget | Interactive positive | Watch data flow through layers |
| Linear collapse proof (W2 * W1 = W) | Negative | Why stacking linear layers doesn't help |

---

## Phase 3: Design

### Narrative Arc

This lesson makes the crucial bridge from linear regression to neural networks by revealing that a neuron IS linear regression with multiple inputs. The "aha moment" is: "You already know this." From there, it builds up layers (multiple neurons) and networks (stacked layers), then delivers the punchline: stacking linear operations is still just linear. This sets up the cliffhanger for the next lesson (XOR problem).

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Symbolic | Neuron formula y = w1x1 + w2x2 + ... + b | Precise mathematical definition |
| Visual | Side-by-side comparison cards (linear regression vs neuron) | Direct visual mapping |
| Interactive | SingleNeuronExplorer | Manipulate weights and see output change |
| Interactive | NetworkDiagramWidget | Watch data flow through layers |
| Symbolic | Linear collapse proof | Mathematical proof that stacking doesn't help |
| Verbal | "It's just multi-input linear regression!" | The key insight in words |

### Cognitive Load

- **New concepts:** 3 (neuron = weighted sum, layer = group of neurons, network = stacked layers + linear collapse)
- **Previous lesson load:** CONSOLIDATE (implementing-linear-regression)
- **This lesson:** STRETCH — new architectural concepts, but grounded in familiar math
- **Assessment:** 3 concepts is acceptable because "neuron" is almost a relabeling of linear regression, reducing true novelty

### Scope Boundaries

- No activation functions yet — explicitly deferred
- No training of neural networks — just architecture
- No backpropagation through multiple layers
- Matrix notation mentioned in TipBlock but NOT required
- "Hidden layers learn internal representations" is stated but not explained
- The linear collapse IS the cliffhanger — doesn't resolve it

---

## What Was Actually Built

1. **Header + Objective** — "a neuron is something you already know"
2. **What is a Neuron?** — Formula: y = w1x1 + w2x2 + ... + b
3. **"It's Just Linear Regression!"** — Side-by-side comparison cards
4. **SingleNeuronExplorer** — Interactive: set weights, see output
5. **Multiple Neurons = A Layer** — 3 neuron formulas with color coding
6. **Stacking Layers = A Network** — Text diagram: Input → Layer 1 → Layer 2 → Output
7. **NetworkDiagramWidget** — Interactive: watch data flow
8. **"But There's a Problem..."** — Linear collapse proof with KaTeX
9. **What We Have So Far** — Summary of architecture terms
10. **Summary** — 4 key takeaways
11. **Next Step** — Link to limits-of-linearity

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (2), TipBlock (2), TryThisBlock (2), ConceptBlock (2)
- SummaryBlock, NextStepBlock
- SingleNeuronExplorer via ExercisePanel
- NetworkDiagramWidget via ExercisePanel
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Neuron = weighted sum + bias** — multi-input linear regression
- **Layer = group of neurons sharing the same inputs** — each computes a different weighted combination
- **Network = stacked layers** — output of one layer becomes input to the next
- **"Hidden layers"** — layers between input and output, learn internal representations
- **"Deep" = many hidden layers** — the "deep" in deep learning
- **Linear collapse** — stacking linear layers is still just one linear transformation

### Analogies Used
- "A neuron is just multi-input linear regression"
- "100-layer linear network = 1-layer linear network"
- "The structure without the magic" — architecture is in place but needs activation functions

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 2
- Polish: 1

### Verdict: MAJOR REVISION

### Findings

### [CRITICAL] — NetworkDiagramWidget uses ReLU activation in a pre-activation-functions lesson

**Location:** NetworkDiagramWidget.tsx, lines 82-86 (hidden layer computation)
**Issue:** The widget computes hidden values using `Math.max(0, ...)` which is ReLU activation. The lesson explicitly teaches that we haven't covered activation functions yet, and the entire final section proves that "stacking linear layers is still linear." But the widget's hidden layer behaves nonlinearly — a student changing inputs from positive to zero would see hidden neurons "clamp" at 0, which contradicts the lesson's core message.
**Student impact:** If the student experiments carefully with the NetworkDiagramWidget (as the TryThisBlock encourages), they may notice hidden neurons hitting exactly 0 even with positive inputs. This contradicts the lesson's claim that everything is just linear. At best, confusing. At worst, the student forms incorrect mental models about what "linear layers" do.
**Suggested fix:** Remove `Math.max(0, ...)` from the hidden layer computation. The widget should use purely linear computations: `hiddenValues = [inputs[0] * 0.5 + inputs[1] * 0.3 + 0.1, ...]` without the ReLU wrapper. This makes the widget consistent with the lesson's scope (no activation functions yet).

### [IMPROVEMENT] — Misconception "neurons are mysterious/brain-like" is not explicitly addressed

**Location:** Entire lesson — the misconception is listed in the planning document but never directly named
**Issue:** The plan says the lesson addresses the misconception that "neurons are mysterious / brain-like" in Section 2. The lesson DOES show that a neuron is just weighted sum + bias, which implicitly demystifies it. But it never explicitly says "Despite the name, artificial neurons have almost nothing in common with biological neurons" or directly confronts the brain analogy. A student coming in with the pop-culture mental model of neurons might not realize their prior conception was wrong — they might just add new info alongside their old model.
**Student impact:** The student might think "oh, so a neuron does weighted sum + bias AND is like a brain cell." The misconception isn't dismantled, just overshadowed.
**Suggested fix:** Add 1-2 sentences early in "What is a Neuron?" explicitly noting that despite the name, artificial neurons are nothing like biological neurons — they're just a math formula. This directly confronts the misconception.

### [IMPROVEMENT] — NetworkDiagramWidget hides weights, reducing pedagogical value

**Location:** NetworkDiagramWidget, the "What's Happening" panel says "Each connection has a weight (not shown)"
**Issue:** The SingleNeuronExplorer shows weights prominently and lets students manipulate them. The NetworkDiagramWidget hides weights entirely. This creates a disconnect — the student just learned that neurons are all about weights, but the network diagram doesn't show them. The widget becomes more of a "black box" demo than an educational tool. The student can change inputs but can't see WHY the hidden values change the way they do.
**Student impact:** Reduced learning. The student sees values propagate but can't connect them to the weighted-sum concept they just learned. The widget demonstrates "something happens" but not "here's how it happens."
**Suggested fix:** Show weight values on the connection lines (even if not adjustable). This lets students mentally trace the computation and connects the network widget to the single-neuron widget. Alternatively, show the computation formula for at least one hidden neuron.

### [POLISH] — TipBlock "Matrix Form" may be premature for some students

**Location:** "Multiple Neurons = A Layer" section aside
**Issue:** The TipBlock says "This is just matrix multiplication! A layer with 3 neurons taking 2 inputs is a 3×2 weight matrix times the input vector, plus a bias vector." While the planning doc says matrix notation is mentioned but not required, this aside might intimidate students who don't remember matrix multiplication well, even though the plan says the user is a software engineer.
**Student impact:** Minimal — it's in an aside, not the main text. Students who don't know matrices can skip it. But some might feel "I should know this" anxiety.
**Suggested fix:** No change required, or optionally soften with "If you know matrix multiplication..." framing.

### Review Notes
The lesson's overall structure and narrative arc are strong. The "you already know this" framing is excellent for reducing activation energy. The progression from single neuron → layer → network → collapse is logical and well-paced. The SingleNeuronExplorer is a standout interactive element.

The critical finding (ReLU in the network widget) is the main issue — it actively contradicts the lesson's core teaching. This must be fixed before the lesson is reliable. The other findings are about strengthening what's already good.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 1

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — NetworkDiagramWidget biases are invisible, breaking mental computation

**Location:** NetworkDiagramWidget.tsx — biases exist in code (lines 83-86: 0.1, 0.2, 0.3) but are not rendered in the diagram
**Issue:** The widget now shows weight values on connections (good!), which implicitly invites the student to verify computations. But biases are hidden. A student trying to verify hidden neuron 1: `0.5 × 0.8 + 0.3 × 0.3 = 0.49`, but the widget shows 0.59 (because of the +0.1 bias). The student either concludes they did the math wrong (discouraging) or that the widget is broken (loss of trust). The SingleNeuronExplorer handles this well by showing a bias arrow — the NetworkDiagramWidget should too.
**Student impact:** Frustration or confusion when trying to mentally verify the computation. The invisible bias makes the "weighted sum + bias" formula seem incomplete when applied to the network widget.
**Suggested fix:** Add small bias labels near each neuron (e.g., "+0.1" below or beside each neuron), similar to how SingleNeuronExplorer shows bias with a dedicated arrow.

### [IMPROVEMENT] — TryThisBlock for NetworkDiagramWidget doesn't leverage visible weights

**Location:** Lesson file, lines 260-265, the TryThisBlock next to the NetworkDiagramWidget
**Issue:** The TryThisBlock says "Change the inputs and watch values propagate" and "Each hidden neuron computes something different." Now that weights are visible on the diagram, the guided experiments should ask the student to actually verify a computation: "Pick one hidden neuron. Multiply each input by the weight on its connection, add the bias. Does it match the value shown?" This turns passive observation into active verification — a much stronger learning activity.
**Student impact:** Without the verification prompt, the student watches passively. With it, they practice the core computation themselves and build confidence.
**Suggested fix:** Update TryThisBlock to include a computation verification experiment, e.g.: "Pick one hidden neuron. Multiply each input by the weight on its connection, add them up, and add the bias. Does your answer match?"

### [POLISH] — Weight labels may be visually cluttered on small screens

**Location:** NetworkDiagramWidget, weight Text elements on connections
**Issue:** There are 12 weight labels on 12 connections (6 input→hidden + 6 hidden→output). On a small canvas width, these labels could overlap each other or be hard to read at fontSize 9. The offset logic (`from.y < to.y ? -8 : 8`) helps but doesn't handle all overlap cases.
**Student impact:** Minor — labels may be hard to read on small screens but the widget is inside ExercisePanel which has a fullscreen button.
**Suggested fix:** Could be left as-is since ExercisePanel fullscreen handles it. Optionally could add a toggle to show/hide weight labels, but this is gold-plating.

### Review Notes
All three iteration 1 findings were addressed:
- CRITICAL (ReLU): Fixed — widget now uses purely linear computation.
- IMPROVEMENT (brain misconception): Fixed — explicit statement about artificial vs biological neurons added.
- IMPROVEMENT (hidden weights): Fixed — weight values now shown on connections.

The lesson is significantly better. The remaining findings are about completing the improvements (biases, guided experiments) rather than fixing broken things. No critical issues remain.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

### Findings

### [POLISH] — Weight labels may overlap on narrow viewports

**Location:** NetworkDiagramWidget, weight Text elements on connections
**Issue:** With 12 weight labels rendered at fontSize 9, some labels may overlap on narrow screen widths (< ~400px). The offset logic helps but doesn't prevent all overlap cases. The ExercisePanel fullscreen button mitigates this since students can expand the widget.
**Student impact:** Minimal. Labels are readable at normal widths and in fullscreen mode.
**Suggested fix:** Leave as-is. ExercisePanel fullscreen handles the edge case.

### Review Notes
All findings from iterations 1 and 2 have been addressed:
- CRITICAL (ReLU in widget): Fixed — purely linear computation.
- IMPROVEMENT (brain misconception): Fixed — explicit "almost nothing in common with biological brain cells" statement.
- IMPROVEMENT (hidden weights): Fixed — orange weight labels on all connections.
- IMPROVEMENT (invisible biases): Fixed — green bias labels below hidden and output neurons.
- IMPROVEMENT (passive TryThisBlock): Fixed — "Verify It" now asks students to manually compute a neuron's output.

The lesson now:
1. Explicitly debunks the brain analogy misconception
2. Makes the "neuron = linear regression" connection via side-by-side comparison and interactive widget
3. Builds from single neuron → layer → network progressively
4. Shows a fully transparent network diagram where students can trace every computation (weights, biases, values)
5. Delivers the linear collapse proof as a compelling punchline
6. Guides students through active verification rather than passive observation
7. Uses purely linear computation in all widgets, consistent with lesson scope

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS

### Changes made:
- [Iteration 1] Fixed CRITICAL: Removed ReLU activation from NetworkDiagramWidget hidden layer computation (was `Math.max(0, ...)`, now purely linear). Added explicit "artificial neurons have nothing in common with biological brain cells" text to debunk brain misconception. Added weight labels on all connection lines in NetworkDiagramWidget.
- [Iteration 2] Fixed IMPROVEMENT: Added green bias labels below hidden and output neurons in NetworkDiagramWidget. Updated TryThisBlock to encourage active computation verification ("Verify It" with step-by-step instructions).

### Remaining items:
- [POLISH] Weight labels may overlap on very narrow viewports — mitigated by ExercisePanel fullscreen button.
