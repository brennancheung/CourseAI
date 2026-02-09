# Lesson: Backpropagation — How Networks Learn

**Module:** 1.3 — Training Neural Networks
**Slug:** backpropagation
**Position:** Lesson 1 of 1 (currently; module needs expansion per curriculum plan)
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Linear model (y-hat = wx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| MSE loss function | DEVELOPED | loss-functions |
| Gradient = slope / steepest ascent | DEVELOPED | gradient-descent |
| Gradient descent update rule | DEVELOPED | gradient-descent |
| Learning rate | DEVELOPED | learning-rate |
| Training loop (forward → loss → backward → update) | DEVELOPED | implementing-linear-regression |
| Gradient formulas for linear regression (dL/dw, dL/db) | DEVELOPED | implementing-linear-regression |
| Chain rule (mentioned in implementing-linear-regression) | MENTIONED | implementing-linear-regression |
| Neuron = σ(w·x + b) | DEVELOPED | neuron-basics, activation-functions |
| Layers, networks, hidden layers | INTRODUCED | neuron-basics |
| Linear collapse (stacking linear = linear) | INTRODUCED | neuron-basics |
| Activation functions (ReLU, sigmoid) | DEVELOPED | activation-functions |
| XOR solved via space transformation | DEVELOPED | activation-functions |
| Vanishing gradients | MENTIONED | activation-functions |
| Differentiability matters for activation | MENTIONED | activation-functions |

Mental models: "Training = forward → loss → backward → update." Student has computed gradients for a 2-parameter linear model. Now they need to scale this to multi-layer networks with many parameters. The key gap: they've only done gradient computation for linear regression, not for composed functions through layers.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student how backpropagation uses the chain rule to efficiently compute gradients for every parameter in a neural network by propagating error signals backward through layers.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent | OK | Backprop computes the gradients that GD uses |
| Training loop structure | DEVELOPED | DEVELOPED | implementing-linear-regression | OK | Backprop is the "backward" step |
| Gradient formulas for 2 parameters | DEVELOPED | DEVELOPED | implementing-linear-regression | OK | Extends from 2 params to many |
| Neural network layers/architecture | INTRODUCED | INTRODUCED | neuron-basics | OK | Need to understand the structure gradients flow through |
| Activation functions (ReLU) | DEVELOPED | DEVELOPED | activation-functions | OK | Need to know ReLU derivative for the concrete example |
| Chain rule (calculus) | INTRODUCED | MENTIONED | implementing-linear-regression | GAP | Was only mentioned, not developed. This lesson addresses it. |
| Partial derivatives notation | INTRODUCED | MENTIONED | implementing-linear-regression | GAP | Notation was used but not formally taught. This lesson addresses it. |

**Gap resolution:** The chain rule and partial derivative notation are gaps that this lesson itself addresses. The lesson opens with the chain rule review (Section 2) specifically to close this gap before using it. This is a valid "dedicated section within this lesson" resolution per the skill.

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Need to compute gradients one parameter at a time | Did it for 2 params in implementing-linear-regression | Naive approach: N forward passes vs backprop's 1 | Section 6: Naive vs Backprop comparison |
| Backprop is a different algorithm from gradient descent | Names sound unrelated | Backprop COMPUTES the gradients; GD USES them for the update | Section 1: Problem statement |
| Gradients require global information | "How does weight in layer 1 know about the loss?" | Each layer only needs its LOCAL derivative — chain rule multiplies | Section 4 + Section 5: "Local × Local × Local" |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| GPT-3's 175B parameters | Positive (scale) | Shows why efficient gradient computation matters |
| Chain rule formula: dy/dx = dy/dg · dg/dx | Positive | Core mathematical tool |
| "Doubling effects" intuition (2× and 3× → 6×) | Positive | Concrete chain rule understanding |
| Single-neuron backprop trace (z → a → L) | Positive | Step-by-step concrete example |
| BackpropFlowExplorer widget | Interactive positive | Visualize gradient flow |
| Naive O(N) vs Backprop O(1) comparison | Positive pair | Why backprop is revolutionary |

---

## Phase 3: Design

### Narrative Arc

The lesson starts with the scale problem: "You computed gradients for 2 parameters. How do you do it for 175 billion?" This creates urgency. Then it introduces the chain rule as the solution — the mathematical tool that makes it possible. The forward/backward pass structure connects to the training loop they already know (the "backward" step they've seen but never opened up). A concrete example walks through one neuron. The key insight section crystallizes the mechanism. Finally, the efficiency comparison shows why backprop was revolutionary.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Symbolic | Chain rule formula, backward pass formula | Mathematical precision |
| Verbal | "Effects multiply through the chain" | Intuitive chain rule explanation |
| Interactive | BackpropFlowExplorer | Watch gradients flow through a network |
| Visual | PhaseCards for Forward/Backward passes | Sequential phases clearly distinguished |
| Concrete | Single-neuron trace: z = wx+b → a = ReLU(z) → L = (y-a)² | Worked example with specific operations |
| Visual | ConceptBlock comparison (Naive vs Backprop) | Efficiency argument |

### Cognitive Load

- **New concepts:** 3 (chain rule for composed functions, backward pass / backpropagation, computational efficiency of backprop)
- **Previous lesson load:** CONSOLIDATE (activation-functions-deep-dive was a reference)
- **This lesson:** STRETCH — the most technically demanding concept so far
- **Assessment:** 3 concepts but chain rule is really a prerequisite recap, and efficiency is a consequence, so the true novelty is the backprop algorithm itself. Appropriate for a STRETCH lesson.

### Scope Boundaries

- Single-neuron backprop example only (not multi-layer)
- Chain rule is reviewed but not derived from first principles
- Vanishing gradients explained at intuition level (ReLU vs sigmoid derivatives)
- Does NOT cover: multi-layer worked example with numbers, automatic differentiation implementation, computational graphs
- Does NOT cover: stochastic gradient descent, batching, momentum
- Partial derivatives used but formal multivariate calculus not required
- The BackpropFlowExplorer widget provides the multi-layer visualization the text does not

---

## What Was Actually Built

1. **Header + Objective** — "the algorithm that computes how each parameter should change"
2. **The Problem: Many Parameters** — Scale problem (175B parameters), need dL/dw for every w
3. **The Key Idea: Chain Rule** — Formula + "effects multiply" intuition
4. **Two Passes: Forward and Backward** — PhaseCards (forward: input→output, backward: gradient flows back)
5. **BackpropFlowExplorer widget** — Interactive: step through forward/backward, watch gradients
6. **A Concrete Example** — Single-neuron trace: z = wx+b → a = ReLU(z) → L = (y-a)²
7. **The Key Insight** — 3-step algorithm: start at loss, flow backward, collect gradients
8. **Why Backprop Changed Everything** — Naive O(N) vs Backprop O(1) ConceptBlock comparison
9. **Summary** — 4 key takeaways
10. **Next Step** — Link to home (future: batching, optimizers)

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (3), TipBlock (2), WarningBlock (1), TryThisBlock (1), ConceptBlock (4)
- PhaseCard (2), SummaryBlock, NextStepBlock
- BackpropFlowExplorer via ExercisePanel
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Backprop = efficient gradient computation** — one forward + one backward pass gives ALL gradients
- **Chain rule = multiply local derivatives** — each layer only needs its own derivative
- **"Local × Local × Local"** — each layer is independent; gradients compose by multiplication
- **Forward pass = compute + save** — save intermediate values for the backward pass
- **Backward pass = error signal propagating back** — from loss through each layer
- **Vanishing gradients = small derivatives multiplied** — why sigmoid is bad for deep networks, ReLU is better
- **Autograd / automatic differentiation** — frameworks implement this automatically

### Analogies Used
- "Effects multiply through the chain" (doubling × tripling = 6×)
- "Error signal propagates backward" for gradient flow
- PhaseCard visual metaphor for sequential passes

### Notes

This lesson currently stands alone in Module 1.3, but the curriculum plan calls for 7 lessons covering the full backpropagation topic (chain rule deep dive, multi-layer worked examples, computational graphs, etc.). The current lesson covers the high-level concept but lacks the depth of a complete module. Future lessons should decompose the pieces further.

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

#### [CRITICAL] — Partial derivative notation introduced without explanation

**Location:** Section 1 "The Problem: Many Parameters" (line 87)
**Issue:** The lesson introduces ∂L/∂w notation (partial derivatives) without explaining what the ∂ symbol means or how it differs from the d in dL/dw that the student learned in implementing-linear-regression. The planning document identifies this as a GAP to be resolved within the lesson, but the resolution was supposed to happen in the chain rule section. Instead, the ∂ notation appears one full section earlier (Section 1) before any explanation.
**Student impact:** Student sees ∂L/∂w and either (a) doesn't notice it's different from dL/dw and gets confused later, or (b) notices and is immediately lost wondering what the curly-d means. Either way, a cognitive bump at a critical moment — the exact sentence that defines what backpropagation does.
**Suggested fix:** Either (1) add a brief parenthetical or aside when ∂ first appears explaining "this is partial derivative notation — it means 'how does L change when we change just w, holding everything else fixed'" or (2) use the familiar dL/dw notation in Section 1 and introduce ∂ notation explicitly in Section 2 when the chain rule is covered.

#### [IMPROVEMENT] — Interactive widget placed before the simple worked example

**Location:** BackpropFlowExplorer appears between Section 3 (Two Passes) and Section 4 (A Concrete Example)
**Issue:** The widget shows a 2-layer network (x → Linear₁ → ReLU → Linear₂ → Loss) with 4 parameters, 11 computation steps, gradient labels, and edge annotations. Section 4 then presents a simpler single-neuron example. This violates simple-before-complex ordering — the student encounters the complex interactive before they've seen any concrete worked example of backprop.
**Student impact:** The student reaches the widget having understood "forward and backward are two passes" conceptually but has never seen actual gradient computation through layers. The widget shows step-by-step formulas like ∂L/∂w₂ = ∂L/∂ŷ · a₁ — which the student can't parse yet because Section 4 hasn't happened. They'll click through the widget without understanding the math, then read Section 4 and wish they could go back.
**Suggested fix:** Move the BackpropFlowExplorer after Section 4 (the concrete example). The student learns the simple case first, then sees it in action in the interactive. Alternatively, move it after Section 5 as a capstone — the student has the full algorithm, then explores it interactively.

#### [IMPROVEMENT] — Indicator function notation unexplained

**Location:** Section 4, backward pass formula (line 243)
**Issue:** The term 𝟙_{z>0} appears in the backward pass computation for ∂a/∂z (the ReLU derivative). This is indicator function notation from mathematics. The student knows ReLU = max(0,x) and knows its derivative is "0 when negative, 1 when positive" from activation-functions, but has never seen the 𝟙 notation.
**Student impact:** Student reads the formula and hits an unfamiliar symbol. Minor confusion — they can probably infer it means "1 if z>0, 0 otherwise" from context, but it breaks the flow of an otherwise carefully grounded example.
**Suggested fix:** Replace 𝟙_{z>0} with a plain-English version in the formula, like `1 if z>0, else 0`, or add a brief inline note: "where 𝟙_{z>0} means 1 when z is positive, 0 otherwise (that's the ReLU derivative)."

#### [IMPROVEMENT] — Misconceptions lack concrete negative examples

**Location:** Throughout the lesson
**Issue:** The planning document identifies 3 misconceptions. Only one (misconception 1: "need to compute gradients one at a time") gets a concrete negative example (the naive O(N) approach in Section 6). Misconception 2 ("backprop is different from gradient descent") is addressed through explanation but never shown concretely — "here's what would happen if backprop and GD were separate." Misconception 3 ("gradients require global info") is addressed via the "Local × Local × Local" insight but lacks a concrete example showing what "global computation" would look like vs local.
**Student impact:** The student may nod along with the explanations but not deeply internalize why the misconceptions are wrong. Concrete counter-examples are more memorable than verbal corrections.
**Suggested fix:** For misconception 2, add a brief clarification sentence or aside: "Backpropagation is not a separate learning algorithm — it's the computation step inside gradient descent that figures out the gradients." For misconception 3, the existing "Local × Local × Local" InsightBlock in Section 4 is nearly sufficient — consider adding one sentence like: "Layer 2 doesn't need to know anything about layer 1's weights — it only needs the gradient coming in from above."

#### [POLISH] — Range input for learning rate lacks cursor style

**Location:** BackpropFlowExplorer, learning rate slider (line 661-669)
**Issue:** The HTML range input uses default cursor. Per the Interaction Design Rule, draggable elements should have explicit cursor styles.
**Student impact:** Minor — most users understand range inputs are draggable, but consistency matters.
**Suggested fix:** Add `cursor-pointer` or `cursor-ew-resize` className to the range input.

#### [POLISH] — Section 4 backward pass is dense

**Location:** Section 4 "A Concrete Example," lines 231-250
**Issue:** The backward pass portion packs the chain rule application, three derivative terms, and a formula with indicator function notation into a short block. After the carefully paced forward pass (3 separate lines), the backward pass is one formula with a decomposition.
**Student impact:** Student may feel the backward pass is rushed compared to the forward pass explanation. The lesson text paces the forward pass as three visually distinct lines but condenses the backward pass into one formula + one decomposition.
**Suggested fix:** Break the backward pass into the same step-by-step visual format as the forward pass — show each local derivative on its own line with labels, then show the multiplication. This mirrors the forward pass structure and reinforces "local × local × local."

### Review Notes

**What works well:**
- The motivation hook (175B parameters) is genuinely compelling
- The "effects multiply through the chain" intuition in the aside is excellent
- The PhaseCards for forward/backward are clear and well-structured
- The Naive vs Backprop efficiency comparison is strong pedagogy
- The BackpropFlowExplorer widget is technically impressive and well-built
- The summary captures the right mental models

**Systemic pattern:** The lesson's main weakness is ordering — it tends to present complex things before simple ones (widget before worked example, ∂ notation before explanation). This is likely because the lesson was built following a conceptual flow (problem → tool → passes → visualization → example) rather than a learning flow (problem → tool → simple example → visualization → insight).

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS (with polish notes)

All critical and improvement findings from iteration 1 have been resolved:
- ✅ Partial derivative notation now explained inline when first introduced (Section 1, line 87-90)
- ✅ Widget moved after concrete example (now between Section 4 and Section 5)
- ✅ Indicator function replaced with plain English "1 if z > 0, else 0" plus ReLU derivative explanation
- ✅ Backward pass expanded to step-by-step format matching the forward pass structure
- ✅ Misconception 2 (backprop ≠ GD) addressed with explicit clarification (lines 95-97)
- ✅ Misconception 3 (global vs local) addressed in revised "Local × Local × Local" aside (lines 259-262)
- ✅ Cursor style added to learning rate slider

### Findings

#### [POLISH] — Concrete example stays symbolic (no plugged-in numbers)

**Location:** Section 4 "A Concrete Example"
**Issue:** The forward pass shows z = w·x + b, a = ReLU(z), L = (y-a)² and the backward pass decomposes the chain rule — but never plugs in actual numbers (e.g., "if x=1, w=0.5, b=0, then z=0.5"). The widget provides numerical computation, but the text stays symbolic.
**Student impact:** Minimal — the widget compensates with live numbers, and the text's purpose is to show the chain rule structure. A student who clicks Step in the widget will see exact numbers. But a student who skips the widget gets only symbolic formulas.
**Suggested fix:** Optionally add one line after the forward pass block: "For example, with w=0.5, x=1, b=0: z=0.5, a=0.5." Low priority.

#### [POLISH] — Chain rule section uses "Remember" framing for MENTIONED-depth concept

**Location:** Section 2, line 118-119
**Issue:** "Remember the chain rule from calculus?" implies the student should recall it well. The student has the chain rule at MENTIONED depth — it was only mentioned in implementing-linear-regression, not developed. The section does develop it fully, so this is a tone issue rather than a pedagogical gap.
**Student impact:** Minimal — the student might feel a moment of "wait, should I know this?" but the section immediately teaches it. The "Remember" framing is a common pedagogical convention even when introducing something fresh.
**Suggested fix:** Could change to "The chain rule from calculus says..." or "Here's the chain rule from calculus:" to avoid implying prior mastery. Very low priority.

### Review Notes

**What improved since iteration 1:**
- The ordering fix (widget after concrete example) significantly improves the learning flow. The student now sees: concept → simple example → interactive → insight → efficiency. This is textbook "simple before complex."
- The step-by-step backward pass mirrors the forward pass structure, making the chain rule feel symmetric and graspable rather than dense.
- The partial derivative explanation is smooth and natural — it doesn't feel like an interruption.
- The misconception clarifications read naturally in context.

**Overall assessment:** The lesson now follows a clean learning arc: problem (why do we need this?) → tool (chain rule) → structure (forward/backward passes) → concrete example (single neuron, step by step) → interactive exploration (2-layer widget) → crystallization (3-step algorithm) → historical context (efficiency revolution). Each section connects to the next, complexity increases gradually, and all new concepts are grounded before use.

---

## Improvement Summary — 2026-02-06

**Iterations:** 2/3
**Final verdict:** PASS (with minor polish notes)

### Changes made:
- [Iteration 1] Fixed: Added inline explanation of ∂ (partial derivative) notation when first introduced in Section 1
- [Iteration 1] Fixed: Moved BackpropFlowExplorer widget from before Section 4 to after it (simple before complex)
- [Iteration 1] Fixed: Replaced indicator function notation (𝟙_{z>0}) with plain English "1 if z > 0, else 0"
- [Iteration 1] Fixed: Expanded backward pass to step-by-step format (Steps 1, 2, 3) matching forward pass structure
- [Iteration 1] Fixed: Added explicit misconception clarification for backprop vs gradient descent (Section 1)
- [Iteration 1] Fixed: Strengthened "Local × Local × Local" aside with locality explanation for misconception 3
- [Iteration 1] Fixed: Added cursor-pointer to learning rate slider in BackpropFlowExplorer

### Remaining items (polish, at user's discretion):
- Concrete example stays symbolic (no plugged-in numbers) — widget compensates
- "Remember the chain rule" framing could be softer for MENTIONED-depth concept
