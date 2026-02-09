# Lesson: Backprop by the Numbers

**Module:** 1.3 — Training Neural Networks
**Slug:** backprop-worked-example
**Position:** Lesson 2 of 7
**Type:** BUILD (applies lesson 1's concepts with concrete numbers)
**Status:** Planning

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| Chain rule for composed functions | DEVELOPED | backpropagation — dy/dx = dy/dg * dg/dx; "effects multiply through the chain" |
| Forward pass (concept) | INTRODUCED | backpropagation — compute + save intermediate values; seen symbolically |
| Backward pass (concept) | INTRODUCED | backpropagation — error signal propagates from loss through layers |
| Local derivatives | INTRODUCED | backpropagation — each layer computes its own derivative independently |
| "Local x Local x Local" analogy | ESTABLISHED | backpropagation — layers are independent, gradients compose by multiplication |
| Single-neuron symbolic backprop | DEVELOPED | backpropagation — z=wx+b, a=ReLU(z), L=(y-a)^2, chain rule decomposition |
| Backprop efficiency (O(1) vs O(N)) | INTRODUCED | backpropagation — why backprop matters at scale |
| Gradient descent update rule | DEVELOPED | gradient-descent — theta_new = theta_old - alpha * grad_L |
| Training loop (forward->loss->backward->update) | DEVELOPED | implementing-linear-regression — 6-step loop, "heartbeat of training" |
| Activation functions (ReLU, sigmoid) | DEVELOPED | activation-functions — formulas, shapes, when to use |
| ReLU derivative | INTRODUCED | backpropagation — "1 if z > 0, else 0" |
| Partial derivative notation (d/dw) | INTRODUCED | backpropagation — "how does L change when we change just w" |
| Neuron = weighted sum + bias + activation | DEVELOPED | neuron-basics, activation-functions |
| Layers and networks | INTRODUCED | neuron-basics — input -> hidden -> output |
| Parameters (weight, bias) | DEVELOPED | linear-regression — "the knobs the model learns" |
| MSE loss function | DEVELOPED | loss-functions — formula, squaring rationale, loss landscape |
| Learning rate | DEVELOPED | learning-rate — step size, failure modes |

**Mental models established:**
- "Effects multiply through the chain" — chain rule intuition
- "Local x Local x Local" — each layer is independent
- "One forward + one backward = ALL gradients" — backprop's efficiency
- "Training = forward -> loss -> backward -> update" — the universal loop
- "Ball rolling downhill" — gradient descent
- "Error signal propagates backward" — gradient flow direction

**What was explicitly NOT covered in lesson 1:**
- Multi-layer worked example with actual numbers
- Computational graphs
- Automatic differentiation details
- Batching, optimizers

**Readiness assessment:** The student has the conceptual framework (chain rule, forward/backward passes, local derivatives) but has only seen it applied to a single neuron symbolically. They're ready to see it with real numbers through multiple layers. This is the natural "make it concrete" step.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to compute every gradient in a 2-layer neural network by hand, using concrete numbers, applying the chain rule backward from the loss through each layer.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Chain rule for composed functions | DEVELOPED | DEVELOPED | backpropagation | OK | Core tool for computing gradients layer by layer |
| Forward pass concept | INTRODUCED | INTRODUCED | backpropagation | OK | Student knows the idea; this lesson DEVELOPS it with numbers |
| Backward pass concept | INTRODUCED | INTRODUCED | backpropagation | OK | Student knows the idea; this lesson DEVELOPS it with numbers |
| Local derivatives | INTRODUCED | INTRODUCED | backpropagation | OK | Student knows each layer has its own; this lesson shows it concretely |
| ReLU and its derivative | DEVELOPED | DEVELOPED (formula) / INTRODUCED (derivative) | activation-functions, backpropagation | OK | Need ReLU(x) = max(0,x), derivative = 1 if x>0 else 0 |
| MSE loss function | DEVELOPED | DEVELOPED | loss-functions | OK | Need to compute loss and its derivative |
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent | OK | Apply updates with computed gradients |
| Neural network architecture (layers) | INTRODUCED | INTRODUCED | neuron-basics | OK | Need to understand 2-layer structure |
| Partial derivative notation | INTRODUCED | INTRODUCED | backpropagation | OK | Will use dL/dw notation throughout |
| Matrix multiplication | INTRODUCED | NOT TAUGHT | — | GAP | A 2-layer network with multiple neurons requires matrix operations |

**Gap resolution:**

The matrix multiplication gap is significant but avoidable. Instead of a full multi-neuron-per-layer network, use a **2-layer network where each layer has exactly 1 neuron**:
- Layer 1: x -> w1*x + b1 -> ReLU -> a1
- Layer 2: a1 -> w2*a1 + b2 -> y_hat
- Loss: L = (y - y_hat)^2

This gives us 4 parameters (w1, b1, w2, b2), multiple composed operations, and a genuine multi-layer backprop experience—without requiring matrix math. The student gets the full chain rule experience through layers without a prerequisite they don't have.

An aside can note that real networks have many neurons per layer (requiring matrices), but the gradient computation principle is identical—you just do it for more paths.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "The backward pass is just the forward pass in reverse" | "Backward" sounds like running the network backward | Show that the backward pass computes different things (derivatives, not activations). Forward: input x -> multiply by w -> add b -> ReLU -> etc. Backward: start at loss, compute dL/dy_hat, then dL/dw2, etc. Different operations, different results. | Section 3 (Backward Pass) — explicitly contrast what each pass computes |
| "I need to re-derive everything from scratch for each network" | The symbolic math in lesson 1 felt specific to that example | Show the pattern: every layer's backward step has the same structure (receive gradient from above, compute local derivative, multiply, pass down). Once you see the pattern, any network is just repetition. | Section 4 (The Pattern) |
| "The gradient of a weight depends on all other weights" | Gradients in a network feel interconnected | Walk through dL/dw1: it depends on the chain of operations FROM w1 TO the loss, but NOT on w2 or b2's values directly. w2 appears in the chain (because the forward pass goes through it), but w1's gradient doesn't depend on what w2's gradient is. | Section 3, when computing dL/dw1 — note what appears and what doesn't |
| "Backprop gives approximate gradients" | The concept seems too good to be true / confused with numerical differentiation | Every number in the worked example is exact. Verify: change w1 by a tiny epsilon, rerun forward pass, check that (L_new - L_old)/epsilon matches the analytical gradient. | Section 5 (Verify) — numerical gradient check |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| 2-layer network with x=2, w1=0.5, b1=0.1, w2=-0.3, b2=0.2, y=1 | Positive (primary) | Full worked example: forward pass, loss, backward pass, all 4 gradients | Simple numbers that produce non-zero ReLU activation, non-trivial gradients. Small enough to track by hand. |
| Numerical gradient verification (epsilon method) | Positive (verification) | Proves backprop gives exact gradients by comparing to finite differences | Dispels "approximation" misconception; gives student a debugging tool they'll use forever |
| What happens when ReLU kills the gradient (z1 < 0) | Negative | Shows that if the hidden neuron's pre-activation is negative, ReLU derivative is 0, and w1/b1 get zero gradient | Connects to "dying ReLU" mentioned in Module 1.2; shows why activation choice matters for gradient flow |
| Weight update step with alpha=0.1 | Positive (application) | Completes the loop: computed gradients -> actual parameter updates -> verify loss decreases | Closes the "training loop" from Module 1.1; the student sees the FULL cycle with real numbers |

---

## Phase 3: Design

### Narrative Arc

In the previous lesson, you learned what backpropagation does: it uses the chain rule to compute gradients for every parameter in one forward + one backward pass. You saw the algorithm symbolically and watched it in an interactive widget. But there's a difference between watching a magic trick and performing one yourself.

This lesson is where you perform the trick. We'll take a tiny 2-layer neural network, plug in actual numbers, and trace every computation—forward pass, loss, backward pass—step by step. By the end, you'll have computed 4 gradients by hand and verified them. No more "trust me, the math works." You'll see exactly why it works, because you'll do it yourself.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example | Full worked example with x=2, y=1, specific weights, every intermediate value | Backprop is a computation, and computations become real when you see actual numbers. The symbolic version in lesson 1 showed the structure; numbers show the substance. |
| Visual | Network diagram with values flowing through (forward=blue, backward=red arrows with gradient values) | The student needs to see WHERE in the network each computation happens. A diagram with actual numbers annotated on edges makes the flow tangible. |
| Interactive | Widget where student can change inputs/weights and watch all forward/backward values update live | Builds intuition for how changing one weight ripples through the network. "What if w1 were larger?" becomes instantly explorable. |
| Symbolic | The chain rule decomposition alongside the numbers (dL/dw1 = dL/dy_hat * dy_hat/da1 * da1/dz1 * dz1/dw1 = numbers) | Shows the correspondence between the abstract chain rule and the concrete numbers. The formula IS the recipe; the numbers are the ingredients. |

### Cognitive Load Assessment
- **New concepts in this lesson:** 2 (multi-layer gradient computation with numbers, numerical gradient verification)
- **Previous lesson load:** STRETCH (backpropagation—chain rule, forward/backward, efficiency)
- **This lesson's load:** BUILD—applies lesson 1's framework to concrete numbers. The chain rule and forward/backward structure are already established. The novelty is doing it with numbers, not learning a new framework.
- **Assessment:** Appropriate. The student is extending, not learning from scratch.

### Connections to Prior Concepts
- **Chain rule (lesson 1):** "Effects multiply through the chain"—now we'll see the actual numbers multiplying
- **Forward pass (lesson 1):** Was introduced as "compute + save"—now we'll compute with real values and see WHY we save them
- **Training loop (Module 1.1):** forward -> loss -> backward -> update. This lesson walks through every step with a real network.
- **Gradient descent update rule (Module 1.1):** theta_new = theta_old - alpha * grad. At the end, we'll apply this with our computed gradients.
- **ReLU (Module 1.2):** We'll use ReLU as the activation and see its derivative in action (1 or 0)
- **MSE loss (Module 1.1):** We'll compute the loss and its derivative with actual numbers

**Analogies to extend:**
- "Local x Local x Local" — we'll see the actual local derivatives as numbers and watch them multiply
- "Forward pass saves values for the backward pass" — we'll see exactly which saved values get used where

**No misleading analogies from prior lessons.**

### Scope Boundaries

**This lesson IS about:**
- Computing forward pass, loss, backward pass, and weight updates for a 2-layer, 1-neuron-per-layer network
- Seeing concrete numbers at every step
- Verifying gradients numerically
- Target depth: forward pass DEVELOPED, backward pass DEVELOPED, multi-layer gradient computation DEVELOPED

**This lesson is NOT about:**
- Multi-neuron layers (requires matrix math—deferred)
- Computational graphs (lesson 3)
- Batching or SGD (lesson 4)
- Optimizers beyond vanilla gradient descent (lesson 5)
- Why training can fail (lesson 6)

### Lesson Outline

1. **Context + Constraints** — "Last time: what backprop does. This time: do it yourself with real numbers." Scope: 2-layer network, 4 parameters, no matrices.

2. **The Network** — Draw the architecture: x -> [w1, b1] -> ReLU -> [w2, b2] -> y_hat. Assign concrete values: x=2, w1=0.5, b1=0.1, w2=-0.3, b2=0.2, y=1. Note: this is the smallest network that's genuinely "deep"—it has a hidden layer.

3. **Forward Pass (with numbers)** — Step through:
   - z1 = w1*x + b1 = 0.5*2 + 0.1 = 1.1
   - a1 = ReLU(1.1) = 1.1
   - z2 = w2*a1 + b2 = -0.3*1.1 + 0.2 = -0.13
   - y_hat = z2 = -0.13 (no activation on output for regression)
   - L = (y - y_hat)^2 = (1 - (-0.13))^2 = 1.2769

   Aside: "See why we SAVE these values? We'll need z1, a1, z2, y_hat in the backward pass."

4. **Backward Pass (with numbers)** — Step through, one derivative at a time:
   - Start: dL/dy_hat = -2(y - y_hat) = -2(1 - (-0.13)) = -2.26
   - Layer 2 weight: dL/dw2 = dL/dy_hat * dy_hat/dw2 = dL/dy_hat * a1 = -2.26 * 1.1 = -2.486
   - Layer 2 bias: dL/db2 = dL/dy_hat * 1 = -2.26
   - Pass gradient to layer 1: dL/da1 = dL/dy_hat * dy_hat/da1 = dL/dy_hat * w2 = -2.26 * (-0.3) = 0.678
   - Through ReLU: dL/dz1 = dL/da1 * da1/dz1 = 0.678 * 1 = 0.678 (z1=1.1 > 0, so ReLU derivative = 1)
   - Layer 1 weight: dL/dw1 = dL/dz1 * dz1/dw1 = 0.678 * x = 0.678 * 2 = 1.356
   - Layer 1 bias: dL/db1 = dL/dz1 * 1 = 0.678

   **Check (predict-and-verify):** "Before we computed dL/dw1, the chain had 4 local derivatives multiplied together. Can you trace them?" Answer: dL/dy_hat * dy_hat/da1 * da1/dz1 * dz1/dw1 = the "Local x Local x Local x Local" from lesson 1.

5. **The Pattern** — Crystallize: every layer's backward step is the same 3-step recipe: (1) receive gradient from the layer above, (2) compute local derivatives, (3) multiply and pass down. This is the "local" insight from lesson 1, now proven with numbers.

6. **Negative Example: Dead ReLU** — "What if w1 were -1 instead of 0.5?" Then z1 = -1*2 + 0.1 = -1.9, a1 = ReLU(-1.9) = 0, and the ReLU derivative is 0. So dL/dz1 = dL/da1 * 0 = 0. Layer 1's weights get ZERO gradient—they can't learn. This is the "dying ReLU" problem. Brief connection to lesson 1's mention of vanishing gradients.

7. **Verify: Numerical Gradient Check** — Change w1 by epsilon=0.001: recompute forward pass, get new loss. (L_new - L_old) / epsilon should approximately match dL/dw1 = 1.356. Show the computation. This proves backprop is exact, not an approximation.

8. **Apply: Weight Update** — theta_new = theta_old - 0.1 * gradient for all 4 parameters. Show the new values. Rerun forward pass: new loss is lower. The network learned. One complete training step.

9. **Interactive Widget** — BackpropCalculator: student can change x, y, w1, b1, w2, b2 and see all forward/backward values update live. Highlight the gradient flow with color. Optional: "step" button that applies the update and shows loss decrease.

10. **Summary** — Key takeaways: (1) Backprop is just the chain rule with real numbers, (2) Each layer only needs its local info + the gradient from above, (3) You can verify any gradient numerically, (4) One forward + backward + update = one training step.

11. **Next step** — Computational graphs: a visual notation that makes this bookkeeping automatic.

### Assessment Moments
- **Check 1 (Section 4):** Predict-and-verify: "How many local derivatives multiply together for dL/dw1?" (4)
- **Check 2 (Section 6):** "What gradient does layer 1 get when ReLU kills the signal?" (Zero—connection to dying ReLU)
- **Check 3 (Section 8):** "After the update, should the loss be higher or lower?" (Lower—gradient descent always moves downhill for small enough learning rate)

---

## What Was Actually Built

Implementation followed the design closely. No significant deviations.

1. **Header + Objective** — "Compute every gradient in a 2-layer neural network by hand"
2. **Constraints** — 2 layers, 1 neuron each, 4 parameters, no matrices
3. **The Network** — Architecture diagram with assigned values (x=2, w1=0.5, b1=0.1, w2=-0.3, b2=0.2, y=1)
4. **Forward Pass** — PhaseCard with 4 NumberSteps (z1, a1, y_hat, L), emphasis on saving intermediate values
5. **Backward Pass** — PhaseCard with 7 NumberSteps organized by layer (loss → layer 2 → pass to layer 1 → ReLU → layer 1), each showing "incoming gradient × local derivative"
6. **Check Your Understanding** — Chain of 4 local derivatives for dL/dw1 with labeled underbrace notation
7. **Dead ReLU** — Negative example: w1=-1 → z1=-1.9 → gradient=0, connecting to dying ReLU from Module 1.2
8. **Numerical Verification** — Epsilon method comparing analytical gradient to finite differences
9. **Weight Update** — All 4 parameters updated with lr=0.1, new loss shown to be lower
10. **Interactive Widget** — BackpropCalculator: forward/backward values, update button, manual parameter sliders, dead ReLU detection
11. **Summary** — 4 items: chain rule with numbers, no global info needed, gradient checks, complete training step
12. **Next Step** — Computational Graphs

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, ConstraintBlock, SectionHeader
- InsightBlock (3), TipBlock (3), TryThisBlock (1), WarningBlock (1), ConceptBlock (1)
- PhaseCard (2), SummaryBlock, NextStepBlock
- Custom: NumberStep (inline), ValueBox (inline), BackpropCalculatorWidget (inline)
- KaTeX: InlineMath, BlockMath

### Design Deviation Notes
- Widget built inline (not as separate file) since it's lesson-specific and simple enough
- Dead ReLU values shown as inline text rather than computed variables
- No separate MermaidDiagram for architecture—used styled HTML boxes instead for more compact inline display

---

## Review — 2026-02-08 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

#### [CRITICAL] — Missing motivation paragraph before the worked example

**Location:** Between ConstraintBlock and "The Network" section (lines 357-370)
**Issue:** The lesson jumps from constraints directly into "Here's our network" without explaining WHY the student is doing this exercise. The planning document specified a narrative: "there's a difference between watching a magic trick and performing one yourself. This lesson is where you perform the trick." This motivation paragraph is entirely absent from the built lesson.
**Student impact:** The student goes from "here are the rules" to "here's a network with these numbers" with no bridge. They don't feel the *need* for this exercise—they just read the conceptual version in the previous lesson and might think "I already understand this, why am I doing it again with numbers?" The motivation paragraph creates the gap between "understanding the concept" and "being able to do it yourself."
**Suggested fix:** Add a motivation paragraph after the constraints and before "The Network" section. Use the planned narrative: "You saw the algorithm symbolically. But there's a difference between watching a magic trick and performing one yourself. This lesson is where you perform the trick—we'll trace every computation with actual numbers, and by the end you'll have computed 4 gradients by hand and verified them."

#### [IMPROVEMENT] — Misconception "backward pass = reverse forward" not addressed

**Location:** Backward pass section (lines 509-621)
**Issue:** The planning document identifies the misconception "The backward pass is just the forward pass in reverse." The lesson shows that the backward pass computes different things (derivatives, not activations), but never explicitly calls this out. The student might not notice the distinction on their own—they see "forward" and "backward" and naturally assume backward is just running things in reverse.
**Student impact:** A student who holds this misconception would read the backward pass section and be confused about why the operations look different from the forward pass. They might attribute their confusion to "this is just hard math" rather than recognizing that backward and forward are fundamentally different operations.
**Suggested fix:** Add a brief callout (WarningBlock or inline text) at the start of the backward pass section: "Don't confuse 'backward' with 'reverse.' The forward pass computes activations: multiply, add, ReLU. The backward pass computes derivatives: completely different operations. 'Backward' refers to the direction (from loss toward input), not running the same operations in reverse."

#### [IMPROVEMENT] — Widget buttons missing cursor-pointer

**Location:** BackpropCalculatorWidget, buttons (lines 227-244)
**Issue:** The "Run Backward Pass" and "Apply Update" buttons don't have explicit `cursor-pointer` class. Per the Interaction Design Rule, all clickable elements should have appropriate cursor styles.
**Student impact:** Minor—buttons are recognizable as buttons, but consistency matters for interactive design quality.
**Suggested fix:** Add `cursor-pointer` to both button elements.

#### [POLISH] — Em dash spacing violation

**Location:** Line 250 in the widget: `Step {stepCount} — Loss: {fmt(loss)}`
**Issue:** Uses ` — ` (space-dash-space) instead of `—` (no spaces) per the Writing Style Rule.
**Student impact:** None functionally, but inconsistent with the lesson's own prose elsewhere.
**Suggested fix:** Change to `Step {stepCount}—Loss: {fmt(loss)}`.

#### [POLISH] — "The Pattern" crystallization is only in a sidebar

**Location:** InsightBlock after the backward pass section (lines 611-619)
**Issue:** The planning document describes a dedicated "Section 5: The Pattern" that crystallizes the 3-step recipe (receive gradient, compute local derivative, multiply and pass down). In the built lesson, this crystallization only appears as an InsightBlock aside—not in the main content. The main content says "Every step was the same pattern: incoming gradient × local derivative" (line 606) but the full 3-step recipe is sidebar-only.
**Student impact:** Minor—the sidebar does contain the insight, and the main text refers to the pattern. But the sidebar might be missed on mobile or by students who focus on main content.
**Suggested fix:** Consider elevating the 3-step recipe into the main content (even briefly), with the InsightBlock aside reinforcing it.

### Review Notes

**What works well:**
- The NumberStep component is excellent—consistent format for every computation, with step numbers, formulas, numerical results, and plain-English explanations
- The backward pass organization by layer (with sub-headings) prevents the 7-step computation from feeling overwhelming
- The Check Your Understanding section with underbrace notation is a strong assessment moment
- The dead ReLU negative example is well-placed and connects to prior knowledge
- The numerical gradient verification is a practical skill the student will use
- The before/after loss comparison in the weight update section provides a satisfying "it worked!" moment
- The widget's progressive disclosure (forward first, then backward on click) is good pedagogy

**Systemic pattern:** The lesson's main weakness is missing connective tissue—the motivation that ties the exercise to the student's learning journey, and the explicit misconception callout that prevents a common misunderstanding. The computational content itself is strong.

---

## Review — 2026-02-08 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS (with polish notes)

All critical and improvement findings from iteration 1 have been resolved:
- Motivation paragraph added between constraints and "The Network" — the "magic trick" framing creates a compelling need
- "Backward ≠ reverse" misconception addressed with amber callout before the backward pass PhaseCard
- Widget buttons now have cursor-pointer
- Em dashes fixed throughout (description, constraints, explanations, summary items)
- "The Pattern" 3-step recipe elevated to main content (numbered list) in addition to the InsightBlock aside

### Findings

#### [POLISH] — InsightBlock "The Pattern" duplicates main content

**Location:** Lines 647-655 (InsightBlock aside) vs lines 632-643 (main content)
**Issue:** The InsightBlock titled "The Pattern" contains a 3-step recipe that is now nearly identical to the numbered list in the main content (which was added to address a previous finding). The main content version is more detailed. The InsightBlock version adds no new perspective.
**Student impact:** Minimal. Redundancy between main content and sidebar is acceptable and can reinforce learning. The sidebar serves as a quick-reference version of the main content list.
**Suggested fix:** Could differentiate the InsightBlock by framing it as a takeaway reference ("Bookmark this:") rather than as a standalone explanation. Or accept the redundancy as intentional reinforcement. Very low priority.

### Review Notes

**What improved since iteration 1:**
- The motivation paragraph ("watching a magic trick vs performing one yourself") immediately establishes why this lesson matters. Without it, the lesson felt like a dry exercise; with it, the student understands they're building a skill, not just reading math.
- The "backward ≠ reverse" callout is well-placed—right before the student enters the backward pass computation, preventing the misconception before it forms.
- The 3-step recipe in main content makes the crystallization accessible to all students, not just those who read sidebars.

**Overall assessment:** The lesson now has a complete arc: motivation (why this matters) → setup (the network) → forward pass (compute) → backward pass (gradients) → understanding check (trace the chain) → negative example (dead ReLU) → verification (numerical check) → application (weight update) → exploration (interactive widget) → summary. Every section connects to the next, complexity increases gradually, misconceptions are addressed proactively, and the student leaves with both conceptual understanding and practical skill. The lesson passes.

---

## Improvement Summary — 2026-02-08

**Iterations:** 2/3
**Final verdict:** PASS (with minor polish notes)

### Changes made:
- [Iteration 1] Fixed: Added motivation paragraph ("watching a magic trick vs performing one") between constraints and "The Network" section
- [Iteration 1] Fixed: Added "backward ≠ reverse" misconception callout (amber box) before the backward pass PhaseCard
- [Iteration 1] Fixed: Added cursor-pointer to all widget buttons (Run Backward Pass, Apply Update, Reset)
- [Iteration 1] Fixed: Elevated "The Pattern" 3-step recipe from sidebar-only to main content (numbered list)
- [Iteration 1] Fixed: Corrected em dash spacing throughout (description, constraints, explanations, summary items)

### Remaining items (polish, at user's discretion):
- InsightBlock "The Pattern" duplicates the main content 3-step recipe — could differentiate or accept as intentional reinforcement
