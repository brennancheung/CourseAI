# Lesson: Computational Graphs

**Module:** 1.3 — Training Neural Networks
**Slug:** computational-graphs
**Position:** Lesson 3 of 7
**Type:** BUILD (visual tool that organizes the computation learned in lessons 1-2)
**Status:** Built

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| Chain rule for composed functions | DEVELOPED | backpropagation — dy/dx = dy/dg * dg/dx; "effects multiply through the chain" |
| Forward pass (concrete) | DEVELOPED | backprop-worked-example — x=2 through 2-layer network, every intermediate value computed and saved |
| Backward pass (concrete) | DEVELOPED | backprop-worked-example — 7-step chain rule with real numbers, all 4 gradients computed |
| Local derivatives ("incoming gradient x local derivative") | DEVELOPED | backprop-worked-example — concretely shown at every step of backward pass |
| Multi-layer gradient computation | DEVELOPED | backprop-worked-example — 4 gradients (dL/dw1, dL/db1, dL/dw2, dL/db2) through 2 layers |
| Weight update with real numbers | DEVELOPED | backprop-worked-example — theta_new = theta_old - 0.1 * gradient, loss verified to decrease |
| Numerical gradient verification | DEVELOPED | backprop-worked-example — (L(w+eps) - L(w))/eps compared to analytical gradient |
| Backprop efficiency (O(1) vs O(N)) | INTRODUCED | backpropagation — million-fold speedup, why backprop matters at scale |
| Autograd / automatic differentiation | MENTIONED | backpropagation — frameworks compute gradients automatically, no detail on how |
| Dying ReLU (concrete) | DEVELOPED | backprop-worked-example — w1=-1 leads to zero gradient, layer 1 can't learn |
| Vanishing gradients (why) | INTRODUCED | backpropagation — small derivatives multiply to near-zero; why ReLU > sigmoid |
| ReLU and its derivative | DEVELOPED | activation-functions, backpropagation — max(0,x), derivative = 1 if z>0 else 0 |
| MSE loss function | DEVELOPED | loss-functions — formula, squaring rationale, loss landscape |
| Gradient descent update rule | DEVELOPED | gradient-descent — theta_new = theta_old - alpha * grad_L |
| Training loop (forward->loss->backward->update) | DEVELOPED | implementing-linear-regression — 6-step loop, "heartbeat of training" |
| Partial derivative notation (d/dw) | INTRODUCED | backpropagation — "how does L change when we change just w" |

**Mental models established:**
- "Effects multiply through the chain" — chain rule as multiplication of local effects
- "Local x Local x Local" — each layer is independent, gradients compose by multiplication
- "Incoming gradient x local derivative" — the recipe for every backward step
- "One forward + one backward = ALL gradients" — backprop's core efficiency insight
- "Error signal propagates backward" — gradient flow direction
- "Training = forward -> loss -> backward -> update" — the universal loop
- "Ball rolling downhill" — gradient descent intuition
- "Gradient check: nudge and measure" — debugging tool for verifying gradients

**What was explicitly NOT covered in lessons 1-2:**
- Computational graphs (deferred explicitly in both lessons)
- Automatic differentiation details (MENTIONED only, no mechanism)
- Multi-neuron layers / matrices
- Batching, optimizers

**Readiness assessment:** The student has computed gradients by hand through a real network. They know the chain rule, the "incoming x local" recipe, and the full backward pass procedure. They're doing the right computation but without a visual organizer. They're ready for a notation/framework that makes the bookkeeping they've been doing by hand systematic and visual. This is a natural "here's a tool that makes what you already do easier" moment, not a new algorithm.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to represent neural network computations as computational graphs and use them to visually trace the chain rule, so they can read and reason about gradient flow in any network architecture.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Chain rule for composed functions | DEVELOPED | DEVELOPED | backpropagation | OK | Must be able to apply the chain rule, not just recognize it; student has done this with real numbers |
| Forward pass (concrete) | DEVELOPED | DEVELOPED | backprop-worked-example | OK | Must understand the sequence of operations that compute outputs from inputs |
| Backward pass (concrete) | DEVELOPED | DEVELOPED | backprop-worked-example | OK | Must be able to trace gradients backward step by step; student did this through 7 steps |
| Local derivatives | DEVELOPED | DEVELOPED | backprop-worked-example | OK | The "incoming x local" recipe is what computational graphs formalize visually |
| Partial derivative notation | INTRODUCED | INTRODUCED | backpropagation | OK | Will use dL/dx notation on graph edges; student knows the meaning from prior lessons |
| Autograd / automatic differentiation | MENTIONED | MENTIONED | backpropagation | OK | This lesson moves it to INTRODUCED; MENTIONED is sufficient starting point since we're building up from what they know |

**Gap resolution:** No gaps. All prerequisites are met at the required depth. The student has strong concrete experience with the chain rule and backward pass, which provides the ideal foundation for introducing the visual notation that organizes that same computation.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Computational graphs are a new algorithm for computing gradients" | The term "computational graph" sounds like a new technique, distinct from the chain rule they already know | Take the same 2-layer network from lesson 2. Show the backward pass done by hand (their existing method) next to the same backward pass traced on a computational graph. The numbers are identical at every step. The graph is a NOTATION, not a new algorithm. | Section 4 (Explain) — side-by-side comparison with lesson 2's computation |
| "Each node in the graph does something different and I need to memorize them all" | Graphs look complex with many different node types and operations | Show that every node follows the exact same rule: multiply incoming gradient by local derivative, pass it along. The only thing that changes is what the local derivative is (which depends on the operation: +, *, ReLU, etc.). The student already knows these local derivatives from lesson 2. | Section 5 (Elaborate) — the "one rule" crystallization |
| "You need computational graphs to compute gradients" | The lesson introduces graphs in the context of gradient computation, so the student might think graphs are required | Point to the worked example from lesson 2: we computed all 4 gradients perfectly without drawing a single graph. Computational graphs are how frameworks (PyTorch, TensorFlow) organize the computation, and they help humans visualize complex networks, but they're a convenience, not a necessity. | Section 3 (Hook) — establish this upfront, before the explanation |
| "The graph flows in one direction (forward) so gradients must too" | Directed graph arrows point forward; the student might expect gradients to flow the same way | On the same graph, show forward values flowing left-to-right along the arrows, and gradients flowing right-to-left against the arrows. The graph is the same; the direction of flow is different for forward vs backward. This connects to "backward" from the backward pass. | Section 4 (Explain) — annotate the graph with both forward (blue) and backward (red) flows |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **f(x) = (x + 1)^2, x=3** — the simplest computational graph | Positive (primary) | Introduce graph notation with the minimum possible complexity: 2 operations (add, square), 1 input, 1 output. Student draws the graph, traces forward values, then backward gradients. | Simple enough that the student can hold the entire graph in their head. Uses operations they understand (addition, squaring). Produces non-trivial gradient (2*(x+1) = 8). Demonstrates the complete concept without any neural network complexity. |
| **The lesson-2 network as a computational graph** — same 2-layer network (x=2, w1=0.5, etc.) | Positive (bridge) | Show that the network they already computed by hand IS a computational graph. Map every operation from lesson 2's forward pass to a node, and every backward step to a gradient flowing through that node. The numbers must match exactly. | This is the critical "aha" moment. The student has already done this computation. Seeing it reorganized as a graph should produce recognition, not confusion. It proves the graph is just notation for what they already know. |
| **A branching computation: f(x) = x * (x + 1)** — where x feeds into two paths | Positive (extension) | Introduce the fan-out pattern: when one variable feeds into multiple operations, its gradient is the SUM of gradients from each path. This is the one genuinely new piece of graph reasoning. | This is the simplest example of multi-path gradient flow. The student can verify: df/dx = (x+1) + x = 2x + 1 (by calculus) matches the sum of gradients from both paths. Prepares them for real networks where a single activation feeds into multiple neurons in the next layer. |
| **Negative example: the graph for "a + b" — trying to compute gradient w.r.t. c** | Negative | Show that a variable NOT in the graph has no gradient path to the output. You can only compute gradients for variables that are connected to the output through the graph. If c doesn't appear in any node, there's no path from c to the output, so dc/doutput = 0. | Defines the boundary of what computational graphs compute. Connects to the real insight: a parameter's gradient depends on the path from that parameter to the loss. No path = no gradient = that parameter doesn't learn. Links to dying ReLU (a broken path). |

---

## Phase 3: Design

### Narrative Arc

In the last two lessons, you learned to compute gradients by hand through a neural network. You applied the chain rule step by step, tracked incoming gradients, multiplied local derivatives, and traced the result all the way from the loss back to each weight. It worked. But if you're honest, the bookkeeping was a lot. You had to remember which values you saved from the forward pass, which derivative belonged to which step, and which gradient to pass to the next layer. For a 4-parameter network, that was manageable. For a network with millions of parameters, doing this in your head is impossible.

Computational graphs solve the bookkeeping problem. They're a visual notation where each operation in your network becomes a node, and the flow of values (forward) and gradients (backward) is shown as arrows between nodes. You already know the computation. This lesson gives you the map.

And here's why this matters beyond just human convenience: computational graphs are exactly how PyTorch and TensorFlow compute gradients automatically. When you write `loss.backward()` in PyTorch, the framework is tracing gradients on a computational graph it built during the forward pass. Understanding graphs means understanding what the framework is doing for you.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Visual (primary) | Computational graph diagrams with nodes for operations and edges for data flow. Forward values annotated in blue above edges, backward gradients annotated in red below edges. At least 3 graphs of increasing complexity. | Computational graphs are inherently visual. The entire point of the notation is to make the computation VISIBLE. This is the one concept where the visual IS the concept, not an illustration of it. |
| Concrete example | The same 2-layer network from lesson 2 (x=2, w1=0.5, etc.) drawn as a computational graph, with the SAME numbers the student already computed. Side-by-side with lesson 2's step-by-step. | Recognition over novelty. The student should look at the graph and think "I know these numbers!" The graph becomes a reorganization of existing knowledge, not new knowledge. |
| Interactive | Widget where the student builds a simple computational graph by connecting operations, then watches forward/backward values propagate. Or: a pre-built graph where the student can change input values and see gradients update in real time. | Building or interacting with the graph cements the connection between the notation and the computation. "Trace the gradient" becomes a spatial activity (follow the red arrows) rather than a sequential one (do the chain rule steps in order). |
| Symbolic | The chain rule written alongside the graph: dL/dw1 = dL/dy_hat * dy_hat/da1 * da1/dz1 * dz1/dw1, where each term maps to a specific edge in the graph. | Shows the 1:1 correspondence between the algebraic chain rule and the visual graph traversal. Each multiplication in the chain rule = one hop backward in the graph. |
| Intuitive | "The graph IS the chain rule, drawn instead of written." The backward pass = walking the graph in reverse, collecting multiplications. | Collapses the concept to a single sentence. If the student remembers nothing else, this sentence reconstructs the concept. |

### Cognitive Load Assessment
- **New concepts in this lesson:** 1 (computational graph notation). Also reinforces automatic differentiation from MENTIONED to INTRODUCED.
- **Previous lesson load:** BUILD (backprop-worked-example — concrete numbers, not new frameworks)
- **This lesson's load:** BUILD — applies a visual notation to the computation the student already knows. The chain rule, forward/backward passes, and local derivatives are all established. The novelty is the visual framework, not the underlying math.
- **Assessment:** Appropriate. Two BUILD lessons in a row is fine because each adds a different dimension (lesson 2 = concrete numbers, lesson 3 = visual organization). The student isn't being stretched, just gaining a new way to see what they already know.

### Connections to Prior Concepts
- **Chain rule (lessons 1-2):** "Effects multiply through the chain" = walking backward through graph nodes, multiplying at each. The chain rule IS the graph traversal.
- **"Incoming gradient x local derivative" (lesson 2):** Each node in the graph implements exactly this recipe. The graph makes the recipe visual.
- **Forward pass saves values (lesson 2):** "See why we SAVE these values?" — the computational graph shows this explicitly: forward values are stored on edges and used during the backward pass through those same edges.
- **Dying ReLU (lesson 2):** A ReLU node with negative input produces a 0 on the backward edge. The graph makes this visible as a "broken link" in the gradient chain.
- **Autograd (lesson 1):** MENTIONED as "frameworks compute gradients automatically." Now we can explain HOW: they build and traverse computational graphs. This is the mechanism behind `loss.backward()`.
- **"One forward + one backward = ALL gradients" (lesson 1):** Visible in the graph: forward pass fills in all edge values left-to-right, backward pass fills in all gradients right-to-left. One traversal in each direction.

**Analogies to extend:**
- "Local x Local x Local" becomes "hop x hop x hop through the graph" — each hop is one edge traversal
- "Incoming gradient x local derivative" becomes the rule at each node visually

**Potentially misleading analogies:** None identified. The graph notation is consistent with all established analogies.

### Scope Boundaries

**This lesson IS about:**
- What computational graphs are (nodes = operations, edges = data flow)
- How to draw one from a sequence of operations
- How to trace the forward pass on a graph (left-to-right, annotate edge values)
- How to trace the backward pass on a graph (right-to-left, "incoming x local" at each node)
- The fan-out rule (when a value feeds into multiple operations, sum the gradients)
- How this connects to autograd / automatic differentiation (conceptual, not implementation)
- Target depths: computational graph notation DEVELOPED, automatic differentiation INTRODUCED

**This lesson is NOT about:**
- Implementing computational graphs in code (that's PyTorch's job)
- Dynamic vs static graphs (PyTorch vs TensorFlow distinction — deferred)
- Higher-order derivatives or Hessians
- Vectorized operations or matrix calculus on graphs
- Batching or SGD (lesson 4)
- Optimizers (lesson 5)

### Lesson Outline

1. **Context + Constraints** — "You've computed gradients by hand. This lesson gives you the visual tool that organizes that computation." Scope: we're learning notation, not a new algorithm. Same math, better map.

2. **Hook (before/after)** — Show two representations of the same gradient computation side by side. Left: the 7-step backward pass from lesson 2, written as a numbered list. Right: the same computation as a graph with arrows. Ask: "Which one would you rather debug when there are 50 layers?" The graph wins. This motivates the notation.

3. **Explain: The Simplest Graph** — Start with f(x) = (x + 1)^2, x=3.
   - Draw the graph: x -> [+ 1] -> [square] -> f
   - Forward pass: x=3, after add=4, after square=16
   - Backward pass: start at f with gradient 1, through square (local derivative 2*input=8) -> gradient 8, through add (local derivative 1) -> gradient 8
   - Verify: df/dx = 2(x+1) = 2*4 = 8. Matches.
   - Label the anatomy: nodes = operations, edges = values (forward) and gradients (backward), "upstream gradient" and "downstream gradient"

4. **Check (predict-and-verify)** — "What would the backward gradient be if f(x) = (x + 2)^2 and x=1?" Student predicts, then we trace: add gives 3, square gives 9, backward through square: 2*3=6, through add: 6*1=6. Verify: df/dx = 2(x+2) = 6.

5. **Explain: The Lesson-2 Network as a Graph** — Map the exact same computation from lesson 2 onto a graph.
   - Draw: x -> [*w1] -> [+b1] -> [ReLU] -> [*w2] -> [+b2] -> [MSE(y)] -> L
   - Forward values: the same numbers from lesson 2 (z1=1.1, a1=1.1, y_hat=-0.13, L=1.2769)
   - Backward gradients: the same numbers from lesson 2 (dL/dy_hat=-2.26, etc.)
   - Explicit side-by-side: "Step 3 in lesson 2 was dL/dw2 = -2.26 * 1.1 = -2.486. On the graph, that's the gradient arriving at the *w2 node (-2.26) times the local derivative (a1=1.1)."
   - InsightBlock: "You already did this computation. The graph is the same thing, drawn instead of written."

6. **Elaborate: The Fan-Out Rule** — The one genuinely new piece.
   - Example: f(x) = x * (x + 1), x=3
   - Draw the graph: x fans out to both [* ] and [+ 1]. Results merge at the multiply node.
   - Forward: x=3, top path: 3, bottom path: 3+1=4, multiply: 3*4=12
   - Backward: through multiply node, gradient to top path = 4 (value from bottom), gradient to bottom path = 3 (value from top). Bottom path through add: gradient stays 3. At x: receives gradient 4 from top AND gradient 3 from bottom. Sum: 4 + 3 = 7.
   - Verify: df/dx = d/dx(x^2 + x) = 2x + 1 = 7.
   - The rule: when a value feeds into multiple operations, ADD the gradients from each path.
   - Connection: in a real network, one neuron's output feeds into ALL neurons in the next layer. Same rule.

7. **Negative Example: No Path = No Gradient** — f(a,b) = a + b. What is df/dc? There is no node involving c. No path from c to the output means dc = 0. Connection to dying ReLU: a ReLU that outputs 0 doesn't literally remove the node, but it puts a 0 on the backward edge, which has the same effect as cutting the path — downstream gradients become zero.

8. **Check (transfer)** — Given a slightly different graph (e.g., f(x,y) = x*y + y, with x=2, y=3), trace forward and backward to find df/dx and df/dy. This tests the fan-out rule (y appears twice) in a context the student hasn't seen.

9. **Explore: Interactive Widget** — ComputationalGraphExplorer: a pre-built graph of the lesson-2 network. Student can adjust input x and weights, and see forward values (blue) and backward gradients (red) update in real time. Hovering over a node shows "incoming gradient x local derivative = outgoing gradient." Optional: a toggle between "graph view" and "step-by-step view" (lesson 2's format) to reinforce that they're the same computation.

10. **Connection to Autograd** — Brief section. "When you write `y = w * x + b` in PyTorch, the framework builds this exact graph behind the scenes. When you call `loss.backward()`, it walks the graph backward exactly as we just did. That's automatic differentiation — the framework does the graph traversal for you." This moves autograd from MENTIONED to INTRODUCED. No code yet; just the conceptual connection.

11. **Summary** — Key takeaways: (1) A computational graph draws operations as nodes and data flow as edges, (2) Forward pass = left-to-right through the graph, filling in values, (3) Backward pass = right-to-left, applying "incoming x local" at every node, (4) When a value fans out to multiple paths, sum the gradients, (5) This is exactly what PyTorch does when you call `loss.backward()`.

12. **Next step** — Batching and SGD: moving from "one input at a time" to training on real datasets.

### Assessment Moments
- **Check 1 (Section 4):** Predict-and-verify on a simple graph: f(x) = (x+2)^2, x=1. Tests basic graph traversal.
- **Check 2 (Section 8):** Transfer to a new graph with fan-out: f(x,y) = x*y + y. Tests the fan-out/summation rule in a new context.

### Widget Specification

**Name:** ComputationalGraphExplorer

**Purpose:** Let the student interactively trace forward and backward passes on a computational graph.

**Core interaction:** Pre-built graph of the lesson-2 network. Nodes are visual boxes with operation labels. Edges show values (blue above) and gradients (red below). Student can:
- Adjust input x and parameter values via sliders
- Watch all forward values and backward gradients update live
- Hover over any node to see the "incoming gradient x local derivative = outgoing gradient" breakdown
- Toggle between "graph view" and "step list view" (lesson 2's sequential format)

**Why this widget:** The computational graph IS the visual concept. An interactive graph where you can change values and watch propagation makes the notation come alive. The toggle to step-list view reinforces that the graph and the sequential computation are the same thing.

**Prerequisite check:** All operations in this widget (multiply, add, ReLU, MSE) are at DEVELOPED depth. The graph notation is being taught in this lesson. No multi-concept jumps — the student knows the computation and is learning to see it differently.

---

## Depth Changes After This Lesson

| Concept | Before | After | Notes |
|---------|--------|-------|-------|
| Computational graph notation | NOT TAUGHT | DEVELOPED | Core concept of this lesson: draw, read, trace forward and backward |
| Automatic differentiation | MENTIONED | INTRODUCED | Student now knows the mechanism (graph traversal) but hasn't implemented it |
| Fan-out gradient summation | NOT TAUGHT | DEVELOPED | When a value feeds multiple paths, sum the gradients |
| Chain rule for composed functions | DEVELOPED | DEVELOPED (reinforced) | Same depth, but now visualized on graphs in addition to algebraic form |
| Forward/backward pass | DEVELOPED | DEVELOPED (reinforced) | Same depth, but now with graph-based mental model in addition to step-by-step |

---

## Planning Checklist

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 planned: visual, concrete, interactive, symbolic, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (4 identified)
- [x] Cognitive load <= 3 new concepts (1 new + 1 reinforcement)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues. The lesson is well-structured, the math is correct, the narrative arc is strong, and the interactive widget works well. However, several improvement findings meaningfully weaken the lesson and should be addressed before shipping.

### Findings

### [IMPROVEMENT] — Check-Your-Understanding sections lack prediction opportunity

**Location:** Sections 4 and 8 (Check Your Understanding)
**Issue:** Both comprehension checks immediately display the full solution inline. The planning document specified "predict-and-verify" but the student never gets a moment to attempt prediction. The trace and answer are visible as soon as the student scrolls to the section. There is no reveal mechanism (e.g., a collapsed section, a "show answer" button, or even a "pause and think" prompt before the solution).
**Student impact:** The student reads the answer passively instead of actively retrieving. Predict-and-verify is one of the most effective learning techniques; showing the answer immediately converts it to a passive reading exercise. The student thinks "yeah, that makes sense" without actually testing whether they could produce the answer.
**Suggested fix:** Either (a) use a collapsible/reveal mechanism (a `<details>` element or a custom "Show Solution" button) so the student sees the problem first and must explicitly choose to reveal the answer, or (b) at minimum add a clear "Pause. Try this yourself before reading on." prompt with visual separation before the solution.

### [IMPROVEMENT] — Symbolic modality is absent

**Location:** The lesson overall, particularly section 5 (Your Network, as a Graph)
**Issue:** The planning document specified a symbolic modality: "The chain rule written alongside the graph: dL/dw1 = dL/dy_hat * dy_hat/da1 * da1/dz1 * dz1/dw1, where each term maps to a specific edge in the graph." This explicit 1:1 mapping between the algebraic chain rule expansion and graph edges does not appear anywhere in the built lesson. The lesson shows the "incoming x local" rule at individual nodes but never writes out the full chain rule expansion and maps each factor to a graph hop.
**Student impact:** A student who thinks algebraically (many engineer-types) loses a key bridge between the chain rule they know and the graph notation. The "each multiplication = one hop backward" insight is stated in the planning document but never shown with the actual symbols. The lesson teaches the graph well visually and concretely but misses the algebraic perspective.
**Suggested fix:** In section 5 (the network graph section), after showing the forward and backward values, add a short subsection showing the full chain rule expansion for one gradient (e.g., dL/dw1) with each factor labeled as "at node X" or "hop through X." Something like: dL/dw1 = (dL/dyhat) * (dyhat/da1) * (da1/dz1) * (dz1/dw1), with each term annotated to show which graph hop it corresponds to.

### [IMPROVEMENT] — Fan-out static diagram is hard to parse

**Location:** Section 6 (When Paths Split), the diagram showing f(x) = x * (x+1)
**Issue:** The static diagram for the fan-out example uses a vertical text layout with labels like "top path: x = 3 -> directly to x" and "bottom path: x = 3 -> +1 -> 4 -> to x" stacked vertically with a node bubble in the middle. This does not look like a graph. Unlike the clean horizontal diagrams in sections 3 and 5, this diagram doesn't show the actual branching structure visually. The multiply node is referenced in the text labels but not drawn as a visible node. The student has to mentally reconstruct the graph from text descriptions rather than seeing it.
**Student impact:** The fan-out concept is the ONE genuinely new piece of graph reasoning in this lesson. The diagram that introduces it should be the clearest, most visual representation in the lesson. Instead, it's the weakest diagram. The student may struggle to see the branching structure and may need to rely on the interactive widget to understand it.
**Suggested fix:** Redraw this as a proper branching graph diagram similar to the horizontal flow diagrams used elsewhere. Show x at the left, two edges branching out (one going to the multiply node directly, one going through the +1 node), both converging at the multiply node, then going to f. The branching structure should be visually obvious, not described in text.

### [IMPROVEMENT] — Misconception #4 (gradients flow same direction as forward) not explicitly addressed

**Location:** Section 3 (The Simplest Graph), should be addressed here per the plan
**Issue:** The planning document identifies misconception #4: "The graph flows in one direction (forward) so gradients must too." The plan says to address it by explicitly showing forward values flowing left-to-right and gradients flowing right-to-left on the same graph. The built lesson does use blue (forward) and red (backward) colors and mentions "left to right" and "right to left" directions, but never explicitly calls out the potential confusion: "Notice that the arrows point left-to-right, but gradients flow in the OPPOSITE direction. This is why we call it the backward pass."
**Student impact:** The color coding is present but subtle. A student who assumes gradients flow in the forward direction could misread the graph. The explicit callout would prevent this misconception from forming. Without it, the student has to infer the directional distinction from context.
**Suggested fix:** Add a brief explicit callout (1-2 sentences) in section 3 or as an aside, stating that forward values travel with the arrows (left-to-right) while gradients travel against the arrows (right-to-left). This directly addresses the misconception before the student can form it. Could be an InsightBlock or WarningBlock.

### [POLISH] — Widget hover-only interaction excludes touch devices

**Location:** Section 9 (Interactive Widget / ComputationalGraphExplorer)
**Issue:** The node detail panel in the widget is triggered only by `onMouseEnter`/`onMouseLeave` (hover). On touch devices (tablets, phones), hover is not available. The hint text says "Hover over any node to see the breakdown." Touch users cannot access this functionality.
**Student impact:** On touch devices, the student misses the "incoming gradient x local derivative = outgoing gradient" breakdown for each node, which is one of the key interactive features. The graph view still shows forward values and gradients on the nodes, so the widget is not broken, but the detail panel is inaccessible.
**Suggested fix:** Add tap-to-select behavior as a fallback: clicking/tapping a node should toggle the detail panel. Update the hint text to say "Hover or tap any node..." This is a polish issue because the step-by-step view provides equivalent information, but the graph-view detail panel is the more natural way to explore.

### [POLISH] — "upstream/downstream gradient" terminology not introduced

**Location:** Section 3 (The Simplest Graph)
**Issue:** The planning document (section 3 outline) specified: "Label the anatomy: nodes = operations, edges = values (forward) and gradients (backward), 'upstream gradient' and 'downstream gradient'." The built lesson's "Graph Anatomy" aside labels nodes, edges, and color coding, but does not introduce the upstream/downstream terminology.
**Student impact:** Minor. The lesson works fine without this terminology. However, "upstream" and "downstream" are standard terms in deep learning literature that the student will encounter later. Introducing them here with the visual context of a graph (where the direction is obvious) would be a natural fit.
**Suggested fix:** Add "upstream" and "downstream" to the Graph Anatomy aside or mention in passing: "Gradients flowing toward the input are sometimes called 'upstream gradients' in the literature."

### [POLISH] — Summary item wording could be tighter

**Location:** Section 11 (Summary), item 1
**Issue:** The first summary item reads: "A computational graph draws your computation so the chain rule becomes visual." The phrase "draws your computation" is slightly awkward. The concept being summarized is the notation itself (operations = nodes, data flow = edges).
**Student impact:** Negligible. The summary is functional and covers all key takeaways.
**Suggested fix:** Consider: "Operations become nodes, data flow becomes edges—the chain rule drawn as a graph." This is the headline already, so the description could be tightened to: "Any sequence of operations can be drawn as a graph, making gradient flow visible."

### Review Notes

**What works well:**
- The narrative arc is strong. The hook ("which would you rather debug with 50 layers?") is genuinely motivating. The before/after ComparisonRow is effective.
- The lesson correctly maintains the "same math, new view" framing throughout. The ConstraintBlock, the InsightBlocks, and repeated phrases like "same numbers" prevent the student from thinking this is a new algorithm.
- The three examples (simple, network, fan-out) progress in complexity exactly as planned, each building on the previous.
- The connection to autograd/PyTorch at the end is well-placed and appropriately brief.
- The negative example (No Path, No Gradient) naturally connects to the dying ReLU concept from the previous lesson.
- The interactive widget is well-designed with three graph modes that correspond to the three lesson examples, plus the graph/step-by-step toggle that reinforces the "same computation" message.
- Math is correct throughout (verified all forward and backward computations).
- All four planned misconceptions are at least partially addressed (though #4 could be more explicit).
- Cognitive load is appropriate: one genuinely new concept (graph notation) plus the fan-out rule, with everything else being recognition of existing knowledge.

**Pattern observation:**
The lesson's main weakness is at the "active engagement" level. The explanations and examples are excellent for reading comprehension, but the two comprehension checks are passive. For a BUILD lesson that should cement a visual framework, there should be more moments where the student must DO something (predict, trace, match) rather than just read.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 3

### Verdict: NEEDS REVISION

All 7 findings from Iteration 1 have been addressed. The lesson is substantially improved: comprehension checks now use collapsible reveals with pause prompts, the symbolic modality (chain rule hop-by-hop expansion) is present and well-executed, the fan-out diagram has proper branching structure, misconception #4 has an explicit WarningBlock ("Opposite Directions"), upstream/downstream terminology is in the Graph Anatomy aside, summary wording is tightened, and the widget supports click-to-toggle for touch devices. One new improvement finding emerged from the fresh read, plus three minor polish items.

### Findings

### [IMPROVEMENT] — Chain rule hop-by-hop expansion groups multiple graph edges into single "hops"

**Location:** Section 5 (Your Network, as a Graph), the "The chain rule, hop by hop" subsection
**Issue:** The lesson states "4 hops backward through the graph = 4 factors in the chain rule" and labels the four factors as Hop 1 through Hop 4. However, the actual graph path from L to w1 traverses 7 edges: L -> MSE -> +b2 -> xw2 -> ReLU -> +b1 -> xw1 -> w1. The "4 hops" are actually 4 semantically meaningful groups, each of which may span 2 graph edges (e.g., Hop 1 labeled "MSE -> +b2" traverses L->MSE->+b2, which is two edges). The grouping works because addition nodes have local derivative 1 and multiply through transparently. But the statement "4 hops = 4 factors" directly contradicts the lesson's own graph diagram, which shows 7 nodes in the chain. A student who carefully counts graph nodes and then reads "4 hops" will be confused about where the other nodes went.
**Student impact:** A careful student tracing the graph will count more than 4 edges and wonder why the chain rule only has 4 factors. This could undermine the "each factor = one hop" mental model, which is the core symbolic insight of this section. The student might think they're counting wrong or that some nodes "don't count" without understanding why.
**Suggested fix:** Either (a) add a brief note acknowledging that addition nodes have local derivative 1, so they pass gradients through unchanged and can be "collapsed" when writing the chain rule, or (b) reframe from "4 hops" to "4 meaningful multiplications" and note that some nodes (like +b) are pass-through steps with local derivative 1. The key point — each chain rule factor maps to a backward traversal segment — is correct and valuable, it just needs the caveat that trivial nodes (local derivative = 1) are absorbed.

### [POLISH] — Fan-out diagram x node positioning may render inconsistently

**Location:** Section 6 (When Paths Split), the branching graph diagram
**Issue:** The x input node uses `absolute left-0 top-1/2 -translate-y-1/2` positioning inside a flex container. This absolute positioning relative to the parent flex-col div may cause the x node to overlap with the path elements on narrow viewports or when the flex container collapses. The two paths (top and bottom) are in a sibling div with `ml-4`, but the x node's position is anchored to the parent, not to the flow. On very narrow screens, the x bubble could overlap the arrows or the +1 node.
**Student impact:** On narrow viewports (mobile, narrow browser), the diagram may look broken or overlapping, making the branching structure hard to parse. On wider screens it likely looks fine.
**Suggested fix:** Test the diagram at mobile widths (320px-375px). If it overlaps, consider replacing the absolute-positioned x node with a flow-based layout (e.g., put x in the same flex row as the paths with explicit spacing).

### [POLISH] — Network backward pass gradients could show the node they belong to

**Location:** Section 5 (Your Network, as a Graph), the forward values grid
**Issue:** The forward values grid shows 6 cards with labels (x, xw1, +b1, ReLU, xw2, +b2) and values. But the lesson does not show a corresponding backward gradient grid with the same labels. The backward computation is described in prose (Step 3 was dL/dw2 = ...). Having a matching backward grid showing each gradient at each node would make the forward/backward symmetry more visually obvious and let the student compare the two passes at a glance.
**Student impact:** Minor. The prose explanation is clear and the numbers are correct. But the visual symmetry between forward and backward is weaker than it could be. The forward pass gets a nice visual grid; the backward pass is text.
**Suggested fix:** Consider adding a matching gradient grid below the forward values grid (with the same 6 labels but showing gradient values in red). This would make the forward/backward duality visible at a glance. Low priority since the chain-rule expansion already visualizes the backward pass symbolically.

### [POLISH] — Widget network graph y=1 is hardcoded and not visible to student

**Location:** ComputationalGraphExplorer widget, computeNetwork function (line 606 in widget)
**Issue:** The network graph mode passes `yVal = 1` hardcoded in the useMemo call: `return computeNetwork(xVal, w1, b1, w2, b2, 1)`. The target value y=1 is not shown in the widget UI or adjustable by the student. The student knows from the lesson text that y=1, but the widget doesn't display this anywhere. The "Adjust network parameters" collapsible shows w1, b1, w2, b2 but not y.
**Student impact:** Very minor. The student could adjust x and weights but not y, and there's no label showing what y is. If the student wonders "what target is the MSE computing against?", they'd need to look at the lesson text. Not a real problem since y=1 is stated in the lesson.
**Suggested fix:** Either display y=1 as a static label in the parameter controls (e.g., "y (target) = 1") or add it as an adjustable slider. Displaying it is simpler and sufficient.

### Review Notes

**What works well:**
- All Iteration 1 findings have been addressed effectively. The collapsible check-your-understanding sections with "Pause" prompts are a clear improvement over inline answers. The symbolic chain-rule expansion with underbraces mapping to graph hops is an excellent addition that completes the modality coverage.
- The "Opposite Directions" WarningBlock in section 3 directly and concisely addresses misconception #4 (gradients flow same direction as forward). Good placement next to the Graph Anatomy aside.
- The fan-out diagram now has a visual branching structure showing x at the left, two diverging paths, and convergence at the multiply node. The caption "x splits into two paths that both converge at the x node" helps.
- The widget's click-to-toggle for node selection (`onClick` that toggles hovered state) addresses the touch device accessibility concern. The hint text says "Hover or tap" which is correct.
- Upstream/downstream terminology is now in the Graph Anatomy aside, naturally placed alongside the other anatomical labels.
- The summary items are tighter and more concrete. "Operations become nodes, data flow becomes edges" is a strong headline.
- Math is correct throughout. Forward and backward computations verified for all three examples (simple, network, fan-out) and both comprehension checks.
- Modality coverage is now strong: visual (3 graph diagrams), concrete (same network numbers from lesson 2), symbolic (chain rule expansion with hop labels), interactive (widget with 3 modes), intuitive ("the graph IS the chain rule, drawn instead of written" appears in the WarningBlock and throughout).
- The lesson stays within scope boundaries: no code, no batching, no optimizers, no matrix calculus.

**Pattern observation:**
The one remaining improvement finding (hop-by-hop grouping) is a precision issue in the symbolic modality, not a structural problem. The lesson is close to shipping quality. Addressing it would require only a sentence or two of additional text acknowledging that addition nodes are pass-through steps.
