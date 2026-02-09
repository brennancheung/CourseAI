# Lesson: Implementing Linear Regression

**Module:** 1.1 — The Learning Problem
**Slug:** implementing-linear-regression
**Position:** Lesson 6 of 6 (module capstone)
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Generalization vs memorization | INTRODUCED | what-is-learning |
| Linear model (y-hat = wx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| ML notation (y-hat, w, b) | INTRODUCED | linear-regression |
| Residuals (y - y-hat) | DEVELOPED | loss-functions |
| MSE loss function | DEVELOPED | loss-functions |
| Loss landscape as a bowl | INTRODUCED | loss-functions |
| Gradient = slope, direction of steepest ascent | DEVELOPED | gradient-descent |
| Gradient descent update rule | DEVELOPED | gradient-descent |
| Learning rate as step size | DEVELOPED | learning-rate |
| Hyperparameters (LR) | INTRODUCED | learning-rate |
| LR failure modes (oscillation, divergence) | DEVELOPED | learning-rate |
| LR schedules (preview) | MENTIONED | learning-rate |

Mental models: All the pieces are in place — model, loss, optimization, learning rate. The student has interacted with each concept separately. This lesson brings them together in code.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand the complete training loop structure and implement linear regression from scratch in Python/NumPy.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Linear model y-hat = wx + b | DEVELOPED | DEVELOPED | linear-regression | OK | Needs to translate to code |
| MSE loss function formula | DEVELOPED | DEVELOPED | loss-functions | OK | Needs to implement |
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent | OK | Needs to implement |
| Learning rate | DEVELOPED | DEVELOPED | learning-rate | OK | Used as hyperparameter in code |
| Python + NumPy | DEVELOPED | APPLIED | Prior experience | OK | Software engineer |
| Partial derivatives (chain rule) | INTRODUCED | INTRODUCED | Prior education | OK | Gradient formulas are given, not derived |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Training loop is complex/magical | ML seems hard | The entire loop is ~10 lines of NumPy | Section 3: The Code |
| Gradient signs don't matter | "Just take the absolute value" | Wrong signs make the model diverge | Section 3: WarningBlock |
| This pattern is specific to linear regression | "Neural networks must be totally different" | "Every neural network follows this exact pattern" | Section 4: InsightBlock |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| TrainingLoopExplorer widget | Interactive positive | Watch loss decrease, parameters converge, line fit |
| 6-step training loop (StepList) | Positive | Universal pattern for all ML training |
| Gradient formulas (4 ConceptBlocks) | Positive | Complete math for implementation |
| Python code implementation | Positive | ~15 lines of actual runnable code |
| Colab notebook link | Positive (practice) | Hands-on implementation exercise |

---

## Phase 3: Design

### Narrative Arc

This is the capstone lesson of Module 1.1 — the moment where all the theory comes together into working code. The lesson doesn't introduce fundamentally new concepts; instead, it connects and integrates everything from lessons 1-5. The structure is: see it work (widget), understand the pattern (training loop), understand the math (gradient formulas), see the code (implementation), then try it yourself (Colab notebook). The payoff is watching a model actually learn — loss dropping, line fitting, parameters converging.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Interactive | TrainingLoopExplorer widget | See the training loop execute visually |
| Symbolic | Gradient formulas (4 KaTeX blocks) | Precise math for implementation |
| Code | Python/NumPy implementation | The actual implementation, line by line |
| Concrete | Specific observations to watch for (loss curve, parameter convergence) | Grounds the abstract in specific visual feedback |
| Practice | Colab notebook exercise | Hands-on implementation |

### Cognitive Load

- **New concepts:** 2 (training loop structure as a pattern, gradient computation formulas for w and b)
- **Previous lesson load:** BUILD
- **This lesson:** CONSOLIDATE — integrates existing concepts into implementation
- **Assessment:** Low new-concept count. The heavy lift is integration, not novelty.

### Scope Boundaries

- Pure Python + NumPy only — no sklearn or PyTorch
- Gradients are given (derived from chain rule), not derived step-by-step
- Single-feature linear regression only
- Does NOT cover: vectorized multi-feature regression, batching, stopping criteria
- Chain rule is mentioned in TipBlock but NOT taught — deferred to Module 1.3

---

## What Was Actually Built

1. **Header + Objective** — "implement from scratch, understand every line"
2. **TrainingLoopExplorer widget** — Interactive visualization of training (loss curve + line fitting)
3. **Training Loop (StepList)** — 6-step universal pattern
4. **Computing Gradients** — 4 ConceptBlocks: model, loss, dL/dw, dL/db with KaTeX
5. **The Code** — CodeBlock with ~15 lines of Python/NumPy
6. **What You'll Observe** — 4 specific things to watch for in the widget
7. **Try It Yourself** — Colab notebook link with exercise description
8. **Summary** — 4 key takeaways
9. **ModuleCompleteBlock** — Lists 7 achievements from Module 1.1
10. **Next Step** — Link to Module 1.2

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (3), TipBlock (2), WarningBlock (1), ConceptBlock (4)
- StepList (1), SummaryBlock, NextStepBlock, ModuleCompleteBlock
- TrainingLoopExplorer via ExercisePanel
- CodeBlock (1 — Python)
- KaTeX: BlockMath
- ExternalLink icon (Colab link)

### Mental Models Established
- **Training loop = forward → loss → backward → update → repeat** — THE universal pattern
- **Gradient formulas for linear regression** — dL/dw and dL/db with the -2 factor
- **"That's deep learning"** — every model from linear regression to GPT follows this pattern
- **Implementation is surprisingly simple** — ~10 lines of core code

### Analogies Used
- "Heartbeat of neural network training" for the training loop
- Forward/backward pass terminology (introduced, becomes standard)

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

The lesson has strong content and appropriate scope, but structural issues weaken the student experience. The widget placement before any explanation is the most impactful issue.

### Findings

#### [CRITICAL] — Widget appears before the training loop concept is explained

**Location:** Lines 67-80 (TrainingLoopExplorer, immediately after ObjectiveBlock)
**Issue:** The TrainingLoopExplorer widget is the first content after the objective. The student sees Train/Step/Reset buttons, a learning rate slider, parameter stats, and an update formula — but they haven't been told what a "training loop" is, what the steps are, or what they should be watching for. The aside says "This is what happens inside the training loop" but the student doesn't know what a training loop IS yet.
**Student impact:** The student clicks buttons and sees numbers changing without understanding what's happening. This wastes the most powerful learning moment — watching a model learn should come AFTER the student understands the process, so they can actively predict what will happen and verify their understanding. Instead, it's just an animation of unexplained numbers.
**Suggested fix:** Move the widget to AFTER the training loop explanation and gradient formulas. The flow should be: explain the pattern → show the math → see it in action (widget) → see the code → try it yourself. The "What You'll Observe" section already exists and should immediately precede or follow the widget. Alternatively, keep the widget early but add a brief "Watch this first, then we'll explain what's happening" framing, and move the "What You'll Observe" bullets to the aside next to the widget.

#### [IMPROVEMENT] — No motivating hook for the capstone

**Location:** Lines 38-65 (Header + Objective)
**Issue:** The lesson jumps directly from objective to widget. There's no payoff moment — "You've learned all the pieces. Now let's put them together and watch a model actually learn." A capstone lesson should feel like arrival, not just another lesson.
**Student impact:** The student doesn't feel the cumulative weight of lessons 1-5 coming together. The excitement of "I can build this NOW" is missed.
**Suggested fix:** Add a brief motivating section after the objective: "Over the last five lessons, you've learned the pieces: a model (ŷ = wx + b), a way to measure how wrong it is (MSE loss), a way to improve (gradient descent), and how to control the improvement speed (learning rate). Now we'll wire them all together and watch a model learn from scratch."

#### [IMPROVEMENT] — No step-by-step walkthrough of one training iteration

**Location:** Between gradient formulas (Section 2) and code (Section 3)
**Issue:** The lesson presents gradient formulas in 4 ConceptBlocks, then shows the complete code. There's no intermediate step where the student walks through one concrete iteration with actual numbers: "Start with w=0, b=0. Forward pass: all predictions are 0. Loss: average of y². Gradient dw = ... . Update: w_new = 0 - 0.01 × (-5.2) = 0.052." This bridges the gap between abstract formulas and the code.
**Student impact:** The student sees formulas, then sees code, but never manually traces through one iteration. The connection between formula and implementation is implicit, not explicit. A software engineer would particularly benefit from tracing through one iteration.
**Suggested fix:** Add a worked example between the gradient formulas and the code: "Let's trace through one iteration..." with concrete numbers. This is the "concrete before abstract" ordering principle — show what one iteration DOES before showing the code that does it in a loop.

#### [IMPROVEMENT] — Partial derivative notation (∂) introduced without explanation

**Location:** Lines 140-148 (ConceptBlocks for dL/dw and dL/db)
**Issue:** The gradient formulas use ∂L/∂w notation. The student learned gradients using ∇ (nabla) notation and "slope of the loss." Partial derivative notation is new. The lesson doesn't bridge between "gradient = slope" (what they know) and "∂L/∂w = how loss changes when weight changes" (what they see).
**Student impact:** A software engineer who hasn't done calculus recently may not recognize ∂ or understand its relationship to the gradient concept they already know. This is a notation gap, not a concept gap — but notation gaps create confusion.
**Suggested fix:** Add a brief bridge: "The symbol ∂L/∂w means 'how much does the loss change when we change w?' — it's the gradient for just one parameter. You already know this idea — it's the slope of the loss curve from lesson 4."

#### [POLISH] — "What You'll Observe" section references widget that requires scrolling back up

**Location:** Lines 212-252 (Section 4)
**Issue:** The section says "In the interactive explorer above, watch for:" but the widget is several scroll-lengths above. The student has to scroll back up to the widget, find the observations, scroll back down to read each one, then back up to verify.
**Student impact:** Minor friction. The student either ignores the instructions (and misses the guided observation) or deals with scroll friction.
**Suggested fix:** If the widget moves per the critical finding above, this is resolved. If the widget stays at the top, consider moving these bullets into the TryThisBlock aside next to the widget.

#### [POLISH] — Em dash with spaces in "That's Deep Learning" aside

**Location:** Line 249 (InsightBlock aside)
**Issue:** "from linear regression to GPT — follows" has spaces around the em dash.
**Student impact:** None (visual only).
**Suggested fix:** Change to "from linear regression to GPT—follows"

### Review Notes

The lesson's content is sound — it has the right concepts at the right depth for a capstone. The training loop StepList, gradient formulas, code, and widget are all well-crafted individually. The primary structural issue is sequencing: the widget appears before the student has context to understand it, which wastes the most powerful learning moment. Moving the widget to after the explanation would create a proper "now watch it in action" payoff.

The lack of a worked example (one traced iteration with numbers) is a significant gap for a software-engineer student who thinks in concrete terms. The partial derivative notation gap is smaller but could cause unnecessary confusion.

The motivating hook is important for ADHD engagement — a capstone should feel like arrival, not just another section.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 1
- Improvement: 0
- Polish: 1

### Verdict: NEEDS REVISION

All iteration 1 findings are resolved. However, the new worked example contains a math error that would teach the student incorrect computations.

### Iteration 1 Fixes Verified
- [CRITICAL] Widget placement — FIXED. Widget now appears after training loop, gradients, worked example, and code.
- [IMPROVEMENT] Motivating hook — FIXED. Added recap of five pieces + "wire them together" framing.
- [IMPROVEMENT] Worked example — FIXED. Tracing one iteration added with concrete numbers. (But math is wrong — see finding below.)
- [IMPROVEMENT] Partial derivative notation — FIXED. Bridge paragraph explains ∂L/∂w as "slope per parameter."
- [POLISH] Widget scroll friction — FIXED. "What to Watch For" is now in the aside next to the widget.
- [POLISH] Em dash — FIXED. "from linear regression to GPT—follows" (no spaces).

### New Findings

#### [CRITICAL] — Math error in worked example

**Location:** Lines 191-228 ("Tracing One Iteration" section)
**Issue:** The gradient computation for w is incorrect. With data x=[1,2,3], y=[3,5,7], w=0, b=0:
- dL/dw = (1/3) × (-2×1×3 + -2×2×5 + -2×3×7) = (1/3) × (-6 - 20 - 42) = -68/3 = **-22.67**, not -30.67
- Consequently, w_new = 0 - 0.1 × (-22.67) = **2.267**, not 3.067
**Student impact:** A student who traces through the computation themselves (which this section is explicitly designed to encourage) will get a different answer. This either makes them doubt their own math skills or lose trust in the lesson. Both are catastrophic for a capstone lesson.
**Suggested fix:** Correct the gradient: -22.67, and the update: w = 2.267.

#### [POLISH] — Em dash with spaces in LessonHeader description

**Location:** Line 44
**Issue:** "Put it all together — implement" uses spaces around the em dash.
**Student impact:** None (visual only).
**Suggested fix:** Change to "Put it all together—implement"

### Review Notes

The restructured lesson flows much better. The motivating hook → training loop → gradients → worked example → code → widget progression is logical and builds understanding incrementally. The worked example is a significant pedagogical addition — tracing through one iteration with numbers bridges the formula→code gap perfectly. The ∂ notation bridge is clean and non-patronizing. The widget placement is now well-motivated ("Now that you understand every piece, watch the full training loop in action").

The math error is the only blocking issue. Once corrected, this lesson will be substantially improved from the original.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

All findings from iterations 1 and 2 are resolved. The lesson is well-structured, mathematically correct, and provides an effective capstone experience.

### Iteration 2 Fixes Verified
- [CRITICAL] Math error in worked example — FIXED. Gradient corrected from -30.67 to -22.67, update from 3.067 to 2.267. Verified by manual computation: (1/3)(-6-20-42) = -22.67. The aside TipBlock was also updated to match.
- [POLISH] Em dash in LessonHeader — FIXED. Also found and fixed one additional em dash with spaces in ObjectiveBlock.

### Review Notes

The lesson now has a strong pedagogical arc:
1. Motivating hook reconnects all 5 prior lessons
2. Training loop as a 6-step universal pattern
3. Gradient formulas with ∂ notation properly bridged to prior knowledge
4. Worked example traces one iteration with verified math
5. Code maps directly to the steps the student just traced
6. Widget is properly contextualized ("now that you understand every piece")
7. Post-widget insight connects to GPT-scale models
8. Colab exercise for hands-on practice
9. Strong summary with concrete mental models

The student experience is well-scaffolded: each section builds on the previous one, and no section assumes knowledge that hasn't been established. The capstone feeling is achieved through the motivating hook and the ModuleCompleteBlock.

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS

### Changes made:
- **[Iteration 1]** Restructured lesson flow: moved widget from top to after explanations; added motivating hook recapping all 5 prior lessons; added "Tracing One Iteration" worked example with concrete numbers; bridged ∂ notation to prior "slope of loss" mental model; moved "What to Watch For" bullets into aside next to widget; fixed em dash spacing; improved summary takeaways
- **[Iteration 2]** Fixed math error in worked example (gradient -30.67 → -22.67, update 3.067 → 2.267); fixed remaining em dash spacing violations

### Remaining items:
- None
