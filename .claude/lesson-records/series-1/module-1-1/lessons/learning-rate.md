# Lesson: Learning Rate Deep Dive

**Module:** 1.1 — The Learning Problem
**Slug:** learning-rate
**Position:** Lesson 5 of 6
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Generalization vs memorization | INTRODUCED | what-is-learning |
| Bias-variance tradeoff (intuitive) | INTRODUCED | what-is-learning |
| Train/val/test splits | INTRODUCED | what-is-learning |
| Linear model (y-hat = wx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| Residuals (y - y-hat) | DEVELOPED | loss-functions |
| MSE loss function | DEVELOPED | loss-functions |
| Loss landscape as a bowl | INTRODUCED | loss-functions |
| Gradient = direction of steepest ascent | DEVELOPED | gradient-descent |
| Gradient descent update rule | DEVELOPED | gradient-descent |
| Learning rate as step size | INTRODUCED | gradient-descent |

Mental models: "Ball rolling downhill." "Subtract the gradient to go downhill." Student has used the GradientDescentExplorer with a learning rate slider and seen that LR matters — this lesson goes deeper.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to develop deep intuition for how the learning rate affects training — the failure modes of too-big and too-small, and the concept of learning rate schedules.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent | OK | Saw formula and interacted with widget |
| Learning rate concept | INTRODUCED | INTRODUCED | gradient-descent | OK | This lesson deepens it |
| Loss surface shape (convex bowl) | INTRODUCED | INTRODUCED | loss-functions | OK | Sufficient for understanding oscillation |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| One learning rate works for all problems | Want a default "safe" value | "No universal answer" — depends on problem | Section 1: InsightBlock |
| Smaller is always safer | "At least it won't diverge" | Stuck in shallow local minima, wastes compute | Section 2: Too Small |
| Loss going up = bug in code | First instinct when loss diverges | It's usually just LR too high | Section 3: WarningBlock |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| LearningRateExplorer comparison mode | Interactive positive | Side-by-side of 4 different LR values |
| LearningRateExplorer interactive mode | Interactive positive | Student picks their own LR |
| Oscillation vs Divergence (ComparisonRow) | Positive pair | Two distinct failure modes of too-large LR |
| Specific LR values (0.05, 0.5, 0.9, 1.1) | Concrete | Gives precise numbers to try |
| Fine-tuning pretrained models | Positive (preview) | When small LR is intentional |
| Learning rate schedule types | Positive (preview) | Four common patterns |

---

## Phase 3: Design

### Narrative Arc

This lesson zooms in on the learning rate, the one knob previewed in the gradient descent lesson. It's structured as an exploration of failure modes: what goes wrong when LR is too small (Section 2), what goes wrong when LR is too large (Section 3), with interactive exploration bookending both. The lesson ends with a preview of learning rate schedules — the practical solution used in modern training. The structure is: play with it first (comparison widget), understand the extremes (theory), play with it again (interactive widget), see the real-world solution (schedules).

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Interactive | LearningRateExplorer comparison mode (4 LRs side-by-side) | Direct visual comparison of behaviors |
| Interactive | LearningRateExplorer interactive mode (choose your own LR) | Hands-on experimentation |
| Symbolic | Update rule repeated with alpha highlighted | Reinforces the formula |
| Visual | ComparisonRow (oscillation vs divergence) | Side-by-side failure modes |
| Verbal | "Goldilocks problem" framing | Connects to familiar concept |

### Cognitive Load

- **New concepts:** 2 (hyperparameters as a concept class, learning rate schedules as preview)
- **Previous lesson load:** STRETCH
- **This lesson:** BUILD — deepens an existing concept (learning rate) rather than introducing something entirely new
- **Assessment:** Appropriate — this is consolidation, not new territory

### Scope Boundaries

- 1D still — not multi-parameter learning rates
- Schedules are previewed, NOT taught (deferred to Module 1.7)
- No implementation / code
- Does NOT cover: adaptive learning rates (Adam, etc.), per-parameter learning rates
- Fine-tuning mention is just a teaser, not explained

---

## What Was Actually Built

1. **Header + Objective** — "most important hyperparameter"
2. **Hyperparameter definition** — TipBlock: "a value YOU choose, not one the model learns"
3. **Goldilocks Problem** — Update rule repeated, too small / too large preview
4. **LearningRateExplorer comparison** — 4 side-by-side (0.05, 0.5, 0.9, 1.1)
5. **Too Small: Slow Convergence** — Symptoms + "when small is OK" (fine-tuning teaser)
6. **Too Large: Oscillation and Divergence** — ComparisonRow of two failure modes
7. **LearningRateExplorer interactive** — Student picks their own LR
8. **Learning Rate Schedules (Preview)** — Step decay, exponential, cosine annealing, warmup
9. **Summary** — 4 key takeaways
10. **Next Step** — Link to implementing-linear-regression

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (2), TipBlock (1), WarningBlock (1), TryThisBlock (2), ConceptBlock (1)
- ComparisonRow (1), SummaryBlock, NextStepBlock
- LearningRateExplorer via ExercisePanel (2 instances: comparison + interactive)
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Hyperparameter = you choose it** — learning rate is not learned, it's set
- **Goldilocks zone** — there's a narrow range that works, specific to each problem
- **Oscillation ≠ divergence** — two distinct failure modes of too-large LR
- **Loss shooting up = LR too high** — practical debugging heuristic
- **Schedules = start big, decay small** — preview of modern practice

### Analogies Used
- Goldilocks: too big, too small, just right
- "Cure worse than the disease" for too-large LR
- Step decay, exponential decay metaphors for schedules

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

### [CRITICAL] — "Local minima" reference assumes untaught concept

**Location:** Section 2: Too Small, bullet list (line 131)
**Issue:** The text says students might get "stuck in shallow local minima." The student has only seen convex (bowl-shaped) loss surfaces with a single minimum. Local minima — multiple valleys in a non-convex surface — have not been introduced. This creates confusion: "Wait, I thought there was only one minimum at the bottom of the bowl?"
**Student impact:** The student either gets confused about what local minima are (they don't know this term), or worse, starts doubting their understanding of the loss landscape from the previous lessons.
**Suggested fix:** Remove the local minima reference. Replace with something the student CAN understand at this point — e.g., "Might take so long that you give up and think the model isn't working" or "Wastes time and compute resources." If local minima need to be previewed, it should be done in a clearly labeled aside, not buried in a symptom list.

### [IMPROVEMENT] — Weak motivation at lesson opening

**Location:** Header + Objective (lines 37-62) and Section 1 opening (lines 64-95)
**Issue:** The lesson goes straight from objective to explanation. There's no motivating question or tension-building opener. The student doesn't feel a problem that needs solving — it's "here's something important" rather than "here's why you should care right now." The gradient descent lesson already showed them that LR matters; this lesson should capitalize on that experience.
**Student impact:** The student engages less deeply because there's no curiosity hook. They'll read it, but they won't be leaning forward.
**Suggested fix:** Add a brief motivating hook after the objective — something like: "In the last lesson, you might have noticed that some learning rate values worked and others didn't. But why? And how would you choose a good one for a new problem? That's what we'll figure out." This connects to their prior experience and creates a question to answer.

### [IMPROVEMENT] — No brief recap of update rule variables

**Location:** Section 1: The Goldilocks Problem (lines 71-87)
**Issue:** The update rule formula θ_new = θ_old − α∇L is presented, but there's no reminder of what θ and ∇L represent. The student learned these one lesson ago, but with ADHD and potentially a gap between sessions, a one-line reminder helps activate prior knowledge.
**Student impact:** A student returning after a few days might need to pause and try to remember "what was ∇L again?" This breaks flow.
**Suggested fix:** Add a brief inline reminder after the formula — e.g., "Remember: θ is our parameter, ∇L is the gradient (slope of the loss), and α is the learning rate — the part we're focusing on today."

### [POLISH] — Em dash spacing violation

**Location:** LessonHeader description (line 41)
**Issue:** "the most important hyperparameter in gradient descent — the learning rate" uses spaces around the em dash. Style rule requires no spaces: "gradient descent—the learning rate."
**Student impact:** None (visual only).
**Suggested fix:** Remove spaces around the em dash.

### [POLISH] — Preset buttons in widget lack cursor-pointer

**Location:** LearningRateExplorer widget, preset buttons (line 241)
**Issue:** The preset buttons (0.1, 0.5, 0.9, 1.05) are `<button>` elements without explicit `cursor-pointer` styling. Most browsers show `cursor: default` on buttons, which makes them feel less interactive despite having hover states.
**Student impact:** Minor — the hover state change partially compensates, but the cursor not changing is a small usability miss.
**Suggested fix:** Add `cursor-pointer` to the preset button className.

### Review Notes

The lesson is well-structured with a strong explore-first → explain → explore-again arc. The two widget modes (comparison + interactive) give both guided and open exploration. The scope boundaries are well-maintained — schedules are appropriately previewed without overteaching. The ComparisonRow for oscillation vs divergence is effective at drawing a clear boundary between two failure modes.

The critical finding (local minima reference) is the most important fix — it introduces a concept the student hasn't encountered and could undermine their mental model of loss surfaces. The motivation improvement would meaningfully increase engagement. The recap improvement follows ADHD-friendly design principles (don't assume the student remembers everything from last session).

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS with notes

### Findings

### [POLISH] — Multiple em dash spacing violations remain

**Location:** Lines 51, 57, 97, 146, 153, 303
**Issue:** Six instances of `word — word` (spaces around em dash) remain throughout the lesson prose. The first iteration only fixed the one in the LessonHeader description. Style rule requires `word—word`.
**Student impact:** None (visual style consistency only).
**Suggested fix:** Remove spaces around all em dashes in the lesson.

### Review Notes

The iteration 1 fixes landed well:
- The motivation hook (lines 72-77) effectively connects to the student's prior experience with the GradientDescentExplorer and creates genuine curiosity.
- The formula recap (lines 85-90) is well-sized (text-sm) and non-patronizing, good for ADHD-friendly re-entry.
- The local minima reference was cleanly replaced with "Might take so long you think the model isn't working" — grounded in the student's experience rather than untaught theory.
- The preset button cursor-pointer fix is appropriate.

No critical or improvement findings remain. The only remaining issue is em dash styling consistency, which is purely cosmetic.

---

## Improvement Summary — 2026-02-06

**Iterations:** 2/3
**Final verdict:** PASS (all findings resolved)

### Changes made:
- [Iteration 1] Fixed CRITICAL: Removed "local minima" reference (untaught concept) — replaced with "Might take so long you think the model isn't working"
- [Iteration 1] Fixed IMPROVEMENT: Added motivation hook connecting to student's prior GradientDescentExplorer experience
- [Iteration 1] Fixed IMPROVEMENT: Added brief formula recap (θ, ∇L, α) for ADHD-friendly re-entry
- [Iteration 1] Fixed POLISH: Em dash spacing in LessonHeader description
- [Iteration 1] Fixed POLISH: Added cursor-pointer to LearningRateExplorer preset buttons
- [Iteration 2] Fixed POLISH: Remaining 6 em dash spacing violations throughout lesson prose

### Remaining items:
- None
