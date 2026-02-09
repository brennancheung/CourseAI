# Lesson: The Limits of Linearity

**Module:** 1.2 — From Linear to Neural
**Slug:** limits-of-linearity
**Position:** Lesson 2 of 4
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| Linear model (y-hat = wx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| Neuron = weighted sum + bias | DEVELOPED | neuron-basics |
| Layer = group of neurons | INTRODUCED | neuron-basics |
| Network = stacked layers | INTRODUCED | neuron-basics |
| Linear collapse (stacking linear = still linear) | DEVELOPED | neuron-basics |
| Gradient descent / training loop | DEVELOPED | gradient-descent, implementing-linear-regression |

Mental models: "A neuron is just multi-input linear regression." "100-layer linear network = 1-layer linear network." Student knows the architecture but has been told (not shown) that linearity is a problem. This lesson makes that concrete.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student that linear decision boundaries cannot separate XOR-like patterns, demonstrating the fundamental limitation that motivates activation functions.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Neuron = weighted sum + bias | DEVELOPED | DEVELOPED | neuron-basics | OK | Need to understand what's being limited |
| Linear collapse theorem | INTRODUCED | INTRODUCED | neuron-basics | OK | The theoretical basis for why stacking doesn't help |
| Linear model equation | DEVELOPED | DEVELOPED | linear-regression | OK | The line they'll try to fit |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| "Just need to try harder / find better parameters" | Optimization mindset from Module 1.1 | XOR widget — try every angle, none works | Section: XOR Challenge + "Why It's Impossible" |
| "Linear can handle any classification" | Linear regression worked well on smooth data | XOR truth table — 4 points, impossible to separate | Section: Why It's Impossible |
| "XOR is an unrealistic edge case" | Seems like a toy problem | Real-world examples: images, language, science | Section: Why This Matters |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| XOR truth table | Positive | Clean definition of the target function |
| XOR plotted as 4 corners | Positive | Visual representation, diagonal pattern |
| XORClassifierExplorer widget | Interactive negative | Student tries and fails to draw separating line |
| AND/OR as linearly separable | Positive (contrast) | Shows what linearly separable looks like |
| Real-world nonlinear examples (images, language, science) | Positive | Generalizes beyond toy problem |
| "Two neurons, two lines" hint | Positive (preview) | Foreshadows the solution |

---

## Phase 3: Design

### Narrative Arc

This lesson is the "problem" that motivates the "solution" (activation functions in the next lesson). It follows the Motivation Rule: make the student feel the need before giving the answer. The structure is: introduce XOR → let them try to solve it → prove it's impossible → explain why → show that the limitation matters for real problems → preview the solution. The interactive widget is crucial — the student should spend a minute trying and failing before being told it's impossible.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Concrete | XOR truth table | Precise definition before visualization |
| Visual | 4 corners plotted, diagonal pattern | Spatial understanding of why separation fails |
| Interactive | XORClassifierExplorer | Student experiences the impossibility firsthand |
| Visual | ComparisonRow (linearly separable vs not) | Side-by-side clarity |
| Verbal | Real-world examples (images, language) | Connects abstract limitation to practical relevance |

### Cognitive Load

- **New concepts:** 2 (linear separability, decision boundary)
- **Previous lesson load:** STRETCH
- **This lesson:** BUILD — applies the linear collapse insight from the previous lesson to a concrete problem
- **Assessment:** Low novelty — mostly demonstrating a limitation, not introducing new tools

### Scope Boundaries

- XOR is the only non-linearly-separable example shown in detail
- Decision boundaries are 2D only (hyperplanes mentioned in aside, not taught)
- Does NOT explain how to solve XOR (next lesson)
- Activation functions are previewed but NOT taught
- No math beyond the truth table — no attempt at formal proof
- "Universal function approximation" mentioned but NOT explained

---

## What Was Actually Built

1. **Header + Objective** — "see where linear networks break down"
2. **XOR: A Simple Function** — Truth table with color-coded outputs
3. **Plotting XOR** — Axes explanation, classification framing
4. **The XOR Challenge** — Setup text + instructions
5. **XORClassifierExplorer widget** — Interactive: adjust slope/intercept, try to separate
6. **Why It's Impossible** — Diagonal pattern explanation
7. **Linear Decision Boundaries** — ComparisonRow (linearly separable vs not)
8. **Why This Matters** — Real-world examples: images, language, science
9. **The Missing Ingredient** — Preview: activation functions break linearity, "two neurons, two lines" hint
10. **Summary** — 4 key takeaways
11. **Next Step** — Link to activation-functions

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (3), TipBlock (2), WarningBlock (1), TryThisBlock (1), ConceptBlock (2)
- ComparisonRow (1), SummaryBlock, NextStepBlock
- XORClassifierExplorer via ExercisePanel
- Custom HTML table for truth table

### Mental Models Established
- **Linear separability** — can a single straight line divide two classes?
- **Decision boundary = the line between classes** — one side = class 0, other = class 1
- **XOR = opposite diagonals** — same-class points on diagonals can't be separated by one line
- **"Not about better parameters — it's mathematically impossible"** — key insight

### Analogies Used
- "One straight cut" for linear decision boundaries
- Diagonal/checkerboard pattern for XOR layout
- "Two neurons, two lines" hint for the solution

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — Spoiler WarningBlock undermines discovery moment

**Location:** Row containing XORClassifierExplorer (lines 189-202), specifically WarningBlock "Spoiler Alert" in the aside
**Issue:** The WarningBlock saying "You can't do it. No matter how you position the line, at least one point will always be on the wrong side." is visible on screen right next to the interactive widget. The planning document emphasizes: "The interactive widget is crucial — the student should spend a minute trying and failing before being told it's impossible." The spoiler being immediately visible alongside the widget undermines the entire pedagogical design of the discovery moment.
**Student impact:** Student reads the spoiler before or while attempting the challenge, removing the "aha" moment of personal failure. The frustration of trying and failing IS the learning. Pre-empting it robs the lesson of its emotional core.
**Suggested fix:** Remove the WarningBlock from the widget's aside. Instead, the widget itself already has a progressive hint system (after 10 attempts, it shows a message). Let the "Why It's Impossible" section serve as the reveal. If a hint is desired near the widget, use the TryThisBlock that's already there (in the row above) to encourage trying harder, not to spoil the answer.

### [IMPROVEMENT] — AND/OR linearly separable contrast is told, not shown

**Location:** "Linear Decision Boundaries" section (lines 239-281), ComparisonRow
**Issue:** The ComparisonRow mentions AND and OR as examples of linearly separable functions, but this is purely text. The student has never seen AND/OR plotted. For a lesson whose core insight is visual (XOR points on diagonals can't be separated), the contrast case should also be visual. The student should SEE what linear separability looks like before being told XOR doesn't have it.
**Student impact:** The student has a strong visual understanding of why XOR fails (from the interactive) but only an abstract understanding of what success looks like. The comparison is weaker than it should be because one side is visual/experiential and the other is just words.
**Suggested fix:** This could be addressed with a simple static visual or by adding a brief callout that explicitly describes the AND case visually (e.g., "For AND, the three (0,0), (0,1), (1,0) points output 0 and only (1,1) outputs 1 — a single line easily separates them"). A full widget isn't needed, just enough visual or concrete detail to make the contrast tangible.

### [POLISH] — Em dash spacing inconsistency

**Location:** LessonHeader description (line 44)
**Issue:** `"See why the neural network structure alone isn't enough — linear networks fail on simple problems like XOR."` — The em dash has spaces around it. The writing style rule requires no spaces: `word—word`.
**Student impact:** Minimal — visual consistency only.
**Suggested fix:** Change to `isn't enough—linear networks fail`.

### [POLISH] — Linear collapse depth inconsistency in planning document

**Location:** Planning document Phase 1 table (line 19)
**Issue:** The planning document lists "Linear collapse" as INTRODUCED from neuron-basics, but the module 1.2 record.md lists it as DEVELOPED (line 15: "Linear collapse | DEVELOPED | neuron-basics"). This is a planning document inconsistency, not a lesson issue.
**Student impact:** None on the lesson itself, but creates confusion for future lesson planning.
**Suggested fix:** Update the planning document to match the module record (DEVELOPED).

### Review Notes

The lesson is well-structured overall. The narrative arc — introduce XOR, let the student try, reveal impossibility, generalize — is sound. The XORClassifierExplorer widget is well-built with good progressive feedback. The main weakness is the spoiler undermining the discovery moment, which is the emotional heart of the lesson. The secondary weakness is the missed opportunity for a visual contrast with linearly separable data. Both are fixable without restructuring the lesson.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: NEEDS REVISION

### Findings

### [POLISH] — Multiple remaining spaced em dashes

**Location:** Multiple locations throughout the lesson
**Issue:** Several em dashes still have spaces around them, violating the writing style rule:
- ObjectiveBlock (line 53): `break down — and understand`
- ConceptBlock "Higher Dimensions" (line 276): `the same — it can only`
- InsightBlock "The Motivation" (line 317): `boundary — including`
- SectionHeader subtitle (line 327): `the structure — we need`
**Student impact:** Minimal — visual consistency.
**Suggested fix:** Replace all spaced em dashes with unspaced em dashes throughout the lesson.

### Review Notes

The iteration 1 fixes are effective. The spoiler removal improves the interactive experience. The AND contrast paragraph makes the ComparisonRow section much more concrete. The only remaining issue is the em dash styling, which is purely cosmetic. The lesson is pedagogically sound.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

### Findings

No findings. All previous issues have been resolved.

### Review Notes

The lesson is now in good shape. The narrative arc is strong: introduce XOR concretely, let the student try and fail interactively, reveal why it's impossible, generalize to real-world problems, preview the solution. The discovery moment around the interactive widget is preserved (no spoiler). The AND contrast provides concrete grounding for the linearly-separable vs non-linearly-separable distinction. Em dashes are consistently formatted. The lesson stays within scope and builds effectively on the previous lesson's linear collapse insight.

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS

### Changes made:
- [Iteration 1] Removed spoiler WarningBlock from XOR widget aside (was undermining discovery moment). Added concrete AND example paragraph before ComparisonRow to strengthen linearly-separable contrast. Fixed em dash spacing in LessonHeader description. Fixed planning doc depth inconsistency (linear collapse INTRODUCED → DEVELOPED). Removed unused WarningBlock import.
- [Iteration 2] Fixed all remaining spaced em dashes throughout the lesson (12 instances).

### Remaining items:
- None
