# Lesson: What is Learning?

**Module:** 1.1 — The Learning Problem
**Slug:** what-is-learning
**Position:** Lesson 1 of 6
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

This is the first lesson in the course. No prior concepts exist.

- Student is a software engineer — understands functions, inputs/outputs, if/else logic
- Comfortable with Python, can read academic papers
- Has used AI tools but wants deeper understanding
- No prior ML concepts from this course

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand machine learning as function approximation and why generalization matters more than training performance.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Programming concepts (functions, I/O) | DEVELOPED | APPLIED | Prior experience | OK | Software engineer background |

No ML prerequisites — this is lesson 1.

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| ML = writing smarter rules | Programming background, "AI" in pop culture | Spam detection — can't enumerate all rules | Section 1: Core Insight |
| Perfect training accuracy = good model | Intuition that accuracy is always good | Memorization example — perfect on training, fails on new data | Section 2: Generalization vs Memorization |
| More complex model = better model | "More capable" feels better | Overfitting in the widget — wiggly line through every point | Section 3: Bias-Variance Tradeoff |
| Lower loss = always better, so fit as closely as possible | Natural optimization instinct — if a metric measures error, zero must be the goal | Stock trading strategy that perfectly fits last year's data but fails on tomorrow's market | "What Goes Wrong" section, WarningBlock aside |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| Spam detection (function approximation) | Positive | Shows why you can't write explicit rules |
| Dog recognition (memorization) | Negative | Shows memorization fails on new data |
| Study analogy (cramming vs understanding) | Positive | Maps to familiar experience |
| OverfittingWidget (underfitting/good fit/overfitting) | Interactive positive | Visual exploration of the tradeoff |
| Stock trading strategy (overfitting) | Negative | Concrete failure: perfect on past data, fails on tomorrow's market. Addresses the "lower loss = better" misconception |
| Test set peeking | Negative | Shows why test set must stay untouched |

---

## Phase 3: Design

### Narrative Arc

The lesson starts from the familiar (traditional programming = writing rules) and reframes ML as a fundamentally different paradigm: instead of writing rules, you provide examples and the machine discovers the rules. This immediately raises a question — how do you know if the machine actually learned the right rules? This motivates generalization vs memorization, which leads to the bias-variance tradeoff, which leads to how we actually measure generalization (train/val/test splits).

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Verbal/Analogy | Studying for a test (cramming vs understanding) | Maps to universal experience |
| Visual | ComparisonRow (memorization vs generalization) | Side-by-side comparison clarifies the distinction |
| Interactive | OverfittingWidget | Lets student directly see underfitting/good fit/overfitting |
| Concrete example | Dog recognition, spam detection | Familiar real-world problems |

### Cognitive Load

- **New concepts:** 4 (function approximation, generalization/memorization, bias-variance tradeoff, train/val/test splits)
- **Assessment:** This is borderline — 4 concepts is above the recommended 2-3. However, as the opening lesson, these concepts are relatively intuitive and don't require mathematical depth. The bias-variance tradeoff is kept at INTRODUCED (intuition only, no formulas).
- **Lesson type:** STRETCH — foundational framing for everything that follows

### Scope Boundaries

- No math — pure intuition
- No code
- Bias-variance is intuition only, not quantitative
- Does NOT cover: specific ML algorithms, loss functions, how training actually works

---

## What Was Actually Built

The lesson follows the design closely, with a post-widget reinforcement section added later:

1. **Header + Objective** — Clear framing with Row layout
2. **Core Insight** — ML as function approximation (traditional programming vs ML)
3. **Generalization vs Memorization** — ComparisonRow with study analogy
4. **Bias-Variance Tradeoff** — Underfitting/Overfitting explained with side-by-side ConceptBlocks
5. **OverfittingWidget** — Interactive exploration of underfitting/good fit/overfitting
6. **What Goes Wrong** — Post-widget reinforcement connecting each fit back to memorization vs generalization frame. Three ConceptBlocks (Underfitting → Good Fit → Overfitting) with WarningBlock aside for the "perfect fit trap" (stock trading example)
7. **Train/Val/Test Splits** — Three ConceptBlocks with percentages and analogy
8. **Summary** — 4 key takeaways
9. **Next Step** — Link to linear regression

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, ConstraintBlock, SectionHeader
- InsightBlock (3), TipBlock (2), TryThisBlock (2), ConceptBlock (8), WarningBlock (1)
- ComparisonRow (1), SummaryBlock, NextStepBlock
- OverfittingWidget via ExercisePanel

### Mental Models Established
- **ML as function approximation** — "There's a true function, we approximate it from examples"
- **Generalization = the real goal** — Not training accuracy
- **Underfitting/Good Fit/Overfitting mapped to memorization framework** — Underfitting = hasn't learned yet, Good fit = generalization, Overfitting = memorization in disguise
- **Perfect fit is a trap** — Natural instinct to minimize error is misleading; stock trading example as concrete proof
- **Data splits as exam metaphor** — Textbook (train), practice tests (val), real exam (test)

### Analogies Used
- Studying for a test (cramming vs understanding)
- Straight line vs wiggly line through data points
- Stock trading strategy that perfectly predicts last year's prices (overfitting failure)
- Textbook / practice test / real exam for data splits

---

## Review — 2026-02-06

### Summary
- Critical: 0
- Improvement: 5
- Polish: 3

### Verdict: NEEDS REVISION → REVISED

### Findings

#### [IMPROVEMENT] — Core Insight section lacked a main-flow concrete example
**Location:** Section 1: The Core Insight
**Issue:** "Function approximation" was explained abstractly. The only concrete example (spam detection) was in the aside, not the main content.
**Student impact:** Student understood the words but didn't feel what function approximation means.
**Fix applied:** Moved spam detection into the main flow as the opening — the student now experiences the problem (whack-a-mole rules) before getting the solution (ML as function approximation). Aside now shows generalization of the pattern (house prices, medical diagnosis, image recognition).

#### [IMPROVEMENT] — Memorization example didn't explain WHY it fails
**Location:** Section 2: Generalization vs Memorization
**Issue:** Student was told memorization fails but not shown why. A software engineer might think a lookup table works fine.
**Fix applied:** Added concrete failure modes: "show it a golden retriever from a different angle, in different lighting, or a breed it's never seen — it's completely lost."

#### [IMPROVEMENT] — "Bias" and "Variance" labels introduced without connection
**Location:** Section 3: Bias-Variance Tradeoff
**Issue:** Terms were presented as synonyms for underfitting/overfitting without grounding.
**Fix applied:** Added inline explanations within each ConceptBlock — bias as "strong bias toward simplicity, can't bend enough" and variance as "train on slightly different data, get wildly different result."

#### [IMPROVEMENT] — Missing transition between bias-variance and data splits
**Location:** Between Section 3 and Section 4
**Issue:** Implicit jump from "what overfitting is" to "we need three datasets."
**Fix applied:** Added bridge paragraph: "How do we detect that? We can't wait until deployment. We need to simulate the real world by holding back data."

#### [IMPROVEMENT] — OverfittingWidget defaults to "Good Fit"
**Location:** OverfittingWidget.tsx
**Issue:** Widget opened showing the ideal result, skipping the narrative arc.
**Fix applied:** Changed default to 'underfit' so the student starts with the problem and discovers the solution.

#### [POLISH] — No ConstraintBlock for scope boundaries
**Location:** After objective
**Fix applied:** Added ConstraintBlock: "Pure intuition — no math, no code / Building the mental model / Does NOT cover specific algorithms or how training works."

#### [POLISH] — "hyperparameters" mentioned without definition
**Location:** Validation Set ConceptBlock
**Fix applied:** Replaced "Used to tune hyperparameters" with "Used to make decisions like 'is my model too simple or too complex?'"

#### [POLISH] — Study analogy only in aside
**Status:** No action needed. The main flow's ComparisonRow already includes "Like cramming answers before a test." The aside expands on this. Working as intended.

### Review Notes
The lesson's structure and concept progression are sound. The main weakness was abstraction-first in Section 1 — the opening section was the weakest despite being the most important. The revision leads with concrete experience (spam filter whack-a-mole) which should make the "function approximation" framing land with more impact. The bias/variance grounding was a small but important fix — these terms will recur throughout the course and needed a foundation beyond just labels.

---

## Revision — 2026-02-06 (post-widget reinforcement)

### What changed
Added a new "What Goes Wrong" section between the OverfittingWidget and the Train/Val/Test section.

**Problem:** After the widget, the lesson had only terse sidebar labels ("Underfitting: The line misses the curve entirely") and jumped straight to data splits. The student saw the visual but got no explanation of *why* each fit is a problem, and the connection to the memorization/generalization frame from Section 2 was never made explicit.

**Solution:** Three ConceptBlocks in the order matching the widget sidebar (Underfitting → Good Fit → Overfitting), each framed through the memorization vs generalization lens:
- Underfitting: "hasn't learned yet — not memorizing, not generalizing, just wrong"
- Good fit: "actual generalization — learned the pattern, not the points"
- Overfitting: "memorization in disguise — looks perfect on training data, falls apart on new data"

**Added misconception: "lower loss = always better."** Natural optimization instinct — if loss measures error, zero must be the goal. Addressed with a WarningBlock aside using a stock trading example (strategy that perfectly predicts last year's prices fails on tomorrow's market). Self-contained, no assumed domain knowledge.

### Design decisions
- Kept as flowing ConceptBlocks, not heavy multi-paragraph explanations. First attempt was too verbose and conflated too many concepts (noise, signal, flexibility, model complexity). Tightened through user feedback.
- Stock trading example in the aside (WarningBlock), not main flow — it adds clarity but isn't essential to the core lesson.
- Underfitting described as "hasn't learned the pattern yet" rather than "model too simple" — avoids claiming only one cause (could also be insufficient training).
- Order matches widget sidebar: Underfitting → Good Fit → Overfitting. Good fit before overfitting reads better pedagogically — student sees the correct answer before the trap.
