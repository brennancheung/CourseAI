# Lesson: Linear Regression

**Module:** 1.1 — The Learning Problem
**Slug:** linear-regression
**Position:** Lesson 2 of 6
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Generalization vs memorization | INTRODUCED | what-is-learning |
| Bias-variance tradeoff (intuitive) | INTRODUCED | what-is-learning |
| Train/val/test splits | INTRODUCED | what-is-learning |

Mental models established: "There's a true function, we approximate from examples." Studying analogy (cramming vs understanding). Goldilocks model complexity.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand a linear model as the simplest function approximator, what parameters (weight/bias) are, and what "fitting" means intuitively.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| ML as function approximation | INTRODUCED | INTRODUCED | what-is-learning | OK | Just needs recognition of the framing |
| y = mx + b (algebra) | DEVELOPED | APPLIED | Prior education | OK | Software engineer background |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Perfect fit = hit every point | Intuition from connecting dots | Interactive widget — can't hit every point with a straight line | Section 2 + Widget |
| Visual estimation is sufficient | Humans can "eyeball" a good line | "How do you compare two fits?" — motivates loss functions | Section 3: Problem with Guessing |
| More parameters = always better | More flexibility = better fit | Not explicitly addressed yet (deferred to bias-variance in later lessons) | — |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| y = mx + b (familiar algebra) | Positive | Connect to prior knowledge before ML notation |
| LinearFitExplorer widget | Interactive positive | Hands-on experience fitting a line |
| "Compare two fits" thought experiment | Negative | Motivates need for quantitative measurement (loss) |

---

## Phase 3: Design

### Narrative Arc

The lesson bridges from the abstract "function approximation" idea to the most concrete instance: a straight line through data points. It starts familiar (y = mx + b from algebra), reframes the slope and intercept as "parameters" that the model learns, then gives the student a chance to try fitting by hand. The failure of visual estimation ("how do you know this is the best?") motivates the next lesson on loss functions. Finally, it introduces ML notation (w, b, y-hat) as a convention shift.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Symbolic | y = mx + b, then y-hat = wx + b | Familiar formula → ML convention |
| Interactive | LinearFitExplorer with draggable slope/intercept | Direct manipulation builds intuition for what parameters do |
| Verbal/Analogy | "Fitting = finding good parameters" | Simple framing |
| Visual | KaTeX block math for the equation | Clean mathematical presentation |

### Cognitive Load

- **New concepts:** 3 (parameters/weights/bias, fitting, ML notation y-hat/w/b)
- **Previous lesson load:** STRETCH
- **This lesson:** BUILD — applies the function approximation frame to a concrete model
- **Assessment:** 3 concepts is at the boundary but appropriate since two are closely related (parameters = weights+bias, and the notation is just a rename)

### Scope Boundaries

- No loss functions yet — fitting is intuitive only
- No optimization — no gradient descent
- No code
- Residuals mentioned in preview but NOT taught
- Widget configured with showResiduals=false and showMSE=false

---

## What Was Actually Built

1. **Header + Objective** — Frames lesson as understanding the simplest model
2. **The Simplest Model** — y = mx + b with KaTeX, lists x/y/m/b meanings
3. **What Fitting Means** — Informal definition, motivates the widget
4. **LinearFitExplorer widget** — Draggable slope and intercept, residuals and MSE hidden
5. **The Problem with Guessing** — Motivates loss functions (next lesson)
6. **ML Notation** — y-hat = wx + b, weight/bias terminology, hat convention
7. **Summary** — 4 key takeaways
8. **Next Step** — Link to loss-functions

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (1), TipBlock (2), TryThisBlock (1), ConceptBlock (1)
- SummaryBlock, NextStepBlock
- LinearFitExplorer via ExercisePanel
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Parameters = the knobs the model learns** — slope (weight) and intercept (bias)
- **Fitting = finding good parameter values** — make the line close to the data
- **y-hat vs y** — prediction vs truth, the difference is the error
- **Linear model as simplest function approximator** — connects back to lesson 1

### Analogies Used
- y = mx + b from algebra (familiar → new)
- "weight" and "bias" generalize to neural networks (forward reference)

---

## Review — 2026-02-06

### Summary
- Critical: 1
- Improvement: 5
- Polish: 2

### Verdict: MAJOR REVISION

Critical finding: lesson lacks a motivating problem scenario, making it abstract where it should be concrete. Combined with multiple improvement findings around interaction instructions, narrative structure, and missing examples.

### Findings

#### [CRITICAL] — No motivating problem/scenario

**Location:** Section 1 (The Simplest Model) and Section 2 (What Fitting Means)
**Issue:** The lesson introduces a linear model and "fitting" without ever giving the student a concrete prediction problem. There's no "here's a scenario where you'd want to predict Y from X." The student gets abstract data points with no real-world grounding.
**Student impact:** Student understands the math but doesn't connect to WHY anyone would do this. The what-is-learning lesson established function approximation with concrete problems (spam detection, dog recognition). This lesson drops back to pure abstraction — "we have data points" with no context for what those data points represent. Reduces engagement and retention significantly.
**Suggested fix:** Open with a concrete scenario. E.g., "You have data on house sizes and sale prices. Can you predict the price of a new house from its size?" Then the data points in the widget have meaning — they represent real measurements. The x-axis is square footage, the y-axis is price. The line becomes a prediction tool, not just a geometric object.

#### [IMPROVEMENT] — Widget aside instructions don't match actual interaction model

**Location:** TryThisBlock "Experiment" (aside next to LinearFitExplorer)
**Issue:** Aside says "Drag the intercept point up and down" and "Drag the slope point to change the angle." The actual widget has no draggable points on the graph. Interaction is via draggable numbers in the equation below the graph and range sliders.
**Student impact:** Student reads the instructions, looks at the graph expecting to grab and drag points, and finds nothing interactive on the graph itself. Confusion and frustration until they discover the equation numbers and sliders below.
**Suggested fix:** Update aside to match the actual widget: "Drag the orange number to change the slope" / "Drag the purple number to change the intercept" / "Or use the sliders below for fine control."

#### [IMPROVEMENT] — Notation section kills narrative momentum

**Location:** Section 4: ML Notation (after "The Problem with Guessing")
**Issue:** The lesson builds a strong arc toward the open question "How do we measure goodness?" — a compelling cliffhanger for the loss functions lesson. Then it pivots to a notation section (ŷ = wx + b) that is dry and disconnected from the narrative. The emotional beat of "you can't solve this by eyeballing" loses its power.
**Student impact:** Student was engaged by the unsolved problem and ready for the next lesson. The notation section deflates that energy and feels like an afterthought.
**Suggested fix:** Move notation earlier — right after introducing y = mx + b in Section 1. "In ML, we write this slightly differently..." is a natural aside at that point. Then the lesson can end on the strong "problem with guessing" cliffhanger, with the NextStep block following immediately.

#### [IMPROVEMENT] — No concrete worked example with numbers

**Location:** Section 1 and Section 4
**Issue:** Two equation forms (y = mx + b and ŷ = wx + b) are presented with variable definitions but never with actual numbers plugged in. The student sees notation but never computes a prediction.
**Student impact:** The notation feels like memorization rather than understanding. The student can recite "w is weight, b is bias" without being able to USE the formula.
**Suggested fix:** Add one concrete calculation, ideally tied to the motivating scenario: "If w = 0.5 and b = 1, then for x = 2: ŷ = 0.5(2) + 1 = 2. Our model predicts 2." This grounds the abstraction.

#### [IMPROVEMENT] — "Can't hit every point" misconception is buried

**Location:** TryThisBlock aside, bullet 4 ("Notice: you can't hit every point perfectly!")
**Issue:** A key misconception (perfect fit = hit every point) is addressed as a single bullet point in an aside. This is the most important insight from the widget interaction and it deserves main-flow emphasis.
**Student impact:** Student might not read the aside, or might read it and not register it as important. The misconception that "good model = passes through every point" goes unaddressed for many students.
**Suggested fix:** After the widget, add a main-flow paragraph: "You probably noticed — no matter how you adjust the line, you can't make it pass through every point. That's not a failure. Real data is noisy, and a straight line is deliberately simple. The gap between the line and the points is the error — and we'll learn to measure it precisely in the next lesson."

#### [IMPROVEMENT] — Building Blocks aside uses undefined terms

**Location:** TipBlock aside next to ObjectiveBlock
**Issue:** "Neural networks are just many linear operations combined with non-linear activations." The student doesn't know what neural networks, "linear operations" (in this context), or "non-linear activations" are. This is a forward reference with zero current value.
**Student impact:** Either ignores it (wasted aside space) or gets anxious about terms they can't parse, wondering if they should already know this.
**Suggested fix:** Replace with something the student CAN appreciate: "Every neural network starts with exactly this operation — a weighted sum plus a bias. Master this and you have the fundamental building block." Or simply motivate why linear regression matters: "This is the starting point for everything in deep learning."

#### [POLISH] — Widget slider labels use "m" not "w"

**Location:** LinearFitExplorer widget, slider labels ("Slope (m)" and "Intercept (b)")
**Issue:** If the notation section is moved earlier (per improvement above), the widget's slider labels would be inconsistent with the ML notation the student just learned.
**Suggested fix:** If notation moves earlier, update widget (or its labeling) to use w/b. If notation stays at the end, current labels are fine since the widget appears before the notation section.

#### [POLISH] — No ConstraintBlock for scope boundaries

**Location:** After ObjectiveBlock
**Issue:** The what-is-learning lesson (post-revision) has a ConstraintBlock setting student expectations. This lesson doesn't have one, and its scope boundaries are important (no loss, no optimization, no code).
**Suggested fix:** Add ConstraintBlock: "The simplest model — no optimization yet / Understanding parameters and fitting / Does NOT cover how to find the best fit (that's next)"

### Review Notes

The lesson has sound concept progression and appropriate scope boundaries. The interactive widget is well-designed with proper feature gating (residuals and MSE hidden). The core structural weakness is the lack of a motivating problem — this lesson is the student's first encounter with a specific ML model and it should feel grounded in a real prediction task, not abstract algebra. The what-is-learning lesson did this well; this lesson regressed.

The notation section placement at the end is a structural issue that disrupts an otherwise clean narrative arc. The lesson should end on its strongest beat (the unsolved problem of measuring fit), not on a notation rename.

The widget interaction mismatch in the aside is the kind of bug that creates unnecessary friction — easy to fix, high impact on the student's first interaction with the widget.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

All critical findings from iteration 1 are resolved. Remaining findings are improvements around density and scenario grounding.

### Iteration 1 Fixes Verified
- [CRITICAL] Motivating scenario — FIXED. Apartment pricing scenario now opens the lesson.
- [IMPROVEMENT] Widget aside instructions — FIXED. Now references orange/purple numbers and sliders.
- [IMPROVEMENT] Notation placement — FIXED. ML notation is now in Section 2, lesson ends on cliffhanger.
- [IMPROVEMENT] No worked example — FIXED. Concrete calculation added (w=0.5, b=1, x=2).
- [IMPROVEMENT] "Can't hit every point" buried — FIXED. Now main-flow paragraph with overfitting callback.
- [IMPROVEMENT] Building Blocks undefined terms — FIXED. Reworded to avoid "non-linear activations."
- [POLISH] ConstraintBlock — FIXED. Added after objective.

### New Findings

#### [IMPROVEMENT] — Section 2 density could overwhelm

**Location:** Section 2: The Simplest Model (lines 108-169)
**Issue:** This section contains: (1) familiar algebra y=mx+b with 4 variables, (2) ML notation ŷ=wx+b with 4 variables, (3) mapping between the two, (4) a worked example, (5) the error concept. It's the longest section in the lesson.
**Student impact:** A student with ADHD might lose focus partway through. The worked example is the most important part but comes last.
**Suggested fix:** Consider splitting into two Rows to create a visual break — the first Row covers the algebra notation, the second Row introduces ML notation + worked example. OR simply add a visual separator (the bg-muted/50 box around the ML equation already provides one).

#### [IMPROVEMENT] — Worked example uses abstract numbers, not the scenario

**Location:** Lines 140-154
**Issue:** The apartment scenario establishes "square footage → price" but the worked example uses w=0.5, b=1, x=2 with no connection to the scenario.
**Student impact:** Slight disconnect. The motivating scenario becomes background rather than the active thread.
**Suggested fix:** Add a brief acknowledgment: "Using simple numbers for clarity — the same principle applies whether x is square footage or any other measurement." OR tie the example to the scenario directly.

#### [POLISH] — Building Blocks aside still uses forward-reference terms

**Location:** Lines 55-59 (TipBlock aside)
**Issue:** "A weighted sum plus a bias" — "weight" and "bias" aren't defined until Section 2. Minor forward reference in an aside.
**Suggested fix:** Simplify to: "Every neural network starts with exactly this — a line. Master this and you have the fundamental building block."

#### [POLISH] — Widget slider labels say "Slope (m)" after lesson introduces w

**Location:** LinearFitExplorer widget
**Issue:** Since notation now comes before the widget, the student has learned "w = weight" but the widget shows "Slope (m)."
**Suggested fix:** Widget code change — deferred since widget is shared across lessons. Note for future.

### Review Notes

Major improvement from iteration 1. The lesson now has a compelling motivating scenario, correct widget instructions, proper narrative arc ending on the cliffhanger, and the key misconception addressed in the main flow. The remaining issues are density management (Section 2 is long) and a minor scenario-example disconnect. Neither is critical — the lesson is functional and effective.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS (with polish notes)

All critical and improvement findings are resolved. The only remaining item is the widget slider label inconsistency (POLISH), which requires a shared widget code change and is deferred.

### Iteration 2 Fixes Verified
- [IMPROVEMENT] Section 2 density — FIXED. Split into two Rows: algebra in first Row, ML notation + worked example in second Row. Each Row is focused and digestible.
- [IMPROVEMENT] Worked example disconnected from scenario — FIXED. Added grounding phrase: "Using simple numbers (the same principle applies to square footage, temperature, or any input)."
- [POLISH] Building Blocks forward reference — FIXED. Simplified to "Every neural network starts with exactly this — a line."
- [POLISH] Widget slider labels — DEFERRED. Shared widget, requires separate change.

### Remaining Items

#### [POLISH] — Widget slider labels say "Slope (m)" after lesson introduces w
**Status:** Deferred. LinearFitExplorer is used by multiple lessons. Changing labels requires updating the widget or adding a prop for label customization. Not a lesson-level fix.

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS (with 1 deferred polish item)

### Changes made:
- **[Iteration 1]** Added motivating apartment-pricing scenario; moved ML notation from end to Section 2; added worked example with numbers; promoted "can't hit every point" to main flow with overfitting callback; fixed widget aside instructions to match actual interaction; replaced Building Blocks aside to remove undefined terms; added ConstraintBlock; restructured narrative to end on loss-function cliffhanger; updated summary takeaways.
- **[Iteration 2]** Split dense Section 2 into two Rows (algebra / ML notation); grounded worked example with "same principle applies to any input" phrase; simplified Building Blocks aside further.

### Remaining items:
- Widget slider labels show "Slope (m)" instead of "Weight (w)" — deferred to widget-level change since LinearFitExplorer is shared across lessons.
