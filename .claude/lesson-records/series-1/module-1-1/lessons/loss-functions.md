# Lesson: Loss Functions — Measuring "Wrongness"

**Module:** 1.1 — The Learning Problem
**Slug:** loss-functions
**Position:** Lesson 3 of 6
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| ML as function approximation | INTRODUCED | what-is-learning |
| Generalization vs memorization | INTRODUCED | what-is-learning |
| Bias-variance tradeoff (intuitive) | INTRODUCED | what-is-learning |
| Train/val/test splits | INTRODUCED | what-is-learning |
| Linear model (y = mx + b) | DEVELOPED | linear-regression |
| Parameters (weight, bias) | DEVELOPED | linear-regression |
| Fitting = finding good parameters | INTRODUCED | linear-regression |
| ML notation (y-hat, w, b) | INTRODUCED | linear-regression |
| y-hat vs y distinction | INTRODUCED | linear-regression |

Mental models: "Parameters are the knobs the model learns." "Fitting = making the line close to data." Student tried fitting manually with LinearFitExplorer but had no way to quantify "goodness." This lesson resolves that open question.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand residuals, MSE as a loss function, and the loss landscape as a surface where training = finding the minimum.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Linear model y-hat = wx + b | INTRODUCED | DEVELOPED | linear-regression | OK | Student interacted with the widget |
| y-hat vs y distinction | INTRODUCED | INTRODUCED | linear-regression | OK | Notation was established |
| "Fitting needs measurement" | INTRODUCED | INTRODUCED | linear-regression | OK | The lesson ended with this open question |
| Basic algebra (squaring, sums) | DEVELOPED | APPLIED | Prior education | OK | Software engineer |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Sum residuals to get total error | Seems natural to add up errors | Positive and negative cancel: +5 and -5 = 0 | Section 2: Why Square? |
| All errors matter equally | Intuitive default | MSE penalizes large errors quadratically — one outlier matters more | Section 2: WarningBlock on large errors |
| Loss landscape is abstract/theoretical | Sounds like a metaphor | Interactive LossSurfaceExplorer — drag the point, see the actual bowl shape | Section 4 + Widget |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| LinearFitExplorer with residuals shown | Interactive positive | See individual residuals as red lines, watch MSE change |
| Cancellation example (+5 and -5) | Negative | Why summing residuals doesn't work |
| MAE vs MSE comparison | Positive | Shows two valid solutions to the cancellation problem |
| LossSurfaceExplorer | Interactive positive | Direct manipulation of the loss landscape |

---

## Phase 3: Design

### Narrative Arc

The previous lesson ended with an open question: "How do you compare two fits?" This lesson answers it. It starts with the most natural idea (residuals = distance from point to line), shows why naively summing them fails (cancellation), introduces squaring as the fix, builds up to the full MSE formula, then introduces the powerful mental model of the loss landscape. The student leaves knowing that training = finding the minimum of a surface.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Symbolic | Residual formula, MSE formula (KaTeX) | Precise mathematical definition |
| Interactive | LinearFitExplorer with residuals=true, MSE=true | See residuals and loss update in real time |
| Interactive | LossSurfaceExplorer | Direct manipulation of the loss surface |
| Verbal | "Wrongness score" framing | Intuitive label for loss |
| Geometric/Spatial | Loss surface as a bowl with a valley | Spatial metaphor for optimization |

### Cognitive Load

- **New concepts:** 3 (residuals, MSE/loss, loss landscape)
- **Previous lesson load:** BUILD
- **This lesson:** STRETCH — introduces the mathematical foundation for optimization
- **Assessment:** 3 concepts at boundary, but residuals → MSE is a natural progression (residuals are a stepping stone to MSE, not independent)

### Scope Boundaries

- MSE only — no cross-entropy or other loss functions
- Formula breakdown but no derivation of gradients
- Loss landscape is visual/interactive, not mathematical
- Does NOT cover: how to find the minimum (that's gradient descent, next lesson)
- Convexity mentioned ("bowl shape, exactly one minimum") but not formally defined

---

## What Was Actually Built

1. **Header + Objective** — Framed as measuring model quality with a single number
2. **Residuals** — Formula + residual_i = y_i - y-hat_i, sign interpretation
3. **LinearFitExplorer with residuals** — Same widget from lesson 2, now with showResiduals=true, showMSE=true
4. **Why Square?** — Cancellation problem, MAE vs MSE comparison, three reasons for MSE
5. **MSE Formula** — Full formula with emphasized KaTeX block, broken down term by term
6. **Loss Landscape** — Mental model of plotting loss for all parameter combinations
7. **LossSurfaceExplorer widget** — Drag point on surface, see line and MSE update
8. **Summary** — 4 key takeaways
9. **Next Step** — Link to gradient-descent

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (2), TipBlock (2), WarningBlock (1), TryThisBlock (2)
- SummaryBlock, NextStepBlock
- LinearFitExplorer via ExercisePanel (with residuals + MSE enabled)
- LossSurfaceExplorer via ExercisePanel
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Residual = actual minus predicted** — individual prediction error
- **MSE = average squared residual** — single "wrongness score"
- **Loss landscape as a bowl** — every parameter combination maps to a loss value, training = finding the valley
- **Squaring penalizes large errors** — one big mistake hurts more than many small ones

### Analogies Used
- "Wrongness score" for loss
- "Bowl" / "valley" for convex loss surface
- Landscape metaphor — finding the lowest point

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — Missing callback to previous lesson's open question

**Location:** Section 1: Residuals (before the formula)
**Issue:** The planning doc explicitly states "The previous lesson ended with an open question: How do you compare two fits?" but the built lesson never calls back to this. It jumps straight into "For each data point, we can measure how far off our prediction is" without reminding the student WHY they need this. The lesson lacks a motivating hook that connects to their experience with LinearFitExplorer in the previous lesson.
**Student impact:** The student starts learning the residual formula without feeling the need for it. The emotional context—"I was manually fitting a line and couldn't tell if I was getting better"—is lost. This weakens engagement in the crucial opening.
**Suggested fix:** Add 2-3 sentences before the residual formula that recall the previous lesson: "In the last lesson, you dragged the line around and tried to make it 'fit.' But how do you know when one fit is better than another? You need a number—a single score that tells you how wrong the model is. Let's build that score step by step."

### [IMPROVEMENT] — "Large errors are penalized more" lacks a concrete numerical example

**Location:** Section 2: Why Square? (WarningBlock aside)
**Issue:** The WarningBlock says "one very wrong prediction hurts more than two slightly wrong predictions" but never shows this with numbers. The body text says "10² = 100, not just 10" which is a step in the right direction, but it's buried in a list item rather than developed as a clear comparison.
**Student impact:** The student gets the general idea but doesn't develop strong intuition for HOW MUCH squaring changes things. A concrete side-by-side comparison would make this "click."
**Suggested fix:** Add a brief concrete comparison, e.g., "Two predictions each off by 3: squared errors = 9 + 9 = 18. One prediction off by 6 (same total): squared error = 36. Same total error, but MSE penalizes the concentrated error twice as much." This could replace or supplement the 10²=100 mention.

### [IMPROVEMENT] — Em dash spacing violations

**Location:** Multiple locations
**Issue:** The lesson uses spaces around em dashes in at least two places:
- Line 50-51: "a single number — the loss — and" should be "number—the loss—and"
- Line 229: "the valley where loss is minimized" (preceding "landscape — the valley" in line 228-229 doesn't have an em dash, but the section prose may have the pattern elsewhere)
- Objective block: "a single number — the loss —"
**Student impact:** Minor visual inconsistency, but the writing style rule requires no spaces around em dashes.
**Suggested fix:** Find and fix all `— ` and ` —` patterns to `—` (no spaces).

### [POLISH] — Cancellation example could use concrete numbers from the widget data

**Location:** Section 2, first paragraph
**Issue:** The cancellation explanation uses abstract "5 too high and 5 too low" rather than numbers the student might see in their widget. Since the student just interacted with LinearFitExplorer, using values closer to the actual data scale (e.g., +0.5 and -0.5) would feel more grounded.
**Student impact:** Minor. The point gets across either way, but widget-scale numbers would reinforce the connection.
**Suggested fix:** Consider using values from the data range (residuals are typically in the -1 to +1 range with the default data) instead of 5. Or keep 5 for dramatic effect—it's fine either way.

### [POLISH] — LossSurfaceExplorer widget explanation text is in the widget, not in the lesson prose

**Location:** Section 4 / LossSurfaceExplorer widget
**Issue:** The widget has its own explanation text at the bottom ("Click or drag on the heatmap..."). The lesson prose introduces the concept, but the widget's built-in text is doing some of the teaching. This is fine architecturally, but the lesson-level prose could be slightly more specific about what the student will see (heatmap with contours, orange dot, green minimum marker) before they encounter the widget.
**Student impact:** Minimal. The student will figure it out. But a sentence preview of what they'll see would reduce the "what am I looking at?" moment.
**Suggested fix:** Add a brief sentence before the widget: "Below, you'll see a heatmap where color represents loss—darker means lower loss. Drag anywhere to explore, and watch how the line on the right changes."

### Review Notes

Overall this is a solid lesson. The two interactive widgets are well-chosen and effectively demonstrate the concepts. The flow from residuals → MSE → loss landscape is logical and well-paced. The main weakness is the opening—it doesn't leverage the motivational setup from the previous lesson. The "why should I care?" moment is implicit rather than explicit. The em dash spacing is a style consistency issue that should be fixed. The numerical example for squaring is a genuine pedagogical gap—concrete numbers make the "squaring penalty" intuition stick.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All three IMPROVEMENT findings from Iteration 1 were addressed:
- Motivation callback added as a new paragraph before Section 1
- Concrete numerical example added to WarningBlock (3²+3² vs 6²)
- Em dash spacing fixed throughout
- Polish: Widget preview sentence added to Section 4

New findings from Iteration 2:

### Findings

### [IMPROVEMENT] — Term "residual" introduced without explaining why it's called that

**Location:** Section 1: Residuals (line 92)
**Issue:** The lesson introduces "residual" as a term and defines it as "actual minus predicted," but never explains WHY it's called "residual." A student would reasonably ask: "Why not just call it 'error'?" The word "residual" comes from "what remains"—the leftover difference between what you observed and what you predicted. The lesson parenthetically says "(or error)" which acknowledges the more intuitive name, but the choice of "residual" as the primary term is unexplained.
**Student impact:** The student memorizes the term without understanding it. The word feels arbitrary. A brief aside explaining the etymology ("residual = what's left over after the model accounts for the pattern") would make the term stick and feel less like jargon.
**Suggested fix:** Add a TipBlock or InsightBlock aside explaining: "Why 'residual'? It means 'what's left over.' After the model explains the pattern in the data, the residual is what remains—the part the model couldn't capture." This also reinforces the mental model that a model tries to capture the signal in data.

### [POLISH] — MSE shown in widget before MSE is explained

**Location:** Widget: LinearFitExplorer (lines 109-131)
**Issue:** The widget is configured with `showMSE={true}` and the TryThisBlock says "Notice the MSE number changing" and "Can you get the MSE below 0.5?" But MSE isn't formally introduced until Section 2-3. The student sees a mysterious number labeled "MSE" that goes down when the fit improves.
**Student impact:** Likely minimal—the "show then explain" pattern works here because the number behaves intuitively (lower = better). The student may even develop curiosity about what "MSE" stands for. However, this could be briefly acknowledged: "You'll see a number called MSE—we'll explain exactly what that means in a moment. For now, just notice: smaller MSE = better fit."
**Suggested fix:** Either add a brief note in the TryThisBlock ("The MSE number measures how wrong the fit is—we'll break it down in the next section"), or leave as-is if the "show then explain" pattern is intentional (which it appears to be per the planning doc).

### Review Notes

The revisions from Iteration 1 are effective. The motivation paragraph creates a natural entry point. The concrete numerical example in the WarningBlock is clear and demonstrates the squaring penalty well. The em dash formatting is consistent. The new widget preview sentence in Section 4 helps orient the student. The remaining IMPROVEMENT finding about the term "residual" was flagged by the student themselves—it's a legitimate pedagogical gap. The POLISH finding about MSE-before-explanation is minor and the "show then explain" pattern is defensible.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 0

### Verdict: PASS

Both findings from Iteration 2 were addressed:
- "Why Residual?" TipBlock aside added to Section 1, explaining etymology and connecting to the mental model of "what the model couldn't capture"
- TryThisBlock updated to note MSE "measures overall wrongness (we'll break it down next)"
- Sign interpretation folded into the main prose paragraph (no longer in a separate aside)

### Findings

No new findings. The lesson is well-structured with:
- Strong motivation that connects to previous lesson experience
- Clear conceptual progression: residuals → why square → MSE formula → loss landscape
- Concrete examples at every stage (widget interactions + numerical comparisons)
- 5 distinct modalities for the core concept
- 3 misconceptions addressed with concrete counter-examples
- Appropriate terminology support ("Why Residual?" aside)
- Consistent writing style (em dashes, no spacing violations)
- Good widget integration with clear prompts and goals

### Review Notes

The lesson passes review. Across 3 iterations, the key improvements were:
1. Added motivating callback to previous lesson's open question
2. Added concrete numerical example for squaring penalty (3²+3² vs 6²)
3. Fixed em dash spacing throughout
4. Added "Why Residual?" etymology aside (student-flagged)
5. Added MSE preview note in TryThisBlock
6. Added widget orientation sentence before LossSurfaceExplorer

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS

### Changes made:
- [Iteration 1] Fixed: Added motivation paragraph recalling previous lesson's open question. Added concrete numerical example for squaring penalty in WarningBlock (3²+3²=18 vs 6²=36). Fixed all em dash spacing violations (7 instances). Added widget preview sentence before LossSurfaceExplorer.
- [Iteration 2] Fixed: Added "Why Residual?" TipBlock aside explaining etymology. Updated TryThisBlock to acknowledge MSE before formal explanation. Folded sign interpretation into main prose.

### Remaining items:
- None
