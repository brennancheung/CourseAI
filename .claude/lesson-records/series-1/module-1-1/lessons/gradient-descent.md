# Lesson: Gradient Descent — Following the Slope

**Module:** 1.1 — The Learning Problem
**Slug:** gradient-descent
**Position:** Lesson 4 of 6
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
| Fitting = finding good parameters | INTRODUCED | linear-regression |
| ML notation (y-hat, w, b) | INTRODUCED | linear-regression |
| Residuals (y - y-hat) | DEVELOPED | loss-functions |
| MSE loss function | DEVELOPED | loss-functions |
| Loss landscape as a bowl | INTRODUCED | loss-functions |
| Training = finding the minimum | INTRODUCED | loss-functions |

Mental models: "Loss is a wrongness score." "The loss landscape is a bowl with one valley." Student can interact with both the line-fitting and loss-surface widgets. The open question from loss-functions: "How do we actually find the minimum?"

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to understand gradient descent as the algorithm for finding parameter values that minimize loss, including the update rule and the role of the learning rate.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Loss function / MSE | DEVELOPED | DEVELOPED | loss-functions | OK | Student saw the formula and interacted with widgets |
| Loss landscape concept | INTRODUCED | INTRODUCED | loss-functions | OK | "Bowl with a valley" is sufficient framing |
| Basic calculus (derivatives) | INTRODUCED | APPLIED | Prior education | OK | Software engineer background, likely has calc |
| Parameters = learnable values | DEVELOPED | DEVELOPED | linear-regression | OK | Interacted with parameters via widget |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| Gradient points downhill | "Going with the gradient" sounds natural | Gradient points uphill — that's why we subtract | Section 3: minus sign emphasis |
| Bigger learning rate = faster learning | Bigger steps = faster | Too-big LR overshoots and diverges | Section 4 + Widget exploration |
| Gradient descent always finds global minimum | Works perfectly on the bowl | Neural networks have non-convex landscapes (mentioned) | Section 5: convexity caveat |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| Blindfolded on a hill | Positive (analogy) | Intuitive entry point for the algorithm |
| Derivative sign interpretation (+/-/0) | Positive | Connects calculus to the gradient concept |
| Update rule breakdown | Positive | Step-by-step formula walkthrough |
| GradientDescentExplorer widget | Interactive positive | Watch the algorithm step by step |
| Too-big vs too-small learning rate | Negative pair | Shows failure modes |

---

## Phase 3: Design

### Narrative Arc

The previous lesson established the loss landscape and posed the question: "How do we find the lowest point?" This lesson answers with gradient descent. It starts with the blindfolded hill analogy (feel the slope, step downhill), connects to calculus (the derivative IS the slope), presents the update rule, then lets the student watch it work interactively. The learning rate is previewed (not fully explored — that's the next lesson). Finally, the "why it works" section gives theoretical grounding and hints at the convexity limitation for neural networks.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Verbal/Analogy | Blindfolded on a hill, ball rolling downhill | Immediately intuitive physical metaphor |
| Symbolic | Update rule: theta_new = theta_old - alpha * grad L | Precise mathematical statement |
| Interactive | GradientDescentExplorer with step button and LR slider | Watch the algorithm execute step by step |
| Visual | Gradient arrow in the widget | See the direction and magnitude of the gradient |
| Concrete | Breaking down each term in the update rule | Demystifies the formula |

### Cognitive Load

- **New concepts:** 3 (gradient/derivative as slope, gradient descent update rule, learning rate as step size)
- **Previous lesson load:** STRETCH
- **This lesson:** STRETCH — core optimization concept
- **Note:** Two STRETCH lessons in a row (loss-functions → gradient-descent). This is borderline but acceptable because loss-functions was more about measurement (passive) while gradient-descent is about action (doing something with that measurement). The cognitive shift keeps engagement.

### Scope Boundaries

- 1D gradient descent only (single parameter)
- Learning rate is previewed but NOT deeply explored (next lesson)
- No implementation / code
- Convexity mentioned but not formally taught
- Does NOT cover: stochastic GD, batches, momentum, multi-parameter gradients
- The notation theta is used generically — not specific to w or b separately

---

## What Was Actually Built

1. **Header + Objective** — "the algorithm that finds parameters minimizing loss"
2. **Intuition: Ball Rolling Downhill** — Blindfolded analogy, 3-step algorithm
3. **What is the Gradient?** — Derivative = slope, positive/negative/zero interpretation, nabla notation
4. **The Update Rule** — Emphasized formula block, term-by-term breakdown
5. **GradientDescentExplorer** — Step-by-step animation with LR slider and gradient arrow
6. **Learning Rate Preview** — Too big vs too small (ConceptBlocks), "just right" WarningBlock
7. **Why It Works** — Tangent line as best local approximation, convexity note
8. **Summary** — 4 key takeaways
9. **Next Step** — Link to learning-rate

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (2), TipBlock (2), WarningBlock (1), TryThisBlock (1), ConceptBlock (3)
- SummaryBlock, NextStepBlock
- GradientDescentExplorer via ExercisePanel
- KaTeX: InlineMath, BlockMath

### Mental Models Established
- **Gradient = slope = direction of steepest ascent** — single most important concept
- **Update rule: subtract the gradient** — go opposite to uphill
- **Learning rate = step size** — too big overshoots, too small is slow
- **Convergence = gradient near zero** — flat spot on the loss surface
- **Ball rolling downhill** — physical intuition for the algorithm

### Analogies Used
- Blindfolded on a hilly landscape
- Ball rolling downhill
- Goldilocks zone for learning rate

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 2
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — Widget LR range doesn't match lesson text

**Location:** Section 5 (Learning Rate Preview) + GradientDescentExplorer widget
**Issue:** The lesson text describes "Too Big (α > 0.7)" and "Too Small (α < 0.1)" but the widget's LR slider only goes from 0.01 to 0.5. The student reads about overshooting at α > 0.7 but cannot experience it in the widget. The misconception "bigger LR = faster learning" is addressed textually but not experientially.
**Student impact:** The student learns that high learning rates overshoot, but they can't SEE it happen. This weakens the negative example significantly. The numbers in the lesson text (0.7) and the widget's max (0.5) are inconsistent.
**Suggested fix:** Either (a) increase the widget slider max to ~1.0 so the student can see overshooting/divergence, or (b) change the lesson text thresholds to match the widget's actual range (e.g., "Too Big (α > 0.4)"). Option (a) is pedagogically stronger since experiencing divergence builds deeper understanding. Also update the `initialLearningRate` prop in the lesson from 0.3 to something that starts in a reasonable range like 0.15.

### [IMPROVEMENT] — LR section tells but doesn't connect to widget

**Location:** Section 5 (Learning Rate Preview)
**Issue:** The section describes too-big and too-small learning rates as abstract ConceptBlocks, but doesn't explicitly tell the student to go back to the widget and try these values. The TryThisBlock for the widget (in Section 4) says "Try different learning rates" but is placed before the LR section that explains what to look for.
**Student impact:** The student may read the LR section passively rather than actively experimenting. The lesson flow puts the widget BEFORE the learning rate explanation, so the student has already moved past the widget when they learn what to test.
**Suggested fix:** Add a brief instruction in the LR section that explicitly directs the student back to the widget: "Go back to the visualization above and try setting α to different extremes." Alternatively, reorder to put the LR explanation before the widget so the student knows what to look for.

### [POLISH] — Initial LR of 0.3 may be too fast for first impression

**Location:** Widget invocation (line 199)
**Issue:** The lesson passes `initialLearningRate={0.3}` to the widget. At 0.3, the ball converges in very few steps, which doesn't give the student much time to observe the process. The widget's own default is 0.15 which would show more steps and let the student observe the behavior more clearly.
**Student impact:** Minor — the student might see the ball reach the minimum in 3-4 steps and not get a strong feel for the iterative process.
**Suggested fix:** Use `initialLearningRate={0.15}` (the widget's own default) or even `0.1` to give more steps before convergence.

### [POLISH] — "Why It Works" section is thin

**Location:** Section 6 (Why Does This Work?)
**Issue:** This section has three paragraphs of text with no visual, formula, or interactive element. It mentions "tangent line" and "best linear approximation" without connecting these back to the widget where the tangent line is actually shown as the red dashed line.
**Student impact:** Minor — the section reads as hand-wavy. Students who want to understand "why" don't get much to grab onto.
**Suggested fix:** Add a brief reference to the tangent line shown in the widget: "The dashed red line in the visualization above IS the tangent line — it shows the best linear approximation of the loss curve at that point." This connects the theory to what they already saw.

### Review Notes

Overall this is a solid lesson. The core pedagogical structure is strong: motivation → intuition → math → interact → theory. The widget is well-designed with step-by-step animation, live stats, and the gradient/update arrows. The main weakness is the disconnect between the learning rate discussion (text says 0.7 is too big) and the widget (slider caps at 0.5), which prevents the student from experiencing the most important negative example. The lesson would benefit from tighter widget-text integration where the LR section explicitly references back to the widget.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS with polish notes

### Findings

### [POLISH] — Em dashes with spaces

**Location:** Multiple locations in lesson prose
**Issue:** The writing style rule requires em dashes with no spaces (`word—word` not `word — word`). Several places in the lesson use spaced em dashes: "minimum loss — following" (line 40), "gradient descent — the algorithm" (line 51), "no universal best value — it" (line 252).
**Student impact:** None — purely stylistic.
**Suggested fix:** Remove spaces around em dashes throughout the lesson.

### Review Notes

The iteration 1 fixes addressed all IMPROVEMENT findings effectively:
- Widget slider now goes to 1.0, matching the lesson's guidance to try α = 0.8+
- LR section now explicitly sends the student back to the widget with specific values
- Initial LR changed from 0.3 to 0.15 for better first-impression observation
- "Why It Works" section now references the tangent line in the widget

The lesson's core strengths remain: strong motivation (blindfolded analogy), clear formula presentation (update rule breakdown), excellent interactive element (GradientDescentExplorer with live stats and equation), and good pacing from intuition → math → interaction → theory.

The only remaining finding is a minor style consistency issue with em dash spacing.

---

## Improvement Summary — 2026-02-06

**Iterations:** 2/3
**Final verdict:** PASS with polish notes (em dash spacing fixed post-review)

### Changes made:
- [Iteration 1] Widget slider max increased from 0.5 to 1.0 so student can experience overshooting/divergence
- [Iteration 1] LR section text changed from abstract thresholds (α > 0.7, α < 0.1) to actionable instructions (try α = 0.8+, try α = 0.02) with explicit direction to use the widget
- [Iteration 1] Initial learning rate changed from 0.3 to 0.15 for longer convergence observation
- [Iteration 1] "Why It Works" section now references the dashed red tangent line in the widget
- [Post-review] Em dashes in prose changed to no-space style (word—word) per style rule

### Remaining items:
- None
