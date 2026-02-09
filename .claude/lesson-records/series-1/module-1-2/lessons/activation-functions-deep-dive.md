# Lesson: Activation Functions — Visual Reference

**Module:** 1.2 — From Linear to Neural
**Slug:** activation-functions-deep-dive
**Position:** Lesson 4 of 4 (module capstone/reference)
**Status:** Built (retroactive record)

---

## Phase 1: Student State (Orient)

| Concept | Depth | Source |
|---------|-------|--------|
| Activation function concept | DEVELOPED | activation-functions |
| Neuron = σ(w·x + b) | DEVELOPED | activation-functions |
| Sigmoid formula and shape | INTRODUCED | activation-functions |
| ReLU formula and shape | INTRODUCED | activation-functions |
| Why activation breaks linearity | DEVELOPED | activation-functions |
| XOR solved via space transformation | DEVELOPED | activation-functions |
| Vanishing gradients (mentioned) | MENTIONED | activation-functions |
| Dying ReLU (mentioned) | MENTIONED | activation-functions |

Mental models: "Activation creates thresholds that can't collapse." "Networks transform space." Student has seen sigmoid and ReLU shapes. This lesson provides the full reference.

---

## Phase 2: Analyze

**Target concept:** This lesson teaches the student to recognize the shapes, properties, and practical usage of all major activation functions — sigmoid, tanh, ReLU, Leaky ReLU, GELU, and Swish.

### Prerequisites

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| Activation function concept | INTRODUCED | DEVELOPED | activation-functions | OK | Student knows what activations are and why they matter |
| Sigmoid and ReLU | INTRODUCED | INTRODUCED | activation-functions | OK | This lesson deepens both |
| Differentiability concept | INTRODUCED | INTRODUCED | gradient-descent | OK | Mentioned in activation properties |

### Misconceptions Addressed

| Misconception | Why They'd Think This | Negative Example | Where Addressed |
|---------------|----------------------|-----------------|-----------------|
| "I need to memorize all activations" | Reference-style lesson | "This is a reference guide — you'll come back to it" | Objective |
| "Activation choice is critical and hard" | Seems like a big decision | "Don't overthink — pick something reasonable and move on" | Decision Guide: InsightBlock |
| "Sigmoid is always bad" | Learned ReLU is the "modern default" | Sigmoid is still the right choice for binary output layers | Decision Guide |

### Examples Used

| Example | Type | Purpose |
|---------|------|---------|
| Sigmoid — formula, shape, range, use | Positive | Historical activation, output-layer use |
| Tanh — formula, shape, range, use | Positive | Zero-centered alternative to sigmoid |
| ReLU — formula, shape, range, use | Positive | Modern default |
| Leaky ReLU — formula, comparison to ReLU | Positive | Fix for dying ReLU |
| GELU and Swish — formula, comparison | Positive | Modern smooth alternatives |
| ActivationFunctionExplorer (3 instances) | Interactive positive | Direct visual comparison |
| Decision guide (4 use cases) | Positive reference | Practical "when to use" summary |

---

## Phase 3: Design

### Narrative Arc

This is a reference lesson, not a conceptual lesson. The narrative is: "here's your visual guidebook for all the activations you'll encounter." Each activation gets the same treatment: formula, 3 ConceptBlocks (range, shape, main use), one-line intuition, and interactive exploration. The lesson ends with a practical decision guide. The "why" behind the differences (vanishing gradients, etc.) is explicitly deferred to after backpropagation.

### Modalities Used

| Modality | What | Rationale |
|----------|------|-----------|
| Symbolic | Formula for each activation (KaTeX) | Precise definition |
| Visual | ConceptBlocks with range/shape/use for each | Consistent visual format |
| Interactive | ActivationFunctionExplorer (3 instances) | Compare shapes interactively |
| Visual | ComparisonRow (ReLU vs Leaky ReLU) | Side-by-side trade-offs |
| Visual | GradientCards (GELU, Swish) | Color-coded modern activations |
| Concrete | Decision guide with 4 color-coded use cases | Practical takeaway |

### Cognitive Load

- **New concepts:** 3 (tanh, Leaky ReLU, GELU/Swish — but these are all variations of a known concept)
- **Previous lesson load:** STRETCH
- **This lesson:** CONSOLIDATE — reference material, low cognitive load per activation
- **Assessment:** Each new activation is a variation on an established pattern. Low effective novelty.

### Scope Boundaries

- Visual shapes and practical usage ONLY
- Deep "why" explanations deferred to after backpropagation
- Vanishing gradients mentioned but NOT explained
- PReLU mentioned in passing (as learnable Leaky ReLU slope)
- Softmax NOT covered (deferred to classification lesson)
- No code implementation
- No comparison of training performance

---

## What Was Actually Built

1. **Header + Objective** — "visual reference guide, come back to it later"
2. **Sigmoid section** — Formula, 3 ConceptBlocks (range, shape, use), intuition
3. **Tanh section** — Formula, 3 ConceptBlocks, vs sigmoid comparison
4. **ActivationFunctionExplorer** — Compare sigmoid and tanh
5. **ReLU section** — Formula, 3 ConceptBlocks, "the surprise" insight
6. **Leaky ReLU section** — Formula, ComparisonRow (ReLU vs Leaky ReLU)
7. **ActivationFunctionExplorer** — Compare ReLU and Leaky ReLU
8. **GELU and Swish section** — Two GradientCards, smooth alternatives description
9. **ActivationFunctionExplorer** — Compare modern activations
10. **Quick Decision Guide** — 4 color-coded use-case cards
11. **Summary** — 4 key takeaways (4th defers "why" to Module 1.3)
12. **ModuleCompleteBlock** — Lists 4 achievements from Module 1.2
13. **Next Step** — Link to home (Module 1.3 preview)

### Components Used
- Row (layout), LessonHeader, ObjectiveBlock, SectionHeader
- InsightBlock (4), TipBlock (2), TryThisBlock (3), ConceptBlock (9)
- ComparisonRow (1), GradientCard (2), SummaryBlock, NextStepBlock, ModuleCompleteBlock
- ActivationFunctionExplorer via ExercisePanel (3 instances)
- KaTeX: BlockMath

### Mental Models Established
- **Sigmoid = squash to (0,1)** — S-curve, output-layer for binary classification
- **Tanh = squash to (-1,1)** — zero-centered S-curve, used in RNNs
- **ReLU = max(0,x)** — default for most networks, simple and fast
- **Leaky ReLU = fix for dying ReLU** — small negative slope prevents neurons from going permanently dead
- **GELU/Swish = smooth ReLU alternatives** — used in transformers and vision
- **"Don't overthink activation choice"** — pick based on architecture, it's rarely the bottleneck

### Analogies Used
- "Confidence score → probability" for sigmoid
- "Less is more" for ReLU's surprising effectiveness
- "Drop-in replacement" for Leaky ReLU

---

## Review — 2026-02-06 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 2
- Polish: 1

### Verdict: NEEDS REVISION

### Findings

### [CRITICAL] — Widget scoping exposes unseen functions

**Location:** All three ActivationFunctionExplorer instances (lines 151, 258, 318)
**Issue:** Each explorer widget renders ALL 7 activation functions in the selector (linear, sigmoid, tanh, relu, leaky-relu, gelu, swish). The first widget appears after only sigmoid and tanh have been introduced. The student sees buttons for ReLU, Leaky ReLU, GELU, Swish — functions they haven't been taught yet. This undermines the carefully sequenced introduction of each function.
**Student impact:** Confusion and distraction. The student might click an unfamiliar function, see a shape they can't interpret, and feel lost. Alternatively, they skip the widget entirely because it's overwhelming.
**Suggested fix:** Use the `visibleFunctions` prop to scope each explorer to only the functions introduced so far:
- First explorer (after sigmoid/tanh): `visibleFunctions={['sigmoid', 'tanh']}`
- Second explorer (after ReLU/Leaky): `visibleFunctions={['relu', 'leaky-relu']}`
- Third explorer (modern): `visibleFunctions={['relu', 'gelu', 'swish']}` (or all 7 as a capstone)

### [IMPROVEMENT] — Leaky ReLU slope inconsistency between lesson and widget

**Location:** Leaky ReLU section (line 218) and ActivationFunctionExplorer widget (line 89 in widget)
**Issue:** The lesson formula shows max(0.01x, x) and the TipBlock says "The small negative slope (0.01)". But the ActivationFunctionExplorer widget uses a slope of 0.1 (ten times larger). When the student explores the widget, the visual won't match the formula they just read.
**Student impact:** Subtle but real confusion. The student might notice the negative slope looks steeper than expected and doubt their understanding.
**Suggested fix:** Either update the widget's Leaky ReLU slope to 0.01 to match the lesson, or update the lesson to say 0.1 and note that the exact value varies. The widget constant is the easier fix since 0.01 is the standard convention.

### [IMPROVEMENT] — Em dash spacing inconsistent with style guide

**Location:** Multiple locations throughout the lesson
**Issue:** The lesson uses spaced em dashes (" — ") in multiple places. The style guide requires no spaces: "word—word" not "word — word". Found in: description (line 44), subtitle (line 72, 112), GELU/Swish description (line 295), and several other locations.
**Student impact:** None pedagogically, but inconsistent with codebase conventions.
**Suggested fix:** Replace all " — " with "—" throughout the lesson.

### [POLISH] — Three identical widget instances are redundant

**Location:** Lines 151, 258, 318
**Issue:** All three ActivationFunctionExplorer instances are configured identically (`defaultFunction="relu"` for two, `defaultFunction="sigmoid"` for one, both with `showDerivatives={false}`). Once scoped with `visibleFunctions`, they'll be differentiated. But even then, three separate widgets for a reference lesson is heavy. Consider whether the third (modern activations) could be merged with a single capstone explorer at the end.
**Student impact:** Minor — the lesson feels a bit repetitive with three separate explorer sections.
**Suggested fix:** After scoping the first two, make the third a capstone that shows all functions together, reinforcing the comparison angle.

### Review Notes

The lesson is well-structured as a reference. The consistent treatment of each function (formula → 3 ConceptBlocks → intuition) creates a predictable pattern that reduces cognitive load. The Decision Guide is the strongest section — practical and memorable. The main issue is the widget scoping: the `ActivationFunctionExplorer` supports a `visibleFunctions` prop that would solve the critical finding immediately. The em dash spacing is a codebase convention issue, not pedagogical.

---

## Review — 2026-02-06 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 0

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — Leaky ReLU visually indistinguishable from ReLU at 0.01 slope

**Location:** Compare ReLU and Leaky ReLU widget (line 258-263) and ActivationFunctionExplorer widget
**Issue:** After fixing the slope to the standard 0.01, Leaky ReLU's negative region produces values so small they're visually identical to ReLU on the canvas (y range -6 to 6). At x=-5, output is -0.05—essentially 0 on the graph. The student is told to "Drag the slider to negative values to see the difference" but there's no visible difference to see.
**Student impact:** The student tries the exercise, sees no difference, and either thinks the widget is broken or that they misunderstand the concept. The TryThisBlock promises a visible difference that doesn't exist.
**Suggested fix:** Use an exaggerated slope (e.g., 0.1) in the widget for visual clarity, while keeping the lesson text at 0.01 as the standard value. Add a note to the TryThisBlock explaining that the widget uses an exaggerated slope for visibility: "The widget uses a larger slope (0.1) so you can see the difference. In practice, the slope is typically 0.01." This is common in educational visualizations—exaggerate for understanding, then note the real value.

### Review Notes

The iteration 1 fixes successfully resolved the critical widget scoping issue and the em dash spacing. The capstone widget approach works well. The remaining issue is a consequence of fixing the Leaky ReLU slope to match convention—the fix was technically correct but created a new pedagogical problem. Educational visualizations should prioritize understanding over exact numerical accuracy.

---

## Review — 2026-02-06 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

### Findings

### [POLISH] — GELU formula in GradientCard uses Φ(x) without definition

**Location:** GELU GradientCard (line 290)
**Issue:** The card shows "x · Φ(x)" but Φ (the standard normal CDF) hasn't been defined. The student might not know what Φ means.
**Student impact:** Minor confusion for the mathematically curious. Most students will focus on the "used in transformers" text and the interactive widget, not the formula.
**Suggested fix:** Could add "(Φ = normal CDF)" in small text, but this might be over-explaining for a reference lesson that explicitly defers the "why" to later. Leave as-is or address at user discretion.

### [POLISH] — Widget formula display shows α=0.1 while lesson text says 0.01

**Location:** Widget formula display for Leaky ReLU
**Issue:** The widget's formula line now shows "LeakyReLU(x) = max(αx, x), α=0.1" while the lesson's KaTeX formula says "max(0.01x, x)". The TryThisBlock explains this, but the dual values across two visible elements could still cause a moment of hesitation.
**Student impact:** Minimal given the explanatory note. The student reads the note and understands.
**Suggested fix:** None needed—the note handles it. This is inherent to the educational exaggeration approach.

### Review Notes

All critical and improvement findings from iterations 1 and 2 have been resolved. The lesson now properly scopes each interactive widget to only show functions already introduced, uses an exaggerated but explained slope for visual clarity, follows em dash conventions, and has a capstone explorer that brings everything together. The lesson is effective as a visual reference guide.

---

## Improvement Summary — 2026-02-06

**Iterations:** 3/3
**Final verdict:** PASS

### Changes made:
- [Iteration 1] Fixed: Scoped all three ActivationFunctionExplorer widgets using `visibleFunctions` prop (sigmoid+tanh, relu+leaky, all). Changed third widget to capstone "All Activations" explorer. Fixed Leaky ReLU slope to 0.01 in widget. Removed spaces around em dashes throughout lesson. Updated TryThisBlock for ReLU/Leaky to remove reference to "Linear" button.
- [Iteration 2] Fixed: Reverted Leaky ReLU slope to 0.1 for visual clarity (0.01 was invisible on canvas). Updated widget formula to show α=0.1. Added explanatory note in TryThisBlock: "The widget uses α=0.1 so the slope is visible. In practice, 0.01 is typical."

### Remaining items (if any):
- GELU formula shows Φ(x) without defining Φ (minor, reference lesson context)
- Widget formula shows α=0.1 vs lesson text 0.01 (explained by note, acceptable)
