# Lesson: Optimizers (Module 1.3, Lesson 5)

**Slug:** `optimizers`
**Type:** BUILD (extends SGD from lesson 4)
**Module position:** 5 of 7

---

## Phase 1: Orient — Student State

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Gradient descent update rule | DEVELOPED | gradient-descent (1.1) | theta_new = theta_old - alpha * grad_L; ball-on-hill analogy |
| Learning rate (alpha) | DEVELOPED | learning-rate (1.1) | Step size, Goldilocks zone, oscillation/divergence failure modes |
| Learning rate schedules | MENTIONED | learning-rate (1.1) | Step/exponential/cosine/warmup previewed, not explained |
| Loss landscape | INTRODUCED | loss-functions (1.1) | Bowl shape, valley = minimum; extended to 2D contour in SGD widget |
| Training loop (forward-loss-backward-update) | DEVELOPED | implementing-linear-regression (1.1) | "Heartbeat of training," 6-step StepList |
| Mini-batch gradient computation | DEVELOPED | batching-and-sgd (1.3) | Polling analogy; same formula but over B random samples |
| Stochastic gradient descent (SGD) | DEVELOPED | batching-and-sgd (1.3) | Mini-batch SGD as the default; spectrum from batch=1 to batch=ALL |
| Gradient noise as beneficial | INTRODUCED | batching-and-sgd (1.3) | Noisy gradients help escape sharp minima; "the hill is shaking" |
| Sharp vs wide minima | INTRODUCED | batching-and-sgd (1.3) | Sharp = fragile, wide = robust; better generalization |
| Batch size / learning rate interaction | MENTIONED | batching-and-sgd (1.3) | "Larger batches can tolerate larger learning rates"; planted for this lesson |
| Backpropagation algorithm | DEVELOPED | backpropagation (1.3) | Forward + backward = all gradients |
| Computational graph notation | DEVELOPED | computational-graphs (1.3) | Nodes = ops, edges = data flow |

**Established mental models available:**
- "Ball rolling downhill" (gradient descent, 1.1) — the core spatial metaphor for optimization
- "Loss landscape = bowl with valley" (loss functions, 1.1) — extended to 2D contour in SGD widget
- "The ball is still rolling downhill, but now the hill is shaking" (batching-and-sgd) — noise from mini-batches
- "Noise as a feature, not a bug" (batching-and-sgd) — gradient noise escapes sharp minima
- "Goldilocks" (learning rate, 1.1) — too big = oscillation, too small = slow

**NOT covered in prior lessons (relevant here):**
- Per-parameter learning rates (no lesson has taught this)
- Exponential moving averages (no lesson has taught this)
- Adaptive learning rates (explicitly deferred from learning-rate lesson: "adaptive LR (Adam)" listed as NOT covered)
- Second-order optimization / curvature (never mentioned)

**Readiness assessment:** Student is well-prepared. They have SGD at DEVELOPED depth with an interactive widget showing the 2D loss landscape. They understand gradient noise, batch sizes, and learning rate tradeoffs. The "batch size / learning rate interaction" was deliberately planted as a seed for this lesson. The ball-on-hill analogy and loss landscape mental model give a strong spatial foundation to build optimizer intuitions on.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand why vanilla SGD struggles on certain loss landscapes and how momentum, RMSProp, and Adam fix those problems by adapting gradient updates.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent (1.1) | OK | Need to modify the update rule; student can write theta_new = theta_old - alpha * grad |
| Learning rate (alpha) | DEVELOPED | DEVELOPED | learning-rate (1.1) | OK | Need to understand why a single global LR is limiting |
| SGD (mini-batch) | DEVELOPED | DEVELOPED | batching-and-sgd (1.3) | OK | This lesson extends SGD; student has worked with the SGD widget |
| Gradient noise | INTRODUCED | INTRODUCED | batching-and-sgd (1.3) | OK | Need recognition that gradients are noisy; don't need to compute noise statistics |
| Loss landscape (2D) | INTRODUCED | INTRODUCED | loss-functions (1.1), batching-and-sgd (1.3) | OK | Need to visualize contours and think about terrain shape; student has done this in the SGD widget |
| Sharp vs wide minima | INTRODUCED | INTRODUCED | batching-and-sgd (1.3) | OK | Context for why optimizer choice matters; recognition-level is sufficient |
| Exponential moving average | INTRODUCED | MISSING | — | MISSING | Momentum and Adam both use exponential moving averages; must teach this |
| Per-parameter adaptation | INTRODUCED | MISSING | — | MISSING | RMSProp and Adam adapt LR per parameter; must teach this concept |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Exponential moving average (EMA) | Medium (related concept "running average" is intuitive, but EMA formula is new) | Dedicated section within this lesson. Teach EMA through a concrete analogy: weather forecasting (today's temperature matters more than last week's). Show the formula v_t = beta * v_{t-1} + (1-beta) * g_t with a concrete numerical walkthrough. This is a prerequisite for understanding momentum. |
| Per-parameter adaptation | Small (student already knows parameters have different gradients from backprop) | Brief motivating section: "ravine problem" — some parameters need big steps, others need small steps. One gradient magnitude example showing 100x difference between parameters. This motivates why a single learning rate is insufficient. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Adam always beats SGD" | Adam is the default in tutorials/frameworks; it converges faster initially; "adaptive = smart = better" | Research results: SGD with tuned LR + momentum often generalizes better than Adam on image classification (e.g., ResNets). Adam can converge to sharper minima. The "faster convergence" of Adam doesn't mean "better final model." | Elaborate section — after teaching Adam, show the tradeoff explicitly with a ComparisonRow |
| "Momentum just makes training faster" | Momentum does speed up training on smooth surfaces, so the student might think that's all it does | Ravine problem: without momentum, SGD oscillates across the narrow dimension and barely progresses along the long dimension. Momentum doesn't just go faster — it changes WHICH DIRECTION the optimizer moves by accumulating a directional signal. | During momentum explanation — show ravine contour plot where vanilla SGD zigzags but momentum cuts through |
| "Adam has no hyperparameters to tune" | Adam is marketed as "adaptive" and has good defaults (lr=0.001, beta1=0.9, beta2=0.999); students assume adaptive = no tuning needed | Learning rate still matters enormously for Adam. beta values matter for specific domains. Weight decay interaction (AdamW) is crucial. Adam with default LR on a new problem can still diverge or converge to bad solutions. | After Adam explanation — "Adam's defaults are good starting points, not universal solutions" |
| "More sophisticated optimizer = always better" | Natural assumption: newer/more complex must be an upgrade | SGD+momentum is still the default for many state-of-the-art vision models. Simpler optimizers have fewer things that can go wrong. Extra state in Adam means 3x memory per parameter. | Elaborate section — "no free lunch" framing |
| "The learning rate means the same thing across optimizers" | Student learned LR = step size in vanilla SGD; assumes same number works everywhere | lr=0.01 for SGD vs lr=0.001 for Adam — same value produces wildly different behavior. Adam internally rescales gradients, so its effective step size is different from the stated LR. | During Adam explanation — explicit callout |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Elongated ravine (2D contour) | Positive | Show the problem momentum solves: vanilla SGD oscillates across the narrow dimension and crawls along the long dimension. Momentum smooths this out. | The ravine is the canonical motivation for momentum. Visually dramatic — zigzag vs smooth path. Connects directly to the 2D contour the student already used in the SGD widget. |
| Per-parameter gradient magnitudes (w1 gradient = 0.001, w2 gradient = 50) | Positive | Show why a single learning rate can't serve both parameters well. lr=0.01 is too big for w2 (overshoots) and too small for w1 (crawls). | Concrete numbers make the problem tangible. Motivates RMSProp's per-parameter scaling. |
| EMA with concrete numbers (5 gradient values, beta=0.9) | Positive | Walk through the EMA formula step by step with real numbers so the student can verify the "recent values matter more" claim numerically. | Abstract formula becomes concrete. Student can trace each step. Builds the foundation for momentum AND Adam. |
| "Adam on everything" counter-example | Negative | Show that Adam with defaults on a well-tuned SGD+momentum problem can actually generalize worse. The "always use Adam" heuristic fails. | Directly addresses the module-level misconception "Adam always beats SGD." |
| Adam rescales gradients (same LR, different behavior) | Negative | SGD with lr=0.01 vs Adam with lr=0.01 — totally different training curves. The learning rate is NOT interchangeable between optimizers. | Prevents the "plug in optimizer, keep same LR" mistake that is extremely common in practice. |

---

## Phase 3: Design

### Narrative Arc

You just learned how mini-batch SGD trains a neural network — take a random sample, compute the gradient, step downhill, repeat. And you saw it work on a 2D loss landscape. But there's a problem: not all loss landscapes are created equal. Imagine a long, narrow valley — like a canyon with steep walls but a gentle downward slope. Vanilla SGD will bounce back and forth between the canyon walls (the gradient is steep there) while barely making progress along the canyon floor (the gradient is gentle there). This is not a contrived problem — most real neural networks have loss landscapes with exactly this shape, because different parameters operate at wildly different scales. This lesson introduces three ideas — momentum, RMSProp, and Adam — that each solve a different aspect of this problem. By the end, you'll understand what Adam is actually doing (not just "use Adam"), when SGD might be the better choice, and why there's no single "best" optimizer.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual/Spatial** | 2D contour plot showing paths of different optimizers on the ravine landscape. Color-coded trajectories: vanilla SGD (zigzag), SGD+momentum (smooth curve), Adam (adaptive path). | The student already has spatial intuition from the SGD widget. Extending to show optimizer paths makes the differences viscerally clear — you SEE the zigzag vs the smooth path. |
| **Concrete example** | EMA walkthrough with 5 gradient values: [2.0, -1.5, 3.0, -0.5, 2.5] at beta=0.9. Table showing v_t at each step. Then the same values used in momentum update. | The formula v_t = beta*v_{t-1} + (1-beta)*g_t is abstract. Walking through 5 concrete values makes it tangible and verifiable. |
| **Symbolic** | Update rules side-by-side: SGD, SGD+Momentum, RMSProp, Adam. KaTeX formulas in a structured comparison, each annotated with what each term does. | The formulas are the ground truth. Having them side-by-side lets the student see how each optimizer adds one more piece to the SGD update rule. |
| **Analogy/Verbal** | "Bowling ball vs tennis ball" for momentum (heavy ball has inertia, rolls through local bumps). "Volume knob per instrument" for RMSProp (each parameter gets its own volume control). "Bowling ball with volume knobs" for Adam (combines both ideas). | These physical analogies map to the mathematical operations and make them memorable. Each builds on the previous one. |
| **Intuitive** | "Why does this make sense?" check after each optimizer: if gradients keep pointing the same direction, momentum amplifies them. If a parameter's gradients are consistently large, RMSProp shrinks its learning rate. Adam does both. | The "of course" feeling — once stated, the logic is obvious. Prevents the student from seeing these as arbitrary tricks. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2 genuinely new concepts — (1) Momentum (EMA of gradients to smooth updates), (2) Adaptive per-parameter learning rates (RMSProp/Adam). Adam is presented as the combination of these two ideas, not as a third independent concept.
- **Supporting concept taught:** Exponential moving average (EMA) — taught as a tool, not as a concept to remember independently. It's a means to understand momentum/Adam.
- **Previous lesson load:** STRETCH (batching-and-sgd introduced 3 new concepts: mini-batches, SGD, epochs)
- **This lesson's load:** BUILD — appropriate. The student has a solid foundation from the STRETCH lesson. This lesson extends SGD rather than introducing a completely new framework. Two new concepts is well within the 2-3 limit.

### Connections to Prior Concepts

- **Gradient descent update rule** (1.1, DEVELOPED): The core formula theta_new = theta_old - alpha * grad. Every optimizer in this lesson is a modification of this rule. "Same heartbeat, new embellishments."
- **Ball-on-hill analogy** (1.1): Extends naturally — momentum = the ball is heavier and has inertia. "The bowling ball keeps rolling through small bumps."
- **Loss landscape** (1.1, 1.3): The 2D contour from the SGD widget is reused but with an elongated ravine shape to motivate momentum. Same visual language, new terrain.
- **"The hill is shaking" analogy** (batching-and-sgd): Momentum smooths out the shaking. "If the hill is shaking, a heavier ball is less affected by individual vibrations."
- **Learning rate failure modes** (1.1): Oscillation and divergence. Momentum can make oscillation worse if LR is too high. RMSProp prevents the "too big for some, too small for others" problem.
- **"Batch size / learning rate interaction"** seed (batching-and-sgd): This lesson pays off that planted seed — optimizers interact with batch size and LR choices.

**Potentially misleading prior analogies:**
- "Ball rolling downhill" could suggest momentum is just about speed (going faster). Need to emphasize that momentum also changes DIRECTION — the accumulated velocity has a directional component that cuts through oscillation.

### Scope Boundaries

**This lesson IS about:**
- Why vanilla SGD struggles (ravine problem, per-parameter scale differences)
- Momentum: what it is, how it works, why it helps
- RMSProp: per-parameter adaptive learning rates
- Adam: momentum + RMSProp combined
- When to use which optimizer (practical guidance)
- Depth target: Momentum at DEVELOPED, Adam at DEVELOPED, RMSProp at INTRODUCED

**This lesson is NOT about:**
- Implementing optimizers in code (deferred to PyTorch series)
- Learning rate schedules (previewed in 1.1, will be covered when relevant)
- Second-order methods (Newton's method, L-BFGS) — too advanced
- AdamW / weight decay correction (lesson 7 covers weight decay)
- Convergence proofs or theoretical guarantees
- Other optimizers (Adagrad, Adadelta, NAG, LAMB, etc.) — mentioned at most
- Hyperparameter tuning strategies

### Lesson Outline

**1. Context + Constraints**
- "This lesson: understand what modern optimizers do and why. NOT: implement them or tune them."
- ConstraintBlock: "We focus on 3 optimizers: Momentum, RMSProp, Adam. There are dozens of others — these 3 cover the core ideas."

**2. Hook — The Ravine Problem (demo)**
- Start with the 2D contour from the SGD widget — but now with an elongated ravine shape instead of two symmetric minima.
- Show vanilla SGD path: dramatic zigzag. "Look at how much wasted motion. The gradient keeps pointing across the ravine, not along it."
- Question: "What if the ball could remember where it's been going?"
- This is a demo hook — the student sees the problem viscerally before hearing the solution.

**3. EMA Primer — The Tool We Need**
- "Before we fix SGD, we need one mathematical tool: the exponential moving average."
- Weather analogy: predicting tomorrow's temperature. You could use the average of all past days (slow to react), or just today's reading (too noisy). EMA: mostly trust the recent history, but let today's reading nudge it.
- Formula: v_t = beta * v_{t-1} + (1 - beta) * x_t
- Concrete walkthrough: 5 gradient values, beta=0.9, table showing v_t at each step.
- Key insight: beta controls memory. beta=0.9 means "90% old, 10% new." Higher beta = smoother, slower to react. Lower beta = noisy, quick to react.
- This is a dedicated section (medium gap resolution) — teach EMA thoroughly here so momentum and Adam explanations are clean.

**4. Momentum — "Give the Ball Some Weight" (Explain)**
- Analogy: bowling ball vs tennis ball. The bowling ball has inertia — it doesn't change direction on every small bump.
- Formula: v_t = beta * v_{t-1} + g_t, then theta = theta - alpha * v_t
- "This is just EMA applied to gradients, then using the smoothed gradient instead of the raw gradient."
- Return to the ravine: show momentum's path. The across-ravine oscillations CANCEL OUT (positive, negative, positive, negative average to ~0). The along-ravine gradients ACCUMULATE (all pointing the same direction).
- Why it works (intuitive): consistent gradient direction = momentum amplifies. Oscillating gradient = momentum dampens. The ball "learns" the right direction.

**5. Check 1 — Predict and Verify**
- "If the ravine curves to the right at the bottom, what happens to the momentum ball?" (It overshoots the curve — momentum has inertia, which is a downside on curved landscapes.)
- Collapsible reveal with explanation.

**6. RMSProp — "A Volume Knob for Each Parameter" (Explain)**
- Motivating problem: gradient for w1 is 0.001, gradient for w2 is 50. One learning rate can't serve both.
- Analogy: mixing a song — each instrument (parameter) gets its own volume knob (learning rate).
- Formula: s_t = beta * s_{t-1} + (1-beta) * g_t^2, then theta = theta - alpha * g_t / sqrt(s_t + epsilon)
- "Parameters with consistently large gradients get a smaller effective learning rate. Parameters with small gradients get a bigger one."
- Key insight: RMSProp is dividing by the recent magnitude of gradients. Large gradients → large divisor → smaller step. Self-normalizing.

**7. Adam — "The Best of Both Worlds" (Explain)**
- "What if we combine momentum (smooth the gradient direction) with RMSProp (normalize the gradient magnitude)?"
- Formula: m_t (momentum) + v_t (RMSProp) + bias correction + combined update
- Show the formulas side by side: SGD, Momentum, RMSProp, Adam — each one adds a piece.
- Bias correction: brief explanation — "the averages start at zero, so early steps are too small. Bias correction fixes the cold start."
- Practical defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
- Connect: "Adam's learning rate is NOT the same as SGD's learning rate." lr=0.01 for SGD vs lr=0.001 for Adam — explicit example.

**8. Explore — OptimizerExplorer Widget**
- Interactive 2D contour plot (extends the SGD widget concept).
- Optimizer selector: Vanilla SGD, SGD+Momentum, RMSProp, Adam.
- Show trajectory on the ravine landscape for each optimizer.
- Learning rate slider to see how each optimizer responds to LR changes.
- Side panel or bottom panel showing live update values (v_t, s_t for Adam).
- Guided experiments:
  1. "Run vanilla SGD. Watch the zigzag."
  2. "Switch to Momentum. Watch the path smooth out."
  3. "Try RMSProp. Notice how it adapts to the ravine shape differently."
  4. "Now try Adam. It combines both effects."
  5. "Increase the learning rate. Which optimizer handles it most gracefully?"

**9. Elaborate — "Adam Always Beats SGD" (Misconception)**
- ComparisonRow: "Fast convergence" (Adam, green) vs "Better generalization" (SGD+momentum, blue)
- Key point: Adam converges faster to A minimum, but SGD+momentum sometimes finds a BETTER minimum.
- "In practice: Adam is a great default. But for final performance on well-understood problems, SGD+momentum with a tuned LR schedule often wins."
- Connect to sharp vs wide minima from batching-and-sgd: Adam can converge to sharper minima because its adaptive LR lets it navigate into tight spaces that SGD would bounce out of.

**10. Check 2 — Transfer Question**
- "You're training a new model. Adam converges in 50 epochs but SGD hasn't converged after 100. Your colleague says 'Adam is clearly better.' What might they be missing?"
- Answer: convergence speed != final quality. Check validation loss, not just training loss. SGD might end up at a better minimum if given enough time with a good LR schedule.

**11. Summary**
- SummaryBlock with key takeaways:
  - Momentum smooths gradient direction (EMA of gradients) — helps with ravines and noise
  - RMSProp adapts learning rate per parameter (EMA of squared gradients) — helps when parameters have different scales
  - Adam = Momentum + RMSProp — good default, but not always the best final answer
  - No free lunch: faster convergence != better generalization
  - The learning rate means different things for different optimizers

**12. Next Step**
- NextStepBlock pointing to `training-dynamics` (lesson 6): "Now you know how to optimize. Next: why training sometimes fails completely — vanishing gradients, exploding gradients, and the initialization strategies that prevent them."

---

## Widget Specification: OptimizerExplorer

**Type:** New widget, extends the visual language of SGDExplorer

**Core interaction:**
- 2D contour loss landscape (elongated ravine shape — elliptical, NOT the two-minima from SGD widget)
- Radio/tab selector for optimizer: Vanilla SGD | Momentum | RMSProp | Adam
- Play/Pause/Step/Reset controls (same as SGDExplorer)
- Learning rate slider
- Animated trajectory showing the optimization path with a trail
- Live stats panel showing current values (loss, gradient, velocity, etc.)

**Key visual behaviors:**
- Vanilla SGD: visible zigzag across the ravine
- Momentum: smooth, sweeping path that cuts through the ravine
- RMSProp: adapts step sizes visibly — shorter steps in steep directions, longer in flat
- Adam: combines both — smooth AND adaptive

**Reuse from SGDExplorer:**
- Contour rendering approach (heatmap + contour lines)
- Play/Pause/Step/Reset control bar
- useContainerWidth pattern
- Animation loop structure

**New loss landscape:**
- Elongated elliptical ravine: lossAt(x, y) with much higher curvature in one direction than the other
- Single minimum (unlike the two-minima SGD landscape) — keeps focus on optimizer behavior, not noise-based escape
- Starting point placed on the rim of the ravine where the zigzag behavior is most visible

**Guided experiment labels:**
- Numbered experiments in TryThisBlock sidebar, similar to SGDExplorer's experiment approach

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (EMA and per-parameter adaptation addressed as gaps with resolution plans)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (ravine problem before solution)
- [x] At least 3 modalities planned for the core concept (visual/spatial, concrete example, symbolic, analogy, intuitive = 5)
- [x] At least 2 positive examples + 1 negative example (3 positive + 2 negative = 5)
- [x] At least 3 misconceptions identified with negative examples (5 identified)
- [x] Cognitive load = 2 new concepts (within limit of 3)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Depth Changes After This Lesson

| Concept | Depth | Notes |
|---------|-------|-------|
| Momentum | DEVELOPED | EMA of gradients; formula + concrete walkthrough + interactive widget + ravine visualization |
| Adam optimizer | DEVELOPED | Momentum + RMSProp combined; formula + widget + defaults + "not always best" nuance |
| RMSProp | INTRODUCED | Per-parameter adaptive LR; formula + analogy + motivation, but no independent deep practice |
| Exponential moving average | DEVELOPED | Taught as a tool; formula + concrete 5-step walkthrough + used in momentum and Adam |
| Per-parameter learning rates | INTRODUCED | Motivated by gradient magnitude example; realized through RMSProp/Adam |
| Learning rate (revisited) | DEVELOPED (reinforced) | Extended: LR means different things for different optimizers; explicit SGD vs Adam comparison |
| Gradient noise (revisited) | INTRODUCED (reinforced) | Momentum smooths noise; connects to "hill is shaking" analogy |

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 2
- Improvement: 4
- Polish: 3

### Verdict: MAJOR REVISION

Critical findings exist around a formula inconsistency that will confuse students and a missing misconception treatment. Fix critical issues, then re-review.

### Findings

#### [CRITICAL] — Momentum formula contradicts the EMA it claims to use

**Location:** Section 4 (Momentum), formula block at line 311
**Issue:** The EMA section teaches `v_t = beta * v_{t-1} + (1-beta) * x_t` and then the lesson explicitly frames momentum as "just EMA applied to gradients." But the momentum formula shown is `v_t = beta * v_{t-1} + g_t`—classical momentum, which does NOT include the `(1-beta)` factor. These are mathematically different operations. The lesson narrator says "we keep a running average of past gradients" but the formula is an accumulated sum, not an average. The widget implementation at line 186 also uses the classical form (`vx = BETA1 * state.vx + gx`), matching the lesson formula but contradicting the EMA framing.
**Student impact:** The student just spent a section learning EMA with a careful table walkthrough. They arrive at momentum expecting to see the same formula with gradients plugged in. Instead the `(1-beta)` factor vanishes with no explanation. The student will either (a) think the `(1-beta)` was accidentally dropped, (b) think EMA and momentum are different after all, or (c) lose trust in the explanation. Any of these outcomes undermines learning.
**Suggested fix:** Choose one consistent framing. Option A: Use the "EMA-style" momentum formulation `v_t = beta * v_{t-1} + (1-beta) * g_t` in both the lesson and widget, which is mathematically equivalent to classical momentum up to a scaling factor absorbed into the learning rate. This matches the EMA teaching and the plan's intent. Option B: Keep classical momentum but add an explicit callout: "Notice this looks like EMA but without the (1-beta) factor. Classical momentum accumulates rather than averages—the learning rate compensates for the different scale." Either way, the current silent discrepancy must be resolved.

#### [CRITICAL] — "Adam has no hyperparameters" misconception planned but not addressed

**Location:** Missing from the lesson entirely
**Issue:** The planning document identifies 5 misconceptions. The "Adam always beats SGD" misconception is addressed in section 9. The "momentum just makes training faster" misconception is addressed in section 4's WarningBlock. The "learning rate means the same across optimizers" misconception is addressed in the amber callout box in section 7. The "more sophisticated = always better" misconception is addressed in section 9's WarningBlock aside. However, the "Adam has no hyperparameters to tune" misconception—explicitly planned as misconception #3—is completely absent. The lesson lists Adam's defaults but never says "these are starting points, not universal solutions" or addresses the student's likely assumption that adaptive = no tuning.
**Student impact:** The student leaves thinking Adam's defaults are universal. This is a common and damaging misconception in practice—students who believe this will never tune Adam's hyperparameters, leading to suboptimal training and confusion when defaults fail.
**Suggested fix:** Add a brief callout after the Adam defaults list (around line 507). A TipBlock or WarningBlock stating that Adam's defaults are good starting points but not universal—learning rate still matters enormously, beta values matter for specific domains, and Adam with default LR on a new problem can still diverge.

#### [IMPROVEMENT] — EMA-to-momentum transition is narratively abrupt

**Location:** Transition between section 3 (EMA) and section 4 (Momentum)
**Issue:** The EMA section ends with "a smoothed signal that filters out the noise while preserving the trend." The momentum section opens with the bowling ball analogy. But there is no explicit bridge: "Now let's apply this EMA idea to gradient descent." The student has to infer the connection. The plan specifies this connection ("This is just EMA applied to gradients, then using the smoothed gradient instead of the raw gradient"), but the built lesson skips it.
**Student impact:** The student understands EMA and understands momentum separately, but the "aha" of seeing that momentum IS EMA applied to gradients is weakened. The connection exists implicitly but should be explicit per the Connection Rule.
**Suggested fix:** Add one sentence before or after the momentum formula: "This is the EMA idea applied to gradients—instead of using today's raw gradient, we use the smoothed running average." (This also interacts with the Critical finding about the formula mismatch—fixing both together is ideal.)

#### [IMPROVEMENT] — RMSProp section lacks a concrete numerical walkthrough

**Location:** Section 6 (RMSProp)
**Issue:** The EMA section has a detailed 5-step numerical table. The momentum section has the ComparisonRow showing +5/-5 oscillation canceling. But RMSProp gets only the formula and a verbal explanation of the self-normalizing property. The planned per-parameter gradient magnitudes example (w1=0.001, w2=50) appears as the motivating GradientCard, but the actual RMSProp formula is never walked through with numbers. The student sees the division by sqrt(s_t) but never traces a concrete step.
**Student impact:** The student understood EMA because of the table. They will understand momentum because of the concrete oscillation example. RMSProp's formula is arguably more complex (squaring, square root, epsilon), and the student has no concrete anchor for it. The depth target for RMSProp is only INTRODUCED, so a full walkthrough may be excessive, but at least one concrete step (e.g., "If g_t = 50, then s_t grows large, and 50/sqrt(large) gives a small step") would ground the formula.
**Suggested fix:** Add a brief 1-2 step concrete example after the formula showing what happens for the large-gradient parameter (g=50) and the small-gradient parameter (g=0.001). This doesn't need a full table—just enough to make the self-normalizing claim tangible.

#### [IMPROVEMENT] — Widget does not expose internal optimizer state

**Location:** OptimizerExplorer widget
**Issue:** The planning document specifies: "Side panel or bottom panel showing live update values (v_t, s_t for Adam)." The built widget shows step count, loss, optimizer name, and position—but NOT the velocity (v_t), second moment (s_t), or bias-corrected values. These internal states are what distinguish the optimizers. Without them, the student can only observe the path difference, not understand WHY the path differs.
**Student impact:** The widget demonstrates that the paths are different but doesn't help the student connect path behavior to the mathematical mechanism. Seeing v_t accumulate along the ravine direction while oscillating cross-ravine would be the "aha" for momentum. Seeing s_t grow for the steep direction in RMSProp would ground the per-parameter adaptation.
**Suggested fix:** Add a small stats section below the existing stats showing the key internal values: for Momentum show vx/vy; for RMSProp show sx/sy; for Adam show all four plus the bias-corrected versions. Even a simplified version showing just the x and y components of the velocity and second moment would be valuable.

#### [IMPROVEMENT] — The "hill is shaking" connection to momentum is missing

**Location:** Sections 3-4 (EMA and Momentum)
**Issue:** The planning document explicitly calls out the connection: "If the hill is shaking, a heavier ball is less affected by individual vibrations." The module record lists "The ball is still rolling downhill, but now the hill is shaking" as a key mental model from batching-and-sgd. The built lesson never makes this connection. The lesson refers to "Batching and SGD" in the hook but only about the loss landscape, not about noise. The Reinforcement Rule requires revisiting concepts from more than 3 lessons ago—gradient noise was INTRODUCED 1 lesson ago, so it's fresh, but the explicit connection "momentum smooths the shaking from mini-batches" is pedagogically important and was planned.
**Student impact:** The student sees momentum as solving the "ravine problem" but misses that it also helps with the noise problem from mini-batches that they learned about in the previous lesson. This is a missed opportunity to reinforce both concepts simultaneously.
**Suggested fix:** Add one sentence in the momentum section connecting to gradient noise: "Momentum also smooths out the mini-batch noise you learned about in Batching and SGD—the heavy bowling ball is less affected by individual vibrations on the shaking hill."

#### [POLISH] — Widget `else` block in canvas rendering

**Location:** OptimizerExplorer.tsx, line 310
**Issue:** There is an `else` block in the canvas pixel-rendering loop. The codebase convention prohibits `else` in favor of early returns. In this case, it's inside a tight rendering loop where early return doesn't apply (it's an if/else assignment pattern within a for loop, not a function body), so this is technically borderline. However, it could be restructured using a ternary or a helper function to avoid the else.
**Student impact:** None (code quality only).
**Suggested fix:** Refactor to use a helper function or ternary. Low priority since this is rendering code and the pattern is clear.

#### [POLISH] — EMA concrete example could note the bias/cold-start issue

**Location:** Section 3 (EMA), the concrete walkthrough table
**Issue:** The EMA table shows v_1 = 0.20, which is much lower than the input of 2.0. This is the cold-start problem that Adam's bias correction fixes. The table is a perfect place to plant this observation ("Notice v_1 = 0.20, much smaller than the input 2.0—the running average starts cold"), which would make the bias correction in the Adam section feel motivated rather than appearing from nowhere.
**Student impact:** Minor—the bias correction is explained in the Adam section. But planting the seed earlier would make the later explanation click faster.
**Suggested fix:** Add a brief note after the table or as an aside: "Notice how v_1 is much smaller than the actual gradient (0.20 vs 2.0). The EMA needs time to 'warm up.' We'll fix this when we get to Adam."

#### [POLISH] — Check understanding sections use custom markup instead of a block component

**Location:** Sections 5 and 10 (Check 1 and Check 2)
**Issue:** Both check-your-understanding sections use inline `<div>` with manual border/bg styling (`border-2 border-primary/30 bg-primary/5`) and a `<details>/<summary>` pattern. Other lessons in the codebase may have an established block component for this pattern. If not, this inline approach is fine, but it creates a pattern that should be consistent across lessons.
**Student impact:** None—the visual presentation works.
**Suggested fix:** If a CheckBlock or similar component exists, use it. If not, this is acceptable as-is but should be noted for potential future extraction into a reusable component.

### Review Notes

**What works well:**
- The narrative arc is strong. The ravine hook is viscerally compelling and motivates the entire lesson.
- The EMA section with the concrete walkthrough table is excellent pedagogy—it makes the abstract formula tangible.
- The ComparisonRows (across vs along ravine, Adam vs SGD+momentum) are effective and well-placed.
- The bowling ball analogy is memorable and extends naturally from the established ball-on-hill mental model.
- The SVG ravine illustration with SGD zigzag vs momentum smooth path is a good visual anchor before the widget.
- The "Adam Always Wins" myth section with the transfer question is strong—it addresses the most common practical misconception.
- The widget is well-built: good contour rendering, responsive sizing, clean animation controls, loss curve display.

**Systemic pattern:**
The formula inconsistency (Critical #1) suggests the builder may have used the classical momentum formulation (which is common in textbooks and frameworks) without fully reconciling it with the EMA-first teaching approach in the plan. This is an understandable deviation, but it needs explicit resolution because the lesson's pedagogical structure depends on EMA being the unifying thread.

**Widget quality:**
The OptimizerExplorer is technically solid—correct gradient computation, proper bias correction for Adam, good UI. The missing internal state display is the main gap between what was planned and what was built. The loss curve is a nice addition not explicitly in the plan.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: NEEDS REVISION

All critical findings from iteration 1 are resolved. One improvement-level finding remains around the widget's use of a ref for internal state display, which produces correct output today but is architecturally fragile. Two polish items are carried forward from iteration 1 (EMA cold-start seed and check-understanding components).

### Verification of Iteration 1 Fixes

**CRITICAL #1 (Momentum formula contradiction): RESOLVED.**
The sky-blue callout at line 328 explicitly addresses the discrepancy between the EMA formula and classical momentum. It explains that classical momentum accumulates rather than averages, and the scaling difference gets absorbed into the learning rate. It also notes that Adam uses the EMA form. The student gets the answer to the question they would naturally ask. The opening sentence of the momentum section ("Replace x_t with today's gradient, and you get momentum") does create a brief expectation that the EMA formula will appear unchanged, but the immediate callout catches this. Acceptable resolution.

**CRITICAL #2 ("Adam has no hyperparameters" misconception): RESOLVED.**
A rose-colored callout at line 562 states: "These defaults are starting points, not universal solutions." It explicitly says the learning rate still matters enormously, beta values matter for specific domains, and Adam with a bad LR will diverge or stall just like SGD. This directly addresses the planned misconception.

**IMPROVEMENT #3 (EMA-to-momentum bridge): RESOLVED.**
Line 298-301 opens the momentum section with: "Now let's put EMA to work. Remember [EMA formula]? Replace x_t with today's gradient, and you get momentum." This is exactly the bridge the plan called for.

**IMPROVEMENT #4 (RMSProp numerical walkthrough): RESOLVED.**
A table at lines 477-502 shows the RMSProp computation for both w1 (g=0.001) and w2 (g=50) with concrete numbers. The math is correct: both parameters end up with ~0.32 step size despite 50,000x gradient difference. The "self-normalizing" claim is now grounded in traceable arithmetic.

**IMPROVEMENT #5 (Widget internal state display): RESOLVED with caveat.**
Lines 633-669 show an "Internal State" section that displays vx/vy for momentum/adam and sx/sy for rmsprop/adam. Includes contextual explanation text per optimizer. However, see finding below about the ref-based rendering approach.

**IMPROVEMENT #6 ("Hill is shaking" connection): RESOLVED.**
Lines 308-309: "Remember the mini-batch noise that makes the hill shake? Momentum smooths that out—a heavier ball is less affected by vibrations." This directly connects to the batching-and-sgd mental model.

**POLISH #7 (else block in widget): RESOLVED.**
No `else` blocks remain in OptimizerExplorer.tsx. Confirmed via grep.

### Findings

#### [IMPROVEMENT] — Widget internal state display reads from ref, not reactive state

**Location:** OptimizerExplorer.tsx, lines 634-669
**Issue:** The internal state panel reads optimizer values from `optStateRef.current.vx`, `.vy`, `.sx`, `.sy`. These are ref values, not React state. Refs do not trigger re-renders when mutated. The display currently updates correctly because the animation loop also calls `setStepCount`, `setPath`, and `setLossHistory`—these state updates trigger re-renders, and during re-render the JSX reads the latest ref values. However, this is a coupling dependency rather than an explicit contract. If anyone refactors `doStep` to skip one of those state setters (e.g., throttling path updates for performance), the internal state display silently goes stale. The step-by-step "Step" button also works because `doStep()` triggers all three state setters.
**Student impact:** None today. The display works correctly. But it is architecturally fragile and could silently break under future refactoring. For a teaching widget where the internal state display is the key pedagogical contribution, reliability of this display matters more than in a typical UI element.
**Suggested fix:** Extract the relevant values from `optStateRef.current` into React state alongside the other state updates in `doStep`. For example, add a `optimizerInternalState` state variable that gets set in `doStep` with `{ vx, vy, sx, sy }`, and read from that in the JSX. This makes the display self-sufficient and removes the implicit dependency on other state updates causing re-renders.

#### [POLISH] — EMA cold-start observation not planted (carried from iteration 1)

**Location:** Section 3 (EMA), the concrete walkthrough table
**Issue:** The EMA table shows v_1 = 0.20 from input 2.0. This is the cold-start problem that Adam's bias correction fixes. Planting a brief observation here ("Notice v_1 is much smaller than the input—the EMA needs time to warm up. We'll fix this when we get to Adam.") would make the bias correction in the Adam section feel motivated rather than appearing from nowhere. This was noted in iteration 1 but not addressed in the fix pass.
**Student impact:** Minor. The bias correction is explained adequately in the Adam section ("since m_0 = 0 and v_0 = 0, the first few estimates are too small"). But the seed would make it click faster.
**Suggested fix:** Add one sentence or an aside note after the EMA table connecting the small v_1 to the cold-start problem.

#### [POLISH] — Check-understanding sections use inline markup (carried from iteration 1)

**Location:** Sections 5 and 10 (Check 1 and Check 2)
**Issue:** Both use inline `<div>` with manual styling and `<details>/<summary>`. If a reusable block component for check-your-understanding patterns is established in the future, these should be migrated. Currently acceptable.
**Student impact:** None.
**Suggested fix:** Note for future component extraction. No action needed now.

### Review Notes

**What improved since iteration 1:**
- The momentum formula discrepancy is now handled cleanly. The sky-blue callout anticipates the student's question ("Wait—where did the (1-beta) go?") and answers it directly. This is actually better than silently using the EMA form, because the student will encounter classical momentum in textbooks and frameworks—now they'll understand why the two forms exist.
- The RMSProp numerical table is a strong addition. The punchline (both parameters take ~0.32 steps despite 50,000x gradient difference) makes the self-normalizing property viscerally clear.
- The "hill is shaking" connection is woven naturally into the bowling ball paragraph, not bolted on. Good integration.
- The widget internal state display adds real pedagogical value—seeing v_x and v_y change while the ball moves connects the formula to the visualization.
- The "Adam has no hyperparameters" callout is appropriately stern without being preachy.

**Overall assessment:**
The lesson is strong. The narrative arc is compelling, all 5 planned misconceptions are addressed, all planned modalities are present, and the fixes from iteration 1 were implemented thoughtfully. The one remaining improvement (ref-based rendering) is a code quality concern that does not affect the student experience today but should be addressed for robustness. The two polish items are minor and optional. This lesson is ready for deployment after the improvement fix.

---

## Review — 2026-02-09 (Iteration 3/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2 (carried forward, non-blocking)

### Verdict: PASS

All critical and improvement findings from iterations 1 and 2 are resolved. The ref-to-state fix from iteration 2 is correctly implemented. Two polish items are carried forward but do not block the lesson. This lesson is ready for deployment.

### Verification of Iteration 2 Fix

**IMPROVEMENT (Widget ref-based rendering): RESOLVED.**
The widget now uses `displayState` (useState, line 248) for rendering internal optimizer values instead of reading directly from `optStateRef.current`. The reactive state is set in `doStep()` at line 337 (`setDisplayState(clamped)`), in `reset()` at line 377, and in `handleOptimizerChange` at line 389. The JSX at lines 647-663 reads from `displayState.vx`, `.vy`, `.sx`, `.sy`. The ref is still maintained for the animation loop (correct pattern for avoiding stale closures in requestAnimationFrame), but the display is now self-sufficient—it will re-render whenever `doStep` is called, regardless of whether other state setters are present. No implicit coupling remains.

**No new issues introduced.** The `initialState()` call is correctly applied in both reset paths. The `displayState` type matches `OptimizerState`. The animation loop correctly writes to both the ref (for its own next-frame reads) and the state (for React rendering).

### Carried Polish Items (Non-Blocking)

#### [POLISH] — EMA cold-start observation not planted (carried from iterations 1 and 2)

**Location:** Section 3 (EMA), the concrete walkthrough table
**Issue:** The EMA table shows v_1 = 0.20 from input 2.0. Planting a brief observation ("Notice v_1 is much smaller than the input—the EMA needs time to warm up") would make Adam's bias correction feel more motivated. The bias correction IS explained in the Adam section ("since m_0 = 0 and v_0 = 0, the first few estimates are too small"), so the student is not left confused—the seed would just make the later explanation click faster.
**Student impact:** Minimal. The Adam section handles this adequately.
**Status:** Optional enhancement for a future pass. Does not block deployment.

#### [POLISH] — Check-understanding sections use inline markup (carried from iterations 1 and 2)

**Location:** Sections 5 and 10 (Check 1 and Check 2)
**Issue:** Both use inline `<div>` with manual border/bg styling and `<details>/<summary>`. If a reusable CheckBlock component is extracted in the future, these should be migrated.
**Student impact:** None. The visual presentation works correctly.
**Status:** Note for future component extraction. Does not block deployment.

### Full Review Walkthrough (Iteration 3)

**Step 1 — Read as Student:** Read the lesson sequentially with the student's mental state (SGD at DEVELOPED, gradient noise at INTRODUCED, loss landscape at INTRODUCED, ball-on-hill and "hill is shaking" analogies established). No confusion points found. Every concept is motivated before introduced. The EMA-to-momentum-to-RMSProp-to-Adam progression builds logically. The sky-blue callout addressing the (1-beta) discrepancy between EMA and classical momentum catches the exact question the student would ask. The RMSProp numerical table grounds the "self-normalizing" claim with traceable arithmetic. All five misconceptions are addressed at appropriate points.

**Step 2 — Check Against Plan:** All planned elements present. Target concept taught correctly. Both gaps (EMA, per-parameter adaptation) resolved with dedicated sections. All 5 planned misconceptions addressed with concrete counter-examples or callouts. All 5 planned examples included. All 5 planned modalities present. Narrative arc matches the plan. Scope boundaries respected (no code implementation, no convergence proofs, no AdamW).

**Step 3 — Pedagogical Principles:** Motivation Rule (problem before solution at every step), Modality Rule (5 modalities), Example Rules (3+ positive, 2 negative), Misconception Rule (5 addressed), Ordering Rules (concrete before abstract, parts before whole, simple before complex), Load Rule (2 new concepts), Connection Rule (every concept linked to prior knowledge), Reinforcement Rule (gradient noise, sharp/wide minima, learning rate all reinforced), Interaction Design Rule (cursor-pointer on all interactive elements), Writing Style Rule (em dashes unspaced throughout). All pass.

**Step 4 — Examples:** EMA walkthrough traceable and strong. Ravine oscillation ComparisonRow clear. RMSProp numerical table verifiable with strong punchline. Per-parameter GradientCards motivating. Adam vs SGD ComparisonRow well-structured. LR interchangeability warning concrete. No missing examples.

**Step 5 — Narrative and Flow:** Hook is compelling (ravine visualization). Transitions are explicit between all sections. Arc builds naturally: problem -> tool -> fix #1 -> fix #2 -> combine -> nuance. Pacing appropriate—EMA and momentum get the most space as foundational sections, RMSProp is appropriately shorter (INTRODUCED depth), Adam is compact as a combination. Conclusion hits all key takeaways.

### Review Notes

**What works well in the final version:**
- The three-iteration review cycle has produced a polished lesson. Each iteration caught genuinely different issues: iteration 1 found structural problems (formula inconsistency, missing misconception treatment, missing examples), iteration 2 found an architectural fragility (ref-based rendering), and iteration 3 confirms all fixes are correctly implemented.
- The sky-blue callout explaining classical vs EMA momentum is actually better pedagogy than silently using the EMA form, because the student will encounter classical momentum in PyTorch and textbooks.
- The RMSProp numerical table added in iteration 2 is one of the lesson's strongest elements—the "both parameters take ~0.32 steps despite 50,000x gradient difference" punchline makes the self-normalizing property viscerally clear.
- The widget's internal state display with contextual per-optimizer explanations adds real pedagogical value that wasn't in the original build.
- All 5 planned misconceptions are addressed, which is unusually thorough. The "Adam has no hyperparameters" callout in particular prevents a common practical mistake.

**Lesson readiness:** This lesson is ready for deployment. The two remaining polish items are genuinely minor and can be addressed in a future quality pass without affecting the student experience.
