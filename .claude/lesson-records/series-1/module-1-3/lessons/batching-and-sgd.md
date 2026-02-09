# Lesson: batching-and-sgd

**Module:** 1.3 — Training Neural Networks
**Position:** Lesson 4 of 7 (first lesson in Act 2: "Making It Work")
**Type:** STRETCH (3 new concepts)
**Slug:** `batching-and-sgd`

---

## Phase 1: Orient — Student State

The student has completed 13 lessons across three modules. They understand the full learning pipeline from "what is ML" through linear regression, neural network architecture, activation functions, and backpropagation. They have computed gradients by hand and visualized them as computational graphs.

### Relevant Concepts With Depths

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Gradient descent update rule | DEVELOPED | gradient-descent (1.1) | theta_new = theta_old - alpha * nabla_L; single parameter, full dataset |
| Learning rate (alpha) | DEVELOPED | learning-rate (1.1) | Step size, Goldilocks zone, oscillation/divergence failure modes |
| Training loop (forward -> loss -> backward -> update) | DEVELOPED | implementing-linear-regression (1.1) | 6-step pattern, "heartbeat of training" |
| MSE loss function | DEVELOPED | loss-functions (1.1) | Full formula, squaring rationale, loss landscape as bowl |
| Loss landscape | INTRODUCED | loss-functions (1.1) | Bowl shape, valley = minimum; convex for linear regression |
| Backpropagation algorithm | DEVELOPED | backpropagation (1.3) | Forward + backward pass = all gradients |
| Multi-layer gradient computation | DEVELOPED | backprop-worked-example (1.3) | 4 gradients from 1 backward pass through 2-layer network |
| Weight update with real numbers | DEVELOPED | backprop-worked-example (1.3) | theta_new = theta_old - 0.1 * gradient, verified loss decreases |
| Computational graph notation | DEVELOPED | computational-graphs (1.3) | Operations as nodes, data flow as edges, forward/backward traversal |
| Automatic differentiation (concept) | INTRODUCED | computational-graphs (1.3) | PyTorch builds graph during forward, loss.backward() walks it |
| Convexity (bowl shape, one minimum) | MENTIONED | gradient-descent (1.1) | Noted for linear regression; non-convex for neural networks |
| Hyperparameters | INTRODUCED | learning-rate (1.1) | "Values YOU choose, not the model" |

### Established Mental Models and Analogies

- **"Ball rolling downhill"** — gradient descent intuition (1.1, reinforced through learning-rate and implementing-linear-regression)
- **"Loss landscape = bowl with valley"** — optimization target (1.1, reinforced in gradient-descent and learning-rate)
- **"Training = forward -> loss -> backward -> update"** — the universal loop (1.1, reinforced throughout 1.3)
- **"One forward + one backward = ALL gradients"** — backprop efficiency (1.3, reinforced in worked example)
- **"Incoming gradient x local derivative"** — the backward pass recipe (1.3)

### What Was Explicitly NOT Covered

- Batching of any kind — all gradient descent has been on a single data point or the full dataset
- Stochastic vs batch vs mini-batch distinction
- Epochs (the concept of cycling through data)
- Non-convex loss landscapes (mentioned but not developed)
- How training works with MORE than one data point flowing through the network at a time
- Noise in gradients, gradient estimation

### Readiness Assessment

The student is well-prepared. They have the training loop at DEVELOPED depth (can trace it step by step), gradient descent at DEVELOPED depth (understand the update rule and learning rate), and backpropagation at DEVELOPED depth (can compute all gradients by hand). The critical gap: everything so far has been one data point at a time. The student has never thought about what happens when you have 50,000 training examples. This lesson addresses exactly that transition.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to understand why and how training data is divided into mini-batches, what stochastic gradient descent is and how it differs from full-batch gradient descent, and how epochs structure the training process.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Gradient descent update rule | DEVELOPED | DEVELOPED | gradient-descent (1.1) | OK | Student needs to run the update rule to understand how batching modifies it |
| Training loop | DEVELOPED | DEVELOPED | implementing-linear-regression (1.1) | OK | Mini-batching modifies the training loop; student must be fluent in the base version |
| MSE loss function | DEVELOPED | DEVELOPED | loss-functions (1.1) | OK | Need to understand that loss is an AVERAGE — this becomes key when computing loss over a batch vs the whole dataset |
| Learning rate | DEVELOPED | DEVELOPED | learning-rate (1.1) | OK | Batch size interacts with effective learning rate; student needs solid LR understanding |
| Backpropagation | INTRODUCED | DEVELOPED | backpropagation (1.3) | OK | Student needs to know backprop gives gradients; does NOT need to compute them by hand here |
| Loss landscape | INTRODUCED | INTRODUCED | loss-functions (1.1) | OK | Need the "bowl" intuition to contrast: full-batch sees the true bowl, mini-batch sees a noisy approximation of it |
| Hyperparameters | INTRODUCED | INTRODUCED | learning-rate (1.1) | OK | Batch size is a new hyperparameter; student needs the framing |
| Non-convex landscapes | MENTIONED | MENTIONED | gradient-descent (1.1) | OK | Sufficient for this lesson — we only INTRODUCE gradient noise as helpful for escaping local minima. Full non-convex treatment comes in training-dynamics (lesson 6) |

**Gap resolution:** No gaps. All prerequisites are at sufficient depth.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "You must use ALL data points to compute an accurate gradient" | All prior lessons computed gradients on either 1 point or the full dataset. The student has never seen a "partial" gradient. Natural assumption: partial = inaccurate = bad. | Show a side-by-side: full-batch gradient vs mini-batch gradient on 4 different batches. The mini-batch gradients point in ROUGHLY the right direction even though each one is "wrong." On average, they converge to the same place — often faster because they take more frequent steps. | During the core explanation of mini-batches, before introducing SGD. This is the central tension the lesson must resolve. |
| "Bigger batch size is always better because it gives a more accurate gradient" | Follows from the previous misconception. If partial data = noisy gradient, then more data = better gradient = better training. | Two training curves: batch_size=full vs batch_size=32. The mini-batch version converges FASTER in wall-clock time despite noisier individual steps, because it takes MANY more update steps per epoch. Also: very large batches can get stuck in sharp minima (introduced gently — this is the "gradient noise is helpful" concept). | After establishing mini-batches, during the "why not just use the full dataset?" section. |
| "One epoch = one gradient update" | The word "epoch" sounds like "step" or "iteration." Without explicit disambiguation, students conflate epoch with update. | Concrete example: 1000 data points, batch size 50 = 20 updates per epoch. One epoch = seeing all data once = 20 separate gradient steps. Walk through the arithmetic explicitly. | When introducing the epoch concept. |
| "SGD means batch_size=1" | Many textbook definitions use "stochastic gradient descent" to mean literally one sample at a time. The student may have encountered this definition elsewhere. | Modern "SGD" in practice means mini-batch SGD. True single-sample SGD is extremely noisy and rarely used. Show: batch_size=1 is too noisy (gradient from one point can point in very wrong direction), batch_size=ALL is too slow. Mini-batch is the sweet spot. This is a terminology clarification, not a deep misconception. | When defining SGD, after the student has seen why mini-batches work. Explicitly disambiguate "pure SGD" vs "mini-batch SGD" vs "batch GD." |
| "Shuffling the data doesn't matter" | Student has never thought about data ordering. Why would order matter if you see all of it in an epoch? | If your data is sorted by label (all cats, then all dogs), consecutive batches are all one class. The gradient from an all-cats batch pushes parameters toward cats, then the all-dogs batch yanks them back. Oscillation instead of convergence. Shuffling ensures each batch is a representative sample. | After explaining epochs, when discussing the mechanics of iterating through batches. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **House prices dataset (1000 homes)** — The core running example | Positive | Motivate batching: "You have 1000 training examples. Computing the gradient over ALL 1000 is expensive. What if you used just 32 at a time?" Ground the entire lesson in a concrete, relatable scenario. | Houses are familiar, the dataset size is large enough to make full-batch feel expensive but small enough to reason about. Connects to linear regression from Module 1.1 (predicting a number from features). |
| **Side-by-side: 3 gradient arrows** — full-batch vs 3 mini-batch arrows | Positive | Visually show that mini-batch gradients are noisy approximations of the true gradient, but they point in roughly the right direction. The average of the mini-batch arrows ~ the full-batch arrow. | This is the key visual insight. It converts the abstract claim "mini-batches approximate the full gradient" into something the student can see. Uses arrows on a 2D loss landscape, which extends the existing "bowl" mental model. |
| **Arithmetic walkthrough: 1000 points, batch=50** | Positive | Make epochs/iterations/batches concrete with specific numbers. 1000/50 = 20 batches per epoch. 5 epochs = 100 total updates. Each number is stated explicitly. | Students need to do the arithmetic at least once to internalize the relationships. Small enough numbers to compute mentally, large enough to feel like "real training." |
| **Sorted dataset (all cats then all dogs)** | Negative | Show why shuffling matters. Without shuffling, consecutive batches are homogeneous. Gradients oscillate instead of converging. | This is the strongest negative example for the lesson. It demonstrates a non-obvious failure mode that students wouldn't predict, making the "shuffle before each epoch" rule feel necessary rather than arbitrary. |
| **batch_size=1 vs batch_size=ALL vs batch_size=32** | Stretch | Show the spectrum from pure SGD (too noisy) through mini-batch (sweet spot) to full-batch (too slow). This is where gradient noise as beneficial gets INTRODUCED. | Extends the core concept into a design decision. The student sees that the "right" batch size is a tradeoff, not a formula. Connects to the hyperparameter concept from 1.1. |

---

## Phase 3: Design

### Narrative Arc

Everything the student has learned about training so far has been on toy problems: one data point, or a handful of points small enough to process all at once. That's fine for understanding the algorithm, but it's not how anyone actually trains a neural network. Real datasets have thousands to millions of examples. Computing the gradient over ALL of them before taking a single step is painfully slow — imagine computing 1 million forward passes, accumulating all the gradients, and only THEN updating your weights once. You'd go broke on compute before finishing a single epoch. The solution is deceptively simple: don't use all the data at once. Use a small random sample — a mini-batch — compute an approximate gradient, and take a step. Then grab another mini-batch and step again. Each individual step is noisier, but you take so many more of them that training is dramatically faster. And here's the surprising part: that noise isn't just tolerable — it's actually helpful. It prevents the model from getting trapped in bad local minima. This is the core idea of stochastic gradient descent, and it's how every modern neural network is trained.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Visual** | 2D loss landscape with gradient arrows: one long true-gradient arrow vs several shorter noisy mini-batch arrows. Show the random walk convergence path vs the smooth full-batch path. | The loss landscape bowl is already a strong mental model from Module 1.1. Putting gradient arrows on it makes the abstract notion of "noisy approximation" viscerally visible. The student can SEE that the noisy path zigzags but still reaches the valley. |
| **Concrete example** | 1000 houses, batch_size=50. Walk through: epoch 1 = shuffle, split into 20 batches, process each (forward, loss, backward, update), repeat. Specific numbers at every step. | Arithmetic grounds abstract concepts. "20 batches per epoch" and "5 epochs = 100 updates" are claims the student can verify. This makes epochs, iterations, and batches feel like real quantities, not vocabulary words. |
| **Intuitive/Analogy** | "Polling analogy" — You don't need to ask every person in a city to estimate the average height. A random sample of 50 gives you a good estimate. Mini-batches do the same for gradients: a random sample of data points gives you a good estimate of the true gradient. | Polling is universally familiar and maps cleanly: population = full dataset, sample = mini-batch, sample mean ~ population mean, larger sample = more accurate but more expensive. This analogy makes the statistical argument feel obvious. |
| **Symbolic** | The gradient formulas: full-batch gradient = (1/N) sum over all N vs mini-batch gradient = (1/B) sum over B. Show they have the same structure — the only difference is which data points you sum over. The mini-batch gradient is an unbiased estimator of the full gradient. | The student has seen gradient formulas in 1.1. Showing the structural similarity between full-batch and mini-batch gradients proves that mini-batching isn't a hack — it's the same math on less data. The word "unbiased" is introduced gently (the average of many mini-batch gradients = the full gradient). |
| **Geometric/Spatial** | The "noisy path" on the loss landscape: full-batch GD follows a smooth curve to the minimum. Mini-batch SGD zigzags drunkenly but still arrives — and sometimes finds a BETTER minimum because the noise bounced it out of a sharp valley into a wide one. | This is the payoff visual for "gradient noise as helpful." It extends the loss landscape from a simple bowl to hint at the non-convex reality of neural network training. The student doesn't need full non-convex theory — just the visual that noise can be a feature, not a bug. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 (mini-batches, SGD, epochs) + 1 INTRODUCED (gradient noise as helpful)
- **Previous lesson load:** BUILD (computational-graphs — 1 new concept)
- **Appropriateness:** This is correctly positioned as STRETCH. The previous two lessons were BUILD, so the student has had room to consolidate. The 3 new concepts are tightly related (mini-batches enable SGD, epochs structure the process) so they reinforce each other rather than competing for attention. The INTRODUCED concept (gradient noise) is a teaser that gets DEVELOPED in later lessons.

### Connections to Prior Concepts

- **Gradient descent update rule** (1.1, DEVELOPED): Mini-batch SGD uses the SAME update rule — the only change is which data computes the gradient. This is the strongest connection: "You already know this. We're just changing what goes into the gradient."
- **Training loop** (1.1, DEVELOPED): The training loop gets a NEW inner structure: loop over epochs, within each epoch loop over batches, each batch runs the familiar forward -> loss -> backward -> update. Same heartbeat, new rhythm.
- **Loss landscape as bowl** (1.1, INTRODUCED): Extended with gradient arrows and noisy vs smooth paths. The bowl is the same — how you traverse it changes.
- **Learning rate** (1.1, DEVELOPED): Batch size and learning rate interact. Smaller batches = noisier gradients = may need to adjust LR. Briefly noted (not DEVELOPED — that's optimizer territory in lesson 5).
- **Hyperparameters** (1.1, INTRODUCED): Batch size is a new hyperparameter. Follows the same framing: "a value you choose, not the model."
- **"One forward + one backward = ALL gradients"** (1.3): Still true — but now "all gradients" means "gradients for THIS batch," not "gradients from all data." The efficiency insight still holds.

**Potentially misleading prior analogies:**
- The "ball rolling downhill" analogy suggests smooth, deterministic movement. SGD is more like a ball bouncing down a rocky hill — still going downhill on average but with random jitter at each step. This needs to be explicitly addressed: "The ball is still rolling downhill, but now the hill is shaking."

### Scope Boundaries

**This lesson IS about:**
- Why batching is necessary (computational argument)
- How mini-batches work mechanically (shuffle, split, iterate)
- The vocabulary: batch, mini-batch, stochastic, epoch, iteration
- Gradient noise as a concept (INTRODUCED — it exists and can help)
- Batch size as a hyperparameter with tradeoffs
- Why shuffling matters

**This lesson is NOT about:**
- Optimizers beyond vanilla SGD (no momentum, no Adam — that's lesson 5)
- Learning rate schedules (lesson 5 territory)
- Data loading, data augmentation, or preprocessing
- Parallelism / GPU utilization (practical concern, but not this lesson)
- The full theory of non-convex optimization and local minima (lesson 6)
- Convergence guarantees or theoretical analysis of SGD
- Matrix/vectorized operations for batch processing

**Target depths:**
- Mini-batches: DEVELOPED (can explain why, how, and trace through a concrete example)
- Stochastic gradient descent: DEVELOPED (can explain the algorithm, compare to full-batch, state tradeoffs)
- Epochs: DEVELOPED (can compute iterations per epoch, total updates over multiple epochs)
- Gradient noise as helpful: INTRODUCED (knows it exists and the intuition for why, hasn't practiced with it)

### Lesson Outline

**1. Context + Constraints**
"This lesson is about how training ACTUALLY works with real datasets. We're NOT covering optimizers, GPU details, or convergence theory. By the end, you'll understand mini-batches, SGD, and epochs — the fundamental mechanics that every training run uses."

**2. Hook — The Scale Problem (demo/reveal)**
Present the problem concretely: "You've trained on 1 data point. What if you have 50,000? Computing ALL gradients before ONE update = painfully slow." Show a number: ImageNet has 1.2 million images. At 1 forward+backward per image, one gradient update takes 1.2 million passes. How long would training take? (Answer: absurdly long.) "There's a better way."

*Why this hook:* Creates an immediate, visceral motivation. The student already knows the training loop — showing it doesn't scale makes them WANT the solution. Problem before solution.

**3. Explain — The Polling Analogy and Mini-Batches**
Introduce the polling analogy: "You don't poll every citizen to estimate public opinion — a random sample works." Connect to gradients: "You don't need all 50,000 data points to get a useful gradient estimate — a random sample of 32 works." Define mini-batch. Show the formula side by side: full gradient vs mini-batch gradient. Same structure, different N.

Concrete example: 1000 houses, batch_size=50. Walk through: shuffle the data, split into 20 batches of 50, process batch 1 (forward on 50 examples, compute average loss, backward, update weights), then batch 2, then batch 3...

**4. Visual — Gradient Arrows on the Loss Landscape**
Show the 2D loss landscape with gradient arrows. Full-batch: one precise arrow pointing to the minimum. Mini-batch: several arrows, each slightly different, each an approximation. Key visual: the arrows cluster around the true direction. "On average, they point the right way."

**5. Check 1 — Predict-and-Verify**
"If you have 600 training examples and a batch size of 32, how many batches per epoch?" (Answer: 18, with the last batch having 24 examples — or 18 batches of 32 if we drop the remainder. Note both conventions.) "How many weight updates per epoch?" (Answer: 18 or 19.) This grounds the arithmetic.

**6. Explain — Epochs and the Full Algorithm**
Define epoch: one complete pass through ALL training data. Formalize: 1 epoch = N/B iterations (where N = dataset size, B = batch size). Define iteration/step = one batch's update.

Spell out the full SGD algorithm explicitly:
```
for epoch in range(num_epochs):
    shuffle(data)
    for batch in split(data, batch_size):
        predictions = forward(batch)
        loss = compute_loss(predictions, batch.targets)
        gradients = backward(loss)
        update_weights(gradients, learning_rate)
```

Connect to the existing training loop: "The inner four lines are EXACTLY the training loop you already know. The new part is the two outer loops: epochs and batches."

**7. Negative Example — The Sorted Dataset**
"What happens if you DON'T shuffle?" Show: data sorted by label (cats, then dogs). First 5 batches are all cats. The gradient says "everything is cats, optimize for cats." Then the dogs batches arrive and yank parameters the other way. Oscillation, slow convergence. "Shuffling ensures each batch is a representative sample — the polling analogy breaks if your 'random' sample is all people from one neighborhood."

**8. Explain — The SGD Spectrum (Introduces Gradient Noise)**
Place three points on a spectrum: batch_size=1 (pure SGD, extremely noisy), batch_size=N (full-batch GD, smooth but slow), batch_size=32-256 (mini-batch SGD, the sweet spot).

Show training curves for each: pure SGD zigzags wildly, full-batch converges smoothly but takes forever, mini-batch converges quickly with moderate noise.

Key reveal: "The noise isn't just tolerable — it's helpful." Show the noisy path on a loss landscape with multiple valleys: the smooth full-batch path rolls into the nearest minimum (which might be sharp/bad), while the noisy SGD path bounces out of sharp minima and settles in wide ones (which generalize better). This is the INTRODUCED concept of gradient noise as beneficial. Don't belabor it — plant the seed, it gets reinforced in lesson 5 (optimizers) and 6 (training dynamics).

**9. Check 2 — Transfer Question**
"Your model is training but loss has plateaued. Someone suggests increasing the batch size from 32 to 512. What are the tradeoffs?" (Expected: more accurate gradients per step = smoother convergence, BUT fewer updates per epoch = potentially slower overall. Also: less noise = might get stuck in sharp minima. Also: uses more memory.)

**10. Explore — Interactive Widget (SGD Explorer)**
Widget concept: A 2D loss landscape (contour plot) with controls for batch size. The student can:
- Toggle between full-batch GD and mini-batch SGD
- Adjust batch size with a slider (1, 8, 32, 128, ALL)
- Watch the optimization path unfold on the landscape
- See the training loss curve update in real time
- Observe: smaller batches = noisier path, more updates per epoch; larger batches = smoother path, fewer updates

The landscape should have at least two minima (one sharp, one wide) so the student can SEE gradient noise helping escape the sharp one.

**11. Elaborate — Practical Details**
- **Batch size as hyperparameter:** Common values (32, 64, 128, 256). Powers of 2 for GPU efficiency (mentioned briefly, not DEVELOPED).
- **Last batch:** If N isn't divisible by B, the last batch is smaller. Two conventions: drop it or use the smaller batch.
- **Batch size and learning rate interaction:** Brief note — larger batches can often tolerate larger learning rates. Not DEVELOPED here; just planted for lesson 5.

**12. Summarize — Key Takeaways**
- Mini-batches: use a random subset of data per gradient update. Same update rule, less data per step, more steps per epoch.
- SGD: the algorithm that uses mini-batch gradients. Mini-batch SGD is the default; pure SGD and full-batch are the extremes.
- Epochs: one full pass through the dataset. Total updates = (N/B) * num_epochs.
- Gradient noise: not a bug, a feature. Helps escape bad local minima.
- Shuffling: essential. Without it, batches aren't representative.

Echo the mental model: "The ball is still rolling downhill. But now the hill is shaking — and that's a good thing."

**13. Next Step**
"SGD with a fixed learning rate is the simplest version. But it struggles: it oscillates in ravines, it can't adapt to the shape of the loss landscape. Next lesson: momentum, RMSProp, and Adam — the optimizers that make SGD actually work well."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (gradient descent from 1.1, training loop from 1.1, backprop from 1.3)
- [x] Depth match verified for each (all OK, no gaps)
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widget (widget uses loss landscape already at INTRODUCED + gradient descent at DEVELOPED)
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (the scale problem)
- [x] At least 3 modalities planned (5: visual, concrete example, intuitive/analogy, symbolic, geometric/spatial)
- [x] At least 2 positive examples + 1 negative example (3 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load <= 3 new concepts (3 DEVELOPED + 1 INTRODUCED)
- [x] Every new concept connected to at least one existing concept (mini-batch -> gradient descent, SGD -> training loop, epochs -> training loop, gradient noise -> loss landscape)
- [x] Scope boundaries explicitly stated

---

## Widget Specification: SGD Explorer

**Purpose:** Let the student SEE the difference between full-batch and mini-batch gradient descent on a 2D loss landscape.

**Inputs/Controls:**
- Batch size slider: 1 / 8 / 32 / 128 / ALL
- Play/pause button + step button
- Reset button
- Speed slider

**Visual Elements:**
- 2D contour plot of a loss landscape with 2 minima (one sharp, one wide)
- Optimization path drawn as a trail of dots/line
- Current position marker
- Training loss curve (small subplot or overlay)

**Key Behaviors:**
- Full-batch: smooth, deterministic path to nearest minimum
- Small batch: noisy, jittery path that sometimes escapes the sharp minimum
- The student should be able to run the same starting point with different batch sizes and compare paths

**Connects to:** GradientDescentExplorer from 1.1 (same idea of watching optimization happen, but now with batching).

**Does NOT include:** Momentum, adaptive learning rates, or any optimizer beyond vanilla SGD. Those are lesson 5.

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues found. The lesson is well-structured, follows the plan closely, and would be usable by the student. However, four improvement-level findings would meaningfully strengthen the lesson. Another pass is warranted.

### Findings

#### [IMPROVEMENT] — "Unbiased estimator" introduced without grounding

**Location:** "The Polling Insight" section (around line 171-176)
**Issue:** The formula block introduces the term "unbiased estimator" in a parenthetical at the bottom. This is a statistics term the student has never encountered. The aside (ConceptBlock "Unbiased Estimator") attempts to define it, but the definition is circular: "if you averaged many mini-batch gradients together, you'd get the full-batch gradient" is the behavior, not an explanation of WHY. The student has no framework for "estimators" or what "biased" vs "unbiased" means. More importantly, the term appears in the formula block first, before the student reads the aside.
**Student impact:** The student encounters a technical-sounding term mid-explanation and may either (a) fixate on it, wondering if they should understand it, or (b) feel like the lesson just jumped to a more advanced level. Either way it disrupts the flow of the core insight (mini-batches approximate the full gradient).
**Suggested fix:** Remove "unbiased estimator" from the formula block. Keep the aside but reframe it as a "nice to know" footnote rather than a core claim. The key point is already stated clearly: "on average, mini-batch gradients equal the full gradient." That's sufficient for DEVELOPED depth on mini-batches. The formal statistical term can wait until a later lesson.

#### [IMPROVEMENT] — Gradient noise benefit section lacks a visual

**Location:** "Noise as a Feature" section (around lines 555-595)
**Issue:** The plan called for a geometric/spatial modality: "the noisy path on the loss landscape: full-batch GD follows a smooth curve to the minimum. Mini-batch SGD zigzags... and sometimes finds a BETTER minimum." The built lesson describes this verbally but does not include any visual representation of this concept. The SGD Explorer widget below does demonstrate this interactively, but the "Noise as a Feature" section itself is purely text-based. The student reads about sharp vs wide minima and bouncing paths but never sees them in this section. They must discover the behavior later in the widget on their own.
**Student impact:** This is the most conceptually novel part of the lesson (gradient noise is helpful is counter-intuitive). Without a visual in-place, the student must hold the mental image purely from text. The widget partially compensates, but the "Noise as a Feature" section reads as a wall of text with only a callout box.
**Suggested fix:** Add a simple static visual showing two paths on a 1D cross-section of the loss landscape: one smooth path descending into a sharp narrow minimum, one noisy path bouncing past it and settling into a wide shallow minimum. This can be a simple SVG similar to the gradient arrows visual above it. The full interactive exploration is still in the widget, but the section needs its own visual anchor.

#### [IMPROVEMENT] — Misconception "SGD means batch_size=1" addressed but could be stronger

**Location:** "The SGD Spectrum" section, terminology note (lines 531-537)
**Issue:** The plan identified "SGD means batch_size=1" as a misconception with a specific plan: "Modern 'SGD' in practice means mini-batch SGD. True single-sample SGD is extremely noisy and rarely used. Show: batch_size=1 is too noisy." The built lesson addresses this with a terminology note box, which is good. However, the three-column card layout presents Pure SGD (batch=1), Mini-batch SGD (32-256), and Full-batch GD as equal peers on a spectrum. The "Pure SGD" card says "Rarely used in practice" but doesn't make it viscerally clear WHY. The student sees "batch_size=1 = extremely noisy gradient" as a bullet point but doesn't see a concrete example of how wrong a single-sample gradient can be.
**Student impact:** The student understands the spectrum intellectually but may not feel the danger of batch_size=1. The misconception is addressed at the vocabulary level but not at the intuition level. A student who encounters "SGD" in a paper might still picture pure single-sample SGD because the lesson didn't make the single-sample failure mode memorable.
**Suggested fix:** Add one concrete sentence to the Pure SGD card or below the spectrum: "Imagine estimating the average height of a city by measuring one random person. You might get a 6'5" basketball player or a 4-year-old child. That single data point could point your gradient in a completely wrong direction." This connects back to the polling analogy and makes the failure mode vivid.

#### [IMPROVEMENT] — The house price example references "Linear Regression from Scratch" by an inconsistent name

**Location:** "Epochs: Cycling Through the Data" section (line 399)
**Issue:** The lesson says "the training loop from Linear Regression from Scratch" which appears to reference the lesson titled "Implementing Linear Regression" (slug: implementing-linear-regression). The name used doesn't match the actual lesson title. This is a small accuracy issue, but in a course where concepts are carefully cross-referenced, inconsistent lesson names can confuse the student about what they've already learned.
**Student impact:** Mild confusion. The student may momentarily wonder "Did I do a lesson called Linear Regression from Scratch?" before connecting it to the right lesson. Not blocking, but it breaks the sense of a well-organized curriculum.
**Suggested fix:** Use the actual lesson name or a generic reference like "the training loop from Module 1.1" or "the training loop you built in the implementing-linear-regression lesson."

#### [POLISH] — Spaced em dashes in vocabulary list

**Location:** Lines 385-387 (Vocabulary check list items)
**Issue:** The three vocabulary definitions use spaced em dashes: "Epoch — one complete pass..." The writing style rule requires no spaces: "Epoch—one complete pass..."
**Student impact:** None functionally, but inconsistent with the rest of the lesson which correctly uses unspaced em dashes throughout.
**Suggested fix:** Remove spaces around the em dashes in lines 385-387.

#### [POLISH] — Leading space before "unbiased estimator" em tag

**Location:** Line 173
**Issue:** `<em> unbiased estimator</em>` has a leading space inside the em tag. This renders correctly but is technically sloppy markup.
**Student impact:** None visible.
**Suggested fix:** Change to `<em>unbiased estimator</em>` (no leading space).

#### [POLISH] — Widget "TryThisBlock" experiments could guide discovery order more explicitly

**Location:** Lines 652-659 (TryThisBlock experiments)
**Issue:** The experiments are listed as bullet points but the discovery order matters pedagogically: full-batch first (baseline), then batch=1 (extreme noise), then batch=32 (sweet spot), then compare deterministic vs stochastic runs. The list presents this order but doesn't number the steps or make the sequence feel mandatory.
**Student impact:** A student might skip around and miss the pedagogical arc (baseline -> extreme -> sweet spot). Not harmful, but the "aha" moment of seeing noise help is strongest when you've seen the smooth-but-trapped full-batch path first.
**Suggested fix:** Consider numbering the experiments (1, 2, 3...) or using language like "First... Then... Now try..." to make the sequence feel intentional.

### Review Notes

**What works well:**
- The hook (Scale Problem) is genuinely motivating. The concrete numbers (1.2 million images, 6 hours per update) make the problem visceral. Problem before solution is executed perfectly.
- The polling analogy is strong and maps cleanly to gradients. It's introduced before the formulas and grounds the statistical argument in familiar territory.
- The house price running example (1000 homes, batch=50) is excellent. The arithmetic is simple enough to verify mentally but large enough to feel real.
- The sorted-dataset negative example is well-placed and compelling. The ComparisonRow format makes the contrast visually clear.
- The SGD Explorer widget is well-designed with appropriate controls and the two-minima landscape directly demonstrates the "noise as feature" concept.
- Section flow is strong. Each section connects to the next with clear transitions.
- All five planned misconceptions are addressed at appropriate locations.
- Layout compliance is perfect: every section uses the Row compound component correctly.
- Interactive elements in the widget have appropriate cursor styles (cursor-pointer on batch size buttons and speed slider).

**Patterns observed:**
- The lesson leans slightly text-heavy in the middle (Noise as Feature section). Adding the suggested visual would break this up.
- The lesson correctly keeps "gradient noise as helpful" at INTRODUCED depth, avoiding over-explanation. The widget provides exploration without over-theorizing.
- Cognitive load is well-managed: the three core concepts (mini-batches, SGD, epochs) are introduced sequentially with checks between them rather than all at once.
- Scope boundaries are respected throughout. No drift into optimizers, GPU parallelism, or convergence theory.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 1
- Polish: 2

### Verdict: PASS

All 7 findings from iteration 1 have been effectively addressed. The "unbiased estimator" term is now safely in an aside with "you don't need the formal term yet" framing. The Noise as a Feature section has a strong 1D cross-section SVG. The Pure SGD card has the vivid basketball-player polling callback. The lesson name reference is corrected. Em dashes are clean. The widget experiments are numbered with sequential language. No critical or significant improvement issues remain. One improvement finding is borderline (could arguably be Polish) and two minor polish items were found.

### Iteration 1 Fix Verification

All 7 iteration 1 findings verified as resolved:

1. **"Unbiased estimator" without grounding** — FIXED. Removed from formula block. Now only in aside (ConceptBlock "Jargon Preview") with "You don't need the formal term yet" framing. The aside says "Statisticians call this..." which makes it feel like optional vocabulary rather than required knowledge.

2. **Noise section lacks visual** — FIXED. A 1D cross-section SVG (lines 581-643) now shows two paths: smooth orange path rolling into the sharp minimum (labeled "stuck") vs noisy green dashed path bouncing past and settling in the wide minimum. Legend, labels, and caption all present. This is exactly what was requested.

3. **"SGD means batch_size=1" could be stronger** — FIXED. The Pure SGD card (line 509) now includes: "Like estimating a city's average height by measuring one random person—you might get a 6'5" basketball player." Vivid, connects back to polling analogy.

4. **Wrong lesson name reference** — FIXED. Line 399 now says "the training loop from Implementing Linear Regression" matching the actual lesson title.

5. **Spaced em dashes in vocabulary list** — FIXED. Lines 385-387 now use unspaced em dashes consistently ("Epoch—one complete pass", "Iteration—one gradient update", "Batch size—how many examples").

6. **Leading space in em tag** — FIXED. The entire "unbiased estimator" was moved to the aside as part of fix #1; the em tag markup issue is resolved.

7. **Widget experiments not numbered** — FIXED. Lines 719-725 now use `<ol>` with numbered items and sequential language ("Start with...", "Reset, switch to...", "Now try...", "Run...several times", "Then run...").

### New Findings

#### [IMPROVEMENT] — Transfer question solution mentions GPU parallelism (scope boundary)

**Location:** Check 2 solution, line 683
**Issue:** The solution to the transfer question says "better utilization of GPU parallelism" as a pro of larger batch sizes. The lesson's own constraint block (line 83) explicitly states "No GPU parallelism or data loading details." The solution introduces a concept the student hasn't been taught and that the lesson itself declared out of scope. This creates an inconsistency: the lesson promises not to cover GPU details, then uses GPU parallelism as a reasoning point in an exercise answer.
**Student impact:** Mild. The student may wonder "what does GPU parallelism mean? Was I supposed to know that?" It doesn't block understanding of the core tradeoff, but it introduces unexplained jargon in a place where the student is trying to synthesize what they've learned. The answer should only reference concepts taught in this lesson.
**Suggested fix:** Replace "better utilization of GPU parallelism" with something within scope, such as "smoother loss curve (less oscillation)" or simply remove that bullet point. The two remaining pros (more accurate gradients, smoother convergence) are sufficient.

#### [POLISH] — Transfer question solution also mentions GPU memory

**Location:** Check 2 solution, line 689
**Issue:** The cons mention "uses more GPU memory per step." This is another GPU-specific detail that's outside the lesson's stated scope. It's less prominent than the parallelism mention (it's the last item in a list) but still introduces a concept the student has no framework for.
**Student impact:** Negligible. The student can understand "more memory" at a surface level. But for consistency with the scope boundaries, it would be cleaner to focus on concepts that were actually taught.
**Suggested fix:** Replace with "each step processes more data, so individual steps take longer" or drop it entirely.

#### [POLISH] — 1D cross-section SVG has no axis labels

**Location:** Noise as a Feature section, lines 581-643
**Issue:** The 1D cross-section SVG shows the loss landscape curve and two optimization paths with a legend, but has no axis labels. The x-axis represents "parameter value" (or position in parameter space) and the y-axis represents "loss." Without labels, the student must infer what the axes represent from context. The gradient arrows SVG above it also lacks axis labels, but that one is simpler (just arrows from a point). The 1D cross-section is more graph-like and would benefit from minimal labeling.
**Student impact:** Minor. The student can likely infer the axes from the "loss landscape" context and the labels "sharp" and "wide." But adding "Loss" on the y-axis and "Parameters" on the x-axis would make it instantly readable without inference.
**Suggested fix:** Add minimal axis labels: "Loss" vertically on the left edge, "Parameters" horizontally on the bottom edge. Small font, muted color.

### Review Notes

**What works well (reinforced from iteration 1 + new observations):**
- All iteration 1 fixes are cleanly integrated. The lesson reads as a cohesive whole, not as a patched document. The "Jargon Preview" aside is natural, the 1D cross-section SVG fits the visual rhythm of the lesson, and the numbered widget experiments create a clear pedagogical arc.
- The lesson's five modalities are all present and genuinely distinct: polling analogy (intuitive), gradient arrows + 1D cross-section SVGs (visual), formulas (symbolic), 1000-house walkthrough (concrete), contour plot widget (geometric/spatial). This is strong modality coverage.
- The three core concepts (mini-batches, SGD, epochs) are introduced sequentially with a check between mini-batches and epochs, and a transfer question after SGD. The pacing gives the student time to consolidate each concept.
- The narrative arc is clean: problem (scale) -> solution (sampling/mini-batches) -> mechanics (epochs/shuffling) -> naming (SGD) -> surprising benefit (noise helps) -> practice (widget) -> practical details. Each section earns the next.
- The mental model update ("ball rolling downhill, but the hill is shaking") is placed perfectly at the end of the Noise as a Feature section as the capstone callout. It takes the existing analogy and extends it rather than replacing it.
- Connection to prior knowledge is strong throughout: the update rule hasn't changed (aside in Hook), the training loop is the same (InsightBlock in Epochs), hyperparameter framing from 1.1 (Practical Details section), loss landscape bowl from 1.1 (Noise as a Feature section).
- Scope boundaries are respected in the prose. The only boundary violation is in the transfer question solution (GPU parallelism/memory mentions), which is a minor fix.

**Overall assessment:** This lesson is pedagogically sound and ready for use with one minor fix to the transfer question solution. The improvement finding is borderline between IMPROVEMENT and POLISH—the GPU parallelism mention is a scope violation but it's in a collapsible solution that the student only sees after attempting the question. It doesn't disrupt the learning flow. Given this, the verdict is PASS. The GPU fix should be made but does not warrant another full review cycle.
