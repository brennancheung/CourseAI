# Lesson: Debugging and Visualization

**Module:** 2.2 (Real Data) — Lesson 3 of 3
**Slug:** `debugging-and-visualization`
**Type:** CONSOLIDATE
**Status:** Planned

---

## Phase 1: Orient — Student State

The student just completed the MNIST project (a STRETCH lesson), their first end-to-end model on real data. They are riding the high of "I trained something real" but have limited diagnostic tools when things go wrong. Their debugging instincts are currently: print shapes, stare at code, restart from scratch.

### Relevant Concepts with Depths

| Concept | Depth | Source Lesson | Notes |
|---------|-------|---------------|-------|
| Tensor shape, dtype, device — "check these first" | DEVELOPED | tensors (2.1.1) | Debugging trinity — already established as first instinct |
| `model.parameters()` for collecting all learnable tensors | DEVELOPED | nn-module (2.1.3) | Can enumerate all parameters; used in optimizer construction |
| nn.Module subclass pattern (__init__ + forward()) | DEVELOPED | nn-module (2.1.3) | Knows how layers are defined and how data flows through forward() |
| Complete PyTorch training loop pattern | DEVELOPED | training-loop (2.1.4) | forward -> loss -> backward -> update; practiced on real data in mnist-project |
| `.grad` attribute (where gradients live) | DEVELOPED | autograd (2.1.2) | Knows gradients are stored on leaf tensors as an attribute |
| `backward()` = press Rewind | DEVELOPED | autograd (2.1.2) | Knows backward walks the computational graph |
| nn.Linear(in, out) weight shape = (out, in) | DEVELOPED | nn-module (2.1.3) | Can predict parameter shapes for linear layers |
| `torch.no_grad()` context manager | DEVELOPED | autograd (2.1.2) | Knows it pauses graph recording; used in evaluation loop |
| model.train() / model.eval() in practice | DEVELOPED | mnist-project (2.2.2) | Practiced for real with dropout and batch norm |
| Train/test evaluation loop pattern | DEVELOPED | mnist-project (2.2.2) | model.eval() + no_grad + iterate test_loader |
| DataLoader stacking / collation | INTRODUCED | datasets-and-dataloaders (2.2.1) | Knows DataLoader calls torch.stack(); inconsistent shapes cause RuntimeError |
| nn.Flatten() module | INTRODUCED | mnist-project (2.2.2) | Flattens [B, 1, 28, 28] to [B, 784] |
| Vanishing gradients | DEVELOPED | backpropagation (1.3.1) | 0.25^N decay; telephone game analogy; "flatline" symptom |
| Exploding gradients | DEVELOPED | backpropagation (1.3.1) | Mirror of vanishing; NaN symptom |
| Training curves as diagnostic tool | DEVELOPED | overfitting-and-regularization (1.3.7) | "Scissors pattern" — train and val loss divergence |
| "Flatline = vanishing, NaN = exploding" | DEVELOPED | training-dynamics (1.3.6) | Diagnostic symptom guide |

### Mental Models Already Established

- "Shape, dtype, device — check these first" (tensors debugging trinity)
- "Flatline = vanishing, NaN = exploding" (gradient symptom guide)
- "The scissors pattern" (train/val divergence = overfitting)
- "Same heartbeat, new instruments" (loop structure doesn't change; tools do)
- "backward() = press Rewind" (gradient computation as graph traversal)
- "Parameters are knobs" (grounded in model.parameters() API)

### What Was NOT Covered That's Relevant Here

- `torchinfo` / `torchsummary` for model inspection (never mentioned)
- TensorBoard or any logging/visualization tool (never mentioned)
- Gradient magnitude inspection during training (concept of vanishing/exploding known, but never checked gradients programmatically in PyTorch)
- Systematic shape debugging workflow (shape errors were seen as "broken Dataset" negative example, but no systematic approach taught)
- Model summary / architecture inspection tools (manually counted parameters in nn-module, but no automated tool)

### Readiness Assessment

The student is fully prepared. All prerequisite concepts are at DEVELOPED or higher depth. The student has hit real bugs (shape mismatches, forgetting model.eval(), accumulation traps) across multiple lessons. What they lack is a systematic toolkit — they know the symptoms but not the diagnostic instruments. This is the perfect moment for a tooling lesson: high motivation (just finished a project with real bugs), low conceptual novelty (no new ML theory), practical payoff (tools they can use immediately).

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to systematically diagnose and visualize common PyTorch training failures using torchinfo for model inspection, gradient magnitude checking for training health, and TensorBoard for monitoring training runs.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Tensor shape/dtype/device | DEVELOPED | DEVELOPED | tensors (2.1.1) | OK | Shape debugging requires fluent shape reasoning |
| model.parameters() | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Gradient inspection iterates over model.parameters() |
| nn.Module subclass pattern | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | torchinfo operates on nn.Module instances |
| Complete training loop | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | TensorBoard integration goes inside the training loop |
| `.grad` attribute | DEVELOPED | DEVELOPED | autograd (2.1.2) | OK | Gradient magnitude checking reads `.grad` on parameters |
| Vanishing/exploding gradients | DEVELOPED | DEVELOPED | backpropagation (1.3.1) | OK | Gradient checking is the practical detection method for these |
| Training curves as diagnostic | DEVELOPED | DEVELOPED | overfitting-and-regularization (1.3.7) | OK | TensorBoard automates what the student plotted manually with matplotlib |
| model.train()/model.eval() | DEVELOPED | DEVELOPED | mnist-project (2.2.2) | OK | TensorBoard logging must respect train/eval mode |
| DataLoader stacking | INTRODUCED | INTRODUCED | datasets-and-dataloaders (2.2.1) | OK | Only need recognition-level for shape error diagnosis |
| nn.Linear weight shape (out, in) | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Needed to verify torchinfo output against manual calculation |

**All prerequisites OK.** No gaps. The student has strong conceptual foundations and practical experience. This lesson layers tools on top of existing understanding.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Shape errors mean my model design is fundamentally wrong" | Shape errors feel catastrophic — big red tracebacks with incomprehensible tensor dimensions. The student hasn't developed the habit of reading the error message systematically. | Show a shape error caused by a single missing `nn.Flatten()` — the model architecture is correct, only the plumbing between layers is wrong. Fix is one line. | Section 4 (Shape Errors), immediately after the first shape error example |
| "If loss is going down, training is working fine" | The student has seen loss decrease in every training run so far. They haven't experienced silent failures where loss decreases but the model is learning the wrong thing (e.g., all-same predictions for an imbalanced dataset, or gradient flow is dead in later layers). | Show a model where loss decreases but accuracy is stuck at ~10% (random chance on MNIST) because the model is outputting nearly identical predictions for every input — loss went down by learning the class prior, not features. | Section 6 (TensorBoard / Logging), after introducing metric logging |
| "Gradient checking is only for when things are obviously broken (NaN/divergence)" | The student's mental model of gradient problems is binary: "working" or "NaN." They don't realize gradients can be unhealthily small (not zero, just ineffective) or that gradient magnitudes vary dramatically across layers even in a "working" model. | Show gradient magnitudes across layers of a deep network: first layer gradients are 1000x smaller than last layer gradients. Loss is still decreasing, no NaN, but early layers are barely learning. | Section 5 (Gradient Checking), as the central motivating example |
| "TensorBoard is for production ML — overkill for learning" | TensorBoard feels like a heavy "real engineering" tool. The student might think matplotlib plots are sufficient for their needs. | Side-by-side comparison: matplotlib requires stopping training, writing plot code, re-running; TensorBoard shows live curves during training with zoom, comparison across runs, and no code changes to view differently. The convenience argument wins when you show "what if you want to compare 3 different learning rates?" | Section 6 (TensorBoard), during the motivation/hook |
| "torchinfo just prints what I already know" | The student manually counted parameters in nn-module (2.1.3) and might think a summary tool is redundant. | Show torchinfo on a model with nn.Flatten — it reveals the exact shape at every layer, catching the "784 vs 28" mismatch that manual parameter counting misses because it doesn't trace shapes through the forward pass. | Section 4 (torchinfo), right after the first example |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Shape mismatch from missing Flatten** — Model takes [B, 1, 28, 28] input but first Linear expects [B, 784]. RuntimeError with shape trace. | Positive | Show systematic shape error diagnosis with torchinfo | Uses their exact MNIST model from last lesson — maximum familiarity. The error is a single missing layer, not a fundamental design flaw. Demonstrates that shape errors are plumbing problems. |
| **Shape mismatch from wrong Linear dimensions** — nn.Linear(784, 128) followed by nn.Linear(64, 10) instead of nn.Linear(128, 10). | Positive | Show how torchinfo catches dimension mismatches between layers | Different root cause than example 1 (typo in layer sizes vs missing transform). Demonstrates that torchinfo's output-shape column makes these errors visible before running data through. |
| **Healthy vs unhealthy gradient magnitudes** — Same architecture, two different initializations. One has balanced gradients (~0.01 across all layers), the other has vanishing gradients in early layers (~1e-8 vs ~0.1 in later layers). | Positive | Show gradient magnitude checking as a diagnostic tool | Connects directly to vanishing gradients concept (DEVELOPED from 1.3). Makes the theoretical concept concrete and actionable with PyTorch code. |
| **Loss decreasing but model not learning** — A model where loss goes down but all predictions converge to the same class. | Negative | Disprove "loss going down = everything is fine" | This is the most important negative example in the lesson. Students need to learn that loss alone is insufficient — you must also monitor accuracy, per-class accuracy, or actual predictions. TensorBoard makes this easy to catch. |
| **TensorBoard comparison of 3 learning rates** — Same model trained with lr=0.001, lr=0.01, lr=0.1. TensorBoard overlays the training curves. | Positive | Show TensorBoard's killer feature: run comparison | Connects to learning rate intuition from gradient-descent (1.1.4) and learning-rate (1.1.5). The student already knows too-high/too-low behavior conceptually; TensorBoard makes it visible in real training. |
| **"It works on my machine" false positive** — A model that appears to train well on the first batch but fails on the full dataset because of a subtle bug (e.g., training on the same batch every epoch due to not iterating the DataLoader properly). | Negative | Disprove "first batch looks good = model is correct" | Teaches the importance of monitoring metrics across the full epoch, not just spot-checking. A common beginner trap. |

### Gap Resolution

No gaps identified. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

You just trained your first real model. It worked — 97% accuracy on MNIST. But what happens when it doesn't work? The reality of deep learning is that most of your time isn't spent writing models — it's spent figuring out why they're broken. Shape errors crash your training with cryptic messages. Gradients silently vanish, leaving early layers frozen while loss slowly decreases. Loss goes down but accuracy stays stuck at random chance. These aren't hypothetical problems — they're the bugs you will hit on your next project. This lesson gives you three diagnostic instruments: torchinfo (X-ray your model's shapes before running any data), gradient magnitude checking (take your model's pulse during training), and TensorBoard (monitor your model's vital signs in real time). After this lesson, you won't just know how to train a model — you'll know how to figure out why it's not training.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example (code)** | Actual Python code for torchinfo.summary(), gradient checking loop, TensorBoard SummaryWriter — all runnable in Colab | This is a tools lesson; the code IS the concept. Every tool must be shown working on familiar models. |
| **Visual** | TensorBoard screenshots showing training curves, gradient histograms, run comparison overlays | TensorBoard is inherently visual — you cannot teach it without showing what the dashboard looks like. Screenshots or embedded images of the TensorBoard UI make the lesson concrete. |
| **Concrete example (worked)** | Step-by-step shape error diagnosis: read the error message, identify the mismatched dimensions, use torchinfo to locate the problem layer, apply the fix | Debugging is a process, not a fact. Walking through the diagnosis step-by-step teaches the workflow, not just the tool. |
| **Verbal/Analogy** | "torchinfo is an X-ray" (see inside the model without running data), "gradient checking is taking the model's pulse" (vital signs during training), "TensorBoard is a flight recorder" (continuous monitoring you review after the fact) | Medical/diagnostic analogies ground each tool in a familiar purpose — the student immediately understands WHY each tool exists, not just how to call it. |
| **Negative example** | Loss-decreasing-but-not-learning scenario; first-batch-looks-good false positive | Negative examples are critical for debugging lessons because the whole point is recognizing when something looks OK but isn't. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 tools (torchinfo, gradient magnitude checking, TensorBoard). These are tools/APIs, not new ML theory. The underlying concepts (shapes, gradients, training curves) are all at DEVELOPED depth already.
- **Previous lesson load:** STRETCH (mnist-project was a major integration project with 2-3 new ML concepts).
- **This lesson's load:** CONSOLIDATE — appropriate. After the STRETCH of MNIST, the student needs breathing room. This lesson introduces new tools but applies them to familiar problems. The cognitive demand is "learn a new API" not "understand a new concept."
- **Assessment:** Load is appropriate. The BUILD -> STRETCH -> CONSOLIDATE trajectory completes cleanly. No new ML theory; all novelty is in tooling and workflow.

### Connections to Prior Concepts

| Existing Concept | Connection in This Lesson |
|-----------------|--------------------------|
| "Shape, dtype, device — check these first" (2.1.1) | torchinfo automates and systematizes this instinct. Instead of printing shape manually, torchinfo shows all shapes at once. |
| Vanishing/exploding gradients (1.3.1) | Gradient magnitude checking is the practical detection method. The student learned the theory; now they learn to check for it in their own models. |
| "Flatline = vanishing, NaN = exploding" (1.3.6) | Extended: gradient checking catches problems BEFORE they manifest as flatline or NaN. Early detection, not just symptom recognition. |
| Training curves as diagnostic (1.3.7) | TensorBoard automates what the student did with matplotlib. Same concept, better instrument. |
| "The scissors pattern" (1.3.7) | TensorBoard makes the scissors visible in real time. The student can watch train/val diverge during training instead of plotting afterward. |
| Learning rate behavior (1.1.4, 1.1.5) | TensorBoard run comparison shows too-high/too-low/just-right learning rates on the same plot. Makes the Goldilocks concept from lesson 5 of Series 1 visually concrete. |
| model.parameters() (2.1.3) | Gradient checking iterates over model.parameters() — the same API the student already uses for optimizer construction. |
| "Same heartbeat, new instruments" (2.1.4) | This lesson literally adds new instruments to the heartbeat: logging calls go inside the training loop without changing its structure. |

**Potentially misleading analogies:** None identified. The established analogies all extend cleanly.

### Scope Boundaries

**This lesson IS about:**
- Systematic shape error diagnosis workflow
- torchinfo.summary() for model inspection (shapes, parameter counts, output sizes)
- Gradient magnitude checking during training (per-layer gradient norms)
- TensorBoard basics (scalar logging, training curves, run comparison)
- Common failure patterns and how to recognize them
- Target depth: DEVELOPED for torchinfo and gradient checking (student practices in Colab), INTRODUCED for TensorBoard (student sees it work, uses it in guided Colab exercise)

**This lesson is NOT about:**
- TensorBoard advanced features (histograms, embeddings, graph visualization, profiler)
- Weights & Biases, MLflow, or other experiment tracking tools
- PyTorch Profiler or performance optimization
- Debugging distributed training
- Unit testing for ML models
- Hyperparameter tuning or search strategies
- Custom logging frameworks
- Advanced torchinfo options (verbose mode, custom input sizes for complex architectures)
- Debugging CUDA/GPU-specific errors

### Lesson Outline

**1. Context + Constraints**
What: This lesson teaches debugging and monitoring tools for PyTorch training. We're learning three instruments, not new ML theory.
Not: No advanced TensorBoard features, no experiment tracking platforms, no performance profiling.
Framing: "You know how to build and train models. Now learn how to figure out what's wrong when they don't work."

**2. Hook — "The Silent Failure"**
Type: Misconception reveal.
Show a training run where loss decreases smoothly from 2.3 to 0.5 over 10 epochs. Ask: "Is this model training correctly?" Then reveal: accuracy is 10% — random chance. The model learned to always predict the most common class. Loss went down because the class prior is a valid (terrible) solution.
Why this hook: Immediately establishes that the student's current diagnostic tools (watching loss decrease) are insufficient. Creates the need for better instruments.

**3. Explain — Tool 1: torchinfo (X-ray Your Model)**
- Motivation: "You manually counted parameters in nn-module. What about a 50-layer model?"
- Install and basic usage: `torchinfo.summary(model, input_size=(1, 1, 28, 28))`
- Walk through output: layer names, output shapes at every layer, parameter counts, total parameters
- **Example 1:** Their MNIST model from last lesson — torchinfo confirms the shapes they already know
- **Example 2:** A broken model (missing Flatten) — torchinfo shows [B, 1, 28, 28] going into Linear(784, 128), immediately reveals the mismatch
- **Example 3 (negative):** Wrong Linear dimensions (128 -> 64 gap) — torchinfo catches it in the output shape column
- Misconception addressed: "torchinfo just prints what I already know" — disproved by the shape mismatch it catches that manual counting misses

**4. Check 1 — Predict-and-Verify**
Show a model definition with 3 Linear layers and ask the student to predict: (a) output shape at each layer, (b) where the shape error will occur, (c) total parameters. Then show torchinfo output to verify.

**5. Explain — Tool 2: Gradient Magnitude Checking (Take the Model's Pulse)**
- Motivation: "Your model from last lesson had vanishing gradient potential — how would you know?"
- Code: iterate `model.named_parameters()`, compute `param.grad.norm()` for each layer after one backward pass
- **Example 4:** Healthy model — gradient norms are ~0.01-0.1 across all layers, relatively balanced
- **Example 4b:** Same architecture, bad initialization — early layer gradients are ~1e-8 while last layer is ~0.1. Loss is still decreasing. No NaN. But early layers are frozen.
- Connection: "Remember 'flatline = vanishing'? Gradient checking catches this BEFORE you see the flatline. It's early detection."
- Misconception addressed: "Gradient checking is only for obvious failures" — disproved by the healthy-looking-but-unhealthy model
- Practical pattern: Add a `log_gradient_norms()` helper function; call it every N iterations

**6. Check 2 — Transfer Question**
"Your colleague's model has loss decreasing but accuracy stuck at 52% on a binary classification task. They say 'the model is learning, just slowly.' What would you check first, and what tool would you use?"
Expected: Check gradient magnitudes per layer (are early layers learning?), check per-class accuracy (is it predicting all one class?), use TensorBoard to visualize both metrics over time.

**7. Explain — Tool 3: TensorBoard (Flight Recorder)**
- Motivation: "You plotted training curves with matplotlib in the MNIST project. What if you want to compare 3 different learning rates? Rerun 3 times, collect data, write plot code, run it..." vs TensorBoard: runs log automatically, dashboard shows live, comparison is one click.
- Setup: `torch.utils.tensorboard.SummaryWriter`
- Basic logging: `writer.add_scalar('Loss/train', loss, epoch)`, `writer.add_scalar('Accuracy/test', accuracy, epoch)`
- Integration into existing training loop: show where the 2 logging lines go (inside the loop, without changing the heartbeat)
- **Example 5:** Side-by-side TensorBoard screenshots of 3 learning rate runs — student can see too-high (oscillating), too-low (barely decreasing), just-right (smooth descent). Connects to learning rate lesson (1.1.5).
- Misconception addressed: "TensorBoard is overkill" — disproved by the 3-LR comparison that would take 30 lines of matplotlib code vs 2 lines of logging
- **Negative example (Example 6):** Training on same batch every epoch — TensorBoard shows suspiciously smooth loss curve with no noise (real training has noise from mini-batches). The smoothness IS the bug signal.

**8. Elaborate — The Debugging Checklist**
Synthesize all three tools into a systematic debugging checklist:
1. **Before training:** Run torchinfo.summary() — verify shapes, parameter count, output sizes
2. **First iteration:** Check gradient magnitudes — are all layers receiving gradients?
3. **During training:** Monitor TensorBoard — loss AND accuracy, train AND test
4. **If loss is stuck:** Check gradients (vanishing?), check learning rate (too small?)
5. **If loss is NaN:** Check gradients (exploding?), check data (NaN in inputs?), check learning rate (too large?)
6. **If train good but test bad:** Scissors pattern — add regularization (callback to 1.3.7)

This checklist connects every tool to a specific diagnostic scenario the student already understands conceptually.

**9. Practice — Colab Notebook**
Scaffolding: Guided -> Supported -> Independent

Exercises:
1. **(Guided)** Run torchinfo on their MNIST model. Verify parameter count matches manual calculation from mnist-project.
2. **(Guided)** Introduce a shape bug (remove Flatten), run torchinfo, identify the problem, fix it.
3. **(Supported)** Write a gradient checking function. Run it on a healthy model and a poorly-initialized model. Compare magnitudes per layer.
4. **(Supported)** Add TensorBoard logging to their MNIST training loop. Train for 10 epochs. Open TensorBoard and examine the curves.
5. **(Supported)** Train 3 runs with different learning rates (0.001, 0.01, 0.1). Compare in TensorBoard. Identify which is too high, too low, and just right.
6. **(Independent)** Given a "broken" training script with 3 intentional bugs (a shape error, missing model.eval(), and a subtle data loading bug), use the debugging checklist to find and fix all three.

**10. Summarize**
Key takeaways:
- torchinfo: X-ray your model BEFORE training — catch shape errors and verify architecture
- Gradient checking: take the model's pulse DURING training — catch vanishing/exploding before symptoms appear
- TensorBoard: monitor vital signs ACROSS training — compare runs, watch for divergence, catch silent failures
- The debugging checklist: a systematic workflow, not random guessing

Echo the mental model: "Same heartbeat, new instruments" — logging goes INTO the training loop; the loop structure doesn't change.

**11. Next Step**
"You can now build, train, evaluate, AND debug models on real data. Module 2.2 is complete. Next up: Module 2.3 (Practical Patterns) — saving and loading models, GPU training, and a capstone project on Fashion-MNIST where you'll use all of these tools for real."

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 2

### Verdict: NEEDS REVISION

No critical structural problems with the lesson's teaching flow. One critical finding (missing visual modality for TensorBoard) that significantly weakens a core tool introduction. Four improvement findings that would make the lesson notably more effective. Two polish items.

### Findings

#### [CRITICAL] — TensorBoard visual modality is completely absent

**Location:** Tool 3: TensorBoard section (lines 631-862)
**Issue:** The planning document explicitly lists "TensorBoard screenshots showing training curves, gradient histograms, run comparison overlays" as a planned visual modality. The built lesson has zero visual representation of what TensorBoard actually looks like. The student reads about "live dashboard with curves, zoom, and overlay comparison" and sees three GradientCards describing what each learning rate does, plus a ComparisonRow for the same-batch bug. But they never see the TensorBoard UI. The student is told to "Open TensorBoard and all three runs appear on the same plot" without ever seeing what that plot looks like.
**Student impact:** The student is learning a visual debugging tool without seeing any visuals. They must imagine what the dashboard looks like based on text descriptions. This is like teaching someone to use a microscope by describing what they would see. The "killer feature" (run comparison) loses most of its persuasive power when it is described rather than shown. The misconception "TensorBoard is overkill" is harder to disprove without a visual comparison.
**Suggested fix:** Add at least one visual representation of TensorBoard output. Options: (1) a static image/screenshot showing the TensorBoard scalar dashboard with overlaid runs, (2) a simplified schematic diagram showing what the student would see (three labeled curves on the same axes), or (3) a Recharts-based inline chart approximating what TensorBoard displays for the 3-LR comparison. Even a simple labeled sketch would be better than nothing. This is the single highest-impact fix for the lesson.

#### [IMPROVEMENT] — Hook reveal buries the key insight in collapsible content

**Location:** Section 2: "The Silent Failure" hook (lines 91-153)
**Issue:** The most important pedagogical moment in the hook -- the reveal that accuracy is 10% despite decreasing loss -- is hidden behind a `<details>` element. The student must click "Reveal the truth" to see it. The lesson's entire motivation depends on this reveal landing emotionally. But `<details>` elements are easy to skip, especially for ADHD learners who may be scanning. The aside (WarningBlock "Loss Is Not Enough") partially spoils the reveal by stating the conclusion before the student clicks.
**Student impact:** Two failure modes: (1) Student clicks reveal but the aside already told them the answer, deflating the "aha." (2) Student skips the reveal entirely, missing the hook, and arrives at the tools section without feeling the need for better instruments.
**Suggested fix:** Move the aside WarningBlock to appear AFTER the reveal, not alongside it. The aside currently spoils the tension. Alternatively, keep the `<details>` but make the aside more neutral (e.g., "Think about what could go wrong here" rather than giving away the answer). The reveal itself should stay collapsible -- that interaction is good -- but the aside must not pre-empt it.

#### [IMPROVEMENT] — Gradient checking section does not address "when should I stop worrying?"

**Location:** Tool 2: Gradient Magnitude Checking (lines 407-573)
**Issue:** The lesson explains what healthy vs unhealthy gradients look like (the ComparisonRow is effective) and provides a "what to look for" checklist. But it never tells the student what to DO when they find unhealthy gradients. The student learns to detect the problem but not to respond. The diagnostic checklist (section 8) mentions "check learning rate" and "add regularization" but those are vague. For a CONSOLIDATE lesson that aims to give the student a "repeatable diagnostic workflow," the workflow is incomplete -- it diagnoses but does not prescribe.
**Student impact:** The student runs gradient checking, sees early-layer norms are 1e-7, and thinks "OK, my gradients are vanishing... now what?" They know the symptom but not the next step. This gap is especially frustrating because gradient checking is positioned as "early detection" -- but early detection without a response plan is just early anxiety.
**Suggested fix:** Add a brief "what to do about it" list after the "what to look for" bullet points. Keep it concise and within scope boundaries (no hyperparameter tuning). Something like: "Vanishing? Try a different activation function (ReLU instead of sigmoid), check your initialization, or try a skip connection. Exploding? Lower your learning rate or add gradient clipping (a future topic)." This connects to concepts the student already has (activation functions, skip connections from nn-module) without introducing new theory.

#### [IMPROVEMENT] — The "loss decreasing but not learning" scenario in the hook lacks a concrete diagnosis path

**Location:** Section 2: Hook + Section 8: Debugging Checklist (lines 91-153 and 864-959)
**Issue:** The hook shows loss=2.302 dropping to loss=0.512 with 10% accuracy. The reveal explains the model learned the class prior. But the lesson never shows HOW the student would have caught this using the tools taught. The debugging checklist says "Loss down, accuracy flat: check predictions" (line 937-939) but the connection between the hook and the checklist is implicit, not explicit. The student has to mentally connect the hook example to checklist item 4 themselves.
**Student impact:** The hook creates a problem. The tools section teaches instruments. But the lesson never closes the loop by saying "here is how you would have caught the hook's bug using the tools you just learned." The narrative arc is setup-build without a satisfying payoff that ties back to the opening.
**Suggested fix:** After the debugging checklist, add a brief "Closing the Loop" paragraph that explicitly revisits the hook: "Remember the silent failure from the opening? Here is exactly how the debugging checklist would have caught it: Step 1 (torchinfo) would pass -- shapes are fine. Step 2 (gradient check) might reveal something -- [brief note]. Step 3 (TensorBoard) with both loss AND accuracy logged would immediately show loss decreasing while accuracy stays at 10%. The bug is caught in the first epoch, not after 10." This closes the narrative arc.

#### [IMPROVEMENT] — Check 2 references TensorBoard before it has been taught

**Location:** Check 2: Transfer Question (lines 575-629)
**Issue:** The check appears between Tool 2 (gradient checking) and Tool 3 (TensorBoard). The expected answer (shown in the details reveal) includes item 3: "Use TensorBoard (next section) to watch loss AND accuracy together." This means the check's answer requires a tool the student has not learned yet. The check acknowledges this with "(next section)" but it still tests knowledge the student does not have.
**Student impact:** The student reads the check after learning about gradient checking. They can answer items 1 and 2 based on what they know. But item 3 is a forward reference. If the student clicks the reveal and sees "Use TensorBoard," they may feel the check was unfair ("I haven't learned that yet") or that the check is testing their ability to read ahead rather than apply what they have learned.
**Suggested fix:** Either (a) move Check 2 to after the TensorBoard section so all three tools are available, or (b) reframe item 3 in the answer to say "you would want to monitor both metrics over time -- you will learn TensorBoard for this in the next section" to explicitly mark it as a preview rather than an expected answer. Option (a) is cleaner.

#### [POLISH] — ModuleCompleteBlock achievement list includes concepts not taught in this lesson

**Location:** Module 2.2 Complete block (lines 1070-1089)
**Issue:** The achievements list includes "Regularization in practice: dropout, batch norm, weight decay" which was taught in lesson 2 (mnist-project), not this lesson. This is not wrong -- it is a module-level summary -- but it could confuse a student who just finished a debugging lesson and wonders why regularization is being claimed as an achievement.
**Student impact:** Minor. The student might briefly wonder "wait, did I miss something about regularization in this lesson?" but will likely figure out it is a module-level summary.
**Suggested fix:** No change needed if ModuleCompleteBlock is understood as a module recap. If it could be misread as "what you learned today," consider adding a brief intro line like "Across this module, you learned:" before the list. Very low priority.

#### [POLISH] — "In nn.Module, you manually counted parameters" phrasing is slightly ambiguous

**Location:** Tool 1: torchinfo section, first paragraph (line 167)
**Issue:** The phrase "In nn.Module, you manually counted parameters" reads as though the student counted parameters while inside an nn.Module class. The intended meaning is "in the nn-module lesson." The phrasing is a lesson name reference but could be read as a location.
**Student impact:** Momentary confusion, quickly resolved by context. The student might briefly wonder "did I count parameters inside __init__?" before understanding the reference.
**Suggested fix:** Change to "In the nn.Module lesson, you manually counted parameters" to make the lesson reference explicit.

### Review Notes

**What works well:**

The lesson has a strong overall structure. The three-tool framework with consistent analogies (X-ray, pulse, flight recorder) gives the student a memorable mental model. The diagnostic checklist in section 8 is genuinely useful -- it synthesizes the three tools into a workflow rather than leaving them as disconnected utilities. The "Same heartbeat, new instruments" callback is effective and well-placed.

The examples are well-chosen. The missing-Flatten shape error uses the student's own MNIST model from last lesson, which is exactly right. The healthy-vs-unhealthy gradient comparison (ComparisonRow) makes the abstract concept of "vanishing gradients" visually concrete with real numbers. The same-batch bug is a great negative example that most students would not catch without TensorBoard.

The checks are well-designed. Check 1 (predict-and-verify on shapes and parameters) is appropriately mechanical. The Colab exercise progression (guided to independent) is properly scaffolded with a satisfying capstone (find 3 bugs in a broken script).

**Systemic pattern:**

The lesson's main weakness is a gap between "diagnosis" and "response." The tools tell the student what is wrong, but the lesson does not consistently tell them what to do about it. This surfaces in the gradient checking section (no remediation guidance) and in the hook (problem is stated but never revisited with the tools). For a CONSOLIDATE lesson aimed at building practical independence, the "so what do I do now?" question needs answering.

**Modality count:**

The core "debugging as a systematic workflow" concept is presented through: (1) verbal/analogy (X-ray, pulse, flight recorder), (2) concrete code examples (multiple), (3) symbolic/structured (the 4-phase debugging checklist), (4) negative examples (silent failure, same-batch bug). That is 4 modalities. The missing 5th is visual -- specifically, what TensorBoard output looks like. Adding a visual representation of TensorBoard would bring this to 5 strong modalities.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 3

### Verdict: PASS

All critical and improvement findings from iteration 1 have been addressed. The lesson is effective and ready for use. Three minor polish items remain that do not affect the student's learning experience.

### Iteration 1 Fix Verification

All 7 iteration 1 findings have been resolved:

1. **[CRITICAL] TensorBoard visual modality absent** — FIXED. A Recharts-based TensorBoardMockup component (lines 63-162) now renders an inline chart approximating the TensorBoard dashboard. Dark theme with orange header, tab selector ("SCALARS / IMAGES / GRAPHS"), three learning rate curves overlaid, and mock run selector at the bottom. This is convincing and informative. The visual modality is now present.

2. **[IMPROVEMENT] Hook aside spoils the reveal** — FIXED. The WarningBlock "Loss Is Not Enough" is now placed in its own Row AFTER the reveal (lines 273-282), with a comment explaining the placement: "placed AFTER the reveal so it does not spoil the tension." The reveal can now land without the aside pre-empting the answer.

3. **[IMPROVEMENT] Gradient checking lacks remediation guidance** — FIXED. A "What to do about it" section (lines 692-717) now appears immediately after the "What to look for" bullet list. Two-column grid: "Vanishing gradients?" (ReLU, initialization, batch norm, skip connections) and "Exploding gradients?" (lower learning rate, gradient clipping, check data, batch norm). Future topics properly marked. The diagnostic workflow is now complete: detect AND respond.

4. **[IMPROVEMENT] Hook not revisited with tools** — FIXED. A "Solving the Opening Puzzle" section (lines 1124-1161) now explicitly walks through all three checklist steps applied to the opening silent failure. It closes the narrative arc: torchinfo passes (structural bug), gradient checking might reveal single-class drift, TensorBoard with accuracy logging catches the bug immediately.

5. **[IMPROVEMENT] Check 2 references TensorBoard before taught** — FIXED. Check 2 is now positioned after the TensorBoard section (line 971), with a comment noting it was moved: "moved after TensorBoard so all 3 tools are available." The answer's three-part response (per-class accuracy, gradient magnitudes, TensorBoard monitoring) now refers only to tools the student has already learned.

6. **[POLISH] ModuleCompleteBlock achievement list** — NOT CHANGED (was assessed as no change needed). The achievement list remains a module-level summary, which is appropriate for a ModuleCompleteBlock. No action required.

7. **[POLISH] "In nn.Module" phrasing** — FIXED. Line 293 now reads "In the nn.Module lesson, you manually counted parameters" — explicit lesson reference.

### Findings

#### [POLISH] — `model.named_parameters()` introduced without noting it differs from `model.parameters()`

**Location:** Tool 2: Gradient Magnitude Checking, line 553-556
**Issue:** The lesson says `model.named_parameters()` is "the same API you used to construct optimizers in nn.Module." The student actually used `model.parameters()` (without names) for optimizer construction. `named_parameters()` is a variant that also yields the parameter name as a string. The connection is correct in spirit (same underlying parameter collection), but the phrasing implies they are identical when they are not.
**Student impact:** Negligible. The code makes the difference obvious — the loop unpacks `name, param` instead of just `param`. The student will understand immediately.
**Suggested fix:** Change "the same API" to "a variant of the same API" or "like model.parameters() but also gives you the name." Very low priority since the code is self-explanatory.

#### [POLISH] — `--` (double hyphen) renders literally in string props instead of em dash

**Location:** LessonHeader description (line 172), ConstraintBlock first item (line 205), SummaryBlock TensorBoard description (line 1255)
**Issue:** String props cannot use `&mdash;` HTML entities. The `--` renders as two literal hyphens in the browser. The lesson's JSX text content consistently uses `&mdash;` for em dashes, but string attributes use `--` because HTML entities are not processed in JSX string literals.
**Student impact:** Purely cosmetic. The student sees `--` instead of an em dash in the header description, constraints block, and one summary item. Does not affect comprehension.
**Suggested fix:** Use the Unicode em dash character (`\u2014`) directly in the string literals, or use a template literal with the actual Unicode character. This is a project-wide pattern (other lessons have the same issue), so it may be best addressed as a project-level convention fix rather than in this individual lesson. Very low priority.

#### [POLISH] — Gradient checking code places `log_gradient_norms` after `optimizer.step()` without noting `.grad` persists through step

**Location:** Gradient checking integration code block (lines 654-665)
**Issue:** The code calls `log_gradient_norms(model)` after `optimizer.step()`. A careful student might wonder: "Does `optimizer.step()` modify or clear the `.grad` values?" It does not — `.grad` is only cleared by `zero_grad()`. But this is not stated. The student who has internalized "clear, compute, use" from the training-loop lesson knows that `step()` uses gradients but does not clear them. However, the question might arise briefly.
**Student impact:** Momentary uncertainty for attentive students. Quickly resolved by the "clear, compute, use" mental model from the training-loop lesson. No actual confusion.
**Suggested fix:** Optionally add a brief comment in the code block: `# .grad values still available after step() -- only zero_grad() clears them`. Very low priority.

### Review Notes

**What works well:**

The lesson has landed in a strong state. All five iteration 1 fixes significantly improved the lesson:

- The TensorBoardMockup is the single biggest improvement. The dark-themed Recharts chart with the mock TensorBoard chrome (header, tabs, run selector) makes the tool concrete and visually compelling. The student can now see what "three learning rates on the same plot" actually looks like, which makes the "TensorBoard vs matplotlib" argument visceral rather than abstract.

- Moving the WarningBlock after the reveal preserves the hook's emotional impact. The reveal now lands cleanly: "Is this training correctly?" -> student thinks "yes, loss is decreasing" -> reveal shows 10% accuracy -> surprise and motivation.

- The "Solving the Opening Puzzle" section closes the narrative arc satisfyingly. The lesson now has a complete setup-build-payoff structure: hook creates the problem, tools provide instruments, checklist provides workflow, and the closing explicitly shows how the checklist would have caught the opening problem. This is textbook narrative structure for a teaching lesson.

- The remediation guidance in the gradient checking section ("what to do about it") completes the diagnostic workflow. The lesson now teaches detect AND respond, not just detect.

- Moving Check 2 after TensorBoard makes the check fair — all three tools are available when the student encounters the transfer question.

**Pedagogical assessment:**

The lesson meets all pedagogical principles:
- **Motivation rule:** Problem (silent failure) stated before solution (three tools). Strong hook.
- **Modality rule:** 5 modalities — verbal/analogy, concrete code, visual (TensorBoardMockup + ComparisonRows), symbolic/structured (4-phase checklist), negative examples. Exceeds minimum of 3.
- **Example rules:** 4 positive + 2 negative examples. All concrete with real numbers. Exceeds minimums.
- **Misconception rule:** 5 misconceptions addressed. All have concrete counter-examples. Exceeds minimum of 3.
- **Ordering rules:** Concrete before abstract throughout. Problem before solution (hook first). Parts before whole (individual tools before checklist). Simple before complex (torchinfo before gradient checking before TensorBoard).
- **Load rule:** 3 tools, no new ML theory. Within limits.
- **Connection rule:** Every tool explicitly connected to existing concepts (model.parameters, .grad, training curves, "same heartbeat").
- **Reinforcement rule:** Vanishing gradients (from 1.3.1, 6+ lessons ago) are properly reinforced with a concrete callback to the telephone game analogy.
- **Interaction design:** All `<details>` elements have `cursor-pointer` on `<summary>`. No other interactive elements in this lesson (TensorBoardMockup is display-only).
- **Writing style:** Em dashes use `&mdash;` with no spaces throughout JSX text. String props use `--` (project-wide convention).

**Scope compliance:** The lesson stays within its stated boundaries. No advanced TensorBoard features, no W&B/MLflow, no profiling, no GPU debugging, no hyperparameter tuning. The remediation guidance appropriately marks future topics ("a future topic") without teaching them.

**Module completion:** This is the final lesson in Module 2.2. The ModuleCompleteBlock correctly summarizes module-level achievements and points toward Module 2.3 (Practical Patterns). The emotional arc of the module ("My loop works on fake data" -> "Now it works on real images" -> "And I know how to fix it when it doesn't") is complete.

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale
- [x] At least 2 positive examples + 1 negative example, each with stated purpose
- [x] At least 3 misconceptions identified with negative examples
- [x] Cognitive load ≤ 3 new concepts (3 tools, no new ML theory)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated
