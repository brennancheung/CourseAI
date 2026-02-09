# Lesson Plan: Saving, Loading, and Checkpoints

**Module:** 2.3 (Practical Patterns)
**Position:** Lesson 1 of 3
**Slug:** `saving-and-loading`
**Load Type:** BUILD

---

## Phase 1: Orient — Student State

The student has completed all of Series 1 (Foundations) and Modules 2.1 (PyTorch Core) and 2.2 (Real Data). They can build, train, evaluate, and debug a model on MNIST end-to-end.

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Complete PyTorch training loop (forward-loss-backward-update) | DEVELOPED | training-loop (2.1.4) | The four-line heartbeat: model(x), criterion(y_hat, y), loss.backward(), optimizer.step(). Practiced on regression and classification. |
| nn.Module subclass pattern (__init__ + forward) | DEVELOPED | nn-module (2.1.3) | Can define layers in __init__, wire data flow in forward(). Understands model.parameters() collects all learnable tensors. |
| model.parameters() for collecting learnable tensors | DEVELOPED | nn-module (2.1.3) | One call returns every nn.Parameter. Used to construct optimizers. |
| torch.optim.Adam optimizer | DEVELOPED | training-loop (2.1.4) | Constructed with model.parameters() and lr. Has .step() and .zero_grad(). Knows it has internal momentum state from optimizers lesson (1.3.5). |
| model.train() / model.eval() mode switching | DEVELOPED | mnist-project (2.2.2) | Practiced: train() enables dropout + batch-stat BN; eval() disables. Forgetting eval() gives noisy results. |
| Early stopping concept | DEVELOPED | overfitting-and-regularization (1.3.7) | Monitor validation loss, save best model weights, patience hyperparameter. The concept of "save best model weights" was taught but never implemented in PyTorch. |
| Training curves as diagnostic tool | DEVELOPED | overfitting-and-regularization (1.3.7) + debugging-and-visualization (2.2.3) | Plot train + val loss; scissors pattern = overfitting. TensorBoard for monitoring. |
| Device management (.to(), .cpu()) | INTRODUCED | tensors (2.1.1) | Standard device pattern, device mismatch errors, GPU vs CPU tradeoffs. Has NOT trained on GPU yet. |
| weight_decay parameter in optimizer | INTRODUCED | mnist-project (2.2.2) | L2 regularization through optimizer constructor. |
| Systematic debugging checklist (4-phase workflow) | DEVELOPED | debugging-and-visualization (2.2.3) | torchinfo before, gradient check first iteration, TensorBoard during, diagnose by symptom. |

### Mental Models Already Established

- **"Same heartbeat, new instruments"** — The training loop pattern is universal; new tools slot in without changing the rhythm.
- **"Parameters are knobs"** — from Series 1, extended through all of Series 2. The student thinks of model state as a collection of learned values.
- **"Not a black box"** — optimizer.step() does the same thing as the manual update rule. Adam has internal momentum and adaptive rate state.
- **"The complete training recipe"** — Xavier/He init + batch norm + AdamW + dropout + early stopping.

### What Was Explicitly NOT Covered

- Saving/loading models — explicitly deferred from nn-module (2.1.3), training-loop (2.1.4), and mnist-project (2.2.2).
- `torch.save()` and `torch.load()` — never used.
- `state_dict()` — never mentioned by name.
- Checkpointing during training — early stopping says "save best model weights" but the student has never actually persisted them.
- Model export or sharing — never discussed.

### Readiness Assessment

The student is fully ready. They have the complete training loop, understand model.parameters() as the collection of learned values, know that Adam has internal state (momentum, adaptive rates), and have the early stopping concept that motivates checkpointing. The only gap is that "save best model weights" from early stopping was conceptual — this lesson makes it concrete.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **persist and restore model state using state_dict, and implement checkpointing to resume interrupted training**.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| nn.Module and model.parameters() | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Need to understand that a model has named parameters to understand what state_dict captures. |
| torch.optim.Adam (or any optimizer) | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | Need to understand optimizers have internal state (momentum buffers) to understand why optimizer state_dict matters. |
| Complete training loop | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | The checkpoint save/load logic wraps around the existing training loop. |
| Early stopping (save best weights) | INTRODUCED | DEVELOPED | overfitting-and-regularization (1.3.7) | OK | The concept of "save best model weights" motivates this entire lesson. Over-prepared — student has the concept, just not the PyTorch implementation. |
| Python dictionaries | DEVELOPED | assumed (SWE background) | N/A | OK | state_dict is an OrderedDict. The student is a software engineer; Python dicts are basic. |
| model.train() / model.eval() | DEVELOPED | DEVELOPED | mnist-project (2.2.2) | OK | Must call eval() after loading for inference, train() after loading to resume training. |
| Device management (.to()) | INTRODUCED | INTRODUCED | tensors (2.1.1) | OK | Relevant for map_location when loading models saved on GPU to CPU (or vice versa). INTRODUCED is sufficient — we only need recognition here, not application. |

**No gaps. All prerequisites met.**

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"torch.save(model) is the right way to save"** | It works, it is simpler, and many tutorials use it. The student may not realize it creates a fragile pickle that depends on exact class definitions and file paths. | Save a model with torch.save(model), rename the class file or move it, then try to load — it fails with a ModuleNotFoundError. Meanwhile state_dict loads fine because it is just numbers in a dictionary. | Core Explain section — compare both approaches head-on, show the failure mode. |
| **"Loading model weights is enough to resume training"** | The student thinks of training as "model gets better" — so restoring the model should be enough. They may not connect that Adam's momentum buffers and adaptive learning rates are separate state that also needs saving. | Resume training from a checkpoint that only saved model weights (no optimizer state). Show the loss spike — Adam restarts from scratch with no momentum, and training regresses before recovering. Compare to a checkpoint that saved both. | After introducing model state_dict, before the full checkpoint pattern. Emphasize with a before/after loss curve comparison. |
| **"state_dict is complicated or opaque"** | The name sounds abstract. Students may expect a complex custom format. | Print model.state_dict() — it is just a dictionary mapping layer names to tensors. Print optimizer.state_dict() — similar structure. Nothing hidden. | Opening of Explain section — print it immediately, before any theory. Concrete before abstract. |
| **"I need to save the model architecture too"** | In frameworks like Keras, model.save() captures architecture + weights. The student may expect PyTorch to work the same way. | Show that state_dict has no architecture information — just parameter names and values. You must define the model class in code, then load the state_dict into an instance. This is a feature: the architecture lives in version-controlled code, not in a binary file. | Elaborate section — after the core pattern is established. Address as a design choice, not a limitation. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Print state_dict of MNIST model** | Positive | Show that state_dict is just a dictionary of named tensors with familiar layer names (fc1.weight, fc1.bias, etc.) | Demystifies the concept immediately. The student recognizes layer names from their own MNIST model. Concrete before abstract. |
| **Save and reload model weights, verify predictions match** | Positive | Complete round-trip: train -> save state_dict -> create fresh model -> load state_dict -> same predictions. Proves nothing was lost. | The "proof" example. Student sees that two separate model objects produce identical output after loading. Establishes trust in the mechanism. |
| **Resume training with and without optimizer state** | Positive | Show that a proper checkpoint (model + optimizer + epoch) resumes smoothly, while model-only causes a loss spike | The "aha" example that proves why optimizer state matters. Loss curves make the difference viscerally obvious. This is the lesson's central insight. |
| **torch.save(model) then rename the class — load fails** | Negative | Proves that saving the full model object creates a fragile dependency on the exact code | Addresses the #1 misconception directly. The error message is concrete and memorable. Contrasts with state_dict which survives the same change. |
| **Load a GPU-saved model on CPU (map_location)** | Stretch | Shows the map_location parameter for cross-device loading | Practical pattern the student will hit soon (GPU training is next lesson). Keeps it brief — just the pattern, not deep device management. |

### Gap Resolution

No gaps found. All prerequisites are at sufficient depth.

---

## Phase 3: Design

### Narrative Arc

You have trained models that achieved 97% accuracy on MNIST. You have debugged training with torchinfo, gradient checks, and TensorBoard. But every model you have trained exists only in your notebook's memory — close the notebook and the model is gone. Imagine training for an hour on a large dataset, and your process crashes at epoch 47 of 50. You would have to start from scratch. Or imagine wanting to share your best model with a friend, or load it next week for inference. Right now, you cannot. This lesson solves the durability problem: how to save a trained model so it survives beyond the current session, and how to checkpoint during training so you never lose progress.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Print model.state_dict() — see the actual dictionary keys and tensor shapes | state_dict is best understood by looking at it. The student needs to see that it is just familiar layer names mapped to tensors. Demystifies before any explanation. |
| **Symbolic (code)** | Complete save/load code patterns: torch.save(state_dict, path), model.load_state_dict(torch.load(path)) | This is fundamentally an API lesson. The code IS the concept. Each pattern needs to be seen, typed, and understood. |
| **Visual (loss curves)** | Side-by-side loss curves: resume with optimizer state (smooth continuation) vs without (loss spike then recovery) | The "why optimizer state matters" insight is best communicated visually. Numbers in a table would work but the curves make the spike viscerally obvious. |
| **Verbal/Analogy** | "state_dict is a snapshot of all the knobs" — extends the "parameters are knobs" metaphor. Saving state_dict is like photographing the position of every knob. Loading is setting them back. | Connects to the student's existing mental model. "Parameters are knobs" has been consistent since Series 1. |
| **Negative example** | torch.save(model) then rename class file — ModuleNotFoundError | Failure modes are memorable. Seeing the error drives home why state_dict is preferred over saving the full model. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2
  1. state_dict (model and optimizer) as the canonical persistence format
  2. Checkpoint pattern (bundling model state, optimizer state, epoch, and loss into a single save)
- **Previous lesson load:** CONSOLIDATE (debugging-and-visualization)
- **This lesson's load:** BUILD — appropriate. Two new concepts, both are API patterns rather than theoretical shifts. The student has all the prerequisites. Coming after a CONSOLIDATE lesson means they have cognitive space.

### Connections to Prior Concepts

- **"Parameters are knobs" (Series 1)** — state_dict is literally a snapshot of all the knobs. The metaphor extends perfectly: save = photograph the knobs, load = set them back.
- **"Not a black box" (training-loop, 2.1.4)** — state_dict is transparent, not opaque. Print it and you see exactly what is stored. Reinforces the "no magic" ethos.
- **Early stopping "save best model weights" (1.3.7)** — This lesson finally implements what was described conceptually. The student has been waiting for this (explicitly deferred from MNIST project).
- **Adam has internal state (optimizers, 1.3.5 + training-loop, 2.1.4)** — The student knows Adam tracks momentum and adaptive rates. This lesson connects that knowledge to "and those need saving too."
- **model.train() / model.eval() (2.2.2)** — After loading, you must set the right mode. Reinforces the habit from MNIST.

No prior analogies are misleading here. "Parameters are knobs" extends cleanly to "snapshot of knob positions."

### Scope Boundaries

**This lesson IS about:**
- model.state_dict() and optimizer.state_dict()
- torch.save() and torch.load() for state dicts
- The checkpoint dict pattern (model + optimizer + epoch + loss)
- Resuming training from a checkpoint
- map_location for cross-device loading (brief)

**This lesson is NOT about:**
- TorchScript or model compilation (out of scope for Series 2)
- ONNX export or model serving (production deployment)
- Distributed checkpointing or multi-GPU saving
- torch.save(model) as a recommended pattern (shown only as a negative example)
- Hyperparameter saving (mentioned as a nice-to-have in checkpoint dict, not developed)
- Model versioning or experiment management tools (MLflow, W&B)
- Saving/loading on GPU specifically (map_location introduced briefly; full GPU training is next lesson)

**Target depth:**
- state_dict: DEVELOPED (understand what it contains, save and load it, explain why it is preferred)
- Checkpoint pattern: DEVELOPED (implement it in a training loop, resume training)
- map_location: INTRODUCED (know the pattern exists, use it once)
- torch.save(model) pitfalls: INTRODUCED (know why to avoid it)

### Lesson Outline

1. **Context + Constraints** — This lesson is about making models durable. We are NOT covering model export for production, distributed training, or TorchScript. By the end, you can save a trained model, load it later, and resume interrupted training.

2. **Hook** — Type: *before/after + real-world impact*. "You just spent 45 minutes training your MNIST model to 97% accuracy. You close your notebook. The model is gone." Present the scenario: training crash at epoch 47 of 50. No way to recover. The student should feel the problem viscerally before seeing the solution. Brief mention that every production ML workflow depends on this — it is not optional.

3. **Explain: state_dict** — Start by printing `model.state_dict()`. The student sees a dictionary: `{'fc1.weight': tensor(...), 'fc1.bias': tensor(...), ...}`. Call out that these are the same layer names from their MNIST model. Introduce the metaphor: "state_dict is a snapshot of all the knobs." Then show the complete save/load pattern:
   - `torch.save(model.state_dict(), 'model.pth')`
   - `model = MNISTModel(); model.load_state_dict(torch.load('model.pth'))`
   - Verify: same predictions on the same input.

4. **Check 1** — Predict-and-verify: "You train a model, save its state_dict, then change the number of hidden units in the model class and try to load. What happens?" (Answer: size mismatch error — state_dict expects the original architecture.)

5. **Explain: Optimizer state_dict** — "What about Adam's momentum and adaptive rates?" Print `optimizer.state_dict()` — the student sees parameter groups, learning rate, and per-parameter state (exp_avg, exp_avg_sq). Connect: "Remember from the optimizers lesson — Adam tracks running averages. Those live here." Show the resume-with vs resume-without comparison (loss curve visual). This is the lesson's central insight.

6. **Explain: Checkpoint pattern** — Bundle everything into one dictionary:
   ```python
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }
   torch.save(checkpoint, 'checkpoint.pth')
   ```
   Show the resume code: load checkpoint, load_state_dict for both model and optimizer, resume from saved epoch. Present the pattern integrated into a training loop with "save every N epochs" and "save best model."

7. **Check 2** — Transfer question: "A colleague trained a model overnight. It crashed at epoch 80. They restart training from epoch 0. What would you tell them?" (Answer: implement checkpointing — save every N epochs, resume from the last checkpoint.)

8. **Elaborate: torch.save(model) and why not** — Negative example. Save the full model, rename or move the class file, try to load. ModuleNotFoundError. Explain: torch.save(model) uses pickle, which stores a reference to the class definition. state_dict stores just the numbers. This is why state_dict is the PyTorch community standard.

9. **Elaborate: map_location** — Brief. "What if you saved on GPU but want to load on CPU?" One pattern: `torch.load('model.pth', map_location='cpu')`. This is a preview — full GPU training is next lesson.

10. **Practice** — Colab notebook exercises:
    - **Guided:** Save and load a trained MNIST model, verify predictions match.
    - **Supported:** Add checkpointing to an existing training loop (save every 5 epochs + save best model by validation loss).
    - **Supported:** Simulate a training crash — train for 10 epochs, save checkpoint, create a fresh model + optimizer, load checkpoint, resume for 10 more epochs. Verify loss curve is continuous.
    - **Independent:** Implement the full early stopping pattern from 1.3.7 using checkpoints — patience counter, save best model, restore best at the end.

11. **Summarize** — Key takeaways:
    - state_dict = snapshot of all the knobs (just a dictionary of tensors).
    - Always save state_dict, not the model object.
    - Checkpoints bundle model + optimizer + metadata for seamless resume.
    - Optimizer state matters — without it, Adam forgets its momentum.

12. **Next step** — "Your models are now durable. But they are still training on CPU. Next lesson: how to put them on a GPU and make training fast."

### Widget Decision

**No custom interactive widget needed.** This is an API lesson where the code IS the interactive element. The student's interaction is in the Colab notebook: saving, loading, verifying, simulating crashes. A widget would add complexity without pedagogical value. The one visual element (loss curves comparing resume-with vs resume-without optimizer state) can be a static or Recharts chart in the lesson content — it does not need interactivity.

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
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative + 1 stretch)
- [x] At least 3 misconceptions identified with negative examples (4 misconceptions)
- [x] Cognitive load <= 3 new concepts (2 new concepts)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical issues. The lesson is well-structured, follows its planning document closely, and would teach the student effectively. However, four improvement-level findings would make it significantly stronger. Another pass is warranted after addressing them.

### Findings

#### [IMPROVEMENT] — Misconception #4 ("I need to save the architecture too") not explicitly addressed

**Location:** Section "What Is a state_dict?" and "The Complete Save/Load Pattern"
**Issue:** The planning document identifies misconception #4: "I need to save the model architecture too" (from Keras-like frameworks where model.save() captures architecture + weights). The plan says to address it in the Elaborate section "after the core pattern is established." The lesson does explain that "The state_dict does not contain the architecture -- just the parameter values" in the save/load section, and the torch.save(model) section contrasts the two approaches. However, it never explicitly names this as a misconception a student would have, never provides a concrete negative example specific to this misconception (e.g., trying to load a state_dict without defining the model class first and getting an error), and never frames it as a deliberate design choice ("this is a feature, not a limitation"). The information is present but scattered across two sections without the misconception-busting framing the plan calls for.
**Student impact:** A student coming from Keras/TensorFlow (or reading tutorials that use model.save()) might still wonder if the PyTorch approach is missing something. Without explicitly naming and disproving the misconception, the student might store "state_dict is limited" rather than "state_dict is intentionally decoupled."
**Suggested fix:** Add 2-3 sentences in the save/load section (or the torch.save(model) section) that explicitly name this as a common expectation: "If you have used Keras, you might expect model.save() to capture both the architecture and the weights. PyTorch deliberately separates them. The architecture lives in your version-controlled code. The state_dict stores only the learned values. This is a feature: you can refactor your model class, rename layers, reorganize files -- and the state_dict still loads as long as the parameter shapes match."

#### [IMPROVEMENT] — The chart lacks a visual marker at the checkpoint resume point

**Location:** ResumeComparisonChart component (line 68-145)
**Issue:** The chart shows two loss curves diverging at epoch 6, but there is no visual marker (vertical line, annotation, or shaded region) at the epoch 5/6 boundary where the checkpoint was saved and training resumed. The only indication is a small footnote below the chart ("Checkpoint saved at epoch 5. Training resumed at epoch 6.") in 10px text. The divergence itself is visible in the data, but the student must infer the causal relationship between "checkpoint saved" and "curves diverge."
**Student impact:** The student sees two curves that diverge but may not immediately register WHY they diverge at that specific point. The chart's pedagogical power comes from the visual "before: identical, after: dramatically different" contrast. Without an explicit marker, the student has to read the footnote, then look back at the chart, then connect the two -- a cognitive detour that weakens the insight.
**Suggested fix:** Add a Recharts ReferenceLine at epoch 5.5 (between 5 and 6) with a dashed stroke and a label like "Resume point." Alternatively, add a ReferenceArea shading the region from epoch 5 to 6. Either approach makes the "checkpoint boundary" visually obvious without requiring the footnote.

#### [IMPROVEMENT] — No explicit recap of Adam's internal state before depending on it

**Location:** Section "Optimizer State Matters Too" (line 418-498)
**Issue:** The section opens with "Remember from the Optimizers lesson -- Adam tracks momentum buffers and adaptive learning rates for every parameter." This is a single-sentence callback to content from lesson 1.3.5 (Optimizers), which is more than 10 lessons ago. The Reinforcement Rule says: "If a concept was INTRODUCED more than 3 lessons ago and hasn't been used since, assume it is fading." Adam's internal state was DEVELOPED in 1.3.5, reinforced in training-loop (2.1.4) at the "Not a black box" level, and used in mnist-project (2.2.2). So it has been reinforced within the last few lessons. However, the specific details (exp_avg = first moment / momentum, exp_avg_sq = second moment / adaptive rate) were taught in 1.3.5 and not reinforced since. The code output on lines 453-458 shows these tensor names without explaining what they are -- the student sees `exp_avg` and `exp_avg_sq` and must recall from 7+ lessons ago what these represent.
**Student impact:** The student likely remembers "Adam has momentum and adaptive rates" but may not remember the specific names `exp_avg` (first moment estimate / momentum) and `exp_avg_sq` (second moment estimate / adaptive rate). They see these in the output and the explanation says "Adam's running averages -- the momentum and adaptive rate state" but this is a compressed recap that assumes more recall than is likely.
**Suggested fix:** Add one brief parenthetical or aside when first showing the optimizer state_dict output: "exp_avg is Adam's momentum (the running average of gradients) and exp_avg_sq is the adaptive rate (the running average of squared gradients) -- the two quantities that let Adam adjust its step size per-parameter." This is 1-2 sentences and grounds the code output in meaning without re-teaching the concept.

#### [IMPROVEMENT] — "model.eval()" in save/load pattern mentioned without reinforcing WHY

**Location:** Save/load code block, line 326
**Issue:** The save/load code includes `model.eval()  # 4. Set to eval mode for inference` as step 4 of the load pattern. The lesson never explains why this step is necessary. The student knows from mnist-project (2.2.2) that eval() disables dropout and uses running batch norm statistics. But in this context, the student just loaded a model that was trained with dropout and batch norm -- why does eval matter? The code comment says "for inference" but does not connect to the student's existing knowledge that forgetting eval() gives noisy, unreliable results (a lesson from 2.2.2).
**Student impact:** The student may include `model.eval()` mechanically without understanding that it is essential for reproducible inference. This is a reinforcement opportunity -- the student learned this in mnist-project but it is being used without reinforcement in a new context. A student who skipped or forgot that lesson would be completely lost about why step 4 matters.
**Suggested fix:** Add one sentence after the code block (or in an aside): "Step 4 matters because your MNIST model uses batch norm and dropout. In eval mode, dropout is disabled and batch norm uses the saved running statistics instead of batch statistics. Without it, your predictions would be noisy and non-reproducible -- the same lesson from the MNIST project."

#### [POLISH] — Double-hyphens used instead of em dashes in visible prose

**Location:** LessonHeader description (line 155), ConstraintBlock item (line 191), SummaryBlock items (lines 873, 888)
**Issue:** Four visible-to-student strings use `--` (double hyphen) instead of proper em dashes (`&mdash;`). The rest of the lesson correctly uses `&mdash;` throughout prose paragraphs. The Writing Style Rule requires em dashes with no spaces (`word&mdash;word`).
**Student impact:** Minor visual inconsistency. The double-hyphens look less polished than the em dashes used elsewhere.
**Suggested fix:** Replace `--` with `\u2014` or `&mdash;` in the four locations: line 155 description prop, line 191 constraint item, line 873 summary description, line 888 summary headline.

#### [POLISH] — Verify predictions code block assumes `original_model` variable exists

**Location:** verify_predictions.py code block (lines 346-352)
**Issue:** The code block uses `original_model(test_input)` but `original_model` was never defined in any prior code block. The previous blocks defined `model` (the loaded model) and the initial model was also called `model`. A student following along would not have an `original_model` variable.
**Student impact:** A student typing along in a notebook would get a NameError. They would need to figure out that they should have kept a reference to the original model before overwriting the `model` variable. This is a minor confusion point in what is otherwise a clear code progression.
**Suggested fix:** Either (a) rename the loaded model to `loaded_model` so the original `model` is still available, or (b) add a comment in the save block: `# Keep a reference: original_model = model` or (c) add a brief note: "Assuming you kept the original model in a separate variable before creating the fresh model."

#### [POLISH] — `weights_only` deprecation warning not mentioned for torch.load

**Location:** All torch.load calls (lines 325, 532, 689, 766-776)
**Issue:** As of PyTorch 2.6+ (and with warnings since 2.4), `torch.load()` without `weights_only=True` emits a FutureWarning about unsafe deserialization via pickle. The recommended pattern is now `torch.load('model.pth', weights_only=True)` for state_dicts. The student following these code examples in a modern PyTorch installation will see deprecation warnings that the lesson does not prepare them for.
**Student impact:** The student runs the code, sees a FutureWarning about `weights_only`, and wonders if they are doing something wrong. The warning text is verbose and mentions "arbitrary code execution" which could be alarming. Not a showstopper, but a moment of unnecessary confusion.
**Suggested fix:** Add `weights_only=True` to all `torch.load()` calls that load state_dicts (not full checkpoints where non-tensor data like epoch/loss are present). For checkpoint loads, either add a brief note about the warning or use `weights_only=False` explicitly with a comment explaining why. This also connects nicely to the "pickle fragility" discussion -- state_dicts are safe to load with `weights_only=True` precisely because they contain only tensors.

### Review Notes

**What works well:**
- The narrative arc is strong. The "durability problem" hook is visceral and well-motivated -- the student immediately feels the pain of losing a trained model. The progression from "what is state_dict" to "save/load" to "optimizer state matters" to "full checkpoint pattern" is logical and well-paced.
- The connection to prior knowledge is excellent. The "parameters are knobs" metaphor extends naturally to "snapshot of all the knobs." References to the optimizers lesson, early stopping, and the MNIST model are all well-placed and concrete.
- Cognitive load is well-managed. Only 2 new concepts (state_dict and checkpoint pattern), both API patterns rather than theoretical shifts. The lesson stays squarely within its scope boundaries.
- The Recharts chart for the resume comparison is the right visual for the central insight. The loss spike is the kind of thing that needs to be seen, not just described.
- The torch.save(model) negative example with the ComparisonRow is well-constructed. The failure mode (ModuleNotFoundError after renaming a file) is concrete and memorable.
- The practice exercises have good scaffolding progression (guided -> supported -> independent) and the independent exercise (early stopping with checkpoints) ties back to the 1.3.7 concept beautifully.
- The plan was followed faithfully. All 12 outline items are present in order. All planned examples, misconceptions, and modalities are represented.

**Systemic observation:**
The lesson leans heavily on code blocks as its primary modality (which the plan acknowledges -- "code IS the concept"). This is appropriate for a BUILD lesson about API patterns. The one risk is that students who learn better from visual/spatial modalities have fewer entry points. The chart partially addresses this, but the core save/load pattern is purely code + prose. This is not a finding (it matches the plan and the lesson type), but worth noting for the module record.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

All 7 findings from iteration 1 (4 improvement, 3 polish) have been addressed. The lesson now explicitly names the Keras misconception (#4), the chart has a ReferenceLine at the resume point, Adam's exp_avg/exp_avg_sq are explained in-line, model.eval() is reinforced with the batch norm/dropout connection, double-hyphens are replaced with em dashes, the verify_predictions code uses correct variable names (model vs loaded_model), and weights_only is specified on all torch.load calls with appropriate comments.

No critical or improvement findings remain. Two minor polish items noted below.

### Findings

#### [POLISH] — Colab notebook does not exist yet

**Location:** Practice section, line 897 (Colab link)
**Issue:** The lesson links to `notebooks/2-3-1-saving-and-loading.ipynb` on GitHub, but this file does not exist in the repository. The link will 404 until the notebook is created.
**Student impact:** A student clicking "Open in Google Colab" will get a GitHub 404 page. They would need to return to the lesson and complete the exercises without the scaffolded notebook environment.
**Suggested fix:** Create the notebook before shipping the lesson, or add a note that the notebook is coming soon. The lesson text accurately describes all four exercises, so the notebook content is well-specified.

#### [POLISH] — torch.save(model) negative example will trigger FutureWarning

**Location:** save_whole_model.py code block (line 726)
**Issue:** `torch.load('model_full.pth')` in the "don't do this" example does not specify `weights_only`. A student who types this code (even as a negative example) in PyTorch 2.6+ will see a FutureWarning about unsafe deserialization. Since this is explicitly a negative example meant to show why not to do this, the warning is arguably reinforcing, but it could confuse a student who does not yet understand pickle security.
**Student impact:** Very minor. The student sees a verbose warning they were not prepared for. It does not break anything and actually supports the lesson's point.
**Suggested fix:** Optionally add `weights_only=False` with a comment like `# (pickle-based, requires trust)` to suppress the warning and reinforce the pickle fragility point. Or leave as-is since the warning actually helps the lesson's argument.

### Review Notes

**All iteration 1 fixes landed cleanly.** Each of the 7 fixes was implemented as suggested or with equivalent quality. Specifically:

- The Keras misconception (#4) is now explicitly named and reframed as a feature (lines 362-371). This is well-placed in the save/load section and does not interrupt the flow.
- The ReferenceLine at x={5.5} with "Resume" label makes the chart's pedagogical point visually immediate. The before/after contrast is now unmistakable.
- The exp_avg/exp_avg_sq parenthetical (lines 516-523) is thorough without being a re-teach. It gives just enough to ground the code output in meaning.
- The model.eval() reinforcement (lines 342-351) explicitly calls back to the MNIST project and names both batch norm and dropout. The "noisy and non-reproducible" consequence is concrete.
- Variable naming, em dashes, and weights_only were all addressed at every affected location.

**Overall assessment:** This is a well-crafted BUILD lesson. It follows the planning document faithfully, addresses all four misconceptions, uses five modalities, maintains appropriate cognitive load (2 new concepts), and connects every new idea to prior knowledge. The narrative arc from "durability problem" through "full checkpoint in a training loop" is logical and well-paced. The two remaining polish items are genuine minor issues, not blocking. The lesson is ready to ship (pending notebook creation).
