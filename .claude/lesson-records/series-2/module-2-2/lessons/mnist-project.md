# Lesson: mnist-project

**Module:** 2.2 Real Data (Lesson 2 of 3)
**Type:** PROJECT (STRETCH)
**Slug:** `mnist-project`

---

## Phase 1: Orient — Student State

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Dataset / DataLoader pattern | DEVELOPED | datasets-and-dataloaders (2.2.1) | `__getitem__`, `__len__`, batching, shuffling, iteration; "menu/kitchen" analogy |
| torchvision.datasets.MNIST | INTRODUCED | datasets-and-dataloaders (2.2.1) | Loaded it, inspected shape [1, 28, 28], visualized sample grid; did NOT train on it |
| torchvision.transforms pipeline | INTRODUCED | datasets-and-dataloaders (2.2.1) | Compose, ToTensor, Normalize; lazy per-sample in `__getitem__` |
| MNIST image tensor shape [B, 1, 28, 28] | INTRODUCED | datasets-and-dataloaders (2.2.1) | Batch dimension first, 1 grayscale channel, 28x28 |
| shuffle=True as random sampling | DEVELOPED | datasets-and-dataloaders (2.2.1) | Connects to polling analogy from 1.3.4 |
| Complete PyTorch training loop | DEVELOPED | training-loop (2.1.4) | forward -> loss -> backward -> update; "same heartbeat, new instruments" |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1.3) | __init__ + forward(); LEGO bricks; nn.Linear IS w*x + b |
| nn.Sequential | INTRODUCED | nn-module (2.1.3) | Simple layer stacks; cannot express branching |
| nn.ReLU as a module | INTRODUCED | nn-module (2.1.3) | max(0,x) from Series 1, wrapped as module |
| torch.optim (SGD, Adam) | DEVELOPED | training-loop (2.1.4) | Uniform interface; .step() and .zero_grad(); swap with one line |
| nn.MSELoss | DEVELOPED | training-loop (2.1.4) | Wraps MSE formula as callable object; stateless |
| MSE loss function (conceptual) | DEVELOPED | loss-functions (1.1.3) | "Wrongness score"; squared residuals; bowl-shaped landscape |
| Overfitting / generalization | DEVELOPED | overfitting-and-regularization (1.3.7) | Training curves, scissors pattern, model capacity framing |
| Dropout (conceptual) | DEVELOPED | overfitting-and-regularization (1.3.7) | Randomly silence neurons during training; p=0.5 default; implicit ensemble |
| Batch normalization (conceptual) | INTRODUCED | training-dynamics (1.3.6) | Normalize activations between layers; learned gamma/beta; stabilizes gradient flow |
| Weight decay / L2 regularization (conceptual) | DEVELOPED | overfitting-and-regularization (1.3.7) | Penalty on large weights; AdamW named but not implemented |
| Early stopping (conceptual) | DEVELOPED | overfitting-and-regularization (1.3.7) | Monitor val loss, patience, save best weights |
| Train/val/test splits (conceptual) | DEVELOPED | overfitting-and-regularization (1.3.7) | Operationalized through validation curve monitoring |
| Training curves as diagnostic | DEVELOPED | overfitting-and-regularization (1.3.7) | Train vs val loss; three phases; gap = overfitting |
| model.train() / model.eval() | MENTIONED | training-loop (2.1.4) | Named as relevant for dropout/batch norm; never practiced |
| weight_decay parameter | MENTIONED | training-loop (2.1.4) | Named in optimizer constructor; not developed |

### Mental Models and Analogies Already Established

- **"Same heartbeat, new instruments"** — training loop structure is universal; new loss/optimizer/model are "instruments" that slot in
- **"LEGO bricks"** — nn.Module composition; snap layers together
- **"Dataset is a menu, DataLoader is the kitchen"** — data pipeline separation
- **"The scissors pattern"** — train/val divergence = overfitting
- **"Not a black box"** — every PyTorch abstraction maps to math the student already knows
- **"Wrongness score"** — loss measures how wrong the model is
- **"The complete training recipe"** — He init + batch norm + AdamW + dropout + early stopping

### What Was Explicitly NOT Covered

- Cross-entropy loss (mentioned nowhere in Series 1 or 2)
- Softmax function (mentioned nowhere)
- Multi-class classification vs regression (all prior work is regression)
- `nn.Dropout` as a PyTorch module (concept known; API unknown)
- `nn.BatchNorm1d` as a PyTorch module (concept known; API unknown)
- `model.train()` / `model.eval()` in practice (only mentioned)
- Train/test split in code (concept known; `train=True/False` in torchvision not practiced)
- Accuracy as a metric (all prior metrics are continuous loss values)
- Flattening images for fully-connected layers (reshape from [B, 1, 28, 28] to [B, 784])
- `nn.CrossEntropyLoss` (not mentioned anywhere)
- `torch.argmax` for predictions
- `torch.no_grad()` for evaluation (concept from autograd, not practiced in eval context)

### Readiness Assessment

The student is well-prepared for this lesson. They have:
- The complete training loop pattern at DEVELOPED depth
- Dataset/DataLoader at DEVELOPED depth with MNIST already loaded
- nn.Module and layer composition at DEVELOPED depth
- All the conceptual understanding of regularization techniques (dropout, batch norm, weight decay, early stopping)

The primary gaps are classification-specific concepts (cross-entropy, softmax, accuracy) and the PyTorch API for regularization layers (nn.Dropout, nn.BatchNorm1d, model.train()/eval()). These are teachable within this lesson because the student has strong conceptual foundations to build on.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to build, train, and evaluate a complete classification model on MNIST end-to-end in PyTorch, including the new concepts needed for multi-class classification (softmax, cross-entropy loss, accuracy metric).

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source | Status | Reasoning |
|---------|---------------|-------------|--------|--------|-----------|
| PyTorch training loop | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | Student will use the loop; needs to run it independently |
| nn.Module subclass pattern | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Student will build a model from scratch |
| nn.Linear | DEVELOPED | DEVELOPED | nn-module (2.1.3) | OK | Primary building block for the MNIST model |
| nn.ReLU | INTRODUCED | INTRODUCED | nn-module (2.1.3) | OK | Used between layers; INTRODUCED is sufficient for use |
| torch.optim.Adam | DEVELOPED | DEVELOPED | training-loop (2.1.4) | OK | Student will choose and configure an optimizer |
| Dataset / DataLoader | DEVELOPED | DEVELOPED | datasets-and-dataloaders (2.2.1) | OK | Student will use DataLoader to feed MNIST data |
| torchvision.datasets.MNIST | INTRODUCED | INTRODUCED | datasets-and-dataloaders (2.2.1) | OK | Student loaded it before; now trains on it |
| Tensor shape [B, 1, 28, 28] | INTRODUCED | INTRODUCED | datasets-and-dataloaders (2.2.1) | OK | Student needs to flatten for FC layers; INTRODUCED is sufficient with a brief section |
| Dropout (concept) | INTRODUCED | DEVELOPED | overfitting-and-regularization (1.3.7) | OK | Student knows the concept well; needs only the PyTorch API |
| Batch normalization (concept) | INTRODUCED | INTRODUCED | training-dynamics (1.3.6) | OK | Concept is INTRODUCED; PyTorch API is a thin wrapper |
| Overfitting / generalization | DEVELOPED | DEVELOPED | overfitting-and-regularization (1.3.7) | OK | Student already uses training curves diagnostically |
| Train/val/test split (concept) | DEVELOPED | DEVELOPED | overfitting-and-regularization (1.3.7) | OK | Concept known; code pattern is new but simple |
| MSE loss | DEVELOPED | DEVELOPED | loss-functions (1.1.3) + training-loop (2.1.4) | OK | Serves as anchor concept for introducing cross-entropy |
| Cross-entropy loss | INTRODUCED | MISSING | — | MISSING | New concept; must be taught in this lesson |
| Softmax function | INTRODUCED | MISSING | — | MISSING | New concept; needed to understand cross-entropy |
| Accuracy metric | INTRODUCED | MISSING | — | MISSING | New concept; needed for classification evaluation |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Cross-entropy loss | Medium | Dedicated section within this lesson. The student has strong loss function intuition (MSE at DEVELOPED). Cross-entropy is motivated by "MSE doesn't work well for classification" — show the problem, then the solution. Teach at INTRODUCED depth with intuition + formula + code; the student uses it but does not derive it. |
| Softmax function | Small | Brief section within the cross-entropy explanation. Softmax converts raw scores to probabilities. The student knows activation functions (sigmoid at DEVELOPED from 1.2). Softmax is "sigmoid generalized to multiple classes." |
| Accuracy metric | Small | Brief introduction within the evaluation section. Accuracy = (correct predictions / total). Intuitive concept; the student just hasn't used it because all prior work was regression. Teach with `torch.argmax` + comparison. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Cross-entropy and MSE are interchangeable — just use MSE for classification" | All prior loss functions have been MSE; the student has no reason to suspect MSE fails for classification | Show MSE loss on a confident-but-wrong classification (output [0.9, 0.1] for class 1 vs [0.1, 0.9]): MSE gives a moderate penalty, cross-entropy gives a huge penalty. MSE treats all errors equally; cross-entropy punishes confident wrong answers severely, which is what classification needs. | Core concept section, before introducing cross-entropy formula |
| "The model outputs probabilities directly" | Sigmoid outputs values in [0,1] which look like probabilities; the student may assume nn.Linear outputs are probabilities too | Show raw model outputs (logits) for a trained model: values like [-2.3, 0.5, 4.1, ...] — these are NOT probabilities (negative, don't sum to 1). Softmax is needed to convert them. | When introducing softmax, before cross-entropy |
| "Higher accuracy = better model (always)" | Accuracy is intuitive and satisfying; the student may ignore loss in favor of accuracy | A model with 98% accuracy on training data but 85% on test data is overfitting. A model at 95% train and 94% test is better. Connect to the scissors pattern — accuracy alone does not detect overfitting. Must track both training and validation metrics. | Evaluation section |
| "I need to get 99%+ on MNIST or my model is bad" | MNIST is often described as "easy" in the ML community; first-timers expect near-perfect results | A simple fully-connected network (no CNNs) typically gets 97-98%. Getting 99%+ requires convolutional architectures (Series 3). The student's model is good for its architecture — CNNs are the tool for that last 1-2%, not bigger FC networks. | After first training run, when reviewing results |
| "nn.CrossEntropyLoss needs softmax first" | Cross-entropy is defined on probabilities, so the student may add softmax before the loss | PyTorch's nn.CrossEntropyLoss applies log-softmax internally for numerical stability. Adding softmax manually double-applies it and produces wrong gradients. Show the PyTorch docs note + incorrect vs correct code. | Immediately after introducing nn.CrossEntropyLoss |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| MSE vs cross-entropy on misclassification | Negative | Show WHY MSE fails for classification — it penalizes confident-wrong and slightly-wrong equally | Motivates the new loss function by demonstrating the problem before the solution; uses the student's existing MSE knowledge as the anchor |
| Simple 2-layer model on MNIST (~97% accuracy) | Positive | First end-to-end success — load, build, train, evaluate, see predictions | The simplest model that gives a satisfying result; proves the full pipeline works; emotional payoff |
| Improved model with dropout + batch norm (~98%) | Positive | Demonstrates regularization improvement; bridges Series 1 theory to practice | Shows that the regularization concepts from 1.3 actually improve real models; validates the "complete training recipe" |
| Visualize correct and incorrect predictions | Positive | Makes the model's behavior concrete — see the actual images it gets wrong | Builds intuition for model failure modes (ambiguous digits, unusual writing styles); grounds abstract accuracy numbers in visual reality |
| Overfit model (no regularization, large model) | Negative | Shows overfitting in practice — training accuracy 99.5%, test accuracy 97% with diverging curves | Connects to the scissors pattern from 1.3.7; the student sees in real training what they saw conceptually in the RegularizationExplorer |

---

## Phase 3: Design

### Narrative Arc

Everything the student has built has been on toy data: synthetic lines, parabolas, tensors with a handful of points. The training loop works, the optimizer works, nn.Module works — but it has all been rehearsal. This lesson is the real performance. We load 60,000 handwritten digits, build a model that learns to read them, train it, and watch it get 97%+ correct. Along the way, we pick up the tools needed for classification (cross-entropy replaces MSE, accuracy replaces "look at the loss curve"), and we put the regularization techniques from Series 1 to work for real. By the end, the student has trained a model that actually does something — it reads handwriting. This is the moment where "I understand deep learning" becomes "I've done deep learning."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Concrete example (worked through) | Softmax on a real logit vector: raw outputs [-1.2, 0.5, 3.8, ...] -> probabilities [0.02, 0.11, 0.80, ...] with actual numbers | Softmax is abstract as a formula; seeing real numbers transform into probabilities makes the "convert to probabilities" intuition concrete |
| Visual | Grid of MNIST predictions: image + predicted label + confidence + green/red border for correct/incorrect | Makes model accuracy tangible — not just "97%" but "this is what the 3% wrong looks like" |
| Symbolic | Cross-entropy formula: L = -log(p_correct) with side-by-side table showing loss values for confident-right (0.01), unsure (0.69), confident-wrong (4.61) | The formula alone is opaque; the table of concrete values builds the "punishes confident-wrong severely" intuition |
| Code (the primary modality) | Complete end-to-end pipeline in annotated code blocks: data loading, model definition, training loop, evaluation loop, prediction visualization | This is a PROJECT lesson — code IS the core deliverable; every other modality supports understanding the code |
| Intuitive/Analogy | Cross-entropy as "confidence penalty" — if you are confident and wrong, the penalty is enormous; if you are uncertain, the penalty is mild | Connects to the "wrongness score" framing from MSE; extends it with the key insight that confidence matters for classification |

### Cognitive Load Assessment

- **New concepts in this lesson:** 3 (cross-entropy loss, softmax, accuracy metric)
- **Previous lesson load:** BUILD (datasets-and-dataloaders — familiar batching concept, new API)
- **This lesson's load:** STRETCH — 3 new concepts plus integration of everything built so far
- **Assessment:** Appropriate. BUILD -> STRETCH follows the trajectory. The 3 new concepts are tightly related (softmax feeds cross-entropy, accuracy is the human-readable version of cross-entropy's judgment). The integration work (putting all pieces together) is cognitive effort but not conceptual novelty — every piece has been individually verified. The emotional payoff ("it works on real data") provides motivation energy to push through the stretch.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| MSE loss ("wrongness score") from 1.1 | Cross-entropy is another wrongness score, but designed for classification; MSE measures distance, cross-entropy measures confidence on the wrong answer |
| Sigmoid from 1.2 | Softmax is "sigmoid generalized to multiple classes" — sigmoid squashes one value to [0,1], softmax squashes N values to probabilities summing to 1 |
| Dropout concept from 1.3.7 | Now implemented as `nn.Dropout(p=0.5)` in an nn.Module — same "randomly silence neurons" idea, one line of code |
| Batch normalization concept from 1.3.6 | Now implemented as `nn.BatchNorm1d(features)` — same "normalize activations between layers" idea |
| model.train()/model.eval() from 2.1.4 | Now practiced for real — dropout and batch norm behave differently in training vs eval mode |
| Training curves / scissors pattern from 1.3.7 | Now observed in real training — compare train vs test accuracy/loss to detect overfitting |
| "Complete training recipe" from 1.3.7 | This lesson implements the recipe: He init + batch norm + AdamW + dropout + early stopping |
| "Same heartbeat, new instruments" from 2.1.4 | The training loop body does not change; we swap MSE for cross-entropy, add accuracy logging, and the heartbeat is the same |

### Analogies That Might Mislead

- **"Wrongness score" (MSE)** could mislead the student into thinking MSE works for classification. Must explicitly show WHY it fails before introducing cross-entropy.
- **"Loss landscape = bowl"** from 1.1 applies to MSE on linear regression. Cross-entropy on a neural network has a more complex landscape. We do not need to address this directly but should not extend the bowl analogy to cross-entropy.

### Scope Boundaries

**This lesson IS about:**
- End-to-end MNIST classification with a fully-connected network
- Cross-entropy loss at INTRODUCED depth (intuition + formula + code; no derivation from information theory)
- Softmax at INTRODUCED depth (what it does + code; no gradient derivation)
- Accuracy as a metric at INTRODUCED depth
- nn.Dropout and nn.BatchNorm1d at INTRODUCED depth (concept-to-code bridge; the "why" is already known)
- model.train()/model.eval() at DEVELOPED depth (practiced in the training + evaluation loop)
- Train/test split in torchvision at INTRODUCED depth (train=True/False argument)
- Visualizing predictions (correct and incorrect)

**This lesson is NOT about:**
- Convolutional networks (MNIST is solved with FC here; CNNs come in Series 3)
- Advanced data augmentation
- Learning rate schedulers
- Hyperparameter tuning (we pick reasonable defaults, not search for optimal)
- Information-theoretic derivation of cross-entropy
- Multi-label classification or non-exclusive classes
- Saving/loading models (next module)
- GPU training (next module)
- Custom loss functions
- Achieving 99%+ accuracy (we celebrate 97-98% as appropriate for FC architecture)

### Lesson Outline

**1. Context + Constraints** (~2 paragraphs)
What: End-to-end MNIST project — everything comes together. Scope: FC network only (no CNNs), celebrate 97-98% accuracy. This is a PROJECT lesson: the Colab notebook is the primary deliverable.

**2. Hook — Before/After Demo**
Show a trained model making predictions: a grid of handwritten digits with the model's predicted labels and confidence scores. "By the end of this lesson, you will have trained the model that produced this." The student already has MNIST images from lesson 1 but has never trained on them. The gap between "I loaded this data" and "a model can read these" is the motivating tension.

**3. The Classification Problem** (~3-4 paragraphs)
Reframe from regression to classification. All prior work: continuous output, MSE loss. Now: 10 discrete categories. Two things change: (a) how the model makes a prediction (pick a class), (b) how we measure wrongness (can't use MSE — motivate with the negative example).

**4. Softmax: From Scores to Probabilities** (~3 paragraphs + worked example)
- Problem: raw model outputs (logits) are arbitrary numbers, not probabilities
- Connection: "remember sigmoid squashes to [0,1]? Softmax does the same thing for multiple classes"
- Worked example with real numbers: logits [-1.2, 0.5, 3.8] -> exp -> normalize -> [0.02, 0.11, 0.80, ...]
- Key property: outputs are positive and sum to 1
- Note: we don't apply softmax in the model; PyTorch's loss function handles it

**5. Cross-Entropy Loss: Confidence Penalty** (~4-5 paragraphs + table)
- Problem: MSE treats all errors equally. Classification needs to punish confident-wrong answers severely.
- The negative example: MSE on classification (moderate penalty for confident-wrong vs cross-entropy's huge penalty)
- Formula: L = -log(p_correct) with concrete value table
- Intuition: -log(0.95) = 0.05 (confident right, small loss), -log(0.5) = 0.69 (unsure, moderate), -log(0.01) = 4.61 (confident wrong, huge)
- `nn.CrossEntropyLoss`: takes raw logits (NOT softmax output); applies log-softmax internally
- WarningBlock: "Do NOT add softmax before nn.CrossEntropyLoss" — this is the most common beginner mistake
- Connection: "same heartbeat, new instrument" — swap nn.MSELoss for nn.CrossEntropyLoss, loop body unchanged

**6. Check 1 — Predict-and-Verify**
Given logits [2.0, -1.0, 0.5] and true class 0: (a) which class does the model predict? (b) is it correct? (c) is the cross-entropy loss high or low? Student reasons before seeing the answer.

**7. Building the Model** (~4-5 paragraphs + code)
- Flatten: [B, 1, 28, 28] -> [B, 784] using `x.view(-1, 784)` or `nn.Flatten()`
- Architecture: 784 -> 256 -> 128 -> 10 (three linear layers with ReLU between)
- First version: simple (no regularization) — get it working before improving
- Code: complete nn.Module subclass with forward()
- Count parameters: ~235K (calculate and state)

**8. The Training Loop — Putting It All Together** (~3 paragraphs + code)
- Emphasize: the loop body is identical to training-loop (2.1.4). Only the model, loss, and data changed.
- Complete training code: DataLoader, model, nn.CrossEntropyLoss, Adam optimizer, epoch loop
- Accuracy tracking within the training loop: `torch.argmax(output, dim=1)` compared to labels
- Print loss + accuracy per epoch

**9. Evaluation — Test Set Performance** (~3 paragraphs + code)
- Use `train=False` for test set (already seen this argument in datasets-and-dataloaders)
- `torch.no_grad()` for evaluation (concept from autograd 2.1.2, now practiced)
- Compute test accuracy + test loss
- First result: expect ~97% accuracy

**10. Seeing What the Model Learned** (~2 paragraphs + visual)
- Visualize correct predictions: image + label + confidence (green border)
- Visualize incorrect predictions: image + predicted + actual + confidence (red border)
- Observation: incorrect ones are often genuinely ambiguous (sloppy handwriting, unusual styles)
- This makes "97% accuracy" concrete — it is not failing on easy cases

**11. Check 2 — Transfer Question**
"Your model gets 99.2% training accuracy but 96.8% test accuracy. What is happening, and what would you try first?" Expected: overfitting (scissors pattern); try adding dropout or using AdamW with weight_decay. Connects to 1.3.7.

**12. Improving the Model — Regularization in Practice** (~4-5 paragraphs + code)
- Add `nn.Dropout(p=0.3)` after ReLU layers
- Add `nn.BatchNorm1d(features)` before ReLU layers
- model.train() / model.eval() — NOW it matters (dropout + batch norm behave differently)
- Explain: model.train() enables dropout + uses batch stats for BN; model.eval() disables dropout + uses running stats for BN
- weight_decay in Adam constructor
- Retrain: expect ~98% accuracy
- Compare training curves: with vs without regularization (the scissors close)

**13. Summarize — Key Takeaways**
- Classification uses cross-entropy (not MSE) because confidence matters
- Softmax converts logits to probabilities; nn.CrossEntropyLoss handles it internally
- Accuracy = human-readable metric; loss = what the model optimizes
- model.train()/model.eval() matters when using dropout or batch norm
- The training loop heartbeat did not change — only the instruments

**14. Next Step**
"You have a working model. But what happens when training goes wrong? Next lesson: debugging tools for when shapes don't match, losses explode, or the model refuses to learn."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (3 MISSING concepts have resolution plans)
- [x] No multi-concept jumps in exercises
- [x] All gaps have explicit resolution plans

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept (concrete example, visual, symbolic, code, intuitive — 5 total)
- [x] At least 2 positive examples + 1 negative example (2 positive + 2 negative + 1 visual)
- [x] At least 3 misconceptions identified with negative examples (5 total)
- [x] Cognitive load = 3 new concepts (at limit)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No critical findings that would leave the student lost. One critical finding regarding a missing planned negative example that significantly weakens the lesson's teaching of the "accuracy is not everything" misconception. Four improvement findings that would make the lesson notably stronger. The lesson's overall structure, narrative arc, motivation, and code quality are strong — the issues are about missing elements and a couple of areas that need more scaffolding.

### Findings

#### CRITICAL — Missing training curve comparison (planned overfit negative example absent)

**Location:** Section 12 (Improving the Model — Regularization in Practice) and general lesson structure
**Issue:** The planning document explicitly listed "Overfit model (no regularization, large model)" as a negative example with the purpose: "Shows overfitting in practice — training accuracy 99.5%, test accuracy 97% with diverging curves. Connects to the scissors pattern from 1.3.7." It also called for "Compare training curves: with vs without regularization (the scissors close)" in the regularization section. Neither is present in the built lesson. The student is told "the gap between train and test accuracy is smaller" and "the scissors pattern is closing" without ever seeing the evidence. The lesson's central promise is connecting theory to practice — telling the student "regularization works" without showing the visual proof undermines that promise.
**Student impact:** The student has the scissors pattern at DEVELOPED depth from 1.3.7 and would benefit enormously from seeing it manifest in their own training. Without the visual comparison, the lesson makes an unsubstantiated claim ("regularization works, the scissors close") that the student must take on faith rather than observe. This also leaves the "higher accuracy = better model" misconception only weakly addressed (an aside, rather than concrete evidence).
**Suggested fix:** Add a code block (or at minimum, a concrete expected-output table) showing epoch-by-epoch train/test accuracy for both models side by side. For example, a simple table or printed output showing the simple model's gap (99%/97%) vs the improved model's gap (98.5%/98%). Even without a matplotlib chart, the printed numbers would make the claim concrete and verifiable. Ideally, the Colab notebook would include a training curve plot comparing both models.

#### IMPROVEMENT — ComparisonRow for MSE vs cross-entropy uses implicit 2-class framing on a 10-class problem

**Location:** Section 5 (Cross-Entropy Loss: Confidence Penalty), the ComparisonRow component
**Issue:** The comparison says "Model says: 90% class 3, 10% class 0" — but the lesson has just established this is a 10-class problem (MNIST, digits 0-9). If probabilities must sum to 1 (as the softmax section just taught), where did the other 80% go? The example implicitly uses a 2-class simplification without stating this. The student just learned that softmax produces probabilities summing to 1 across all classes, and this example appears to violate that property.
**Student impact:** A careful student would notice the inconsistency and wonder if they misunderstood softmax. A less careful student would internalize an incorrect example. Either way, it creates unnecessary confusion at a critical teaching moment.
**Suggested fix:** Either (a) state explicitly "For simplicity, imagine a 2-class problem..." before the comparison, or (b) use a 10-class example: "Model assigns 90% to class 3, 1% to each of the other 9 classes (including the correct class 0)." Option (b) is better because it stays grounded in the MNIST context.

#### IMPROVEMENT — Softmax worked example lacks step-by-step computation

**Location:** Section 4 (Softmax: From Scores to Probabilities), the code block `softmax_example.py`
**Issue:** The formula $e^{z_i} / \sum_j e^{z_j}$ is shown, then the code block shows `F.softmax(logits, dim=0)` and the probabilities appear as comments. The student sees the input and output but not the intermediate computation. For a concept being INTRODUCED, the student should see at least one element computed manually: "For class 7: $e^{3.8} = 44.7$, total sum of exponentials = 55.6, so $44.7 / 55.6 = 0.80$." This would connect the formula to the numbers and make the "exponential makes everything positive, dividing by sum makes them add to 1" explanation concrete rather than stated.
**Student impact:** Without the intermediate computation, the student takes the output on faith. They understand the formula abstractly but cannot verify it concretely. This weakens the "concrete before abstract" principle — the formula IS abstract, and the code output IS concrete, but the bridge between them (working through the math) is missing.
**Suggested fix:** Add a brief worked-through computation for one element (class 7) between the formula and the code block. Two or three lines showing $e^{3.8}$, the sum, and the division. This does not need to cover all 10 classes — one is sufficient to build the bridge.

#### IMPROVEMENT — nn.Flatten() used without introduction

**Location:** Section 7 (Building the Model), `mnist_model_v1.py`
**Issue:** The model uses `self.flatten = nn.Flatten()` and the comment says `# [B, 1, 28, 28] -> [B, 784]`. The student knows `view()` and `reshape()` from the tensors lesson (2.1.1) at INTRODUCED depth, but `nn.Flatten()` as an nn.Module has never been shown. The planning doc acknowledges both options ("x.view(-1, 784) or nn.Flatten()") but the lesson uses nn.Flatten without explaining what it is or why it is preferred over view/reshape.
**Student impact:** Minor confusion. The student understands flattening conceptually (28x28=784) but may wonder what nn.Flatten is and why it is used instead of the reshape they already know. This is not a blocking issue because the comment makes the effect clear, but it is an unresolved gap.
**Suggested fix:** Add one sentence when introducing the model: "nn.Flatten() is a module version of reshape — it flattens every dimension after the batch dimension into a single vector. Using it as a module means it slots into the forward() pipeline cleanly, instead of calling x.view(-1, 784) separately."

#### IMPROVEMENT — Regularization section is dense; too many API introductions in one block

**Location:** Section 12 (Improving the Model — Regularization in Practice)
**Issue:** This section introduces nn.BatchNorm1d API, nn.Dropout API, the Linear->BatchNorm->ReLU->Dropout ordering convention, weight_decay as a concrete parameter, model.train()/model.eval() as mandatory (not optional), AND shows a complete new training+evaluation loop — all in one section. While the student has conceptual foundations for each of these, the API density is high. Six code-level new things in one section approaches the cognitive load boundary.
**Student impact:** The student may feel overwhelmed by the volume of new API details, even though each individual piece is conceptually familiar. The section reads more like a reference dump than a guided build. Compared to the rest of the lesson (which carefully introduces one idea at a time), this section breaks the pacing.
**Suggested fix:** Consider splitting into two sub-sections: (1) "The Improved Model" — show the nn.Module subclass with BatchNorm + Dropout, explain the layer ordering pattern, explain why model.train()/eval() now matters. (2) "Training the Improved Model" — show the training loop with weight_decay, show the epoch-by-epoch comparison (the missing training curve comparison from the critical finding). This gives the student a breathing point between understanding the model architecture and understanding the training changes.

#### POLISH — model.eval()/model.train() aside is misplaced

**Location:** Section 9 (Evaluation — Test Set Performance), the ConceptBlock aside
**Issue:** The aside explains model.train() vs model.eval() in terms of dropout and batch norm behavior. But the student's current model (the simple model being evaluated in this section) has neither dropout nor batch norm. The aside is teaching about something that does not yet apply. It would be more effective in section 12, where dropout and batch norm are actually added.
**Student impact:** Mild confusion — "why is this aside telling me about dropout when my model doesn't have dropout?" The aside is accurate but premature.
**Suggested fix:** Move this ConceptBlock aside to section 12 (regularization section) where it directly applies. In section 9, keep the simpler inline explanation already present ("For now this has no effect...").

#### POLISH — Normalize magic numbers unexplained

**Location:** Section 8 (The Training Loop), `train_mnist.py`, line `transforms.Normalize((0.1307,), (0.3081,))`
**Issue:** The normalization values 0.1307 and 0.3081 are MNIST-specific (the dataset's mean and standard deviation). The student saw transforms.Normalize in 2.2.1 but these specific numbers appear without explanation. A brief note would prevent the student from wondering "where did these come from?"
**Student impact:** Minor curiosity/confusion. The student might try to look up these numbers or worry they need to compute them for other datasets.
**Suggested fix:** Add an inline comment: `# MNIST mean and std — precomputed from the training set` or a brief note in the surrounding text.

#### POLISH — "Let us" / "Let us see" phrasing is slightly stiff

**Location:** Multiple sections (e.g., "Let us see why MSE fails", "Let us make it concrete")
**Issue:** The "Let us" construction reads formally compared to the rest of the lesson's conversational tone. The contractions used elsewhere ("you will have", "it has") create a warmer voice. "Let us" stands out as unusually formal.
**Student impact:** Negligible — stylistic only. Does not affect comprehension.
**Suggested fix:** Consider "Let's" for a more conversational tone that matches the rest of the lesson, or rephrase (e.g., "To see why MSE fails..."). Very low priority.

### Review Notes

**What works well:**
- The narrative arc is strong. The progression from "classification is different" through softmax/cross-entropy to a working model to an improved model feels natural and well-motivated.
- The "same heartbeat, new instruments" reinforcement is excellent. The student sees their training loop pattern working on real data without structural changes — this builds deep confidence.
- The emotional payoff ("I have done deep learning") is well-earned. The lesson delivers on its promise.
- Cross-entropy is taught particularly well: the motivation (MSE fails), the intuition (confidence penalty), the formula, the concrete values, and the code are all aligned and mutually reinforcing.
- The scope boundaries are well-managed. The lesson stays focused and does not drift into CNN territory or hyperparameter tuning.
- Check questions are well-designed: Check 1 tests softmax/cross-entropy understanding with a concrete exercise, Check 2 bridges to the regularization section via the scissors pattern.

**Systemic observation:**
The lesson's primary weakness is in section 12 (regularization). It is the densest section and also has the most significant deviations from the plan (missing training curve comparison, missing overfit negative example). This suggests the section was either rushed during building or that the builder ran out of energy/context. The fix is focused: split the section, add the training curve comparison, and the lesson will be significantly stronger.

**Verdict rationale:**
NEEDS REVISION, not MAJOR REVISION, because the student can follow the lesson and learn the core concepts effectively. The critical finding (missing training curve comparison) weakens one section but does not leave the student lost or confused — it leaves them with an unsubstantiated claim where they should have concrete evidence. The improvement findings would each make the lesson notably better but none blocks the student's learning path.

---

## Review — 2026-02-09 (Iteration 2/3)

### Iteration 1 Fix Verification

All 8 findings from iteration 1 have been addressed:

| Finding | Severity | Status | Notes |
|---------|----------|--------|-------|
| Missing training curve comparison | CRITICAL | FIXED | Side-by-side epoch tables + matplotlib code added in new section 12b |
| ComparisonRow 2-class framing | IMPROVEMENT | FIXED | Now uses 10-class example: "80% class 3, 2% class 0, ~2% each elsewhere" with setup paragraph |
| Softmax lacks step-by-step | IMPROVEMENT | FIXED | Three-step worked computation added for class 7 (e^3.8 = 44.70, sum = 54.58, divide = 0.82) |
| nn.Flatten() unintroduced | IMPROVEMENT | FIXED | Explanatory sentence added: "nn.Flatten() does the same thing as a module — it flattens every dimension after the batch dimension" |
| Regularization section too dense | IMPROVEMENT | FIXED | Split into 12a (architecture) and 12b (training + comparison) with separate SectionHeaders |
| model.eval()/train() aside misplaced | POLISH | FIXED | Moved to section 12a where dropout/batch norm are introduced; section 9 aside now covers torch.no_grad() |
| Normalize magic numbers | POLISH | FIXED | Inline comment added: "# MNIST mean and std, precomputed from training set" |
| "Let us" phrasing | POLISH | FIXED | Changed to "Let's" throughout |

### Summary
- Critical: 1
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

One critical math error in a check answer that would actively mislead a student who computes the values. One improvement around an unexplained phenomenon in the training curves. One polish item. The lesson is otherwise strong — the iteration 1 fixes landed cleanly and the pedagogical quality is high.

### Findings

#### CRITICAL — Check 1 answer has incorrect softmax probability

**Location:** Section 6 (Check Your Understanding), question 3 answer (lines 481-488)
**Issue:** The answer states "Softmax assigns class 0 the highest probability (about 0.71 for these logits), so -log(0.71) approximately 0.34." The actual softmax computation for logits [2.0, -1.0, 0.5] gives: e^2.0 = 7.389, e^-1.0 = 0.368, e^0.5 = 1.649, sum = 9.406, p(class 0) = 7.389/9.406 = 0.786. The correct answer is approximately 0.79 (not 0.71), and -log(0.79) is approximately 0.24 (not 0.34). The qualitative conclusion ("small loss, model is correct and reasonably confident") is right, but the specific numbers are wrong.
**Student impact:** This is a predict-and-verify exercise. A careful student who actually computes the softmax will get 0.79, see the answer says 0.71, and conclude they made a mistake. This undermines the exercise's purpose (building confidence through correct self-verification) and could shake the student's trust in their own computation right after learning softmax. A less careful student takes the wrong number as ground truth.
**Suggested fix:** Change "about 0.71" to "about 0.79" and "-log(0.71) approximately 0.34" to "-log(0.79) approximately 0.24". Keep the qualitative framing unchanged.

#### IMPROVEMENT — Training curves show test > train for improved model without explanation

**Location:** Section 12b (Training the Improved Model), the improved model comparison table (lines 1039-1051)
**Issue:** The improved model shows train=97.5% and test=98.1% at epoch 5 — test accuracy is HIGHER than training accuracy. This is a real phenomenon caused by dropout (which suppresses neurons during training but not during evaluation), and the numbers are realistic. However, the lesson has just spent multiple sections establishing that overfitting means "train > test," and the student has the scissors pattern at DEVELOPED depth. Seeing test > train without explanation could confuse the student: "How can the model do better on data it has never seen? Did I misunderstand overfitting?"
**Student impact:** A thoughtful student would notice the inversion and wonder if the numbers are wrong or if they misunderstand something. The lesson does not address this, leaving the student to either ignore it or form an incorrect mental model (e.g., "regularization makes test accuracy always higher than training accuracy").
**Suggested fix:** Add one sentence after the comparison tables acknowledging this: "You may notice the improved model's test accuracy slightly exceeds its training accuracy. This is normal with dropout — during training, 30% of neurons are silenced, making training harder. During evaluation, all neurons fire, so the model performs at full strength. The important signal is the small gap, not which number is higher." This takes 2 sentences and prevents a real source of confusion.

#### POLISH — Redundant aside content between section 12a and warning block in 12b

**Location:** Section 12a aside (ConceptBlock "model.eval() vs model.train()", line 950) and section 12b aside (WarningBlock "train() vs eval()", line 1115)
**Issue:** Both asides cover the same topic — model.train() vs model.eval(). The 12a aside is a ConceptBlock explaining the behavior. The 12b aside is a WarningBlock about forgetting model.eval(). The content overlaps substantially: both mention dropout behavior and batch norm behavior. The inline text of section 12a (lines 938-946) also covers this same ground. Three instances of the same information in close proximity is slightly redundant.
**Student impact:** Negligible. Repetition of an important point is not harmful. But the second aside could be replaced with something that adds new information (e.g., a tip about a common debugging pattern for checking if model is in the right mode).
**Suggested fix:** Low priority. Either (a) remove the 12b WarningBlock since the same point was made in 12a, or (b) change the 12b WarningBlock to focus on a different angle (e.g., "A common debugging trick: if your test accuracy is noisy or suspiciously close to training accuracy, check that you called model.eval() before the test loop"). The current state is not wrong, just slightly redundant.

### Review Notes

**What works well:**
- All iteration 1 fixes landed cleanly. The lesson is substantially stronger than the previous version.
- The regularization section split (12a/12b) dramatically improved pacing. The student gets a clear breath between understanding the architecture and seeing the training results.
- The training curves comparison tables are exactly what was needed. The scissors pattern claim is now backed by concrete numbers.
- The softmax step-by-step computation bridges formula to code effectively.
- The ComparisonRow now correctly uses 10-class framing consistent with the softmax section.
- The relocated model.eval()/train() aside makes section 12a much more cohesive.
- Cross-entropy teaching remains the lesson's strongest section: motivation -> intuition -> formula -> concrete values -> code -> warning.

**The one critical issue:**
The math error in Check 1 is the only blocking finding. It is a straightforward fix (two numbers), but it must be fixed before the lesson is usable because predict-and-verify exercises are worthless if the answer is wrong.

**Verdict rationale:**
NEEDS REVISION because of the math error in Check 1. Without that error, this would be a PASS with one improvement suggestion. The improvement finding (test > train unexplained) is a real source of potential confusion but does not rise to critical because the qualitative message ("regularization works, gap is small") is correct even if the inversion is confusing. Fix the math error, optionally add the dropout explanation, and this lesson is ready to ship.
