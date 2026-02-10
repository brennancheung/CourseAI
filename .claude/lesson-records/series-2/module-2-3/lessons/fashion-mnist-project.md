# Lesson Plan: Fashion-MNIST Project

**Module:** 2.3 (Practical Patterns)
**Position:** Lesson 3 of 3 (FINAL lesson in Module 2.3 and Series 2)
**Slug:** `fashion-mnist-project`
**Load Type:** CONSOLIDATE / PROJECT

---

## Phase 1: Orient --- Student State

The student has completed all of Series 1 (17 lessons: the learning problem, neural networks, backpropagation, optimization, regularization) and all of Series 2 up to this point (9 lessons: tensors, autograd, nn.Module, training loop, datasets/dataloaders, MNIST project, debugging/visualization, saving/loading, GPU training). This is the graduation exercise. There should be zero new concepts. Every skill the student needs has been taught and practiced. The question is whether they can combine those skills independently with minimal scaffolding.

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Complete PyTorch training loop | DEVELOPED | training-loop (2.1.4) | forward-loss-backward-update heartbeat. Practiced on regression (2.1.4), classification (2.2.2), GPU (2.3.2). |
| nn.Module subclass pattern | DEVELOPED | nn-module (2.1.3) | __init__ + forward(). Used in MNIST project (2.2.2) with regularization layers. |
| nn.Sequential | INTRODUCED | nn-module (2.1.3) | Simple stacks. Used as alternative to subclass in MNIST. |
| nn.Linear, nn.ReLU | DEVELOPED | nn-module (2.1.3) | Core building blocks. Used in every model since 2.1.3. |
| nn.Dropout(p) as PyTorch module | INTRODUCED | mnist-project (2.2.2) | Bridges concept (DEVELOPED 1.3.7) to API. Active in train(), disabled in eval(). |
| nn.BatchNorm1d(features) as module | INTRODUCED | mnist-project (2.2.2) | Bridges concept (INTRODUCED 1.3.6) to API. Batch stats in train(), running stats in eval(). |
| Linear -> BatchNorm -> ReLU -> Dropout ordering | INTRODUCED | mnist-project (2.2.2) | Standard pattern for regularized FC hidden layers. Output layer has no activation/dropout/BN. |
| Cross-entropy loss (nn.CrossEntropyLoss) | INTRODUCED | mnist-project (2.2.2) | Takes raw logits + integer labels. Applies log-softmax internally. "Confidence penalty" intuition. |
| Softmax function | INTRODUCED | mnist-project (2.2.2) | Converts logits to probabilities. "Sigmoid generalized to multiple classes." |
| Accuracy metric (torch.argmax) | INTRODUCED | mnist-project (2.2.2) | argmax(outputs, dim=1) = predicted class. Human-readable evaluation metric vs loss. |
| model.train() / model.eval() | DEVELOPED | mnist-project (2.2.2) | Practiced for real: dropout/BN behavior changes. Critical for evaluation. |
| Train/test evaluation loop | DEVELOPED | mnist-project (2.2.2) | model.eval() + torch.no_grad() + iterate test_loader + compute accuracy. |
| weight_decay in optimizer | INTRODUCED | mnist-project (2.2.2) | L2 regularization via optimizer constructor. |
| Dataset / DataLoader pattern | DEVELOPED | datasets-and-dataloaders (2.2.1) | Dataset: __getitem__ + __len__. DataLoader: batching, shuffling, iteration. |
| torchvision.datasets and transforms | INTRODUCED | datasets-and-dataloaders (2.2.1) | MNIST loaded with datasets.MNIST, transforms.Compose/ToTensor/Normalize. |
| torchinfo.summary() | DEVELOPED | debugging-and-visualization (2.2.3) | "X-ray" -- see inside model without running data. First step of debugging checklist. |
| Gradient magnitude checking | DEVELOPED | debugging-and-visualization (2.2.3) | Per-layer gradient norms. Healthy: balanced. Unhealthy: orders-of-magnitude mismatch. |
| TensorBoard SummaryWriter | INTRODUCED | debugging-and-visualization (2.2.3) | Scalar logging, run comparison. "Flight recorder." |
| Systematic debugging checklist (4-phase) | DEVELOPED | debugging-and-visualization (2.2.3) | torchinfo -> gradient check -> TensorBoard -> diagnose by symptom. |
| model.state_dict() / torch.save/load | DEVELOPED | saving-and-loading (2.3.1) | "Snapshot of all the knobs." Round-trip persistence pattern. |
| Checkpoint pattern (model + optimizer + epoch + loss) | DEVELOPED | saving-and-loading (2.3.1) | Bundle state in one dict. Periodic saves + best-model save. |
| Resume training from checkpoint | DEVELOPED | saving-and-loading (2.3.1) | Restore model + optimizer state_dicts, resume from saved epoch. |
| Early stopping with checkpoints | DEVELOPED | saving-and-loading (2.3.1) | Patience counter + save best model + restore best at end. |
| Device-aware training loop | DEVELOPED | gpu-training (2.3.2) | model.to(device), inputs.to(device), labels.to(device) inside loop. |
| Portable device detection pattern | DEVELOPED | gpu-training (2.3.2) | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| Device-aware checkpoint save/load | DEVELOPED | gpu-training (2.3.2) | map_location=device for portable files. |
| Mixed precision (autocast + GradScaler) | INTRODUCED | gpu-training (2.3.2) | 4-line addition. Automatic dtype management. |
| "When does GPU help?" decision | DEVELOPED | gpu-training (2.3.2) | Model size, batch size, transfer overhead. Under 30s CPU -> GPU probably no help. |
| Overfitting / generalization | DEVELOPED | overfitting-and-regularization (1.3.7) | Scissors pattern. Model capacity framing. Training curves diagnostic. |
| Training curves as diagnostic | DEVELOPED | overfitting-and-regularization (1.3.7) | Train vs val loss. Three phases. Gap = overfitting. |
| Dropout conceptual | DEVELOPED | overfitting-and-regularization (1.3.7) | Randomly silence neurons. Implicit ensemble. |
| Weight decay / L2 conceptual | DEVELOPED | overfitting-and-regularization (1.3.7) | Penalty on large weights. |
| Early stopping conceptual | DEVELOPED | overfitting-and-regularization (1.3.7) | Monitor val loss, patience, save best. |
| nn.Flatten() | INTRODUCED | mnist-project (2.2.2) | Module form of reshape. Flattens after batch dim. |

### Mental Models Already Established

- **"Same heartbeat, new instruments"** -- The training loop rhythm is universal. Reinforced across regression, classification, GPU, checkpointing.
- **"Assembly line with four stations"** -- Forward, loss, backward, update. GPU upgrades the workers. Device placement is logistics.
- **"The scissors pattern"** -- Train/val divergence = overfitting. Every regularization technique aims to keep scissors closed.
- **"Dataset is a menu, DataLoader is the kitchen"** -- Data pipeline separation of concerns.
- **"state_dict is a snapshot of all the knobs"** -- Persistence format for models and optimizers.
- **"Architecture in code, values in file"** -- State_dict decouples learned values from code structure.
- **"Debugging is a systematic workflow, not random guessing"** -- 4-phase checklist.
- **"torchinfo is an X-ray"** -- See model structure before training.
- **"Gradient checking is taking the pulse"** -- Monitor gradient health.
- **"TensorBoard is a flight recorder"** -- Continuous training monitoring.
- **"Loss going down does not mean training is working"** -- Always monitor accuracy alongside loss.
- **"Not a black box"** -- Every abstraction maps to math the student already knows.
- **"Not magic -- automation"** -- Autograd and autocast automate manual work the student understands.
- **"Micrometer -> ruler -> tape measure"** -- Float64 -> float32 -> float16 precision spectrum.
- **"The complete training recipe"** -- He init + batch norm + AdamW + dropout + early stopping.

### What Was Explicitly NOT Covered (Relevant to This Lesson)

- Hyperparameter tuning strategies (grid search, random search) -- never taught. The student picks reasonable defaults.
- Learning rate schedulers -- never taught. Not needed for this project.
- Data augmentation strategies (RandomFlip, RandomCrop) -- mentioned only in datasets-and-dataloaders. NOT developed.
- Fashion-MNIST as a dataset -- never loaded, never discussed. It is the same API as MNIST with different classes.
- Per-class accuracy / confusion matrix -- never shown. Could be a useful addition here.
- Convolutional networks -- explicitly deferred to Series 3. This project uses FC only.

### Readiness Assessment

The student is fully prepared. Every concept, tool, and pattern needed for this project has been taught at DEVELOPED or INTRODUCED depth. The student has completed one end-to-end project (MNIST, 2.2.2) with significant scaffolding. This project uses the exact same tools on a harder dataset with less hand-holding. The challenge is integration and independent decision-making, not new concepts.

The MNIST project (2.2.2) had 3 MISSING prerequisites that needed in-lesson teaching (cross-entropy, softmax, accuracy). This project has zero MISSING prerequisites. The student has every tool. The question is: can they wield them independently?

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to **independently design, train, debug, and improve a classification model on Fashion-MNIST by combining all Series 2 skills without step-by-step scaffolding**.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| nn.Module subclass with regularization layers | INTRODUCED | INTRODUCED | mnist-project (2.2.2) | OK | Student built this once; needs to do it again with their own architecture choices. |
| Cross-entropy loss (nn.CrossEntropyLoss) | INTRODUCED | INTRODUCED | mnist-project (2.2.2) | OK | Classification loss. Used once. Applied here again. |
| Accuracy metric (torch.argmax) | INTRODUCED | INTRODUCED | mnist-project (2.2.2) | OK | Evaluation metric. Used once. Applied here again. |
| model.train() / model.eval() | DEVELOPED | DEVELOPED | mnist-project (2.2.2) | OK | Practiced in MNIST project. Critical for dropout/BN correctness. |
| Train/test evaluation loop | DEVELOPED | DEVELOPED | mnist-project (2.2.2) | OK | Complete evaluation pattern practiced in MNIST. |
| Dataset / DataLoader with torchvision | DEVELOPED | DEVELOPED | datasets-and-dataloaders (2.2.1) | OK | MNIST loading pattern transfers directly to FashionMNIST. |
| Debugging checklist (torchinfo, gradients, TensorBoard) | DEVELOPED | DEVELOPED | debugging-and-visualization (2.2.3) | OK | Student should use these tools proactively when something goes wrong. |
| Checkpoint pattern + early stopping | DEVELOPED | DEVELOPED | saving-and-loading (2.3.1) | OK | Save best model, resume training. Practiced in exercises. |
| Device-aware training loop | DEVELOPED | DEVELOPED | gpu-training (2.3.2) | OK | Full portable GPU pattern. Used in exercises. |
| Mixed precision (autocast + GradScaler) | INTRODUCED | INTRODUCED | gpu-training (2.3.2) | OK | Optional addition. Used in exercises. |
| Overfitting diagnosis + regularization toolkit | DEVELOPED | DEVELOPED | overfitting-and-regularization (1.3.7) + mnist-project (2.2.2) | OK | Scissors pattern, dropout, batch norm, weight decay, early stopping -- all available. |
| torchvision.transforms | INTRODUCED | INTRODUCED | datasets-and-dataloaders (2.2.1) | OK | ToTensor + Normalize. Same pattern for Fashion-MNIST. |

**No gaps. All prerequisites met.** This is by design -- a CONSOLIDATE lesson should have zero gaps.

### Gap Resolution

No gaps found. All prerequisites are at sufficient depth. This lesson introduces no new concepts.

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| **"Fashion-MNIST should be as easy as MNIST -- I should get 97%+ immediately"** | MNIST gave 97% with a simple 3-layer FC model. Fashion-MNIST has the same image size (28x28) and the same number of classes (10). The student may expect similar performance with the same architecture. | A simple 3-layer FC model on Fashion-MNIST gets ~87-88% accuracy, not ~97%. The dataset is harder: distinguishing shirts from coats requires more nuanced features than distinguishing 0 from 1. The 10-point accuracy drop with the same architecture is the signal that this is a harder problem. | After the baseline model. The gap between expected and actual performance is the motivating tension for the experimentation phase. |
| **"More layers or more neurons will always improve the model"** | The student has seen bigger models work better on MNIST. The natural instinct when accuracy is disappointing is to add capacity. | A very wide model (784->1024->512->256->10) with no regularization overfits: train accuracy climbs to 95%+ while test accuracy plateaus at ~88%. Adding capacity without regularization makes the scissors open wider. The fix is regularization, not more neurons. | During the experimentation phase. The student should try a bigger model, see overfitting, then apply regularization. |
| **"I should use the same hyperparameters as MNIST"** | The student's MNIST setup (lr=0.001, batch_size=64, 5 epochs) worked well. Why change what worked? | Fashion-MNIST may need more epochs (15-20) because the classes are harder to separate. The student should observe that loss is still decreasing at epoch 5 and extend training. The assumption "5 epochs is enough" from MNIST does not transfer to a harder dataset. | Baseline training results. The student sees the loss curve is still trending down at epoch 5 and decides to train longer. |
| **"If my accuracy is stuck, the code is broken"** | In MNIST, the model improved rapidly and continuously. Plateau feels like a bug. | A model at ~88% accuracy for 3 consecutive epochs is not broken -- it has reached the limit of what its architecture can do. The debugging checklist (gradients healthy, loss still decreasing slowly, no NaN) confirms the code is correct. The bottleneck is the model, not the code. This is the difference between a bug and a capacity limit. | When the student checks accuracy and sees a plateau. Connects to the debugging checklist -- use it to confirm the code is fine, then shift focus to architecture/regularization. |
| **"This dataset needs CNNs; I cannot do well with FC layers"** | The student may have heard or read that Fashion-MNIST "requires" convolutional networks. Feeling limited by FC-only architecture could be demoralizing. | State-of-the-art FC models on Fashion-MNIST reach ~89-90% accuracy. CNNs reach ~93-95%. There is a gap, but the FC model is not far off. The student can be proud of 89-90% from FC layers and understand that CNNs (Series 3) close the remaining gap. | Lesson conclusion / framing. Set expectations upfront and reaffirm at the end. |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| **Baseline model (~87% accuracy)** | Positive | Establishes the starting point. A simple model without regularization that the student builds themselves. The accuracy is respectable but clearly improvable. | The gap between MNIST accuracy (97%) and Fashion-MNIST accuracy (87%) on the same architecture is the central insight. Same tools, harder problem, worse result. This motivates experimentation. |
| **"MNIST architecture on Fashion-MNIST" comparison** | Negative | Shows that directly copying the MNIST project's architecture without adaptation gives disappointing results. Fashion-MNIST is a different problem. | Disproves "same setup will work." The student should not blindly reuse hyperparameters. Each dataset requires thought about architecture and training. |
| **Improved model with regularization (~89-90% accuracy)** | Positive | The student adds dropout, batch norm, weight decay, and trains longer. Accuracy improves by 2-3 points. The scissors pattern closes. | Validates the entire regularization toolkit from Series 1 and the MNIST project. The improvement is real but not magical -- proving that these tools help incrementally. |
| **Overfitting model (big, unregularized)** | Negative | Student tries a large model without regularization. Train accuracy climbs high while test accuracy plateaus or drops. Clear scissors pattern. | Reinforces the overfitting lesson (1.3.7) in a new context. The student sees the scissors pattern in their own training run, not a textbook example. Also disproves "more neurons = better." |
| **Per-class accuracy breakdown** | Stretch (positive) | Shows which classes are easy (sneakers, trousers) and which are hard (shirt vs coat vs pullover). The model's confusion patterns make visual sense. | Gives the student a richer understanding of their model's behavior than a single accuracy number. Also previews the kind of analysis that becomes important for real projects. |

---

## Phase 3: Design

### Narrative Arc

You have spent nine lessons building your PyTorch toolkit piece by piece. You learned tensors, autograd, nn.Module, training loops, data loading, debugging, saving, checkpointing, and GPU training. You proved the tools work on MNIST and on toy data. Now it is time to use them on your own. Fashion-MNIST looks like MNIST on the surface -- same image size, same number of classes, same API to load it. But the classes are harder: telling a shirt from a coat is not the same as telling a 0 from a 1. Your MNIST architecture will not perform as well, and you will need to make decisions about how to improve it. This is not a lesson with answers at the bottom of the page. This is a project where you experiment, observe, diagnose, adapt, and iterate -- the actual workflow of machine learning. The emotional payoff is not "I followed the steps and it worked." It is "I figured it out myself."

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Code (primary)** | Starter template with gaps. The student fills in architecture, training loop, evaluation, and experimentation. Code blocks are scaffolds, not solutions. | This is a PROJECT lesson. The code IS the deliverable. The student should be writing more code than reading explanations. |
| **Visual** | Fashion-MNIST sample grid showing all 10 classes, so the student sees what they are classifying. Per-class accuracy breakdown showing easy vs hard classes. | The dataset should feel concrete, not abstract. Seeing a coat next to a shirt makes the classification difficulty visceral. Per-class accuracy adds texture beyond a single number. |
| **Concrete example (training curves)** | Recharts or printed epoch tables showing baseline vs improved model. Train/test accuracy side by side. The scissors pattern visible in the baseline, closing in the improved model. | Training curves are the diagnostic tool the student has used since 1.3.7. Seeing them on their own training run closes the loop. |
| **Verbal/Analogy** | "Same tools, harder problem" -- the theme of the lesson. Everything transfers from MNIST, but the dataset demands more careful architecture and regularization choices. | Frames the lesson not as "learn new things" but as "prove you can use what you have." Reduces anxiety about a new dataset. |
| **Intuitive** | The "why is this hard?" moment: looking at ambiguous Fashion-MNIST images (shirt vs pullover, coat vs shirt, ankle boot vs sneaker) and recognizing that even humans sometimes struggle. | Grounds accuracy numbers in visual reality. If the images look confusing to a human, 88% accuracy is impressive, not disappointing. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 0 (zero genuinely new concepts)
- **Previously untaught items introduced within lesson:** 2 minor items:
  - Fashion-MNIST as a dataset (same API as MNIST, different classes -- trivial to introduce)
  - Per-class accuracy / class-level analysis (conceptually trivial given torch.argmax is known)
- **Previous lesson load:** STRETCH (gpu-training -- device management, mixed precision)
- **This lesson's load:** CONSOLIDATE -- integration project, no new concepts, cognitive effort is in decision-making and synthesis rather than learning
- **Assessment:** Appropriate. STRETCH -> CONSOLIDATE is the correct trajectory. The student recovers from the GPU training lesson's coordination complexity by applying familiar patterns in a new context. The challenge is independence, not novelty.

### Connections to Prior Concepts

| Prior Concept | Connection |
|---------------|-----------|
| MNIST project (2.2.2) | Direct sequel. Same structure (load data, build model, train, evaluate), harder dataset, less scaffolding. The student should consciously recall and adapt their MNIST approach. |
| Cross-entropy / softmax / accuracy (2.2.2) | Reused without re-explanation. The student should recognize these as classification tools from MNIST. |
| Regularization toolkit: dropout, BN, weight_decay (1.3.7 + 2.2.2) | Applied to a new problem. The student chooses which techniques to use and observes their effect on the scissors pattern. |
| Debugging checklist (2.2.3) | Used when things go wrong. If accuracy plateaus or loss spikes, the student should reach for the 4-phase checklist rather than panicking. |
| Saving/loading + checkpointing (2.3.1) | The student should save their best model. Early stopping with patience is a practiced pattern. |
| GPU training + mixed precision (2.3.2) | The student should use the portable device pattern. Mixed precision is optional but available. |
| Training curves diagnostic (1.3.7) | The primary diagnostic tool. The student monitors train vs test metrics to detect overfitting and guide improvements. |
| "Same heartbeat, new instruments" (2.1.4) | The training loop is the same. The student should feel confident that the loop structure transfers. |
| "The scissors pattern" (1.3.7) | The student will see scissors in their own training and apply regularization to close them. |

### Analogies That Might Mislead

- **"MNIST was easy, so similar datasets should be easy"** -- Fashion-MNIST looks identical on the API surface (28x28, 10 classes, same loading code). The student may not realize the classification difficulty is significantly higher. Must address this explicitly in framing.
- **"The complete training recipe is a fixed recipe"** -- The student has a recipe (He init + BN + AdamW + dropout + early stopping) but may apply it rigidly rather than experimenting. The recipe is a starting point, not a final answer.

### Scope Boundaries

**This lesson IS about:**
- Loading Fashion-MNIST and understanding its 10 classes
- Designing and building an FC classification model independently
- Training with GPU, checkpointing, and early stopping (all patterns already known)
- Diagnosing issues using the debugging checklist
- Experimenting with architecture size, regularization, and training duration
- Analyzing per-class accuracy to understand model behavior
- The experience of independent ML experimentation

**This lesson is NOT about:**
- Convolutional networks (deferred to Series 3 -- stated explicitly)
- Data augmentation (transforms beyond ToTensor + Normalize)
- Learning rate schedulers or gradient clipping
- Hyperparameter search strategies (grid search, Bayesian optimization)
- Achieving state-of-the-art Fashion-MNIST accuracy
- New PyTorch APIs or concepts
- Information-theoretic analysis of dataset difficulty

**Target depths:**
- Fashion-MNIST dataset: INTRODUCED (load it, understand the 10 classes, see samples)
- Independent model design: APPLIED (the student makes architecture decisions and justifies them)
- All Series 2 skills in combination: APPLIED (the graduation exercise -- use everything together independently)
- Per-class accuracy analysis: INTRODUCED (compute it, interpret it, draw conclusions)

### Lesson Outline

**1. Context + Constraints** (~2-3 paragraphs)
What: Fashion-MNIST project -- second end-to-end project, less scaffolding than MNIST. Scope: FC networks only (no CNNs). Realistic target: ~88-90% with FC. This is a CONSOLIDATE lesson: no new concepts, all the tools are already in your hands. The Colab notebook is the primary deliverable.

What makes this different from MNIST: you are making the decisions. The lesson gives you a baseline, a set of experiments to try, and the tools to diagnose what is happening. It does not give you the answers.

**2. Hook -- "Same Shape, Different Challenge"**
Type: *before-after comparison.* Show a grid of Fashion-MNIST samples next to MNIST samples. Same 28x28, same 10 classes, same API. But look at the images: T-shirt vs shirt, pullover vs coat -- some of these are hard even for humans. Then reveal: "Your MNIST model got 97%. On Fashion-MNIST, it gets 87%. Same tools, harder problem. Your job: close that gap."

The hook establishes the central tension. The student expects high performance (MNIST set that expectation), sees a gap, and is motivated to close it.

**3. Fashion-MNIST: The Dataset** (~3-4 paragraphs + visual)
- Load with torchvision.datasets.FashionMNIST (same API as MNIST -- one word changes)
- Show the 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Show a sample grid (matplotlib) with class labels
- Key observation: some classes look very similar (shirt/coat/pullover). This is WHY the problem is harder.
- Normalization values: Fashion-MNIST has different mean/std than MNIST. Show how to compute them or provide them.

**4. The Baseline: Your MNIST Architecture** (~3 paragraphs + code)
- Take the architecture from the MNIST project (784->256->128->10, no regularization) and run it on Fashion-MNIST
- Train for 5 epochs (same as MNIST)
- Show results: ~87-88% accuracy. Loss is still decreasing at epoch 5.
- The observation: "This is 10 points below your MNIST score. Same model, same training, different data."
- This sets up the experimentation phase. The baseline is the thing to beat.

**5. Check 1 -- Diagnose the Baseline** (predict-and-verify)
Before experimenting, the student diagnoses the baseline using their debugging toolkit:
(a) "The loss is still decreasing at epoch 5. What does this tell you?" (Answer: the model has not converged. Train longer.)
(b) "The train accuracy is 92% but test accuracy is 87%. What pattern is this?" (Answer: the scissors pattern -- overfitting. Apply regularization.)
(c) "Which debugging tool would you use to check if the model's gradients are healthy?" (Answer: log_gradient_norms() -- per-layer gradient check.)

This check activates the debugging mindset BEFORE the student starts experimenting. It is diagnostic, not just knowledge recall.

**6. Experiment 1: Train Longer** (~2-3 paragraphs + results)
- Train for 20 epochs instead of 5
- Show that accuracy improves by 1-2 points (88-89%) but then plateaus
- The loss curve flattens -- the model has reached its capacity
- Key insight: more epochs helped, but the model needs more than just more training time
- This is the simplest experiment. Low activation energy. Start here.

**7. Experiment 2: Add Regularization** (~3-4 paragraphs + code)
- Apply the regularization toolkit from the MNIST project: BatchNorm + Dropout(0.3) + weight_decay
- Use the Linear->BatchNorm->ReLU->Dropout ordering pattern
- model.train() / model.eval() -- the student should add these automatically by now
- Train for 20 epochs with early stopping (patience=5)
- Show results: ~89-90% accuracy. The scissors pattern closes.
- Comparison table: baseline (20 epochs, no reg) vs improved (20 epochs, with reg). Train accuracy may go down but test accuracy goes up. "Regularization increases training loss -- that is the point" (callback to 1.3.7).

**8. Experiment 3: Architecture Decisions** (~3-4 paragraphs + guidance)
- Wider layers? Deeper network? Different dropout rate?
- Provide guidance, not answers: "Try 784->512->256->128->10 (more capacity). Try 784->256->256->128->10 (deeper). Try dropout=0.5 instead of 0.3."
- The student experiments in the Colab notebook and compares results
- Key point: there is no single right answer. ML experimentation is about trying things and observing. Use TensorBoard (or printed metrics) to compare runs.
- Negative example opportunity: a very large model (1024->512->256->10) with no regularization shows severe overfitting. The student should be able to diagnose this.

**9. Understanding Your Model: Per-Class Accuracy** (~3-4 paragraphs + code + visual)
- Show how to compute accuracy per class: group predictions by true label, compute correct/total for each
- Display results: sneakers and trousers are easy (95%+), shirt vs pullover vs coat is hard (75-85%)
- Show a confusion matrix or per-class bar chart
- The insight: "Your model is not uniformly 89% accurate. It is excellent at some classes and mediocre at others. The hard classes are the ones that look similar to each other."
- This is a richer analysis than a single accuracy number and builds the student's diagnostic skills

**10. The Complete Pipeline** (~2-3 paragraphs + code scaffold)
- Show the student the full pipeline they should have in their notebook:
  1. Data loading with transforms
  2. Model definition with regularization
  3. Device-aware training loop with checkpointing and early stopping
  4. Evaluation loop with per-class accuracy
  5. Best model saving
- This is a reference scaffold, not new teaching. The student should recognize every piece.
- Emphasize: this pipeline carries forward to every future project. The specifics (architecture, dataset, hyperparameters) change, but the structure is the same.

**11. Check 2 -- Transfer Question**
"You are starting a new project on a dataset with 50 classes and 100x100 color images. You have not learned CNNs yet. How would you approach this using only the tools you have? What would you try first, and how would you know if your model is working?"

Expected answer: Flatten to 30,000 (100x100x3), start with a simple FC model, cross-entropy loss with 50 output units, train with GPU, use debugging checklist, monitor per-class accuracy to identify hard classes, add regularization if overfitting. The student should recognize this is the same workflow applied to different data.

**12. Summarize -- What You Can Do Now** (~3-4 paragraphs)
- Recap what the student proved they can do: load data, design a model, train it on GPU, save checkpoints, diagnose issues, improve with regularization, analyze per-class performance
- This is the Series 2 graduation. Every lesson from tensors (2.1.1) through this project built one piece of this workflow.
- Name the mental models that carried through: "same heartbeat, new instruments," the debugging checklist, the scissors pattern, the training recipe
- Emotional payoff: "You did not follow a tutorial. You made decisions, observed results, and adapted. That is machine learning."

**13. What Comes Next -- Series 3 Preview** (~2 paragraphs)
- "Your FC model topped out around 89-90%. To go further, you need a different kind of architecture -- one that understands spatial structure. Series 3 introduces convolutional neural networks."
- Brief tease: CNNs can reach ~93-95% on Fashion-MNIST. The ~5% gap between your best FC model and a CNN motivates Series 3.
- Frame it as expansion, not correction: "Your FC model is good. CNNs are the tool for the next level."

### Widget Decision

**No custom interactive widget needed.** This is a project lesson. The primary interaction happens in the Colab notebook. The lesson page provides:
- Fashion-MNIST sample grid (static image or matplotlib in Colab)
- Training results tables/charts (Recharts for baseline vs improved comparison, or printed tables)
- Per-class accuracy breakdown (bar chart or table)
- Code scaffolds that the student fills in

The lesson page serves as a guide and reference. The Colab notebook is where the student does the work.

### Notebook Design: More Autonomous Than MNIST

The MNIST project notebook (2.2.2) was heavily guided: every cell had complete code or near-complete code with small gaps. This notebook should have significantly less scaffolding:

- **Section 1 (Data Loading):** Provided, since FashionMNIST API is identical to MNIST. The student should not spend time on boilerplate.
- **Section 2 (Baseline Model):** Provided as a starting point. The student runs it, observes results, and diagnoses the gap.
- **Section 3 (Experimentation):** Minimal scaffolding. The student gets a structure ("Build your improved model here", "Train and compare") but writes the code. Hints available in collapsible cells.
- **Section 4 (Analysis):** Partially scaffolded. Per-class accuracy computation provided or lightly guided. The student interprets the results.
- **Section 5 (Full Pipeline):** Independent. The student puts it all together with checkpointing, GPU, and early stopping.
- **Stretch goals** (optional): Confusion matrix visualization, TensorBoard comparison of 3+ architectures, finding the best model they can build.

The progression: guided (data loading) -> supported (baseline) -> lightly scaffolded (experimentation) -> independent (full pipeline).

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (zero MISSING, zero GAP)
- [x] No multi-concept jumps in exercises
- [x] All gaps have explicit resolution plans (no gaps found)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 0 new concepts (well within limit; effort is synthesis/independence)
- [x] Every prior concept connected to this lesson's application
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-09 (Iteration 1/3)

### Summary
- Critical: 1
- Improvement: 4
- Polish: 3

### Verdict: NEEDS REVISION

No show-stopping conceptual errors, but one critical finding (the missing Colab notebook) and several improvement-level findings that would meaningfully strengthen the lesson as a graduation experience.

### Findings

#### [CRITICAL] — Colab notebook does not exist in the repository

**Location:** "Build It Yourself" section (line ~906), link to `notebooks/2-3-3-fashion-mnist-project.ipynb`
**Issue:** The lesson links to a Colab notebook at `https://colab.research.google.com/github/brennancheung/CourseAI/blob/main/notebooks/2-3-3-fashion-mnist-project.ipynb`, but no file matching `2-3-3-fashion-mnist-project*` exists in the repository. The planning document explicitly states "The Colab notebook is the primary deliverable" and the lesson itself says "The notebook is where the real work happens." Without the notebook, the lesson is purely informational and the student cannot do the project.
**Student impact:** The student clicks the Colab link and gets a 404. The entire experimentation and independence arc of the lesson collapses. The lesson becomes a walkthrough to read rather than a project to do.
**Suggested fix:** Create the notebook `notebooks/2-3-3-fashion-mnist-project.ipynb` following the scaffolding plan from the design section: Section 1-2 provided (data loading + baseline), Section 3 lightly scaffolded (experimentation with hints in collapsible cells), Section 4 partially guided (per-class analysis), Section 5 independent (full pipeline). Include stretch goals (confusion matrix, TensorBoard multi-run comparison).

---

#### [IMPROVEMENT] — No sample images of Fashion-MNIST shown in the lesson

**Location:** "Same Shape, Different Challenge" section (lines ~126-177) and "Fashion-MNIST: The Dataset" section (lines ~179-249)
**Issue:** The planning document specifies a visual modality: "Fashion-MNIST sample grid showing all 10 classes, so the student sees what they are classifying" and "Show a sample grid (matplotlib) with class labels." The built lesson describes the classes in text and uses GradientCards to list them, but never shows actual images. The student reads about shirts and coats being hard to distinguish but never sees why. The planning document also says: "looking at ambiguous Fashion-MNIST images (shirt vs pullover, coat vs shirt, ankle boot vs sneaker) and recognizing that even humans sometimes struggle."
**Student impact:** The student is told the problem is hard but does not viscerally see it. The text says "if you squint at a tiny grayscale image of a shirt and a coat, they look nearly identical" but the student has never squinted at these images. The motivation relies on the student taking the lesson's word for it rather than experiencing the difficulty firsthand.
**Suggested fix:** Add a static image grid (or a simple component that renders a few sample images) showing representative examples from each class. Even a static PNG embedded in the lesson would work. Alternatively, explicitly note that the Colab notebook's first cell displays sample images and direct the student there before the hook section. The visual does not need to be interactive, but it needs to exist somewhere the student sees it before being told the problem is hard.

---

#### [IMPROVEMENT] — Mixed precision not mentioned anywhere in the lesson or pipeline

**Location:** "The Complete Pipeline" section (lines ~706-808) and the full lesson generally
**Issue:** The immediately prior lesson (gpu-training, 2.3.2) introduced mixed precision (autocast + GradScaler) at INTRODUCED depth. The planning document lists it as an available prerequisite ("Mixed precision (autocast + GradScaler): INTRODUCED"). The complete pipeline section enumerates every prior lesson it connects to (nine lessons named in the TipBlock aside) but does not mention mixed precision at all. The full pipeline code does not include autocast/GradScaler, and there is no mention of it as an optional addition.
**Student impact:** The student just learned mixed precision and has no opportunity to apply or even consider it. This is a reinforcement gap. The concept was INTRODUCED one lesson ago and then completely ignored in the capstone project that is supposed to integrate all Series 2 skills. The student may forget it exists by the time they need it in Series 3.
**Suggested fix:** Add mixed precision as an optional experiment or stretch goal. A brief mention in the "Architecture Decisions" section ("You can also try adding mixed precision from the GPU Training lesson") or a note in the pipeline section ("For faster GPU training, wrap the forward pass in autocast and add a GradScaler") would suffice. It does not need to be a full section, but it should be acknowledged as a tool the student has.

---

#### [IMPROVEMENT] — "Debugging and Visualization" and "Overfitting and Regularization" referenced without lesson titles

**Location:** Check 1 section (lines ~314-393), specifically: "Use what you know from Debugging and Visualization and Overfitting and Regularization."
**Issue:** The text references lesson names as though they are proper nouns the student would recognize, but they are run together without any formatting or separation. "Debugging and Visualization and Overfitting and Regularization" reads as a single garbled phrase. Other places in the lesson (e.g., "Debugging and Visualization" in section 9, "Overfitting and Regularization" in section 7) have the same issue but are less confusing because they appear individually.
**Student impact:** Mild confusion parsing the sentence. The student may need to re-read it to understand these are two separate lesson references.
**Suggested fix:** Use italic or bold formatting to distinguish lesson names: "Use what you know from *Debugging and Visualization* and *Overfitting and Regularization*." Alternatively, restructure the sentence: "Use your debugging checklist and your overfitting diagnosis skills."

---

#### [IMPROVEMENT] — Misconception "If my accuracy is stuck, the code is broken" not explicitly addressed

**Location:** Entire lesson; planned misconception from the design document
**Issue:** The planning document identifies five misconceptions. Four are addressed in the lesson: (1) "should get 97%" is addressed by the baseline results and the WarningBlock; (2) "more neurons = better" is addressed by the negative example in Experiment 3; (3) "same hyperparameters from MNIST" is addressed by Experiment 1 (train longer); (5) "needs CNNs" is addressed in the conclusion and expectations. However, misconception (4) "If my accuracy is stuck, the code is broken" is not directly addressed anywhere. The lesson tells the student to use the debugging checklist in Check 1 part (c), but it never explicitly addresses the moment when accuracy plateaus and the student might think the code is buggy. The closest is the implicit suggestion that regularization is the fix, not code changes, but the misconception itself is never surfaced or named.
**Student impact:** When the student hits a plateau during experimentation (which the lesson says will happen), they may waste time hunting for bugs instead of recognizing a capacity limit. The debugging checklist will reveal healthy gradients, but the student needs to be told: "If the checklist says the code is fine and accuracy is still stuck, the bottleneck is the model, not the code."
**Suggested fix:** Add a brief callout (InsightBlock or inline text) in the Experiment 1 or Experiment 3 section that explicitly names this: "If accuracy plateaus for several epochs, use the debugging checklist. If gradients are healthy and loss is still decreasing slowly, the code is not broken. The model has reached its capacity. The fix is a different architecture or more regularization, not a code change."

---

#### [POLISH] — Em dash spacing inconsistency in one location

**Location:** Line 99, "That lesson gave you complete code with small gaps to fill. This one gives you a baseline and a set of experiments&mdash;you write the code and make the decisions."
**Issue:** This is correct (&mdash; with no spaces). However, on line 113: "Same heartbeat, your decisions." The word "heartbeat" could benefit from an em dash for consistency with the lesson's rhetorical style ("Same heartbeat—your decisions"), but this is subjective. More importantly, all other em dashes in the lesson are correctly formatted with no spaces.
**Student impact:** None. This is purely stylistic consistency.
**Suggested fix:** No action required. All em dashes are correctly formatted. The "heartbeat, your decisions" phrasing is fine as-is with a comma.

---

#### [POLISH] — The "Emotional payoff" block uses violet styling with no explanation

**Location:** Lines 966-979, the centered violet block: "You did not follow a tutorial..."
**Issue:** This block uses a custom `border-violet-500/30 bg-violet-500/5` style that differs from all the standard block components used elsewhere in the lesson (InsightBlock, ConceptBlock, WarningBlock, etc.). While it is visually distinct and effective for an emotional payoff moment, it is not a reusable component. This is not a pedagogical issue but a pattern consistency observation.
**Student impact:** None. The block is visually effective and emotionally resonant. It reads well.
**Suggested fix:** Consider extracting this as a reusable `CelebrationBlock` or `PayoffBlock` if more lessons will use this pattern. For this lesson alone, the inline styling is fine. No action needed.

---

#### [POLISH] — Series 2 Complete block lists "Mixed precision with autocast and GradScaler" as an achievement but the lesson never uses it

**Location:** ModuleCompleteBlock (line ~1033) and Series 2 Complete block (lines ~1042-1093)
**Issue:** The ModuleCompleteBlock lists "Mixed precision with autocast and GradScaler" as a Module 2.3 achievement. The Series 2 Complete block lists "Saving/loading, GPU training, Fashion-MNIST project" for Module 2.3. Mixed precision was taught in the GPU training lesson, so it is technically a Module 2.3 achievement, but the Fashion-MNIST project (the culminating exercise) never uses it. This creates a slight disconnect: the module claims the student proved competence in mixed precision, but the capstone project did not require or even mention it.
**Student impact:** Minimal. The student may notice the disconnect if they are reading carefully.
**Suggested fix:** This resolves naturally if mixed precision is added as an optional experiment (see the IMPROVEMENT finding above). If mixed precision is mentioned in the project even as a stretch goal, the achievement listing is justified.

---

### Review Notes

**What works well:**

1. **The graduation arc is genuinely effective.** The progression from "here is a baseline" to "here are experiments to try" to "you make the decisions" to "here is everything you have built" is well-structured. The student genuinely moves from supported to independent across the lesson.

2. **The hook is strong.** The side-by-side MNIST vs Fashion-MNIST class comparison using GradientCards immediately establishes the tension. The specific accuracy gap (97% vs 87%) gives a concrete target. The framing "Same tools, harder problem" is compelling.

3. **The checks are diagnostic, not rote.** Check 1 asks the student to diagnose the baseline using their debugging skills. Check 2 is a genuine transfer question (50-class, 100x100 dataset). These test understanding, not recall.

4. **Per-class accuracy is a well-placed introduction.** It provides a richer diagnostic than overall accuracy and naturally previews the kind of analysis needed in Series 3. The Easy/Hard class split with the visual reasoning (silhouette similarity) is intuitive.

5. **The emotional payoff block and Series 2 Complete celebration are appropriate.** This is a graduation lesson and it feels like one. The student should feel proud at the end.

6. **Scope boundaries are well maintained.** The lesson explicitly defers CNNs, data augmentation, schedulers, and hyperparameter search. The ConstraintBlock at the top sets expectations cleanly.

**Patterns to watch:**

1. The primary interaction mode for this lesson is the Colab notebook, which does not exist yet. Until the notebook is built, the lesson is incomplete regardless of how well the lesson page is written.

2. The lesson is text-heavy for a project lesson. The planning document emphasizes that "The code IS the deliverable" and "the student should be writing more code than reading explanations." The lesson page has substantial explanatory prose. This is appropriate as a reference guide, but the Colab notebook needs to be the thing the student actually spends time on.

3. This is a good candidate for revisiting after the student actually completes it, since the real test of a project lesson is whether the scaffolding level is right. Too much scaffolding and it is not a graduation; too little and the student gets stuck.

---

## Review — 2026-02-09 (Iteration 2/3)

### Summary
- Critical: 1
- Improvement: 1
- Polish: 1

### Verdict: NEEDS REVISION

All four IMPROVEMENT findings from iteration 1 have been fixed. Mixed precision is now included as an optional experiment with complete code. Lesson name formatting uses `<em>` tags throughout. The "stuck != broken" misconception is explicitly addressed in Experiment 1 with both inline text and a dedicated WarningBlock. The one remaining critical finding (missing notebook) was deferred as a project-wide task in iteration 1 and remains outstanding. One new improvement finding and one polish finding identified on this pass.

### Iteration 1 Fix Verification

1. **[CRITICAL] Colab notebook does not exist** -- STILL MISSING. No file `notebooks/2-3-3-fashion-mnist-project.ipynb` exists. Deferred as project-wide task (notebook creation is separate from lesson page review). Carrying forward.
2. **[IMPROVEMENT] No sample images of Fashion-MNIST** -- PARTIALLY ADDRESSED. The lesson now provides matplotlib visualization code (lines 228-248) with specific instructions ("Look closely at rows 2, 4, and 6"). The student will see the images when they run the notebook cell. The lesson page still has no embedded images, but for a project lesson where the student is expected to open the notebook early, directing them to run the visualization code is a reasonable approach. Downgrading to POLISH (see below).
3. **[IMPROVEMENT] Mixed precision not mentioned** -- FIXED. Lines 848-882 add mixed precision as an optional experiment with a complete code block showing autocast + GradScaler integration. The TipBlock aside (lines 894-901) mentions mixed precision by name. The previous POLISH finding about the ModuleCompleteBlock listing mixed precision as an achievement is now resolved by this inclusion.
4. **[IMPROVEMENT] Lesson names formatting** -- FIXED. Lines 363-365 now use `<em>` tags to distinguish lesson names: "debugging checklist from *Debugging and Visualization* and your overfitting diagnosis skills from *Overfitting and Regularization*." The sentence was restructured to avoid the garbled double-and construction.
5. **[IMPROVEMENT] "Stuck = broken" misconception not addressed** -- FIXED. Lines 457-464 explicitly address this: "If accuracy plateaus for several epochs, that does not mean your code is broken. Use the debugging checklist: run the gradient check, verify loss is still decreasing (even slowly), confirm no NaNs. If the checklist says the code is healthy, the bottleneck is the model architecture---not a bug. The fix is a different architecture or more regularization, not a code change." The WarningBlock at lines 484-487 reinforces: "A plateau means the model has reached its capacity, not that the code has a bug."

### Findings

#### [CRITICAL] — Colab notebook still does not exist (carried forward)

**Location:** "Build It Yourself" section (line ~960), link to `notebooks/2-3-3-fashion-mnist-project.ipynb`
**Issue:** Same as iteration 1. The notebook file does not exist in the repository. The lesson page is complete and well-structured, but the primary deliverable (the notebook) is missing.
**Student impact:** The student clicks the Colab link and gets a 404. The project cannot be completed without the notebook.
**Suggested fix:** Create the notebook following the design section's scaffolding plan. This is a separate work item from the lesson page review. The lesson page itself is ready; the notebook is the remaining deliverable.
**Note:** This was deferred in iteration 1 as a project-wide task. It remains the only blocking item for this lesson to be fully functional.

---

#### [IMPROVEMENT] — No sample images visible on the lesson page itself

**Location:** "Same Shape, Different Challenge" (lines 126-177) and "Fashion-MNIST: The Dataset" (lines 179-288)
**Issue:** The planning document's visual modality calls for "Fashion-MNIST sample grid showing all 10 classes, so the student sees what they are classifying." The lesson provides matplotlib code to generate this grid (lines 228-248) but the student must run it in the notebook to see it. On the lesson page itself, the student reads about shirt-vs-coat similarity without ever seeing it. The GradientCards listing class names are informative but not visual in the way images would be.
**Student impact:** The student reads through the hook and dataset sections understanding intellectually that the classes are similar, but does not experience the "even humans struggle" moment the planning document intended. The motivation is slightly weaker on the lesson page than it could be. However, once the student opens the notebook and runs the visualization cell, this gap closes.
**Suggested fix:** Either (a) add a static image (a pre-rendered PNG of the Fashion-MNIST sample grid) embedded in the lesson page, or (b) add an explicit callout before the hook: "Before reading further, open the notebook and run the first cell to see what these images actually look like." Option (b) is lighter weight and aligns with the project lesson's notebook-first philosophy.

---

#### [POLISH] — Fashion-MNIST sample images note in "visualize" code block could be more directive

**Location:** Lines 222-248, the matplotlib visualization code block
**Issue:** The text before the code says "Run this in your notebook to see what your model is actually classifying" (line 224). This is a reasonable instruction, but it comes after the hook section has already told the student the problem is hard. The student reads about the difficulty before they have the tool to see it. Moving the visualization code or the directive to run it earlier (before the hook's accuracy comparison) would make the narrative flow better: see the data, understand the difficulty, then see the accuracy gap.
**Student impact:** Minor. The student will eventually see the images. The ordering is slightly suboptimal but not confusing.
**Suggested fix:** No action required for this pass. If the lesson page gets a static image (per the IMPROVEMENT above), this ordering issue resolves naturally.

### Review Notes

**What improved since iteration 1:**

1. **Mixed precision integration is clean.** The optional experiment section (lines 848-882) fits naturally into the "Complete Pipeline" section without disrupting the flow. The code is complete and the context ("Fashion-MNIST is small enough that the speedup may be modest, but it is a good habit to practice") sets appropriate expectations. This was the most impactful fix.

2. **The "stuck != broken" callout is well-placed.** Putting it in Experiment 1 (Train Longer) is exactly right because that is where the student first encounters a plateau. The WarningBlock reinforces it in the aside. The student is armed with the right mental model before they start experimenting.

3. **Lesson name formatting is clean throughout.** All lesson references use `<em>` tags consistently across the entire lesson.

**Remaining gap:**

The notebook is the only truly blocking item. The lesson page is pedagogically sound and ready for the student. Once the notebook is created following the design document's scaffolding plan, this lesson is complete.

**Assessment for next iteration:**

If the notebook creation is treated as a separate work item (which is reasonable given its scope), the lesson page can be considered ready with one minor improvement (adding a visual element for Fashion-MNIST images). The improvement finding is not blocking---the lesson works without it, just slightly less powerfully. A third review iteration should focus on whether the image improvement was addressed, and if so, this lesson page should PASS.

---

## What Was Actually Built

The lesson was built closely following the Phase 3 design with no significant structural deviations. All 13 sections from the outline are present in the built component.

### Deviations from Design

1. **Colab notebook not created** -- The planning document states "The Colab notebook is the primary deliverable" and the design specifies a 5-section notebook with decreasing scaffolding. The notebook file `notebooks/2-3-3-fashion-mnist-project.ipynb` does not exist. The lesson page links to it and describes its structure, but the notebook itself is a separate work item. This is the only blocking gap.

2. **No embedded sample images on the lesson page** -- The design called for a "Fashion-MNIST sample grid showing all 10 classes." The built lesson uses emoji icons in a CSS grid to represent classes visually (with red/green grouping for hard/easy classes) and provides matplotlib code the student runs in the notebook. The review accepted this as reasonable for a project lesson where the student opens the notebook early.

3. **Training curves not shown as Recharts charts** -- The design mentioned "Recharts or printed epoch tables showing baseline vs improved model." The built lesson uses a ComparisonRow with bullet-pointed metrics (train accuracy, test accuracy, gap) rather than interactive charts. Simpler and appropriate for a lesson where the student generates their own curves in the notebook.

4. **Confusion matrix not shown** -- The design listed confusion matrix as a stretch goal. The built lesson mentions it in the notebook structure description but does not show one on the lesson page. Per-class accuracy (the primary analysis tool) is fully implemented with code and visual breakdown.

### What Worked Well

- The graduation arc (baseline -> diagnose -> experiment -> analyze -> full pipeline) transferred cleanly from design to implementation
- Two review iterations caught and fixed all improvement-level issues before recording
- Mixed precision integration as optional experiment resolved the reinforcement gap from review
- The "stuck != broken" misconception callout was placed exactly where the student first encounters a plateau (Experiment 1)
