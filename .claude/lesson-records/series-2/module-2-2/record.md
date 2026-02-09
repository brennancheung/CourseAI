# Module 2.2: Real Data — Record

**Goal:** The student can load, transform, batch, and iterate over real datasets in PyTorch, then build, train, evaluate, and debug a complete model on MNIST — their first end-to-end project on real data.
**Status:** Complete (3 of 3 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| torch.utils.data.Dataset abstraction | DEVELOPED | datasets-and-dataloaders | `__len__` returns count, `__getitem__` returns one (input, target) pair by index. Custom subclass pattern demonstrated with SimpleDataset. |
| torch.utils.data.DataLoader | DEVELOPED | datasets-and-dataloaders | Wraps a Dataset; handles batching (`batch_size`), shuffling (`shuffle=True`), iteration, and last-batch handling. Yields tensors with batch dimension prepended. |
| Dataset/DataLoader two-layer separation of concerns | DEVELOPED | datasets-and-dataloaders | Dataset = data access (one sample). DataLoader = logistics (batching, shuffling, iteration). Neither knows about the other's job. |
| Lazy loading pattern in __getitem__ | INTRODUCED | datasets-and-dataloaders | Store file paths in __init__, load data from disk in __getitem__. Shown as skeleton (LazyImageDataset), not built end-to-end. |
| torchvision.transforms pipeline | INTRODUCED | datasets-and-dataloaders | Compose(), ToTensor(), Normalize(). Run per-sample in __getitem__, not upfront. Original data on disk unchanged. Random augmentations can produce different results per access. |
| torchvision.datasets (pre-built datasets) | INTRODUCED | datasets-and-dataloaders | MNIST loaded via `datasets.MNIST(root, train, download, transform)`. 60,000 images, shape [1, 28, 28], labels 0-9. |
| DataLoader stacking / collation | INTRODUCED | datasets-and-dataloaders | DataLoader calls torch.stack() on individual samples to form batch tensors. Requires uniform shapes — inconsistent shapes in __getitem__ cause RuntimeError. |
| shuffle=True as implementation of random sampling | DEVELOPED | datasets-and-dataloaders | Connects polling analogy (1.3.4) to practice: shuffle=True re-shuffles indices each epoch, making each batch a random sample. Sorted data without shuffling demonstrated as catastrophic negative example. |
| MNIST image tensor shape [B, 1, 28, 28] | INTRODUCED | datasets-and-dataloaders | 64 images, 1 channel (grayscale), 28x28 pixels. Batch dimension is always first. |
| Iterations per epoch = ceil(N/B) in practice | DEVELOPED | datasets-and-dataloaders | 100 samples / batch_size 8 = 13 iterations (12 full + 1 partial). Connects formula from 1.3.4 to concrete DataLoader iteration count. |
| Softmax function | INTRODUCED | mnist-project | Converts raw logits to probabilities (all positive, sum to 1). Taught as "sigmoid generalized to multiple classes." Formula shown, one element worked through step-by-step (e^3.8 / sum). Key point: do NOT apply manually — nn.CrossEntropyLoss handles it internally. |
| Cross-entropy loss | INTRODUCED | mnist-project | Loss function for classification: L = -log(p_correct). Motivated by MSE's failure on classification (treats all errors equally regardless of confidence). Taught via "confidence penalty" intuition + concrete value table (0.05 for confident-right, 0.69 for unsure, 4.61 for confident-wrong). Extends "wrongness score" framing from MSE. |
| nn.CrossEntropyLoss (PyTorch API) | INTRODUCED | mnist-project | Takes raw logits + integer class labels. Applies log-softmax internally for numerical stability. Drop-in replacement for nn.MSELoss in the training loop. Warning: do NOT add softmax before it (double-application). |
| Accuracy metric (% correct) | INTRODUCED | mnist-project | torch.argmax(outputs, dim=1) to get predicted class, then compare to labels. Human-readable evaluation metric vs loss (what the model optimizes). Distinction: loss and accuracy correlate but are not identical. |
| Classification vs regression framing | INTRODUCED | mnist-project | Regression: continuous output, MSE loss. Classification: discrete categories, cross-entropy loss. Same training loop, different loss function. |
| nn.Flatten() module | INTRODUCED | mnist-project | Module version of reshape — flattens every dimension after batch dim into a single vector. Used to convert [B, 1, 28, 28] images to [B, 784] for fully-connected layers. |
| nn.Dropout(p) as PyTorch module | INTRODUCED | mnist-project | Bridges concept (DEVELOPED from 1.3.7) to API. nn.Dropout(p=0.3) silences 30% of neurons. Active in model.train(), disabled in model.eval(). |
| nn.BatchNorm1d(features) as PyTorch module | INTRODUCED | mnist-project | Bridges concept (INTRODUCED from 1.3.6) to API. Normalizes activations between layers. Uses batch statistics in model.train(), running averages in model.eval(). |
| model.train() / model.eval() in practice | DEVELOPED | mnist-project | Practiced for real: model.train() enables dropout + batch-stat BN; model.eval() disables dropout + running-stat BN. Forgetting model.eval() before testing gives noisy, unreliable results. Elevated from MENTIONED (2.1.4) to DEVELOPED. |
| weight_decay parameter in optimizer | INTRODUCED | mnist-project | L2 regularization penalty applied through optimizer constructor: optim.Adam(..., weight_decay=1e-4). Bridges concept (DEVELOPED from 1.3.7) to concrete PyTorch parameter. |
| Linear -> BatchNorm -> ReLU -> Dropout layer ordering | INTRODUCED | mnist-project | Standard pattern for each hidden layer in a regularized fully-connected network. Output layer has no activation, no dropout, no batch norm. |
| Train/test evaluation loop pattern | DEVELOPED | mnist-project | model.eval() + torch.no_grad() + iterate test_loader + compute accuracy. Three differences from training: no optimizer.zero_grad/backward/step, torch.no_grad() for memory, shuffle=False. |
| torch.argmax for classification predictions | INTRODUCED | mnist-project | argmax(outputs, dim=1) returns index of highest logit = predicted class. How a classification model "picks" its answer. |
| Logits (raw model outputs) | INTRODUCED | mnist-project | The raw, unnormalized output of the final layer before softmax. Can be negative, do not sum to 1, are NOT probabilities. Term used consistently throughout classification context. |
| torchinfo.summary() for model architecture inspection | DEVELOPED | debugging-and-visualization | Pass model + input_size, get every layer's output shape, parameter count, total trainable params. "X-ray" analogy — see inside the model without running data. Catches shape mismatches before training. |
| Gradient magnitude checking (per-layer gradient norms) | DEVELOPED | debugging-and-visualization | Iterate model.named_parameters(), compute param.grad.norm() after backward(). Healthy: all layers within 10-100x of each other. Unhealthy: early layers 100,000x smaller than later layers (vanishing) or any layer >100/NaN (exploding). "Taking the pulse" analogy. |
| log_gradient_norms() helper pattern | DEVELOPED | debugging-and-visualization | Reusable function to print gradient magnitude per named parameter. Called every N iterations inside training loop. Practical implementation of gradient health monitoring. |
| TensorBoard SummaryWriter for training monitoring | INTRODUCED | debugging-and-visualization | torch.utils.tensorboard.SummaryWriter logs scalars (loss, accuracy) per epoch. Two lines of logging code added to training loop. Launches as separate process on localhost:6006. "Flight recorder" analogy. |
| TensorBoard run comparison (multi-run overlay) | INTRODUCED | debugging-and-visualization | Each run logs to a separate directory (runs/experiment_name). TensorBoard overlays all runs on same plot automatically. Killer feature: compare 3 learning rates with zero plotting code. |
| "Loss going down does not mean training is working" diagnostic principle | DEVELOPED | debugging-and-visualization | A model can minimize loss by learning the class prior (always predict most common class) without extracting useful features. Always monitor accuracy alongside loss. Central lesson insight, demonstrated with silent failure hook (loss 2.3→0.5, accuracy stuck at 10%). |
| Systematic debugging checklist (4-phase workflow) | DEVELOPED | debugging-and-visualization | Phase 1: torchinfo before training (shapes). Phase 2: gradient check on first iteration (health). Phase 3: TensorBoard during training (loss + accuracy, train + test). Phase 4: diagnose by symptom (stuck→gradients, NaN→exploding, train-good-test-bad→regularization, loss-down-accuracy-flat→check predictions). |
| "Suspiciously smooth loss curve" as bug signal | INTRODUCED | debugging-and-visualization | Real mini-batch training has noise in the loss curve. A perfectly smooth loss means you are likely training on the same batch every epoch (not iterating DataLoader). Negative example using next(iter(train_loader)) outside the loop. |
| model.named_parameters() for parameter inspection | INTRODUCED | debugging-and-visualization | Variant of model.parameters() that also yields the parameter name as a string. Used in gradient checking to identify which layer has problematic gradients. |

## Per-Lesson Summaries

### datasets-and-dataloaders (Lesson 1) — BUILD

**Concepts taught:** Dataset (__getitem__ + __len__), DataLoader (batch_size, shuffle, iteration), torchvision.transforms (Compose, ToTensor, Normalize), torchvision.datasets.MNIST.

**Mental models established:**
- "Dataset is a menu, DataLoader is the kitchen" — Dataset says what is available and how to get one item; DataLoader batches, shuffles, and serves.
- "shuffle=True is how you get random sampling in practice" — bridges polling analogy from 1.3.4 to the actual API parameter.
- "The training loop body does not change" — reinforces the "same heartbeat, new instruments" model from 2.1.4. Side-by-side comparison shows only the data-feeding line changes.

**Analogies used:**
- Menu/kitchen for Dataset/DataLoader separation of concerns.
- Polling analogy (from 1.3.4) extended to explain shuffle=True.

**How concepts were taught:**
- Dataset: Code-first with SimpleDataset on familiar y=2x+1 data, then lazy loading skeleton for large datasets.
- DataLoader: Wrapped SimpleDataset, iterated and printed shapes, showed iteration count matching ceil(N/B) formula.
- Shuffling: Negative example with SortedDataset (targets sorted low-to-high) showing biased batches without shuffling. Training curve comparison code provided.
- Transforms: torchvision.transforms.Compose with ToTensor + Normalize on MNIST. Emphasized lazy per-sample application.
- Integration: Side-by-side ComparisonRow of old training loop (raw tensors) vs new loop (DataLoader). Loop body identical.
- MNIST: Loaded with torchvision.datasets.MNIST, inspected batch shapes, visualized sample grid with matplotlib.
- Broken Dataset: Negative example showing inconsistent shapes causing collation RuntimeError.

**What is NOT covered:**
- Training MNIST to convergence (deferred to mnist-project)
- Cross-entropy loss, softmax, multi-class classification (deferred to mnist-project)
- Data augmentation strategies (RandomFlip, RandomCrop) — mentioned only
- Advanced DataLoader options (num_workers, pin_memory, custom collate)
- Train/val/test splitting in code
- Writing custom image loading code (pillow internals)

**Checks/exercises:**
- Check 1: Predict iterations per epoch, shuffle behavior, batch shapes (predict-and-verify)
- Quick Check: Parse MNIST batch tensor shape [64, 1, 28, 28] — identify batch size and channel count
- Check 2: Transfer question — advise a colleague using manual index slicing on PIL images (suggest lazy Dataset + transforms + DataLoader)
- Colab notebook: 5 exercises (2 guided, 2 supported, 1 independent) covering custom Dataset, MNIST loading, training loop integration, batch size experiments, and CSV Dataset

### mnist-project (Lesson 2) — STRETCH / PROJECT

**Concepts taught:** Softmax function, cross-entropy loss, accuracy metric, nn.CrossEntropyLoss, nn.Flatten, nn.Dropout, nn.BatchNorm1d, model.train()/model.eval() in practice, weight_decay parameter, train/test evaluation loop, torch.argmax, logits terminology, classification vs regression framing, Linear->BatchNorm->ReLU->Dropout layer ordering.

**Mental models established:**
- "Wrongness score v2" — cross-entropy extends the "wrongness score" framing from MSE; MSE measures distance, cross-entropy measures confidence on the wrong answer.
- "Sigmoid to softmax" — softmax is sigmoid generalized to multiple classes. Sigmoid squashes one value to [0,1], softmax squashes N values to probabilities summing to 1.
- "Loss vs accuracy" — the model optimizes loss (cross-entropy), the human evaluates with accuracy (% correct). They correlate but are not identical.
- "Same heartbeat, new instruments" (reinforced) — the training loop body is identical; cross-entropy replaces MSE, DataLoader replaces raw tensors, but the four-line pattern (forward, loss, backward, update) does not change.

**Analogies used:**
- Cross-entropy as "confidence penalty" — if you are confident and wrong, the penalty is enormous; if you are uncertain, the penalty is mild.
- "Wrongness score v2" extending the MSE wrongness score framing.
- Sigmoid-to-softmax bridge connecting familiar activation functions to the new multi-class context.

**How concepts were taught:**
- Softmax: Formula shown, then one element (class 7, logit 3.8) worked step-by-step through e^3.8 = 44.70, sum = 54.58, divide = 0.82. PyTorch code confirms full vector. Key property: positive, sum to 1.
- Cross-entropy: Motivated by MSE's failure on classification via 10-class ComparisonRow (model confidently wrong: 80% on class 3, 2% on correct class 0). Formula L = -log(p_correct) with GradientCard trio showing concrete values: confident-right (0.05), unsure (0.69), confident-wrong (4.61). 90x difference emphasized.
- nn.CrossEntropyLoss: Code showing drop-in replacement for nn.MSELoss. Warning about not adding softmax before it (double-application bug). Takes raw logits + integer labels.
- Model architecture: Simple 3-layer FC model (784->256->128->10) with nn.Flatten + ReLU. Parameter count calculated (~235K). No activation on output layer.
- Training loop: Complete code with DataLoader integration. Accuracy tracking via torch.argmax + comparison. Expected output: ~97% in 5 epochs.
- Evaluation: Test set with model.eval() + torch.no_grad(). Three differences from training explained.
- Visualization: matplotlib code for correct (green) and incorrect (red) predictions with confidence scores. Observation that errors cluster on genuinely ambiguous digits.
- Improved model: nn.BatchNorm1d + nn.Dropout(p=0.3) + weight_decay=1e-4 in Adam. Split into architecture section (12a) and training section (12b). Side-by-side epoch tables showing scissors pattern closing. Dropout causing test > train explained.
- Checks: (1) Predict-and-verify with logits [2.0, -1.0, 0.5] — which class, correct?, loss high/low? (2) Transfer question about 99.2% train / 96.8% test — connects to scissors pattern and regularization.

**What is NOT covered:**
- Convolutional networks (deferred to Series 3)
- Information-theoretic derivation of cross-entropy
- Advanced data augmentation
- Learning rate schedulers or gradient clipping
- Hyperparameter tuning / search
- Multi-label classification or non-exclusive classes
- Saving/loading models (deferred to Module 2.3)
- GPU training (deferred to Module 2.3)
- Custom loss functions
- Achieving 99%+ accuracy (celebrated 97-98% as appropriate for FC architecture)

**Checks/exercises:**
- Check 1: Predict-and-verify — given logits and true label, predict class, correctness, and loss magnitude
- Check 2: Transfer question — diagnose overfitting from train/test accuracy gap, prescribe regularization
- Colab notebook: 6-step project (load MNIST, build simple model, training loop, evaluate, visualize, build improved model with regularization)

### debugging-and-visualization (Lesson 3) — CONSOLIDATE

**Concepts taught:** torchinfo.summary() for model inspection, gradient magnitude checking (per-layer gradient norms via model.named_parameters()), TensorBoard SummaryWriter for scalar logging and run comparison, systematic 4-phase debugging checklist, "loss going down does not mean training is working" diagnostic principle.

**Mental models established:**
- "torchinfo is an X-ray" — see inside your model's shape flow without running any data through it. Catches plumbing problems (shape mismatches) before training starts.
- "Gradient checking is taking the model's pulse" — healthy pulse has balanced magnitudes across layers; weak pulse (tiny early-layer gradients) means the model is struggling even if it looks alive. Early detection, not just symptom recognition.
- "TensorBoard is a flight recorder" — captures data continuously during training; review it when something goes wrong or when comparing experiments.
- "Debugging is a systematic workflow, not random guessing" — the 4-phase checklist (torchinfo before, gradient check first iteration, TensorBoard during, diagnose by symptom) replaces "stare at code and restart."

**Analogies used:**
- X-ray for torchinfo (medical imaging — see structure before operating).
- Taking the pulse for gradient checking (vital signs during training).
- Flight recorder for TensorBoard (continuous monitoring, reviewed after the fact).
- "Shape errors are plumbing problems, not design flaws" — shape mismatches feel catastrophic but are usually a single mismatched number between adjacent layers.

**How concepts were taught:**
- Hook: "Silent failure" — training run with loss 2.3→0.5 (looks good) but 10% accuracy (random chance). Collapsible reveal. Establishes that loss alone is insufficient.
- torchinfo: Basic usage on MNIST model, then two shape-error examples (missing Flatten, wrong Linear dimensions). Both caught by torchinfo before running data. Aside confirms "plumbing, not design" framing.
- Gradient checking: log_gradient_norms() function iterating model.named_parameters(). ComparisonRow of healthy (balanced ~0.03) vs unhealthy (early layers 100,000x smaller). Connects to telephone game analogy from backpropagation (1.3.1). Remediation guidance in two-column grid (vanishing: ReLU, initialization, batch norm, skip connections; exploding: lower LR, gradient clipping, check data, batch norm).
- TensorBoard: SummaryWriter setup, two logging lines in training loop. TensorBoardMockup component (Recharts dark-themed chart) showing 3-LR comparison. GradientCards explaining each LR behavior. Negative example: same-batch bug with suspiciously smooth loss curve.
- Debugging checklist: 4 PhaseCards synthesizing all tools into a workflow. "Solving the Opening Puzzle" section revisits the hook with all three tools, closing the narrative arc.
- Check 1: Predict-and-verify on shapes and parameter counts (3 questions with collapsible answers).
- Check 2: Transfer question — colleague's model at 52% on binary classification. Expected answer uses all three tools.

**What is NOT covered:**
- Advanced TensorBoard features (histograms, embeddings, graph visualization, profiler)
- Weights & Biases, MLflow, or other experiment tracking platforms
- PyTorch Profiler or performance optimization
- Hyperparameter tuning or search strategies
- Debugging CUDA/GPU-specific errors
- Unit testing for ML models
- Custom logging frameworks
- Advanced torchinfo options

**Checks/exercises:**
- Check 1: Predict-and-verify — predict output shapes, parameter counts, and shape error location for a 3-layer model
- Check 2: Transfer question — diagnose 52% binary classification accuracy (near random chance) using all three tools
- Colab notebook: 6 exercises (2 guided, 2 supported, 1 supported, 1 independent) covering torchinfo verification, shape bug diagnosis, gradient checking implementation, TensorBoard logging, 3-LR comparison, and a capstone "find 3 bugs in a broken script" exercise

## Key Mental Models and Analogies

| Model/Analogy | Established In | Available For |
|---------------|---------------|---------------|
| "Dataset is a menu, DataLoader is the kitchen" | datasets-and-dataloaders | Any future lesson using data pipelines; reinforces separation of concerns |
| "shuffle=True is how you get random sampling in practice" | datasets-and-dataloaders | Any lesson involving training on real data; grounds the 1.3.4 polling analogy in API |
| "The training loop body does not change" (reinforced) | datasets-and-dataloaders (originally 2.1.4) | All future training lessons; DataLoader changes data feeding, not the heartbeat |
| "Wrongness score v2" — cross-entropy as confidence penalty | mnist-project | Any future classification lesson; extends MSE wrongness score framing |
| "Sigmoid to softmax" — generalization to multiple classes | mnist-project | Future multi-class or attention lessons; bridges familiar sigmoid to new contexts |
| "Loss vs accuracy" — model metric vs human metric | mnist-project | All future training/evaluation lessons; always track both |
| "Same heartbeat, new instruments" (reinforced again) | mnist-project (originally 2.1.4) | Proven across regression and classification; the loop pattern is universal |
| "torchinfo is an X-ray" — see inside model without running data | debugging-and-visualization | Any future model debugging; first step in debugging checklist |
| "Gradient checking is taking the pulse" — per-layer gradient health | debugging-and-visualization | Any future training debugging; connects to vanishing/exploding from 1.3 |
| "TensorBoard is a flight recorder" — continuous training monitoring | debugging-and-visualization | Any future training run; run comparison for hyperparameter experiments |
| "Debugging is a systematic workflow, not random guessing" — 4-phase checklist | debugging-and-visualization | All future training; replaces ad-hoc debugging; the workflow students will use going forward |
| "Loss going down does not mean training is working" | debugging-and-visualization | All future training evaluation; always track accuracy alongside loss |
| "Shape errors are plumbing problems, not design flaws" | debugging-and-visualization | Future architecture debugging; reduces panic when shape errors occur |
