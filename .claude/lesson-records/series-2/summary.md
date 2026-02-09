# Series 2: PyTorch — Summary

**Status:** In Progress (Module 2.1: complete, Module 2.2: complete, Module 2.3: planned)

## Series Goal

Bridge theory to practice. The student understands backprop, optimizers, and training dynamics conceptually from Series 1 — now they build and train real models in PyTorch. Every lesson has a Colab notebook. The emotional arc: "I already understand this" -> "PyTorch automates what I did manually" -> "I can build real things now."

## Rolled-Up Concept List

### From Module 2.1: PyTorch Core (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| PyTorch tensor creation API | DEVELOPED | `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`, `torch.from_numpy()` — mapped to NumPy equivalents |
| Tensor attributes: shape, dtype, device | DEVELOPED | "Three attributes that determine everything"; debugging first instinct |
| Tensor reshaping | INTRODUCED | `view()` vs `reshape()` — use reshape() by default |
| Element-wise arithmetic on tensors | DEVELOPED | Same as NumPy; `*` (element-wise) vs `@` (matmul) distinction |
| Matrix multiplication (`@` operator) | DEVELOPED | Shorthand for `torch.matmul()`; forward pass `y_hat = X @ w + b` |
| PyTorch float32 default | DEVELOPED | "Rough sketch with a micrometer" — float64 is overkill for approximate gradients |
| PyTorch dtype system | INTRODUCED | float32, float64, float16, int64, bool with use cases |
| GPU as parallel processor | INTRODUCED | CPU (few fast cores) vs GPU (thousands of simple cores) |
| Device management | INTRODUCED | `.to(device)`, `.cpu()`, standard device pattern |
| GPU transfer overhead | INTRODUCED | Small tensors faster on CPU; empirical timing comparison |
| NumPy-PyTorch interop | DEVELOPED | Shared memory (`from_numpy`, `.numpy()`) vs copying (`torch.tensor()`) |
| `requires_grad` flag | DEVELOPED | "Press Record" — tells PyTorch to track operations on a tensor |
| `backward()` method | DEVELOPED | "Press Rewind" — walks computational graph backward applying chain rule |
| `.grad` attribute | DEVELOPED | Gradient results stored on leaf tensors (parameters), not returned |
| Gradient accumulation / `zero_grad()` | DEVELOPED | Gradients add by default; must clear before each backward pass; #1 beginner bug |
| `torch.no_grad()` context manager | DEVELOPED | Pauses recording; semantically necessary during parameter updates |
| `.detach()` method | DEVELOPED | Severs tensor from graph; "snip the tape" |
| Autograd as computational graph traversal | DEVELOPED | backward() performs the same traversal done by hand in Series 1.3 |
| nn.Module subclass pattern | DEVELOPED | __init__ (what layers) + forward() (how data flows); auto-tracks parameters |
| nn.Linear as w*x + b | DEVELOPED | Not mysterious — IS the matrix multiply the student knows; requires_grad automatic |
| model.parameters() | DEVELOPED | Collects all learnable tensors; model.zero_grad() clears all gradients at once |
| nn.Sequential | INTRODUCED | Convenience for simple layer stacks; cannot express skip connections or branching |
| nn.ReLU as a module | INTRODUCED | Same max(0,x), wrapped as a module for Sequential use |
| Skip/residual connection pattern | INTRODUCED | Custom forward() with bypass path; previews ResNets |
| nn.MSELoss (loss function object) | DEVELOPED | Stateless callable wrapping the MSE formula; `criterion(y_hat, y)` = `((y_hat - y)**2).mean()` |
| torch.optim.SGD (optimizer object) | DEVELOPED | Wraps gradient descent update rule; `optimizer.step()` = `param -= lr * grad` for all parameters |
| torch.optim.Adam (optimizer object) | DEVELOPED | Momentum + RMSProp + bias correction; defaults lr=0.001, betas=(0.9, 0.999); one-line swap from SGD |
| optimizer.step() / optimizer.zero_grad() | DEVELOPED | Step performs update rule; zero_grad clears all parameter gradients; canonical order: clear, compute, use |
| Complete PyTorch training loop | DEVELOPED | forward -> loss -> backward -> update; same pattern from Series 1 with PyTorch instruments |
| Uniform optimizer interface | DEVELOPED | All torch.optim optimizers share .step() and .zero_grad(); algorithm encapsulated; swap freely |
| model.train() / model.eval() | MENTIONED | Named for dropout/batch norm context; not practiced |

### From Module 2.2: Real Data (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| torch.utils.data.Dataset abstraction | DEVELOPED | `__len__` returns count, `__getitem__` returns (input, target) by index. Custom subclass pattern. |
| torch.utils.data.DataLoader | DEVELOPED | Wraps Dataset; handles batching, shuffling, iteration. Yields tensors with batch dimension prepended. |
| Dataset/DataLoader separation of concerns | DEVELOPED | Dataset = data access (one sample). DataLoader = logistics (batching, shuffling). "Menu/kitchen" analogy. |
| torchvision.transforms pipeline | INTRODUCED | Compose(), ToTensor(), Normalize(). Per-sample in __getitem__. |
| torchvision.datasets (pre-built datasets) | INTRODUCED | MNIST loaded via datasets.MNIST(root, train, download, transform). |
| shuffle=True as random sampling implementation | DEVELOPED | Bridges polling analogy (1.3.4) to practice; re-shuffles indices each epoch. |
| Softmax function | INTRODUCED | Converts logits to probabilities (positive, sum to 1). "Sigmoid generalized to multiple classes." |
| Cross-entropy loss | INTRODUCED | L = -log(p_correct). "Confidence penalty" — confident-wrong costs 90x more than confident-right. |
| nn.CrossEntropyLoss (PyTorch API) | INTRODUCED | Takes raw logits + integer labels. Applies log-softmax internally. Do NOT add softmax before it. |
| Accuracy metric (% correct) | INTRODUCED | torch.argmax(outputs, dim=1) for predicted class. Human metric vs model metric (loss). |
| nn.Flatten() module | INTRODUCED | Flattens [B, 1, 28, 28] to [B, 784] for fully-connected layers. |
| nn.Dropout(p) as PyTorch module | INTRODUCED | Bridges concept (1.3.7) to API. Active in train(), disabled in eval(). |
| nn.BatchNorm1d(features) as PyTorch module | INTRODUCED | Bridges concept (1.3.6) to API. Batch stats in train(), running averages in eval(). |
| model.train() / model.eval() in practice | DEVELOPED | Elevated from MENTIONED (2.1.4). Dropout + batch norm behavior changes between modes. |
| weight_decay in optimizer | INTRODUCED | L2 regularization via optimizer constructor: optim.Adam(..., weight_decay=1e-4). |
| Train/test evaluation loop pattern | DEVELOPED | model.eval() + no_grad() + iterate test_loader + compute accuracy. |
| torchinfo.summary() for model inspection | DEVELOPED | Layer output shapes, parameter counts, total params. "X-ray" before training. Catches shape mismatches. |
| Gradient magnitude checking (per-layer norms) | DEVELOPED | model.named_parameters() + param.grad.norm(). Detects vanishing/exploding before symptoms appear. |
| TensorBoard SummaryWriter | INTRODUCED | Two lines of logging; live dashboard with run comparison. "Flight recorder" analogy. |
| Systematic 4-phase debugging checklist | DEVELOPED | torchinfo before, gradient check first iteration, TensorBoard during, diagnose by symptom. |
| "Loss going down ≠ training working" principle | DEVELOPED | Always monitor accuracy alongside loss. Silent failures (class prior learning) are real. |

## Key Mental Models Carried Forward

1. **"Tensors are NumPy arrays that know where they live"** — Core framing for PyTorch's data structure
2. **"Same interface, different engine"** — PyTorch mirrors NumPy; the new capabilities are device management and autograd
3. **"Shape, dtype, device — check these first"** — The debugging trinity for PyTorch
4. **"Rough sketch with a micrometer"** — Why float32 is right-sized for deep learning
5. **"GPU wins at scale, CPU wins for small operations"** — Transfer overhead determines the crossover
6. **"requires_grad = press Record / backward() = press Rewind"** — Unified recording metaphor for autograd API
7. **"Not magic — automation"** — Autograd automates the same algorithm the student did manually in Series 1.3
8. **"Same math, better organization"** — nn.Module packages known computation into reusable building blocks
9. **"nn.Linear IS w*x + b"** — Layer abstractions are not mysterious; same formula, better packaging
10. **"Three representations of the same network"** — Math (1.3) -> raw tensors (autograd) -> nn.Module; same numbers at every abstraction level
11. **"Same heartbeat, new instruments"** — The training loop pattern is identical to Series 1; PyTorch objects are instruments for each beat
12. **"Of course it works"** — Every piece was individually verified; the training loop runs verified pieces in sequence
13. **"Clear, compute, use"** — Canonical gradient lifecycle: zero_grad, backward, step
14. **"Dataset is a menu, DataLoader is the kitchen"** — Dataset says what is available; DataLoader batches, shuffles, and serves
15. **"shuffle=True is how you get random sampling in practice"** — Bridges polling analogy from 1.3.4 to the actual API parameter
16. **"Wrongness score v2"** — Cross-entropy as confidence penalty; extends MSE wrongness score framing from Series 1
17. **"Sigmoid to softmax"** — Softmax is sigmoid generalized to multiple classes
18. **"Loss vs accuracy"** — Model optimizes loss, human evaluates with accuracy; they correlate but are not identical
19. **"torchinfo is an X-ray"** — See inside the model without running data; first step in debugging checklist
20. **"Gradient checking is taking the pulse"** — Per-layer gradient health; early detection before symptoms appear
21. **"TensorBoard is a flight recorder"** — Continuous monitoring; review when things go wrong
22. **"Debugging is a systematic workflow, not random guessing"** — 4-phase checklist replaces ad-hoc debugging
23. **"Loss going down does not mean training is working"** — Always track accuracy alongside loss
24. **"Shape errors are plumbing problems, not design flaws"** — Reduces panic; usually a single mismatched number

## What This Series Does NOT Cover (Yet)

- Saving/loading models (Module 2.3)
- GPU training patterns (Module 2.3)
