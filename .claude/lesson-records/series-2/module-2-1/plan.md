# Module 2.1: PyTorch Core — Plan

**Status:** In Progress (1 of 4 built)

## Module Goal

The student can build, train, and debug a neural network in PyTorch by understanding tensors, autograd, nn.Module, and the training loop — translating every concept from Series 1 into working code.

## Narrative Arc

Series 1 gave the student the complete conceptual toolkit: they understand gradient descent, backpropagation, computational graphs, optimizers, and regularization. But they built everything in NumPy or explored it through interactive widgets. The gap is: "I understand HOW this works, but I've never actually trained a real model."

Module 2.1 closes that gap by mapping every Foundations concept to its PyTorch equivalent. The arc is bottom-up:

1. **Tensors** (the data) — Start where the student is comfortable (NumPy) and show how PyTorch tensors are the same idea with GPU superpowers. Low activation energy: "you already know this."
2. **Autograd** (the gradients) — The student hand-computed gradients in 1.3. Now PyTorch does it automatically. Connect `loss.backward()` directly to the computational graph lesson. The "aha" moment: all that manual work from backprop-worked-example happens in one line.
3. **nn.Module** (the model) — Package neurons, layers, and parameters into reusable building blocks. The student defined neurons mathematically in 1.2 — now they define them in code.
4. **Training Loop** (putting it together) — Reimplement the same linear regression from lesson 1.1 and the same neural net concepts from 1.2, but in PyTorch. Full circle: the universal training loop pattern (forward -> loss -> backward -> update) is the same, just expressed in PyTorch.

The emotional arc: "I already understand this" -> "Oh, PyTorch just automates what I did manually" -> "I can build real things now." Every lesson should feel like recognition, not new learning. The NEW learning is the API — the concepts are review.

## Lesson Sequence

| # | Slug | Core Concept | Type | Rationale for Position |
|---|------|-------------|------|----------------------|
| 1 | tensors | PyTorch tensors as the fundamental data structure | BUILD | Familiar ground (NumPy arrays); low activation energy start; everything else depends on tensors | **Built** |
| 2 | autograd | Automatic differentiation via PyTorch's autograd | STRETCH | New API concept (requires_grad, backward); connects to computational graphs (1.3); must come before nn.Module |
| 3 | nn-module | Building networks with nn.Module, Parameters, layers | BUILD | Applies autograd knowledge; packages neurons (1.2) into code; must come before training loop |
| 4 | training-loop | Complete training loop in PyTorch with optim | CONSOLIDATE | Integrates all three prior lessons; reimplements Series 1 concepts; capstone for module |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| tensors | Tensor creation, manipulation, GPU placement | Familiar ground — NumPy bridge |
| autograd | `requires_grad`, `backward()`, `.grad`, `no_grad()`, `zero_grad()` | Connects to computational graphs (1.3) |
| nn-module | `nn.Module`, `nn.Linear`, `nn.Sequential`, `model.parameters()` | Packages neurons (1.2) into code |
| training-loop | `torch.optim`, loss function objects, the PyTorch training idiom | Reimplements Series 1 concepts in PyTorch |

Each lesson's "new" content is the PyTorch API — the underlying concepts are review from Series 1.

## Cognitive Load Trajectory

```
Lesson 1 (tensors):       BUILD        — Familiar territory with new syntax
Lesson 2 (autograd):      STRETCH      — New mechanism, requires linking to theory
Lesson 3 (nn-module):     BUILD        — Applies autograd, packages known concepts
Lesson 4 (training-loop): CONSOLIDATE  — Integrates everything, reimplements known tasks
```

BUILD -> STRETCH -> BUILD -> CONSOLIDATE. The STRETCH lesson (autograd) is sandwiched between familiar territory.

## Module-Level Misconceptions

- **"PyTorch is doing something different from what I learned"** — The API looks different from the math. Each lesson should explicitly map PyTorch to its Foundations equivalent.
- **"`loss.backward()` replaces understanding backprop"** — Autograd automates what they did manually — it doesn't replace the understanding.
- **"More PyTorch API = better code"** — Software engineers tend toward abstraction. Show explicit operations before convenience wrappers.
- **"`optimizer.zero_grad()` is just boilerplate"** — Gradients accumulate by default. This isn't ceremony.
