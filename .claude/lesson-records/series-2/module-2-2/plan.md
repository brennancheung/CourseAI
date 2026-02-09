# Module 2.2: Real Data — Plan

**Status:** Planning (0 of 3 built)

## Module Goal

The student can load, transform, batch, and iterate over real datasets in PyTorch, then build, train, evaluate, and debug a complete model on MNIST — their first end-to-end project on real data.

## Narrative Arc

Module 2.1 gave the student the full PyTorch training loop, but on toy synthetic data — hand-crafted tensors with a few dozen points. The student can write `model(x)`, `loss.backward()`, `optimizer.step()`, but has never fed real data through that loop. The gap: "I can train on fake data, but how do I actually use a dataset?"

Module 2.2 closes that gap in three steps:

1. **Datasets and DataLoaders** (the plumbing) — Real data doesn't fit in one tensor. It needs loading, transforming, batching, and shuffling. PyTorch's Dataset/DataLoader pattern solves this cleanly. The student already knows WHY batching matters from Series 1.3 (SGD lesson) — now they learn HOW to batch in practice. This is the "missing piece" between their training loop and real data.
2. **MNIST Project** (the payoff) — Everything comes together. Load real images, build a model, train it, evaluate it, see actual predictions. This is the student's first "I trained a real model" moment. Dropout, batch norm, and cross-entropy (all concepts from Series 1 that haven't been implemented in PyTorch yet) get their first practical use here.
3. **Debugging and Visualization** (the safety net) — Real training goes wrong. Shape mismatches, gradient issues, silent bugs. This lesson gives the student diagnostic tools (torchinfo, gradient checking, TensorBoard) so they can independently troubleshoot when things break.

The emotional arc: "My training loop works on fake data" -> "Now it works on real images" -> "And I know how to fix it when it doesn't."

## Lesson Sequence

| # | Slug | Core Concept | Type | Rationale for Position |
|---|------|-------------|------|----------------------|
| 1 | datasets-and-dataloaders | PyTorch Dataset/DataLoader pattern for feeding batched data into training loops | BUILD | Must come before MNIST project — the project needs DataLoader to function; connects batching theory (1.3) to practice; extends the training loop from 2.1 |
| 2 | mnist-project | End-to-end model training on real data (MNIST) | STRETCH | Integrates everything: DataLoader (lesson 1) + nn.Module + training loop + new concepts (cross-entropy, softmax); first real project; must come after DataLoader is available |
| 3 | debugging-and-visualization | Diagnostic tools for training failures | CONSOLIDATE | After the student has trained a real model and potentially hit real bugs; debugging is reflection, not new concepts; wraps the module with practical independence |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| datasets-and-dataloaders | `torch.utils.data.Dataset`, `DataLoader`, batching, shuffling, `torchvision.transforms`, `torchvision.datasets` | Connects directly to mini-batch SGD from 1.3.4; the "why" is already established, this is the "how" |
| mnist-project | End-to-end pipeline: load data, build model (with dropout + batch norm), cross-entropy loss, train, evaluate, visualize predictions | First real project; cross-entropy/softmax are new concepts but heavily supported by loss function intuition from 1.1 |
| debugging-and-visualization | Shape error diagnosis, `torchinfo.summary()`, gradient magnitude checking, TensorBoard basics, common failure patterns | Practical tools, not theoretical concepts; builds debugging instincts |

## Cognitive Load Trajectory

```
Lesson 1 (datasets-and-dataloaders): BUILD        — Familiar batching concept, new API
Lesson 2 (mnist-project):            STRETCH      — Integration project, 2-3 new concepts
Lesson 3 (debugging-and-visualization): CONSOLIDATE — Tools and patterns, no new theory
```

BUILD -> STRETCH -> CONSOLIDATE. The STRETCH lesson (MNIST project) is the module's central payoff. It introduces new concepts (cross-entropy, softmax, multi-class classification) but in the context of a concrete, motivating project. The CONSOLIDATE lesson gives breathing room after the project.

## Module-Level Misconceptions

- **"DataLoader is just a for loop"** — Students may not see why they need a special object when they could just slice tensors. DataLoader handles shuffling, batching, parallel loading, and drop_last — things a naive for loop gets wrong.
- **"Batch size doesn't matter much"** — The student saw this conceptually in SGD (1.3); now they need to internalize it practically — batch size affects memory usage, training speed, AND generalization.
- **"The model should get 99%+ accuracy immediately"** — MNIST is "easy" by ML standards but a first model won't be perfect. Setting expectations for iterative improvement.
- **"Shape errors mean I did something fundamentally wrong"** — Shape mismatches are the most common PyTorch error and almost always fixable; they indicate a plumbing problem, not a conceptual one.
