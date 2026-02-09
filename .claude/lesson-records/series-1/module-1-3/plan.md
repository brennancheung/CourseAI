# Module 1.3: Training Neural Networks

**Status:** In progress (6 of 7 lessons built)

## Module Goal

Teach the student how neural networks actually learn—from understanding backpropagation conceptually, through computing gradients by hand on a real network, to practical training skills (batching, optimizers, regularization) that make training work in practice.

## Narrative Arc

The student enters this module knowing *what* neural networks are (Module 1.2) and *how* single-parameter optimization works (Module 1.1). The gap: they've never trained a neural network. This module bridges that gap in three acts:

**Act 1 — The Algorithm (lessons 1-3):** Understand backpropagation conceptually (lesson 1, built), then make it concrete by computing every gradient by hand through a real 2-layer network (lesson 2), then see the visual framework that makes the computation manageable: computational graphs (lesson 3).

**Act 2 — Making It Work (lessons 4-5):** Move from theory to practice. How does training actually work when you have thousands of data points? Batching and stochastic gradient descent (lesson 4). Why does vanilla SGD struggle, and what do modern optimizers like Adam actually do? (lesson 5).

**Act 3 — Making It Work Well (lessons 6-7):** The "dark art" of training. Why do networks fail to train? Vanishing/exploding gradients and initialization (lesson 6). Why do networks memorize instead of generalize? Overfitting and regularization (lesson 7).

## Lesson Sequence

| # | Slug | Core Concept | Type | Rationale for Position | Status |
|---|------|-------------|------|----------------------|--------|
| 1 | backpropagation | Chain rule, forward/backward pass concept, efficiency | STRETCH | Foundation: what backprop IS, conceptually | Built |
| 2 | backprop-worked-example | Multi-layer backprop with concrete numbers | BUILD | Makes lesson 1 concrete; the "aha" of seeing real numbers flow | Built |
| 3 | computational-graphs | Visual framework for derivatives in networks | BUILD | Gives a visual tool that organizes what lesson 2 did by hand | Built |
| 4 | batching-and-sgd | Mini-batches, stochastic gradient descent, epochs | STRETCH | Pivots from "how" to "practice"; new concepts but builds on training loop | Built |
| 5 | optimizers | Momentum, RMSProp, Adam | BUILD | Extends SGD with practical improvements; uses intuition from lesson 4 | Built |
| 6 | training-dynamics | Vanishing/exploding gradients, initialization, batch norm | STRETCH | Why training fails; connects back to activation functions (1.2) | Built |
| 7 | overfitting-and-regularization | Dropout, weight decay, early stopping | CONSOLIDATE | Capstone: when to stop training, preventing memorization | Not started |

## Concept Distribution

| Concept | Lesson | New or Reinforced |
|---------|--------|-------------------|
| Chain rule for composed functions | 1 | New (DEVELOPED) |
| Forward/backward pass | 1 | New (INTRODUCED) |
| Local derivatives | 1 | New (INTRODUCED) |
| Backprop efficiency | 1 | New (INTRODUCED) |
| Multi-layer gradient computation | 2 | New (DEVELOPED) |
| Weight update with real numbers | 2 | New (DEVELOPED) |
| Forward pass (concrete) | 2 | Reinforced: INTRODUCED -> DEVELOPED |
| Backward pass (concrete) | 2 | Reinforced: INTRODUCED -> DEVELOPED |
| Computational graph notation | 3 | New (DEVELOPED) |
| Automatic differentiation (concept) | 3 | Reinforced: MENTIONED -> INTRODUCED |
| Mini-batches | 4 | New (DEVELOPED) |
| Stochastic gradient descent | 4 | New (DEVELOPED) |
| Epochs | 4 | New (DEVELOPED) |
| Gradient noise (helpful) | 4 | New (INTRODUCED) |
| Momentum | 5 | New (DEVELOPED) |
| Adam optimizer | 5 | New (DEVELOPED) |
| Learning rate (revisited for Adam) | 5 | Reinforced |
| Vanishing gradients (deep) | 6 | Reinforced: INTRODUCED -> DEVELOPED |
| Exploding gradients | 6 | New (DEVELOPED) |
| Weight initialization strategies | 6 | New (DEVELOPED) |
| Batch normalization | 6 | New (INTRODUCED) |
| Overfitting (revisited) | 7 | Reinforced from 1.1 |
| Dropout | 7 | New (DEVELOPED) |
| Weight decay / L2 regularization | 7 | New (DEVELOPED) |
| Early stopping | 7 | New (DEVELOPED) |

## Cognitive Load Trajectory

```
Lesson:   1         2         3         4         5         6         7
Type:     STRETCH   BUILD     BUILD     STRETCH   BUILD     STRETCH   CONSOLIDATE
New:      3         2         1         3         2         3         3
          ▓▓▓       ▓▓░       ▓░░       ▓▓▓       ▓▓░       ▓▓▓       ▓▓▓
```

No two STRETCH lessons are adjacent. The pattern is S-B-B-S-B-S-C, giving the student room to breathe after each challenging lesson.

Note: Lesson 7 has 3 new concepts but is CONSOLIDATE because dropout/weight-decay/early-stopping are independently simple techniques—each is a single idea applied to a familiar context (overfitting, which was established in Module 1.1). The lesson's cognitive work is applying known patterns, not understanding new ones.

## Module-Level Misconceptions

| Misconception | Progression |
|---------------|------------|
| "Backprop is a different algorithm from gradient descent" | Addressed in lesson 1: backprop COMPUTES gradients, GD USES them |
| "Each weight needs to know the full network" | Addressed in lessons 1-2: each layer only needs LOCAL derivatives |
| "Backprop is an approximation" | Addressed in lesson 2: exact gradients computed with real numbers |
| "More data always helps / batch size doesn't matter" | Addressed in lesson 4: mini-batch tradeoffs |
| "Adam always beats SGD" | Addressed in lesson 5: no free lunch, different optimizers for different situations |
| "Vanishing gradients = gradients are zero" | Addressed in lesson 6: near-zero, accumulates over many layers |
| "Regularization is just for small datasets" | Addressed in lesson 7: even large models need regularization |
