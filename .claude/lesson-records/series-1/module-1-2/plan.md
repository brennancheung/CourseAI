# Module 1.2: From Linear to Neural

**Status:** Complete (4 lessons built)

## Module Goal

Show why linear models are insufficient, introduce nonlinearity via activation functions, and demonstrate how neural networks can solve problems linear models cannot (XOR).

## Narrative Arc

This module follows a deliberate "problem → solution" structure. First, show that a neuron is just linear regression extended to multiple inputs (lesson 1). Then demonstrate the fundamental limitation: stacking linear layers is still linear (lesson 1, proven with XOR in lesson 2). Then provide the solution: activation functions (lesson 3). Finally, give a reference guide to all activations (lesson 4). The XOR problem is the through-line — can't solve it → can solve it.

## Lesson Sequence

| # | Lesson | Core Concept | Type | Rationale for Position |
|---|--------|-------------|------|----------------------|
| 1 | neuron-basics | Neuron = weighted sum + bias, layers, networks, linear collapse | STRETCH | Must establish what neurons ARE before showing their limitations |
| 2 | limits-of-linearity | Linear separability, XOR impossibility | BUILD | Concrete demonstration of the limitation stated in lesson 1 |
| 3 | activation-functions | Activation = nonlinearity, thresholds, XOR solved | STRETCH | The solution — must come AFTER the problem is felt |
| 4 | activation-functions-deep-dive | Visual reference to all activations | CONSOLIDATE | Reference material, low cognitive demand |

## Concept Distribution

- neuron-basics: neuron formula, layer, network, hidden layer, linear collapse
- limits-of-linearity: linear separability, decision boundary, XOR impossibility
- activation-functions: activation function concept, sigmoid, ReLU, thresholds, space transformation
- activation-functions-deep-dive: tanh, Leaky ReLU, GELU, Swish, decision guide

## Cognitive Load Trajectory

STRETCH → BUILD → STRETCH → CONSOLIDATE

Clean trajectory — no adjacent STRETCH lessons.

## Module-Level Misconceptions

| Misconception | Progression |
|---------------|------------|
| "Neurons are mysterious/brain-like" | Addressed in lesson 1 ("it's just linear regression!") |
| "More layers = more power (always)" | Introduced in lesson 1 (collapse), demonstrated in lesson 2 (XOR), resolved in lesson 3 (need activation) |
| "Networks draw many lines in input space" | Natural intuition from lesson 2, corrected in lesson 3 ("networks TRANSFORM the space") |
