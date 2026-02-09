# Module 1.1: The Learning Problem

**Status:** Complete (6 lessons built)

## Module Goal

Teach the student what machine learning is (function approximation), how to measure model quality (loss functions), and how to optimize (gradient descent) — culminating in a from-scratch implementation.

## Narrative Arc

The module tells a single story: "What is ML? → What's the simplest model? → How do we measure if it's good? → How do we make it better? → How do we set the speed? → Build it yourself." Each lesson answers a question that the previous lesson raised.

## Lesson Sequence

| # | Lesson | Core Concept | Type | Rationale for Position |
|---|--------|-------------|------|----------------------|
| 1 | what-is-learning | Function approximation, generalization | STRETCH | First lesson — establishes the framing for everything |
| 2 | linear-regression | Parameters, fitting, y-hat = wx + b | BUILD | Simplest concrete model to apply the framing |
| 3 | loss-functions | Residuals, MSE, loss landscape | STRETCH | "How do we measure goodness?" — answers question from lesson 2 |
| 4 | gradient-descent | Gradient, update rule | STRETCH | "How do we find the minimum?" — answers question from lesson 3 |
| 5 | learning-rate | Hyperparameters, LR failure modes | BUILD | Deepens the most important knob from lesson 4 |
| 6 | implementing-linear-regression | Training loop, from-scratch code | CONSOLIDATE | Integration — puts all pieces together in code |

## Concept Distribution

- what-is-learning: function approximation, generalization, memorization, bias-variance, train/val/test
- linear-regression: parameters, weight, bias, fitting, ML notation (y-hat, w, b)
- loss-functions: residuals, MSE, loss landscape
- gradient-descent: gradient, derivatives, update rule, learning rate (preview)
- learning-rate: hyperparameters, LR failure modes, schedules (preview)
- implementing-linear-regression: training loop, gradient formulas, NumPy implementation

## Cognitive Load Trajectory

STRETCH → BUILD → STRETCH → STRETCH → BUILD → CONSOLIDATE

Note: lessons 3-4 are back-to-back STRETCH. Acceptable because lesson 3 (loss) is about measurement (passive) while lesson 4 (gradient descent) is about action (doing something with the measurement). Different cognitive modes.

## Module-Level Misconceptions

| Misconception | Progression |
|---------------|------------|
| "More complex = better" | Introduced in lesson 1 (bias-variance), revisited in lesson 5 (LR too high) |
| "Visual estimation is sufficient" | Raised in lesson 2, resolved in lesson 3 (MSE) |
| "Gradient descent always finds THE best answer" | Previewed in lesson 4 (convexity note), deepened in later modules |
