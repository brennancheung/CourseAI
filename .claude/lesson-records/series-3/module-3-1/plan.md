# Module 3.1: Convolutions — Plan

## Module Goal

The student can explain what a convolutional layer computes, why spatial structure matters for images, and build a working CNN in PyTorch that outperforms a dense network on image data.

## Narrative Arc

**Beginning (the problem):** Dense networks treat every pixel as independent — a 28x28 image becomes a flat vector of 784 numbers, destroying all spatial relationships. A pixel's meaning depends on its neighbors (edges, textures, shapes), but dense layers cannot see neighborhoods. Worse, dense networks need separate weights for every pixel position, so a cat in the top-left requires entirely different weights than the same cat shifted right — they cannot share what they learn across positions.

**Middle (building the tools):** Convolutions solve both problems at once. A small filter slides across the image, looking at local neighborhoods and sharing the same weights everywhere. We start by computing convolutions by hand on tiny grids to build intuition for what filters detect (edges, gradients). Then we assemble these pieces into the CNN pattern: convolution extracts features, pooling shrinks spatial dimensions, and fully-connected layers at the end make decisions. We learn the engineering knobs (stride, padding, pooling) that control how the network processes spatial information.

**End (the payoff):** The student builds a CNN for MNIST and sees it dramatically outperform the dense network from Series 2. The improvement is not from more parameters or longer training — it is from architecture that respects the structure of the data. This is the first time the student sees that *how you connect layers* matters as much as *how many neurons you have*.

## Lesson Sequence

| # | Slug | Core Concepts | Type | Rationale for Position |
|---|------|--------------|------|----------------------|
| 1 | what-convolutions-compute | Filters, feature maps, edge detection, spatial structure | STRETCH | Must come first: establishes what convolutions are and why they exist; connects to known concepts (weights, linear combinations) |
| 2 | building-a-cnn | Pooling, stride, padding, the conv-pool-fc pattern | BUILD | Uses convolution understanding from L1; adds engineering controls; assembles pieces into architecture |
| 3 | mnist-cnn-project | End-to-end CNN implementation, architecture choices | CONSOLIDATE | Applies everything from L1+L2 in a real project; proves the payoff vs dense networks |

## Rough Topic Allocation

| Lesson | Topics | Notes |
|--------|--------|-------|
| what-convolutions-compute | Convolution as sliding filter, feature maps, edge detection, spatial structure, weight sharing | Core "what is a convolution" lesson |
| building-a-cnn | Pooling, stride, padding, conv-pool-fc pattern, `nn.Conv2d` | Engineering knobs + assembling the architecture |
| mnist-cnn-project | CNN vs dense comparison, architecture design decisions, feature hierarchies | Project — applies L1+L2, no new concepts |

## Cognitive Load Trajectory

```
Lesson 1 (what-convolutions-compute): STRETCH      — New operation, new spatial reasoning
Lesson 2 (building-a-cnn):            BUILD        — Applies convolutions, adds engineering knobs
Lesson 3 (mnist-cnn-project):         CONSOLIDATE  — Project, no new concepts
```

STRETCH-BUILD-CONSOLIDATE. Ideal ramp for a short module.

## Module-Level Misconceptions

- **"Convolution is just a smaller dense layer"** — Both compute weighted sums, but convolutions share weights across positions. Key difference to highlight.
- **"More filters = better"** — Students know "more parameters = more capacity" from Foundations. Filter count is a design choice with tradeoffs.
- **"The filter values are designed by hand"** — Edge detection examples use hand-crafted filters. Must clarify the network LEARNS its own filters via backprop.
- **"Pooling is just throwing away information"** — Max pooling discards values, but it provides spatial invariance.
- **"CNNs only work on images"** — All examples here are images. Worth a brief mention of 1D convolutions for sequences.
