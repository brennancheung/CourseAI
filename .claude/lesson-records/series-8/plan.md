# Series 8: Special Topics

## Goal

Standalone deep dives into interesting models, techniques, and ideas that don't belong to a structured series — giving the student a place to explore any topic with the same rigor as the core curriculum.

## Structure

This series has no fixed scope. Modules are added as needed, grouped loosely by domain. Lessons within a module are independent — each is self-contained with its own prerequisites stated inline.

## Module Breakdown

| Module | Title | Lessons | Focus |
|--------|-------|---------|-------|
| 8.1 | Vision & Vision-Language Models | 2 (complete) | Modern vision architectures and vision-language models |
| 8.2 | Safety & Content Moderation | 1 (complete) | How production image generation systems prevent harmful content — the multi-layered safety stack |

Future modules can cover any domain: reinforcement learning, audio models, efficiency techniques, emerging architectures, etc. Add a new module when a topic doesn't fit an existing one.

## Lesson Titles

### Module 8.1: Vision & Vision-Language Models

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | siglip-2 | SigLIP 2 | Sigmoid loss for vision-language pretraining — how SigLIP removes the need for global softmax in contrastive learning |
| 2 | sam-3 | SAM 3 | Segment Anything Model — promptable segmentation, the vision foundation model approach |

### Module 8.2: Safety & Content Moderation

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | image-generation-safety | How Image Generation Safety Works | The multi-layered safety stack — prompt filtering, inference-time guidance, CLIP-based image classification, and model-level concept erasure |

## Scope Boundaries

- **In scope:** Anything interesting. No domain restrictions.
- **Out of scope:** Nothing permanently — if it's worth a lesson, it belongs here.
- **Constraint:** Each lesson must be self-contained. State prerequisites explicitly since there's no guaranteed ordering. Brief recap sections are expected to be heavier than in structured series.

## Connections to Prior Series

Lessons draw on the full course as needed. Module 8.1 specifically builds on:
- Series 3 (CNNs) — convolutional architectures, feature maps, spatial reasoning
- Series 4 (LLMs & Transformers) — attention mechanism, transformer architecture
- Series 6-7 (Diffusion) — U-Net architecture, conditioning mechanisms

Module 8.2 builds on:
- Series 6 (Stable Diffusion) — CLIP embeddings, classifier-free guidance, cross-attention conditioning, U-Net architecture
- Series 7 (Post-SD) — full diffusion pipeline understanding
- Series 3 (CNNs) — image classification as the basis for NSFW classifiers
- Series 4 (LLMs) — text classification concepts, transformer encoders
