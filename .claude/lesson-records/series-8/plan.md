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
| 8.3 | Architecture Analysis | 1 (active) | Reverse-engineering production AI systems by reasoning from observable behavior, disclosed fragments, and published precedents |
| 8.4 | Image Generation Landscape | 1 (complete) | Survey of open weight image generation models -- architecture taxonomy, historical evolution, and the tools to navigate new releases |
| 8.5 | Preference Optimization Deep Dives | 1 (complete) | Mathematical deep dives into preference optimization techniques -- derivations, loss functions, and implementations |

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

### Module 8.3: Architecture Analysis

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | nano-banana-pro | Nano Banana Pro | Speculative architecture analysis of Google's Gemini 3 Pro Image — autoregressive image generation, discrete visual tokenization, and reasoning-guided synthesis |

### Module 8.4: Image Generation Landscape

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | open-weight-image-gen | Open Weight Image Generation Models | Survey of the open weight image generation ecosystem -- architecture taxonomy, historical timeline, key innovations per model, and a framework for placing new releases |

### Module 8.5: Preference Optimization Deep Dives

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | direct-preference-optimization | Direct Preference Optimization | The mathematical derivation of DPO -- Bradley-Terry preference model, closed-form RLHF solution, the DPO loss function, and implementation |

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

Module 8.3 builds on:
- Series 4 (LLMs & Transformers) — autoregressive generation, decoder-only transformer, KV caching
- Series 6 (Stable Diffusion) — VAE encoding/decoding, diffusion pipeline, cross-attention conditioning
- Series 7 (Post-SD) — DiT, MMDiT, flow matching, S3-DiT, full frontier architecture understanding
- Series 6.1 (Generative Foundations) — VAE latent spaces as bridge to discrete visual tokenization

Module 8.4 builds on:
- Series 6 (Stable Diffusion) — complete SD pipeline understanding, VAE, CLIP, U-Net, CFG, samplers
- Series 7 (Post-SD) — SDXL, DiT, MMDiT, S3-DiT, flow matching, consistency models, ControlNet, LoRA
- Module 8.3 — autoregressive image generation paradigm (provides contrast to diffusion-based models)

Module 8.5 builds on:
- Series 4.4 (Beyond Pretraining) — RLHF, DPO at INTRODUCED depth, reward models, KL penalty, human preference data
- Series 5.1 (Advanced Alignment) — design space axes framework, IPO/KTO/ORPO at INTRODUCED depth, Chatbot Arena/Elo ratings
- Series 4.1-4.3 (LLM fundamentals) — log-probabilities, cross-entropy loss, language model training
