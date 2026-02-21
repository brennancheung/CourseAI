# Module 8.1: Vision & Vision-Language Models -- Plan

**Module goal:** The student can explain how modern vision-language models bridge image and text understanding, tracing the architectural and loss-function innovations that make contrastive pretraining practical and effective, and how these models serve as building blocks for larger systems.

## Narrative Arc

This module explores standalone vision and vision-language models that have become foundational components across deep learning. Unlike the structured series (1-7), these lessons are self-contained deep dives. The student arrives with strong foundations in CNNs, transformers, attention, contrastive learning (CLIP), and the full diffusion pipeline. Each lesson here picks one model and examines it in depth, connecting back to these foundations while introducing the specific innovations that make each model interesting.

## Lesson Sequence

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| siglip-2 | Sigmoid loss for contrastive vision-language pretraining | BUILD | Extends CLIP knowledge from 6.3.3; introduces sigmoid loss as a cleaner alternative to softmax cross-entropy. Self-contained. |
| sam-3 | Promptable image segmentation | BUILD | Independent of SigLIP 2; can be done in any order. Extends CNN/transformer knowledge to dense prediction tasks. |

## Rough Topic Allocation

- **Lesson 1 (SigLIP 2):** CLIP recap and limitations (batch size dependency), sigmoid loss replacing softmax cross-entropy, why per-pair binary classification removes global normalization, SigLIP 2 improvements (multi-stage training, self-distillation, multi-resolution), downstream uses (zero-shot classification, VLM vision encoders like PaliGemma)
- **Lesson 2 (SAM 3):** Image segmentation primer, promptable segmentation concept, SAM architecture (image encoder + prompt encoder + mask decoder), foundation model approach to vision

## Cognitive Load Trajectory

| Lesson | Load | Rationale |
|--------|------|-----------|
| siglip-2 | BUILD | Extends familiar CLIP framework; sigmoid loss is a targeted replacement, not a paradigm shift |
| sam-3 | BUILD | Applies known architectural components (ViT, transformers) to a new task domain |

No STRETCH lessons -- these are special topics that build on deep existing knowledge rather than introducing paradigm shifts.

## Module-Level Misconceptions

- Students may think SigLIP is fundamentally different from CLIP (it is the same contrastive framework with one key change: the loss function)
- Students may think SAM is a classification model (it is a dense prediction / segmentation model, a different output modality)
- Students may conflate "vision-language model" with "multimodal LLM" (SigLIP is a pretraining method for encoders, not a chat model)

## Notes

Lessons are independent and self-contained. Order does not matter. Each includes heavier recap sections than structured series lessons, since the student may arrive at any time with varying recency of prerequisite knowledge.
