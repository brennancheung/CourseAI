# Module 7.1: Controllable Generation -- Plan

**Goal:** The student can explain how ControlNet and IP-Adapter add structural and semantic conditioning to a frozen Stable Diffusion model, understanding the architectural patterns (trainable encoder copy, zero convolution, decoupled cross-attention) well enough to reason about when and why to use each technique.

## Narrative Arc

Series 6 built the full Stable Diffusion pipeline and customized it via LoRA, img2img, and textual inversion. But all of those conditioning pathways are limited: text prompts describe *what* to generate, img2img controls *how much* to change, and LoRA shifts *style*. None of them give precise *spatial* control--"put the figure here, follow this edge map, match this pose."

Module 7.1 opens Series 7 by showing how to add new conditioning channels to a frozen SD model without breaking anything it already knows. The story follows a progression of control specificity:

1. **ControlNet** (structural control): The student learns how to inject spatial maps (edges, depth, pose) by cloning the U-Net encoder, training only the clone, and feeding its outputs back into the frozen decoder via zero convolutions. This is the architectural pattern--a trainable copy that cannot corrupt the original.
2. **ControlNet in Practice**: Hands-on use of real preprocessors (Canny, depth estimation, OpenPose), stacking multiple ControlNets, and understanding the conditioning scale as a control-creativity tradeoff. This is the practical payoff.
3. **IP-Adapter** (semantic control): Instead of spatial maps, condition on a *reference image*. The student learns how decoupled cross-attention adds a second set of K/V projections for image features alongside the existing text K/V, enabling image prompting without replacing text control.

The arc moves from "add spatial structure" to "use it in practice" to "add semantic meaning from images"--three layers of control beyond text, each building on the same frozen-model pattern.

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| controlnet | ControlNet architecture: trainable encoder copy + zero convolution | STRETCH | First lesson in a new series; introduces the key architectural innovation. STRETCH because the trainable-copy-with-zero-conv pattern is genuinely new, even though all building blocks (U-Net encoder, skip connections, cross-attention) are well-established. |
| controlnet-in-practice | Preprocessors, multi-ControlNet stacking, conditioning scale | CONSOLIDATE | Applies the architecture from lesson 1 with real tools. No new theory--uses diffusers to explore ControlNet with different spatial maps. A hands-on "win" after the architectural work. |
| ip-adapter | Decoupled cross-attention for image conditioning | BUILD | Extends the cross-attention mechanism the student already has at DEVELOPED depth. Small conceptual delta (add a parallel K/V path for image embeddings) but meaningful architectural insight. Requires ControlNet context for comparison. |

## Rough Topic Allocation

- **Lesson 1 (controlnet):** The problem of spatial control (motivating example: edge map, depth map, pose). ControlNet architecture: clone the encoder, freeze the original, train the clone. Zero convolution as the safe connection mechanism. How ControlNet outputs merge with the frozen decoder. Brief cross-attention reactivation (needed for understanding where text conditioning still operates).
- **Lesson 2 (controlnet-in-practice):** Preprocessors that extract spatial maps from images (Canny edge detection, MiDaS depth estimation, OpenPose skeleton). Using ControlNet via diffusers. Conditioning scale parameter (how much to follow the control vs allow creative freedom). Stacking multiple ControlNets (edge + depth simultaneously). Limitations and failure modes.
- **Lesson 3 (ip-adapter):** The problem of "describe this image in words" (motivating limitation of text-only conditioning). CLIP image embeddings as conditioning signal (callback to shared embedding space from 6.3.3). Decoupled cross-attention: separate K/V projections for text and image, added in parallel. IP-Adapter as a lightweight adapter (not fine-tuning). Comparison with textual inversion and LoRA for concept transfer.

## Cognitive Load Trajectory

| Lesson | Load | Notes |
|--------|------|-------|
| controlnet | STRETCH | New architectural pattern (trainable encoder copy + zero conv). All building blocks are familiar but the assembly is novel. First lesson in a new series--high engagement expected. |
| controlnet-in-practice | CONSOLIDATE | Hands-on application of lesson 1 with real tools. No new theory. |
| ip-adapter | BUILD | Small conceptual delta on deep cross-attention knowledge. Decoupled cross-attention is a clean extension. |

## Module-Level Misconceptions

- **"ControlNet fine-tunes or modifies the original Stable Diffusion model"** -- The entire point is that the original model is frozen. ControlNet trains a copy and connects it via zero convolutions that start at zero, guaranteeing no corruption.
- **"ControlNet replaces text conditioning"** -- ControlNet adds spatial conditioning alongside text conditioning. Both operate simultaneously. The text prompt still matters for style, content, and semantics.
- **"IP-Adapter and ControlNet do the same thing"** -- ControlNet provides structural/spatial control (edges, depth, pose). IP-Adapter provides semantic/stylistic control from a reference image. Different conditioning types for different purposes.
- **"You need to retrain ControlNet for every new spatial map type"** -- Each ControlNet is trained for one map type, but the architectural pattern is identical. You swap ControlNet checkpoints, not retrain.
- **"Image prompting (IP-Adapter) means replacing the text prompt with an image"** -- Decoupled cross-attention means both text and image conditioning coexist. The image provides semantic guidance while text provides additional specificity.
