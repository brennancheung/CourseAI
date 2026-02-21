# Module 7.4: Next-Generation Architectures -- Plan

**Status:** Complete (4 of 4 lessons built)
**Prerequisites:** Module 7.3 (complete -- consistency models, LCM, adversarial distillation, speed landscape), Module 7.2 (complete -- score functions, SDE/ODE duality, flow matching), Module 7.1 (complete -- ControlNet, IP-Adapter, conditioning channels), Series 6 (complete -- full SD pipeline, U-Net, conditioning, CLIP, latent diffusion, LoRA, img2img)

## Module Goal

The student can trace the architectural evolution from Stable Diffusion v1.5 through SDXL to the Diffusion Transformer (DiT) and SD3/Flux, understanding why the field moved from refined U-Nets to transformers and how flow matching, joint text-image attention, and stronger text encoders converge in the current frontier architecture.

## Narrative Arc

The student finished Module 7.3 with a complete taxonomy of acceleration approaches--better solvers, straighter paths, and trajectory bypass. They understand the speed landscape but every model they have worked with still uses the same U-Net backbone from Stable Diffusion v1.5. This module asks: what about the architecture itself?

**Beginning (Lesson 9 -- sdxl):** The U-Net's final evolution. SDXL pushes the U-Net as far as it can go: larger backbone, higher base resolution (1024x1024), dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG) for richer text conditioning, a refiner model for high-frequency detail, and micro-conditioning to eliminate resolution-dependent artifacts. The student sees a model that is dramatically better than SD v1.5 but architecturally the same species. The lesson should feel like "the U-Net refined to its limit"--every improvement is about scaling or conditioning, not about changing the fundamental denoising architecture. The implicit question: how much further can this go?

**Middle (Lesson 10 -- diffusion-transformers):** Replace the U-Net entirely. The DiT paper (Peebles & Xie, 2023) asks: what if the denoising network were a plain vision transformer instead of a U-Net? Patchify the latent into tokens, process with standard transformer blocks, use adaptive layer norm for timestep conditioning. The student already knows transformers deeply from Series 4 and understands latent diffusion from Series 6. DiT connects these two threads. The key insights: transformers scale more predictably than U-Nets (scaling laws), removing the inductive biases of convolutions lets the model learn spatial relationships from data, and the architecture is simpler (fewer hand-designed components). The lesson should feel like "of course this was the next step"--the student has all the pieces and should see why a transformer backbone makes sense.

**Convergence (Lesson 11 -- sd3-and-flux):** SD3 and Flux combine DiT's transformer backbone with flow matching's training objective and stronger text encoding (T5-XXL). MMDiT (multimodal diffusion transformer) goes further: instead of injecting text via cross-attention, text tokens and image tokens are concatenated and attend to each other jointly. This is the architectural convergence the entire series has been building toward. The student should feel that every concept from Series 4 (transformers), Series 6 (diffusion, conditioning, latent space), Module 7.2 (flow matching), and this module (DiT) converges in one architecture.

**End (Lesson 12 -- z-image):** What comes after convergence. Z-Image (Alibaba Tongyi Lab, November 2025) starts from the same converged building blocks and asks: how can we simplify and improve? S3-DiT replaces MMDiT's dual-stream with a fully single-stream design (shared projections + shared FFN), using lightweight refiner layers for upfront modality pre-processing instead of per-layer duplication. An LLM (Qwen3-4B) replaces the triple encoder setup. Decoupled-DMD decomposes distillation into "spear" (quality) and "shield" (stability). DMDR combines distillation with RL to break the teacher ceiling. The lesson should feel like "reading a frontier paper together"--the student identifies every building block and evaluates each design choice. This is the capstone test of "you can read frontier papers and understand the design choices."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| sdxl | SDXL as the U-Net's final evolution: dual text encoders, refiner, micro-conditioning | BUILD | Follows the CONSOLIDATE lesson (the-speed-landscape). BUILD is appropriate: the student learns new concepts (dual encoders, micro-conditioning, refiner) but all are extensions of familiar patterns (CLIP text encoding, cross-attention conditioning, img2img partial denoising). No conceptual leap--just scaling and refinement. Must come before DiT to establish the baseline that DiT replaces. |
| diffusion-transformers | Replace U-Net with a vision transformer: patchify latents, transformer blocks, adaptive layer norm, scaling properties | STRETCH | The biggest conceptual shift in the module: an entirely different architecture for the denoising network. Vision Transformer applied to latent patches is a new concept (ViT was only MENTIONED in CLIP lesson). Follows BUILD appropriately. Mitigated by deep transformer knowledge from Series 4 and the "of course" framing. |
| sd3-and-flux | MMDiT (joint text-image attention), T5 text encoder, rectified flow scheduling, architectural convergence | BUILD | Applies the DiT concept from lesson 10 with modifications (MMDiT joint attention replaces cross-attention, T5 added alongside CLIP). No conceptual leap beyond lesson 10--the student extends the transformer architecture with joint attention and better text encoding. Follows STRETCH appropriately. The consolidation happens naturally as the student sees every prior concept converge. |
| z-image | S3-DiT single-stream architecture, LLM text encoder (Qwen3-4B), Decoupled-DMD distillation, DMDR post-training (DMD + RL) | BUILD | All concepts are extensions or clever combinations of building blocks the student has at DEVELOPED depth: single-stream simplifies MMDiT, Qwen3-4B extends the text encoder trajectory, Decoupled-DMD extends the distillation taxonomy from 7.3, DMDR applies DPO/GRPO from Series 5 to images. Follows BUILD appropriately. Tests the "read frontier papers" claim from the convergence lesson. |

## Rough Topic Allocation

**Lesson 9 (sdxl):**
- Dual text encoders: CLIP ViT-L/14 (SD v1.5's encoder) + OpenCLIP ViT-bigG/14 (much larger, trained on LAION-2B). Why two? Different strengths, concatenated embeddings, richer text conditioning.
- Larger U-Net: expanded channel dimensions, more attention at higher resolutions. Still the same architecture, just bigger.
- Higher base resolution: 1024x1024 vs 512x512. Training resolution matters--SD v1.5 struggles above 512 because it never saw higher resolutions.
- Refiner model: a second U-Net that takes the base model's output and performs SDEdit-style partial denoising focused on high-frequency detail. Connection to img2img from 6.5.2.
- Micro-conditioning: feeding original_size, crop_top_left, and target_size as additional conditioning inputs. Why? Eliminates artifacts from multi-resolution training data (cropped images, upscaled images).
- NOT: full SDXL training procedure, SDXL Turbo (covered in 7.3.2), every architectural detail of the larger U-Net, production deployment

**Lesson 10 (diffusion-transformers):**
- ViT applied to latent patches: patchify operation, patch size, sequence length calculation. Elevate ViT from MENTIONED to DEVELOPED.
- Standard transformer blocks: self-attention + feedforward, no convolutions, no skip connections in the U-Net sense (though residual connections within blocks exist).
- Adaptive layer norm (adaLN-Zero): timestep and class conditioning via scale/shift/gate on layer norm. Connection to adaptive group norm from 6.3.2.
- Why transformers scale better: scaling laws for DiT (loss vs compute follows predictable curves), U-Nets lack clear scaling recipes.
- Removing U-Net inductive biases: convolutions assume local spatial structure; transformers learn spatial relationships from data via attention. The tradeoff: more data-hungry but more flexible.
- NOT: ViT pretraining on classification, full DiT training from scratch, MAE or other self-supervised vision transformer methods, every DiT variant (U-ViT, etc.)

**Lesson 11 (sd3-and-flux):**
- MMDiT: text tokens and image tokens concatenated into one sequence, joint self-attention (every token attends to every other token). No cross-attention needed--text and image interact through the same attention mechanism.
- T5-XXL text encoder: a language model encoder (encoder-only transformer, 4.7B parameters) providing much richer text understanding than CLIP. Why a third encoder?
- Rectified flow training: flow matching (from 7.2.2) as the training objective instead of DDPM noise prediction. Connection to straight trajectories and fewer inference steps.
- Logit-normal timestep sampling: sampling more training steps from intermediate noise levels where the model learns the most. A scheduling optimization.
- How it all converges: DiT architecture (lesson 10) + flow matching training (7.2.2) + better text encoding (this lesson) = the current frontier.
- NOT: full SD3/Flux training procedure, every Flux variant (dev/schnell/pro), video extensions, distilled versions, licensing or business considerations

**Lesson 12 (z-image):**
- S3-DiT single-stream architecture: shared Q/K/V projections and shared FFN across all token types (text, image, visual semantic), lightweight refiner layers for upfront modality pre-processing. Contrast with MMDiT's dual-stream per-layer duplication.
- Qwen3-4B as text encoder: LLM replacing CLIP+T5 triple encoder setup. Text encoder evolution trajectory.
- 3D Unified RoPE: temporal axis for text, spatial axes for image, orthogonal encoding.
- Decoupled-DMD: spear (CFG augmentation) and shield (distribution matching) decomposition with separate noise schedules.
- DMDR: combining DMD with DPO and GRPO to break the teacher ceiling. Transfer of RL alignment from LLMs to image generation.
- Performance positioning: 6.15B vs Flux 32B, Z-Image Turbo 8-step generation.
- Source code references for the open-source implementation.
- NOT: implementing S3-DiT from scratch, training data curation details, Prompt Enhancer VLM internals, every ablation from the paper, production deployment

## Cognitive Load Trajectory

| Lesson | Load | Notes |
|--------|------|-------|
| sdxl | BUILD | 2-3 new concepts (dual text encoders, micro-conditioning, refiner model) but all are extensions of familiar patterns the student already has at DEVELOPED depth. Dual encoders extend CLIP text encoding. The refiner extends img2img. Micro-conditioning is a new conditioning channel using the student's deep understanding of how conditioning works. Follows CONSOLIDATE (the-speed-landscape) appropriately. |
| diffusion-transformers | STRETCH | The biggest conceptual shift: replacing the U-Net with a vision transformer. ViT on latent patches is genuinely new (only MENTIONED in CLIP lesson). The student has deep transformer knowledge from Series 4, which mitigates the stretch, but applying transformers to image patches at a diffusion model's denoising backbone is a new integration. Follows BUILD appropriately. |
| sd3-and-flux | BUILD | Extends DiT with joint attention (MMDiT) and stronger text encoding (T5). The student already has transformers (Series 4), DiT (lesson 10), flow matching (7.2.2), and cross-attention (6.3.4). MMDiT's joint attention is a simplification of cross-attention (concatenate and attend), not a fundamentally new mechanism. Follows STRETCH appropriately. Serves as a convergence and near-consolidation of the entire series. |
| z-image | BUILD | All new concepts are extensions or combinations of familiar building blocks at DEVELOPED depth. Single-stream simplifies MMDiT (the student evaluates a tradeoff, not learning a new mechanism). Decoupled-DMD extends the distillation taxonomy from 7.3. DMDR applies DPO/GRPO from Series 5 to image generation (same techniques, new domain). Follows BUILD appropriately. The cognitive work is integration and evaluation, not learning fundamentally new mechanisms. |

## Module-Level Misconceptions

- **"SDXL is a fundamentally different architecture from SD v1.5"** -- SDXL is the same U-Net architecture, scaled up. Dual text encoders, larger channels, higher resolution training, but the denoising backbone is still a convolutional U-Net with cross-attention at middle resolutions. The fundamental architecture did not change--only scale and conditioning did.

- **"DiT is a completely new idea unrelated to what the student knows"** -- DiT is the student's existing transformer knowledge (from Series 4) applied to the student's existing latent diffusion framework (from Series 6). The "patchify" operation is the image equivalent of tokenization, and transformer blocks are the same self-attention + feedforward the student has built from scratch. The only genuinely new element is how timestep conditioning works (adaLN-Zero instead of adaptive group norm).

- **"MMDiT's joint attention is fundamentally different from cross-attention"** -- Joint attention (concatenate text and image tokens, do standard self-attention on the combined sequence) is arguably simpler than cross-attention (separate Q from spatial, K/V from text). The student already understands both self-attention and cross-attention deeply. Joint attention removes the asymmetry: all tokens attend to all tokens, text included. It is a simplification, not a complication.

- **"Flow matching in SD3/Flux is new content for this module"** -- Flow matching was DEVELOPED in 7.2.2. Rectified flow was INTRODUCED in 7.2.2. SD3/Flux's use of flow matching is an application of concepts the student already has, not a new teaching moment. The lesson should activate and apply the concept, not re-teach it.

- **"The refiner model in SDXL is a new mechanism"** -- The refiner is img2img applied with a specialized model. The student learned img2img at DEVELOPED depth in 6.5.2. The refiner takes the base model's output, adds noise to an intermediate noise level, and denoises with a model fine-tuned on high-quality, high-resolution images. Same mechanism, specialized execution.

- **"Single-stream means no modality awareness"** -- Z-Image's S3-DiT uses shared projections and shared FFN, which might seem like it ignores the difference between text and image tokens. But the modality-specific processing is concentrated in lightweight refiner layers that pre-process each modality BEFORE fusion. The awareness is not removed--it is allocated differently (upfront, not per-layer).

- **"Distillation cannot produce students better than the teacher"** -- True for pure distillation, but DMDR combines distillation with reinforcement learning. The distillation component regularizes (prevents reward hacking), while RL provides quality signal beyond the teacher's distribution. The teacher becomes the guardrail, not the ceiling.
