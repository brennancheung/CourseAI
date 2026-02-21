# Series 7: Post-SD Advances -- Plan

**Status:** Complete
**Prerequisites:** Series 1-3 (complete), Series 4 Modules 4.1-4.2 (complete), Series 6 (complete)

## Series Goal

Take the student from "I understand how Stable Diffusion v1.5 works end-to-end" to "I can read current diffusion research and understand why the field moved beyond DDPM, U-Net, and multi-step sampling"--covering the key architectural, theoretical, and practical advances that define post-2023 generative image models, culminating in the Diffusion Transformer (DiT) architecture that powers SD3 and Flux.

## Modules

| Module | Title | Lessons | Status | Focus |
|--------|-------|---------|--------|-------|
| 7.1 | Controllable Generation | 3 | complete | ControlNet, IP-Adapter -- structural and semantic conditioning beyond text prompts |
| 7.2 | The Score-Based Perspective | 2 | complete | Score functions, SDE/ODE duality, flow matching -- the theoretical reframing underlying modern architectures |
| 7.3 | Fast Generation | 3 | complete | Consistency models, latent consistency models, SDXL Turbo -- collapsing the multi-step process |
| 7.4 | Next-Generation Architectures | 4 | complete | SDXL, Diffusion Transformers (DiT), SD3/Flux, Z-Image -- the architectural shift from U-Net to transformer and beyond convergence |

## Lessons

### Module 7.1: Controllable Generation

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | controlnet | ControlNet | Add structural control (edges, depth, pose) to a frozen SD model. A trainable copy of the encoder learns to condition on spatial maps without forgetting anything. |
| 2 | controlnet-in-practice | ControlNet in Practice | Use ControlNet with real preprocessors (Canny, depth, OpenPose). Stack multiple ControlNets. Understand the control-creativity tradeoff via conditioning scale. |
| 3 | ip-adapter | IP-Adapter | Image prompting: use a reference image instead of (or alongside) text to guide generation. Decoupled cross-attention separates text and image conditioning. |

### Module 7.2: The Score-Based Perspective

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 4 | score-functions-and-sdes | Score Functions & SDEs | The score function (gradient of log probability) as the unifying concept behind diffusion. SDE forward process generalizes DDPM's discrete steps to continuous time. The ODE view from 6.4.2 formalized. |
| 5 | flow-matching | Flow Matching | Straight paths between noise and data instead of curved diffusion trajectories. Conditional flow matching as a simpler, more stable training objective. Rectified flow and why it enables fewer steps. |

### Module 7.3: Fast Generation

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 6 | consistency-models | Consistency Models | Map any point on the ODE trajectory directly to the endpoint. Self-consistency as a training objective. Consistency training vs consistency distillation. |
| 7 | latent-consistency-and-turbo | LCM & SDXL Turbo | Consistency distillation applied to latent diffusion models. 1-4 step generation from existing SD/SDXL checkpoints. Adversarial diffusion distillation (ADD) in SDXL Turbo. |
| 8 | the-speed-landscape | The Speed Landscape | Compare all acceleration approaches: better samplers (Series 6), consistency distillation, adversarial distillation, flow matching with fewer steps. When to use what. Quality-speed-flexibility tradeoffs. |

### Module 7.4: Next-Generation Architectures

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 9 | sdxl | SDXL | Dual text encoders (CLIP + OpenCLIP), larger U-Net, higher base resolution, refiner model, micro-conditioning. What changed and why it matters. |
| 10 | diffusion-transformers | Diffusion Transformers (DiT) | Replace U-Net with a transformer: patchify latents into tokens, use standard transformer blocks with adaptive layer norm. Why transformers scale better than U-Nets. |
| 11 | sd3-and-flux | SD3 & Flux | MMDiT (joint text-image attention), T5 text encoder, rectified flow scheduling. How flow matching + DiT + better text encoding converge into the current frontier. |
| 12 | z-image | Z-Image & Z-Image Turbo | S3-DiT single-stream architecture, Qwen3-4B LLM encoder, Decoupled-DMD distillation, DMDR post-training. What comes after convergence--simplification and RL post-training. |

## Scope Boundaries

### In scope

- ControlNet architecture and zero-convolution mechanism
- IP-Adapter and decoupled cross-attention
- Score functions and the score-based generative model perspective (intuition-level, not measure theory)
- SDE/ODE duality for diffusion (connecting to the ODE perspective from 6.4.2)
- Flow matching and rectified flow (the training objective shift)
- Consistency models (both consistency training and consistency distillation)
- Latent consistency models (LCM) and SDXL Turbo / adversarial diffusion distillation
- SDXL architecture changes (dual encoders, refiner, micro-conditioning)
- Diffusion Transformers (DiT) architecture
- SD3 and Flux architecture (MMDiT, T5 encoder, rectified flow)
- Practical use of these models via diffusers

### Out of scope

- Full SDE/ODE mathematical derivations (Ito calculus, Fokker-Planck equation)
- Training any of these models from scratch (too compute-intensive; Series 6 capstone was the training experience)
- Video diffusion models (Sora, Runway, Kling)
- 3D generation (Zero-1-to-3, MVDream, instant3D)
- Audio/music diffusion
- Image editing models (InstructPix2Pix, prompt-to-prompt beyond brief mention)
- Production deployment, quantization, inference optimization
- GANs or GAN-based acceleration beyond brief context in adversarial distillation
- Autoregressive image models (DALL-E, Parti, Chameleon)
- Every ControlNet variant (T2I-Adapter, etc.--mentioned for context, not developed)

## Connections to Prior Series

| Series 7 Concept | Earlier Concept | Source |
|------------------|----------------|--------|
| ControlNet encoder copy | U-Net encoder-decoder architecture | 6.3.1 (unet-architecture) |
| ControlNet zero-convolution | Residual connections, "highway and detour" | 3.2 (ResNet), 4.4.4 (LoRA) |
| ControlNet spatial conditioning | Cross-attention spatially-varying conditioning | 6.3.4 (text-conditioning-and-guidance) |
| IP-Adapter decoupled cross-attention | Cross-attention Q/K/V mechanism | 4.2, 6.3.4 |
| IP-Adapter image embeddings | CLIP shared embedding space | 6.3.3 (clip) |
| Score function as gradient | Gradient descent, loss landscape | 1.1 (gradient-descent) |
| SDE continuous-time diffusion | DDPM discrete forward/reverse process | 6.2.1-6.2.4 |
| ODE trajectory view | ODE perspective on diffusion | 6.4.2 (samplers-and-efficiency) |
| Flow matching straight paths | DDIM predict-and-leap, ODE trajectory | 6.4.2 |
| Consistency model self-consistency | ODE trajectory, sampler as ODE solver | 6.4.2 |
| Consistency distillation | Knowledge distillation pattern (teacher-student) | New, but connects to LoRA as "learning from a trained model" |
| SDXL dual text encoders | CLIP text encoder, cross-attention text injection | 6.3.3-6.3.4 |
| SDXL refiner model | Img2img partial denoising | 6.5.2 (img2img-and-inpainting) |
| DiT patchify | Vision Transformer / ViT patch tokenization | 6.3.3 (MENTIONED in clip lesson) |
| DiT adaptive layer norm | Adaptive group normalization for timestep conditioning | 6.3.2 (conditioning-the-unet) |
| DiT transformer blocks | Full transformer architecture, self-attention | 4.2 (attention-and-transformer) |
| SD3 MMDiT joint attention | Cross-attention, self-attention | 4.2, 6.3.4 |
| SD3 T5 text encoder | Language model text encoding | 4.1-4.3 |
| Rectified flow in SD3 | Flow matching (this series), DDPM noise schedule | 7.2, 6.2.2 |

## Key Pedagogical Notes

- **Capstone series tone:** This is the final series. The student has built SD from scratch and customized it. The tone shifts from "let me teach you" to "let's read the frontier together." Lessons should feel more like guided paper-reading than hand-holding. The student should be deriving insights, not receiving them.

- **Score-based theory scoped to intuition:** Module 7.2 is the theoretical core. The score function and SDE/ODE perspective must be taught at INTRODUCED/DEVELOPED depth--enough to understand why flow matching works and why DiT uses it--but NOT at the measure-theory level. Keep it grounded: "the score function points toward higher probability," not "the Stein score is the gradient of the log-density with respect to the Hessian of the Ito diffusion." Concrete examples and geometric intuition over formalism.

- **Flow matching is the conceptual pivot:** The single most important idea in this series is flow matching. It explains why modern models (Flux, SD3) train differently from DDPM, why they generate in fewer steps, and why the architecture could shift from U-Net to transformer. Module 7.2 must land this concept solidly because Modules 7.3 and 7.4 both depend on it.

- **ODE perspective from 6.4.2 is the bridge:** The sampler lesson already INTRODUCED the ODE view of diffusion and Euler's method. Module 7.2 formalizes and extends this. It should feel like a natural deepening, not a retcon. "Remember when we said DDIM is approximately Euler's method on the diffusion ODE? Let's make that precise."

- **ControlNet before theory:** Module 7.1 comes first because it is the most concrete and immediately useful advance. It requires zero new theory--just clever architecture on top of the existing SD stack. This gives the student a "win" before the theoretical work in 7.2.

- **Consistency models need the ODE view:** The student cannot understand consistency models without the trajectory/ODE perspective. This is why 7.2 (theory) must precede 7.3 (fast generation). The consistency property ("any point on the same trajectory maps to the same endpoint") only makes sense if you see generation as following a trajectory.

- **DiT as the convergence:** Module 7.4 is where everything comes together. DiT replaces U-Net with a transformer (the student knows transformers deeply from Series 4). SD3/Flux combines DiT + flow matching + better text encoding. The student should feel that every piece of their knowledge--transformers, diffusion, attention, flow matching--converges in these architectures.

- **ViT elevation needed:** Vision Transformer was only MENTIONED in the CLIP lesson (6.3.3). The DiT lesson (7.4.2) needs to elevate this to DEVELOPED. The "patchify" operation (split image into patches, project to token dimension, process with standard transformer blocks) is the key architectural move. This is a medium gap--the student knows transformers deeply and knows CNNs, but has not seen transformers applied to images.

- **SDXL is a bridge, not a destination:** SDXL is still a U-Net model. It matters because (a) it is widely used and (b) its innovations (dual encoders, micro-conditioning, refiner) appear in modified form in SD3/Flux. But the real destination is DiT. The SDXL lesson should feel like "the U-Net's last stand" before the paradigm shift.

- **No training from scratch in this series:** Series 6 had the capstone training experience (pixel-space diffusion model). This series focuses on architectural understanding and practical use. Notebooks use pre-trained models via diffusers/ComfyUI, not training runs. The exception is consistency models, where a toy consistency training exercise on 2D data could build intuition.

- **The speed lesson (7.3.3) is a consolidation checkpoint:** By lesson 8, the student has seen four approaches to faster generation: better ODE solvers (6.4.2), consistency models (7.3.1), latent consistency distillation (7.3.2), and flow matching with fewer steps (7.2.2). The speed landscape lesson organizes these into a coherent taxonomy. It should feel like the sampler comparison lesson (6.4.2) but at a higher level.

- **Reinforcement rule for cross-attention:** Cross-attention was DEVELOPED in 6.3.4 (~15+ lessons ago by the time we reach Module 7.1). ControlNet and IP-Adapter both depend on it. Brief reactivation needed--not re-teaching, but reconnecting. "Remember: Q from spatial features, K/V from text. ControlNet adds a new source of spatial features. IP-Adapter adds a new source of K/V."
