# Series 7: Post-SD Advances -- Summary

**Status:** Complete (Module 7.1: complete, Module 7.2: complete, Module 7.3: complete, Module 7.4: complete)

## Series Goal

Take the student from "I understand how Stable Diffusion v1.5 works end-to-end" to "I can read current diffusion research and understand why the field moved beyond DDPM, U-Net, and multi-step sampling"--covering the key architectural, theoretical, and practical advances that define post-2023 generative image models, culminating in SD3 and Flux as the current frontier architecture where every design choice traces to a lesson already completed.

## Rolled-Up Concept List

### From Module 7.1: Controllable Generation (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Trainable encoder copy architecture (clone frozen U-Net encoder, initialize from original weights, train the copy on spatial maps, add outputs to skip connections via zero convolution) | DEVELOPED | "Of course" design chain: spatial features live in encoder, you cannot retrain frozen encoder, so copy it. ~300M trainable params (~35% of U-Net). |
| Zero convolution mechanism (1x1 conv initialized to all-zero weights and bias; guarantees zero initial contribution; control signal fades in gradually) | DEVELOPED | "Nothing, Then a Whisper" metaphor. Concrete numerical examples. Connected to LoRA B=0 initialization--same safety pattern. |
| Decoupled cross-attention for IP-Adapter (parallel K/V projections for CLIP image embeddings alongside text K/V; shared Q; weighted addition of two attention outputs) | DEVELOPED | "Two reference documents, one reader." Shape walkthrough at 16x16 resolution. ~22M trainable params, trained once on millions of pairs, works with any reference image. |
| Four conditioning channels (WHEN via timestep/adaptive norm, WHAT via text/cross-attention, WHERE via ControlNet/additive encoder features, WHAT-IT-LOOKS-LIKE via IP-Adapter/decoupled cross-attention) | DEVELOPED | Module-level synthesis. All four channels additive, composable, target different U-Net parts. Extended to five with SDXL micro-conditioning (AT-WHAT-QUALITY). |
| Conditioning scale / control-creativity tradeoff (volume knob for spatial or image control at inference; connected to CFG guidance scale and img2img strength) | DEVELOPED | Six-value sweep (0.3-2.0). Sweet spot 0.7-1.0. Same "volume knob" pattern across ControlNet, IP-Adapter, and CFG. |
| Canny edge detection, MiDaS depth estimation, OpenPose skeleton detection (preprocessing tools for ControlNet spatial maps) | DEVELOPED | Black-box tools with practical tuning. "Garbage in, garbage out" insight. |
| Multi-ControlNet stacking (additive feature composition from multiple ControlNets) | DEVELOPED | Complementary vs conflicting maps. Per-ControlNet scales. Same source image recommended. |
| IP-Adapter vs LoRA distinction (LoRA modifies existing W_K/W_V, IP-Adapter adds new K/V path; "remove and check" test) | DEVELOPED | Highest-priority comparison. "Same highway, add a detour" (LoRA) vs "build a second highway" (IP-Adapter). |
| Zero initialization pattern across adapters (zero conv in ControlNet, B=0 in LoRA, zero K/V in IP-Adapter) | REINFORCED | Third reinforcement. Same safety pattern: start contributing nothing. |
| IP-Adapter is NOT img2img (CLIP semantic encoding vs VAE pixel-level encoding) | DEVELOPED | Reference image never enters denoising loop as latent. |

### From Module 7.2: The Score-Based Perspective (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Score function as gradient of log probability (nabla_x log p(x); direction toward higher probability; compass toward likely data) | DEVELOPED | 1D Gaussian concrete example. 2D vector field visualization. WarningBlock: gradient w.r.t. data, not parameters. |
| Score-noise equivalence (epsilon_theta approx -sqrt(1-alpha_bar_t) * score; DDPM was always a score-based model in disguise) | DEVELOPED | Three-step algebraic walkthrough. "The score was hiding inside DDPM all along." |
| SDE forward process (continuous-time generalization of DDPM; staircase-to-ramp analogy) | INTRODUCED | Concrete numerical example. WarningBlock: no Ito calculus. |
| Probability flow ODE (deterministic reverse SDE; DDIM was approximately Euler's method on this ODE) | INTRODUCED | Deepened from MENTIONED (6.4.2). Named and formalized. |
| Conditional flow matching (straight-line interpolation x_t = (1-t)*x_0 + t*epsilon; velocity prediction v = epsilon - x_0; simpler alternative to DDPM) | DEVELOPED | "The Flip": design the trajectory first, derive the training objective. Worked example with specific numbers. "Of Course" chain. Student trained a 2D flow matching model. |
| Straight-line vs curved trajectories (flow matching paths straight by construction; diffusion ODE paths curve because score field changes with noise level) | DEVELOPED | Core geometric insight. Euler's method on straight path is exact in one step. "GPS recalculating vs straight highway." |
| Velocity prediction parameterization (third member of noise/score/velocity family; interconvertible with simple algebra) | DEVELOPED | Three-column comparison. Explicit conversion formulas. Same architecture works for all three. |
| Rectified flow (iterative trajectory straightening via model-generated aligned pairs) | INTRODUCED | "Retracing a hand-drawn line with a ruler." 1-2 rounds sufficient. |
| Flow matching as same family as diffusion (not a new paradigm; same type of generative model) | INTRODUCED | "Same landscape, different lens" extended to three lenses (SDE, ODE, flow matching). |

### From Module 7.3: Fast Generation (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| Self-consistency property of ODE trajectories (any point on the same trajectory maps to the same clean endpoint; a trivial ODE fact used as a training objective) | DEVELOPED | Trajectory diagram. Worked 2D example. "Not a New Mathematical Result"--the insight is using it as a training objective. |
| Consistency function f(x_t, t) = x_0 (maps any noisy input directly to clean endpoint; one evaluation replaces multi-step trajectory-following) | DEVELOPED | "Not Fewer Steps--No Steps." ComparisonRow vs 1-step ODE solver. "Predict-and-Leap, Perfected." |
| Consistency distillation (pretrained diffusion teacher provides trajectory estimates; student learns to make adjacent points produce same output) | DEVELOPED | Four-step training procedure. ASCII diagram. EMA target for collapse prevention. |
| Consistency training without a teacher (slower, lower quality than distillation) | INTRODUCED | ComparisonRow across 5 dimensions. FID comparison. "Discovery vs Learning." |
| "Three levels of speed" framework (Level 1: better solvers/15-20 steps, Level 2: straighter trajectories/20-30 steps, Level 3: bypass trajectory/1-4 steps) | DEVELOPED | Organizes entire speed narrative. Extended to 3a (consistency-based) and 3b (adversarial). Reframed from progression to menu. |
| LCM (consistency distillation on latent diffusion; "same recipe, bigger kitchen") | DEVELOPED | Four scaling adaptations. Not a new architecture. Augmented PF-ODE for text faithfulness. |
| LCM-LoRA (consistency distillation as a ~4 MB LoRA adapter; "speed as a skill") | DEVELOPED | Reframes LoRA from style adapter to speed adapter. One adapter works across compatible fine-tunes. |
| Adversarial diffusion distillation / ADD (hybrid loss: diffusion distillation + adversarial discriminator; SDXL Turbo) | DEVELOPED | "Sharpness problem" motivation. Contrastive quality comparison. "Two teachers, two lessons." |
| Quality-speed curve nonlinearity (1000 to 20 steps is free, 20 to 4 is cheap, 4 to 1 is expensive) | DEVELOPED | Recharts visualization with three regions. "Most speedup is free." |
| Cross-level composability rule (approaches compose across levels but conflict within levels) | DEVELOPED | ComposabilityGrid (5x5 matrix). Kitchen Sink negative example. |
| Four-dimensional decision framework (speed, quality, flexibility, composability) | DEVELOPED | Three worked scenarios reaching different conclusions. "Which is right for what I need?" not "which is fastest?" |

### From Module 7.4: Next-Generation Architectures (complete)

| Concept | Depth | Key Teaching |
|---------|-------|-------------|
| SDXL dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG concatenated to [77,2048]) | DEVELOPED | Tensor shapes traced. Single cross-attention path. Pooled embedding for global conditioning. |
| Micro-conditioning (original_size, crop_top_left, target_size via adaptive norm) | INTRODUCED | Problem-before-solution: three bad multi-resolution options. Solution: tell the model the training context. AT-WHAT-QUALITY as fifth conditioning dimension. |
| SDXL as U-Net ceiling ("the U-Net's last stand") | INTRODUCED | Five changes all go IN, AROUND, or ALONGSIDE the U-Net. Limitations: no scaling recipe, limited text-image interaction. |
| Patchify operation (latent [C,H,W] to non-overlapping patches, flatten, project to d_model) | DEVELOPED | Full tensor shape trace. "Tokenize the image" analogy. Patch size as resolution-compute tradeoff. |
| DiT block as standard transformer block (MHA + FFN + residual + norm, no convolutions) | DEVELOPED | Every component checked off against Series 4 knowledge. ComparisonRow: U-Net vs DiT. |
| adaLN-Zero conditioning (adaptive layer norm with scale, shift, gate; 6 params per block; zero-initialized gate = identity) | DEVELOPED | Full formula trace. Connected to adaptive group norm (6.3.2) and zero convolution (7.1.1). |
| DiT scaling recipe (increase d_model and N; same as GPT-2 to GPT-3) | INTRODUCED | "Two knobs, not twenty." LEGO vs house renovation. |
| MMDiT joint attention (concatenate text + image tokens, standard self-attention on combined sequence, four attention types) | DEVELOPED | "One room, one conversation." Concrete shape trace (77+256=333). ComparisonRow vs cross-attention. Design challenge with three options. |
| Modality-specific Q/K/V projections (separate projections per modality, shared attention, separate FFNs) | DEVELOPED | "Shared listening, separate thinking." Negative example: naive shared projections. CodeBlock trace through full block. |
| T5-XXL as text encoder (4.7B params, deep linguistic understanding alongside CLIP visual alignment) | INTRODUCED | Gap fill from MENTIONED. Complementary, not replacement. Triple encoder progression. |
| Rectified flow applied in SD3 (same flow matching from 7.2.2 at scale; 20-30 steps) | INTRODUCED | "You already know everything about flow matching that SD3 uses." |
| SD3/Flux as convergence architecture (every component traces to a prior lesson) | INTRODUCED | Five-thread convergence map. Annotated 13-step pipeline. "Convergence, not revolution." |
| S3-DiT single-stream architecture (shared Q/K/V projections and SwiGLU FFN after lightweight refiner layers) | INTRODUCED | "Translate once, then speak together." Eliminates ~50% parameter overhead of MMDiT dual-stream. Parameter counting with d_model=3840. |
| Qwen3-4B as text encoder (single LLM replacing triple encoder setup) | INTRODUCED | Text encoder evolution: 1 CLIP -> 2 CLIPs -> 3 encoders -> 1 LLM. "Convergence to simplicity." |
| 3D Unified RoPE (temporal/height/width axes for multi-modal position encoding) | INTRODUCED | Extension of 1D RoPE to 3D. Orthogonal axes, zero rotation for non-applicable modality. |
| DMD / Distribution Matching Distillation (CFG augmentation + distribution matching) | INTRODUCED | Third distillation paradigm. "Spear" (quality) and "shield" (stability). |
| Decoupled-DMD (spear/shield with separate noise schedules) | INTRODUCED | High noise for spear, low noise for shield. Eliminates artifacts from coupling. |
| DMDR (DMD + DPO + GRPO post-training breaking the teacher ceiling) | INTRODUCED | "The teacher becomes the guardrail, not the ceiling." DPO/GRPO from Series 5 applied to image generation. |

## Key Mental Models Carried Forward

1. **"WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE/AT-WHAT-QUALITY"** -- Five conditioning channels in a frozen SD model, each targeting a different mechanism. Extended from three (WHEN/WHAT/WHERE in ControlNet) to four (IP-Adapter adds WHAT-IT-LOOKS-LIKE) to five (SDXL micro-conditioning adds AT-WHAT-QUALITY).
2. **"Nothing, Then a Whisper"** -- Zero initialization as a safety pattern across adapters: ControlNet zero conv, LoRA B=0, IP-Adapter zero K/V, adaLN-Zero alpha=0. New components start contributing nothing to the frozen model.
3. **"Two reference documents, one reader"** -- IP-Adapter's decoupled cross-attention reads from two sources (text K/V and image K/V) with shared Q. Distinct from MMDiT's "one room" approach.
4. **"Compass toward likely data"** -- The score function at every point in data space tells you which direction to go for higher probability.
5. **"The score was hiding inside DDPM all along"** -- The noise prediction network has been learning the score function. Nothing new was trained.
6. **"Staircase to ramp"** -- DDPM (discrete steps) to SDE (continuous time). Same process, smoother description.
7. **"Same landscape, different lens"** -- Four lenses on the same generative process: diffusion SDE (stochastic, curved), probability flow ODE (deterministic, curved), flow matching ODE (deterministic, straight), consistency models (teleport to endpoint).
8. **"Curved vs straight"** -- Flow matching's core geometric insight. Straight trajectories need fewer ODE solver steps. "GPS recalculating vs straight highway."
9. **"Design the trajectory, then derive the training objective"** -- Flow matching flips DDPM's order. Choose straight lines, derive velocity prediction.
10. **"Three levels of speed"** -- Level 1: better solvers (walk smarter), Level 2: straighter trajectories (straighten the road), Level 3: bypass trajectory (teleport). Reframed as a menu, not an upgrade path.
11. **"Teleport to the destination"** -- Consistency models do not step along the trajectory; they jump directly from any noisy point to the clean endpoint.
12. **"Predict-and-Leap, Perfected"** -- DDIM predicts x_0 and iterates. Consistency model predicts x_0 and stops.
13. **"Same recipe, bigger kitchen"** -- LCM scales consistency distillation from toy 2D to latent diffusion. Same pattern, bigger scale.
14. **"Speed as a skill"** -- LCM-LoRA captures "how to generate fast" as a LoRA adapter. One ~4 MB file turns any compatible model into a 4-step model.
15. **"Menu, not upgrade path"** -- The three speed levels are options with different tradeoffs, not version numbers.
16. **"Most speedup is free"** -- The first 50x (1000 to 20 steps) costs nothing in quality. The tradeoff only becomes real below ~10 steps.
17. **"Across, not within"** -- Acceleration approaches compose across levels but conflict within levels.
18. **"Tokenize the image"** -- Patchify is to images what tokenization is to text. Both produce [n, d_model] sequences.
19. **"Two knobs, not twenty"** -- DiT scales with d_model and N, same as GPT-2 to GPT-3. LEGO vs house renovation.
20. **"Same pipeline, different denoising network"** -- DiT (and then MMDiT) replaces only the U-Net. VAE, text encoder, sampling unchanged.
21. **"The U-Net's last stand"** -- SDXL as the final U-Net refinement before the paradigm shift to transformers.
22. **"One room, one conversation"** -- MMDiT's joint attention: text and image tokens in one sequence, bidirectional interaction. Simpler (one attention operation) and richer (four attention types) than cross-attention.
23. **"Shared listening, separate thinking"** -- MMDiT's modality-specific projections and FFNs with shared attention. They hear each other through shared attention but maintain distinct representational identities.
24. **"Convergence, not revolution"** -- SD3/Flux combines concepts from across the entire course. Every component traces to a lesson already completed. The frontier is the synthesis of understanding.
25. **"Translate once, then speak together"** -- S3-DiT concentrates modality-specific processing in lightweight refiner layers, then uses fully shared projections and FFN. Contrasts with MMDiT's per-layer duplication.
26. **"Spear and shield"** -- Decoupled-DMD separates distillation into quality-driving CFG augmentation (spear) and distribution-preserving regularization (shield). Separate noise schedules for each.
27. **"The teacher becomes the guardrail, not the ceiling"** -- DMDR uses DMD as regularizer and RL as quality driver. Student can exceed teacher because RL provides external reward signal.
28. **"Simplicity beats complexity"** -- Z-Image matches 32B Flux with 6.15B parameters through architecture simplification + training innovation + post-training, not architectural novelty.
29. **"Convergence to simplicity"** -- Text encoder trajectory from multiple specialized encoders (CLIP -> dual CLIP -> CLIP+T5) back to one powerful LLM (Qwen3-4B).

## What This Series Has NOT Covered

- Full training of any of these models from scratch (too compute-intensive; Series 6 capstone was the training experience)
- Video diffusion models (Sora, Runway, Kling)
- 3D generation (Zero-1-to-3, MVDream)
- Audio/music diffusion
- Image editing models (InstructPix2Pix, prompt-to-prompt)
- Production deployment, quantization, inference optimization
- GANs beyond brief context in adversarial distillation
- Autoregressive image models (DALL-E, Parti, Chameleon)
- Ito calculus, stochastic integration, Fokker-Planck equation
- Implementing MMDiT from scratch
- Every ControlNet variant (T2I-Adapter, etc.)
- Every Flux variant (dev/schnell/pro/fill) beyond vocabulary positioning
- T5 internal architecture (used as a black box)
- ControlNet/IP-Adapter for SD3/Flux (same concepts, different weight dimensions)

## Module Completion Notes

### Module 7.1 (complete)
Three lessons covering controllable generation: ControlNet (structural conditioning via trainable encoder copy + zero convolution), ControlNet in Practice (preprocessors, conditioning scale, multi-ControlNet stacking), IP-Adapter (image conditioning via decoupled cross-attention). The module establishes the four conditioning channels framework (WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE) and the zero initialization safety pattern that recurs throughout the series. All three lessons have notebooks. The cognitive load progression is STRETCH -> CONSOLIDATE -> BUILD.

### Module 7.2 (complete)
Two lessons covering the score-based perspective: Score Functions & SDEs (score function definition, score-noise equivalence, SDE/ODE formalization) and Flow Matching (straight-line interpolation, velocity prediction, rectified flow). This is the theoretical core of the series. The score-noise equivalence ("DDPM was always a score-based model") is the emotional reveal. Flow matching is the conceptual pivot that explains why modern models train differently and generate in fewer steps. Both lessons have notebooks, including training a 2D flow matching model from scratch. The cognitive load progression is STRETCH -> BUILD.

### Module 7.3 (complete)
Three lessons covering fast generation: Consistency Models (self-consistency property, consistency distillation, teleporting to the endpoint), LCM & SDXL Turbo (latent consistency models, LCM-LoRA as speed adapter, adversarial diffusion distillation), The Speed Landscape (comprehensive taxonomy, four-dimensional decision framework, composability rules). The "three levels of speed" framework organizes the entire acceleration narrative from Series 6 through this module. All three lessons have notebooks. The cognitive load progression is STRETCH -> BUILD -> CONSOLIDATE.

### Module 7.4 (complete)
Four lessons covering next-generation architectures: SDXL (dual encoders, micro-conditioning, refiner--"the U-Net's last stand"), Diffusion Transformers (patchify, adaLN-Zero, transformer scaling recipe), SD3 & Flux (MMDiT joint attention, T5-XXL, flow matching applied at scale--the convergence), Z-Image & Z-Image Turbo (S3-DiT single-stream architecture, Qwen3-4B LLM encoder, Decoupled-DMD, DMDR post-training breaking the teacher ceiling). The module arc traces the architectural evolution from refined U-Nets (SDXL) through the paradigm shift (DiT) to the convergence architecture (SD3/Flux) and then beyond--what comes after convergence (Z-Image). The Z-Image lesson serves as the series capstone, testing the claim that "you can read frontier papers and understand the design choices" by reading a November 2025 paper together. All four lessons have notebooks. The cognitive load progression is BUILD -> STRETCH -> BUILD -> BUILD.

### Series 7 (complete)
Twelve lessons across four modules. From "I understand how Stable Diffusion v1.5 works" (start of 7.1) to "I can read frontier diffusion research and trace every design choice to concepts I built from scratch" (end of 7.4). The student has understanding of: spatial conditioning via ControlNet and IP-Adapter, the score-based perspective unifying DDPM with continuous-time frameworks, flow matching as the training objective shift enabling fewer inference steps, consistency models for 1-4 step generation, the full acceleration taxonomy with composability rules, SDXL as the U-Net ceiling, DiT as the architecture shift from U-Net to transformer, SD3/Flux as the convergence of transformers + latent diffusion + flow matching + better text encoding, and Z-Image as what comes after convergence--S3-DiT single-stream simplification, LLM text encoding, Decoupled-DMD distillation, and DMDR RL post-training that breaks the teacher ceiling. 29 mental models accumulated. The student can look at the Z-Image pipeline and explain every component from first principles, tracing each design choice back through 50+ lessons across 7 series.
