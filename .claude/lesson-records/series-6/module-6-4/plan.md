# Module 6.4: Stable Diffusion -- Plan

**Module goal:** The student can work with the full Stable Diffusion pipeline end-to-end -- assembling the components they learned individually (CLIP, U-Net, VAE, conditioning, CFG), understanding why advanced samplers let them generate in 20 steps instead of 1000, and using the diffusers library to generate images with full comprehension of every parameter.

## Narrative Arc

Module 6.3 ended with the student understanding every component of Stable Diffusion individually and seeing how latent diffusion combines them conceptually. But the student has never seen these components running together as one system. They know the pieces; they have not watched the assembly.

The arc begins with the assembly lesson (Lesson 15). The student has CLIP, U-Net with conditioning, and VAE as separate mental models. This lesson puts them in one room and traces a single generation from text prompt to pixel image, with real tensor shapes at every stage. The emphasis is on data flow: what tensor leaves CLIP, what tensor enters the U-Net, where cross-attention and CFG happen in the loop, and how the VAE decodes the result. This is CONSOLIDATE -- zero new concepts, pure integration. The student's reward is seeing their accumulated knowledge form one coherent system.

Then the module addresses the remaining pain point: DDPM requires 1000 denoising steps because it was designed as a Markov chain where each step depends on the previous one. Lesson 16 introduces DDIM (which reinterprets the same trained model as a non-Markov process, enabling large step skips), Euler (the ODE perspective where denoising is following a trajectory), and DPM-Solver (higher-order methods that look ahead for faster convergence). The student understands why 20 steps can produce results comparable to 1000 -- not because of a different model, but because of a different way to traverse the same learned landscape.

The module culminates in Lesson 17, where the student uses the diffusers library to generate images with real Stable Diffusion. This is the capstone: every parameter the student sets (guidance_scale, num_inference_steps, scheduler choice, negative_prompt) maps to a concept they built from scratch across Modules 6.1-6.4. The lesson is not a tutorial -- it is a victory lap where the student proves to themselves that they understand the system.

The emotional arc: "I can see all the pieces working together as one pipeline" -> "So THAT is why 20 steps works" -> "I know what every parameter does because I built each concept from scratch."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| stable-diffusion-architecture | Full pipeline data flow: text prompt -> CLIP -> text embeddings -> VAE encode (training) / random latent (inference) -> U-Net denoising loop with cross-attention + CFG -> VAE decode -> pixel image | CONSOLIDATE | Must come first because the student needs to see the assembled system before learning about sampler optimizations (Lesson 16) or using the real pipeline (Lesson 17). Zero new concepts -- pure assembly of existing knowledge from Modules 6.1-6.3. Positioned after 6.3's CONSOLIDATE lesson; two CONSOLIDATE lessons in a row is appropriate because this module IS the integration payoff for three modules of piece-by-piece learning. |
| samplers-and-efficiency | DDIM deterministic sampling, Euler method (ODE perspective), DPM-Solver (higher-order solver). Why 20 steps produces results comparable to 1000. | STRETCH | Genuinely new concepts: DDIM reinterpretation, ODE formulation, higher-order solvers. Must come after the pipeline assembly so the student understands where the sampler fits in the system. Must come before the generation lesson so the student can choose samplers with understanding, not blind defaults. |
| generate-with-stable-diffusion | Using diffusers to generate images with full parameter comprehension | CONSOLIDATE | The capstone. Must come last because every parameter maps to a concept from earlier lessons. No new theory -- the student applies everything by generating real images and understanding why each setting matters. |

## Rough Topic Allocation

- **Lesson 15 (stable-diffusion-architecture):** Full pipeline walkthrough tracing one generation from text prompt to pixel output. Real tensor shapes at every stage (prompt -> tokenizer -> 77 token IDs -> CLIP text encoder -> 77x768 text embeddings -> U-Net denoising loop [z_T 64x64x4 -> cross-attention with text embeddings, adaptive group norm with timestep, CFG with two forward passes -> z_0 64x64x4] -> VAE decode -> 512x512x3 pixel image). Training pipeline vs inference pipeline (encode image vs start from noise). How the frozen-VAE, U-Net, and CLIP connect at the code level. The "two forward passes for CFG" shown concretely in the loop. Component parameter counts for scale intuition.

- **Lesson 16 (samplers-and-efficiency):** Why DDPM needs 1000 steps (Markov chain, each step is a small correction). DDIM as a reinterpretation of the same model (non-Markov, can skip steps, deterministic option). The ODE perspective (denoising as following a trajectory through latent space, not a random walk). Euler method as the simplest ODE solver. DPM-Solver as a higher-order method (takes bigger, more accurate steps). Practical guidance: which sampler, how many steps. Sampler comparison with concrete quality/speed tradeoffs.

- **Lesson 17 (generate-with-stable-diffusion):** Load and use Stable Diffusion via the diffusers library. Map every API parameter to the concept it controls: `guidance_scale` -> CFG (6.3.4), `num_inference_steps` -> sampler steps (6.4.2), `scheduler` -> sampler choice (6.4.2), `prompt` -> CLIP encoding -> cross-attention (6.3.3-6.3.4), `negative_prompt` -> unconditioned direction in CFG, `height`/`width` -> latent dimensions -> VAE decode (6.3.5), `generator` -> seed -> initial z_T. Explore parameter effects by generating and comparing. This is NOT a diffusers tutorial -- it is a demonstration that the student understands the system deeply enough to predict what each parameter will do before running the code.

## Cognitive Load Trajectory

CONSOLIDATE -> STRETCH -> CONSOLIDATE

The module opens with a pure assembly lesson (CONSOLIDATE) that rewards three modules of piece-by-piece learning with the full picture. The sampler lesson (STRETCH) introduces genuinely new mathematical ideas (ODE formulation, higher-order methods). The capstone (CONSOLIDATE) is a hands-on generation session with full parameter comprehension. The STRETCH lesson is sandwiched between CONSOLIDATE lessons on both sides, which provides comfortable cognitive bookends around the hardest material.

## Module-Level Misconceptions

- **"Stable Diffusion is one big model trained end-to-end"** -- It is a pipeline of independently trained, frozen components. The CLIP text encoder, the VAE, and the U-Net were trained separately. Only the U-Net is trained on the diffusion objective; the others are frozen and reused. The student knows each piece was trained independently (frozen-VAE from 6.3.5, CLIP from 6.3.3), but may not have internalized how modular the full system is.

- **"Better samplers require retraining the model"** -- DDIM, Euler, and DPM-Solver all work with the EXACT same trained U-Net. The model predicts noise (or a score) at each step; the sampler decides how to use that prediction to take a step. Swapping samplers is a post-training, inference-time decision. The student may assume that making generation faster requires a fundamentally different model.

- **"Fewer steps always means worse quality"** -- Advanced samplers can match DDPM quality at 20-50 steps. The quality-steps relationship is not linear and depends on the sampler. Some samplers even produce better results at moderate step counts than DDPM does at 1000, because they avoid the accumulated noise from stochastic sampling.

- **"The diffusers library is a black box I'm just calling"** -- The entire point of this module is that the student understands what happens inside every API call. `pipe(prompt, guidance_scale=7.5, num_inference_steps=50)` is not magic -- it is CLIP encoding, then a 50-step denoising loop with CFG at scale 7.5, then VAE decoding. Every parameter maps to a concept the student built.

- **"Negative prompts are a separate conditioning mechanism"** -- A negative prompt is simply the text used for the "unconditional" prediction in CFG. Instead of using an empty string for epsilon_uncond, you use the negative prompt. The CFG formula is unchanged: epsilon_cfg = epsilon_neg + w * (epsilon_cond - epsilon_neg). The student already knows the CFG formula from 6.3.4; negative prompts are a direct application, not a new mechanism.
