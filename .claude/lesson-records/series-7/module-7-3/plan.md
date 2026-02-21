# Module 7.3: Fast Generation -- Plan

**Status:** In progress
**Prerequisites:** Module 7.2 (complete -- score functions, SDE/ODE duality, flow matching), Module 6.4 (samplers, ODE perspective, Euler's method), Module 6.2 (DDPM forward/reverse process, noise prediction training)

## Module Goal

The student can explain how consistency models, latent consistency distillation, and adversarial distillation collapse the multi-step generation process into 1-4 steps, understanding the self-consistency property, the distillation pattern, and the quality-speed-flexibility tradeoffs across all acceleration approaches.

## Narrative Arc

The student finished Module 7.2 understanding two ways to define generation trajectories: the diffusion ODE (curved, score-guided) and flow matching (straight, velocity-guided). Both produce high-quality samples, but both still require multiple ODE solver steps--even flow matching needs 20-30. The last line of the flow matching lesson posed the question: "What if you could collapse the ENTIRE trajectory into a SINGLE step?" That is where this module begins.

**Beginning (Lesson 6 -- consistency-models):** The student knows that generation follows a trajectory from noise to data, and that ODE solvers step along this trajectory. Consistency models ask: what if a network could learn to jump directly from ANY point on the trajectory to the endpoint, without stepping? The self-consistency property ("any two points on the same ODE trajectory should map to the same clean image") is the key insight. This is not about better solvers or straighter paths--it is about bypassing the trajectory entirely. The lesson introduces consistency training (learning from the ODE directly) and consistency distillation (learning from a pretrained diffusion model's trajectory), with a toy 2D exercise to build intuition.

**Middle (Lesson 7 -- latent-consistency-and-turbo):** The consistency idea meets real-world scale. Latent Consistency Models (LCM) apply consistency distillation to latent diffusion models (SD/SDXL), enabling 1-4 step generation from existing checkpoints. SDXL Turbo takes a different approach: adversarial diffusion distillation (ADD), using a GAN-style discriminator to push samples toward realism in 1-4 steps. The student sees two distillation strategies side by side, both solving the same problem with different trade-offs.

**End (Lesson 8 -- the-speed-landscape):** Consolidation. The student has now seen four acceleration approaches: better ODE solvers (6.4.2), flow matching with fewer steps (7.2.2), consistency distillation (7.3.1-7.3.2), and adversarial distillation (7.3.2). This lesson organizes them into a coherent taxonomy with clear quality-speed-flexibility tradeoffs. It is a map lesson, not a teaching lesson--the student already knows the pieces and now sees how they fit together.

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| consistency-models | Self-consistency property on ODE trajectories; consistency training vs consistency distillation | STRETCH | Must come first: introduces the foundational concept (self-consistency) that all subsequent speed methods in this module build on. Requires the ODE trajectory view from 7.2. STRETCH because the self-consistency property is a genuinely new idea that has no direct analogue in prior lessons. |
| latent-consistency-and-turbo | Consistency distillation at scale (LCM); adversarial diffusion distillation (ADD/SDXL Turbo) | BUILD | Uses the consistency concept from lesson 6 and applies it at scale. LCM is consistency distillation on latent diffusion (connecting to SD from 6.4). ADD introduces adversarial distillation as a contrasting approach. Two practical methods that build on one theoretical foundation. |
| the-speed-landscape | Taxonomy of all acceleration approaches; quality-speed-flexibility tradeoffs | CONSOLIDATE | Organizing lesson. No new concepts. Synthesizes better solvers (6.4.2), flow matching (7.2.2), consistency distillation (7.3.1-7.3.2), adversarial distillation (7.3.2) into a decision framework. Should feel like a satisfying "now I see the whole picture" moment. |

## Rough Topic Allocation

**Lesson 6 (consistency-models):**
- The self-consistency property: any point on the same ODE trajectory maps to the same clean image endpoint
- Self-consistency as a training objective: the consistency function f(x_t, t) should be constant along any trajectory
- Boundary condition: f(x_0, 0) = x_0 (the endpoint IS the clean image)
- Consistency training: learning directly from the ODE using adjacent timesteps (no pretrained teacher needed)
- Consistency distillation: learning from a pretrained diffusion model's ODE trajectory (faster, better results)
- Multi-step generation: use a small number of consistency steps for progressive refinement
- NOT: full mathematical derivations of the consistency training loss, latent diffusion integration, adversarial methods

**Lesson 7 (latent-consistency-and-turbo):**
- Latent Consistency Models (LCM): applying consistency distillation to latent diffusion (SD/SDXL)
- LCM-LoRA: LoRA-based LCM that can be plugged into existing SD/SDXL checkpoints
- 1-4 step generation from existing SD/SDXL models
- Adversarial diffusion distillation (ADD) in SDXL Turbo: GAN-style discriminator loss alongside diffusion loss
- ADD vs consistency distillation: teacher signal (ODE consistency vs discriminator realism)
- NOT: training these models, implementation details of the discriminator, full ADD loss derivation

**Lesson 8 (the-speed-landscape):**
- Taxonomy of acceleration approaches: better solvers, straighter paths, trajectory collapse, adversarial distillation
- Quality-speed-flexibility tradeoffs for each approach
- When to use what: decision framework based on quality requirements, step budget, model compatibility
- Connecting back to Series 6 (better solvers) and Module 7.2 (flow matching)
- NOT: new techniques, deep technical details (this is synthesis, not new teaching)

## Cognitive Load Trajectory

| Lesson | Load | Notes |
|--------|------|-------|
| consistency-models | STRETCH | 2 genuinely new concepts (self-consistency property, consistency distillation). The self-consistency property is conceptually novel--the student has not seen anything quite like "collapse the trajectory to its endpoint." Mitigated by connecting to the ODE trajectory the student already understands well. |
| latent-consistency-and-turbo | BUILD | 2 new concepts (LCM as consistency distillation at scale, adversarial diffusion distillation) but both are applications of patterns the student knows: distillation from 7.3.1, adversarial training from GAN context. Follows STRETCH appropriately. |
| the-speed-landscape | CONSOLIDATE | 0 new concepts. Pure synthesis and organization. Follows BUILD appropriately. Provides a satisfying conclusion to the module and a consolidation checkpoint for the series. |

## Module-Level Misconceptions

- **"Consistency models are a type of diffusion model that just uses fewer steps"** -- Consistency models are NOT running the ODE with fewer steps. They learn a direct mapping from any noise level to the clean endpoint. The trajectory is bypassed, not shortened. This is fundamentally different from better solvers (which step along the trajectory faster) or flow matching (which makes the trajectory straighter).

- **"Consistency models and flow matching are competing approaches to the same problem"** -- They address different aspects of the speed problem. Flow matching makes the trajectory straighter (so ODE solvers need fewer steps). Consistency models bypass the trajectory entirely (map directly to the endpoint). You can combine them: a flow matching model with consistency distillation gets the best of both.

- **"One-step generation means the quality is as good as multi-step"** -- There is a quality-speed tradeoff. One-step consistency model output is good but not as good as 50-step diffusion. Multi-step consistency (2-4 steps) recovers much of the quality. The right number of steps depends on the use case.

- **"Consistency distillation and consistency training are the same thing"** -- Consistency distillation requires a pretrained teacher model and uses it to define the trajectory. Consistency training learns directly from the data without a teacher. Distillation is faster to train and produces better results (because the teacher already knows the trajectory), but it requires an existing model.

- **"Adversarial distillation (ADD/SDXL Turbo) is just GANs with extra steps"** -- ADD combines a GAN-style discriminator with a diffusion loss. The diffusion component provides stable training and diversity (GANs alone suffer from mode collapse). The discriminator provides realism at low step counts (where pure diffusion output is blurry). It is a hybrid, not a GAN.
