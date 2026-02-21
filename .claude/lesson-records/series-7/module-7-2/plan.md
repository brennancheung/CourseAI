# Module 7.2: The Score-Based Perspective -- Plan

**Status:** Complete
**Prerequisites:** Module 7.1 (complete), Module 6.2 (DDPM forward/reverse process), Module 6.4 (samplers, ODE perspective), Module 1.1 (gradient descent)

## Module Goal

The student can explain the score function as the gradient of log probability that unifies diffusion models, connect DDPM's discrete noise schedule to a continuous SDE framework, and understand flow matching as a simpler alternative that enables straighter, fewer-step generation paths.

## Narrative Arc

The student finished Module 7.1 with practical mastery of controllable generation (ControlNet, IP-Adapter). Now we shift from "how to use diffusion" to "why diffusion works at a deeper level"--the theoretical reframing that explains why the field moved beyond DDPM.

**Beginning (Lesson 4):** The student has a working model of DDPM: add noise in T discrete steps, train a network to predict noise, reverse the process step by step. The sampler lesson (6.4.2) hinted that something deeper is going on--the model's predictions define a vector field, and samplers are ODE solvers following trajectories. But what IS that vector field, really? The score function answers this: it is the gradient of log probability, and it points toward higher-probability regions of data space. This reframes everything the student already knows about diffusion into a unified continuous-time picture where the forward process is an SDE (noise destroys structure continuously) and the reverse process is another SDE (the score function guides reconstruction). The ODE perspective from 6.4.2 gets formalized as the probability flow ODE.

**End (Lesson 5):** With the score function and SDE/ODE framework in place, flow matching emerges as a natural simplification: instead of curved diffusion trajectories, use straight paths between noise and data. This is simpler to train, more stable, and enables fewer sampling steps. The student understands why modern architectures (SD3, Flux) use flow matching instead of DDPM-style training.

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| score-functions-and-sdes | Score function as gradient of log probability; SDE/ODE duality for diffusion | STRETCH | Must come first: introduces the theoretical vocabulary (score function, SDE, probability flow ODE) that flow matching builds on. Extends the ODE perspective INTRODUCED in 6.4.2 to a formal framework. |
| flow-matching | Conditional flow matching; straight paths between noise and data; rectified flow | BUILD | Uses the score/SDE/ODE framework from lesson 4. The "straight paths" insight only makes sense in contrast to the "curved trajectories" of diffusion SDEs. Relief after STRETCH. |

## Rough Topic Allocation

**Lesson 4 (score-functions-and-sdes):**
- The score function: what it is (gradient of log probability), geometric intuition (points toward higher probability), connection to DDPM's noise prediction (epsilon_theta IS a scaled score)
- SDE forward process: generalizes DDPM's discrete steps to continuous time, Brownian motion as continuous noise addition
- Reverse SDE: score function guides the reverse process, Anderson's result (reverse of an SDE is an SDE that depends on the score)
- Probability flow ODE: deterministic version of the reverse SDE, formalizes the ODE view from 6.4.2
- NOT: Ito calculus, Fokker-Planck equation, score matching training objective derivation

**Lesson 5 (flow-matching):**
- Curved vs straight trajectories: diffusion paths curve through high-dimensional space, what if we straightened them?
- Conditional flow matching: define straight-line paths between noise and data, train a network to predict the velocity along these paths
- Velocity prediction vs noise prediction vs score prediction: different parameterizations of the same underlying field
- Rectified flow: iteratively straighten trajectories for even fewer steps
- Connection to modern architectures: why SD3 and Flux use flow matching

## Cognitive Load Trajectory

| Lesson | Load | Notes |
|--------|------|-------|
| score-functions-and-sdes | STRETCH | 2-3 genuinely new concepts (score function, SDE framework, probability flow ODE). The most theoretical lesson in the series. Mitigated by connecting every concept to existing knowledge: score connects to gradients (Series 1), SDE connects to DDPM forward process (6.2), probability flow ODE connects to sampler lesson (6.4.2). |
| flow-matching | BUILD | 2 new concepts (conditional flow matching, rectified flow) but they build directly on the score/SDE framework from lesson 4. Should feel like a natural payoff of the theoretical work. |

## Module-Level Misconceptions

- **"Score-based models are a different thing from diffusion models"** -- Score-based and diffusion models are the same thing viewed from different angles. DDPM is a discrete approximation of a score-based continuous process. The score function was hiding inside DDPM all along.

- **"The SDE/ODE framework requires understanding stochastic calculus"** -- At the intuition level, an SDE is just "DDPM's forward process with infinitely small steps." You do not need Ito calculus to understand what the score function does or why flow matching works.

- **"Flow matching is an entirely new paradigm that has nothing to do with diffusion"** -- Flow matching is a different training objective for the same kind of generative model. It produces a vector field, just like score-based diffusion. The difference is the trajectory shape (straight vs curved) and the training simplicity.

- **"The score function is an abstract mathematical concept with no physical intuition"** -- The score function points in the direction of increasing data probability. If you are at a noisy image, the score says "move this way to make the image more likely." It IS the denoising direction.

- **"Continuous time means you need infinitely many steps at inference"** -- The continuous-time framework is about modeling, not inference. The whole point of the ODE perspective is that you can take large steps with good solvers (the student already knows this from 6.4.2 / DPM-Solver).
