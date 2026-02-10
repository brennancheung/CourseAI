# Module 6.2: Diffusion -- Plan

**Module goal:** The student understands the core diffusion mechanism -- gradually destroying images with noise (forward process) and training a neural network to reverse that destruction (reverse process) -- well enough to implement and train a pixel-space DDPM from scratch, generating real images and experiencing both the magic and the slowness that motivates latent diffusion.

## Narrative Arc

Module 6.1 ended with a question: the VAE proved that generation works, but the images are blurry. What does better look like? This module answers that question with a radically different approach. Instead of learning a latent space and sampling from it, diffusion models learn to reverse a gradual destruction process. The arc begins with the key insight that destruction is easy but creation is hard -- unless you break creation into many small, learnable steps (Lesson 5). Then it formalizes the destruction side: adding noise according to a schedule, with a mathematical shortcut that makes training practical (Lesson 6). The turning point is the training objective itself: predict the noise that was added, using MSE loss the student already knows (Lesson 7). Then the student sees the reverse process in action: starting from pure static and iteratively denoising to produce an image (Lesson 8). The module culminates in building a working pixel-space diffusion model that generates real images -- but painfully slowly, creating the visceral motivation for latent diffusion in Module 6.3 (Lesson 9).

The emotional arc: "Interesting idea" -> "The math is surprisingly elegant" -> "Wait, the training objective is just MSE on noise?" -> "This actually works, I can see an image emerging from static" -> "This is amazing but SO SLOW -- there must be a better way."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| the-diffusion-idea | Forward and reverse process intuition -- destruction is easy, undoing small steps of destruction is learnable | BUILD | Must come first: establishes the conceptual framework before any math. Without this intuition, the forward process formula looks arbitrary rather than motivated. Directly extends the "VAE quality ceiling" concept from 6.1. |
| the-forward-process | Noise schedules, Gaussian noise properties, closed-form formula q(x_t\|x_0) | STRETCH | The mathematical formalization of the forward process. Genuinely new: Gaussian reparameterization for noise addition, variance schedules, the alpha-bar shortcut. Positioned after the intuition lesson so the student knows WHY these formulas exist. |
| learning-to-denoise | DDPM training objective: predict the noise, MSE loss on epsilon | BUILD | The "surprisingly simple" moment. After the STRETCH of formal math, this lesson delivers a payoff: the training objective is just MSE loss on noise -- something the student already knows deeply from Series 1. Cognitive relief after the STRETCH. |
| sampling-and-generation | Reverse process algorithm: iteratively denoise from pure noise to image | BUILD | Applies the trained model to actually generate. The algorithm is a loop that the student can trace step by step. Positioned after training so the student understands what the model learned before seeing it used. |
| build-a-diffusion-model | Implement and train DDPM on MNIST/CIFAR; generate images; experience the slowness | CONSOLIDATE | The capstone: no new theory, pure implementation and experiential learning. Deliberately preserves the pain of pixel-space diffusion (slow training, slow sampling, 1000 steps). This felt frustration motivates latent diffusion in Module 6.3. |

## Rough Topic Allocation

- **Lesson 5 (the-diffusion-idea):** The VAE quality ceiling as motivation, the "destruction is easy, creation is hard" insight, breaking creation into small steps, forward process as gradual noise addition, reverse process as learned denoising, the key bet that "undo a tiny bit of noise" is learnable. Conceptual only, no notebook. Connects to VAE blurriness from 6.1 Lesson 4.
- **Lesson 6 (the-forward-process):** Gaussian noise and its properties (why Gaussian), noise schedules (linear beta schedule), the variance-preserving formulation, the closed-form shortcut (q(x_t|x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon), what images look like at various timesteps. Interactive widget showing the noise schedule and its effect on images.
- **Lesson 7 (learning-to-denoise):** The DDPM training objective, why predict noise rather than the clean image, the simplified loss (MSE between predicted and actual noise), the training algorithm (sample image, sample timestep, sample noise, predict, compute loss). Connection to MSE loss from Series 1.
- **Lesson 8 (sampling-and-generation):** The reverse process algorithm step by step, why you add noise back during sampling (stochastic sampling), the 1000-step sampling loop, visualizing the denoising process from pure static to image. No notebook yet -- understand the algorithm before implementing.
- **Lesson 9 (build-a-diffusion-model):** Implement a simple U-Net (minimal, not the full architecture from 6.3), implement the forward process, implement the training loop, implement the sampling loop, train on MNIST or CIFAR-10, generate images, time the generation (feel the slowness). Full Colab notebook.

## Cognitive Load Trajectory

BUILD -> STRETCH -> BUILD -> BUILD -> CONSOLIDATE

The first lesson is purely conceptual intuition-building (BUILD). The forward process formalization is the hardest lesson with genuinely new math (STRETCH). Then two BUILD lessons: the training objective (surprisingly simple) and the sampling algorithm (procedural but not mathematically heavy). The final lesson is pure implementation and experience (CONSOLIDATE). The STRETCH lesson is sandwiched between BUILD lessons on both sides.

## Module-Level Misconceptions

- **"Diffusion models learn to generate images from scratch"** -- They learn to denoise. Generation happens because pure noise is the starting point of repeated denoising. The model never sees a blank canvas and produces an image; it always starts from something noisy and makes it slightly less noisy.
- **"More noise steps = better quality"** -- The number of steps is a tradeoff. More steps gives the model an easier per-step task, but makes sampling slower. The 1000-step DDPM convention is not optimal; it is the simplest formulation. Advanced samplers in Module 6.4 reduce this dramatically.
- **"The noise added at each step is the same"** -- The noise schedule varies across timesteps. Early steps add small amounts; later steps add more. The schedule shapes the entire process and is a design choice, not a fixed constant.
- **"Diffusion is fundamentally different from everything we've learned"** -- The training loop is familiar: sample data, compute loss (MSE), backpropagate, update weights. The loss function is the same MSE from Series 1. The U-Net is an encoder-decoder like the autoencoder. The conceptual framework is new; the building blocks are not.
- **"You need to iterate through all timesteps during training"** -- The closed-form formula lets you jump to any timestep directly. Training samples random timesteps, not sequential ones. This is the key insight that makes training practical.
