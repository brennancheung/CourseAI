# Series 6: Stable Diffusion -- Plan

**Status:** In progress (Module 6.1 complete)
**Prerequisites:** Series 1-3 (complete), Series 4 Modules 4.1-4.2 (attention, cross-attention, embeddings, positional encoding)

## Series Goal

Take the student from "I can classify and understand language models" to "I understand how Stable Diffusion works end-to-end and can customize it" -- building every piece from scratch before assembling the full system.

## Modules

| Module | Title | Lessons | Focus |
|--------|-------|---------|-------|
| 6.1 | Generative Foundations | 4 | Shift from discriminative to generative thinking; autoencoders, VAEs, latent spaces |
| 6.2 | Diffusion | 5 | Core diffusion theory: forward process, DDPM training, reverse sampling, build pixel-space model |
| 6.3 | Architecture & Conditioning | 5 | U-Net, timestep conditioning, CLIP, text-guided generation, latent diffusion |
| 6.4 | Stable Diffusion | 3 | Full pipeline assembly, advanced samplers, generate with real SD |
| 6.5 | Customization | 3 | LoRA fine-tuning, img2img/inpainting, textual inversion |

## Lessons

### Module 6.1: Generative Foundations

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 1 | from-classification-to-generation | From Classification to Generation | You've learned to label images; what does it mean to create them? Generative vs discriminative, sampling from distributions |
| 2 | autoencoders | Autoencoders | Force images through a bottleneck; the network learns what matters. Encoder-decoder, reconstruction loss, latent representations |
| 3 | variational-autoencoders | Variational Autoencoders | Regular autoencoders memorize points; VAEs learn a smooth space you can sample from. KL divergence as regularizer -- scoped to intuition, not full ELBO derivation |
| 4 | exploring-latent-spaces | Exploring Latent Spaces | Interpolate, do arithmetic, generate new images by sampling. First real generative payoff -- "I made something that doesn't exist" |

### Module 6.2: Diffusion

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 5 | the-diffusion-idea | The Diffusion Idea | Destruction is easy, creation is hard -- but you can learn to undo small steps of destruction. Forward and reverse process intuition |
| 6 | the-forward-process | The Forward Process | Noise schedules, Gaussian properties, the closed-form trick that lets you jump to any timestep without iterating |
| 7 | learning-to-denoise | Learning to Denoise (DDPM) | The training objective: predict the noise that was added. MSE loss on noise. Surprisingly simple |
| 8 | sampling-and-generation | Sampling: Images from Noise | The reverse process algorithm. Iteratively denoise pure static into an image, step by step |
| 9 | build-a-diffusion-model | Build a Pixel-Space Diffusion Model | Train DDPM on MNIST/CIFAR. Generate real images. Feel the magic -- and feel why it's painfully slow |

### Module 6.3: Architecture & Conditioning

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 10 | unet-architecture | U-Net: The Diffusion Backbone | Encoder-decoder with skip connections -- "ResNet meets autoencoder." Why it's perfect for denoising at multiple scales |
| 11 | conditioning-the-unet | Conditioning the U-Net | Timestep embeddings (callback to positional encoding from 4.1), adaptive normalization -- how the network knows what noise level to denoise |
| 12 | clip | CLIP: Connecting Language and Vision | Contrastive learning: text-image pairs into a shared embedding space. A new training paradigm the student hasn't seen |
| 13 | text-conditioning-and-guidance | Text Conditioning & Classifier-Free Guidance | Cross-attention injects text into U-Net (review from 4.2). CFG: run conditioned + unconditioned, amplify the difference |
| 14 | from-pixels-to-latents | From Pixels to Latents | Pixel-space is too slow (felt in lesson 9). Compress with a VAE first (re-activate from 6.1), diffuse in latent space, decode back |

### Module 6.4: Stable Diffusion

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 15 | stable-diffusion-architecture | The Stable Diffusion Architecture | CLIP + U-Net + VAE assembled. Full pipeline walkthrough with real tensor shapes |
| 16 | samplers-and-efficiency | Samplers & Efficiency | DDIM, Euler, DPM-Solver -- why 20 steps works when DDPM needed 1000 |
| 17 | generate-with-stable-diffusion | Generate with Stable Diffusion | Use diffusers to generate images. You know what every parameter means because you built the concepts |

### Module 6.5: Customization

| # | Slug | Title | Description |
|---|------|-------|-------------|
| 18 | lora-finetuning | LoRA Fine-Tuning | Low-rank adaptation: inject small trainable matrices into frozen weights. Train a style/subject LoRA on your own images |
| 19 | img2img-and-inpainting | Img2img & Inpainting | Start from a noised image instead of pure noise. Mask regions for selective editing |
| 20 | textual-inversion | Textual Inversion | Teach the model new concepts by optimizing a token embedding -- no weight changes |

## Scope Boundaries

### In scope

- Generative model foundations (autoencoders, VAEs)
- Diffusion theory (DDPM, forward/reverse process, noise schedules)
- U-Net architecture and conditioning
- CLIP and contrastive learning
- Cross-attention text conditioning and classifier-free guidance
- Latent diffusion / Stable Diffusion architecture
- Samplers (DDIM, Euler, DPM-Solver)
- LoRA fine-tuning for diffusion models
- Img2img, inpainting
- Textual inversion

### Out of scope (Series 7)

- ControlNet
- SDXL
- Consistency models
- Flow matching
- Score-based / SDE formulation (mentioned for context, not derived)

### Out of scope (general)

- GANs (mentioned briefly for landscape context, not built)
- Normalizing flows
- Autoregressive image generation
- Video diffusion
- 3D generation
- Full ELBO derivation (VAE scoped to intuition)

## Connections to Prior Series

| SD Concept | Earlier Concept | Source |
|------------|----------------|--------|
| Encoder-decoder architecture | CNN architecture, skip connections | Series 3 |
| U-Net skip connections | ResNet residual connections | Series 3.2 |
| Reconstruction loss | MSE loss | Series 1.1 |
| VAE KL regularization | Regularization (L2, dropout) | Series 1.3 |
| Timestep conditioning | Positional encoding | Series 4.1 |
| Cross-attention in U-Net | Cross-attention mechanism | Series 4.2 |
| Text embeddings (CLIP) | Token embeddings | Series 4.1 |
| LoRA fine-tuning | Transfer learning, fine-tuning | Series 3.2-3.3 |
| Training loop, optimization | Full PyTorch pipeline | Series 2 |

## Key Pedagogical Notes

- **VAE depth:** Lesson 3 scoped to intuition (KL as "keep the space smooth"), not full ELBO derivation. Enough to use VAEs as a tool; mathematicians can go deeper on their own.
- **Deliberate pain in lesson 9:** The pixel-space diffusion model is deliberately slow/painful to train and generate with. This felt experience motivates latent diffusion in lesson 14. Do not optimize away the frustration.
- **Reinforcement rule for VAEs:** Lesson 14 must re-activate VAE concepts. By lesson 14, the VAE was taught ~10 lessons ago (lesson 3). Per the Reinforcement Rule, assume it's fading. Lesson 14 should reconnect, not re-teach from scratch.
- **Cross-attention is review:** Lesson 13's cross-attention is REVIEW from Series 4.2, not newly taught. The new concept is classifier-free guidance. Cross-attention just gets a new application context (text into U-Net instead of decoder attending to encoder).
- **ControlNet stays in Series 7.** Even though it's closely related, it's a post-SD advance.
