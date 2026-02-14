# Module 6.5: Customization -- Plan

**Module goal:** The student can customize Stable Diffusion for specific styles, subjects, and editing tasks -- understanding LoRA fine-tuning for diffusion models, img2img and inpainting as variations of the denoising process, and textual inversion as embedding-space optimization -- and can explain why each technique works by connecting it to the diffusion pipeline they already understand.

## Narrative Arc

Module 6.4 ended with the student generating images using the full Stable Diffusion pipeline with complete parameter comprehension. They know what every component does, how tensors flow through the system, and why each setting matters. But the model only generates what it was trained on. The student cannot make it produce their own style, edit an existing image, or teach it a new concept. This module addresses the three most practical customization techniques.

The arc begins with LoRA fine-tuning (Lesson 18). The student already understands LoRA deeply from Module 4.4 (LLM context: low-rank decomposition, bypass architecture, merge at inference). The genuinely new content is where LoRA gets applied in a diffusion U-Net (cross-attention projections are the primary target), how the training loop differs from LLM fine-tuning (noise-conditioned, with timestep sampling), and the practical workflow for training a style or subject LoRA with your own images. This is BUILD -- the mechanism is familiar, the application context is new.

Then img2img and inpainting (Lesson 19). These do not require any new training -- they are inference-time modifications to the denoising process the student already knows. Img2img starts from a noised version of an existing image instead of pure noise (partial denoising). Inpainting adds a spatial mask so only some regions are denoised while others are preserved. Both connect directly to the forward process (adding noise to an image) and the sampler (denoising from a specific timestep). This is BUILD -- the concepts are reconfigurations of known mechanisms, not new algorithms.

The module ends with textual inversion (Lesson 20). This is the most conceptually surprising technique: instead of modifying the U-Net weights (LoRA) or the inference process (img2img/inpainting), you optimize a new token embedding in CLIP's embedding space to represent a novel concept. The model weights stay completely frozen. This connects to CLIP's shared embedding space, the tokenizer, and cross-attention -- all DEVELOPED concepts. This is STRETCH because the optimization target (a single embedding vector) is conceptually novel, even though the components are familiar.

The emotional arc: "I can make it generate MY style" -> "I can edit and control existing images" -> "I can teach it entirely new concepts without changing any weights."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| lora-finetuning | LoRA applied to diffusion U-Net cross-attention projections; training loop with noise-conditioned loss | BUILD | Must come first because it is the most natural extension of Module 4.4's LoRA lesson (familiar mechanism, new context). Establishes the "customization without full finetuning" theme. Must come before textual inversion because understanding LoRA's weight modification makes the "no weight modification" surprise of textual inversion land harder. |
| img2img-and-inpainting | Start denoising from a noised real image (partial denoising); mask regions for selective editing | BUILD | Comes second because it requires no training at all -- purely inference-time. Contrasts with LoRA (training required) and previews the spectrum of customization approaches. Must come before textual inversion because spatial control via masking is conceptually simpler than embedding-space optimization. |
| textual-inversion | Optimize a single token embedding to represent a new concept; freeze all model weights | STRETCH | Comes last because it is the most surprising technique (optimization target is unusual) and benefits from contrast with LoRA (LoRA changes weights, textual inversion changes embeddings). The module ends with the broadest conceptual payoff: three fundamentally different customization strategies, each modifying a different part of the pipeline. |

## Rough Topic Allocation

- **Lesson 18 (lora-finetuning):** Brief recap of LoRA mechanism from Module 4.4 (low-rank bypass, merge at inference). Where LoRA is applied in the diffusion U-Net (cross-attention W_Q, W_K, W_V, W_out as primary targets, attention to text conditioning pathway). How the training loop differs: sample a random timestep t, noise the target image, predict the noise, MSE loss on noise prediction -- same as DDPM training but with LoRA adapters unfrozen and base U-Net frozen. Training data: 5-20 images for a subject, 50-200 for a style. Practical workflow with diffusers + PEFT. LoRA rank and alpha for diffusion (typically r=4-8, lower than LLM LoRA). Multiple LoRA composition (applying style + subject LoRAs simultaneously by summing the adaptations).

- **Lesson 19 (img2img-and-inpainting):** Img2img: take a real image, encode with VAE, add noise to timestep t_start (using the forward process closed-form from 6.2.2), then denoise from t_start instead of t=T. Strength parameter as the fraction of the denoising process: strength=0.8 means start at 80% noise, major changes; strength=0.2 means start at 20% noise, minor changes. Inpainting: add a spatial mask that preserves original latents in unmasked regions at each denoising step. Why the boundary between masked and unmasked regions blends naturally (the denoising process handles the transition). Practical examples with diffusers.

- **Lesson 20 (textual-inversion):** The idea: create a new pseudo-token (e.g., `<my-cat>`) and optimize its embedding vector while keeping the entire model frozen. Training: for each image of the concept, encode the prompt containing `<my-cat>`, run the diffusion training loop (noise prediction MSE), backpropagate only into the embedding vector. Why it works: CLIP's embedding space is meaningful and continuous (from 6.3.3); a well-placed vector in that space can represent a novel concept. Limitations: less expressive than LoRA (one vector vs thousands of parameters), slower convergence, works best for specific objects/styles. Comparison of all three customization approaches.

## Cognitive Load Trajectory

BUILD -> BUILD -> STRETCH

Two BUILD lessons back-to-back is appropriate here because they are genuinely different types of BUILD. Lesson 18 applies a familiar training technique (LoRA) to a new domain (diffusion). Lesson 19 reconfigures a known inference process (denoising) with new starting conditions (noised image, mask). Neither introduces fundamentally new concepts -- they extend existing ones. The STRETCH at the end (textual inversion) is cushioned by two comfortable BUILD lessons and benefits from the contrast with what came before.

## Module-Level Misconceptions

- **"LoRA for diffusion works the same way as LoRA for LLMs"** -- The mechanism is the same (low-rank bypass matrices), but the training loop is fundamentally different: LLM LoRA uses next-token prediction on text, diffusion LoRA uses noise prediction on images at random timesteps. The target modules also differ: LLM LoRA typically targets attention projections in transformer blocks, diffusion LoRA targets cross-attention projections in the U-Net where text interacts with spatial features.

- **"Img2img modifies the original image directly"** -- Img2img does not edit pixels. It encodes the image to latent space, adds noise (partially destroying it), then denoises with the text prompt guiding the result. The output is a new image influenced by both the original structure and the prompt. Higher strength means more noise, more creative freedom, less resemblance to the original.

- **"Inpainting uses a separate model"** -- Standard inpainting uses the same U-Net. The mask simply controls which latent regions are denoised versus preserved at each step. (Specialized inpainting models do exist but are fine-tuned variants, not a different architecture.)

- **"Textual inversion changes the model's weights"** -- This is the defining feature of textual inversion: zero weight changes. Only a single embedding vector is optimized. The entire U-Net, VAE, and CLIP text encoder remain frozen. The new concept exists entirely as a position in CLIP's embedding space.

- **"More images always means better LoRA results"** -- For subject LoRA (learning a specific face, object), 5-20 high-quality images often outperform hundreds of low-quality ones. Overfitting is the primary risk with too many training steps, not too few images. This connects to the "finetuning is a refinement" mental model from Module 4.4.
