# Module 6.3: Architecture & Conditioning -- Plan

**Module goal:** The student understands how the components of Stable Diffusion work individually -- U-Net architecture for multi-scale denoising, timestep and text conditioning mechanisms, CLIP for bridging language and vision, and the move from pixel-space to latent-space diffusion -- well enough to explain why each piece exists and how they connect, preparing for full pipeline assembly in Module 6.4.

## Narrative Arc

Module 6.2 ended with a working pixel-space diffusion model that generates real images -- but painfully slowly, with no way to control what it generates. The student felt both the magic and the limitations firsthand. Module 6.3 answers two questions that emerged from that experience: "Why is this architecture shaped like that?" and "How do I tell it what to generate?"

The arc begins by opening the black box that was the neural network in Module 6.2. The student used a minimal U-Net in the capstone but did not build it or deeply understand it. Lesson 10 develops the U-Net architecture in depth -- why encoder-decoder for denoising, why skip connections are not just nice-to-have but essential, and how the multi-resolution structure maps to the coarse-to-fine denoising progression the student already observed. This is re-entering theoretical mode after a pure implementation capstone, so the lesson rebuilds analytical thinking with familiar architectural concepts from Series 3.

Lesson 11 adds the first conditioning mechanism: how does the network know what noise level it is denoising? Timestep embeddings (connecting to positional encoding from Series 4.1) and adaptive normalization give the U-Net noise-level awareness. This completes the "unconditional" diffusion architecture.

Then the module pivots to the control question. Lesson 12 introduces CLIP -- a genuinely new training paradigm (contrastive learning) that creates a shared embedding space for text and images. This is the STRETCH lesson: contrastive learning is unfamiliar, but the embedding space concept connects to latent spaces from Module 6.1.

Lesson 13 connects CLIP to the U-Net via cross-attention (review from Series 4.2, not newly taught) and introduces classifier-free guidance -- the technique that makes text conditioning actually work well. The student understands how text descriptions steer the denoising process.

The module culminates in Lesson 14, which circles back to the pain point from the Module 6.2 capstone: pixel-space diffusion is too slow. The solution is to compress images with a VAE first (re-activating concepts from Module 6.1 that have been dormant for ~10 lessons) and run diffusion in the latent space. This is the conceptual bridge to Stable Diffusion itself.

The emotional arc: "Oh, THAT is why the architecture looks like that" -> "Now I see how it knows the noise level" -> "Wait, you can train on text-image pairs without labels?" -> "Cross-attention lets me steer generation with words" -> "And running it in latent space makes it fast. That is Stable Diffusion."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| unet-architecture | U-Net as encoder-decoder with skip connections for multi-scale denoising | BUILD | First lesson after a CONSOLIDATE capstone. Opens the black box from Module 6.2. Re-enters theoretical mode but with familiar architectural concepts (encoder-decoder from 6.1, skip connections from 3.2). Must come before conditioning because the student needs to understand the architecture before learning how to inject information into it. |
| conditioning-the-unet | Timestep embeddings and adaptive normalization -- how the network knows the noise level | BUILD | Extends the architecture from Lesson 10 with the conditioning mechanism. Timestep embedding connects to positional encoding from 4.1 (familiar concept, new application). Must come before text conditioning because timestep conditioning is simpler and establishes the pattern of "injecting information into the U-Net." |
| clip | Contrastive learning and CLIP's shared text-image embedding space | STRETCH | Genuinely new training paradigm (contrastive learning). Must come before text conditioning because CLIP produces the text embeddings that get injected into the U-Net. The embedding space concept connects to latent spaces from 6.1, grounding the new idea in familiar territory. |
| text-conditioning-and-guidance | Cross-attention for text injection, classifier-free guidance for steering generation | BUILD | Cross-attention is review from 4.2 (new application, not new concept). Classifier-free guidance is the genuinely new concept but is procedurally simple. Must come after CLIP (needs text embeddings) and after U-Net architecture (needs to understand where cross-attention lives). |
| from-pixels-to-latents | VAE compression + latent-space diffusion as the speed solution | CONSOLIDATE | Connects the pain point from Module 6.2 capstone (pixel-space is slow) to the solution (compress with VAE, diffuse in latent space). Re-activates VAE concepts from Module 6.1 (~10 lessons ago, per Reinforcement Rule). Positioned last because it synthesizes everything: U-Net operates in latent space, conditioning mechanisms stay the same, but now it is fast. Prepares for full pipeline assembly in 6.4. |

## Rough Topic Allocation

- **Lesson 10 (unet-architecture):** Why encoder-decoder for denoising (connects to autoencoder from 6.1 and CNN encoder-decoder from Series 3), why skip connections are essential (not optional -- without them, fine details lost through the bottleneck), multi-resolution feature hierarchy mapping to coarse-to-fine denoising, the downsampling/upsampling path with channel dimension changes, residual blocks within the U-Net. Timestep conditioning is explicitly DEFERRED. Focus is purely on the spatial architecture.

- **Lesson 11 (conditioning-the-unet):** How the network receives the timestep t (sinusoidal embedding, callback to positional encoding from 4.1), why sinusoidal rather than the simple linear projection from the capstone, adaptive group normalization (how timestep information modulates feature maps), the pattern of "global conditioning" (information that affects every spatial location). This completes the unconditional DDPM architecture.

- **Lesson 12 (clip):** Contrastive learning as a training paradigm (the idea of "same pair" vs "different pair"), how CLIP trains on 400M text-image pairs from the internet, the shared embedding space where text and images live, why cosine similarity works as a training signal, what CLIP representations look like. No U-Net integration yet -- this lesson is about CLIP as a standalone concept.

- **Lesson 13 (text-conditioning-and-guidance):** Cross-attention as the injection mechanism (review from 4.2 -- the U-Net's spatial features attend to CLIP text embeddings), where cross-attention layers live in the U-Net (interleaved with self-attention and residual blocks), classifier-free guidance (train with randomly dropped text, run two forward passes at inference, amplify the difference). The new concept is CFG; cross-attention is a reviewed concept in a new context.

- **Lesson 14 (from-pixels-to-latents):** The computational cost problem revisited (felt in capstone, now quantified), compressing to latent space with a pre-trained VAE (re-activate VAE concepts from 6.1), what "diffusing in latent space" means (same noise schedule, same training algorithm, smaller tensors), decoding back to pixel space, why this works (VAE preserves perceptual content). This lesson synthesizes 6.1 (VAE) + 6.2 (diffusion) + 6.3 (architecture) into latent diffusion.

## Cognitive Load Trajectory

BUILD -> BUILD -> STRETCH -> BUILD -> CONSOLIDATE

The module begins gently after the CONSOLIDATE capstone. The first two lessons (U-Net architecture, timestep conditioning) are BUILD -- re-entering theoretical mode but with heavily leveraged prior knowledge (encoder-decoder, skip connections, positional encoding). CLIP is the STRETCH lesson with genuinely new concepts (contrastive learning). Text conditioning is BUILD -- cross-attention is review, and CFG is procedurally simple. The final lesson is CONSOLIDATE -- synthesizing everything into latent diffusion, no new theoretical concepts.

The STRETCH lesson (CLIP) is sandwiched between BUILD lessons on both sides, following the same pattern as Module 6.2.

## Module-Level Misconceptions

- **"The U-Net is just an autoencoder used for a different task"** -- The skip connections fundamentally change the architecture's behavior. An autoencoder forces ALL information through the bottleneck; the U-Net's skip connections let high-resolution details bypass the bottleneck. This is why the U-Net preserves fine details while still processing at multiple scales.

- **"Timestep conditioning is just concatenating t to the input"** -- Simple concatenation does not give the network enough flexibility. The timestep needs to modulate the network's behavior at every layer, not just at the input. Sinusoidal embeddings + adaptive normalization achieve this.

- **"CLIP understands images/text the way humans do"** -- CLIP learns statistical associations between text and image patterns. It can be fooled by adversarial examples, misses spatial relationships, and has biases from its internet training data. It produces useful embeddings, not understanding.

- **"Cross-attention in the U-Net works differently from cross-attention in the Transformer"** -- It is the same mechanism. Query comes from one source (U-Net spatial features), key and value come from another (CLIP text embeddings). The student already knows this from Series 4.2.

- **"Latent diffusion is a fundamentally different algorithm from pixel-space diffusion"** -- The diffusion algorithm is identical. The only change is the input space: instead of noising/denoising pixel tensors, you noise/denoise latent tensors. The VAE handles the translation between pixel space and latent space. Everything the student learned in Module 6.2 still applies.
