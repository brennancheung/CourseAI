# Module 6.1: Generative Foundations — Plan

**Module goal:** The student understands the generative framing (modeling data distributions, sampling to create), builds intuition for latent representations through autoencoders and VAEs, and experiences the first payoff of generating novel images by sampling from a learned latent space.

## Narrative Arc

Everything the student has built so far is discriminative: given an input, produce a label or a score. Classification asks "what is this?"; generation asks "what could exist?" This module bridges that gap by reframing neural networks from classifiers to distribution learners. The arc moves from the conceptual shift (Lesson 1: what does it even mean to generate?), through the encoder-decoder architecture that compresses and reconstructs (Lesson 2: autoencoders), to making that architecture probabilistic so it can be sampled (Lesson 3: VAEs), and finally to the reward of actually generating and exploring (Lesson 4: latent space exploration). The emotional arc: "I've only ever classified" -> "wait, you can learn to reconstruct?" -> "a probabilistic twist makes it generative" -> "I just generated a face by moving through latent space."

## Lesson Sequence with Rationale

| Lesson | Core Concept | Type | Rationale for Position |
|--------|-------------|------|----------------------|
| from-classification-to-generation | Generative vs discriminative framing, probability distributions over data, sampling | BUILD | Must come first: establishes the conceptual shift before any architecture. Without this framing, autoencoders look like a compression trick, not a step toward generation. |
| autoencoders | Encoder-decoder structure, reconstruction loss, bottleneck as learned compression | BUILD | Uses the generative framing to motivate learning representations. Encoder-decoder is a prerequisite for VAE. Reconstruction loss is concrete and familiar (MSE). Two BUILD lessons in a row is fine because the concepts are complementary, not stacking. |
| variational-autoencoders | Probabilistic latent space, KL divergence as regularizer, sampling from the latent space | STRETCH | The hardest lesson in the module. Requires understanding autoencoders AND the generative framing. KL divergence is genuinely new math. Scoped to intuition (not full ELBO derivation). Positioned after two BUILD lessons so the student has headroom. |
| exploring-latent-spaces | Interpolation, arithmetic, generation by sampling | CONSOLIDATE | First generative payoff. Applies VAE concepts hands-on. No new theory — purely about experiencing what the latent space can do. The reward lesson after the STRETCH. Motivates the rest of Series 6: "this is cool, but the images are blurry — how do we do better?" |

## Rough Topic Allocation

- **Lesson 1 (from-classification-to-generation):** Discriminative vs generative framing, P(y|x) vs P(x), what "learning a distribution" means intuitively, sampling as generation, why this is hard (high-dimensional data). Conceptual only, no notebook.
- **Lesson 2 (autoencoders):** Encoder-decoder architecture, bottleneck / latent representation, reconstruction loss (MSE on pixels), undercomplete vs overcomplete, what the bottleneck forces the network to learn. Notebook: build and train an autoencoder on Fashion-MNIST.
- **Lesson 3 (variational-autoencoders):** Why autoencoder latent spaces have gaps, encoding to a distribution (mean + variance), the reparameterization trick (briefly), KL divergence as "keep the latent space organized," ELBO at intuition level only. Notebook: convert the autoencoder to a VAE.
- **Lesson 4 (exploring-latent-spaces):** Sampling random z vectors, interpolation between two images, latent arithmetic (smile vector), visualizing the latent space with t-SNE/UMAP, quality limitations that motivate diffusion. Notebook: explore the trained VAE.

## Cognitive Load Trajectory

BUILD -> BUILD -> STRETCH -> CONSOLIDATE

Two BUILD lessons establish the conceptual and architectural foundations. The STRETCH lesson (VAE) introduces the most challenging new concept (KL divergence, probabilistic encoding). The CONSOLIDATE lesson lets the student play and internalize. No adjacent STRETCH lessons.

## Module-Level Misconceptions

- **"Generative models memorize and regurgitate training data"** — They learn the structure/distribution of the data. Sampling produces novel instances that were never in the training set. Address primarily in Lesson 1 (framing) and Lesson 4 (demonstrate novel samples).
- **"You need a different kind of neural network for generation"** — Same building blocks (linear layers, activations, loss functions), different objective. The encoder in an autoencoder uses the same conv layers from Series 3. Address in Lesson 1 and reinforce in Lesson 2.
- **"Latent space is just dimensionality reduction like PCA"** — PCA finds linear projections; autoencoders learn nonlinear representations that capture higher-order structure. More importantly, a well-organized latent space (VAE) can be *sampled from* — PCA cannot generate. Address in Lesson 2 and Lesson 3.
- **"VAEs and autoencoders are the same thing"** — The probabilistic aspect is the key difference. An autoencoder's latent space has gaps and no smooth structure; a VAE's latent space is continuous and organized, enabling sampling. Address across Lessons 2-4 as the difference becomes experiential.
