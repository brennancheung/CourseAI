# Lesson: Diffusion Transformers (DiT)

**Slug:** `diffusion-transformers`
**Series:** 7 (Post-SD Advances), **Module:** 7.4 (Next-Generation Architectures), **Position:** Lesson 10 of 11 in series, Lesson 2 of 3 in module
**Cognitive Load:** STRETCH (2-3 genuinely new concepts: ViT on latent patches elevated from MENTIONED to DEVELOPED, adaLN-Zero conditioning, scaling laws for DiT)

---

## Phase 1: Orient (Student State)

### Relevant Concepts the Student Has

| Concept | Depth | Source | Notes |
|---------|-------|--------|-------|
| Full transformer block (MHA + FFN + residual connections + layer norm, shape-preserving: (n, d_model) in and out, stacks N times) | DEVELOPED | the-transformer-block (4.2.5) | The student built mental models of "attention reads, FFN writes," residual stream as cross-layer backbone, pre-norm ordering. Knows the complete block formula: x' = x + MHA(LayerNorm(x)), output = x' + FFN(LayerNorm(x')). Has TransformerBlockDiagram and StackedBlocksDiagram SVGs. GPT-2: 12 blocks, GPT-3: 96 blocks. |
| Multi-head self-attention with Q/K/V projections (h independent heads, each with W_Q^i, W_K^i, W_V^i, dimension splitting d_k = d_model/h, output projection W_O) | DEVELOPED | multi-head-attention (4.2.4) | The student traced a full worked example with 4 tokens, d_model=6, h=2, d_k=3. Understands head specialization, dimension splitting as budget allocation, W_O as learned cross-head mixing. Has the complete formula: MultiHead(X) = Concat(head_1, ..., head_h) W_O. |
| Full GPT architecture end-to-end (token embedding + positional encoding -> N transformer blocks with causal masking -> layer norm -> output projection -> softmax) | DEVELOPED | decoder-only-transformers (4.2.6) | The student can trace the complete architecture, count parameters (GPT-2: 124M), and understands training vs inference asymmetry. Built GPT from scratch in Module 4.3. |
| Layer normalization (normalize across features within a single token, per-token mean/variance, learned gamma/beta) | INTRODUCED | the-transformer-block (4.2.5) | The student knows pre-norm vs post-norm ordering, the gradient argument for pre-norm, and the contrast with batch norm. Knows the formula and that gamma/beta are learnable. Has not implemented it from scratch. |
| Adaptive group normalization for timestep conditioning (AdaGN(x, t) = gamma(t) * Normalize(x) + beta(t), per-block linear projection pattern) | DEVELOPED | conditioning-the-unet (6.3.2) | The student knows the formula, traced a concrete example with specific gamma/beta values at two timesteps, understands the distinction between global conditioning (timestep via adaptive norm) and spatially-varying conditioning (text via cross-attention). Connected to "learned lens" pattern from Q/K/V projections. |
| U-Net encoder-decoder architecture (encoder downsampling, bottleneck, decoder upsampling, skip connections, channel progression 320/640/1280/1280) | DEVELOPED | unet-architecture (6.3.1) | "Bottleneck decides WHAT, skip connections decide WHERE." The student traced dimensions through the architecture. Knows attention layers interleaved at middle resolutions (16x16, 32x32). |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, per-spatial-location conditioning) | DEVELOPED | text-conditioning-and-guidance (6.3.4) | The student can trace Q/K/V through cross-attention, knows the attention weight shape (e.g., [256, 77] at 16x16), understands per-spatial-location text conditioning. Reactivated multiple times across Series 7. |
| Latent diffusion (encode with VAE, diffuse in [4, 64, 64] latent space, decode back to pixels; 48x compression) | DEVELOPED | from-pixels-to-latents (6.3.5) | "Same orchestra, smaller hall." The student understands frozen-VAE pattern, 8x spatial downsampling, why diffusion in latent space is computationally tractable. |
| SDXL as U-Net ceiling (every improvement is conditioning or scale, not architecture; dual text encoders, micro-conditioning, refiner, larger U-Net, higher resolution) | INTRODUCED | sdxl (7.4.1) | "The U-Net's last stand." The student understands SDXL's five changes and why they represent the limit of U-Net-based scaling. Three limitations identified: no scaling recipe, limited text-image interaction, convolutional inductive biases. This lesson directly addresses those limitations. |
| SDXL resolution scaling and quadratic attention cost (256 tokens at 16x16 SD v1.5, 1024 at 32x32 SDXL, 4096 at 64x64; O(n^2) attention) | INTRODUCED | sdxl (7.4.1) | Concrete token counts and quadratic cost relationship. The student understands that higher resolution means dramatically more compute at attention layers. |
| Vision Transformer / ViT (transformer on image patches) | MENTIONED | clip (6.3.3) | Named as one of CLIP's image encoder options. 2-sentence ConceptBlock. The student knows ViT exists and processes patches like tokens but has not seen the patchify operation, positional embedding for patches, or how ViT blocks work. Also referenced in IP-Adapter lesson as CLIP image encoder producing 257 tokens (256 patches + 1 CLS). |
| Flow matching (straight-line interpolation, velocity prediction, fewer ODE steps, same model family as diffusion) | DEVELOPED | flow-matching (7.2.2) | The student trained a 2D flow matching model. Understands curved vs straight trajectories, velocity parameterization, conversion formulas, and that SD3/Flux use flow matching (INTRODUCED in 7.2.2 as independent of architecture change). |
| Pooled text embedding for global conditioning (OpenCLIP ViT-bigG CLS token [1280], injected via adaptive norm alongside timestep) | INTRODUCED | sdxl (7.4.1) | The student knows SDXL uses pooled embedding for global conditioning through the same adaptive norm pathway as timestep. Two injection points, two purposes. |
| "Three levels of speed" framework + decision framework for acceleration | DEVELOPED | the-speed-landscape (7.3.3) | The student has a comprehensive taxonomy of acceleration approaches organized as a four-dimensional decision framework. |
| Micro-conditioning (original_size, crop_top_left, target_size injected via adaptive norm) | INTRODUCED | sdxl (7.4.1) | The student understands micro-conditioning as additional conditioning inputs via the adaptive norm pathway. |

### Mental Models and Analogies Already Established

- **"Attention reads, FFN writes"** -- the transformer block's dual-function structure. Attention gathers context, FFN transforms it. Both needed. The student built this from scratch.
- **"Residual stream as backbone"** -- the central highway flowing from embedding to output through every sub-layer. Each sub-layer reads and annotates. 24 residual additions in GPT-2 provide direct gradient highway.
- **"The U-Net's last stand"** -- SDXL as the final refinement of the U-Net paradigm. Every improvement goes IN, AROUND, or ALONGSIDE the U-Net.
- **"Bottleneck decides WHAT, skip connections decide WHERE"** -- U-Net dual-path information flow.
- **"WHEN/WHAT/WHERE/WHAT-IT-LOOKS-LIKE/AT-WHAT-QUALITY"** -- five conditioning dimensions, each targeting a different mechanism.
- **"Same orchestra, smaller hall"** -- latent diffusion as diffusion in compressed space.
- **"Curved vs straight"** -- flow matching's core geometric insight about trajectory shape.
- **"Same landscape, different lens"** -- diffusion SDE, probability flow ODE, flow matching, consistency models as four lenses on the same generative process.
- **"Of Course" chain** -- design insights that feel inevitable given the right framework. Used in ControlNet, IP-Adapter, flow matching.
- **"Learned lens"** -- linear projection as viewing the same input through a different matrix (Q/K/V projections, per-block timestep projection).

### What Was Explicitly NOT Covered

- **ViT architecture details:** The student knows ViT exists (MENTIONED in CLIP lesson, referenced in IP-Adapter as producing 256+1 tokens), but has never seen: the patchify operation (splitting an image into patches and linearly embedding them), patch size and its effect on sequence length, positional embeddings for patches, or how standard transformer blocks process patch tokens. This is the primary elevation target: MENTIONED -> DEVELOPED.
- **adaLN-Zero or adaptive layer norm:** The student knows adaptive GROUP norm from U-Net conditioning (6.3.2) where timestep-dependent gamma/beta modulate GroupNorm. adaLN-Zero is a different mechanism: it modulates LayerNorm (not GroupNorm), adds a GATE parameter (not just scale/shift), and initializes the gate to zero (so each transformer block starts as an identity function). This is genuinely new.
- **Scaling laws for diffusion models:** The student has never seen scaling laws (predictable loss-vs-compute curves). The concept of "double compute, halve loss" as a predictable relationship does not exist in their knowledge. They know transformers scale by stacking blocks (GPT-2 12 blocks -> GPT-3 96 blocks) but not the theoretical framework of scaling laws.
- **DiT architecture or any transformer-based denoising network:** The student has never seen a transformer used as a denoising backbone. Every diffusion model they have encountered uses a U-Net.
- **Removing U-Net inductive biases:** The student has not explicitly considered what convolutions assume (local spatial structure, translational equivariance) vs what attention provides (global receptive field, learned spatial relationships). They implicitly know attention's all-pairs property from Series 4 and convolutions' locality from Series 3, but the tradeoff has not been articulated.

### Readiness Assessment

The student is well-prepared despite the STRETCH designation. They have:

1. **Deep transformer knowledge (DEVELOPED across 6 lessons in Module 4.2):** self-attention from raw dot products through multi-head Q/K/V, transformer block structure, residual stream, layer norm, FFN role, parameter counting. They built GPT from scratch in Module 4.3. This is not surface familiarity -- it is deep, implementation-level understanding.

2. **Deep latent diffusion knowledge (DEVELOPED across Series 6):** the full pipeline, conditioning mechanisms (timestep via adaptive norm, text via cross-attention), U-Net architecture with tensor shapes traced, latent space mechanics. They have built a diffusion model from scratch.

3. **The SDXL lesson establishing the motivation (INTRODUCED in 7.4.1):** "The U-Net's last stand" framing with three explicit limitations (no scaling recipe, limited text-image interaction, convolutional inductive biases). The student is primed to see DiT as the answer to these limitations.

4. **ViT at MENTIONED depth with a clear mental hook:** The student knows ViT processes patches like tokens (from CLIP lesson) and that CLIP's image encoder produces 256+1 patch tokens (from IP-Adapter). The patchify operation will feel familiar once explained: "tokenize the image, just like you tokenize text."

The main challenge is that the STRETCH designation comes from three genuinely new concepts arriving together: (1) ViT applied to latent patches (elevation from MENTIONED to DEVELOPED), (2) adaLN-Zero conditioning, and (3) scaling laws as a motivation for the architecture choice. The deep transformer knowledge from Series 4 substantially mitigates this: the student will recognize self-attention + FFN + residual connections immediately. The genuinely new elements are the patchify operation, the conditioning mechanism (adaLN-Zero), and the scaling argument.

The lesson should feel like "of course this was the next step" -- two familiar threads (transformers from Series 4, latent diffusion from Series 6) converging into one architecture. The student has all the pieces; this lesson shows how they fit together.

---

## Phase 2: Analyze

### Target Concept

This lesson teaches the student to explain how the Diffusion Transformer (DiT) replaces the U-Net's convolutional encoder-decoder with a standard vision transformer operating on latent patches, using adaptive layer norm (adaLN-Zero) for timestep/class conditioning, and why this architecture scales more predictably than U-Nets.

### Prerequisites Table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Transformer block (MHA + FFN + residual + layer norm, shape-preserving, stacks N times) | DEVELOPED | DEVELOPED | the-transformer-block (4.2.5) | OK | DiT blocks are standard transformer blocks. The student needs to recognize MHA + FFN + residual + norm immediately so the lesson can focus on what is different (conditioning via adaLN-Zero) rather than re-teaching the block structure. |
| Multi-head self-attention (Q/K/V projections, dimension splitting, output projection W_O) | DEVELOPED | DEVELOPED | multi-head-attention (4.2.4) | OK | DiT uses standard self-attention on patch tokens. The student needs to understand that the same attention mechanism they know from LLMs now operates on image patch tokens instead of word tokens. |
| Layer normalization (normalize across features, learned gamma/beta, pre-norm placement) | INTRODUCED | INTRODUCED | the-transformer-block (4.2.5) | OK | adaLN-Zero modulates LayerNorm parameters. The student needs to know what LayerNorm does (normalize, then scale with gamma and shift with beta) so they can understand "replace the learned gamma/beta with conditioning-dependent gamma/beta." INTRODUCED depth is sufficient -- they know the formula and role, they do not need to have implemented it. |
| Adaptive group normalization (gamma(t) * Normalize(x) + beta(t), timestep-dependent scale/shift) | DEVELOPED | DEVELOPED | conditioning-the-unet (6.3.2) | OK | adaLN-Zero is the transformer analogue of adaptive group norm. Same idea (make normalization parameters depend on the conditioning signal), different normalization type (LayerNorm vs GroupNorm), with an additional gate parameter. The student's deep understanding of AdaGN makes adaLN-Zero a clear extension. |
| Latent diffusion (diffusion in [4, H, W] latent space via frozen VAE) | INTRODUCED | DEVELOPED | from-pixels-to-latents (6.3.5) | OK | DiT operates on VAE latents, not raw pixels. The student needs to know that the input to the denoising network is a latent tensor [4, H, W], not a pixel image. |
| U-Net architecture (encoder-decoder, skip connections, channel progression, attention at middle resolutions) | INTRODUCED | DEVELOPED | unet-architecture (6.3.1) | OK | Need to understand the U-Net baseline that DiT replaces. The student should be able to identify what DiT removes (encoder-decoder hierarchy, skip connections between encoder and decoder, convolutions) and what it keeps (residual connections within blocks, attention). |
| SDXL as U-Net ceiling with three identified limitations (no scaling recipe, limited text-image interaction, convolutional inductive biases) | INTRODUCED | INTRODUCED | sdxl (7.4.1) | OK | The three limitations from SDXL are exactly what DiT addresses. The student needs this narrative context to understand WHY the field moved to transformers. |
| Vision Transformer / ViT (transformer on image patches) | MENTIONED | MENTIONED | clip (6.3.3) | GAP | ViT is the architectural basis for DiT. It must be elevated from MENTIONED (name-dropped) to DEVELOPED (student can explain patchify, sequence length calculation, positional embeddings for patches, and standard transformer blocks on patch tokens). This is the primary new concept. |
| Cross-attention (Q from spatial, K/V from text, per-spatial-location conditioning) | INTRODUCED | DEVELOPED | text-conditioning-and-guidance (6.3.4) | OK | Need to understand what DiT does NOT use for conditioning. The original DiT paper uses class-conditional generation (not text), and conditions via adaLN-Zero rather than cross-attention. This is important context for the sd3-and-flux lesson where cross-attention returns (as joint attention). |

### Gap Resolution

| Gap | Size | Resolution |
|-----|------|------------|
| ViT at MENTIONED, need DEVELOPED | Medium (related concept exists: the student deeply understands transformers on text tokens from Series 4, and knows ViT exists and processes patches. The gap is specifically: the patchify operation, patch size -> sequence length calculation, and positional embeddings for patches.) | Dedicated section within this lesson. The patchify operation is the image analogue of tokenization, which the student deeply understands. The section will: (1) show how a latent [4, H, W] is split into patches of size p x p, (2) each patch is flattened and linearly projected to d_model dimensions (like a token embedding), (3) positional embeddings are added (same concept from 4.1.3), (4) the result is a sequence of patch tokens fed to standard transformer blocks. Concrete example with numbers: [4, 32, 32] latent with p=2 produces (32/2)^2 = 256 patch tokens, each of dimension 4 * 2 * 2 = 16, projected to d_model. This is the "tokenize the image" analogy. |

### Misconceptions Table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "DiT is a completely new architecture unrelated to what I know" | The name "Diffusion Transformer" and the paradigm shift from U-Nets might suggest something entirely foreign. The student has spent all of Series 6 and 7 with U-Nets. | DiT's transformer blocks are the SAME self-attention + FFN + residual connections + layer norm the student built in Series 4. The only genuinely new element is the conditioning mechanism (adaLN-Zero). Strip away the conditioning and you have a standard ViT. Strip away the patches and you have the same transformer block from GPT. The student can identify every component except the conditioning. | Early in the lesson, after presenting the DiT block diagram. Explicit statement: "You already know every component of this block except one." List the components and check them off against Series 4. |
| "DiT must have skip connections like the U-Net to preserve spatial detail" | The student deeply understands U-Net skip connections as essential for combining global semantics (bottleneck) with local detail (encoder features). They might assume any denoising architecture needs this. | DiT has NO encoder-decoder hierarchy and NO skip connections in the U-Net sense. It has residual connections WITHIN each block (the same x + f(x) pattern from the transformer block in 4.2.5), but NOT skip connections BETWEEN distant layers. The transformer's self-attention at every layer gives every patch token a global receptive field from layer 1 -- there is no need for skip connections to recover spatial detail because spatial detail is never thrown away by downsampling in the first place. | When explaining the architecture difference. ComparisonRow: U-Net (downsample -> bottleneck -> upsample, skip connections bridge resolution levels) vs DiT (same resolution throughout, no downsampling, no skip connections needed). Key insight: U-Net skip connections exist because downsampling loses spatial detail. DiT never downsamples, so there is nothing to skip. |
| "adaLN-Zero is just adaptive group norm applied to LayerNorm" | The student knows adaptive group norm: gamma(t) and beta(t) from a conditioning signal. adaLN seems like the same thing with LayerNorm instead of GroupNorm. | adaLN-Zero has a THIRD parameter: a gate (alpha) that is initialized to zero. This means each DiT block starts as an identity function (the residual connection passes input through unchanged). This is a different design choice from adaptive group norm, which has no gate and starts with nonzero gamma/beta. The zero initialization serves the same purpose as zero convolution in ControlNet: ensure the model starts undamaged, then gradually learns to use the new pathway. The "Zero" in adaLN-Zero is not a name -- it is the critical design choice. | When explaining adaLN-Zero conditioning. Explicit ComparisonRow: adaptive group norm (gamma + beta, no gate, nonzero initialization) vs adaLN-Zero (gamma + beta + alpha gate, zero-initialized gate, block starts as identity). Connect to zero convolution pattern from ControlNet (7.1.1): same principle, different context. |
| "Transformers need more data and compute so they must be worse for images" | The student knows from Series 3 that CNNs have useful inductive biases (translational equivariance, local connectivity) that make them data-efficient for images. Moving to transformers seems like giving up these advantages. | DiT's scaling law results show that transformers SCALE MORE PREDICTABLY than U-Nets. At sufficient scale (which modern diffusion models operate at), the transformer's ability to learn spatial relationships from data outperforms the CNN's hard-coded local assumptions. The tradeoff: CNNs are more data-efficient at small scale, transformers are more capable at large scale. Modern diffusion training datasets are large enough that the crossover has been passed. This mirrors the NLP story: RNNs had useful sequential inductive biases, but transformers won because they scale better with data and compute. | In the scaling laws section. Present the tradeoff honestly: convolutions embed spatial priors that are useful, but at the scale of modern diffusion training, learned spatial relationships beat hard-coded ones. The data efficiency argument was valid for small-scale models but the field has moved past that regime. |
| "Removing the U-Net means starting from scratch -- nothing from the SD architecture carries over" | The student might think DiT is a complete replacement where nothing from the Stable Diffusion pipeline applies. | DiT replaces ONLY the denoising network. The VAE (frozen, encoding to latent space) is unchanged. The noise schedule or flow matching objective is unchanged. The text conditioning mechanism changes (from cross-attention in U-Net to adaLN or joint attention in DiT), but the CLIP/T5 text encoders themselves are unchanged. DiT slots into the same latent diffusion pipeline the student knows from 6.4.1: text encode -> denoise in latent space -> VAE decode. Only the middle box changes. | When positioning DiT within the pipeline. Explicit: "The VAE still encodes to [4, H, W]. The text encoder still produces embeddings. The noise schedule still applies. DiT replaces one component: the denoising network. Everything else is preserved." |

### Examples Planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Patchify operation on a concrete latent tensor: [4, 32, 32] with patch size p=2 produces 256 tokens of dimension 16, projected to d_model=1152 | Positive | Make the patchify operation concrete and numerical. The student sees exactly how an image latent becomes a sequence of tokens -- the same kind of sequence they processed with transformers in Series 4. The specific numbers (256 tokens at p=2 on a 32x32 latent) connect to the token counts from SDXL (256 at 16x16 in SD v1.5). | Uses the format the student is comfortable with (tensor shape tracing from 6.4.1). Connects to specific numbers from prior lessons. The student can verify: (32/2)^2 = 256 tokens, each 4*2*2 = 16 dims before projection. This is "tokenize the image" made concrete. |
| adaLN-Zero conditioning traced through one DiT block: conditioning vector c -> MLP -> (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2) -> scale/shift LayerNorm before MHA, gate MHA output, scale/shift LayerNorm before FFN, gate FFN output | Positive | Make the conditioning mechanism concrete. The student sees that one conditioning vector produces SIX parameters per block (three for the MHA sub-layer, three for the FFN sub-layer), and that the alpha parameters (gates) control how much the block contributes. At initialization (alpha=0), the block is an identity function. | Connects directly to adaptive group norm from 6.3.2 (same idea of conditioning-dependent normalization parameters) and to zero convolution from ControlNet 7.1.1 (same zero-initialization principle). The student has both anchors. Showing all six parameters explicitly prevents the misconception that adaLN-Zero is just AdaGN with different normalization. |
| DiT block vs U-Net residual block side-by-side comparison | Positive (stretch) | Show that DiT's architecture is simpler. The U-Net block has: conv -> GroupNorm -> activation -> conv -> GroupNorm -> activation -> skip connection addition -> cross-attention -> self-attention. The DiT block has: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual. Fewer hand-designed components, no convolutions, no encoder-decoder hierarchy. | Addresses the "completely new architecture" misconception. The student can visually see that DiT removes components (convolutions, skip connections, encoder-decoder) rather than adding them. The transformer block is simpler, not more complex. |
| Negative example: trying to scale a U-Net by "just making it bigger" -- no clear recipe for how to allocate additional parameters (wider channels? more attention layers? deeper encoder? which resolution levels?) vs DiT scaling recipe (increase d_model and N, the same two knobs as GPT-2 -> GPT-3) | Negative | Show WHY the field moved to transformers for the denoising backbone. The U-Net's ad hoc scaling problem was identified in the SDXL lesson. This example makes it concrete: with a U-Net, "double the parameters" requires many engineering decisions. With DiT, "double the parameters" means increase d_model and/or N -- the same recipe that scaled GPT-2 to GPT-3. | Connects the SDXL "no scaling recipe" limitation to DiT's solution. Uses the student's knowledge of GPT-2 -> GPT-3 scaling (same architecture, different scale, from 4.2.6) as the positive example of what a scaling recipe looks like. The student already believes transformers scale well -- this lesson shows the same property applies to image generation. |

---

## Phase 3: Design

### Narrative Arc

The student ended the SDXL lesson feeling that the U-Net has been pushed to its limit. The five SDXL innovations -- dual text encoders, larger U-Net, higher resolution, refiner model, micro-conditioning -- are all about what goes IN, AROUND, or ALONGSIDE the U-Net. The U-Net backbone itself is the same convolutional encoder-decoder from SD v1.5, just bigger. Three limitations were explicitly identified: no scaling recipe (how do you systematically make a U-Net bigger?), limited text-image interaction (cross-attention only at certain resolutions), and convolutional inductive biases (convolutions assume local structure, which is both useful and limiting).

This lesson answers the implicit question from SDXL: what if you replaced the U-Net entirely? The Diffusion Transformer (DiT) by Peebles & Xie (2023) does exactly this. Take the noisy latent, split it into patches (like tokenizing text), feed the patch tokens through standard transformer blocks (the same self-attention + FFN the student built in Series 4), and use adaptive layer norm for conditioning (extending the adaptive group norm the student knows from Series 6). The result is an architecture that is simpler (fewer hand-designed components), more scalable (the same d_model + N recipe that scaled GPT-2 to GPT-3), and achieves better results at sufficient scale.

The lesson should feel like a convergence. The student has two deep knowledge threads -- transformers from Series 4 and latent diffusion from Series 6 -- that have run in parallel. DiT is where they meet. Every component of the DiT block is something the student already knows, except the conditioning mechanism. The lesson's job is to show how these known pieces fit together in a new context, teach the one genuinely new mechanism (adaLN-Zero), and explain why this combination scales better than U-Nets.

### Modalities Planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| **Concrete example** | Tensor shape trace through the DiT pipeline, mirroring the SD v1.5 trace from 6.4.1 and the SDXL trace from 7.4.1. Noisy latent [4, 32, 32] -> patchify with p=2 -> 256 tokens of dim 16 -> linear projection to [256, 1152] -> positional embedding -> N transformer blocks -> unpatchify back to [4, 32, 32]. At each stage, the student can compare shapes to the U-Net pipeline they already know. | The student learned both SD v1.5 and SDXL through tensor shape tracing. Using the same format for DiT makes the architectural change immediately tangible: the tensor shapes are different (a flat sequence instead of a spatial feature map at multiple resolutions), and the student can trace exactly where the change happens. |
| **Visual** | Side-by-side architecture comparison: U-Net (vertical hourglass with encoder, bottleneck, decoder, skip connections, convolutions at every level, attention only at middle resolutions) vs DiT (flat stack of identical transformer blocks, same resolution throughout, no skip connections, attention at every layer). The visual makes it immediately clear that DiT is structurally simpler. | The U-Net is a complex architecture with many custom components. DiT replaces it with a uniform stack. A visual comparison is the most efficient way to communicate "simpler, not more complex." The student can see at a glance: hourglass vs rectangle, many component types vs two repeating components. |
| **Verbal/Analogy** | "Tokenize the image" -- the patchify operation is to images what tokenization is to text. In Series 4, the student saw text go from words to tokens to embeddings to a sequence fed into transformers. DiT does the same thing with images: pixels to patches to embeddings to a sequence. The transformer does not know or care whether its input tokens came from text or image patches. | This analogy leverages the student's deepest knowledge (transformer processing of token sequences from Series 4) to make the patchify operation feel natural rather than foreign. "You already know how transformers process sequences. We are just giving it a different kind of sequence." |
| **Symbolic** | The adaLN-Zero formulas traced concretely: (1) conditioning MLP: c -> (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2), (2) MHA sub-layer: x' = x + alpha_1 * MHA(gamma_1 * LayerNorm(x) + beta_1), (3) FFN sub-layer: output = x' + alpha_2 * FFN(gamma_2 * LayerNorm(x') + beta_2). At initialization: alpha_1 = alpha_2 = 0, so x' = x and output = x'. Block is identity. | The student understands adaptive group norm at formula level. Presenting adaLN-Zero as a formula with the same structure (condition-dependent gamma/beta) plus one addition (the alpha gate) makes the extension clear. The zero-initialization insight is immediately visible in the formula: alpha=0 means the entire MHA/FFN contribution vanishes. |
| **Intuitive** | The "two knobs" scaling intuition: scaling a U-Net is like renovating a house with custom parts (each room is different, each wall has bespoke plumbing). Scaling a transformer is like stacking LEGO: increase the brick size (d_model) or add more layers (N). The DiT paper showed that both knobs produce predictable loss improvements. GPT proved this for text. DiT proves it for images. | Makes the scaling argument visceral rather than abstract. The student already intuitively understands that GPT-2 -> GPT-3 was "same architecture, more scale." This lesson extends that intuition to image generation: same scaling recipe, same predictability, different domain. |

### Cognitive Load Assessment

- **New concepts in this lesson:** 2-3 genuinely new concepts:
  1. **ViT applied to latent patches (patchify -> transformer blocks -> unpatchify)** -- This is the primary new concept. ViT was MENTIONED in the CLIP lesson but never developed. However, the student deeply understands both transformers (Series 4) and latent diffusion (Series 6). The patchify operation is the bridge: "tokenize the image." The conceptual delta is smaller than it appears because the transformer blocks ARE the transformer blocks the student already knows.
  2. **adaLN-Zero conditioning (scale + shift + gate, zero-initialized gate)** -- Genuinely new mechanism, but clearly connected to adaptive group norm (same idea: conditioning-dependent normalization parameters) and zero convolution (same idea: zero initialization for safe training start). The student has both anchors at DEVELOPED depth.
  3. **Scaling laws for DiT (predictable loss-vs-compute curves, two knobs: d_model and N)** -- New argument, but the student already accepts that transformers scale (GPT-2 -> GPT-3). The new element is that this scaling property extends to image generation. This is more of a realization than a new concept.

- **Previous lesson load:** SDXL was BUILD (2-3 concepts extending familiar patterns)
- **Is this appropriate?** STRETCH following BUILD is the standard escalation pattern. The student had a CONSOLIDATE (the-speed-landscape) then BUILD (sdxl), so a STRETCH is well-timed. The stretch is mitigated by the convergence of two deep knowledge threads: the student's transformer knowledge from Series 4 means that ~80% of the DiT architecture is already familiar. The genuinely new elements are patchify (medium delta from tokenization), adaLN-Zero (medium delta from AdaGN + zero conv), and the scaling argument (small delta from existing GPT scaling knowledge).

### Connections to Prior Concepts

| New Concept | Connects To | How |
|-------------|------------|-----|
| Patchify operation (split latent into p x p patches, flatten, project to d_model) | Tokenization + embedding from 4.1.2-4.1.3 | "In Series 4, text goes from words to tokens to embeddings. Here, the latent goes from a spatial grid to patches to embeddings. Same pipeline, different input modality. The transformer does not know whether its tokens came from text or image patches." |
| Standard transformer blocks in DiT (self-attention + FFN + residual + norm) | Transformer block from 4.2.5 | "This is the exact block you built in Series 4. Self-attention reads from other patch tokens ('what is elsewhere in the image?'), FFN processes ('transform what attention found'), residual connection preserves input, layer norm stabilizes. Attention reads, FFN writes -- same mental model." |
| adaLN-Zero (conditioning-dependent gamma, beta, alpha on LayerNorm) | Adaptive group norm from 6.3.2 + zero convolution from 7.1.1 | "AdaGN makes GroupNorm parameters depend on the timestep. adaLN-Zero makes LayerNorm parameters depend on the conditioning -- same idea, different normalization. The 'Zero' part: the gate alpha is initialized to zero, making each block start as an identity function. You saw this zero-initialization principle in ControlNet: start contributing nothing, gradually learn to contribute." |
| No U-Net skip connections needed (self-attention provides global receptive field at every layer) | U-Net skip connections from 6.3.1 + all-pairs attention from 4.2.1 | "U-Net skip connections exist because downsampling destroys spatial detail that the decoder needs. DiT never downsamples -- every layer operates at the same spatial resolution. And self-attention at every layer means every patch token can attend to every other patch token. There is no lost spatial information to recover." |
| Scaling laws for DiT (predictable loss improvement with d_model and N) | GPT-2 -> GPT-3 scaling from 4.2.6 + SDXL "no scaling recipe" from 7.4.1 | "GPT-2 (124M) and GPT-3 (175B) use the same architecture at different scale. The scaling recipe is simple: increase d_model and N. DiT shows the same recipe works for image generation. This directly addresses SDXL's limitation: there is now a clear scaling recipe for the denoising network." |
| Patch size -> sequence length tradeoff (smaller patches = more tokens = higher resolution = quadratic compute) | SDXL token count discussion from 7.4.1 + attention O(n^2) from Series 4 | "In SDXL, you saw that higher resolution means more attention tokens (256 at 16x16, 1024 at 32x32, 4096 at 64x64) with quadratic scaling. DiT has the same tradeoff: smaller patch size p means more tokens (L = (H/p)^2), and attention is O(L^2). Patch size is DiT's resolution knob." |

### Analogies to Extend

- **"Attention reads, FFN writes"** from 4.2.5 -- applies directly to DiT blocks. The student should immediately recognize the dual-function structure.
- **"Tokenize the image"** -- new analogy extending the tokenization concept from 4.1.2. Patches are to images what tokens are to text.
- **"Of Course" chain** from ControlNet, IP-Adapter, flow matching -- can be applied to DiT: (1) U-Nets scale ad hoc, (2) transformers scale predictably, (3) latent diffusion already operates on a fixed-size tensor, (4) a fixed-size tensor can be patchified into tokens, (5) transformer blocks can process those tokens. "Of course the field moved to transformers."
- **"Learned lens"** from Q/K/V projections (4.2.2) and per-block timestep projection (6.3.2) -- adaLN-Zero's MLP is another learned projection, this time mapping the conditioning vector to six per-block modulation parameters.
- **Zero initialization** from ControlNet (7.1.1) and LoRA (4.4.4) -- adaLN-Zero's gate initialized to zero follows the same safety pattern: ensure new components start contributing nothing.

### Analogies That Could Be Misleading

- **"Bottleneck decides WHAT, skip connections decide WHERE"** from U-Net 6.3.1 -- could mislead if the student expects DiT to have an equivalent mechanism. DiT has NO bottleneck (same resolution throughout) and NO skip connections (in the U-Net sense). The student needs to understand that the transformer's global attention at every layer serves both functions simultaneously. Address by explicit contrast: "In the U-Net, spatial detail flows through skip connections because the bottleneck throws it away. In DiT, no information is thrown away -- every layer processes at full resolution."
- **"Same orchestra, smaller hall"** from latent diffusion 6.3.5 -- could mislead if the student thinks DiT changes the latent space. DiT uses the same VAE latent space. The "hall" (latent space) is unchanged; only the "orchestra" (denoising network) is different.

### Scope Boundaries

**This lesson IS about:**
- ViT applied to latent patches: the patchify operation, patch size, sequence length calculation, linear projection to d_model, positional embeddings
- Standard transformer blocks as the denoising backbone: self-attention + FFN + residual + norm, NO convolutions, NO encoder-decoder hierarchy, NO U-Net-style skip connections
- adaLN-Zero: conditioning via adaptive layer norm with scale, shift, AND gate; zero-initialized gate making each block start as identity; connection to adaptive group norm and zero convolution
- Why transformers scale better than U-Nets for diffusion: the DiT scaling law results (loss vs compute), the "two knobs" (d_model, N) recipe, and the contrast with U-Net ad hoc scaling
- Removing U-Net inductive biases: what convolutions assume (local spatial structure) vs what attention learns (global relationships from data); the tradeoff at different scales
- DiT's position within the latent diffusion pipeline: same VAE, same text encoding, same noise process, only the denoising network changes

**This lesson is NOT about:**
- ViT pretraining on classification tasks (ImageNet pretraining, MAE, DINO, etc.) -- DiT is trained from scratch on diffusion, not pretrained as a classifier
- Full DiT training from scratch (no notebook training exercise; computational requirements are too large)
- Every DiT variant (U-ViT, MDT, etc.) -- only the original DiT from Peebles & Xie 2023
- Text conditioning in DiT (the original DiT paper is class-conditional on ImageNet, not text-conditioned; text conditioning via joint attention is covered in the next lesson, sd3-and-flux)
- SD3/Flux architecture (next lesson)
- MMDiT / joint text-image attention (next lesson)
- Detailed training procedures, learning rate schedules, or data requirements
- The complete set of conditioning variants from the DiT paper (in-context conditioning, cross-attention conditioning) -- we focus on adaLN-Zero because it won

**Depth targets:**
- ViT applied to latent patches: DEVELOPED (student can explain patchify, compute sequence length from image size and patch size, trace tensor shapes through the full pipeline)
- adaLN-Zero conditioning: DEVELOPED (student can explain the three parameters per sub-layer, why the gate is zero-initialized, trace conditioning through one block, compare to adaptive group norm)
- DiT scaling laws: INTRODUCED (student understands the qualitative argument that transformers scale more predictably than U-Nets, knows the "two knobs" recipe, but has not seen the actual scaling law curves in detail)
- Removal of U-Net inductive biases (convolutions vs attention tradeoff): INTRODUCED (student understands the tradeoff at a conceptual level, knows that at sufficient scale transformers win, but has not done mathematical analysis)

---

### Lesson Outline

#### 1. Context + Constraints

- "This is the second lesson in Module 7.4. In the previous lesson, you saw SDXL push the U-Net to its limit: dual text encoders, micro-conditioning, a refiner model, higher resolution, a larger backbone. Every innovation was about what goes IN or AROUND the U-Net. The U-Net itself was the same convolutional encoder-decoder. Three limitations stood out: no clear scaling recipe, limited text-image interaction, and convolutional inductive biases."
- "This lesson replaces the U-Net entirely."
- ConstraintBlock: This lesson covers the Diffusion Transformer (DiT) architecture from Peebles & Xie (2023). DiT is class-conditional on ImageNet -- it does NOT use text prompts. Text conditioning (via joint attention) is covered in the NEXT lesson (SD3/Flux). We also do NOT cover ViT pretraining (ImageNet classification), full DiT training, or every DiT variant. We focus on the architecture, the conditioning mechanism, and why it scales.

#### 2. Recap

Brief reactivation of four concepts (one gap fill for ViT deferred to the main explanation):

- **Transformer block** (from 4.2.5): 2-3 sentences. "You built the transformer block in Series 4: LayerNorm -> Multi-Head Attention -> residual -> LayerNorm -> FFN -> residual. 'Attention reads, FFN writes.' Shape-preserving: (n, d_model) in, (n, d_model) out. Stack N of them identically."
- **Adaptive group norm** (from 6.3.2): 2-3 sentences. "In the U-Net, timestep conditioning works through adaptive group normalization: the timestep produces gamma(t) and beta(t) that scale and shift the normalized features. The normalization parameters become conditioning-dependent. Every residual block gets its own projection from the timestep embedding."
- **SDXL limitations** (from 7.4.1): 2-3 sentences. "SDXL identified three limitations of the U-Net approach: (1) no clear scaling recipe -- making a U-Net bigger requires many ad hoc engineering decisions, (2) text-image interaction only at cross-attention layers, not everywhere, (3) convolutions hard-code local spatial structure, which is both a useful prior and a limiting assumption."
- Transition: "What if you replaced the U-Net with an architecture that scales predictably, has global interaction at every layer, and learns spatial relationships from data instead of hard-coding them? You already know such an architecture. You built it in Series 4."

#### 3. Hook

Type: **"Of Course" chain + convergence reveal**

"You have two deep knowledge threads running through this course:"

Two GradientCards:
- **Thread 1: Transformers (Series 4)** (sky blue): "You built GPT from scratch. Self-attention lets every token attend to every other token. FFN transforms what attention finds. The block stacks identically. GPT-2 (124M) scales to GPT-3 (175B) by increasing d_model and N. Same architecture, different scale."
- **Thread 2: Latent Diffusion (Series 6)** (violet): "You built a diffusion model. The denoising network operates on a fixed-size latent tensor [4, H, W]. Timestep conditioning via adaptive normalization. The U-Net processes this tensor with convolutions and attention."

"This lesson is where these two threads meet."

InsightBlock: "The 'Of Course' Chain": (1) The denoising network needs to process a fixed-size tensor. (2) A fixed-size spatial tensor can be split into patches. (3) Patches are just tokens. (4) You already know the best architecture for processing token sequences. (5) Of course the field replaced the U-Net with a transformer.

Challenge: "Before we see how, predict: if you were designing a transformer-based denoising network, what would you need to figure out? Think about what changes when the input is an image latent instead of text tokens." (Expected answers: how to turn the image into tokens, how to inject the timestep, how to get an image back out at the end.)

#### 4. Explain: Tokenize the Image (Patchify)

**The core new concept: ViT on latent patches.**

**Part A: From latent tensor to token sequence**

"In Series 4, text became a token sequence: words -> tokenizer -> token IDs -> embedding lookup -> [n, d_model]. The transformer processes this sequence."

"For images, the equivalent operation is patchify: take the latent tensor [C, H, W] and split it into non-overlapping patches of size p x p."

Concrete example with numbers (trace the shapes):
```
Noisy latent: [4, 32, 32]  (e.g., from a 256x256 image with 8x VAE downsampling)
Patch size: p = 2

Step 1: Split into patches
  32/2 = 16 patches per row, 16 patches per column
  Total patches: 16 x 16 = 256 tokens

Step 2: Flatten each patch
  Each patch is [4, 2, 2] = 4 * 2 * 2 = 16 dimensions per patch
  Sequence: [256 tokens, 16 dims]

Step 3: Linear projection to model dimension
  Project [256, 16] -> [256, d_model] via a learned linear layer (nn.Linear(16, d_model))
  For DiT-XL/2: d_model = 1152
  Sequence: [256, 1152]

Step 4: Add positional embeddings
  Same concept as positional encoding from 4.1.3
  Learned positional embeddings: [256, 1152]
  Final input: [256, 1152]
```

"The transformer now has a sequence of 256 tokens, each with 1152 dimensions. It does not know or care that these tokens came from image patches rather than text. The architecture is the same."

**Part B: Sequence length and patch size**

"The number of tokens depends on the latent resolution and patch size:"

Formula: L = (H/p) x (W/p)

Three concrete examples:
- SD v1.5 latent [4, 64, 64], p=8: L = 8 x 8 = 64 tokens
- SD v1.5 latent [4, 64, 64], p=4: L = 16 x 16 = 256 tokens
- SD v1.5 latent [4, 64, 64], p=2: L = 32 x 32 = 1024 tokens

"Smaller patch size = more tokens = finer spatial detail = quadratic attention cost. This is the same tradeoff you saw in SDXL: more tokens means O(L^2) attention compute. Patch size is the resolution knob for DiT."

Connect to SDXL: "In SDXL, you traced these exact token counts: 256 at 16x16, 1024 at 32x32, 4096 at 64x64. DiT gives you explicit control over this tradeoff via the patch size parameter."

**Part C: Positional embeddings for patches**

"Positional embeddings for patches work the same way as positional encodings for text tokens. Each patch position gets a learned vector added to the patch embedding. Without positional embeddings, the transformer cannot distinguish patch (0,0) from patch (15,15) -- it treats the sequence as an unordered set."

"DiT uses standard learnable positional embeddings (same as GPT-2). The positional information is 2D (row, column) but encoded as a 1D position index after flattening. This is sufficient because the transformer can learn 2D spatial relationships from the data."

#### 5. Check #1

Three predict-and-verify questions:

1. "A DiT model uses patch size p=4 on a latent of size [4, 64, 64]. How many tokens does the transformer process?" (Answer: (64/4) x (64/4) = 16 x 16 = 256 tokens. Each token has dimension 4 * 4 * 4 = 64 before projection.)

2. "If you halve the patch size from p=4 to p=2 on the same latent, what happens to: (a) the number of tokens, (b) the attention compute cost?" (Answer: (a) tokens quadruple from 256 to 1024 because (64/2)^2 = 1024. (b) Attention compute increases by 16x because it is O(L^2): (1024/256)^2 = 16. Same quadratic tradeoff from SDXL, now controlled by patch size.)

3. "A colleague claims: 'DiT's patchify is fundamentally different from tokenization in LLMs.' Is this accurate?" (Answer: The operations are analogous but not identical. Both convert raw input into a sequence of embedding vectors that transformers process. Text tokenization uses a learned vocabulary with discrete token IDs; patchify uses a learned linear projection on continuous pixel values. The key similarity: both produce [n, d_model] sequences that the transformer processes identically. The transformer does not know what kind of tokens it received.)

#### 6. Explain: The DiT Block

**Part A: It is the transformer block you already know**

"A DiT block is a standard transformer block: self-attention + FFN + residual connections + normalization. Here is what you recognize:"

Checklist (checking off against Series 4 knowledge):
- Multi-head self-attention with Q/K/V projections -- same as 4.2.4
- Feed-forward network with expansion (d_model -> 4*d_model -> d_model, GELU) -- same as 4.2.5
- Residual connections (x + f(x)) around both sub-layers -- same as 4.2.5
- Normalization before each sub-layer (pre-norm) -- same as 4.2.5

"Every component is the one you built. The difference: how the normalization parameters are computed."

Address Misconception #1: "You already know every component of this block except the conditioning mechanism. DiT is not a new architecture -- it is your transformer applied to patches with a specific conditioning technique."

**Part B: Architecture comparison**

ComparisonRow: U-Net vs DiT architecture

| | U-Net (SD v1.5, SDXL) | DiT |
|---|---|---|
| Basic unit | Conv residual block + optional attention | Standard transformer block (MHA + FFN) |
| Spatial processing | Multi-resolution (64x64 -> 32x32 -> 16x16 -> 8x8 -> back up) | Single resolution (all patches at same level) |
| Skip connections | Encoder-to-decoder across resolution levels | None (only within-block residual connections) |
| Attention | Only at middle resolutions (16x16, 32x32) -- too expensive at full resolution | Every layer, every patch token |
| Convolutions | Every layer (the primary spatial operation) | None (only in the linear projection for patchify/unpatchify) |
| Spatial inductive bias | Strong (local connectivity, translational equivariance from convolutions) | Minimal (position embeddings only; spatial relationships learned from data) |
| Scaling recipe | Ad hoc (which channels to widen? which resolutions to add attention? how deep?) | Systematic (increase d_model and/or N -- same as GPT) |

Address Misconception #2: "U-Net skip connections exist because downsampling loses spatial detail that the decoder needs to recover. DiT never downsamples. Every layer operates on the full set of patch tokens at the same resolution. There is no lost spatial information to skip across."

InsightBlock: "Notice what DiT removes: convolutions, encoder-decoder hierarchy, skip connections, resolution changes. And what it keeps: self-attention, FFN, residual connections, normalization. The transformer block is simpler than the U-Net block. Fewer hand-designed components, fewer engineering decisions."

#### 7. Explain: adaLN-Zero Conditioning

**Part A: The problem**

"The DiT block needs to know the timestep (and optionally the class label). In the U-Net, this was handled by adaptive group normalization: gamma(t) and beta(t) from the timestep embedding modulate GroupNorm. What is the equivalent for a transformer block that uses LayerNorm?"

**Part B: adaLN -- adaptive layer norm**

"The simplest approach: make LayerNorm parameters depend on the conditioning signal. Instead of learned gamma and beta, predict gamma(c) and beta(c) from the conditioning vector c (which encodes timestep + class label)."

"This is the same idea as adaptive group norm: replace fixed normalization parameters with conditioning-dependent ones. The difference is the normalization type (LayerNorm vs GroupNorm) and the conditioning signal (timestep + class vs timestep alone)."

**Part C: adaLN-Zero -- the gate**

"The DiT paper tested several conditioning approaches. The winner adds one more parameter: a gate alpha."

Formula traced through one DiT block:

```
Conditioning vector c (timestep + class embedding)
  -> MLP -> six parameters: (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)

MHA sub-layer:
  h = gamma_1 * LayerNorm(x) + beta_1      (adaptive layer norm)
  h = MultiHeadAttention(h)                  (standard MHA)
  x' = x + alpha_1 * h                      (gated residual)

FFN sub-layer:
  h = gamma_2 * LayerNorm(x') + beta_2      (adaptive layer norm)
  h = FFN(h)                                 (standard FFN)
  output = x' + alpha_2 * h                  (gated residual)
```

"Six parameters per block, from one conditioning signal. Three per sub-layer: scale (gamma), shift (beta), gate (alpha)."

**Part D: Why zero?**

"At initialization, ALL alpha values are set to zero. This means:"

```
x' = x + 0 * MHA(...) = x
output = x' + 0 * FFN(...) = x'  = x
```

"The entire block is an identity function. The input passes through unchanged."

Connect to zero convolution (ControlNet, 7.1.1): "You have seen this pattern before. In ControlNet, zero convolution ensures the trainable encoder starts contributing nothing to the frozen decoder. In LoRA, the B matrix is initialized to zero so the bypass starts at zero. adaLN-Zero follows the same principle: the transformer block starts as a no-op, then gradually learns what to contribute."

Address Misconception #3: ComparisonRow: Adaptive Group Norm vs adaLN-Zero

| | Adaptive Group Norm (U-Net) | adaLN-Zero (DiT) |
|---|---|---|
| Normalization type | Group Normalization | Layer Normalization |
| Parameters from conditioning | gamma, beta (2 per layer) | gamma, beta, alpha (3 per sub-layer, 6 per block) |
| Gate parameter | No | Yes (alpha) |
| Initialization | gamma=1, beta=0 (standard norm behavior) | alpha=0 (block starts as identity function) |
| Zero-initialization principle | Not used | Core design choice |
| Context in this course | 6.3.2 (conditioning-the-unet) | This lesson |

InsightBlock: "The 'Zero' in adaLN-Zero is not just a name. It is the critical design choice. Every DiT block starts as an identity function: input in, same input out. The model gradually learns which blocks should contribute what. This is the same safety pattern you have seen in ControlNet and LoRA: start contributing nothing, learn what to add."

#### 8. Check #2

Three predict-and-verify questions:

1. "A DiT model has N=28 blocks, each with 6 adaLN-Zero parameters (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2). At initialization, what does the entire N-block model compute?" (Answer: Every block is an identity function because all alpha values are zero. The entire model outputs its input unchanged: patchified latent in, same patchified latent out. The model must LEARN to denoise through training. This is different from the U-Net, where the architecture starts with nonzero contributions from every layer.)

2. "You know adaptive group norm from 6.3.2 uses gamma(t) and beta(t) but no gate. What happens if you add a gate to AdaGN and initialize it to zero?" (Answer: You would get the same identity-at-initialization property for U-Net residual blocks. This was not done historically because the U-Net architecture predates the zero-initialization insight. adaLN-Zero was discovered empirically: the DiT paper compared multiple conditioning variants and adaLN-Zero performed best. The gate+zero-initialization combination outperformed the standard scale+shift.)

3. "A DiT block at initialization: alpha=0, so the block is identity. After training, alpha has learned nonzero values. Some blocks might have small alpha values (near-identity) and others might have large alpha values (strong contribution). Does this remind you of anything from Series 4?" (Answer: This is similar to how different transformer blocks in GPT serve different roles -- some early blocks primarily copy/route information, some later blocks do more transformation. The alpha values provide a learned measure of each block's contribution magnitude. In practice, different blocks specialize for different noise levels and spatial scales.)

#### 9. Explain: From Patches Back to Image (Unpatchify)

Brief section -- this is the inverse of patchify and should feel straightforward.

"After N transformer blocks, the model has a sequence of [L, d_model] patch tokens. To get back to a spatial latent, reverse the patchify operation:"

```
Transformer output: [256, 1152]  (L=256 tokens, d_model=1152)

Step 1: Linear projection to patch dimensions
  [256, 1152] -> [256, 4*p*p] = [256, 16]  via nn.Linear(1152, 16)

Step 2: Reshape to spatial grid
  [256, 16] -> [256, 4, 2, 2] -> rearrange to [4, 32, 32]

Predicted noise (or velocity): [4, 32, 32]  -- same shape as the noisy latent input
```

"The output is a denoised latent (or a noise prediction, or a velocity prediction -- the output parameterization is independent of the architecture). It enters the same diffusion sampling loop and VAE decode that you know from Series 6."

Address Misconception #5: "DiT replaces ONLY the denoising network. The VAE still encodes to [4, H, W]. The text encoder (when used) still produces embeddings. The noise schedule or flow matching objective still applies. DiT slots into the same latent diffusion pipeline: text encode -> denoise in latent space -> VAE decode. Only the middle box changes."

#### 10. Check #3

Two predict-and-verify questions:

1. "After the DiT produces its [4, 32, 32] output, what happens next in the generation pipeline?" (Answer: The same thing as with a U-Net: the output is used in the sampling step (DDPM reverse step, DDIM step, ODE solver step, etc.) to compute the next latent. After all sampling steps, the final latent z_0 is decoded by the frozen VAE to pixel space [3, 256, 256]. The rest of the pipeline is identical.)

2. "Could you use a DiT denoising network with DPM-Solver++ or LCM-LoRA?" (Answer: In principle, yes for any solver (DPM-Solver++ works on any denoising network). For LCM-LoRA specifically, the LoRA would need to target the DiT's attention projections (W_Q, W_K, W_V) instead of U-Net cross-attention projections, and the matrix dimensions would be different. The acceleration approaches from Module 7.3 are solver/training techniques that are architecture-independent in concept, though practical implementations need to match the specific architecture's weight shapes.)

#### 11. Explain: Why Transformers Scale Better

**Part A: The U-Net scaling problem**

"In the SDXL lesson, you identified a limitation: there is no clear recipe for making a U-Net bigger. To double the parameters, you must decide:"

- Wider channels at which resolution levels?
- More attention blocks at which resolutions?
- Deeper encoder or deeper decoder?
- Add attention at higher resolutions (expensive)?
- How do skip connections work with asymmetric encoder/decoder?

"Each decision is a manual engineering choice. There is no 'GPT-2 -> GPT-3' recipe for U-Nets."

**Part B: The transformer scaling recipe**

"Transformers have two primary scaling knobs:"

1. **d_model** -- the hidden dimension per token (wider)
2. **N** -- the number of transformer blocks (deeper)

"GPT-2 (d_model=768, N=12, 124M params) -> GPT-3 (d_model=12288, N=96, 175B params). Same architecture, different scale. The scaling recipe is the same: increase d_model and N."

"The DiT paper tested this systematically:"

DiT model family:
- DiT-S: d_model=384, N=12, ~33M params
- DiT-B: d_model=768, N=12, ~130M params
- DiT-L: d_model=1024, N=24, ~458M params
- DiT-XL: d_model=1152, N=28, ~675M params

"Each model uses the same architecture with different scale parameters. The result: loss decreases predictably as compute increases. Bigger DiT = better images, with a smooth, predictable improvement curve."

Address Misconception #4: "Convolutions embed spatial priors that are genuinely useful: local connectivity and translational equivariance make CNNs data-efficient for small datasets. But at the scale of modern diffusion training (hundreds of millions of images), the transformer's ability to learn spatial relationships from data outperforms the CNN's hard-coded assumptions. The tradeoff: CNNs are more efficient at small scale, transformers are more capable at large scale. Modern diffusion training operates in the regime where transformers win. This mirrors the NLP story: RNNs had useful sequential inductive biases, but transformers won because they scale better."

**Part C: DiT scaling results**

"The DiT paper showed class-conditional ImageNet generation results:"

Key result: DiT-XL/2 (patch size 2, 675M params) achieved FID 2.27 on ImageNet 256x256 class-conditional generation -- state-of-the-art at the time, outperforming the best U-Net models (ADM, LDM).

"The FID number is less important than the trend: bigger DiT = better results, with a smooth relationship. No engineering plateaus, no diminishing returns from ad hoc architectural decisions. The same scaling law behavior that made GPT successful applies to image generation."

InsightBlock: "Two knobs, not twenty. Scaling a U-Net requires deciding which channels to widen, which resolutions to add attention, how deep to make each stage. Scaling a DiT requires choosing d_model and N. The simplicity of the scaling recipe is the architectural advantage."

#### 12. Check #4

Two transfer questions:

1. "Suppose you want to build a DiT twice as large as DiT-XL. What would you change?" (Answer: Increase d_model beyond 1152, increase N beyond 28, or both. The same two knobs. No structural decisions required. Compare: doubling a U-Net requires choosing WHERE to add the parameters -- wider channels at what resolution? More attention blocks where? There is no clear answer.)

2. "The original DiT paper uses class-conditional generation (ImageNet classes, no text). Does the scaling argument depend on this? Would you expect the same scaling behavior with text conditioning?" (Answer: The scaling argument is about the denoising backbone architecture, not the conditioning type. Whether you condition on class labels, text embeddings, or something else, the transformer's scaling properties come from the architecture itself. SD3 and Flux -- which use text conditioning with DiT-style backbones -- confirm this: they scale the transformer and get predictable improvements. This is the subject of the next lesson.)

#### 13. Elaborate: What DiT Changes and What It Preserves

**Part A: The complete DiT pipeline in context**

Brief pipeline trace placing DiT within the full latent diffusion framework:

```
Full DiT Pipeline (class-conditional ImageNet):
1. Sample class label y -> class embedding [d_model]
2. Timestep t -> sinusoidal embedding -> MLP -> timestep embedding [d_model]
3. Combine: c = class_emb + timestep_emb  (conditioning vector)
4. Noisy latent z_t [4, 32, 32] -> patchify -> [256, d_model]
5. Add positional embeddings -> [256, d_model]
6. N DiT blocks with adaLN-Zero conditioned on c -> [256, d_model]
7. Final adaLN-Zero + linear projection -> [256, 4*p*p]
8. Unpatchify -> [4, 32, 32]  (predicted noise or velocity)
9. Sampling step (DDPM, DDIM, etc.) -> next z_{t-1}
10. Repeat steps 4-9 for all timesteps
11. VAE decode z_0 -> [3, 256, 256] image
```

"Steps 1-2 are conditioning preparation. Steps 4-8 are the DiT-specific part (the denoising network). Steps 9-11 are the same as every diffusion model you have seen."

**Part B: What DiT enables for the next generation**

Three GradientCards previewing what the architecture change makes possible:

- **Joint text-image attention (MMDiT)** (emerald): "If the denoising network is a transformer processing a sequence of tokens, you can concatenate text tokens and image tokens into one sequence and let them attend to each other. No more separate cross-attention -- text and image interact through the same self-attention mechanism. This is what SD3 and Flux do."
- **Architecture-agnostic training objectives** (sky blue): "DiT's architecture is independent of the training objective. You can train with DDPM noise prediction, velocity prediction, or flow matching. SD3 uses flow matching (from 7.2.2) with a DiT backbone. Two independent improvements that compose."
- **Clear scaling path** (amber): "From DiT-S to DiT-XL, the same architecture scales smoothly. This path continues: SD3 and Flux use larger DiT variants with billions of parameters. The scaling recipe discovered in DiT directly enables frontier models."

"The next lesson shows how these possibilities converge in SD3 and Flux."

#### 14. Practice (Notebook)

Design: 4 exercises, Guided -> Guided -> Supported -> Independent progression.

**Exercise 1 (Guided): Patchify and Unpatchify**
- Implement the patchify operation: take a random tensor [4, 32, 32], split into patches of size p=2, flatten, project with nn.Linear
- Verify shapes at every step: [4, 32, 32] -> [256, 16] -> [256, d_model]
- Implement unpatchify: reverse the operation
- Verify round-trip: patchify -> unpatchify should recover the original tensor (minus projection)
- Predict-before-run: "How many patches will a [4, 64, 64] latent with p=4 produce?"
- What it tests: the patchify operation is concrete and implementable, not just a diagram

**Exercise 2 (Guided): adaLN-Zero Forward Pass**
- Implement one adaLN-Zero conditioning step: take a conditioning vector, produce (gamma, beta, alpha) via MLP
- Apply to a LayerNorm output: gamma * LayerNorm(x) + beta
- Apply the gate: x + alpha * attention_output
- Verify: at alpha=0, the block output equals the input (identity property)
- Gradually increase alpha and observe how the block's contribution grows
- Predict-before-run: "What will the output be when alpha=0?" (Same as input.)
- What it tests: adaLN-Zero is not abstract -- it is a concrete computation the student can trace

**Exercise 3 (Supported): DiT Architecture Inspection**
- Load a pretrained DiT model from the `DiT` library or HuggingFace
- Inspect the model: count parameters, list layer types, verify no convolutions in the transformer blocks
- Compare to U-Net: print both models' layer summaries side-by-side
- Identify the adaLN-Zero components: find the MLP that produces the six conditioning parameters
- Vary the model size (DiT-S, DiT-B, DiT-L, DiT-XL) and observe parameter count scaling
- What it tests: the DiT architecture is inspectable and matches the lesson's description

**Exercise 4 (Independent): DiT Generation and Scaling Comparison**
- Use a pretrained DiT-XL/2 to generate class-conditional ImageNet images (e.g., class 207 = golden retriever)
- Compare generation quality across DiT model sizes (if VRAM allows: DiT-S vs DiT-XL on the same class and seed)
- Vary the number of sampling steps and classifier-free guidance scale
- Observe: larger DiT = better image quality, consistent with the scaling argument
- What it tests: the scaling argument is not just theory -- the student can see it in generated images

#### 15. Summarize

Key takeaways (echo mental models):

1. **Tokenize the image.** The patchify operation splits the latent into patches and projects them to d_model dimensions -- the image equivalent of text tokenization. The transformer processes patch tokens the same way it processes word tokens. Patch size controls the resolution-compute tradeoff.

2. **The transformer block you already know.** DiT blocks are standard transformer blocks: self-attention + FFN + residual + normalization. No convolutions, no encoder-decoder hierarchy, no U-Net-style skip connections. Every component except the conditioning mechanism is from Series 4.

3. **adaLN-Zero: condition via the normalization.** Conditioning-dependent scale (gamma), shift (beta), and gate (alpha) on LayerNorm -- extending adaptive group norm with a zero-initialized gate. Each block starts as an identity function and learns what to contribute. Same zero-initialization safety pattern as ControlNet and LoRA.

4. **Two knobs, not twenty.** Scaling DiT means increasing d_model and N -- the same recipe that scaled GPT-2 to GPT-3. No ad hoc architectural decisions. Predictable loss improvement with scale. This directly addresses SDXL's "no scaling recipe" limitation.

5. **Same pipeline, different denoising network.** DiT replaces only the U-Net. The VAE, text encoder, noise schedule, and sampling algorithm are unchanged. The latent diffusion pipeline you learned in Series 6 still applies -- with a transformer in the middle instead of a U-Net.

#### 16. Next Step

"DiT showed that transformers can replace U-Nets for diffusion, with better scaling. But the original DiT is class-conditional on ImageNet -- it uses class labels, not text prompts. How do you add text conditioning to a transformer-based denoising network?"

"The next lesson introduces SD3 and Flux. They take DiT's transformer backbone and add three things: (1) T5-XXL text encoder for richer text understanding, (2) MMDiT -- Multimodal Diffusion Transformer -- where text tokens and image tokens are concatenated into one sequence and attend to each other jointly (no separate cross-attention needed), and (3) flow matching as the training objective (from 7.2.2). Every concept from Series 4 (transformers), Series 6 (diffusion, conditioning, latent space), Module 7.2 (flow matching), and this lesson (DiT) converges in one architecture."

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (one gap: ViT at MENTIONED, resolved with dedicated section)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (ViT elevation from MENTIONED to DEVELOPED via dedicated patchify section)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (convergence of two knowledge threads, answering SDXL's three limitations)
- [x] At least 3 modalities planned for the core concept, each with rationale (5 modalities: concrete example, visual, verbal/analogy, symbolic, intuitive)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 1 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load: 2-3 new concepts (ViT on latent patches, adaLN-Zero, scaling laws)
- [x] Every new concept connected to at least one existing concept (patchify to tokenization, adaLN-Zero to AdaGN + zero conv, scaling to GPT, no skip connections to all-pairs attention)
- [x] Scope boundaries explicitly stated

---

## Review — 2026-02-20 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

### Findings

### [IMPROVEMENT] — Positional embedding section is too thin and split awkwardly from the patchify narrative

**Location:** "Sequence Length and Patch Size" section (TipBlock aside) and the outline's planned "Part C: Positional embeddings for patches"
**Issue:** The planning document has a dedicated Part C for positional embeddings (6 sentences explaining 2D positions encoded as 1D, why positions matter, connection to GPT-2). The built lesson reduces this to a 4-sentence TipBlock aside. More importantly, the key insight from the plan is missing: "Without positional embeddings, the transformer treats the sequence as an unordered set." This IS stated in the aside, but positional embeddings for patches are genuinely important for the student to understand how 2D spatial relationships are preserved. Moving this entirely into an aside signals "optional detail" when it is actually a necessary part of understanding how the transformer handles spatial data. The student deeply understands 1D positional encoding from Series 4 (sinusoidal encoding for text token positions). The extension to 2D patch positions deserves at least 2-3 sentences in the main body connecting to what they already know.
**Student impact:** The student may underweight positional embeddings and not fully understand how the transformer knows spatial relationships between patches. This matters for understanding why DiT can learn spatial relationships from data (a core claim of the scaling argument).
**Suggested fix:** Add 2-3 sentences in the main body of the patchify section (after Step 4 in the CodeBlock trace) explaining: (1) positional embeddings work the same as in GPT-2 but encode 2D patch positions, (2) without them the transformer cannot distinguish spatial location, (3) the positions are learned (same as GPT-2), and (4) this is how DiT's "minimal spatial bias" works -- position embeddings provide the only spatial prior, everything else is learned from data. Keep the TipBlock but promote the core explanation to main content.

### [IMPROVEMENT] — The "Tokenize the Image" analogy modality is present but the analogy itself is not developed enough to serve as a lasting mental model

**Location:** "Tokenize the Image" section, first paragraph + InsightBlock aside
**Issue:** The planning document identifies "Tokenize the image" as a key verbal/analogy modality and a new analogy to establish. The lesson states the analogy in the InsightBlock aside ("Same pipeline, different input modality") and in one sentence in the main body ("In Series 4, text became a token sequence: words -> tokenizer -> token IDs -> embedding lookup -> [n, d_model]. The transformer processes this sequence."). But the analogy deserves explicit side-by-side mapping to really land. The student has deep knowledge of the text tokenization pipeline (from 4.1.2 and 4.1.3). A brief parallel mapping -- text: words -> BPE tokenizer -> token IDs -> embedding table -> [n, d_model] vs. image: spatial latent -> patchify -> flattened patches -> linear projection -> [L, d_model] -- would make the structural equivalence concrete and memorable. Currently the lesson states the conclusion ("the transformer does not know or care") without giving the student the full structural comparison to arrive at that conclusion themselves.
**Student impact:** The analogy is stated rather than built. The student hears "patchify is like tokenization" but does not get to see the exact structural correspondence that makes the analogy powerful. The mental model will be weaker than it could be.
**Suggested fix:** Add a brief ComparisonRow or side-by-side list immediately after the first paragraph of "Tokenize the Image" that maps the text pipeline to the image pipeline step by step. Three rows: (1) Input -> tokens/patches, (2) Token -> embedding / Patch -> linear projection, (3) Result -> [n, d_model] sequence. Then the existing paragraph continues with "The transformer now has a sequence..." This takes 5-8 lines and significantly strengthens the analogy.

### [IMPROVEMENT] — Notebook Exercise 3 loads DiT model via raw torch.load rather than a DiT model class, preventing actual architecture inspection

**Location:** Notebook Exercise 3, cells 24-31
**Issue:** Exercise 3 instructs the student to "Load a pretrained DiT model" and "Inspect the model: count parameters, list layer types, verify no convolutions." But the implementation loads the model as a raw state dict via `torch.load()` and then inspects parameter tensor names and shapes. This means the student cannot use `count_parameters_by_type()` (which requires `model.named_modules()`) on the DiT model, only on the U-Net. The exercise's TODO for Step 5 calls `count_parameters_by_type(unet)` but has no equivalent for the DiT because it is not loaded as a model object. The solution in the details block also only shows the U-Net type breakdown, not DiT. This weakens the comparison: the student can inspect the U-Net's module types but can only inspect the DiT's parameter name strings. The planning document says "Load a pretrained DiT model from the `DiT` library or HuggingFace" and "Compare to U-Net: print both models' layer summaries side-by-side" -- this implies both should be loaded as inspectable model objects.
**Student impact:** The side-by-side comparison is asymmetric. The student gets a clean module-type breakdown for the U-Net but must infer DiT's structure from parameter name strings. This weakens the "no convolutions in DiT" verification and the architectural comparison that is the exercise's main purpose.
**Suggested fix:** Load the DiT model as a model object rather than a raw state dict. Options: (1) use `DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")` and access `pipe.transformer` as the model object, or (2) instantiate the DiT model class from the `diffusers` library (e.g., `DiTTransformer2DModel`) and load weights into it. Then `count_parameters_by_type()` works on both models and the side-by-side comparison is symmetric. This also sets up Exercise 4 more smoothly since the pipeline is already loaded.

### [POLISH] — Spaced em dash in CodeBlock text on line 800

**Location:** "From Patches Back to Image" section, unpatchify_trace.txt CodeBlock
**Issue:** The line reads: `Predicted noise (or velocity): [4, 32, 32]  — same shape as the noisy latent input`. The em dash has spaces around it (`  — `). The writing style rule requires `word—word` not `word — word`. This is visible rendered text (inside a CodeBlock), not an HTML comment.
**Student impact:** Minor inconsistency with the rest of the lesson's em dash usage, which is correct elsewhere.
**Suggested fix:** Change to: `Predicted noise (or velocity): [4, 32, 32]—same shape as the noisy latent input`

### [POLISH] — Recap section describes three threads but the header says "Three Threads to Reactivate" while only SDXL limitations is not a "thread" in the course-level sense

**Location:** "Three Threads to Reactivate" section header
**Issue:** The header says "Three Threads to Reactivate" but the items are: (1) the transformer block, (2) adaptive group norm, (3) SDXL's three limitations. Items 1 and 2 are genuine deep knowledge threads (Series 4 and Series 6). Item 3 is a narrative setup from the previous lesson, not a "deep knowledge thread." The next section (the Hook) correctly identifies "Two Threads Converge" as the transformer thread and the latent diffusion thread. Having the recap say "three threads" when the hook says "two threads" creates a minor narrative inconsistency. The recap is really reactivating two deep knowledge threads plus one motivational context.
**Student impact:** Minimal -- the student will understand the content regardless. But the narrative framing slightly overloads the word "thread."
**Suggested fix:** Change the subtitle to "Concepts you already know deeply" (which is already the subtitle) and change the header to something like "Quick Reactivation" or "Three Concepts to Reactivate" instead of "Three Threads to Reactivate." Alternatively, keep it as is -- this is a minor framing issue.

### Review Notes

**What works well:**
- The "convergence of two threads" narrative is genuinely compelling. The lesson successfully makes DiT feel like an inevitable conclusion rather than a surprising new architecture. The "Of Course" chain in the InsightBlock aside is well-constructed.
- The tensor shape trace through patchify (Section 5) is excellent -- it mirrors the format the student is comfortable with from Series 6 and makes the operation concrete with specific numbers.
- The adaLN-Zero section is well-structured: problem statement (how to condition a transformer block), connection to AdaGN (same idea, different normalization type), the gate parameter as the new element, zero initialization with explicit math showing identity property, and the connection to ControlNet zero convolution. This hits all the planned modalities.
- All four planned misconceptions are addressed at their planned locations (Misconception #1 in the DiT Block section, #2 in the architecture comparison, #3 in the adaLN-Zero comparison, #4 in the scaling section, #5 in the unpatchify section).
- The lesson stays within scope boundaries -- it correctly identifies DiT as class-conditional, defers text conditioning to the next lesson, and avoids DiT variants.
- Check questions are well-calibrated: they test prediction (sequence length calculations), boundary testing (what happens when you halve patch size), and transfer (is patchify "fundamentally different" from tokenization).
- The notebook is well-scaffolded: Guided (patchify) -> Guided (adaLN-Zero) -> Supported (architecture inspection) -> Independent (generation). The progression mirrors the lesson arc. Exercises 1 and 2 are excellent -- they verify the exact claims from the lesson with concrete code.
- Cognitive load is appropriate for STRETCH: two genuinely new concepts (ViT on latent patches, adaLN-Zero) with the third (scaling laws) being more of a realization than a new concept. The deep transformer knowledge from Series 4 means ~80% of the architecture is familiar.

**Pattern observation:**
- The lesson has a slight tendency to state conclusions in asides (InsightBlocks, TipBlocks) rather than developing them in the main body. The positional embedding issue and the "tokenize the image" analogy both suffer from this -- key content is in asides rather than in the main explanatory flow. Asides should reinforce and contextualize, not carry primary explanatory weight.

---

## Review — 2026-02-20 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 2

### Verdict: PASS

### Iteration 1 Fix Verification

All five iteration 1 findings were addressed effectively:

1. **Positional embedding promoted to main body (IMPROVEMENT -> FIXED).** A full paragraph now appears after Step 4 in the patchify CodeBlock (lines 320-328). It explains: (a) positional embeddings are the same concept as GPT-2's positional encoding extended to 2D patch positions, (b) without them the transformer treats the sequence as an unordered set, (c) they are DiT's only spatial prior, and (d) this is what "minimal spatial bias" means. The TipBlock aside is retained as reinforcement but no longer carries the primary explanatory weight. The aside-in-main-body pattern flagged in iteration 1 is resolved.

2. **"Tokenize the Image" analogy developed via ComparisonRow (IMPROVEMENT -> FIXED).** A four-row ComparisonRow (lines 265-286) now maps the text pipeline to the image pipeline step by step: Input -> Tokenize/Patchify -> Embed/Project -> Result. The student can see the exact structural correspondence before reading the concrete shape trace. The analogy is now built rather than stated. This is a significant improvement.

3. **Notebook Exercise 3 loads DiT as proper model object (IMPROVEMENT -> FIXED).** The notebook now loads via `DiTPipeline.from_pretrained('facebook/DiT-XL-2-256')` and accesses `dit_pipe.transformer` as a proper `nn.Module`. The `count_parameters_by_type()` function now works symmetrically on both models. The solution block in the details section includes both U-Net and DiT breakdowns. The comparison is no longer asymmetric.

4. **Spaced em dash in CodeBlock (POLISH -> FIXED).** Line 842 now reads `[4, 32, 32]—same shape as the noisy latent input` with no spaces around the em dash.

5. **Recap header (POLISH -> FIXED).** Changed from "Three Threads to Reactivate" to "Three Concepts to Reactivate." This resolves the "three threads" vs "two threads" narrative inconsistency with the Hook section.

### Findings

### [POLISH] — Positional embedding TipBlock aside partially redundant with new main body paragraph

**Location:** "Sequence Length and Patch Size" section, TipBlock aside (lines 397-403)
**Issue:** After the iteration 1 fix added a strong paragraph about positional embeddings in the main body of the patchify section (lines 320-328), the TipBlock aside in the next section ("Sequence Length and Patch Size") now repeats much of the same content. The main body paragraph says: "Step 4 adds learned positional embeddings--the same concept as GPT-2's positional encoding, extended to 2D patch positions. Without them, the transformer treats the patch sequence as an unordered set: it cannot distinguish patch (0,0) from patch (15,15)." The TipBlock aside says: "Positional embeddings for patches work the same way as for text tokens. Each patch position gets a learned vector. Without them, the transformer cannot distinguish patch (0,0) from patch (15,15). DiT uses standard learnable positional embeddings--same as GPT-2." The overlap is nearly verbatim. The aside could add value by saying something the main body does not (e.g., 2D positions encoded as 1D index, or that sinusoidal embeddings are also possible but DiT chose learned).
**Student impact:** Minimal. The student reads the same explanation twice in close proximity, which may feel repetitive but will not cause confusion.
**Suggested fix:** Either (a) trim the TipBlock to a single sentence like "Positional embeddings are DiT's only spatial prior--the transformer learns everything else about spatial relationships from data" (which echoes the "minimal spatial bias" insight without repeating the mechanics), or (b) add a new detail the main body does not cover, such as: "DiT encodes 2D patch positions as a 1D index after flattening. Sinusoidal embeddings are also possible, but DiT uses learned embeddings--same choice as GPT-2."

### [POLISH] — Notebook Exercise 3 "What Just Happened" states DiT has ~675M and U-Net has ~860M but these numbers should be verified

**Location:** Notebook cell-33 (markdown "What Just Happened" for Exercise 3)
**Issue:** The summary states "DiT has ~675M parameters, U-Net has ~860M." The DiT-XL/2 count of ~675M matches the paper and is consistent with the lesson. The SD v1.5 U-Net parameter count of ~860M is stated as fact in the summary but the actual count will be computed by the student in Step 4 of the exercise. The `diffusers` SD v1.5 U-Net has approximately 860M parameters, so this is likely correct. However, the exact count depends on the `diffusers` version and whether the model includes all sub-modules. If the actual runtime value differs slightly, the student will see a mismatch. The sentence also claims "DiT achieves better results with fewer parameters" which is comparing a class-conditional ImageNet model (DiT-XL/2) to a text-conditional general model (SD v1.5 U-Net)--these are not directly comparable in terms of "better results."
**Student impact:** Minimal. The student will see the actual numbers when running the code. If the U-Net count is slightly different, the summary paragraph may feel slightly misleading. The "better results" claim conflates class-conditional ImageNet FID with general text-to-image quality, which are different benchmarks.
**Suggested fix:** Change "DiT achieves better results with fewer parameters" to "Different architectures at a similar scale, designed for different tasks (class-conditional ImageNet vs text-to-image)." The lesson already correctly notes that DiT-XL/2 is class-conditional, so this keeps the notebook consistent with the lesson's framing.

### Review Notes

**What works well (carried forward from iteration 1 + new observations):**
- All three iteration 1 improvements were effectively addressed. The ComparisonRow for the tokenization analogy is particularly strong--it builds the analogy step by step rather than stating it, and the student can now arrive at "the transformer does not know or care" themselves by seeing the structural correspondence.
- The positional embedding fix is clean: the main body now carries the explanatory weight with a coherent paragraph connecting to GPT-2, explaining the unordered-set problem, and tying to the "minimal spatial bias" claim. This resolves the iteration 1 pattern observation about primary content in asides.
- The notebook Exercise 3 fix is the most substantial improvement. The symmetric comparison using `count_parameters_by_type()` on both models makes the architectural difference between DiT and U-Net concrete and verifiable. Loading via `DiTPipeline` also makes the transition to Exercise 4 smoother.
- The lesson continues to be pedagogically strong across all dimensions: motivation (SDXL limitations answered by DiT), modalities (5 present: concrete shape trace, visual comparison tables, verbal tokenization analogy, symbolic formulas for adaLN-Zero, intuitive LEGO vs house renovation), examples (3 positive + 1 negative as planned), misconceptions (5 addressed at planned locations), cognitive load (appropriate STRETCH), and connections to prior knowledge (every new concept linked to at least one existing concept).
- The check questions are well-distributed (4 sets, 10 total questions) and well-calibrated for a STRETCH lesson.
- The notebook exercises cover the lesson arc completely: patchify (Guided), adaLN-Zero (Guided), architecture inspection (Supported), generation (Independent). The scaffolding progression is correct.

**No new issues introduced by the fixes.** The fixes were surgical and did not create inconsistencies or disrupt the narrative flow.

**Verdict rationale:** Zero critical findings. Zero improvement findings. Two polish items, both minor (redundant aside content and a slightly inaccurate comparison claim in the notebook summary). The lesson is ready to ship. The student will come away with a clear understanding of how DiT works, why it replaces the U-Net, and how it connects to everything they learned in Series 4 and Series 6.
