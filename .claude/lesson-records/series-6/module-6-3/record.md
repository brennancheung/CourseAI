# Module 6.3: Architecture & Conditioning -- Record

**Goal:** The student understands how the components of Stable Diffusion work individually -- U-Net architecture for multi-scale denoising, timestep and text conditioning mechanisms, CLIP for bridging language and vision, and the move from pixel-space to latent-space diffusion -- well enough to explain why each piece exists and how they connect, preparing for full pipeline assembly in Module 6.4.
**Status:** Complete (5 of 5 lessons built)

## Concept Index

| Concept | Depth | Lesson | Notes |
|---------|-------|--------|-------|
| U-Net encoder-decoder architecture (encoder downsampling path, bottleneck, decoder upsampling path) | DEVELOPED | unet-architecture | Taught via dimension walkthrough (64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512 -> back up), Mermaid architecture diagram, pseudocode forward pass. Connected to autoencoder encoder-decoder from 6.1 and CNN feature hierarchy from Series 3. |
| U-Net skip connections as essential for denoising (encoder features concatenated with decoder at matching resolutions) | DEVELOPED | unet-architecture | Elevated from INTRODUCED (capstone 6.2). Core argument: without skips, decoder must reconstruct from bottleneck alone = blurry (autoencoder problem). With skips, fine-grained spatial details bypass bottleneck. Taught via autoencoder comparison (ComparisonRow), "keyhole and side doors" analogy, predict-and-verify exercise. |
| Multi-resolution processing mapped to coarse-to-fine denoising | DEVELOPED | unet-architecture | New connection between existing concepts. At t=900, bottleneck features dominate (global structural decisions). At t=50, skip connection features dominate (fine detail corrections). Same architecture, same weights -- the noise level determines which path matters more. Connected to DenoisingTrajectoryWidget experience from 6.2. |
| Dual-path information flow (bottleneck = WHAT, skip connections = WHERE) | DEVELOPED | unet-architecture | Core mental model. Bottleneck provides global composition (shoe vs shirt, overall structure). Skip connections provide exact edge positions, textures, pixel-precise spatial information. Both needed at every timestep. Addressed misconception that bottleneck is unnecessary if skips exist. |
| Receptive field growth through downsampling (each lower resolution level sees a larger region of the input) | INTRODUCED | unet-architecture | Explained as motivation for why conv stacks at same resolution fail for heavy noise. At 8x8 bottleneck, each pixel has global receptive field over the 64x64 input. Not practiced, but essential for understanding why the encoder path exists. |
| Residual blocks within the U-Net (Conv -> Norm -> Activation -> Conv -> Norm -> Activation + skip) | INTRODUCED | unet-architecture | Brief treatment. Connected to ResNet residual connections (Series 3.2). Mentioned group normalization (replaces batch norm for small batch sizes) but did not explain. Distinguished short-range skips (residual blocks within levels) from long-range skips (encoder-to-decoder across the U). |
| Short-range vs long-range skip connections (different purposes: gradient flow vs spatial information) | INTRODUCED | unet-architecture | Explicitly contrasted: ResNet skips primarily help gradients flow during training; U-Net long-range skips primarily carry high-resolution spatial information. Same mechanism, fundamentally different purpose. |
| Feature map channel concatenation at skip connections (encoder channels + decoder channels, then conv to reduce) | INTRODUCED | unet-architecture | Shown in pseudocode: `cat(up(b), e2)` produces doubled channels, subsequent conv reduces back. Explained in TipBlock but not practiced. |
| Attention layers within the U-Net (self-attention, cross-attention) | MENTIONED | unet-architecture | Brief preview paragraph: interleaved at middle resolutions (16x16, 32x32), self-attention relates distant parts of feature map, cross-attention connects to text embeddings. Explicitly deferred to later lessons. |
| Group normalization (replaces batch norm in diffusion U-Nets) | MENTIONED | unet-architecture | Named as a component in residual blocks. "Works better than batch norm for small batch sizes common in diffusion training. You do not need the details." |
| Group normalization (divide channels into groups, normalize within each group, per-example) | INTRODUCED | conditioning-the-unet | Elevated from MENTIONED. Taught via comparison to batch norm (across batch per channel) and layer norm (across all channels per example). ASCII diagram showing normalization axes on a [B, C, H, W] tensor. Three GradientCards comparing the three variants. Motivated by small batch sizes in diffusion training. |
| Sinusoidal timestep embedding (same formula as positional encoding, applied to noise level instead of position) | DEVELOPED | conditioning-the-unet | Direct transfer from sinusoidal positional encoding (4.1.3). Side-by-side formula comparison (PE vs TE -- identical, replacing "pos" with "t"). Concrete numerical example with d=8 showing four dimensions at t=500 and t=50, demonstrating frequency progression. Clock analogy callback ("same clock, different question"). ComparisonRow contrasting simple linear projection (capstone) vs sinusoidal + MLP. Negative example: random embeddings lack smoothness. MLP refinement step shown in code. |
| Adaptive group normalization (timestep-dependent gamma and beta applied after standard group norm) | DEVELOPED | conditioning-the-unet | Core mechanism for timestep conditioning. Built from batch norm gamma/beta (DEVELOPED). Formula: AdaGN(x, t) = gamma(t) * Normalize(x) + beta(t). Concrete example with different gamma/beta values at t=500 vs t=50. Residual block pseudocode and Mermaid diagram showing two AdaGN injection points per block. Per-block linear projection pattern (one embedding, different lens per block -- connected to Q/K/V). WarningBlock: "adaptive" does NOT mean conv weights change, only scale and shift after normalization. |
| Global conditioning pattern (information that affects every spatial location, injected at every processing stage) | INTRODUCED | conditioning-the-unet | Named and explained. Timestep is the canonical example: noise level is the same for every pixel. Contrasted with spatially-varying conditioning (text via cross-attention, previewed for later lesson). Motivated by negative examples: input-only conditioning gets diluted, bottleneck-only conditioning misses skip connections. |
| FiLM conditioning (Feature-wise Linear Modulation -- predicting scale and shift from a conditioning signal) | MENTIONED | conditioning-the-unet | Name-dropped as the general framework of which adaptive group norm is a specific instance. ConceptBlock with 2-sentence description. Useful for reading papers. |
| Per-block timestep projection (one shared embedding, per-block linear layer to produce resolution-specific gamma/beta) | INTRODUCED | conditioning-the-unet | Each residual block has its own Linear(512, 2*channels) to extract the relevant aspect of the timestep for its resolution. Connected to "learned lens" pattern from Q/K/V projections (4.2.2). WarningBlock addresses misconception that separate conditioning signals are needed per resolution level. |
| U-Net origin in medical image segmentation (Ronneberger et al. 2015) | MENTIONED | unet-architecture | Transfer question exercise: same architecture useful for segmentation and denoising because both require pixel-precise output combining global understanding with local detail. |
| ConvTranspose2d / nearest-neighbor upsampling (decoder upsampling mechanisms) | MENTIONED | unet-architecture | Named as decoder upsampling methods but not explained further. Student already has ConvTranspose2d at INTRODUCED from autoencoders (6.1). |
| Contrastive learning paradigm (push matching pairs together, non-matching apart, using batch structure for negatives) | DEVELOPED | clip | Taught via conference analogy (name tag matching), 4x4 similarity matrix heatmap with walkthrough, symmetric cross-entropy loss formula, predict-and-verify exercises. Student can explain training setup, loss function, why negatives come from the batch, and why scale matters. |
| CLIP dual-encoder architecture (separate image encoder + text encoder, no shared weights, loss-only coupling) | INTRODUCED | clip | Two GradientCards (image encoder: CNN/ViT -> 512-dim, text encoder: Transformer -> 512-dim). Mermaid diagram showing parallel paths to cosine similarity. WarningBlock addressing misconception that CLIP is a single network. Student knows the structure and data flow but has not built it. |
| Shared embedding space (text and image vectors in the same geometric space, enabling cross-modal cosine similarity) | DEVELOPED | clip | Before/after SVG diagram showing alignment. Connected to EmbeddingSpaceExplorer (token embeddings, 4.1.3) and VAE latent space (6.1.3-6.1.4). ComparisonRow explicitly contrasting VAE latent space (has decoder, for generation) vs CLIP embedding space (no decoder, for comparison). Negative example: independently trained encoders produce unaligned spaces. |
| Zero-shot classification via shared embedding space (classify images by comparing to text descriptions of classes, no task-specific training) | INTRODUCED | clip | Explained conceptually with 4-step procedure and PyTorch pseudocode. Connected to transfer learning (3.2.3) -- goes further because no retraining of classification head needed. Student understands how and why it works but has not implemented it. |
| Contrastive loss formula (symmetric cross-entropy on cosine similarity matrix, labels = diagonal) | INTRODUCED | clip | Full formula in LaTeX (similarity matrix, labels, symmetric CE, average). PyTorch pseudocode showing the complete training step. Connected to cross-entropy from classification (3.2.3) and language modeling (4.1.1). Temperature parameter explained with explicit convention inversion note vs LM temperature. |
| Batch size as task difficulty in contrastive learning (more negatives = harder discrimination task = more discriminative features) | INTRODUCED | clip | Three GradientCards (batch=2 vs 32 vs 32,768) showing progressive difficulty. WarningBlock contrasting with batch size role in standard training (gradient noise/speed). CLIP's actual batch size of 32,768 as concrete reference point. |
| CLIP limitations (typographic attacks, spatial relationships, counting, novel compositions) | INTRODUCED | clip | Four GradientCards covering specific failure modes. Framed as "statistical co-occurrence, not deep understanding." ConceptBlock on typographic attacks explaining why they work. Tempers zero-shot enthusiasm without undermining utility. |
| Vision Transformer / ViT (transformer architecture applied to image patches instead of tokens) | MENTIONED | clip | Named as one of CLIP's image encoder options. ConceptBlock with 2-sentence description. Not developed -- student knows ViT exists and processes patches like tokens. |
| Natural language supervision (using internet text-image co-occurrence as training signal, no predefined categories) | INTRODUCED | clip | Concrete example (golden retriever alt-text). Connected to self-supervised training pattern (autoencoder: input=target, LM: shifted text=target, CLIP: internet pairs=target). TipBlock contrasting with ImageNet's 1,000 predefined categories. |
| Cross-attention mechanism (Q from spatial features, K/V from text embeddings, same QKV formula as self-attention) | DEVELOPED | text-conditioning-and-guidance | Taught as a one-line change from self-attention: Q = W_Q X_spatial, K = W_K X_text, V = W_V X_text. Same output formula softmax(QK^T / sqrt(d_k)) V. ComparisonRow side-by-side with self-attention. Custom SVG data flow diagram showing 4x4 spatial grid producing Q, text tokens producing K/V, and varying-thickness attention lines. Shape walkthrough (Q: 256xd_k, K: Txd_k, attention weights: 256xT -- rectangular, not square). Concrete attention weight table for "a cat sitting in a sunset" showing per-spatial-location attention over 6 text tokens. Three-lens callback (seeking/advertising/contributing from 4.2.2-4.2.3). |
| Per-spatial-location cross-attention (each spatial location generates its own query, attends independently over all text tokens, creating spatially-varying conditioning) | DEVELOPED | text-conditioning-and-guidance | Core insight distinguishing text conditioning from timestep conditioning. Taught via concrete attention weight table: cat's face region attends to "cat" (0.62), sky region attends to "sunset" (0.80), cat's body attends to "cat" + "sitting" (0.31 + 0.45). Different locations get different signals from the same prompt. |
| Classifier-free guidance (training with random text dropout, inference with two forward passes, amplifying the text direction) | DEVELOPED | text-conditioning-and-guidance | Taught in three stages: (1) problem (model can learn to ignore text), (2) training trick (randomly drop text embedding ~10-20% of the time, one model learns both conditional and unconditional), (3) inference formula: epsilon_cfg = epsilon_uncond + w * (epsilon_cond - epsilon_uncond). Concrete numerical example with 3-dim noise vectors at w=7.5. Geometric SVG diagram showing unconditional/conditional predictions, text direction vector, and CFG extrapolation. "Contrast slider" analogy. WarningBlock addressing one-model-not-two misconception. |
| Guidance scale tradeoff (w=1 no amplification, w=7.5 typical, w=20 oversaturated -- fidelity vs quality) | DEVELOPED | text-conditioning-and-guidance | Three GradientCards showing w=1 (diverse but unfaithful), w=7.5 (balanced, SD default), w=20 (oversaturated, distorted). WarningBlock: higher is not always better. Connected to user-facing CFG scale parameter in Stable Diffusion UIs. |
| Spatially-varying vs global conditioning (timestep via adaptive norm is global, text via cross-attention is spatially varying) | INTRODUCED | text-conditioning-and-guidance | ComparisonRow contrasting timestep conditioning (adaptive group norm, every resolution, global: same signal at all spatial locations, WHEN) vs text conditioning (cross-attention, attention resolutions only, spatially varying: each location gets its own signal, WHAT). WarningBlock: they coexist, not compete. |
| Self-attention within the U-Net (spatial features attend to other spatial features for global spatial coherence) | INTRODUCED | text-conditioning-and-guidance | Brief treatment. Enables spatial consistency (left eye and right eye can coordinate). Sits alongside cross-attention in the block ordering. Not the focus -- distinguished from cross-attention by purpose (spatial consistency vs text guidance). |
| Block ordering at attention resolutions (residual block -> self-attention -> cross-attention, repeated at 16x16 and 32x32) | INTRODUCED | text-conditioning-and-guidance | Mermaid diagram showing the three-block pattern repeated twice at one resolution. Color-coded: amber (residual/timestep), blue (self-attention), violet (cross-attention). Explained why only middle resolutions: O(n^2) cost constraint. At 64x64 = 4096 tokens, attention is too expensive. At 16x16 = 256 tokens, affordable. |
| Classifier guidance (predecessor to CFG, used a separate classifier's gradients to steer diffusion) | MENTIONED | text-conditioning-and-guidance | Historical context: Dhariwal & Nichol, 2021. Required a separately trained classifier, only supported class labels, was complex. CFG (Ho & Salimans, 2022) eliminated the classifier entirely. |
| CFG doubles inference compute (two forward passes per denoising step, can be batched for efficiency) | INTRODUCED | text-conditioning-and-guidance | Brief note: 50 denoising steps = 100 forward passes. The two passes can be batched (batch of 2) but memory cost is real. TipBlock on batching efficiency. |
| Latent diffusion as an architectural pattern (encode with VAE, diffuse in latent space, decode back to pixels -- same DDPM algorithm on smaller tensors) | DEVELOPED | from-pixels-to-latents | Core concept of the lesson. Taught via side-by-side training algorithm comparison (ComparisonRow: 7-step DDPM vs 7-step latent diffusion -- only steps 1, 5, 7 differ), pipeline comparison Mermaid diagram (pixel-space vs latent-space with identical denoising loop), dimension walkthrough code block (512x512x3 -> 64x64x4 -> diffuse -> decode), and "of course" intuitive chain (VAE preserves content + diffusion denoises tensors = of course combine them). "Translator between two languages" analogy (VAE translates pixel language to latent language and back). "Same orchestra, smaller hall" analogy. |
| Frozen-VAE pattern (VAE trained separately on image reconstruction, then frozen during diffusion training) | INTRODUCED | from-pixels-to-latents | Taught via WarningBlock explaining the two-stage pipeline: (1) train VAE, (2) freeze VAE, train diffusion on frozen latents. VAE does not know about diffusion; diffusion model does not know it operates on VAE latents. Modularity is the point. TipBlock explaining why not end-to-end: diffusion gradients would shift latent space geometry, invalidating U-Net's learned denoising. Connected to student's own VAE training experience ("you did not know you would later use latent codes for anything"). |
| SD VAE improvements (perceptual loss + adversarial training for sharper reconstructions than MSE-only VAEs) | MENTIONED | from-pixels-to-latents | Brief treatment explaining why SD's VAE produces sharper results than the student's toy VAE (which used MSE loss). Perceptual loss compares features from a pre-trained network, adversarial training penalizes fake-looking reconstructions. Explicitly scoped as MENTIONED: "you do not need the details of these losses." |
| Computational cost reduction via latent space (512x512x3 = 786,432 values compressed to 64x64x4 = 16,384 values, 48x compression; convolution cost scales with spatial area) | DEVELOPED | from-pixels-to-latents | Taught via ComparisonRow (pixel-space U-Net at 262,144 spatial positions vs latent-space at 4,096), code block with specific tensor dimensions, and concrete scaling comparison to capstone experience: 512x512 pixel-space is ~335x larger than 28x28 capstone, while 64x64 latent-space is only ~5x larger. The difference between "impractical" and "runs on a consumer GPU" is the VAE compression. TipBlock: VAE encode/decode each run once, cheap vs 50 U-Net steps. |
| Latent representation is not a small image (64x64x4 latent is an abstract learned representation, not a downscaled RGB image; decoder is essential, not optional) | INTRODUCED | from-pixels-to-latents | Negative example addressing misconception that latent diffusion is "just small-resolution diffusion." The 4-channel latent tensor has no direct visual interpretation. Connected to autoencoder lesson (6.1.2): bottleneck representation is a learned compression, not a thumbnail. WarningBlock reinforcing the distinction. |

## Per-Lesson Summaries

### unet-architecture (Lesson 1)
**Status:** Built, reviewed (PASS on iteration 2), two polish items noted
**Cognitive load:** BUILD (re-entering theoretical mode after CONSOLIDATE capstone)
**Notebook:** None (conceptual/theoretical lesson; implementation was the 6.2 capstone)

**Concepts taught:**
- U-Net encoder-decoder architecture with skip connections (DEVELOPED) -- elevated from INTRODUCED in capstone
- Multi-resolution processing mapped to coarse-to-fine denoising (DEVELOPED) -- new connection
- Residual blocks within the U-Net (INTRODUCED)
- Short-range vs long-range skip connections (INTRODUCED)
- Receptive field growth through downsampling (INTRODUCED)
- Attention layers, group normalization, medical U-Net origin (MENTIONED)

**Mental models established:**
- "The bottleneck decides WHAT. The skip connections decide WHERE." -- core mental model for dual-path information flow
- "Keyhole and side doors" -- bottleneck is a keyhole (global structure passes through), skip connections are side doors (fine details bypass the keyhole entirely)
- Multi-resolution processing mirrors the denoising task -- at high noise, bottleneck features matter most; at low noise, skip connection features matter most

**Analogies used:**
- Keyhole and side doors (bottleneck vs skip connections)
- Autoencoder blurry reconstructions as negative example (student experienced this in 6.1)
- "Same building blocks, different question" (continuing recurring theme)
- CNN feature hierarchy IS the encoder path (edges -> textures -> parts -> objects)

**How concepts were taught:**
- Hook: "Why not just conv layers?" -- receptive field argument shows same-resolution stacks fail at heavy noise because they cannot make global decisions
- Encoder path: dimension walkthrough (64x64x64 -> 32x32x128 -> 16x16x256 -> 8x8x512) with what each level captures
- Bottleneck: connected to autoencoder bottleneck from 6.1, but distinguished by not relying solely on it for reconstruction
- Skip connections: motivated by autoencoder blurriness problem, then shown as the solution via ComparisonRow (Autoencoder vs U-Net)
- Full architecture: Mermaid diagram with active tracing exercise (TryThisBlock)
- Multi-resolution mapping: t=900 vs t=50 walkthrough showing differential importance of each path
- Pseudocode: high-level `cat(up(b), e2)` pattern showing information flow, timestep parameter omitted (noted in comment, deferred to next lesson)

**Misconceptions addressed:**
1. "The U-Net is just an autoencoder" -- addressed via ComparisonRow showing autoencoder (blurry) vs U-Net (sharp)
2. "Skip connections are just a training trick (like ResNet)" -- explicitly contrasted in ConceptBlock: ResNet skips = gradient flow, U-Net skips = spatial information
3. "The bottleneck is unnecessary if skip connections exist" -- addressed in Section 8 WarningBlock: without bottleneck, no global context for heavy noise
4. "Each resolution level handles one specific noise level" -- WarningBlock "Not a Routing Mechanism": every timestep processes all levels, but importance shifts

**What is NOT covered (deferred):**
- Timestep conditioning (how the network receives t) -- next lesson
- Self-attention and cross-attention mechanisms -- mentioned only, developed in later lessons
- Group normalization details
- Text conditioning or guided generation
- Implementation (capstone already covered this)

**Review notes:**
- Iteration 1: MAJOR REVISION -- 1 critical (missing autoencoder recap), 4 improvements, 2 polish
- Iteration 2: PASS -- all iteration 1 findings fixed, 2 remaining polish items (Mermaid diagram does not visually form a U shape; pseudocode omits timestep without inline comment -- both were fixed in the final build)

### conditioning-the-unet (Lesson 2)
**Status:** Built, reviewed (PASS on iteration 2), one polish item noted
**Cognitive load:** BUILD (re-entering theoretical mode with heavily leveraged prior knowledge)
**Notebook:** `notebooks/6-3-2-conditioning-the-unet.ipynb` (4 exercises: sinusoidal embedding, timestep MLP, adaptive group norm, comparison)

**Concepts taught:**
- Sinusoidal timestep embedding (DEVELOPED) -- direct transfer from positional encoding (4.1.3)
- Adaptive group normalization (DEVELOPED) -- timestep-dependent gamma/beta after standard group norm
- Group normalization (INTRODUCED) -- elevated from MENTIONED in unet-architecture
- Global conditioning pattern (INTRODUCED)
- Per-block timestep projection (INTRODUCED)
- FiLM conditioning (MENTIONED)

**Mental models established:**
- "Same clock, different question" -- sinusoidal positional encoding reused for timestep encoding; position in sequence becomes noise level in diffusion
- "The conductor's score" -- timestep embedding is the conductor, adaptive norm is how the conductor communicates dynamics to each section; same orchestra (weights), different performance depending on the measure number (timestep)
- "Tone controls that respond to the signal" -- standard norm has fixed gamma/beta set during training; adaptive norm has gamma/beta that depend on the timestep, like a graphic equalizer adjusting based on the genre

**Analogies used:**
- Clock with many hands (callback from 4.1.3, now encoding noise level)
- Conductor and orchestra (adaptive normalization)
- GPS coordinates vs single number (sinusoidal encoding vs simple linear projection)
- Learned lens pattern (callback from Q/K/V, applied to per-block projection)
- Skeleton + nervous system (U-Net architecture + timestep conditioning)

**How concepts were taught:**
- Gap fill: Group normalization elevated from MENTIONED via comparison to batch norm and layer norm. ASCII diagram showing normalization axes on [B, C, H, W] tensor. Three GradientCards. Motivated by small batch sizes.
- Hook: "The pseudocode is incomplete" -- callback to unet-architecture's omitted timestep parameter. Problem framed as: one network, two radically different tasks (t=900 structural hallucination vs t=50 detail polish).
- Sinusoidal embedding: side-by-side formula comparison with positional encoding. Concrete numerical example (d=8, t=500 and t=50 at four dimensions showing frequency progression). ComparisonRow (simple projection vs sinusoidal + MLP). Negative example (random embeddings lack smoothness). MLP refinement step in code.
- Adaptive group norm: built from batch norm gamma/beta (DEVELOPED). Minimal conceptual delta: "make gamma and beta depend on t." Formula side-by-side (standard GN vs AdaGN). Concrete example with specific gamma/beta values at two timesteps. Residual block pseudocode + Mermaid diagram showing injection points. Per-block projection connected to Q/K/V pattern.
- Global conditioning: motivated by negative examples (input-only gets diluted, bottleneck-only misses skip connections). Named the pattern. Previewed spatially-varying conditioning contrast (text, next lessons).
- Complete forward pass: pseudocode fulfilling the promise from unet-architecture, with t_emb flowing into every block. Full U-Net Mermaid diagram with conditioning arrows. "Skeleton + nervous system" analogy.
- Predict-and-verify: two check sections testing smoothness intuition and transfer to what/where mental model.

**Misconceptions addressed:**
1. "Timestep conditioning is just concatenating t to the input" -- negative example showing dilution through conv layers; motivated conditioning at every layer
2. "Timestep embedding is a lookup table (like token embeddings)" -- WarningBlock contrasting nn.Embedding with sinusoidal encoding; smoothness requirement for 1000 timesteps
3. "Adaptive normalization changes the network's weights based on timestep" -- WarningBlock: only gamma and beta change, not conv weights or normalization statistics
4. "You need separate conditioning for each resolution level" -- WarningBlock "One Embedding, Not Many": same embedding, per-block linear projection extracts different aspects
5. "The simple linear projection from the capstone works fine with more training" -- ComparisonRow showing the representational poverty of a single direction vs multi-frequency encoding

**What is NOT covered (deferred):**
- Text conditioning or cross-attention (Lesson 4: text-conditioning-and-guidance)
- CLIP or text embeddings (Lesson 3: clip)
- Self-attention within the U-Net (deferred)
- Classifier-free guidance (Lesson 4)
- Full implementation (Module 6.4 capstone)

**Review notes:**
- Iteration 1: NEEDS REVISION -- 1 critical (missing notebook), 4 improvements (fabricated sinusoidal denominators, missing normalization axis visual, missing residual block diagram, implicit misconception #4), 2 polish
- Iteration 2: PASS -- all iteration 1 findings fixed, 1 remaining polish item (spaced em dashes in concrete example)

### clip (Lesson 3)
**Status:** Built, reviewed (PASS pending notebook fix on iteration 2), two polish items noted
**Cognitive load:** STRETCH (genuinely new training paradigm with familiar building blocks)
**Notebook:** `notebooks/6-3-3-clip.ipynb` (4 exercises: hand-computed similarity matrix, pretrained CLIP exploration, zero-shot classification, probing limitations)

**Concepts taught:**
- Contrastive learning paradigm (DEVELOPED) -- genuinely new training objective; taught via conference analogy, 4x4 similarity matrix heatmap, symmetric cross-entropy formula, and predict-and-verify exercises
- Shared embedding space (DEVELOPED) -- text and images in the same geometric space; before/after SVG diagram, ComparisonRow vs VAE latent space, negative example with independently trained encoders
- CLIP dual-encoder architecture (INTRODUCED) -- two separate encoders (image + text), no shared weights, loss-only coupling; GradientCards, Mermaid diagram
- Zero-shot classification (INTRODUCED) -- classify by comparing image embedding to text prompt embeddings; conceptual walkthrough + pseudocode
- Contrastive loss formula (INTRODUCED) -- symmetric cross-entropy on similarity matrix; full LaTeX formula + PyTorch pseudocode
- Batch size as task difficulty, CLIP limitations, natural language supervision, Vision Transformer (INTRODUCED/MENTIONED)

**Mental models established:**
- "Two encoders, one shared space -- the loss function creates the alignment, not the architecture." -- core mental model for contrastive learning and CLIP
- "Negatives come from the batch" -- in N pairs, each example has 1 positive and N-1 negatives for free; the batch structure IS the supervision signal
- "The shared space enables zero-shot transfer" -- matching any text to any image without task-specific training

**Analogies used:**
- Name tag matching at a conference (contrastive learning: N people, photo<->name tag, more people = harder task)
- "Like comparing GPS coordinates to temperatures" (why independently trained encoders produce meaningless cosine similarities)
- "Same building blocks, different question" (continuing recurring theme -- CNN + transformer + cross-entropy, but the question is "do this text and image match?")
- "The internet already paired these for you" (natural language supervision from 400M web text-image pairs)

**How concepts were taught:**
- Recap: reinforced embedding space intuition (EmbeddingSpaceExplorer callback, VAE latent space callback) and self-supervised training pattern (autoencoder, LM, CLIP as three forms of free supervision)
- Hook: "The Control Problem" -- U-Net processes tensors, has no idea what "cat" means; need vectors that MEAN the same thing as the words; train two encoders together
- Dual-encoder architecture: two GradientCards (image encoder, text encoder), Mermaid dataflow diagram, WarningBlock about two encoders not one
- Shared embedding space: before/after training SVG diagram (circles=images, triangles=text, green push-together arrows, red push-apart dashes); ComparisonRow (VAE for generation vs CLIP for comparison); negative example (pretrained ResNet + pretrained BERT = random correlation)
- Contrastive learning: conference analogy, then 4x4 similarity matrix as styled HTML table (violet diagonal = matches, muted off-diagonal = negatives); walked through one row (classification framing) and one column; showed that negatives come from batch structure
- Loss function: LaTeX formula (S_ij, labels, symmetric CE, average) + PyTorch pseudocode; temperature parameter with explicit convention inversion note ("same principle, opposite convention" vs LM temperature from 4.1.1)
- Scale: three GradientCards (batch=2 easy, batch=32 moderate, batch=32768 requires fine-grained features); WarningBlock contrasting with standard training batch size role
- Zero-shot classification: 4-step procedure + pseudocode; callback to transfer learning (3.2.3) -- CLIP goes further, no retraining needed
- Limitations: four GradientCards (typographic attacks, spatial relationships, counting, novel compositions); ConceptBlock on statistical co-occurrence
- Predict-and-verify: two check sections -- first tests diagonal labels and loss mechanics (3 questions), second tests transfer (independently trained encoders, can CLIP generate?)

**Misconceptions addressed:**
1. "CLIP is a single neural network" -- WarningBlock: two separate encoders, no shared weights, only loss connects them
2. "CLIP's embedding space is like the VAE's latent space" -- ComparisonRow: no decoder, no generation, for comparison not generation; highest-risk misconception addressed early
3. "Contrastive learning needs explicit negative labels" -- explained that other batch items ARE the negatives, for free; WarningBlock
4. "CLIP understands images/text like humans do" -- typographic attacks, spatial relationships, counting failures as concrete evidence of statistical co-occurrence not understanding
5. "Bigger batch size is just for faster training" -- three GradientCards showing qualitative difference: batch size changes task difficulty, not just speed

**What is NOT covered (deferred):**
- How CLIP connects to the U-Net / cross-attention injection (Lesson 4: text-conditioning-and-guidance)
- Classifier-free guidance (Lesson 4)
- CLIP implementation from scratch (too compute-intensive)
- ViT architecture details (MENTIONED only)
- CLIP variants (SigLIP, OpenCLIP differences)
- Full Stable Diffusion pipeline integration (Module 6.4)

**Review notes:**
- Iteration 1: MAJOR REVISION -- 1 critical (missing notebook), 4 improvements (missing before/after SVG, temperature convention contradiction, similarity matrix not visual enough, repetitive ending), 2 polish
- Iteration 2: NEEDS REVISION -- 0 critical, 1 improvement (notebook says "400M pairs" but loads LAION-2B model), 2 polish (notebook "same model" misleading, module record outdated)

### text-conditioning-and-guidance (Lesson 4)
**Status:** Built, reviewed (PASS on iteration 2), one polish item noted
**Cognitive load:** BUILD (cross-attention is a small delta from deep self-attention knowledge; CFG is procedurally simple but conceptually important)
**Notebook:** `notebooks/6-3-4-text-conditioning-and-guidance.ipynb` (4 exercises: cross-attention from self-attention, visualize cross-attention weights, implement CFG, CFG with a real diffusion model)

**Concepts taught:**
- Cross-attention mechanism in the U-Net (DEVELOPED) -- elevated from MENTIONED; taught as a one-line change from self-attention (same formula, K/V come from text instead of spatial features)
- Classifier-free guidance (DEVELOPED) -- genuinely new; training with random text dropout, inference with two forward passes, amplifying the text direction
- Guidance scale tradeoff (DEVELOPED) -- w=1 vs w=7.5 vs w=20 spectrum with practical consequences
- Per-spatial-location cross-attention / spatially-varying conditioning (DEVELOPED) -- each spatial location attends independently to all text tokens
- Spatially-varying vs global conditioning contrast (INTRODUCED)
- Self-attention within the U-Net (INTRODUCED)
- Block ordering at attention resolutions (INTRODUCED)
- Classifier guidance as predecessor (MENTIONED)
- CFG doubles inference compute (INTRODUCED)

**Mental models established:**
- "Same formula, different source for K and V" -- cross-attention is the QKV formula the student already knows deeply, with the only change being that K and V are projected from text embeddings instead of spatial features
- "The timestep tells the network WHEN (noise level, via adaptive norm, global). The text tells it WHAT (semantic content, via cross-attention, spatially varying). CFG turns up the volume on the WHAT." -- complete conditioning mental model integrating both conditioning mechanisms
- "Contrast slider" -- CFG as adjusting the contrast between text-conditioned and unconditioned predictions; medium contrast = vivid images, too much = oversaturation

**Analogies used:**
- "One-line change" (cross-attention vs self-attention: same formula, different input sources)
- "Contrast slider" (guidance scale as a contrast dial -- medium = good, extreme = distorted)
- "Access vs influence" (cross-attention provides the channel, CFG controls the volume)
- "Reading from a reference document" (self-attention reads from spatial features, cross-attention reads from text -- callback to "attention reads" from 4.2.5)
- "Three lenses" callback (W_Q = seeking lens on spatial features, W_K = advertising lens on text, W_V = contributing lens on text)
- "Directions in space" connection (CFG text direction analogous to attribute directions in latent space from 6.1.4, but computed fresh at every denoising step)

**How concepts were taught:**
- Recap: clarified that CLIP produces a SEQUENCE of token embeddings (not just the single summary vector used for similarity). Quick callback to QKV formula from 4.2.
- Hook: "The U-Net still cannot understand text" -- two pieces on the table (U-Net + CLIP) but disconnected. Motivated cross-attention by showing why global conditioning (averaging text into one vector + adaptive norm) loses spatial selectivity.
- Cross-attention: ComparisonRow (self-attention vs cross-attention side-by-side formulas). Custom SVG data flow diagram showing 4x4 spatial grid -> W_Q -> Q, text tokens -> W_K/W_V -> K/V, with varying-thickness attention lines. Shape walkthrough (256 spatial locations x T text tokens = rectangular attention matrix). Concrete attention weight table for "a cat sitting in a sunset" with three spatial locations showing different attention patterns.
- Where cross-attention lives: Mermaid diagram showing block ordering (ResidualBlock -> SelfAttention -> CrossAttention, color-coded). Why only at 16x16 and 32x32: O(n^2) cost argument with concrete numbers.
- Coexistence: ComparisonRow (timestep conditioning via adaptive norm vs text conditioning via cross-attention). WarningBlock: they coexist, not compete.
- CFG: motivated by "access does not equal influence" -- model can learn to ignore text. Training trick: random text dropout (10-20%), one model learns both tasks. WarningBlock: one model, not two. Inference formula with concrete 3-dim numerical example. Geometric SVG diagram showing unconditional/conditional predictions, text direction, and CFG extrapolation. Rearranged form showing extrapolation beyond conditional prediction.
- Guidance scale: three GradientCards (w=1, w=7.5, w=20). WarningBlock: higher is not always better. Historical context: classifier guidance as predecessor. Cost note: two forward passes per step.
- Predict-and-verify: three questions (swapped Q/KV roles, w=0.5 blend, why not just train longer)

**Misconceptions addressed:**
1. "Cross-attention in the U-Net is fundamentally different from attention in Series 4.2" -- addressed via side-by-side ComparisonRow showing identical formula with different K/V source
2. "Cross-attention processes the entire text and image at once as a single dot product" -- addressed via per-spatial-location attention table showing different locations attend to different words
3. "Classifier-free guidance means training two separate models" -- WarningBlock: one model, random text dropout during training
4. "Higher guidance scale always means better images" -- three GradientCards showing oversaturation at w=20; WarningBlock
5. "Text conditioning replaces timestep conditioning" -- ComparisonRow and WarningBlock showing coexistence: global WHEN vs spatially-varying WHAT

**What is NOT covered (deferred):**
- Implementing cross-attention or CFG from scratch (Module 6.4 capstone)
- Latent diffusion / VAE encoder-decoder (next lesson: from-pixels-to-latents)
- Self-attention within the U-Net beyond brief introduction
- Negative prompts, prompt engineering
- Alternative text encoders (T5 in Imagen)
- CLIP architecture details (covered in previous lesson)

**Review notes:**
- Iteration 1: NEEDS REVISION -- 0 critical, 3 improvements (missing cross-attention data flow diagram, notebook Exercise 3 crash risk, module record outdated), 2 polish (latent spaces reference ambiguity, notebook model path mismatch)
- Iteration 2: PASS -- all iteration 1 findings fixed, 1 remaining polish item (SVG hardcoded colors may have low contrast in light mode -- future-proofing concern only)

### from-pixels-to-latents (Lesson 5)
**Status:** Built, reviewed (PASS on iteration 3), three improvement findings and two polish items resolved across iterations
**Cognitive load:** CONSOLIDATE (zero new algorithms -- synthesizes VAE from 6.1, diffusion from 6.2, and architecture/conditioning from 6.3 into latent diffusion)
**Notebook:** `notebooks/6-3-5-from-pixels-to-latents.ipynb` (4 exercises: explore SD's VAE encoder/decoder, visualize the latent space, compute compression ratio, trace the full pipeline)

**Concepts taught:**
- Latent diffusion as an architectural pattern (DEVELOPED) -- the same DDPM algorithm running on 64x64x4 latent tensors instead of 512x512x3 pixel tensors, with VAE encode before and decode after
- Frozen-VAE pattern (INTRODUCED) -- VAE trained separately on image reconstruction, then frozen during diffusion training; modularity is the design principle
- SD VAE improvements: perceptual + adversarial loss (MENTIONED) -- why SD's VAE is sharper than the student's MSE-only toy VAE
- Computational cost reduction (DEVELOPED) -- 48x compression ratio, concrete dimension walkthrough, scaling comparison to capstone experience
- Latent representation is not a small image (INTRODUCED) -- negative example: 64x64x4 is an abstract learned representation, not a thumbnail

**Mental models established:**
- "The VAE is a translator between two languages" -- pixel language (what humans see) and latent language (what the diffusion model speaks); the encoder translates pixel to latent, the decoder translates latent to pixel
- "Same orchestra, smaller hall" -- the U-Net, conditioning mechanisms, and diffusion algorithm are the same orchestra; the latent space is a smaller, more efficient venue; same music, less energy to fill the room
- "The VAE built the roads. Diffusion walks them." -- the VAE's organized, continuous latent space is what makes the denoising trajectory stay in semantically meaningful territory
- "The VAE proves the concept; diffusion delivers the quality. Together, they are Stable Diffusion." -- fulfillment of the Module 6.1 promise

**Analogies used:**
- Translator between two languages (VAE as pixel-to-latent and latent-to-pixel translator)
- Same orchestra, smaller concert hall (latent space as more efficient venue for the same performance)
- "Same building blocks, different question" (continuing recurring theme -- same diffusion algorithm, now asked "can you denoise latents?")
- Callback to "describe a shoe in 32 words" from 6.1.3 (compression analogy, now scaled to SD's 48x compression)
- Callback to "clouds, not points" from 6.1.3 (VAE distributional encoding enabling organized latent space)

**How concepts were taught:**
- VAE reactivation (Section 3): Dedicated recap section following the Reinforcement Rule for concepts ~10 lessons old. Callbacks to specific student experiences (T-shirt/sneaker interpolation, "clouds not points," quality ceiling comparison). Not re-teaching -- reactivating the encoder-decoder data flow and latent space properties. Scale transition from 28x28 toy VAE to 512x512 SD VAE.
- Hook (Section 4): "Remember how slow that was?" -- reactivates the capstone pain point (generating 28x28 MNIST images took minutes, 512x512 would be ~335x more expensive). Frames every lesson since as adding capability without addressing speed. Reveals the student already has the compression tool from Module 6.1.
- Core insight (Section 5): Pipeline overview (encode, diffuse in latent space, decode) followed by the "of course" chain (VAE preserves content + diffusion denoises tensors = of course combine them). Frozen-VAE pattern in WarningBlock. Pipeline comparison Mermaid diagram showing identical denoising loop with VAE bookends.
- Algorithm comparison (Section 6): Side-by-side ComparisonRow of pixel-space DDPM training (7 steps) vs latent diffusion training (7 steps) -- only steps 1, 5, and 7 differ. Side-by-side sampling comparison. "The algorithm does not change" is the core reveal.
- Computational cost (Section 7): Dimension walkthrough code block with specific tensor shapes. ComparisonRow (262,144 spatial positions vs 4,096). Concrete scaling: 64x64 latent is only ~5x larger than the 28x28 capstone, vs 512x512 pixel-space being ~335x larger.
- Why it works (Section 8): Latent space preserves perceptual content (callback to interpolation). SD's VAE uses perceptual + adversarial loss for sharper results. Negative example: raw 64x64x4 latent is not a viewable image.
- Check sections (Section 9): three predict-and-verify questions (quality tradeoff misconception, frozen VAE reasoning, standard autoencoder vs VAE for diffusion). Transfer question (audio latent diffusion).
- Complete picture (Section 10): six GradientCards mapping every SD component to the lesson/module where it was taught, plus a seventh for latent space (this lesson). ModuleCompleteBlock for Module 6.3.

**Misconceptions addressed:**
1. "Latent diffusion is a fundamentally different algorithm from pixel-space diffusion" -- ComparisonRow showing identical training algorithms except encode step, variable names (z vs x), and frozen VAE
2. "The VAE is trained jointly with the diffusion model (end-to-end)" -- WarningBlock explaining frozen-VAE pattern; TipBlock explaining why gradients through VAE would destabilize latent space
3. "Diffusion in latent space produces blurry images because the VAE is blurry" -- addressed in Section 8; SD's VAE uses perceptual + adversarial loss; latent space discards imperceptible detail, concentrates on meaningful structure
4. "You need a special VAE designed for diffusion" -- TipBlock: the VAE was designed for compression, not for diffusion; modularity is the point; any sufficiently good VAE could work
5. "Latent diffusion sacrifices image quality for speed" -- predict-and-verify question with reveal; result is both faster AND often better because U-Net focuses on semantically meaningful aspects
6. "The latent output is a small image" -- WarningBlock: 64x64x4 is an abstract representation, not a downscaled image; decoder is essential

**What is NOT covered (deferred):**
- Implementing latent diffusion from scratch (Module 6.4 capstone)
- SD VAE architecture details (KL-regularized autoencoder with ResNet blocks)
- Perceptual loss or adversarial training in depth (MENTIONED only)
- DDIM or accelerated samplers (Module 6.4)
- LoRA, fine-tuning, inference optimization (Module 6.5)
- SD v1 vs v2 vs XL differences
- Negative prompts, prompt engineering

**Review notes:**
- Iteration 1: NEEDS REVISION -- 0 critical, 3 improvements (Check section before quality explanation was earned, missing "of course" intuitive chain, notebook Exercise 4 code cell comments too detailed for Independent), 3 polish (spaced em dash in NextStepBlock, ModuleCompleteBlock referencing inconsistency, notebook dtype note)
- Iteration 2: NEEDS REVISION -- 0 critical, 1 improvement (notebook Exercise 4 markdown tips provide implementation-level scaffolding undermining Independent label), 0 polish
- Iteration 3: PASS -- all findings resolved; lesson and notebook pedagogically sound
