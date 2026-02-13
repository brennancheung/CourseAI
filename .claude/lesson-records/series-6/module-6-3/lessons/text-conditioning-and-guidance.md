# text-conditioning-and-guidance -- Lesson Planning Document

**Module:** 6.3 (Architecture & Conditioning)
**Position:** Lesson 4 of 5 (Lesson 13 in Series 6)
**Type:** BUILD (cross-attention builds on deep self-attention knowledge; CFG is procedurally simple but conceptually new)
**Previous lesson:** clip (STRETCH)
**Next lesson:** from-pixels-to-latents (CONSOLIDATE)

---

## Phase 1: Student State

### Relevant concepts the student has

| Concept | Depth | Source Lesson | Notes |
|---------|-------|--------------|-------|
| Self-attention mechanism (Q/K/V projections, scaled dot-product attention, weighted average of V vectors) | DEVELOPED | Series 4.2 (the-problem-attention-solves through values-and-attention-output) | Deeply taught over 3 lessons with worked examples, formula evolution, interactive widget. Student can trace Q = W_Q X, K = W_K X, V = W_V X, output = softmax(QK^T / sqrt(d_k)) V. This is the foundation for understanding cross-attention -- same formula, different source for K and V. |
| Multi-head attention (h independent heads in parallel, dimension splitting d_k = d_model/h, output projection W_O) | DEVELOPED | multi-head-attention (4.2.4) | Student understands that multiple heads capture diverse relationship types. "Multiple lenses, pooled findings." Relevant because the U-Net uses multi-head cross-attention at each layer. |
| Cross-attention as an architectural concept (Q from one source, K/V from another; encoder-decoder variant of transformer) | MENTIONED | decoder-only-transformers (4.2.6) | Mentioned in the three-variant comparison: "encoder-decoder has two stacks + cross-attention." The student knows cross-attention EXISTS and that it connects two different sequences, but has never seen the mechanism explained or the formula written out. This is a GAP -- the module plan calls cross-attention "review from 4.2," but the records show it was only MENTIONED, not DEVELOPED. |
| "Learned lens" / nn.Linear as projection (same input, different matrix, different view) | DEVELOPED | queries-and-keys (4.2.2) | The Q/K/V projection pattern. Directly applicable: in cross-attention, Q comes from spatial features, K and V come from text embeddings via different learned projections. |
| U-Net encoder-decoder architecture with skip connections | DEVELOPED | unet-architecture (6.3.1) | Full dimension walkthrough, dual-path mental model (bottleneck = WHAT, skips = WHERE). Student knows the spatial architecture thoroughly. |
| Attention layers within the U-Net (self-attention, cross-attention interleaved at middle resolutions) | MENTIONED | unet-architecture (6.3.1) | Brief preview: "interleaved at middle resolutions (16x16, 32x32), self-attention relates distant parts of feature map, cross-attention connects to text embeddings." Student knows they exist and roughly where they live, but has no mechanistic understanding. |
| Sinusoidal timestep embedding + adaptive group normalization (timestep conditioning at every residual block) | DEVELOPED | conditioning-the-unet (6.3.2) | Student understands global conditioning: one signal (timestep) injected at every layer via adaptive normalization. This is the "first conditioning mechanism." This lesson adds the second. |
| Global conditioning pattern (information that affects every spatial location, injected at every processing stage) | INTRODUCED | conditioning-the-unet (6.3.2) | Named and explained. Timestep is the canonical example. The lesson previewed "spatially-varying conditioning" as a contrast (text via cross-attention). This lesson delivers on that preview. |
| CLIP dual-encoder architecture (image encoder + text encoder, shared embedding space, contrastive training) | INTRODUCED | clip (6.3.3) | Student knows CLIP produces 512-dim text embeddings that live in a shared space with image embeddings. Knows the training paradigm. Does NOT know how these embeddings connect to the U-Net. |
| Shared embedding space (text and image vectors in the same geometric space, cosine similarity as alignment measure) | DEVELOPED | clip (6.3.3) | Student experienced the before/after training diagram, computed similarity matrices, explored a real CLIP model. Understands that CLIP text embeddings encode visual meaning. The key bridge: these embeddings ARE the conditioning signal for cross-attention. |
| Contrastive learning paradigm | DEVELOPED | clip (6.3.3) | Student understands the training objective, the loss function, why negatives come from the batch, why scale matters. Relevant context for understanding what the text embeddings represent. |
| CLIP limitations (typographic attacks, spatial relationships, counting, novel compositions) | INTRODUCED | clip (6.3.3) | Important context for understanding CFG: CLIP embeddings capture statistical co-occurrence, not perfect understanding. CFG amplifies whatever signal the text embeddings provide. |
| Batch normalization gamma/beta (learned scale and shift after normalization) | DEVELOPED | training-dynamics (1.3.6), resnets (3.2) | Foundation for understanding that adaptive group norm is one conditioning mechanism (global) and cross-attention is another (spatially varying). |
| Residual blocks within the U-Net (Conv -> AdaGN -> Activation -> Conv -> AdaGN -> Activation + skip) | DEVELOPED | conditioning-the-unet (6.3.2) | The student saw the complete residual block with timestep conditioning injection points. This lesson adds cross-attention blocks interleaved with these residual blocks. |

### Mental models and analogies already established

- **"Three lenses, one embedding" / "Learned lens"** -- Q, K, V as three different projections of the same input. Directly applicable: in cross-attention, Q projects from spatial features, K and V project from text embeddings.
- **"Bottleneck decides WHAT, skip connections decide WHERE"** -- U-Net dual-path information flow. Text conditioning via cross-attention tells the bottleneck WHAT to generate.
- **"The conductor's score"** -- timestep embedding as global conditioning. Text conditioning is a DIFFERENT kind of signal: it varies spatially (different patches attend to different words).
- **"Two encoders, one shared space -- the loss function creates the alignment"** -- CLIP mental model. The text embeddings that enter cross-attention are the output of CLIP's text encoder.
- **"Same building blocks, different question"** -- recurring meta-pattern. Cross-attention is the same QKV formula, applied to a different question (what text is relevant to this spatial location?).
- **"Attention reads, FFN writes"** -- transformer block mental model. Cross-attention in the U-Net "reads" from text embeddings.
- **"Geometry encodes meaning"** -- embedding spaces organize semantically. The CLIP embedding space encodes visual-linguistic meaning.

### What was explicitly NOT covered in prior lessons (relevant here)

- **Cross-attention mechanism** -- only MENTIONED. The student knows Q comes from one source and K/V from another, but has never seen the formula, a worked example, or the intuition developed. Module 4.2 explicitly listed "cross-attention" as NOT covered ("Self-attention vs cross-attention distinction" was listed as not covered in the-problem-attention-solves; "Cross-attention" was listed as not covered in multi-head-attention, the-transformer-block, and decoder-only-transformers).
- **How CLIP text embeddings connect to the U-Net** -- deferred from clip (6.3.3). The student knows CLIP produces embeddings but not how they enter the denoising process.
- **Classifier-free guidance** -- entirely untaught. The idea of training with randomly dropped conditioning, running two forward passes at inference, and amplifying the difference is completely new.
- **Where attention layers sit within the U-Net architecture** -- MENTIONED briefly ("interleaved at middle resolutions") but never developed. Which resolutions, how interleaved with residual blocks, the block ordering.
- **Self-attention within the U-Net** -- MENTIONED. How the U-Net's spatial features attend to other spatial features. Related to cross-attention but a separate concept.
- **The practical effect of guidance scale** -- how varying the guidance scale trades off between fidelity to the text prompt and image quality/diversity.

### Readiness assessment

The student is well-prepared for the self-attention-to-cross-attention transfer, but cross-attention requires more development than the module plan anticipated. The module plan describes cross-attention as "review from 4.2," but the records show it was only MENTIONED in 4.2 (in the encoder-decoder variant comparison). It was never explained mechanistically, never given a formula, and never practiced. This means the lesson needs to DEVELOP cross-attention, not merely review it.

The good news: the conceptual delta from self-attention to cross-attention is small. The student has the full QKV formula at DEVELOPED depth. Cross-attention changes only WHERE Q, K, and V come from (Q from one source, K/V from another), not HOW the computation works. The formula is identical. The student should feel "oh, you just change which input you project K and V from."

For classifier-free guidance (CFG), the student has no prior exposure. CFG is procedurally simple (train with random text dropout, run two forward passes, take a weighted difference) but conceptually subtle (what does "amplifying the text signal" mean geometrically? why does this work?). The student's strong embedding space intuition from CLIP (DEVELOPED) provides the foundation for understanding what CFG does in the output space.

The cognitive load is appropriate for BUILD: one concept that is a small delta from deep prior knowledge (cross-attention), one genuinely new but procedurally simple concept (CFG).

---

## Phase 2: Analysis

### Target concept

This lesson teaches the student how text descriptions steer the diffusion U-Net's denoising process -- via cross-attention that lets spatial features attend to CLIP text embeddings, and classifier-free guidance that amplifies the text's influence at inference time.

### Prerequisites table

| Concept | Required Depth | Actual Depth | Source Lesson | Status | Reasoning |
|---------|---------------|-------------|---------------|--------|-----------|
| Self-attention QKV formula (output = softmax(QK^T / sqrt(d_k)) V) | DEVELOPED | DEVELOPED | values-and-attention-output (4.2.3) | OK | Cross-attention uses the identical formula. Student needs the formula at DEVELOPED to see that only the source of K/V changes. Three-lesson arc built this thoroughly with worked examples. |
| Q/K/V as learned projections (W_Q, W_K, W_V as "learned lenses") | DEVELOPED | DEVELOPED | queries-and-keys (4.2.2) / values-and-attention-output (4.2.3) | OK | Student needs to understand that projections can come from DIFFERENT inputs. The "learned lens" framing transfers directly: W_Q applied to spatial features, W_K and W_V applied to text embeddings. |
| Multi-head attention (parallel heads, dimension splitting, W_O output projection) | DEVELOPED | DEVELOPED | multi-head-attention (4.2.4) | OK | The U-Net uses multi-head cross-attention. Student needs to understand that multiple heads attend to different aspects of the text simultaneously. |
| Cross-attention as a concept (Q from one source, K/V from another) | INTRODUCED | MENTIONED | decoder-only-transformers (4.2.6) | GAP (medium) | Student recognizes the term and knows it connects two different sequences, but has never seen the mechanism explained. The formula is identical to self-attention; the difference is the source of K/V. A dedicated section is needed to develop this from MENTIONED to DEVELOPED. |
| CLIP text embeddings (512-dim vectors encoding visual-linguistic meaning in a shared space) | INTRODUCED | INTRODUCED | clip (6.3.3) | OK | Student knows what CLIP text embeddings represent. Does not need to know implementation details. Needs to understand that a text prompt produces a SEQUENCE of embeddings (one per token), not a single vector. This is a small clarification, not a gap. |
| U-Net architecture with residual blocks and skip connections | DEVELOPED | DEVELOPED | unet-architecture (6.3.1) | OK | Student needs to know the spatial architecture to understand where cross-attention layers are inserted. Fully developed. |
| Timestep conditioning via adaptive group normalization | DEVELOPED | DEVELOPED | conditioning-the-unet (6.3.2) | OK | The student already knows one conditioning mechanism. Text conditioning via cross-attention is the second. The contrast (global vs spatially-varying) is a key conceptual move. |
| Attention layers within the U-Net (mentioned location: middle resolutions) | INTRODUCED | MENTIONED | unet-architecture (6.3.1) | GAP (small) | The student knows attention layers exist in the U-Net and roughly where. This lesson develops the specifics: which resolutions, what ordering within a block (residual -> self-attention -> cross-attention), why not at every resolution. |
| Global conditioning pattern (information that affects every spatial location) | INTRODUCED | INTRODUCED | conditioning-the-unet (6.3.2) | OK | Needed as the contrast: timestep conditioning is global, text conditioning is spatially varying. The previous lesson explicitly previewed this contrast. |
| Softmax and probability distributions | DEVELOPED | DEVELOPED | Multiple (4.1.1, 4.2.1-4.2.3) | OK | Needed for understanding attention weights in cross-attention and for intuiting what CFG does to the predicted noise. |

### Gap resolution

| Gap | Size | Resolution |
|-----|------|------------|
| Cross-attention (MENTIONED -> DEVELOPED needed) | Medium (student has self-attention at DEVELOPED depth; cross-attention is the same formula with K/V sourced from a different input, but the module plan's assumption that it is "review" is incorrect -- it needs a dedicated explanation section) | Dedicated section within this lesson. Present cross-attention as a one-line change from self-attention: in self-attention, Q = W_Q X, K = W_K X, V = W_V X (all from the same input X). In cross-attention, Q = W_Q X_spatial, K = W_K X_text, V = W_V X_text (Q from the U-Net's spatial features, K and V from the text embeddings). Same formula, different inputs. Side-by-side comparison (self-attention vs cross-attention). The student should feel "oh, I already know this formula." |
| Attention layers' position in U-Net (MENTIONED -> INTRODUCED needed) | Small (student knows they exist at middle resolutions) | Brief subsection showing the block ordering at a cross-attention resolution: ResidualBlock(with AdaGN) -> SelfAttention -> CrossAttention -> ResidualBlock. Explain why attention is only at middle resolutions (16x16, 32x32): attention is O(n^2) in sequence length, and 64x64 = 4096 tokens would be prohibitively expensive. At 16x16 = 256 tokens, attention is affordable. |

### Misconceptions table

| Misconception | Why They'd Think This | Negative Example That Disproves It | Where to Address |
|---------------|----------------------|-----------------------------------|-----------------|
| "Cross-attention in the U-Net is fundamentally different from the attention mechanism I learned in Series 4.2" | The U-Net is a convolutional architecture, and the student learned attention in the context of language modeling. The application domain is so different that the student may assume the mechanism itself differs. | Write out the cross-attention formula side-by-side with the self-attention formula. They are identical: output = softmax(QK^T / sqrt(d_k)) V. The only difference is where Q, K, V come from. Q = W_Q (spatial features), K = W_K (text embeddings), V = W_V (text embeddings). Same three projection matrices, same dot-product scoring, same softmax, same weighted average. | Section 4 (Explain -- Cross-Attention), as the core reveal. Address immediately because if the student thinks the mechanism is different, everything else in the lesson fails to connect. |
| "Cross-attention processes the entire text and entire image at once (like a single big dot product)" | The student might think cross-attention computes one global similarity between the full text and the full image feature map, producing one number. | Cross-attention is per-spatial-location. Each pixel in the 16x16 feature map generates its own query vector and computes its own attention weights over ALL text tokens. Different spatial locations can attend to different words. The cat's ear region attends to "cat," the background region attends to "sunset." The output is a feature map, not a scalar. | Section 4 (Explain -- Cross-Attention), after the formula. Concrete example: 16x16 feature map = 256 query vectors, each attending over a text sequence of ~77 tokens. |
| "Classifier-free guidance means training two separate models (one with text, one without)" | The word "classifier-free" and the description "run two forward passes" might suggest two different networks. Also, the original classifier guidance DID use a separate classifier model. | CFG uses ONE model. During training, the text conditioning is randomly dropped (replaced with a null/empty embedding) some percentage of the time (e.g., 10%). This teaches the single model to operate both conditionally and unconditionally. At inference, the same model runs twice with the same weights: once with the text prompt, once with the empty embedding. | Section 6 (Explain -- CFG), as a WarningBlock immediately after introducing the concept. |
| "Higher guidance scale always means better images" | The student might assume that since guidance amplifies the text signal, more amplification = better adherence = better images. Linear thinking: if some is good, more is better. | At extreme guidance scales (e.g., w=20), images become oversaturated, distorted, and artifact-heavy. The model over-optimizes for the text signal at the expense of image coherence. This is like cranking the volume on one instrument until it drowns out the rest of the orchestra and distorts. The sweet spot is typically w=7-9 for Stable Diffusion. Show the tradeoff: low w = diverse but may ignore text, medium w = balanced, high w = high fidelity to text but distorted. | Section 7 (Elaborate), with a concrete tradeoff spectrum. |
| "Text conditioning replaces timestep conditioning (the network now responds to text instead of the timestep)" | The student might see text conditioning as a replacement rather than an addition. After all, the previous lesson was about timestep conditioning, and this one is about text conditioning. | Both conditioning signals are present simultaneously and serve different purposes. The timestep tells the network WHEN it is in the denoising process (noise level). The text tells it WHAT to generate (semantic content). Adaptive group norm handles the timestep at every block. Cross-attention handles the text at attention-resolution blocks. They coexist in the same forward pass. The residual block + attention block ordering makes this clear: ResidualBlock(with AdaGN for timestep) -> CrossAttention(with text embeddings). | Section 5 (Explain -- Where Cross-Attention Lives), in a WarningBlock after showing the block ordering. |

### Examples planned

| Example | Type | Purpose | Why This Example |
|---------|------|---------|-----------------|
| Self-attention vs cross-attention side-by-side formula | Positive (bridge) | Show that cross-attention is the same formula as self-attention with one change: K and V are projected from the text embeddings instead of from the spatial features. Side-by-side: Self: Q = W_Q X, K = W_K X, V = W_V X. Cross: Q = W_Q X_spatial, K = W_K X_text, V = W_V X_text. Same softmax(QK^T / sqrt(d_k)) V. | Leverages the student's deepest prior knowledge (self-attention at DEVELOPED from a three-lesson arc). Makes cross-attention feel immediately familiar. Directly addresses misconception #1. The minimal delta approach mirrors how adaptive group norm was taught as a minimal delta from batch norm gamma/beta (conditioning-the-unet). |
| Per-spatial-location cross-attention: "a cat sitting in a sunset" | Positive (concrete) | Show how different spatial locations in a 16x16 feature map attend to different parts of the text. The patch near the cat attends strongly to "cat," the sky region attends to "sunset," and the overall composition attends to "sitting." Concrete attention weight numbers for 2-3 spatial locations over a 6-token text. | Makes the spatially-varying nature of text conditioning visceral. Prevents misconception #2 (single global similarity). Shows that cross-attention produces a rich, position-dependent signal, not a uniform global one. Contrasts with timestep conditioning (uniform across all spatial positions). |
| CFG as "amplifying the text direction" in noise prediction space | Positive (geometric) | The unconditional prediction points in one direction (what the model would generate without any text). The conditional prediction points in a slightly different direction (what the model generates with text). CFG amplifies the DIFFERENCE: noise_cfg = noise_uncond + w * (noise_cond - noise_uncond). The vector from unconditional to conditional is the "text direction" -- the effect the text has on the prediction. CFG scales this vector by w. | Makes CFG geometric and spatial rather than purely formulaic. The student has strong embedding space intuition from CLIP. This example puts CFG in the same visual language: directions in a space, scaling a difference vector. The formula becomes obvious once the geometric picture is clear. |
| Negative: guidance scale too high (oversaturation and artifacts) | Negative | Show that extreme guidance scale (w=20) produces distorted images. The model over-commits to the text signal, producing images that technically match the prompt but look unnatural -- oversaturated colors, repeated textures, distorted anatomy. CFG is not "free quality" -- it is a tradeoff between text fidelity and image naturalness. | Addresses misconception #4 (higher is always better). Establishes that CFG has a sweet spot and the guidance scale is a user-facing parameter that requires tuning. Grounds the formula in practical consequences. |
| Negative: text conditioning via concatenation or global averaging | Negative | What if instead of cross-attention, you averaged all text token embeddings into one vector and injected it via adaptive normalization (like the timestep)? The text would affect all spatial locations uniformly. "A cat sitting in a sunset" would push the ENTIRE image toward cat-ness and sunset-ness equally. No spatial selectivity. The cat's ear and the sky get the same text signal. Cross-attention solves this: each location queries the text and takes what it needs. | Addresses misconception #5 indirectly and motivates why a NEW conditioning mechanism (cross-attention) is needed rather than reusing the existing one (adaptive normalization). The student already understands global conditioning from the previous lesson -- this shows why text requires something different. Connects to the global vs spatially-varying contrast previewed in conditioning-the-unet. |

---

## Phase 3: Design

### Narrative arc

The student has spent two lessons building the unconditional diffusion architecture (U-Net with timestep conditioning) and one lesson understanding CLIP (the model that turns text into vectors encoding visual meaning). The pieces are on the table but not connected. The gap is palpable: the student knows the U-Net can denoise images, knows CLIP can turn "a cat sitting in a sunset" into a sequence of 512-dim vectors, but has no idea how those vectors enter the denoising process. This lesson makes the connection -- and it turns out the mechanism is one the student already knows deeply from Series 4.2. Cross-attention, the same QKV formula that powers transformers, is how the U-Net's spatial features "read" from the text embeddings. Each pixel's features generate a query; the text tokens provide keys and values. Different spatial locations attend to different words, creating a spatially-varying conditioning signal that tells each part of the image what it should become. But there is a catch: cross-attention alone does not produce strong text adherence. The model can learn to partially ignore the text if the training signal is weak. Classifier-free guidance solves this -- a beautifully simple technique where you train the model to work with AND without text, then at inference time amplify the difference. The emotional arc: "Wait, cross-attention is just the same QKV formula?" (relief and connection) followed by "CFG is just subtracting two predictions and scaling the difference?" (elegant simplicity) followed by "But the guidance scale matters a lot -- too much and it distorts" (practical nuance).

### Modalities planned

| Modality | What Specifically | Why This Modality for This Concept |
|----------|------------------|------------------------------------|
| Verbal/Analogy | (1) Cross-attention as "reading from a reference document" -- self-attention is the U-Net's spatial features talking to each other ("what is nearby?"), cross-attention is the spatial features consulting a text description ("what should I be?"). (2) CFG as "turning up the contrast on the text signal" -- like adjusting the contrast slider on a photo, except the "contrast" is between the text-conditioned and unconditioned predictions. | The "reference document" analogy maps directly to the "attention reads" mental model from 4.2.5 (the-transformer-block). Self-attention reads from other spatial features; cross-attention reads from the text. The "contrast slider" analogy makes the guidance scale intuitive and immediately suggests that too much contrast looks bad (oversaturation). |
| Visual (diagram) | (1) Cross-attention data flow diagram: U-Net spatial features (16x16 grid) on one side, CLIP text token embeddings (sequence of ~77 vectors) on the other, Q arrows from spatial features, K/V arrows from text, attention weights as lines of varying thickness between them. (2) U-Net block ordering diagram at an attention resolution: ResidualBlock(AdaGN) -> SelfAttention -> CrossAttention -> ResidualBlock(AdaGN), with timestep arrows into residual blocks and text embedding arrows into cross-attention. (3) CFG vector diagram: two arrows from a common origin (unconditional and conditional noise predictions), a difference vector between their endpoints, and the CFG formula showing the scaled amplification. | Cross-attention needs a spatial diagram because the student's prior attention experience was with 1D sequences (text tokens). Seeing 2D spatial features attend to 1D text tokens is a new spatial configuration. The block ordering diagram shows how both conditioning signals (timestep and text) coexist in the architecture. The CFG vector diagram makes the formula geometric rather than algebraic. |
| Symbolic (formula/code) | (1) Cross-attention formula side-by-side with self-attention: Self: Q=W_Q X, K=W_K X, V=W_V X; Cross: Q=W_Q X_spatial, K=W_K X_text, V=W_V X_text. Output = softmax(QK^T / sqrt(d_k)) V in both cases. (2) CFG formula: noise_cfg = noise_uncond + w * (noise_cond - noise_uncond). Rearranged: noise_cfg = (1-w) * noise_uncond + w * noise_cond. (3) Shape annotations: X_spatial is (H*W, d_model), X_text is (T, d_text), Q is (H*W, d_k), K is (T, d_k), attention weights are (H*W, T). | The side-by-side formula is the core pedagogical move for cross-attention -- it proves the mechanism is identical. The CFG formula is simple enough that the student should see it as one line of code. The shape annotations ground the abstract formula in concrete tensor dimensions, connecting to the U-Net's spatial resolution and text sequence length. |
| Concrete example | (1) Cross-attention with numbers: 4x4 spatial feature map (16 locations), 6-word text prompt, show attention weights for 2-3 specific spatial positions attending differently across the 6 text tokens. (2) CFG with specific noise vectors: unconditional prediction = [0.3, -0.1, 0.5], conditional prediction = [0.5, -0.3, 0.8], CFG at w=7.5: [0.3 + 7.5 * (0.2, -0.2, 0.3)] = [1.8, -1.6, 2.55]. Show how the text direction is amplified. | Numbers make the abstract formula tangible. The cross-attention example shows that different spatial locations produce genuinely different attention patterns. The CFG example shows how a modest difference between conditional and unconditional predictions becomes a large signal after amplification. |
| Intuitive | The "of course" chain: (1) The U-Net needs text information, but text information is spatially varying (the word "cat" should influence the cat region, not the sky). So a spatially-uniform mechanism (like adaptive norm) is wrong for text. Each spatial location needs to selectively extract relevant information from the text. That is exactly what attention does -- selective extraction. So cross-attention is the natural choice. (2) Cross-attention alone gives a weak signal -- the model can learn to partially ignore it. To strengthen the signal, compare "what the model predicts with the text" to "what it predicts without" and amplify the difference. That difference IS the text's effect. CFG is just "measure the effect and scale it up." | The intuitive chain makes both cross-attention and CFG feel inevitable rather than arbitrary design choices. The student should feel "of course that is how you would do it" rather than "this is a clever trick someone invented." |

### Cognitive load assessment

- **New concepts in this lesson:** 2 genuinely new concepts:
  1. **Cross-attention in the U-Net** -- but this is a small delta from self-attention (DEVELOPED). The formula is identical; only the source of K/V changes. The novelty is the APPLICATION (2D spatial features attending to 1D text embeddings) and the LOCATION (where it sits in the U-Net). The mechanism itself is not new.
  2. **Classifier-free guidance** -- genuinely new training and inference technique. The student has no prior exposure. However, the procedure is simple: randomly drop text during training, run two forward passes at inference, compute a weighted combination. The formula is one line. The subtlety is in understanding WHY it works and what the guidance scale controls.

  The block ordering within the U-Net (residual -> self-attention -> cross-attention) is a structural detail, not a new concept.

- **Previous lesson load:** STRETCH (clip -- genuinely new training paradigm with contrastive learning)
- **This lesson's load:** BUILD -- appropriate. After a STRETCH lesson, the student needs lower cognitive load. Cross-attention is heavily leveraged from prior knowledge. CFG is new but procedurally simple. The trajectory: BUILD -> BUILD -> STRETCH -> **BUILD** -> CONSOLIDATE.

### Connections to prior concepts

| Prior Concept | How It Connects | Risk of Misleading? |
|---------------|----------------|--------------------|
| Self-attention QKV formula (4.2.1-4.2.3) | Cross-attention is the same formula. The student's three-lesson arc building up to output = softmax(QK^T / sqrt(d_k)) V applies directly. The only change: K and V are projected from a different input (text embeddings instead of spatial features). | Low risk. The transfer is clean. The main difference (Q from one source, K/V from another) is exactly what the student needs to learn. |
| "Learned lens" / nn.Linear projections (4.2.2) | W_Q projects spatial features into the query space. W_K and W_V project text embeddings into key and value spaces. Same "learned lens" pattern, but now the two lenses are applied to different inputs. | Low risk. The analogy extends naturally. |
| "Attention reads, FFN writes" (4.2.5) | Self-attention reads from other spatial features. Cross-attention reads from the text embeddings. Same "reading" operation, different "document." The U-Net's spatial features read from two sources: themselves (self-attention for spatial coherence) and the text (cross-attention for semantic guidance). | Low risk. The reading analogy extends cleanly. |
| Global conditioning / adaptive group norm (6.3.2) | Text conditioning via cross-attention is explicitly contrasted with timestep conditioning via adaptive norm. Timestep = global (same signal everywhere). Text = spatially varying (different locations attend to different words). The previous lesson previewed this contrast. | Low risk. The contrast reinforces both concepts. |
| CLIP text embeddings (6.3.3) | The text embeddings produced by CLIP's text encoder are the K/V input to cross-attention. The student knows what these embeddings represent (visual-linguistic meaning). This lesson shows where they go. | Low risk. The connection is the whole point of the narrative arc. One clarification needed: CLIP produces a SEQUENCE of token embeddings (one per text token), not just the single [CLS] embedding used for similarity. |
| Embedding space directions / vector arithmetic (6.1.4, 6.3.3) | CFG amplifies the difference between conditional and unconditional predictions. This difference is a DIRECTION in noise prediction space -- the direction the text pushes the prediction. The student has experience with meaningful directions in latent spaces (latent arithmetic from 6.1.4). | Moderate risk. Latent arithmetic was shown to be unreliable for most directions (6.1.4 tempered this). CFG's "text direction" is more reliable because it is computed per-step, not as a global property of the space. Note the difference. |
| Temperature scaling (4.1.1, 6.3.3) | The guidance scale w in CFG functions similarly to an inverse temperature: it controls how sharply the model commits to the text signal. Higher w = sharper commitment to text, like lower temperature = sharper commitment to top prediction. | Moderate risk. The analogy is imperfect -- temperature scales logits before softmax (a probability distribution), while guidance scale scales a noise prediction (a continuous vector). Use the analogy briefly for intuition but note the structural difference. |

### Scope boundaries

**This lesson IS about:**
- Cross-attention as the mechanism for injecting text embeddings into the U-Net (Q from spatial features, K/V from text)
- Where cross-attention layers live in the U-Net (interleaved at middle resolutions: 16x16 and 32x32)
- The spatially-varying nature of text conditioning (different spatial locations attend to different words)
- Classifier-free guidance: training with random text dropout, inference with two forward passes, amplifying the text signal
- The guidance scale parameter and its tradeoff (low = diverse but unfaithful, medium = balanced, high = oversaturated)
- The contrast between global conditioning (timestep via adaptive norm) and spatially-varying conditioning (text via cross-attention)

**This lesson is NOT about:**
- Implementing cross-attention or CFG from scratch (the capstone in Module 6.4 handles implementation)
- Self-attention within the U-Net (mentioned for context but not developed)
- The full Stable Diffusion pipeline (Module 6.4)
- Latent diffusion / the VAE encoder-decoder (next lesson: from-pixels-to-latents)
- Negative prompts or prompt engineering techniques
- CLIP architecture or training details (covered in the previous lesson)
- Classifier guidance (the predecessor to CFG that DID require a separate classifier -- mentioned for historical context only)
- Alternative text conditioning approaches (e.g., T5 encoder in Imagen)
- LoRA or fine-tuning (Series 6 later modules or Series 7)
- The specific text encoder used in Stable Diffusion v1 vs v2 (a detail that does not affect understanding)

**Depth targets:**
- Cross-attention mechanism: DEVELOPED (student can explain the formula, trace Q from spatial features and K/V from text, understand the per-spatial-location nature, identify where it sits in the U-Net)
- Classifier-free guidance: DEVELOPED (student can explain training with text dropout, inference with two forward passes, the CFG formula, and the guidance scale tradeoff)
- Spatially-varying vs global conditioning: INTRODUCED (can contrast the two patterns and identify which mechanism serves which type of conditioning)
- Self-attention within the U-Net: INTRODUCED (knows it exists, knows it enables spatial features to communicate, knows it sits alongside cross-attention)
- Classifier guidance (predecessor): MENTIONED (name-dropped for context, motivated CFG as the replacement that does not need a separate classifier)

### Lesson outline

**1. Context + Constraints**
- This lesson connects CLIP's text embeddings to the U-Net's denoising process. Two topics: how text enters the architecture (cross-attention) and how to amplify the text signal at inference (classifier-free guidance).
- Scope: cross-attention mechanism and CFG. NOT implementing either (that is the capstone). NOT latent diffusion (next lesson).
- By the end: the student can explain the full conditioning pipeline -- timestep via adaptive norm (global), text via cross-attention (spatially varying), CFG to strengthen text adherence.

**2. Recap (brief -- cross-attention gap fill + CLIP embedding clarification)**
- **Cross-attention clarification:** In decoder-only-transformers (4.2.6), the student learned that encoder-decoder transformers have "cross-attention" where the decoder attends to the encoder's output. This lesson develops the mechanism. Quick callback: "You know the QKV formula. In self-attention, Q, K, and V all come from the same input. In cross-attention, Q comes from one input and K/V come from another. Same formula, different sources."
- **CLIP embedding clarification:** CLIP produces a SEQUENCE of token embeddings (one per text token in the prompt), not just a single summary vector. When we say "CLIP text embeddings," we mean this sequence. For the U-Net, these token-level embeddings become the keys and values in cross-attention. The prompt "a cat sitting in a sunset" produces ~7 token embeddings, each 512-dim. This is the "document" that spatial features will read from.

**3. Hook: "The U-Net still cannot understand text"**
- Type: Problem-before-solution, callback to conditioning-the-unet.
- Recall the previous lesson's end note: "The timestep tells the network WHEN it is in the denoising process. The text will tell it WHAT to generate."
- The student now has both pieces (U-Net with timestep conditioning, CLIP text embeddings) but they are disconnected. How do you inject a sequence of text embeddings into a convolutional architecture?
- First instinct might be: average all text tokens into one vector, inject via adaptive normalization (like the timestep). But this loses spatial selectivity -- "a cat sitting in a sunset" should not push the entire image toward cat-ness and sunset-ness uniformly. The word "cat" should influence the cat region, "sunset" should influence the sky.
- What you need: a mechanism where each spatial location can selectively extract relevant information from the text. "You already know a mechanism that does exactly this."

**4. Explain -- Cross-Attention in the U-Net (DEVELOPED)**

**4a. The one-line change from self-attention:**
- Self-attention: Q = W_Q X, K = W_K X, V = W_V X (all from the same input X)
- Cross-attention: Q = W_Q X_spatial, K = W_K X_text, V = W_V X_text
- Output = softmax(QK^T / sqrt(d_k)) V -- SAME FORMULA
- Side-by-side in a ComparisonRow. The student should feel immediate recognition.
- "In self-attention, each token asks 'what in my sequence is relevant to me?' In cross-attention, each spatial location asks 'what in the text is relevant to me?'"
- Callback to "attention reads": self-attention reads from other spatial features, cross-attention reads from the text. Same reading operation, different source document.
- InsightBlock: "This is the same formula you built across three lessons in Module 4.2. The only change is where K and V come from."

**4b. Per-spatial-location attention (addressing misconception #2):**
- The 16x16 feature map has 256 spatial locations. Each location generates its own query vector. Each query attends independently over ALL text tokens.
- Shape walkthrough: X_spatial reshaped to (H*W, d) = (256, d). X_text is (T, d_text) where T is the number of text tokens (~77 max in CLIP). Q is (256, d_k), K is (T, d_k), V is (T, d_v). Attention weights are (256, T) -- each of 256 spatial locations has its own distribution over T text tokens.
- Concrete example: "a cat sitting in a sunset." The spatial location near the cat's face generates a query that attends strongly to "cat." The sky region generates a query that attends to "sunset." The overall composition attends to "sitting." Different queries, same keys, different attention patterns. Show hypothetical attention weights for 2-3 locations.
- "This is what makes text conditioning SPATIALLY VARYING. The timestep modulates all locations the same way (global). The text gives each location its own signal."

**4c. Connection to the "three lenses" mental model:**
- W_Q is the spatial features' "seeking" lens: "what kind of text information do I need here?"
- W_K is the text's "advertising" lens: "what information does each word offer?"
- W_V is the text's "contributing" lens: "what information does each word actually deliver when matched?"
- Same three-lens pattern from values-and-attention-output (4.2.3). The novelty: the Q lens looks at spatial features, the K/V lenses look at text.

**5. Explain -- Where Cross-Attention Lives in the U-Net (INTRODUCED)**

**5a. Block ordering at attention resolutions:**
- At middle resolutions (16x16, 32x32), the U-Net interleaves three types of processing:
  1. ResidualBlock with AdaGN (timestep conditioning)
  2. Self-attention (spatial features attend to each other)
  3. Cross-attention (spatial features attend to text embeddings)
- This pattern repeats at each attention resolution, on both the encoder and decoder paths.
- Diagram: Mermaid or inline showing ResidualBlock(t_emb) -> SelfAttn -> CrossAttn(text_emb) -> ResidualBlock(t_emb) -> SelfAttn -> CrossAttn(text_emb) at one resolution level.

**5b. Why only at middle resolutions:**
- Attention is O(n^2) in sequence length. At 64x64 = 4096 "tokens," the attention matrix would be 4096 x 4096 = 16.7M entries. Expensive and memory-heavy.
- At 16x16 = 256 "tokens," the self-attention matrix is 256 x 256 = 65K entries. The cross-attention matrix is 256 x 77 = ~20K entries. Feasible.
- At 8x8 = 64 "tokens," attention is cheap but the bottleneck already has global receptive field, so self-attention adds less value.
- The sweet spot: 16x16 and 32x32. Large enough to benefit from long-range spatial dependencies, small enough for attention to be affordable.

**5c. WarningBlock: text and timestep conditioning coexist**
- Address misconception #5: both conditioning signals are present simultaneously.
- The timestep enters via adaptive group normalization in every residual block (global, every resolution).
- The text enters via cross-attention at attention resolutions (spatially varying, middle resolutions).
- They serve different purposes and do not compete. Timestep = WHEN (noise level). Text = WHAT (semantic content).
- The residual block handles "how much noise to remove." Cross-attention handles "what should be revealed as noise is removed."

**5d. Self-attention within the U-Net (INTRODUCED, brief):**
- Self-attention lets different spatial locations communicate. At 16x16, one part of the image can attend to another part 16 pixels away in a single layer.
- This enables global spatial coherence: if the model is generating a face, the left eye and right eye can "coordinate" through self-attention to be symmetric.
- Brief -- not the focus of this lesson. The point: self-attention handles spatial consistency, cross-attention handles text guidance. They sit side by side.

**6. Explain -- Classifier-Free Guidance (DEVELOPED)**

**6a. The problem CFG solves:**
- Cross-attention gives the model ACCESS to text information, but it does not guarantee the model will USE it strongly.
- During training, the model minimizes reconstruction error. If it can achieve low error by mostly ignoring the text (relying on the noisy image itself for predictions), it will. The text signal is one of many inputs, and the model may learn to weight it weakly.
- The result: text conditioning works but produces images that only loosely match the prompt. "A cat" generates something cat-like but with low fidelity.

**6b. The CFG training trick:**
- During training, randomly replace the text embedding with a null/empty embedding some fraction of the time (typically 10-20%).
- When the text is replaced with the null embedding, the model must predict noise unconditionally (like the Module 6.2 capstone model).
- When the text is present, the model predicts noise conditionally.
- ONE model learns BOTH tasks. No separate networks.
- WarningBlock: address misconception #3. "CFG uses one model. The text is randomly dropped during training, not a different model trained without text."

**6c. The CFG inference formula:**
- At inference, run the model TWICE for each denoising step:
  1. noise_uncond = model(x_t, t, null_embedding) -- "what would you predict without any text?"
  2. noise_cond = model(x_t, t, text_embedding) -- "what would you predict with the text?"
- Combine: noise_cfg = noise_uncond + w * (noise_cond - noise_uncond)
- The difference (noise_cond - noise_uncond) is the DIRECTION of the text's influence on the prediction. It captures "how does the text change what the model predicts?"
- The guidance scale w amplifies this direction. w=1 gives the conditional prediction (no amplification). w=0 gives unconditional. w=7.5 amplifies the text's effect by 7.5x.
- Rearranged: noise_cfg = (1-w) * noise_uncond + w * noise_cond. At w > 1, you are EXTRAPOLATING beyond the conditional prediction in the direction away from the unconditional one.
- Concrete example with numbers: 3-dim noise vectors showing unconditional, conditional, their difference, and the CFG output at w=7.5.

**6d. Geometric intuition for CFG:**
- Visual: Two arrows from a common origin point. One is the unconditional prediction, the other is the conditional prediction. The difference vector between their tips is the "text direction." CFG extends along this direction by a factor of w.
- Connection to latent arithmetic (6.1.4): the text direction is analogous to an attribute direction in latent space. But this is computed fresh at every denoising step, not as a global property of the space.
- "Contrast slider" analogy: CFG turns up the contrast between what the model predicts with vs without text. Medium contrast (w=7.5) produces vivid, text-faithful images. Too much contrast (w=20) oversaturates.

**7. Elaborate -- Guidance Scale Tradeoff + Historical Context**

**7a. The guidance scale tradeoff (addressing misconception #4):**
- Spectrum with 3 GradientCards:
  - w=1.0 (no guidance): Model uses text but does not amplify. Diverse, creative, but may loosely follow the prompt.
  - w=7.5 (typical for SD): Strong text adherence while maintaining image quality and coherence. The default in most Stable Diffusion UIs.
  - w=20 (extreme): Over-committed to text. Oversaturated colors, distorted anatomy, artifact-heavy. The model over-optimizes for the text signal at the expense of realism.
- "The guidance scale is a user-facing parameter. When you use Stable Diffusion and adjust 'CFG scale,' this is the w in the formula."

**7b. Why "classifier-free"? (historical context, MENTIONED depth):**
- Before CFG, there was "classifier guidance" (Dhariwal & Nichol, 2021): train a separate image classifier on noisy images, use its gradients to steer the diffusion process toward a class label.
- Problem: requires a separately trained classifier. Does not work with free-form text (only class labels). Complex.
- CFG (Ho & Salimans, 2022) eliminated the classifier entirely. Train ONE model with random text dropout. No external classifier needed. Works with any text prompt, not just class labels.
- "Classifier-free guidance is the standard approach in Stable Diffusion and virtually all modern text-to-image diffusion models."

**7c. CFG doubles compute:**
- Brief note: CFG requires TWO forward passes per denoising step (one conditional, one unconditional). This doubles the inference compute.
- In practice, the two passes can be batched together (batch of 2) for efficiency, but the fundamental cost is 2x.
- This is a real cost. At 50 denoising steps, that is 100 forward passes.

**8. Check (predict-and-verify)**
- "In cross-attention, the query comes from the U-Net's spatial features and the keys/values come from the text embeddings. If you swapped this -- queries from text, keys/values from spatial features -- what would happen?" (Answer: each text token would compute a weighted average of spatial features. The output would be a sequence of text-length vectors, each summarizing what the image looks like at locations relevant to that word. This is what CLIP does internally for image understanding -- not what the U-Net needs for generation. The U-Net needs spatial features enriched by text, not text features enriched by spatial information.)
- "Your colleague says: 'With CFG at w=1, you get the conditional prediction. With w=0, you get unconditional. So w=0.5 gives you a blend that is less text-dependent.' Is this correct?" (Answer: Yes, technically. noise_cfg = noise_uncond + 0.5 * (noise_cond - noise_uncond) = 0.5 * noise_uncond + 0.5 * noise_cond. This is a 50/50 blend. w < 1 produces images that are LESS text-faithful than the base conditional model. In practice, w < 1 is rarely used -- the whole point of CFG is to AMPLIFY the text signal beyond w=1.)
- "Why can't you achieve the same effect as CFG by just training the model longer or with a stronger loss weight on text-conditioned examples?" (Answer: CFG works at INFERENCE time by extrapolating in a specific direction. Training longer makes the conditional prediction slightly better but does not extrapolate beyond it. CFG actively amplifies the difference between conditional and unconditional, which training alone cannot do.)

**9. Practice -- Notebook exercises (Colab)**
- **Exercise design rationale:** Cross-attention and CFG are both conceptually clean but benefit from hands-on exploration. The exercises should verify understanding of the cross-attention formula, demonstrate the per-spatial-location nature, and let the student experiment with CFG guidance scales.
- **Exercise sequence (cumulative):**
  1. **(Guided) Cross-attention from self-attention:** Start with a working self-attention implementation (provided). Modify it to perform cross-attention by changing where K and V come from. Predict-before-run: "After this one-line change, what shape will the attention weight matrix be?" (Answer: no longer square -- (H*W) x T instead of (H*W) x (H*W)). Verify with a dummy spatial feature map (4x4, d=32) and dummy text embeddings (6 tokens, d=32).
  2. **(Guided) Visualize cross-attention weights:** Use the modified function to compute attention weights for a 4x4 spatial feature map attending to a 6-token text. Visualize as a heatmap: rows are spatial locations (16), columns are text tokens (6). Observe that different spatial locations attend to different tokens. Compute mean attention per token -- which text token gets the most attention overall?
  3. **(Supported) Implement CFG:** Given a dummy model function that produces different noise predictions for conditional and unconditional inputs, implement the CFG formula. Test at w=0, w=1, w=3, w=7.5, w=15. Plot the L2 norm of the CFG noise prediction as a function of w. Observe that higher w produces larger-magnitude predictions (more aggressive denoising).
  4. **(Independent) CFG with a real diffusion model:** Load a small pretrained text-conditioned diffusion model (provided). Generate images from the same prompt at w=1, 3, 7.5, 12, 20. Observe the quality/fidelity tradeoff. Identify the sweet spot. Reflection: why does extreme guidance produce artifacts?
- **Solutions should emphasize:** The one-line difference between self-attention and cross-attention (K and V source), the non-square attention weight matrix shape, the direction interpretation of CFG (noise_cond - noise_uncond is the text direction), and the practical tradeoff of guidance scale.

**10. Summarize**
- Two mechanisms complete the text-conditioned diffusion architecture:
  1. **Cross-attention** injects CLIP text embeddings into the U-Net. Each spatial location generates a query; the text tokens provide keys and values. Different locations attend to different words, creating spatially-varying text conditioning. The formula is identical to self-attention from Module 4.2 -- only the source of K/V changes.
  2. **Classifier-free guidance** amplifies the text signal at inference. Train with randomly dropped text; at inference, run two forward passes (with and without text) and scale the difference. noise_cfg = noise_uncond + w * (noise_cond - noise_uncond). The guidance scale w controls the fidelity/quality tradeoff.
- Mental model echo: "The timestep tells the network WHEN (noise level, via adaptive norm, global). The text tells it WHAT (semantic content, via cross-attention, spatially varying). CFG turns up the volume on the WHAT."
- The student now understands all the conditioning mechanisms of the Stable Diffusion U-Net.

**11. Next step**
- The architecture is now complete: U-Net with skip connections, timestep conditioning, text conditioning via cross-attention, and CFG. But there is one remaining problem the student felt in the Module 6.2 capstone: pixel-space diffusion is painfully slow.
- The next lesson addresses this by moving the diffusion process from pixel space to latent space. A pre-trained VAE (from Module 6.1) compresses images before the diffusion process, and the U-Net denoises in the smaller latent space. Same algorithm, smaller tensors, much faster.
- "The next lesson is the last piece: latent diffusion. After that, you have Stable Diffusion."

---

## Widget Assessment

**Widget needed:** No dedicated interactive widget.

**Rationale:** The core concepts are best served by diagrams, formulas, and notebook exercises rather than an interactive widget.

- **Cross-attention** is a formula-level concept that transfers directly from self-attention. The key insight (same formula, different K/V source) is best conveyed by a side-by-side comparison, not interactivity. An interactive attention heatmap would be valuable, but the student already experienced the AttentionMatrixWidget in 4.2.1 -- another attention heatmap would not add new insight. The notebook exercises provide hands-on exploration of cross-attention weights.
- **CFG** is a simple arithmetic formula. The interesting part is the guidance scale tradeoff, which is best demonstrated by generating images at different scales (notebook Exercise 4). An interactive slider could be fun but would require a running diffusion model in the browser, which is infeasible.

The lesson uses:
- Side-by-side ComparisonRow (self-attention vs cross-attention)
- Mermaid diagram for block ordering (ResidualBlock -> SelfAttn -> CrossAttn)
- Cross-attention data flow diagram (spatial features <-> text embeddings)
- CFG vector diagram (unconditional, conditional, difference, amplified)
- GradientCards for guidance scale spectrum (w=1, w=7.5, w=20)
- Concrete numerical examples for both cross-attention and CFG
- Colab notebook with 4 exercises (Guided -> Guided -> Supported -> Independent)

---

## Checklists

### Prerequisite Audit
- [x] Every assumed concept listed with required depth
- [x] Each traced via the records (not the curriculum plan)
- [x] Depth match verified for each
- [x] No untaught concepts remain (cross-attention gap resolved with dedicated section; attention layer location gap resolved with brief subsection)
- [x] No multi-concept jumps in widgets/exercises
- [x] All gaps have explicit resolution plans (cross-attention: dedicated section building from self-attention at DEVELOPED; attention layer location: brief structural subsection)

### Pedagogical Design
- [x] Narrative motivation stated as a coherent paragraph (problem before solution)
- [x] At least 3 modalities planned for the core concept, each with rationale (verbal/analogy, visual, symbolic, concrete example, intuitive -- 5 modalities)
- [x] At least 2 positive examples + 1 negative example, each with stated purpose (3 positive + 2 negative)
- [x] At least 3 misconceptions identified with negative examples (5 misconceptions)
- [x] Cognitive load = 2 new concepts (within limit of 3)
- [x] Every new concept connected to at least one existing concept
- [x] Scope boundaries explicitly stated

---

## Review -- 2026-02-13 (Iteration 1/3)

### Summary
- Critical: 0
- Improvement: 3
- Polish: 2

### Verdict: NEEDS REVISION

No critical findings. The lesson is well-structured, pedagogically sound, and covers both core concepts (cross-attention and CFG) with strong connections to prior knowledge. All five misconceptions are addressed. The narrative arc is compelling and the examples are concrete and effective. Three improvement findings exist that would meaningfully strengthen the lesson.

### Findings

### [IMPROVEMENT] -- Missing cross-attention data flow diagram

**Location:** Section 5 (Explain -- Cross-Attention in the U-Net), between the ComparisonRow and the shape walkthrough
**Issue:** The planning document's Visual modality (#1) specified a "Cross-attention data flow diagram: U-Net spatial features (16x16 grid) on one side, CLIP text token embeddings (sequence of ~77 vectors) on the other, Q arrows from spatial features, K/V arrows from text, attention weights as lines of varying thickness between them." This diagram was not built. The Widget Assessment section even lists it as included ("Cross-attention data flow diagram (spatial features <-> text embeddings)"), but it does not exist in the lesson.
**Student impact:** The student goes from the side-by-side formula comparison (symbolic) directly to the shape walkthrough (also symbolic) and then to the attention weight table (concrete numerical). A diagram showing 2D spatial features on one side and 1D text tokens on the other, with Q/K/V arrows flowing between them, would give the student a spatial/visual anchor for understanding the rectangular attention matrix. Without it, the student has to construct this mental image entirely from text and numbers.
**Suggested fix:** Add an SVG or Mermaid diagram between the ComparisonRow and the shape walkthrough showing: a small grid (4x4 or simplified) of spatial features on the left, a column of text token embeddings on the right, W_Q arrows from spatial features producing Q vectors, W_K/W_V arrows from text tokens producing K and V vectors, and the resulting rectangular attention weight connections between them. This completes the visual modality for cross-attention and gives the student a picture that maps to the formula.

### [IMPROVEMENT] -- Notebook Exercise 3 norm computation cell will crash on the marker plotting code

**Location:** Notebook cell-15 (the plotting cell for Exercise 3)
**Issue:** The plotting cell has two parts: (1) a loop where the student fills in `norms`, and (2) marker plotting code that calls `apply_cfg` directly for specific guidance scales. Both parts depend on the student having implemented `apply_cfg`. However, the `norms` list is built in the loop with `pass`, and the subsequent `ax.plot(guidance_scales, norms, ...)` call will crash because `norms` is empty (length 0 vs `guidance_scales` length 100). Then the marker plotting code below also calls `apply_cfg` which returns `None` and would crash on `.norm()`. The problem is that the marker code and the main plot code are in the same cell, so the student cannot run the markers independently to verify their `apply_cfg` implementation before tackling the loop. This creates a situation where the student fills in `apply_cfg`, runs the test cell (cell-14) to verify it works, then has to fill in the loop AND run the whole plotting cell at once.
**Student impact:** Minor friction. The student might get confused by a crash in code they did not write (the marker plotting section) if they only partially complete the TODOs. However, since cell-14 tests `apply_cfg` independently first, and the "What to fill in" instructions are clear, most students would implement both TODOs before running the plotting cell.
**Suggested fix:** Move the marker plotting code inside a conditional check (`if norms:`) or add a brief comment above the marker section: `# Note: this section also uses apply_cfg. Make sure you filled in both TODOs above.` Alternatively, no change needed -- the error message would be self-explanatory.

### [IMPROVEMENT] -- Module record still shows "Not built" for lesson 4

**Location:** `.claude/lesson-records/series-6/module-6-3/record.md`, line 197-198
**Issue:** The module record's per-lesson summary for text-conditioning-and-guidance says `**Status:** Not built`. The lesson has been built, reviewed, and has a notebook. The record needs updating to reflect the actual state.
**Student impact:** None directly (the student does not read the module record). But the record is used for planning future lessons -- if a future lesson's planning phase reads the module record, it will incorrectly think this lesson has not been built, potentially leading to wrong prerequisite assessments.
**Suggested fix:** After the review cycle completes, update the module record with the full per-lesson summary (status, concepts taught, mental models, etc.) following the pattern of the three existing lesson summaries.

### [POLISH] -- The "Exploring Latent Spaces" reference in CFG vector aside may be ambiguous

**Location:** Section 7d, the TipBlock aside titled "Directions in Space"
**Issue:** The aside references "In **Exploring Latent Spaces**, you saw meaningful directions in the VAE's latent space." The lesson name "Exploring Latent Spaces" was from Module 6.1 (lesson 6.1.4). The student has had many lessons since then. The reference is correct but the lesson title alone may not immediately trigger the student's memory. Adding "from Module 6.1" would help.
**Student impact:** Very minor. The student might think "which lesson was that?" for a moment. The aside still works without the clarification.
**Suggested fix:** Change to "In **Exploring Latent Spaces** (Module 6.1)..." to help the student locate the memory.

### [POLISH] -- Notebook Exercise 4 references a specific HuggingFace model path that may change

**Location:** Notebook cell-18, the instructions for Exercise 4
**Issue:** The instructions say `StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", ...)`. HuggingFace model paths can change over time (the original Runway path was deprecated in favor of the current one). Additionally, the solution uses a different model (`nota-ai/bk-sdm-small`). Having two different model names (one in instructions, one in solution) could confuse the student.
**Student impact:** If the model path changes, the student gets an error they did not cause. The mismatch between instruction and solution model names is a minor source of confusion.
**Suggested fix:** Use the same model in both the instructions and the solution (recommend `nota-ai/bk-sdm-small` in both places for speed), and add a comment noting that `stable-diffusion-v1-5/stable-diffusion-v1-5` is the full-size alternative.

### Review Notes

**What works well:**
- The lesson does an excellent job leveraging the student's deep self-attention knowledge. The cross-attention reveal ("same formula, different K/V source") is exactly the right pedagogical move. The side-by-side ComparisonRow makes the minimal delta visceral.
- The attention weight table with "a cat sitting in a sunset" is one of the best concrete examples in the series. The three spatial locations (cat's face, sky, cat's body) attending to different words makes spatially-varying conditioning immediately tangible.
- The CFG explanation follows the motivation rule perfectly: the problem (model ignores text) is stated before the solution (random dropout + amplification). The geometric intuition SVG diagram and the "contrast slider" analogy give the student two mental models for understanding guidance scale.
- All five misconceptions from the planning document are addressed at the right locations.
- The notebook is well-structured with a clean scaffolding progression and predict-before-run prompts in the Guided exercises.

**Patterns observed:**
- The lesson is thorough and well-connected. The main weakness is the missing cross-attention data flow diagram, which would complete the visual modality and give the student a spatial anchor for the rectangular attention matrix. This is a gap between what was planned and what was built.
- The module record needs updating -- this is a recurring pattern where the record trails the actual build state.

---

## Review -- 2026-02-13 (Iteration 2/3)

### Summary
- Critical: 0
- Improvement: 0
- Polish: 1

### Verdict: PASS

All iteration 1 findings have been addressed. The lesson is well-structured, pedagogically sound, and complete. Both core concepts (cross-attention and CFG) are taught with strong connections to prior knowledge, multiple modalities, concrete examples, and effective misconception handling. The narrative arc is compelling and well-paced. The notebook is well-scaffolded with good solution quality. One minor polish finding remains.

### Iteration 1 Findings -- Verification

**[IMPROVEMENT] Missing cross-attention data flow diagram -- FIXED.**
An SVG diagram was added at lines 292-418, between the ComparisonRow and the shape walkthrough. The diagram shows a 4x4 grid of blue spatial features on the left, a column of 6 violet text token embeddings on the right, W_Q arrow from spatial features producing Q, W_K/W_V arrows from text tokens producing K and V, and amber attention weight lines of varying thickness showing a "cat region" spatial location attending strongly to "cat." Includes dimension annotations (Q: 16, d_k; K: 6, d_k; V: 6, d_v), a legend, and a descriptive caption. Correctly wrapped in `<Row><Row.Content>...<Row.Aside>...</Row.Aside></Row>`. Uses unique SVG marker IDs that do not collide with the CFG vector diagram's markers. The diagram fulfills the planning document's visual modality specification and provides the spatial anchor the student needs.

**[IMPROVEMENT] Notebook Exercise 3 norm computation cell crash -- FIXED.**
Cell-15 now wraps the plotting code in an `if norms:` guard. If the student runs the cell without filling in the TODO loop, they get a clear message: "norms list is empty. Fill in the TODO loop above." The marker plotting code is also inside the guard, so no crash occurs. The figure is closed cleanly in the else branch.

**[IMPROVEMENT] Module record still shows "Not built" -- DEFERRED.**
Handled separately per task instructions. Not re-evaluated here.

**[POLISH] "Exploring Latent Spaces" reference ambiguity -- FIXED.**
The TipBlock now reads "In **Exploring Latent Spaces** (Module 6.1), you saw meaningful directions..." The "(Module 6.1)" addition helps the student locate the memory.

**[POLISH] Notebook Exercise 4 model path mismatch -- FIXED.**
Cell-18 instructions now use `"nota-ai/bk-sdm-small"` as the primary model, consistent with the solution in cell-20. The full-size model `"stable-diffusion-v1-5/stable-diffusion-v1-5"` is mentioned as an alternative for users with more VRAM.

### Findings

### [POLISH] -- SVG hardcoded fill colors may have reduced contrast in light mode

**Location:** Cross-attention data flow diagram (lines 298-401) and CFG vector diagram (lines 984-1036)
**Issue:** Both SVG diagrams use hardcoded hex colors for text labels and fills (e.g., `#94a3b8` for muted labels, `#e2e8f0` for text on violet backgrounds, `#64748b` for secondary labels). These colors are calibrated for a dark background and render well in the app's dark theme. If the app ever supports light mode, these colors would have low contrast against a light background (e.g., light gray text on white). The surrounding `bg-muted/30` container provides some protection, but the SVG text colors do not adapt to CSS variables.
**Student impact:** None currently -- the app uses a dark theme. Only relevant if a light mode is added in the future.
**Suggested fix:** No action needed now. If light mode is ever added, refactor SVG text colors to use `currentColor` or CSS variables. This is a future-proofing concern, not a current issue.

### Review Notes

**What works well:**
- All iteration 1 findings were properly addressed. The cross-attention data flow SVG is a strong addition -- it gives the student a spatial/visual anchor between the symbolic formula (ComparisonRow) and the numerical examples (attention weight table). The two-input-one-output structure is visually clear, and the varying-thickness attention lines preview the spatially-varying nature that the subsequent sections develop.
- The notebook's `if norms:` guard is a clean solution to the Exercise 3 issue. The error message is student-friendly and actionable.
- The model path consistency fix in the notebook eliminates a potential source of confusion in Exercise 4.
- The lesson as a whole is one of the strongest in the series. The cross-attention section leverages the student's deep prior knowledge beautifully (the "one-line change" framing makes the concept feel immediately accessible). The CFG section follows the motivation rule perfectly and provides multiple modalities (verbal analogy, geometric diagram, algebraic formula, concrete numbers). The guidance scale tradeoff is grounded in practical consequences.
- The five misconceptions from the planning document are all addressed at the right locations with concrete negative examples. The "They Coexist, Not Compete" ComparisonRow (timestep vs text conditioning) is particularly effective.

**Patterns observed:**
- The lesson demonstrates a mature implementation pattern: consistent Row layout throughout, appropriate use of block components (ComparisonRow for contrasts, GradientCards for spectra, WarningBlocks for misconceptions, InsightBlocks for key takeaways), and well-placed asides that complement rather than distract from the main content.
- The SVG diagrams are well-crafted but use hardcoded colors. This is a systemic pattern across multiple lessons and could be addressed project-wide if light mode is ever added.
